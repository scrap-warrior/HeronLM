"""
Fake quantization, for the PTQ delta the QAT decision is gated on.

Three things this file is deliberate about.

1. It lives in the harness, not in the reference. The reference is the parity oracle, and a fake-quant hook
   inside it would make the parity test depend on quantizer details: per-block scales, block size, and the
   order of dequantization relative to nGPT normalization. The reference stays bit-clean; this file is the
   training-side question.

2. The scheme is recorded as data. Only a single-block symmetric int4 version is simulated. The Q4_K
   superblocks and the min/max scales that produce the roadmap's 4.9 effective bits are NOT simulated, so
   the number here is 5.0 and the gap between the two is the block overhead. Calling a simulated
   single-block scheme "Q4_K_M" would make the measured delta unattributable: a loss change could be coming
   from the scheme being simulated or from the overhead that is not simulated.

3. Normalization order is tested both ways, because it is a parity risk rather than a preference. The rule
   the engine must follow is normalize AFTER dequant. A converter that normalized before quantizing would
   store weights whose per-block scales were computed from magnitudes that normalization removes, and those
   weights would not reproduce the reference. The difference between the two orders is measured below.

QAT does nothing for the components excluded here: there is no differentiable rounding for the hash table
rows, no head to train through the tied embedding, and no trained head in the router, because the
prototypes are data. It also buys quality, not speed.

Run through scripts/phase0.py.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from heronlm.memory import slot_ids
from heronlm.norms import PARITY_RULE
from phase0 import gates

NORM_ORDERS = ("after_dequant", "before_dequant")


@dataclass(frozen=True)
class QuantScheme:
    name: str
    bits: int
    # ggml blocks run along the input dimension of a weight matrix, which is dim 1 in the reference's
    # [out_features, in_features] layout. That is the same axis normalization sums over, which is why the
    # order of the two steps is load-bearing.
    axis: int
    block_size: int
    code_min: int
    code_max: int
    scale_bits: int
    applies_to: tuple[str, ...]
    excludes: tuple[str, ...]
    source: str
    note: str

    def effective_bits(self) -> float:
        """Bits per weight including the block scale, which is the number the budget uses."""
        if self.block_size <= 0:
            return float(self.bits)
        return self.bits + self.scale_bits / self.block_size


# Single-block symmetric int4: 4.0 bits of code plus 4 bytes of scale per 32 weights, i.e. 1.0 bit of
# overhead. Not the roadmap's 4.9, because Q4_K superblocks and min/max scales are not simulated.
DENSE_INT4 = QuantScheme(
    name="q4_0_single_block_sim",
    bits=4,
    axis=1,
    block_size=32,
    code_min=-8,
    code_max=7,
    scale_bits=32,
    applies_to=("weight",),
    excludes=("embed", "memory.table"),
    source="proposed by this harness; the roadmap's Q4_K_M = 4.9 includes overhead this simulation omits",
    note="Symmetric, per-block, no imatrix. The QAT decision has to be made against THIS scheme recorded as data, not against a scheme name.",
)

# Table rows are int8 by design, one scale per row, not decayed and not quantized with an imatrix. The order
# question does not apply to them: the reference does not normalize table rows, so normalize-then-quantize
# and quantize-then-normalize are the same operation for a row.
TABLE_INT8 = QuantScheme(
    name="int8_per_row",
    bits=8,
    axis=1,
    block_size=0,  # one scale per row, so the whole row is one block
    code_min=-128,
    code_max=127,
    scale_bits=16,
    applies_to=("memory.table",),
    excludes=(),
    source="ROADMAP device table: 1.5M x 256 int8",
    note="ROADMAP uses 8.0 effective bits; with a per-row half-precision scale this simulation gives 8.0625. That is scale accounting, not architecture, and the roadmap number is the one the budget test asserts.",
)

Q8_0 = QuantScheme(
    name="q8_0",
    bits=8,
    axis=1,
    block_size=32,
    code_min=-128,
    code_max=127,
    scale_bits=16,
    applies_to=("embed",),
    excludes=(),
    source="the tied embedding is q8_0 by rule, never Q4",
    note="ROADMAP uses 8.03; the block overhead is 34 bytes per 32 weights (half-precision scale plus 32 codes), which is 8.5. Recorded as the discrepancy to resolve rather than silently picked.",
)

# Parameters that are never quantized as matrices. Each exclusion has a reason, because "it is small" is not
# the reason: these are parameters whose values define semantics rather than compute.
EXCLUSION_REASONS = {
    "stream.read_weights": "the per-pass read mixture. Quantizing it would change which branch a block reads, which is the mechanism under test, not a rounding detail.",
    "reinject_gate": "one scalar per block, a ratio between the current state and the original token identity.",
    "gate_scale": "one scalar per expert slot, the amount that makes hot-swapping a slot safe.",
    "gate_bias": "the memory gate's bias, a single value.",
    "gate_key": "the gate's key direction, a vector rather than a matrix. The engine stores it at higher precision because the relevance cosine is computed on it before projection.",
}

NORM_ORDERS = (PARITY_RULE, "before_dequant")
# PARITY_RULE itself comes from heronlm.norms, which is the reference and therefore the authority: metadata
# records the same value for general.arch.heron.weight_norm_order.


def scheme_for(param_name: str, dense: QuantScheme = DENSE_INT4, table: QuantScheme = TABLE_INT8,
              embedding: QuantScheme = Q8_0) -> QuantScheme | None:
    """Which scheme a parameter gets, or None if it is excluded."""
    for excluded in EXCLUSION_REASONS:
        if excluded in param_name:
            return None
    if "memory.table" in param_name:
        return table
    if "embed" in param_name:
        # The tied embedding is the output head, so its scheme follows the rule: q8_0, never Q4.
        return embedding if dense.bits < embedding.bits else None
    if any(a in param_name for a in dense.applies_to):
        return dense
    return None


def _blocks(t: torch.Tensor, scheme: QuantScheme):
    axis = scheme.axis
    n = t.shape[axis]
    block = n if scheme.block_size == 0 else min(scheme.block_size, n)
    if n % block != 0:
        raise ValueError(
            f"block size {block} does not divide axis {axis} of a {tuple(t.shape)} tensor. "
            "This is not a rounding detail: ggml requires exact multiples, so a width like the roadmap's "
            "expert_ffn_width 20345 (not a multiple of 32) has to become a multiple of the block, or the "
            "last block is partial and the effective bits change."
        )
    moved = torch.moveaxis(t, axis, -1) if t.dim() > 1 else t.reshape(1, n)
    return moved, moved.reshape(*moved.shape[:-1], n // block, block), block


def quantize(t: torch.Tensor, scheme: QuantScheme):
    """
    Returns (codes, scales), blocked along the scheme's axis.

    Codes plus scales are what the engine stores; the dequantized values are what the forward pass sees. The
    invariant that matters for parity is the stored scale: re-rounding the dequantized values against the
    scale that was stored reproduces the same codes. Re-quantizing from scratch does not have that property,
    because the recomputed max-abs is a floating-point product rather than the stored scale, and a block whose
    recomputed scale differs in the last bit can round one weight to a different code. That is why the engine
    stores the scale, and why the reference has to read the stored scale instead of re-deriving it. The test
    asserts the stored-scale form, not the naive one.
    """
    if t.dim() == 0:
        raise ValueError("scalar parameters are excluded from quantization; see EXCLUSION_REASONS")

    _, blocks, _ = _blocks(t, scheme)
    qmax = max(abs(scheme.code_min), abs(scheme.code_max))
    max_abs = blocks.abs().amax(dim=-1, keepdim=True)
    # An all-zero block stores scale 0.0, which is what ggml does. Table rows are initialized to zero, so this
    # path is hit on the first forward pass: a guard that silently used scale 1 would produce the same
    # dequantized value while disagreeing with the file about what the scale is.
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.zeros_like(max_abs))
    codes = torch.round(blocks / scale.clamp_min(1e-30)).clamp(scheme.code_min, scheme.code_max).to(torch.long)
    return codes, scale


def dequantized(t: torch.Tensor, scheme: QuantScheme) -> torch.Tensor:
    """
    The weight as the engine reads it: stored codes, then the stored scale, in the original shape.

    The reconstruction is codes x scale, not the original value x scale. The distinction is not cosmetic:
    multiplying the unquantized tensor by the block scale produces values that look plausible but are the
    original magnitudes inflated by the scale, so every fake-quant delta would measure a scale factor rather
    than quantization error.
    """
    moved, _, _ = _blocks(t, scheme)
    codes, scales = quantize(t, scheme)
    out = (codes * scales).reshape(moved.shape)
    return torch.moveaxis(out, -1, scheme.axis) if t.dim() > 1 else out.reshape(t.shape)


def requantize_is_stable(t: torch.Tensor, scheme: QuantScheme) -> bool:
    """
    Re-round the dequantized values against the STORED scale and check the codes come back identical.

    This is the testable parity invariant, and it holds exactly. It is what makes engine parity testable at
    all: a reader that applies the same rule to the same stored codes and the same stored scale lands on the
    same numbers. Zero blocks are excluded, because their stored scale is 0 and re-rounding is undefined for
    them - the engine never re-rounds a stored block anyway.
    """
    _, _, _ = _blocks(t, scheme)
    codes, scales = quantize(t, scheme)
    deq = codes * scales
    r = torch.round(deq / scales.clamp_min(1e-30)).clamp(scheme.code_min, scheme.code_max).to(torch.long)
    nonzero = (scales > 0).expand_as(codes)
    return bool(torch.equal(r[nonzero], codes[nonzero]))


def pre_normalized(t: torch.Tensor, cfg, scheme: QuantScheme) -> torch.Tensor:
    """
    The weight as it would be if normalization happened BEFORE quantizing.

    Only 2D weights have a normalization rule to apply, and only the nGPT arm normalizes. In the layernorm arm
    the two orders are the same operation, so the measured delta between orders is zero by construction on
    that arm - worth stating, because otherwise a zero delta looks like a result.
    """
    if cfg.norm_mode != "ngpt" or t.dim() < 2:
        return t
    return F.normalize(t, dim=scheme.axis)


def quantized_stored_weight(t: torch.Tensor, cfg, scheme: QuantScheme, order: str) -> torch.Tensor:
    """
    The stored weight, under either order.

    after_dequant: stored = qdq(w). The forward rule then normalizes what was read, which is exactly what the
        reference does and what the engine must reproduce.
    before_dequant: stored = qdq(normalize(w)). The forward rule still normalizes on read, so the result is
        normalize(qdq(normalize(w))) - a tensor that does not reproduce the reference, because the block scales
        were computed from magnitudes that normalization removes.
    """
    if order == PARITY_RULE:
        return dequantized(t, scheme)
    if order == "before_dequant":
        return dequantized(pre_normalized(t, cfg, scheme), scheme)
    raise ValueError(f"unknown normalization order: {order}")


def fake_quantize_model(model, dense: QuantScheme = DENSE_INT4, order: str = PARITY_RULE,
                        table: QuantScheme = TABLE_INT8, embedding: QuantScheme = Q8_0):
    """
    A quantized copy of the model. The original is left untouched, so the delta is attributable to
    quantization alone rather than to the copy.

    Nothing here changes the forward semantics: the reference's own rule is applied to quantized weights,
    which is the only way the comparison measures quantization error rather than two different models.
    """
    q = copy.deepcopy(model)
    for name, p in q.named_parameters():
        scheme = scheme_for(name, dense, table, embedding)
        if scheme is None:
            continue
        p.data = quantized_stored_weight(p.data, model.cfg, scheme, order)
    return q


@torch.no_grad()
def path_facts(model, tokens: torch.Tensor, k: int) -> dict:
    """
    The decisions a forward pass makes, independent of the values it computes.

    Three, and only three, because these are the parts of a forward pass that choose a route rather than
    do arithmetic: which expert slot fires, whether a table recall is injected at all, and which branch
    each block reads. Everything else is arithmetic on numbers, and rounding numbers is what quantization
    is for.

    The gate bucket is `>= 0.5` rather than the raw value because the engine's decision is binary: the
    recalled value either enters the reserved branch or it does not. Comparing raw gate values would
    report a change for every position, which is rounding, not a path change.
    """
    cfg = model.cfg
    ids = slot_ids(tokens, cfg)
    _, pooled = model.forward(tokens, k, None, None, ids, return_pooled=True)
    # decide returns (state, margin); the slot is the decision, the margin is what flapping is measured on.
    slot = model.router.decide(pooled)[0].active_slot
    _, gate = model.memory.read(ids)
    # Branch assignment is a config rule, so it cannot change; only the read mixture can, and that is the
    # parameter the exclusion protects.
    branches = [int(w.argmax()) for w in model.stream.read_weights]
    return {
        "slot": slot,
        "gate_bucket": [int(g >= 0.5) for g in gate[0].flatten().tolist()],
        "read_branches": branches,
    }


@torch.no_grad()
def path_stability(model, samples, k: int, dev: torch.device | None = None,
                   dense: QuantScheme = DENSE_INT4, order: str = PARITY_RULE,
                   table: QuantScheme = TABLE_INT8, embedding: QuantScheme = Q8_0) -> dict:
    """
    Whether PTQ changes which path a token takes. This is the half of the QAT decision that a delta in nats
    cannot answer: a model can lose 0.01 nats and still route every conversation to the same expert, or
    lose nothing and start firing a different slot.

    One path is reported as undecidable rather than as stable, and that distinction is the point. An
    untrained table is all zeros, so its gate sits at exactly 0.5 at every position and no rounding can
    move it. Reporting "no changes" there would look like a result for ROADMAP line 157, when it is a
    property of an untrained table. The memory path only becomes decidable after the table has been
    trained on data disjoint from the eval set, which is Phase 1.
    """
    from phase0.train import device

    dev = dev or device()
    q = fake_quantize_model(model, dense, order, table, embedding)
    cfg = model.cfg

    slot_changes, gate_changes, branch_changes = 0, 0, 0
    for s in samples:
        tokens = torch.tensor([list(s.tokens)], device=dev)
        a, b = path_facts(model, tokens, k), path_facts(q, tokens, k)
        slot_changes += int(a["slot"] != b["slot"])
        gate_changes += sum(int(x != y) for x, y in zip(a["gate_bucket"], b["gate_bucket"]))
        branch_changes += sum(int(x != y) for x, y in zip(a["read_branches"], b["read_branches"]))

    # The degenerate case, checked rather than assumed: a table initialized to zeros has no rounding to
    # lose, so the memory path cannot change here.
    table_untrained = bool((model.memory.table.weight == 0).all())
    paths = {
        "router_slot": {"samples": len(samples), "changes": slot_changes},
        "memory_gate": {
            "positions": len(samples) * len(samples[0].tokens),
            "changes": gate_changes,
            "undecidable": table_untrained,
            # Only stated when the path actually is undecidable; otherwise the row reads as if a result came
            # from an untrained table.
            **({"reason": "untrained table rows are zero, so every gate is exactly 0.5 and rounding cannot move it"}
               if table_untrained else {}),
        },
        "gr_read_branch": {
            "blocks": int(model.stream.read_weights.shape[0]),
            "changes": branch_changes,
            # Not a measurement, a consequence of the exclusion: the read mixture is excluded because
            # quantizing it would change which branch a block reads. If this ever reports a change, the
            # exclusion was violated rather than the model behaving badly.
            "cannot_change_by_construction": "stream.read_weights" in EXCLUSION_REASONS,
        },
    }
    # The claim is about every path, so one undecidable path makes the whole rule undecidable. Reporting a
    # pass from the decidable half alone would let an untrained table count as evidence.
    undecidable = [name for name, p in paths.items() if p.get("undecidable")]
    return {
        "gate": "PTQ path stability",
        "k": k,
        "paths": paths,
        # None while any path is undecidable: `not measured` is the honest state, and a pass here would come
        # from an untrained table rather than from PTQ being safe.
        "ptq_does_not_change_paths": None if undecidable else bool(sum(p["changes"] for p in paths.values()) == 0),
        "undecidable_paths": undecidable,
    }


@torch.no_grad()
def ptq_delta(model, samples, k: int, dev: torch.device | None = None,
             dense: QuantScheme = DENSE_INT4, order: str = PARITY_RULE,
             table: QuantScheme = TABLE_INT8, embedding: QuantScheme = Q8_0) -> dict:
    """
    Held-out loss before and after fake quantization.

    This is the number the QAT decision is gated on. The rule I am proposing, and stating as a proposal rather
    than as a threshold: adopt QAT only where the PTQ delta exceeds about 0.05 nats, because below that the
    training-side complexity is not paying for itself. ROADMAP states no value for this.
    """
    from phase0.train import device, evaluate

    dev = dev or device()
    base = evaluate({"model": model}, samples, k, dev)
    by_order: dict[str, dict] = {}
    for o in NORM_ORDERS:
        qmodel = fake_quantize_model(model, dense, o, table, embedding)
        by_order[o] = evaluate({"model": qmodel}, samples, k, dev)

    quantized_params, excluded_params = 0, 0
    for name, p in model.named_parameters():
        if p.numel() == 0:
            continue
        if scheme_for(name, dense, table, embedding) is None:
            excluded_params += p.numel()
        else:
            quantized_params += p.numel()

    # The adoption number is read from gates.py rather than restated here. One source is what keeps the
    # test that checks numbers against ROADMAP meaningful: if the harness could invent its own threshold
    # locally, the check would be against the harness rather than against the design.
    adopt = next(t for t in gates.gate("QAT").thresholds if t.metric == "qat_adopt_if_delta_above")

    return {
        "gate": "PTQ delta",
        "k": k,
        "scheme": dense.name,
        "effective_bits": dense.effective_bits(),
        "order": order,
        "parity_rule": PARITY_RULE,
        "bf16_loss": base["loss"],
        "by_order": by_order,
        "delta_nats": by_order[order]["loss"] - base["loss"],
        # Stated separately from the quantization error: this is the parity risk, the cost of the wrong order.
        "normalization_order_gap_nats": by_order["before_dequant"]["loss"] - by_order[PARITY_RULE]["loss"],
        "quantized_params": quantized_params,
        "excluded_params": excluded_params,
        "qat_proposal": {"adopt_if_delta_above": adopt.limit, "status": adopt.source},
    }

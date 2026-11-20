"""
Phase 4.1, the other direction: a file in, a module out.

`heronlm.converter` writes what the reference says. This file reads what a file says and checks that the two
agree, which is the proof that the converter's claims are consumable rather than merely written. A converter can
state a name, a shape and a block in metadata and still produce a file a loader cannot use, because nothing in
the format forces a loader to agree with the metadata: it forces the loader to agree with the bytes. So this is
where "the file describes the reference" becomes testable.

Four things this file deliberately does not do.

1. It does not normalize. `heronlm.norms.PARITY_RULE` says normalization happens after dequantization, and the
   only way that rule survives is that the reader hands the module the stored values. The moment a loader
   normalizes on the way in, the order is decided by the loader rather than by the rule, and the parity test can
   no longer tell which of the two produced the numbers. So there is no normalize step here at all: the module
   normalizes at forward time, exactly as the reference does, and the loader's job ends at "these are the file's
   bytes, in the file's order".

   What this file can therefore check is the claim, and it refuses a file that states a different order. The
   limit of that check is worth stating plainly, because it is the limit of what a reader can prove: for the
   unit-norm arm normalization is idempotent, so a file written from pre-normalized weights is indistinguishable,
   by looking at loaded values alone, from a file written from raw weights. The check that catches it is a byte
   comparison against the reference, which is what tests/test_converter.py does.

2. It does not invent names. Every name it loads has to be explained by the file's own claim
   (`general.arch.heron.tensor_names`), and the module path is derived from that claim rather than from a second
   copy of the mapping. The claim is written as patterns with placeholders, so a concrete name is matched by
   substituting the block it carries, not by dictionary lookup: the claim covers the family, not the individual
   tensor. A name the claim cannot explain is refused. That is the failure the converter exists to prevent, seen
   from the other side.

3. It does not guess the slot. A per-expert file is one slot set, and which slot it is is a claim
   (`general.arch.heron.expert_file_slot`), not something derivable from the bytes. Without the claim the loader
   cannot tell a code expert's file from a math expert's file, so it refuses to place the bytes.

4. It does not treat the hash table as a tensor. There is no ggml probe op, so the table is an mmap'd blob:
   codes, then scales. The loader reads it as the blob the metadata describes, and reports what no file can
   supply rather than silently leaving it at initialization - which is the failure mode this phase is about,
   because a randomly initialized parameter in a loaded model produces perfectly plausible numbers.

Two shape facts are kept apart, because conflating them is the classic silent converter bug: the file records
dims fastest-first, so a PyTorch [out, in] weight appears as [in, out], while the bytes stay in the module's
element order. The shape the loader uses is the module's, and the file's dims are checked against its reverse.

Run tests: python tests/test_loader.py
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import quants
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as Quant

from heronlm import converter
from heronlm.config import HeronConfig, ROPE_FREQ_BASE
from heronlm.model import HeronLM
from heronlm.norms import EPS, PARITY_RULE


# Config facts, as a file states them. Only the facts a file can state are here, and that is the point: the
# training fields describe how a model was trained, not what it is, so a loader given only a file cannot rebuild
# an arm. `config_from_metadata` reports what it cannot state instead of guessing at a training default.
METADATA_FIELDS = {
    "general.arch.heron.vocab_size": "vocab_size",
    "general.arch.heron.embedding_length": "d_model",
    "general.arch.heron.attention.head_count": "heads",
    "general.arch.heron.attention.head_count_kv": "kv_heads",
    "general.arch.heron.attention.head_dim": "head_dim",
    "general.arch.heron.feed_forward_length": "ffn_width",
    "general.arch.heron.expert_ffn_width": "expert_ffn_width",
    "general.arch.heron.window_size": "swa_window",
    "general.arch.heron.cache_policy": "cache_policy",
    "general.arch.heron.loop.max_loop": "max_loop",
    "general.arch.heron.loop.default_loop": "loop_curriculum_bulk",
    "general.arch.heron.gr.branches": "gr_branches",
    "general.arch.heron.gr.read_enabled": "ngpt_gr_read",
    "general.arch.heron.norm_mode": "norm_mode",
    "general.arch.heron.ngpt_weight_scale": "ngpt_weight_scale",
    "general.arch.heron.experts.count": "n_experts",
    "general.arch.heron.experts.generalist_slot": "generalist_slot",
    "general.arch.heron.table.entries": "ngram_memory_entries",
    "general.arch.heron.table.value_dim": "ngram_memory_dim",
    "general.arch.heron.table.read_block": "ngram_memory_read_block",
    "general.arch.heron.ngram.max_entries": "ngram_max_entries",
    "general.arch.heron.ngram.top_k": "ngram_top_k",
    "general.arch.heron.ngram.min_cnt": "ngram_min_cnt",
    "general.arch.heron.ngram.orders": "ngram_orders",
    "general.arch.heron.router.temperature": "router_temperature",
    "general.arch.heron.router.margin": "router_margin",
    "general.arch.heron.router.hysteresis_turns": "router_hysteresis_turns",
}

# The three-part split, as the file states it. `block_count` is not a config field, so it is not in
# METADATA_FIELDS: it is the length of these three lists, which is the same fact read a different way.
BLOCK_MAP_KEYS = {
    "input": "general.arch.heron.loop.input_layers",
    "loop": "general.arch.heron.loop.layers",
    "output": "general.arch.heron.loop.output_layers",
}


@dataclass
class GGUFFile:
    """One file, as the file states it: its claims and the tensors it actually holds."""

    path: str
    metadata: dict[str, Any]
    tensors: dict[str, dict[str, Any]]

    def claims(self) -> dict[str, str]:
        """
        The name claim, as the file wrote it: module pattern -> name pattern.

        Read from `general.arch.heron.tensor_names`, which is the same dict the converter looked up, so the
        loader cannot know a name the file does not claim. If the loader carried its own copy of the mapping, a
        file holding a tensor the metadata does not describe would still load - and loading it would mean placing
        bytes somewhere the writer never said they belonged.

        Both sides of the mapping are patterns, not concrete names: `blocks.{bid}.attn_q.weight ->
        blk.{bid}.attn_q` covers every block at once. So matching a name in a file is a substitution against the
        block index the name carries, which is what `name_pattern` and `name_claims` do.
        """
        out: dict[str, str] = {}
        for line in self.metadata.get("general.arch.heron.tensor_names", []):
            pattern, _, name = line.partition(" -> ")
            out[pattern] = name
        return out

    def name_claims(self) -> dict[str, str]:
        """The claim read in the direction a reader needs: name pattern -> module pattern."""
        return {name: pattern for pattern, name in self.claims().items()}

    def slot(self) -> int | None:
        """The expert this file is, for a per-expert file. None for the core file."""
        return self.metadata.get("general.arch.heron.expert_file_slot")

    def block_map(self) -> dict[str, list[int]]:
        """The three-part split as the file states it, not as the config computes it. Empty when the file does
        not state it, which is what makes a file that omits the split show up as missing names rather than as a
        KeyError in a reader that assumed the split was there."""
        return {part: [int(b) for b in self.metadata.get(key, [])] for part, key in BLOCK_MAP_KEYS.items()}

    def expert_blocks(self) -> list[int]:
        """The blocks this file's slots sit in, as the file states them."""
        return [int(b) for b in self.metadata.get("general.arch.heron.expert_slot_layers", [])]

    def scope(self) -> list[str]:
        """
        Which patterns this file is allowed to hold.

        The claim is model-wide, so a per-expert file cannot be checked against the whole claim: it does not hold
        `blk.0.attn_q`, and it is not supposed to. A file is checked against the patterns its own scope covers,
        which is the same distinction `converter.tensor_entries` makes when it decides what a file carries - the
        check has to be written the same way from both ends or the two directions check different things.

        The hash table is not in the claim at all: `TENSOR_NAMES` records it as None, and a pattern that names
        nothing is not a claim a file can hold. It is the blob's, and `open_blob` is where its bytes are read.
        """
        patterns = self.claims()
        if self.slot() is None:
            return [p for p in patterns if not p.startswith("experts.")]
        return [p for p in patterns if p.startswith("experts.")]

    def names(self) -> dict[str, str]:
        """
        The names this file should hold, as the file states them: name -> module pattern.

        Built from the claim plus the file's own block map and slot claim, so nothing here comes from the config
        method that computes the same thing. That is deliberate: a reader that reconstructed names from the config
        would be checking the file against the reader's assumptions, and a file that disagreed with the reference
        about which block holds which tensors would pass.

        Only the block placeholder is filled here. `{eid}` and `{slot}` stay in the module pattern because a name
        cannot carry them: `blk.6.ffn_gate_exps` says which slot position it is, not which expert. `module_path`
        fills them from the slot claim, and refuses when there is no claim to fill them from.
        """
        out: dict[str, str] = {}
        blocks = sum(self.block_map().values(), [])
        for pattern in self.scope():
            name = self.claims()[pattern]
            if pattern.startswith("experts."):
                for bid in self.expert_blocks():
                    out[name.replace("{bid}", str(bid))] = pattern
                continue
            if "{" not in name:
                # `memory_gate_key` and `router_prototypes` are applied at a block named by metadata, so the block
                # is not part of the name and there is nothing to substitute.
                out[name] = pattern
                continue
            for bid in blocks:
                out[name.replace("{bid}", str(bid))] = pattern
        return out


def name_pattern(name: str) -> str:
    """
    A name the file holds, written back as the pattern the claim is written in.

    The block index is the only placeholder a concrete name can supply: `blk.6.ffn_gate_exps` is `blk.{bid}.
    ffn_gate_exps` for some pattern. What the name cannot supply is which expert it is, which is why the slot
    claim exists, and why `module_path` refuses without it.

    Names with no block index (`token_embd`, `router_prototypes`, `memory_gate_key`) are their own pattern, which
    is what `TENSOR_NAMES` says they are: the block they are applied at is a metadata fact, not part of the name.
    """
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blk" and parts[1].isdigit():
        parts[1] = "{bid}"
    return ".".join(parts)


def pattern_for(name: str, file: GGUFFile) -> str:
    """The module pattern the claim says this name came from. Refused rather than guessed, per point 2."""
    pattern = name_pattern(name)
    claims = file.name_claims()
    if pattern in claims:
        return claims[pattern]
    raise ValueError(
        f"{name}: the claim cannot explain this name. The file's tensor_names covers {sorted(claims)} only as "
        "these name patterns; a name outside them has no module path, so it cannot be placed."
    )


def open_file(path: Path) -> GGUFFile:
    """
    Read the claims and the bytes, then refuse a file the loader cannot honour.

    Three groups of checks are done on the way in, and all of them are about the loader's own behaviour rather
    than about tidiness:

      - the order rule, at the point where a loader would be about to act on it. A file that states a different
        order is refused rather than loaded with the reference's rule applied anyway, because then the file's
        claim and the loader's behaviour would disagree while both looked fine on their own.
      - names in both directions, against the file's scope. A held name the claim cannot explain is the failure
        the converter exists to prevent; a name the claim covers that the file does not hold is the same failure
        as a reader would see it, and it is what a loader would otherwise fill in with initialization.
      - the claims that are module constants rather than config fields, which a loader cannot honour at all.

    The per-block cache rule is checked in `load_model` rather than here, because it is the one check that needs
    a config from outside the file: a file checked against the config built from its own claims cannot fail, since
    the interval that produced `layer_pattern` is not in the file.

    `general.file_type` is deliberately not checked here, because it is a fact about the file the writer
    produced, not a fact a loader acts on: the loader learns each tensor's type from the tensor itself, so a
    recorded file type that disagrees is a mislabelled header, not a wrong tensor.
    """
    file = GGUFFile(str(Path(path)), converter.read_metadata(Path(path)), converter.read_tensors(Path(path)))
    problems = file_problems(file)
    if problems:
        raise ValueError(f"{file.path}: loader refuses the file: " + "; ".join(problems))
    return file


def name_problems(file: GGUFFile) -> list[str]:
    """
    The claim against the bytes, in both directions, for the patterns this file's scope covers.

    A held name the claim cannot explain is the failure the converter exists to prevent. A name the claim covers
    that the file does not hold is the same failure as a reader experiences it, and it is the more dangerous of
    the two: a missing tensor is not an error, it is a parameter that keeps whatever the freshly built module
    happened to initialize it to. A loaded model with a randomly initialized router produces perfectly plausible
    logits, so the report has to name the missing pieces rather than assume them.
    """
    wanted = file.names()
    held = file.tensors
    problems = [f"{name}: the file holds it but the claim cannot explain it" for name in held if name not in wanted]
    problems += [f"{name}: the claim covers it but the file does not hold it" for name in sorted(wanted) if name not in held]
    return sorted(problems)


def file_problems(file: GGUFFile) -> list[str]:
    """
    Everything about this file the loader cannot honour, as a list rather than as an exception.

    Written as a list because the same checks have to run from two places: `open_file` refuses a file on its own,
    and `load_model` refuses a file against the config it was given. A check that could only run inside
    `open_file` could not be pointed at a file built by hand, and a hand-built file is how the refusal paths get
    tested - a writer that produces a bad file is a writer the tests already caught, so what has to be tested is
    a reader being handed one.
    """
    problems = []
    order = file.metadata.get("general.arch.heron.weight_norm_order")
    if order != PARITY_RULE:
        problems.append(
            f"weight_norm_order: the file states {order!r}, the reference's rule is {PARITY_RULE!r} "
            "(dequantize, then normalize). This loader does not normalize, so it cannot honour a file that "
            "says normalization happens somewhere else."
        )
    if file.slot() == 0:
        problems.append(
            f"expert_file_slot: the file states slot {file.slot()}. Slot 0 is the generalist, which is each "
            "expert block's base FFN, so its bytes are already named by blk.{bid}.ffn_* and a file for slot 0 "
            "has no content. The writer refuses to write one; a file that states it is not a file the loader "
            "can place."
        )
    # The other half of the slot claim: a file that holds a name from an expert pattern is a per-expert file, and
    # only the claim says which. That is not a nicety - five files with the same tensor names and the same shapes
    # are five different domains, and a reader that placed them by iteration order would produce a model that
    # computes plausible numbers through the wrong expert.
    held_patterns = []
    for name in file.tensors:
        try:
            held_patterns.append(pattern_for(name, file))
        except ValueError:
            continue  # an unexplainable name, which name_problems reports
    if file.slot() is None and any(p.startswith("experts.") for p in held_patterns):
        problems.append(
            "expert_file_slot: the file holds names from an expert pattern, so it is a per-expert file, but it "
            "states no slot. Which slot a file is cannot be read from its bytes, so the loader refuses to place "
            "them."
        )
    return problems + name_problems(file) + claim_problems(file)


def stored_float(value) -> float:
    """
    A metadata number as the format actually holds it.

    gguf writes a float field as float32, so `1e-06` does not survive the round trip: the file states
    9.999999974752427e-07. That is not a bug to work around, it is a fact about what a file can state, and the
    loader has to compare at the precision the format has rather than at Python's - otherwise a faithful writer
    is refused by a reader that checks with more precision than the file has. So every numeric claim a loader acts
    on is compared as float32, which is the only comparison that can be satisfied by bytes in a file.
    """
    return float(np.float32(value))


def claim_problems(file: GGUFFile) -> list[str]:
    """
    Claims the loader cannot honour because they are module constants rather than config fields.

    `norm_eps` and the rope base are recorded for the fork, and the reference's values are module constants, so a
    loader has no way to honour a different number: it would either silently use the reference's constant or
    change a constant. Refusing is the only honest option, and it is what makes these keys checkable rather than
    decorative. The comparison is made at the precision the format has, for the reason `stored_float` gives.
    """
    problems = []
    eps = file.metadata.get("general.arch.heron.norm_eps")
    if eps is not None and stored_float(eps) != stored_float(EPS):
        problems.append(
            f"norm_eps: the file states {stored_float(eps)}, the reference clamps at {stored_float(EPS)}; a "
            "loader cannot honour a different floor without changing the constant unit_norm clamps at"
        )
    base = file.metadata.get("general.arch.heron.rope.freq_base")
    if base is not None and stored_float(base) != stored_float(ROPE_FREQ_BASE):
        problems.append(
            f"rope.freq_base: the file states {stored_float(base)}, the reference builds angles from "
            f"{stored_float(ROPE_FREQ_BASE)}; a loader cannot honour a different base without changing the "
            "constant"
        )
    order = file.metadata.get("general.arch.heron.table.blob_order")
    if order is not None and list(order) != ["codes", "scales"]:
        problems.append(f"table.blob_order: the file states {list(order)}, not codes then scales")
    hash_claim = file.metadata.get("general.arch.heron.table.hash")
    if hash_claim is not None and hash_claim != "blake2b-64-mod-entries":
        problems.append(
            f"table.hash: the file states {hash_claim!r}; the reference hashes with blake2b, 64-bit digest, "
            "modulo entries. A loader cannot honour a different hash and still land on the same slot."
        )
    return problems


def module_path(name: str, file: GGUFFile, cfg: HeronConfig) -> str:
    """
    The reference's own path for a name the file holds, resolved from the file's claim.

    Three kinds of name resolve differently, and the difference is why this function exists:

      - `blk.{bid}.<field>` resolves by block index, so the path is `blocks.<bid>.<field>`.
      - `blk.{bid}.residual_alpha` resolves to a ROW of one shared tensor, not to a submodule: the module has one
        tensor for the whole model and the file has one tensor per block. The row index is the block index,
        because `ResidualStream.read` is called with the block index - it is not a sublayer position, and the
        three parts of the block map do not reorder the rows.
      - `blk.{bid}.ffn_gate_exps` cannot resolve without the file's slot claim. The block index says which slot
        position it is, but not which expert: that is the hot-swap fact, and only the claim says it.

    A read-weight row is returned as `stream.read_weights[bid]`, in the same spelling `converter.module_pattern`
    uses, because brackets are what says "element selection on one Parameter" rather than "submodule".
    """
    pattern = pattern_for(name, file)

    # The row branch comes first because this pattern has no placeholder at all: `stream.read_weights -> blk.{bid}
    # .residual_alpha` is a mapping from one module tensor to n names, so the placeholder is on the name side
    # rather than on the pattern side. Checking for a placeholder before this branch would return the module
    # itself, and the assignment would then fail with "9 elements do not match (8, 3, 3)" - a shape error that
    # tells you nothing about the actual mistake.
    if pattern == "stream.read_weights":
        bid = int(name.split(".")[1])
        return f"stream.read_weights[{bid}]"
    if "{" not in pattern:
        return pattern

    bid = int(name.split(".")[1])

    resolved = pattern.replace("{bid}", str(bid))
    if pattern.startswith("experts."):
        slot = file.slot()
        if slot is None:
            raise ValueError(
                f"{name}: cannot be loaded without a slot claim. A per-expert file is one slot set, and the "
                "bytes cannot say which domain they are for; general.arch.heron.expert_file_slot is what says it."
            )
        layers = file.expert_blocks()
        if bid not in layers:
            # The claim's slot layers are the blocks this file's bytes belong to, and a name for a block outside
            # them cannot be resolved: the config puts every routed expert at each of its expert blocks, so a
            # block outside that set has a base FFN - named blk.{bid}.ffn_* - rather than a slot. The block index
            # is the fact here, not the pattern: the name matches the pattern exactly, and is still wrong.
            raise ValueError(
                f"{name}: block {bid} is not one of this file's slot layers {layers}. The config puts every "
                f"routed expert at each of {cfg.expert_blocks()}; a block outside that set has a base FFN, not a "
                "slot, so this name cannot be resolved to a module."
            )
        return resolved.replace("{eid}", str(slot)).replace("{slot}", str(layers.index(bid)))
    return resolved


def value_for(name: str, file: GGUFFile, cfg: HeronConfig) -> np.ndarray:
    """
    The file's values, in the module's shape and the module's element order.

    Dequantize, then stop. No normalization, no re-rounding against a recomputed scale. Three checks are made
    together, because each one alone can pass while the tensor is wrong:

      - the shape, taken from the config rather than from the entry, so a check cannot agree with itself;
      - the file's dims, checked as the reverse of the module's shape, which is the half that rules out a
        transposed tensor;
      - the byte count, checked against the scheme's block size, which is the half that rules out a header that
        states a type the bytes cannot be.

    Within a quantized block, ggml stores the first `block_size` elements of the flattened tensor, so the
    dequantized values come back in the order the module wrote them. That is why `reshape` is the right operation
    here and `T` is not.
    """
    entry = file.tensors[name]
    pattern = pattern_for(name, file)
    shape = converter.expected_shapes(cfg)[pattern]

    if tuple(reversed(entry["dims"])) != shape:
        raise ValueError(f"{name}: file dims {entry['dims']} are not the reverse of the module's {shape}")

    quant = getattr(Quant, entry["type"], None)
    storage = GGML_QUANT_SIZES.get(quant)
    if storage is None:
        raise ValueError(f"{name}: type {entry['type']!r} cannot be read")
    block, type_size = int(storage[0]), int(storage[1])
    numel = int(np.prod(shape))
    if int(entry["n_bytes"]) != numel // block * type_size:
        raise ValueError(
            f"{name}: {entry['type']} stores {numel} elements as {numel // block} blocks of {type_size} bytes, "
            f"the file says {entry['n_bytes']}"
        )

    values = np.asarray(quants.dequantize(np.asarray(entry["data"]).reshape(-1), quant))
    return values.reshape(shape)


@dataclass
class TableBlob:
    """
    The hash table as the file holds it: every code, then every scale, one scale per row.

    The blob has no header, so its shape and dtypes come from the metadata. The order is recorded rather than
    assumed because it is the property a probe depends on: codes-then-scales is what makes a probe one contiguous
    read of `value_dim` bytes plus one half-precision load. Interleaving them would be the same values and a
    different operation.
    """

    path: str
    entries: int
    value_dim: int
    codes: np.ndarray
    scales: np.ndarray

    def row(self, slot: int) -> np.ndarray:
        """The dequantized row, as the gate sees it: code x the stored scale, never a recomputed max-abs."""
        if not 0 <= slot < self.entries:
            raise ValueError(f"slot {slot} is outside the table's {self.entries} entries")
        return (self.codes[slot].astype(np.float32) * self.scales[slot].astype(np.float32)).reshape(self.value_dim)

    def values(self) -> np.ndarray:
        """Every row. `memory.read` takes one row per token; this is the same operation over the whole table,
        which is what makes a comparison against the module's weight possible."""
        return self.codes.astype(np.float32) * self.scales.astype(np.float32).reshape(-1, 1)


def open_blob(path: Path, file: GGUFFile) -> TableBlob:
    """
    Read the blob as the metadata describes it.

    There is no ggml tensor to read here, so every fact has to come from the claims, and the byte count is the
    check: a blob whose length does not match the claimed shape cannot be read as rows, and counting is the only
    way to find that out. `TableBlob.row` is the operation the engine performs host-side.
    """
    md = file.metadata
    entries = int(md["general.arch.heron.table.entries"])
    dim = int(md["general.arch.heron.table.value_dim"])
    order = list(md["general.arch.heron.table.blob_order"])
    if order != ["codes", "scales"]:
        raise ValueError(f"{path}: the file states blob order {order}, not codes then scales")

    code_dtype = np.dtype(md["general.arch.heron.table.dtype"])
    scale_dtype = np.dtype(md["general.arch.heron.table.scale_dtype"])
    raw = Path(path).read_bytes()
    n_codes, n_scales = entries * dim, entries
    if len(raw) != n_codes + n_scales * scale_dtype.itemsize:
        raise ValueError(
            f"{path}: holds {len(raw)} bytes; the metadata's shape needs {n_codes} code bytes plus {n_scales} "
            f"scales of {scale_dtype.itemsize} bytes"
        )

    codes = np.frombuffer(raw[:n_codes], dtype=code_dtype).reshape(entries, dim)
    scales = np.frombuffer(raw[n_codes:], dtype=scale_dtype).reshape(n_scales)
    return TableBlob(str(path), entries, dim, codes, scales)


def config_from_metadata(file: GGUFFile) -> tuple[HeronConfig, dict[str, str]]:
    """
    The config as the file states it, plus what it cannot state.

    This is the read side of the same argument the converter makes about names: a metadata dict that states a
    shape is a claim a loader acts on, and a loader that acts on a claim cannot also check it. So the config is
    built from the claims only, and what the file never said is returned as `unrecorded` rather than guessed.

    One case looks like a gap and is not: `full_attention_every` is not in the file, only its result is
    (`full_attention_layers` and `layer_pattern`). For the blocks a file has, the result IS the behaviour - a
    block is either bounded by the window or it is not - so a loader can reproduce the model from the result
    alone. What it cannot reproduce is the rule that generated the result, and that is reported rather than
    guessed: `attention_interval` says whether the pattern implies an interval, and `check_mixer_rule` refuses a
    pattern that no interval could have produced.

    `expert_slots` is the same kind of case in the other direction: the file states the layers, and the count is
    only derivable because the design puts one slot in each block, which is what makes a name per slot possible.
    """
    md = file.metadata
    cfg = HeronConfig()
    for key, field_name in METADATA_FIELDS.items():
        if key not in md:
            continue
        value = md[key]
        # A number comes back from the reader as a numpy integer and a bool as a bool, and an array as a list of
        # numpy integers. The reader's types are not the config's types, so this is where the two are reconciled -
        # once, for every field.
        if field_name == "ngpt_weight_scale":
            setattr(cfg, field_name, stored_float(value))
        elif isinstance(value, bool):
            setattr(cfg, field_name, bool(value))
        elif isinstance(value, (list, tuple)):
            setattr(cfg, field_name, [int(x) for x in value])
        elif isinstance(value, (int, np.integer)):
            setattr(cfg, field_name, int(value))
        elif isinstance(value, (float, np.floating)):
            # A metadata float comes back as float32, and the config's fields are Python floats. This is the same
            # argument as `stored_float`: the config holds the value the format can state, not Python's reading of
            # it, so a config built from a file and a config compared against a file measure at one precision.
            setattr(cfg, field_name, stored_float(value))
        else:
            setattr(cfg, field_name, value)

    blocks = file.block_map()
    cfg.input_blocks = len(blocks["input"])
    cfg.loop_blocks = len(blocks["loop"])
    cfg.output_blocks = len(blocks["output"])

    # `expert_slots` is not a key the file states; `expert_slot_layers` is. The count follows from the layers only
    # under the design's rule that every slot sits in its own block - which is what makes a name per slot possible
    # at all, since `blk.{bid}.ffn_gate_exps` names one slot per block. If a checkpoint put two slots in one
    # block, the file could not name them separately, so the derivation is only valid when the layers are distinct.
    layers = file.expert_blocks()
    if len(set(layers)) != len(layers):
        raise ValueError(
            f"{file.path}: expert_slot_layers repeats a block ({layers}), which the name scheme cannot state: "
            "a slot is named through its block, so two slots in one block would need two names for the same "
            "block."
        )
    cfg.expert_slots = len(layers)

    return cfg, unrecorded()


def unrecorded() -> dict[str, str]:
    """
    Config fields no file can state, with the reason for each.

    A loader that guessed a training default would be inventing facts the file never made, so these are reported
    as absent rather than filled in. They are not a gap to fix: a file describes a model, and a training arm is
    not a property of a model.
    """
    reasons = {
        "loop_curriculum_expand": "the loop-expansion curriculum is how a model was trained, not what it is",
        "ngram_optimizer": "the optimizer treatment is a training fact; the file records the table's storage",
        "full_attention_every": (
            "the file states the per-block result, not the interval that produced it. The interval follows only "
            "when the pattern is an arithmetic progression from the output block's first block, which "
            "`attention_interval` reports as `implied`; a pattern with one full block does not imply one at all"
        ),
        "ngpt": "derived from norm_mode; the file states the mode as the authority and the boolean as an alias",
        "expert_slots": (
            "the file states the slot layers, not a slot count; the count follows only because the design puts "
            "one slot in each block, which is what makes a name per slot possible"
        ),
    }
    return dict(sorted(reasons.items()))


def attention_interval(pattern: list[int], output_start: int) -> tuple[str, int | None]:
    """
    What this per-block pattern says about the interval that produced it, as (kind, interval).

    `layer_pattern` is the rule itself: which blocks keep an unbounded cache. An interval is a fact about how the
    pattern was produced, and the reference's config expresses the rule as an interval, so a loader can honour a
    pattern only by writing it back as one. The three kinds are different situations and have to be reported
    differently:

      - ("implied", step): the full positions inside the output block are evenly spaced from its first block, so
        the pattern can only have come from that interval, and a config claiming a different one is refused.
      - ("underdetermined", None): at most one block is full. Any interval at least as large as the output block
        produces the same pattern, so the pattern does not state an interval; the pattern itself is still
        honoureable, block by block.
      - ("not expressible", None): the full positions are not evenly spaced from the output block's first block,
        so no interval produces this pattern. The file states a rule the reference's config cannot hold, which is
        a refusal rather than a report.
    """
    full = [b for b in range(output_start, len(pattern)) if pattern[b]]
    if len(full) == 0:
        return ("underdetermined", None)
    if len(full) == 1:
        return ("underdetermined", None)
    step = full[1] - full[0]
    if full[0] != output_start or full != [full[0] + i * step for i in range(len(full))]:
        return ("not expressible", None)
    return ("implied", step)


def check_mixer_rule(cfg: HeronConfig, file: GGUFFile) -> list[str]:
    """
    The per-block cache rule, checked against the config the loader built.

    A full-attention block and a sliding-window block have the same tensors, so this cannot be derived from a
    name or a shape: it is the one behaviour fact that only a claim states. Checking it is what makes
    `layer_pattern` load-bearing rather than decorative, and it is the only way a loader can notice that the
    interval it assumed is not the interval the file was written from.

    The check has two directions and they are not the same check:
      - a pattern the config cannot produce is refused, because the module would be built with a different cache
        rule than the file states;
      - a pattern that no interval can produce is refused on its own, because the reference expresses the rule as
        an interval and there is no interval that would give this file's pattern. A pattern with at most one full
        block is not refused: it is honoureable by any interval at least as large as the output block, it just
        cannot be traced back to one, which is why the interval is reported as unrecorded.
    """
    pattern = [int(x) for x in file.metadata.get("general.arch.heron.layer_pattern", [])]
    problems = []
    if len(pattern) != cfg.n_blocks():
        problems.append(f"layer_pattern has {len(pattern)} entries, config expects {cfg.n_blocks()}")
        return problems

    if len(file.block_map()["output"]):
        full = [b for b, code in enumerate(pattern) if int(code)]
        if full and full[0] != cfg.output_start_block():
            problems.append(
                f"layer_pattern: the first full-attention block is {full[0]}, the config's output block starts at "
                f"{cfg.output_start_block()}; a full block can only be in the output block"
            )
        kind, implied = attention_interval(pattern, cfg.output_start_block())
        if kind == "not expressible":
            problems.append(
                f"layer_pattern: the full-attention blocks {full} are not evenly spaced from the output block's "
                "first block, so no interval produces this pattern; the loader cannot honour it because the "
                "reference expresses the rule as an interval"
            )
        elif kind == "implied" and implied != cfg.full_attention_every:
            problems.append(
                f"full_attention_every: the config uses {cfg.full_attention_every}, the pattern implies "
                f"{implied}; the loader would build a different cache rule than the file states"
            )

    for bid, code in enumerate(pattern):
        mode = cfg.mixer_mode(bid)
        if int(mode == "full") != code:
            problems.append(
                f"block {bid}: the file states a {'full' if code else 'sliding-window'} cache rule, the "
                f"loader's config computes {mode}"
            )
    return problems


def _target(model: HeronLM, module_path: str) -> tuple[Any, str]:
    """
    The module that owns a path, and the name at the end of it.

    `experts.1.0.ffn.gate.weight` walks two ModuleList indices; `router.prototypes` stops at the router and
    leaves `prototypes` as a plain attribute. That distinction is not cosmetic: `load_state_dict` cannot carry the
    prototypes at all, because the router stores them as a tensor rather than as a Parameter, so a
    state-dict-driven loader would produce a model with no router and no error.
    """
    parts = module_path.split(".")
    obj = model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj, parts[-1]


def assign(model: HeronLM, module_path: str, values: np.ndarray) -> None:
    """
    Put the file's values into the module, without normalizing them on the way in.

    A scalar is recorded as a one-element tensor because gguf tensor-info needs n_dims >= 1, so the element count
    is what has to match rather than the shape: the file's (1,) is the module's 0-dimensional Parameter, and the
    value is unchanged by that.
    """
    owner, attr = _target(model, module_path)
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    current = getattr(owner, attr)
    if isinstance(current, torch.nn.Parameter):
        if tensor.numel() != current.numel():
            raise ValueError(f"{module_path}: {tensor.numel()} elements do not match {tuple(current.shape)}")
        current.data = tensor.reshape(current.shape)
    else:
        # Not a Parameter, so there is no .data to write into. This is the router's case, and it is the only
        # reason this function branches on the type at all: the branch exists because of one real asymmetry, not
        # as a general nicety.
        if tensor.shape != current.shape:
            raise ValueError(f"{module_path}: {tuple(tensor.shape)} does not match {tuple(current.shape)}")
        setattr(owner, attr, tensor)


def coverage(model: HeronLM, files: list[GGUFFile], loaded: dict[str, str], blob: TableBlob | None = None) -> dict[str, str]:
    """
    Every tensor the reference has - parameters and buffers together - and whether any of these files can supply
    its bytes.

    Reported rather than assumed, because the gaps are the interesting part and each has a different reason:

      - the hash table is not a tensor, so it has no name and can only come from the blob;
      - the generalist's tensors have no destination in any file, because the generalist is each expert
        block's base FFN. They are non-persistent buffers rather than parameters, so they are outside the saved
        state in both directions: no checkpoint carries them and the load path does not expect them. A
        checkpoint that does carry them came from a model where slot 0 was an ordinary expert, and the CLI
        refuses it before any file is written;
      - the router's prototypes are not a Parameter, so a state dict would not carry them.

    The reasons come from `converter.tensor_name`, the same mapping the writer used, so the two directions cannot
    disagree about what a name is. What is *not* a gap is decided by `loaded`, because a parameter can be supplied
    in a form a name lookup cannot express: `stream.read_weights` has no single name, and its bytes arrive as one
    row per block. Reporting it as a gap whenever the lookup raises would be reporting the mapping's shape as
    though it were the file's.

    Slot 0 has to be reported apart from the parameter loop, because slot 0 is not in `named_parameters()`. Its
    tensors are still paths the mapping covers, and the mapping refuses them, so the reason here is the refusal
    itself rather than a sentence written next to it.
    """
    held = {n for f in files for n in f.tensors}
    out: dict[str, str] = {}
    covered = set(loaded)
    for path, _ in model.named_parameters():
        if path in covered:
            continue
        if path == "stream.read_weights":
            # Rows, which is the only form this parameter can take in a file: `tensor_name` raises for it by
            # design, so the check has to be made before the lookup rather than after it. A parameter supplied as
            # n element selections is supplied; reporting it as a gap whenever the lookup raises would report the
            # mapping's shape as though it were the file's.
            if not any(p.startswith("stream.read_weights[") for p in covered):
                out[path] = "the claim names one row per block; no file holds a row"
            continue
        try:
            name = converter.tensor_name(path, model.cfg)
        except ValueError as e:
            out[path] = str(e)
            continue
        if name is None:
            out[path] = "not a tensor in any file; only the blob can supply it"
            continue
        if name not in held:
            out[path] = f"named {name!r} by the claim, but no file holds it"

    # Slot 0, which the parameter loop cannot see: its tensors are non-persistent buffers, so they are outside
    # the saved state in both directions. They are still paths TENSOR_NAMES covers, and the mapping refuses them,
    # so the reason reported here is the mapping's own refusal - caught rather than allowed to propagate, because
    # coverage is a report and a report is the answer to the question, not an exception. That is what makes this a
    # statement about the name rather than about the number of bytes: the generalist that computes is each expert
    # block's base FFN, already named by blk.{bid}.ffn_*, so these paths have no name and no destination. The count
    # is the module's shape, not a number chosen here - four tensors per slot, times the slots the reference
    # instantiates.
    for path, _ in model.named_buffers():
        if path in covered:
            continue
        try:
            out[path] = str(converter.tensor_name(path, model.cfg))
        except ValueError as e:
            out[path] = str(e)

    # The one thing neither a state dict nor a parameter loop carries on its own: the router stores its
    # prototypes as a tensor, not as a Parameter. Reported here so the asymmetry shows up in the same report as
    # the other gaps rather than only in a comment.
    if "router.prototypes" not in covered:
        name = converter.tensor_name("router.prototypes", model.cfg)
        if name not in held:
            out["router.prototypes"] = (
                f"named {name!r} by the claim, but no file holds it; it is not a Parameter, so a state dict "
                "would not carry it either"
            )

    if blob is not None:
        out.pop("memory.table.weight", None)
    return out


@dataclass
class LoadedModel:
    path: str
    cfg: HeronConfig
    model: HeronLM
    files: list[GGUFFile]
    blob: TableBlob | None = None
    loaded: dict[str, str] = field(default_factory=dict)  # module path -> which file and name supplied its bytes
    gaps: dict[str, str] = field(default_factory=dict)    # module path -> why no file can supply its bytes
    unrecorded: dict[str, str] = field(default_factory=dict)

    def forward(self, tokens: torch.Tensor, **kwargs):
        """
        Run the reference's forward on the loaded weights.

        The memory path is not run here: the table is read host-side through the blob, so a probe has to be fed in
        as `slots_ids`. Calling forward without them is the honest form of "the file describes the model except
        for the blob", and the tests assert the blob separately rather than pretending a GGUF file contains a
        probe op.
        """
        return self.model.forward(tokens, **kwargs)

    def derived(self) -> dict[str, str]:
        """
        What the loaded model's pattern implies about a fact no file states.

        `unrecorded` says what a file cannot state; this says what the file's data implies anyway. The two are not
        the same statement and a report that merged them would lose the distinction: a pattern with two or more
        full blocks implies its interval, and a pattern with one full block implies nothing at all, even though
        both are honoureable by a config that holds an interval.
        """
        core = [f for f in self.files if f.slot() is None][0]
        pattern = [int(x) for x in core.metadata.get("general.arch.heron.layer_pattern", [])]
        kind, implied = attention_interval(pattern, self.cfg.output_start_block())
        full = [b for b, code in enumerate(pattern) if int(code)]
        if kind == "implied":
            return {"full_attention_every": f"implied {implied} by layer_pattern {full}"}
        if kind == "underdetermined":
            return {
                "full_attention_every": (
                    f"not implied: {full or 'no'} block(s) are full, so any interval at least as large as the "
                    f"output block ({self.cfg.output_blocks}) produces this pattern"
                )
            }
        return {
            "full_attention_every": f"not implied: {full} cannot be produced by any interval, so the loader "
            "cannot honour this pattern and refuses it"
        }

    def report(self) -> str:
        """The loaded model as a report, so the gaps are visible to a person rather than only to an assert."""
        lines = [f"{self.path}  {len(self.files)} file(s)  {sum(len(f.tensors) for f in self.files)} tensors"]
        for path, reason in sorted(self.gaps.items()):
            lines.append(f"  gap  {path}  {reason}")
        for name, why in sorted(self.unrecorded.items()):
            lines.append(f"  unrecorded  {name}  {why}")
        for name, what in sorted(self.derived().items()):
            lines.append(f"  derived  {name}  {what}")
        return "\n".join(lines)


def load_model(files: GGUFFile | list[GGUFFile], cfg: HeronConfig | None = None, blob: TableBlob | None = None) -> LoadedModel:
    """
    Rebuild a module from files.

    Every tensor a file holds has to land somewhere, and every place it lands has to be a place the file's claim
    says it belongs to. A name the claim cannot explain is refused; a parameter the reference has that no file can
    supply is reported with its reason, not silently left at its initialization.

    Two structures are assembled rather than assigned, because the file and the module hold them differently:

      - `stream.read_weights`: one tensor in the module, one tensor per block in the file. The rows are read in
        block order and assembled only if every row agrees with the shape the file claims for the shared tensor.
        A file that wrote the aliases as n separate tensors would look identical byte for byte and load
        identically, so what is asserted here is the structure the claim states, not only the values.
      - the router's prototypes, which are not a Parameter, so they cannot come through a state dict.

    `files` may be one file or several. A core file plus the per-expert files is the whole model; a core file
    alone is the model with its routed experts missing, which is a normal thing to load and a thing the report has
    to say.
    """
    files = [files] if isinstance(files, GGUFFile) else list(files)
    core = [f for f in files if f.slot() is None]
    if len(core) != 1:
        raise ValueError(f"load needs exactly one core file, got {len(core)}")

    if cfg is None:
        cfg, un = config_from_metadata(core[0])
    else:
        un = unrecorded()

    for file in files:
        problems = file_problems(file) + check_mixer_rule(cfg, file)
        if problems:
            raise ValueError(f"{file.path}: loader refuses the file: " + "; ".join(problems))

    model = HeronLM(cfg)
    loaded: dict[str, str] = {}
    rows: dict[int, tuple[GGUFFile, str]] = {}

    for file in files:
        for name in file.tensors:
            path = module_path(name, file, cfg)
            if path.startswith("stream.read_weights["):
                # An element selection on the shared tensor, so it cannot be assigned as a submodule; it is
                # collected here and assembled below.
                rows[int(name.split(".")[1])] = (file, name)
                loaded[path] = f"{file.path}  {name}"
                continue
            # Keyed by the module path rather than by the tensor name, because the name is not unique across a
            # set of files: two per-expert files hold the same names for the same slot positions, and only their
            # slot claims tell them apart. Keyed by name, the second file's entry overwrote the first's and the
            # report said a parameter was never supplied although its bytes were assigned.
            assign(model, path, value_for(name, file, cfg))
            loaded[path] = f"{file.path}  {name}"

    if rows:
        shape = [int(x) for x in core[0].metadata["general.arch.heron.gr.read_weights_shape"]]
        rows = dict(sorted(rows.items()))
        if any(value_for(name, file, cfg).shape != tuple(shape[1:]) for file, name in rows.values()):
            raise ValueError(f"read-weight rows are not the shape the file states: {shape}")
        if len(rows) != shape[0]:
            raise ValueError(
                f"the file states {shape[0]} read-weight rows and holds {len(rows)}; the shape fact is what "
                "rules out a file that dropped one row and still looked consistent"
            )
        assembled = torch.stack([torch.from_numpy(value_for(name, file, cfg)) for file, name in rows.values()], dim=0)
        if assembled.shape != tuple(shape):
            raise ValueError(f"assembled read weights are {tuple(assembled.shape)}, the file states {shape}")
        model.stream.read_weights.data = assembled.reshape(model.stream.read_weights.shape)

    if blob is not None:
        # The table is not a tensor, so this is the only way its bytes can reach the module. Writing the
        # dequantized rows is what `memory.read` sees, and it is the same arithmetic the engine does at probe
        # time: code x the stored scale.
        model.memory.table.weight.data = torch.from_numpy(blob.values()).reshape(model.memory.table.weight.shape)

    gaps = coverage(model, files, loaded, blob)

    return LoadedModel(core[0].path, cfg, model, files, blob, loaded, gaps, un)

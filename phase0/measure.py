"""
Per-branch magnitudes, for gate G0.3.

The claim being measured is not "branches are nice to have", it is a specific pattern: one branch is
written once and barely updated afterwards, so what landed in it stays readable at every loop pass. That
pattern has to be observed from the outside, which is what this file does.

It observes by wrapping the stream's read and write, calling the originals unchanged and recording what
passed through. The wrapper lives in the harness rather than in gr.py for the reason stated in
phase0/__init__.py: the reference is the parity oracle, and measurement code inside it would make the
parity test depend on harness details. The numbers are the same either way, which is what makes the
wrapper safe to use.

Two quantities are recorded per read: the effective read weight per branch, and the share of the read
attributable to the memory injection. The second is the one the design depends on, and it is measured the
same way in both arms, so the arms are comparable.
"""

import torch
import torch.nn.functional as F

from heronlm.memory import slot_ids
from phase0.corpus import Sample


@torch.no_grad()
def measure_branches(model, sample: Sample, k: int, dev: torch.device | None = None) -> dict:
    """
    Records one forward pass at loop count k.

    No gradients, no state mutation: the wrappers call the originals and return their results unchanged,
    so the forward pass measured here is the same forward pass the reference would run.
    """
    dev = dev or torch.device("cpu")
    cfg = model.cfg
    model.eval()

    record: dict = {"writes": [], "reads": [], "memory_write": None}
    stream = model.stream
    orig_read, orig_write = stream.read, stream.write
    current_loop = [0]  # the read that precedes a write is the same sublayer call, so the loop pass is known

    def effective_weights(block: int, loop: int) -> list[float]:
        """
        The mixture the read actually uses.

        With the GR read on, it is the softmax of the per-block, per-pass weights. With it off, the read
        returns branch 0 alone, which is a one-hot mixture - and that is why the off arm must also write
        to branch 0, or the writes would land in branches the read never looks at.
        """
        if not cfg.ngpt_gr_read:
            return [1.0 if b == 0 else 0.0 for b in range(cfg.gr_branches)]
        w = F.softmax(stream.read_weights[block, min(loop, stream.read_weights.shape[1] - 1)], dim=0)
        return [float(x) for x in w]

    def read(state, block, loop):
        out = orig_read(state, block, loop)
        current_loop[0] = loop
        w_eff = effective_weights(block, loop)
        entry = {
            "block": block,
            "loop": loop,
            "weights": w_eff,
            "branch_norms": [float(state[:, b].norm()) for b in range(cfg.gr_branches)],
            "read_norm": float(out.norm()),
        }
        if record["memory_write"] is not None:
            # The recalled value contributes to the read through the branch it was written into, weighted
            # by that branch's share of the mixture. This is the quantity the memory path lives or dies on.
            share = w_eff[record["memory_write"]["branch"]] * record["memory_write"]["norm"]
            entry["memory_share"] = share / entry["read_norm"] if entry["read_norm"] else 0.0
        record["reads"].append(entry)
        return out

    def write(state, block, value):
        # The loop pass is taken from the read that precedes this write, which is the same sublayer call.
        # stream.write does not receive the pass number, so inferring it here is what keeps the reference
        # signature unchanged: the wrapper records, it does not ask the reference to carry extra state.
        record["writes"].append({
            "block": block,
            "branch": stream.branch_of[block],
            "norm": float(value.norm()),
            "loop": current_loop[0],
        })
        return orig_write(state, block, value)

    def inject(state, slots, branch):
        value, gate = model.memory.read(slots)
        contribution = float((gate * model.memory.value_proj(value)).norm())
        record["memory_write"] = {"branch": branch, "norm": contribution, "gate_mean": float(gate.mean())}
        return orig_inject(state, slots, branch)

    stream.read, stream.write = read, write
    orig_inject = model.memory.inject
    model.memory.inject = inject

    try:
        tokens = torch.tensor([list(sample.tokens)], device=dev)
        targets = torch.tensor([list(sample.targets)], device=dev)
        logits = model.forward(tokens, k, None, None, slot_ids_for(model, sample))
        record["loss"] = float(F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
                                               ignore_index=-1)) if int((targets[0] >= 0).sum()) else float("nan")
        record["logits_last"] = logits[0, -1]
    finally:
        stream.read, stream.write = orig_read, orig_write
        model.memory.inject = orig_inject

    return record


def slot_ids_for(model, sample: Sample) -> torch.Tensor:
    """Slot ids for a whole sequence, from the reference's own prefix-aware function."""
    from heronlm.memory import slot_ids

    return slot_ids(torch.tensor([list(sample.tokens)]), model.cfg)


def branch_summary(record: dict) -> dict:
    """
    The pattern, as numbers.

    The pattern is about writers, not counts, and the distinction is the gate. Two sublayers per block mean
    block 0 contributes two writes to the reserved branch, so a count cannot tell "only block 0 owns branch 0"
    from "every block writes branch 0". What the gate claims is that exactly one block writes the reserved
    branch, so `reserved_branch_writers` is the list of blocks that wrote it: in the arm that keeps the
    property it is [0], in the arm that does not it is every block. The loss difference between the arms is
    the other half of gate G0.3, and it comes from the training run, not from this record.
    """
    per_branch: dict[int, dict] = {}
    for w in record["writes"]:
        b = per_branch.setdefault(w["branch"], {"writes": 0, "writers": set(), "total_norm": 0.0})
        b["writes"] += 1
        b["writers"].add(w["block"])
        b["total_norm"] += w["norm"]

    last = record["reads"][-1]
    return {
        "reserved_branch_writers": sorted(per_branch.get(0, {"writers": set()})["writers"]),
        # The memory path writes the reserved branch too, but not through stream.write, so it has to be
        # recorded separately: the rule is "block 0 plus the n-gram injection own branch 0", and a summary
        # that lumped them together could not tell a legitimate memory write from a block overwriting it.
        "memory_branch": record["memory_write"]["branch"] if record["memory_write"] else None,
        "writes_to_reserved_branch": per_branch.get(0, {"writes": 0})["writes"],
        "branches_written": len(per_branch) or 1,
        "per_branch": {b: {"writes": v["writes"], "writers": sorted(v["writers"]), "total_norm": v["total_norm"]}
                        for b, v in sorted(per_branch.items())},
        "memory_share_at_last_read": last.get("memory_share", 0.0),
        "last_read_weights": last["weights"],
        "gate_mean": record["memory_write"]["gate_mean"] if record["memory_write"] else 0.0,
    }

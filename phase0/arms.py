"""
The Phase 0 arms, as single-variable deltas from one baseline.

A gate is only decidable if the two arms differ in exactly one thing. Everything else - seed, steps,
batch, data order, initialization - has to be shared, because a loss difference between two arms is only
attributable to the mechanism being tested if the arms start from the same weights and see the same data
in the same order. That is what `verify_arms` checks, and it is the reason arms are built from one
baseline rather than written out separately.

Two kinds of arm variable:
  - a config flag (norm_mode, ngpt_gr_read, cache_policy). The model's behaviour changes; its size does
    not, which is what keeps the param budget valid for both arms.
  - a run variable (train_k). Same config, different training distribution. Gate G0.6 is the case where
    the variable is not in the config at all.
"""

import dataclasses
from dataclasses import dataclass

from heronlm.config import HeronConfig, PHASE0, TINY


@dataclass(frozen=True)
class RunSpec:
    seed: int = 7
    steps: int = 300
    batch: int = 8
    seq_len: int = 48
    lr: float = 3e-3
    wd: float = 0.01
    # n_train sizes the fixed corpus `make_corpus` builds for the tests and the Phase 1 drivers. The Phase 0
    # gate driver streams fresh samples (SampleStream) and does not use this pool: repeating 64 sequences for
    # a gate-scale run measures memorization, not depth generalization.
    n_train: int = 64
    # An eval set of 16 was too small to separate a depth effect from draw noise once there are four depths
    # and a dozen arms. 32 is still cheap: it is 16 per family over 48- and 48-squared-sequence spaces, so it
    # leaves the majority of each synthetic task available for training.
    n_test: int = 32
    # Trained depth, as the distribution of k seen during training. A model trained only at k=1 does not
    # behave at k=4, which is why the loop curriculum is a gate of its own rather than a training detail.
    train_k: tuple[int, ...] = (1,)
    eval_k: tuple[int, ...] = (1, 2, 3, 4)
    families: tuple[str, ...] = ("lookup", "recurrence")


@dataclass(frozen=True)
class Arm:
    gate: str
    name: str
    cfg: HeronConfig
    spec: RunSpec
    variable: str
    claim: str


def _arm(gate: str, name: str, base: Arm, variable: str, claim: str,
         cfg_changes: dict | None = None, spec_changes: dict | None = None) -> Arm:
    cfg = dataclasses.replace(base.cfg, **(cfg_changes or {}))
    spec = dataclasses.replace(base.spec, **(spec_changes or {}))
    return Arm(gate=gate, name=name, cfg=cfg, spec=spec, variable=variable, claim=claim)


def build_arms(scale: HeronConfig | None = None, spec: RunSpec | None = None) -> dict[str, list[Arm]]:
    """
    The arms, one gate at a time.

    `scale` is the model size. TINY runs on a laptop; PHASE0 is the scale the gates are actually decided
    at and needs a card. The arms are the same in both, which is the point of generating them from the
    config rather than hardcoding a tiny variant: a mechanism that only exists at TINY scale is not the
    mechanism the fork has to reproduce.
    """
    scale = scale or TINY()
    # Eval depth is capped by the config's max_loop. Requesting k=4 on a config whose max_loop is 3 would
    # index read weights past the trained distribution, so the number it produced would not be a measurement
    # of anything. The gate scale is capped by the same rule as the target model.
    spec = dataclasses.replace(spec or RunSpec(), eval_k=tuple(range(1, scale.max_loop + 1)))

    gates: dict[str, list[Arm]] = {}

    # G0.1: does loss still improve past the depth the model was trained at? The trained depth has to sit
    # between the two evaluated depths, or the comparison is not "beyond trained depth" at all. At the gate
    # scale that is k=3 with k=4 beyond it, which is what the roadmap names; at tiny scale max_loop is 3, so
    # the same shape lands on trained depth 2 and evaluated depth 3.
    mid = min(3, scale.max_loop - 1)
    base = Arm(gate="G0.1", name="train@k=1", cfg=scale, spec=spec, variable="", claim="")
    gates["G0.1"] = [
        base,
        _arm("G0.1", f"train@k={mid}", base, "spec.train_k",
             f"trained at depth {mid}, so k > {mid} is the out-of-distribution case",
             spec_changes={"train_k": (mid,)}),
    ]

    # G0.2: the normalized stack has to beat the plain stack specifically above trained depth.
    base = Arm(gate="G0.2", name="ngpt", cfg=scale, spec=spec, variable="", claim="")
    gates["G0.2"] = [
        base,
        _arm("G0.2", "layernorm", base, "cfg.norm_mode",
             "plain standardized stack, same param count",
             cfg_changes={"norm_mode": "layernorm"}),
    ]

    # G0.3: the widened stream, or a plain single residual stream.
    base = Arm(gate="G0.3", name="gr_on", cfg=scale, spec=spec, variable="", claim="")
    gates["G0.3"] = [
        base,
        _arm("G0.3", "gr_off", base, "cfg.ngpt_gr_read",
             "branches collapse to one stream; writes all land in branch 0",
             cfg_changes={"ngpt_gr_read": False}),
    ]

    # G0.5: the approximation the phone has to live with, against the exact version.
    base = Arm(gate="G0.5", name="shared_cache", cfg=scale, spec=spec, variable="", claim="")
    gates["G0.5"] = [
        base,
        _arm("G0.5", "per_loop_cache", base, "cfg.cache_policy",
             "exact per-pass cache, which is a memory multiplier on a phone",
             cfg_changes={"cache_policy": "per_loop"}),
    ]

    # G0.6: the variable here is the training distribution, not a config flag.
    base = Arm(gate="G0.6", name="fixed_k1", cfg=scale, spec=spec, variable="", claim="")
    gates["G0.6"] = [
        base,
        _arm("G0.6", "sampled_k", base, "spec.train_k",
             "per-batch k in 1..max_loop, so every depth is in distribution",
             spec_changes={"train_k": tuple(range(1, scale.max_loop + 1))}),
    ]

    # G0.4 is not represented here. It is a mixer choice measured as device time on the S23 build, and
    # the reference has no GDN path to measure. A PyTorch harness cannot decide it, so no arm is claimed.
    return gates


def arm_variable(a: Arm, b: Arm) -> dict[str, list[str]]:
    """Everything that differs between two arms of the same gate."""
    cfg = [f.name for f in dataclasses.fields(a.cfg) if getattr(a.cfg, f.name) != getattr(b.cfg, f.name)]
    spec = [f.name for f in dataclasses.fields(a.spec) if getattr(a.spec, f.name) != getattr(b.spec, f.name)]
    return {"cfg": cfg, "spec": spec}


def verify_arms(gates: dict[str, list[Arm]]) -> list[str]:
    """
    Returns the problems, empty if the arms are single-variable.

    Two arms must share the seed, the step count, the batch, the data and the initialization. `variable`
    is what the author intended to test; this checks the actual difference rather than trusting the label,
    because an arm that silently differs in two things produces a number that looks like a gate result.
    """
    problems = []
    for gate, arms in gates.items():
        for arm in arms[1:]:
            diff = arm_variable(arms[0], arm)
            found = [f"cfg.{n}" for n in diff["cfg"]] + [f"spec.{n}" for n in diff["spec"]]
            if len(found) != 1:
                problems.append(f"{gate}: {arm.name} differs from {arms[0].name} in {found}")
            elif found[0] != arm.variable:
                problems.append(f"{gate}: {arm.name} labelled variable {arm.variable}, actual {found[0]}")
        for spec_field in ("seed", "steps", "batch", "n_train", "n_test", "seq_len"):
            values = {getattr(a.spec, spec_field) for a in arms}
            if len(values) > 1:
                problems.append(f"{gate}: arms do not share {spec_field}: {values}")
    return problems

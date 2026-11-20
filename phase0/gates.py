"""
The Phase 0 thresholds, as data.

Why the thresholds are a table rather than prose in the driver: every number here came from ROADMAP.md,
and the difference between a threshold and a guess has to be visible. Each row records the source, and
where the roadmap states no number the limit is None rather than a plausible value. That is the whole
point of this file: a gate decided against an invented threshold is not a gate.

Two states that a normal pass/fail check would hide:
  - `not stated`: the roadmap says "by a real margin" or "costs measurable loss" without a number. The
    direction is testable, the magnitude is not, so the verdict is pending until the number is chosen
    from data rather than from the report.
  - `not decidable here`: the threshold is stated at the gate scale (200-250M params), so a tiny run can
    show that a mechanism changes loss at all but cannot decide the number. Both states are reported.

Proposed values are marked as proposals. They exist so the harness produces a number in the right shape,
not so a number can be called a result.

The gate scale ROADMAP states has two parts, model size and token budget. The third part a numeric threshold
needs is an eval size, and ROADMAP does not state one; `GATE_EVAL_TOKENS` and `eval_condition` are the harness
stating its own condition, and they are labelled as that rather than as a roadmap number.

Run through scripts/phase0.py.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Threshold:
    metric: str
    comparison: str          # ">", "<", "direction", "pattern"
    limit: float | None
    source: str
    proposal: bool = False


@dataclass(frozen=True)
class Gate:
    id: str
    phase: str
    claim: str
    thresholds: tuple[Threshold, ...]
    # Some gates cannot be decided by a PyTorch harness at all, which has to be stated rather than worked
    # around. G0.4 is device time on the S23 build; there is no GDN path in the reference to measure.
    harness_decidable: bool
    reason: str


GATES: dict[str, Gate] = {
    "G0.1": Gate(
        id="G0.1",
        phase="Phase 0",
        claim="Loss still improves when k exceeds the trained depth.",
        thresholds=(
            # ROADMAP: "k = 3 beats k = 1 on held-out by a real margin". The margin is not a number, so
            # it cannot be checked as a number; an arm that wins by 0.001 nats on tiny data is not a pass.
            Threshold("depth_gain", "direction", None, "ROADMAP line 96, 'by a real margin' - no number stated"),
            Threshold("degradation_beyond_trained", "<", 0.02, "ROADMAP line 96"),
        ),
        harness_decidable=True,
        reason="Fail means elastic depth is not real and the design changes fundamentally.",
    ),
    "G0.2": Gate(
        id="G0.2",
        phase="Phase 0",
        claim="The normalized stack beats the plain LayerNorm stack specifically at k above trained depth.",
        thresholds=(
            # Direction only, as the roadmap states it. No magnitude, and the arms are param-count-neutral
            # so a difference cannot be attributed to size.
            Threshold("normalized_beats_plain_above_trained_depth", "direction", None, "ROADMAP line 97"),
        ),
        harness_decidable=True,
        reason="If it does not, the loop is decorative. Half of this gate is the mixer, measured on the S23.",
    ),
    "G0.3": Gate(
        id="G0.3",
        phase="Phase 0",
        claim="A branch shows the frozen-write pattern AND removing it costs measurable loss.",
        thresholds=(
            # The pattern is a set, not a magnitude: the reserved branch has exactly one block writer.
            Threshold("reserved_branch_written_by_one_block", "pattern", None, "ROADMAP line 98"),
            Threshold("loss_cost_of_removing", ">", 0.02, "proposal, not in ROADMAP - pending measurement", True),
        ),
        harness_decidable=True,
        reason="Otherwise drop it from v1: it is engine work with no evidence yet.",
    ),
    "G0.4": Gate(
        id="G0.4",
        phase="Phase 0",
        claim="Looped-SWA vs looped-GDN, measured on the S23 build.",
        thresholds=(
            Threshold("device_time", "direction", None, "ROADMAP line 99"),
        ),
        harness_decidable=False,
        reason="Device time on the phone, not loss on a GPU. The reference has no GDN path, so no arm is claimed.",
    ),
    "G0.5": Gate(
        id="G0.5",
        phase="Phase 0",
        claim="Shared cache loses under 0.05 nats against the exact per-loop cache at k = 4.",
        thresholds=(
            Threshold("shared_cache_loss_at_k4", "<", 0.05, "ROADMAP line 100"),
        ),
        harness_decidable=True,
        reason="Per-loop cache is a memory multiplier on a phone, so the approximation has to be priced.",
    ),
    "G0.6": Gate(
        id="G0.6",
        phase="Phase 0",
        claim="Per-batch k sampling produces a model that is good at every k individually.",
        thresholds=(
            # "Good at every k" has no number in the roadmap. What is stated is the failure mode: a model
            # trained only at k = 1 will not behave at k = 4. So the gate is read as two conditions - the
            # failure mode must actually appear in the fixed arm, and the sampled arm must not show it.
            Threshold("fixed_arm_degrades_beyond_trained_depth", "direction", None, "ROADMAP line 101"),
            Threshold("sampled_no_degradation_beyond_k1", "<", 0.02, "proposal, not in ROADMAP - pending measurement", True),
        ),
        harness_decidable=True,
        reason=(
            "The variable is the training distribution, so this gate needs two training runs, not one model at two "
            "depths. The arms are not compute-neutral: a model that samples k in 1..3 receives more average passes "
            "per step than one fixed at k = 1, so a between-arm loss difference cannot be attributed to the "
            "distribution alone. That is why the gate is measured as degradation within each arm rather than as "
            "arm A minus arm B."
        ),
    ),

    # The QAT decision, stated as two conditions rather than as one number.
    # ROADMAP line 158 requires routing accuracy to be measured on the quantized model, which is the
    # measurable form of "PTQ must not change which path a token takes". It states no pass number, so the
    # path half is direction-only and the adoption number below is a proposal, not a gate.
    "QAT": Gate(
        id="QAT",
        phase="Phase 0",
        claim="PTQ must not change which path a token takes, and the quantization error has to be worth the training-side complexity.",
        thresholds=(
            Threshold("ptq_does_not_change_paths", "direction", None, "ROADMAP line 158 (routing accuracy measured on the quantized model); the path rule itself is the proposal, no number stated", True),
            Threshold("qat_adopt_if_delta_above", ">", 0.05, "proposal, not in ROADMAP - pending measurement", True),
        ),
        harness_decidable=True,
        reason=(
            "A path decision is not a rounding detail: the router slot, the memory gate and the GR read mixture decide "
            "which sublayers a token actually passes through. The read mixture is excluded from quantization for "
            "exactly that reason, so its path cannot change. QAT buys quality, not speed, and does nothing for the "
            "table rows or the scalars."
        ),
    ),

    # Phase 1 now has an implementation, so this gate is decided by the harness rather than by a future run.
    # What changed: the guard is a rule the harness can check at any size, so it is a pattern threshold; the
    # three measured numbers still cannot be decided below the conditions they are stated for, and the delta is
    # only attributable to the memory path because the decode path cannot change loss.
    "P1.1": Gate(
        id="P1.1",
        phase="Phase 1",
        claim="Table gates: hit rate, delta loss, speculative acceptance.",
        thresholds=(
            Threshold("hit_rate", ">", 0.40, "ROADMAP line 105"),
            Threshold("delta_loss", ">=", 0.02, "ROADMAP line 105. Measurable only through the memory path: the decode path verifies proposals and never blends, so it cannot change loss. Positive means the path improves loss.", True),
            Threshold("speculative_acceptance", ">=", 0.25, "ROADMAP line 105"),
            # The guard, as a rule rather than a number. It decides at any scale because it is not a
            # measurement: an eval sequence inside the build corpus is a leak at any size.
            Threshold("eval_disjoint_from_build", "pattern", None, "ROADMAP line 105, 'provably disjoint from every eval set'; the overlap check is reported by phase0/table.py"),
            # The conditions the three numbers are stated for. A table built at min_cnt 2 is not a smaller
            # version of the same measurement, it is a different measurement.
            Threshold("min_cnt_at_stated", "pattern", None, "ROADMAP line 105, min_cnt >= 3"),
            Threshold("top_k_at_stated", "pattern", None, "ROADMAP line 105, top-K = 4"),
        ),
        harness_decidable=True,
        reason=(
            "Implemented in phase0/table.py: the guard, the recall axis as its own denominator, the n-gram length "
            "distribution, and the memory path at the stated dims and after int8. The reported-only requirements "
            "(overlap check, length distribution, verbatim recall as a separate axis) are printed by the driver "
            "and are not thresholds, because ROADMAP asks for them as reports. The delta cannot be decided below "
            "the stated 256 dims, so at tiny scale it stays not measured rather than being reported as a reading "
            "at 64 dims."
        ),
    ),
    "P2.1": Gate(
        id="P2.1",
        phase="Phase 2",
        claim="Router gates: conversation-level accuracy, flapping, generalist not worse out of domain.",
        thresholds=(
            Threshold("routing_accuracy", ">=", 0.85, "ROADMAP line 113"),
            Threshold("turn_to_turn_flapping", "<", 0.10, "ROADMAP line 113"),
            Threshold("generalist_not_worse_out_of_domain", "direction", None, "ROADMAP line 113"),
        ),
        harness_decidable=False,
        reason="Needs 500-1,000 domain-tagged conversations; the hysteresis and margin values are fixed here, not later.",
    ),
}


def gate(gate_id: str) -> Gate:
    return GATES[gate_id]


# --- The eval size, stated before the run rather than after it ------------------------
#
# ROADMAP line 95 gives the training budget for a gate run (1-2B tokens per arm) and says nothing about the
# eval size. The numeric thresholds (0.02 and 0.05 nats) are stated at that budget, so the eval size is part
# of the scale the thresholds are decided at: a reading has to be large enough for its error bar to sit
# comfortably inside the threshold, otherwise the number exists and the decision does not.
#
# The error bar is computed over windows, not over tokens, and that distinction is the whole point. A held-out
# window is one document: the losses at its positions are one observation of that document split into seq_len
# pieces, not seq_len independent observations. Dividing the dispersion by sqrt(tokens) gives a number an order
# of magnitude smaller and would make a 32-window run look decisive at 0.02 nats, which is the mistake this
# constant exists to prevent.
#
# `WINDOW_LOSS_STD` is an assumption named as an assumption. It is the spread of held-out loss between
# documents, and it is not measured here: nothing in the harness has a corpus yet, so the only honest form is a
# named value a run can replace with its own measured one. What is not an assumption is the consequence - at
# 32 windows the error bar is at least 0.05 nats for any dispersion that is not far below 0.3, which is already
# larger than the 0.02 threshold G0.1 states. The gate cannot be decided at that eval size whatever the
# dispersion turns out to be.

# 4M held-out tokens: about 1,950 windows at seq 2048, which puts the error bar near 7e-3 nats under the
# assumption above - an order of magnitude below the smallest numeric threshold in the table. The number is
# chosen so the condition is met with margin, not so a particular reading can be made to pass.
GATE_EVAL_TOKENS = 4_000_000
WINDOW_LOSS_STD = 0.3
SE_FACTOR = 2.0


def eval_condition(windows: int, seq_len: int, *, gate_id: str = "G0.1",
                   window_loss_std: float = WINDOW_LOSS_STD) -> dict:
    """
    Whether an eval set of `windows` held-out windows can resolve the gate's numeric thresholds.

    Reported as a condition with its assumption visible, not as a result: `standard_error` is what the assumed
    window dispersion implies, `assumption` names the value it came from, and `run_measured` says plainly that
    no corpus has been measured yet. A run that measures its own window dispersion passes it here, and the
    record then states the dispersion the decision was made under.

    `per_token_error` is included because it is the number a careless calculation produces, and the difference
    between the two is the point of the function.
    """
    g = gate(gate_id)
    numeric = [t for t in g.thresholds if t.limit is not None and t.comparison != "pattern"]
    limit = min(t.limit for t in numeric) if numeric else None
    tokens = windows * seq_len
    se = window_loss_std / math.sqrt(windows) if windows else float("inf")
    needed_windows = None if limit is None else int(math.ceil((window_loss_std * SE_FACTOR / limit) ** 2))
    return {
        "gate": g.id,
        "windows": windows,
        "tokens": tokens,
        "assumed_window_loss_std": window_loss_std,
        "standard_error": se,
        "smallest_numeric_limit": limit,
        "per_token_error": window_loss_std / math.sqrt(tokens) if tokens else float("inf"),
        "needed_windows": needed_windows,
        "needed_tokens": None if needed_windows is None else needed_windows * seq_len,
        "matches_gate_eval_tokens": tokens >= GATE_EVAL_TOKENS,
        "resolves": limit is not None and se <= limit / SE_FACTOR,
        "status": "no numeric threshold to resolve" if limit is None
        else ("resolvable" if se <= limit / SE_FACTOR else "not resolvable"),
        "measured_from_a_corpus": False,
    }


def check(threshold: Threshold, value) -> dict:
    """
    One threshold against one measured value.

    `pending` is a real state, not a hedge: it means either the roadmap stated no number, or the number
    was not measured. Reporting it as pending is what stops a tiny-scale direction from being written as a
    pass in the summary.

    `decides_at_any_scale` is true only for `pattern` thresholds, because those check a rule rather than a
    measurement: "block 0 is the only writer of the reserved branch" is true or false at any size. A measured
    number or a measured direction is not decidable below the scale it was stated for, so verdict() must not
    turn a tiny-scale reading of one into a fail.
    """
    if value is None:
        status = "not measured"
    elif threshold.comparison == "direction":
        status = "pass" if value else "fail"
    elif threshold.comparison == "pattern":
        status = "pass" if value else "fail"
    elif threshold.limit is None:
        status = "not stated"
    elif threshold.comparison == ">":
        status = "pass" if value > threshold.limit else "fail"
    elif threshold.comparison == ">=":
        status = "pass" if value >= threshold.limit else "fail"
    elif threshold.comparison == "<":
        status = "pass" if value < threshold.limit else "fail"
    else:
        raise ValueError(f"unknown comparison: {threshold.comparison}")
    return {
        "metric": threshold.metric,
        "limit": threshold.limit,
        "comparison": threshold.comparison,
        "value": value,
        "status": status,
        "source": threshold.source,
        "proposal": threshold.proposal,
        "decides_at_any_scale": threshold.comparison == "pattern",
    }


def verdict(gate_id: str, measured: dict, scale_label: str, at_gate_scale: bool) -> dict:
    """
    Thresholds against measurements.

    Even when every stated number passes, the overall status stays `indicative` unless the run is at the
    scale the thresholds are stated for. A tiny run cannot decide a threshold stated at 200-250M params;
    it can only show that the mechanism changes loss at all, which is the precondition for the gate being
    decidable later.

    The same restriction applies in the other direction, which is what this edit is about: a tiny run cannot
    produce a `fail` either. A direction reading at tiny scale is an indication, not a decision, and the
    roadmap treats a fail as a design change ("Fail means elastic depth is not real"). The exception is a
    pattern threshold, which checks a rule rather than a measurement and is decided at any size - if the
    harness misimplements the rule, that is a real failure at any scale.

    `at_gate_scale` is the caller's answer to four conditions, not three: the model size, the token budget, a
    corpus of real text, and an eval set large enough to resolve the numeric thresholds (see eval_condition).
    A run at 1B tokens with 32 held-out windows meets three of them and cannot decide a 0.02-nat threshold, so
    the fourth condition is not a formality.
    """
    g = gate(gate_id)
    checks = [check(t, measured.get(t.metric)) for t in g.thresholds]
    if not g.harness_decidable:
        overall = "not decidable by this harness"
    elif any(c["status"] == "fail" and c["decides_at_any_scale"] for c in checks):
        overall = "fail"
    elif any(c["status"] in ("not stated", "not measured") for c in checks):
        # A missing number cannot be made indicative by a small run either: there is nothing to indicate from.
        overall = "pending"
    elif not at_gate_scale:
        overall = "indicative"
    elif any(c["status"] == "fail" for c in checks):
        overall = "fail"
    else:
        overall = "pass"
    return {
        "gate": g.id,
        "claim": g.claim,
        "checks": checks,
        "status": overall,
        "reason": g.reason,
        "scale": scale_label,
        "at_gate_scale": at_gate_scale,
    }

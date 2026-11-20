"""
Phase 0 driver: run the arms, print the numbers, and say what those numbers can and cannot decide.

The honest version of what this script does: at TINY scale it shows that each mechanism changes loss at
all, which is the precondition for a gate being decidable. It cannot decide a threshold stated at 200-250M
params, and it prints that rather than printing a pass. Running it at the PHASE0 scale on a card uses the
same arms and the same thresholds, but the model size is only half of the gate condition: ROADMAP line 95
also states a 1-2B-token budget per arm. The driver prints both and labels a run below either one as a
reading, not a decision.

Training data is streamed, not a fixed pool: `phase0.corpus.SampleStream` generates sample i on demand, so
steps scale the data budget instead of replaying 64 sequences. `--checkpoint-dir` makes a long arm
resumable; `--resume` continues whatever is there; `--checkpoint-keep` bounds how many records survive,
which is what a shared storage quota actually requires.

The driver writes one JSON file per gate under `--record-dir`, as that gate is decided. Printing alone is not
a record: a rented session that ends takes its numbers with it, and a number that cannot be attributed to the
conditions it was measured under - device, torch build, TF32, eval size, corpus, checkpoint policy - is not
evidence later. Without `--record-dir` the run prints and loses, so a gate-scale run should name one.

Run:
  python scripts/phase0.py                                   # tiny arms, whatever device is available
  python scripts/phase0.py --scale phase0 --steps 2000       # gate-size pilot, needs a GPU
  python scripts/phase0.py --scale phase0 --steps 122070 --batch 8 --seq-len 2048 \
      --corpus-dir runs/corpus --eval-windows 2000 \
      --checkpoint-dir runs/phase0 --checkpoint-every 500 --checkpoint-keep 3 \
      --record-dir runs/records --resume
                                                             # 122,070 x 8 x 2048 = 2.0B tokens per arm, which is
                                                             # the ROADMAP line 95 budget; 2,000 windows x 2048 =
                                                             # 4.1M eval tokens, which is what makes a 0.02-nat
                                                             # threshold resolvable. On this machine the same
                                                             # budget is around 634 CPU days, so this is a card
                                                             # command, and throughput on the card has to be
                                                             # measured for one arm before --steps is chosen.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heronlm.config import FULL, PHASE0, TINY
from phase0 import corpus, gates
from phase0.arms import RunSpec, build_arms, verify_arms
from phase0.measure import branch_summary, measure_branches
from phase0.quant import path_stability, ptq_delta
from phase0.train import device, precision_flags, run_gate, set_tf32, training_key
from phase0.tokens import read_manifest

SCALES = {"tiny": TINY, "phase0": PHASE0, "full": FULL}

# Depth is only expected to help on a task with more than one stage, so the family is chosen deliberately
# rather than averaged over both. lookup is the control: if a gate shows a depth gain on lookup too, the
# gain is a harness artifact.
CONTROL_FAMILY = "lookup"
TWO_STAGE_FAMILY = "recurrence"

# ROADMAP line 95 states 1-2B tokens per arm for a gate run. Below the lower bound, even a model at the gate
# size is a pilot: the measured thresholds must stay readings rather than decisions. Keeping this next to the
# verdict call is deliberate, so a short run cannot be printed as a pass because the scale label said phase0.
GATE_TOKEN_BUDGET = 1_000_000_000


def gate_scale_conditions(scale_label: str, tokens_per_arm: int,
                          corpus_train_tokens: int | None, eval_condition: dict) -> dict:
    """
    Whether a run is at the scale the numeric thresholds are stated for, and which parts it is short of.

    Four conditions, not one. The model size comes from the scale, the budget from steps x batch x seq, the
    corpus from --corpus-dir, and the eval size from the resolution argument in gates.eval_condition. A
    synthetic stream can be asked for 1B samples, but its task space is a few thousand sequences, so the token
    count would be repetition rather than a token budget.

    The fourth condition is the one a run can satisfy and still not be able to decide: 1B tokens with 32
    held-out windows meets the first three and cannot resolve a 0.02-nat threshold, because 32 documents give
    an error bar larger than the threshold. Being at the budget is not the same as being able to read it.

    There is one definition, and every gates.verdict call is read against its result. The QAT call once used
    the scale condition alone, which printed a `fail` against a 0.05-nat proposal threshold for a run of 48
    tokens. A fail is a design change in the roadmap's language, so a short run must not be able to print one,
    and the place where that difference shows up is in which value each verdict call receives. The conditions
    are returned as a dict rather than as a single boolean so the record can name which part a run is short of
    - a run that is at the gate size and short of the budget is a different run from one that is short of both.
    """
    checks = {
        "model_size_at_gate_scale": scale_label == "phase0",
        "token_budget_met": tokens_per_arm >= GATE_TOKEN_BUDGET,
        "real_text_corpus_and_the_budget_fits_inside_it": (corpus_train_tokens is not None
                                                           and tokens_per_arm <= corpus_train_tokens),
        "eval_resolves_the_smallest_numeric_threshold": eval_condition["resolves"],
    }
    short_of = [name for name, ok in checks.items() if not ok]
    return {"at_gate_scale": not short_of, "conditions": checks, "short_of": short_of}


def family_roles(corpus_dir: str | None, family: str | None,
                 control_family: str | None) -> tuple[str, str | None, tuple[str, ...], list[str]]:
    """
    Which family's number the gate reads, which family that number is read against, and which families the run
    has to draw. Refusals are returned as a list rather than raised, so the driver can refuse a run before it
    spends compute on a comparison it cannot make.

    The run must draw both families: the gate's number is meaningless without the control, and a run cannot
    report a number for a family it never draws. So these arguments drive the data, not only the label. The
    control is put first because that is the order RunSpec already had, which keeps a run that passes neither
    argument the same run as before, with the same data order and the same run key.

    A real corpus has one family, "text". The synthetic family names do not exist in it, and asking for one would
    compare NaN losses; the control-family check is specific to the synthetic design and is reported as not
    applicable rather than as a control that showed no effect. On that path the arguments cannot be handed a
    family they do not describe: a run cannot read "needle" out of a token corpus, and a corpus has no second
    task to act as a control.
    """
    if corpus_dir:
        problems = []
        if family not in (None, "text"):
            problems.append(f"--family {family} cannot be read from a token corpus. A tokenized corpus has one "
                            f"family, \"text\"; the synthetic family names are tasks, not text, and the run would "
                            f"be measuring a family it never drew")
        if control_family is not None:
            problems.append("--control-family is not applicable to a token corpus: a control is a second task "
                            "measured the same way, and a corpus has one family")
        if problems:
            return family, control_family, ("text",), problems
        return "text", None, ("text",), []

    read, control = family or TWO_STAGE_FAMILY, control_family or CONTROL_FAMILY
    if read == control:
        return read, control, (read,), [
            f"--family and --control-family both name {read}. A control has to be a different task: the whole "
            f"point of the control is that a depth gain appearing on it as well cannot be attributed to depth"
        ]
    return read, control, tuple(dict.fromkeys((control, read))), []


def value(entry: dict, k: int, family: str | None = None) -> float:
    ev = entry["by_k"][k]
    return ev["loss"] if family is None else ev["by_family"].get(family, float("nan"))


def measured_for(gate_id: str, entries: list[dict], family: str,
                 control_family: str | None = CONTROL_FAMILY) -> dict:
    """
    The numbers one gate's thresholds are written against.

    Where the roadmap states a direction rather than a number, the measured value is a boolean and the
    threshold stays unstated in gates.py. That is deliberate: a boolean cannot be made to look like a pass.
    """
    base, other = entries[0], entries[1]
    deepest = max(max(e["by_k"]) for e in entries)

    if gate_id == "G0.1":
        # Trained depth comes from the arm's own spec, not from its name.
        deep_arm = max(entries, key=lambda e: e["arm"].spec.train_k[0])
        trained = deep_arm["arm"].spec.train_k[0]
        beyond = [k for k in sorted(deep_arm["by_k"]) if k > trained]
        return {
            "trained_depth": trained,
            "depth_gain": bool(beyond and value(deep_arm, max(beyond), family) < value(deep_arm, trained, family)),
            "degradation_beyond_trained": (max(value(deep_arm, k, family) - value(deep_arm, trained, family) for k in beyond)
                                           if beyond else None),
            # The same check on the control family, which should show nothing. If it shows a gain, the gain
            # is an artifact of the harness rather than of depth. A real-text corpus has no control family;
            # None records that the check is not applicable instead of reporting a NaN comparison as false.
            "depth_gain_on_control_family": (
                bool(beyond and value(deep_arm, max(beyond), control_family)
                     < value(deep_arm, trained, control_family))
                if control_family is not None else None
            ),
        }

    if gate_id == "G0.2":
        # Both arms train at the same depth, so the comparison is made above that depth and reported at it
        # too, where the two should be close.
        return {
            "normalized_beats_plain_above_trained_depth": bool(value(base, deepest, family) < value(other, deepest, family)),
            "loss_at_trained_depth": {base["arm"].name: value(base, base["arm"].spec.train_k[0], family),
                                      other["arm"].name: value(other, other["arm"].spec.train_k[0], family)},
            "loss_at_deepest_k": {base["arm"].name: value(base, deepest, family), other["arm"].name: value(other, deepest, family)},
        }

    if gate_id == "G0.3":
        # The pattern is observed from the arm that is supposed to keep it; the loss cost comes from the pair.
        record = measure_branches(base["result"]["model"], base["result"]["corpus_test"][0], deepest, base["result"]["device"])
        summary = branch_summary(record)
        return {
            # The rule as it is written in gr.py: block 0 owns the reserved branch and every other block
            # rotates away from it. Compared against the writers, not a write count, because each block
            # writes twice (attention then FFN).
            "reserved_branch_written_by_one_block": summary["reserved_branch_writers"] == [0],
            "loss_cost_of_removing": value(other, deepest, family) - value(base, deepest, family),
            "reserved_branch_writers": summary["reserved_branch_writers"],
            "memory_branch": summary["memory_branch"],
            "branches_written": summary["branches_written"],
            "memory_share_at_last_read": summary["memory_share_at_last_read"],
        }

    if gate_id == "G0.5":
        # The sign is the convention the threshold is written against: positive means the shared cache is the
        # worse of the two, which is what "shared cache loses under 0.05 nats" means. A negative delta is not
        # a pass by accident; it means the approximation came out better than the exact version at this size.
        delta = value(base, deepest, family) - value(other, deepest, family)
        # The roadmap states this number at k = 4. A config whose max_loop is 3 cannot be run at 4: the read
        # weights are sized [blocks, max_loop, branches], so a deeper pass indexes out of range. Putting a
        # k = 3 number under a k = 4 metric name would be a measurement under the wrong label, so the value
        # is None here and the gate reports as not measured rather than as a small pass.
        at_k4 = bool(4 in base["by_k"] and 4 in other["by_k"])
        return {
            "shared_cache_loss_at_k4": delta if at_k4 else None,
            "deepest_evaluated_k": deepest,
            # Whether the two policies produce different numbers at all, which is the precondition for the
            # gate. It is reported separately from the threshold value because a difference at k = 3 is an
            # indication, not the measurement the gate asks for.
            "arms_differ_at_deepest_k": abs(delta) > 1e-9,
            "loss_at_deepest_k": {base["arm"].name: value(base, deepest, family),
                                  other["arm"].name: value(other, deepest, family)},
        }

    if gate_id == "G0.6":
        # Measured as degradation within each arm, not as arm A minus arm B, for the reason given in
        # gates.py: the sampled arm receives more average passes per step, so a between-arm difference is
        # confounded by compute. Both arms are still reported, as context.
        fixed, sampled = base, other
        fixed_trained = fixed["arm"].spec.train_k[0]
        fixed_within = {k: value(fixed, k, family) - value(fixed, fixed_trained, family) for k in sorted(fixed["by_k"])}
        sampled_min = min(sorted(sampled["by_k"]))
        sampled_within = {k: value(sampled, k, family) - value(sampled, sampled_min, family) for k in sorted(sampled["by_k"])}
        return {
            "fixed_arm_degrades_beyond_trained_depth": bool(fixed_within and max(fixed_within.values()) > 0.0),
            "sampled_no_degradation_beyond_k1": max(sampled_within.values()) if sampled_within else None,
            "fixed_arm_per_k_degradation": fixed_within,
            "sampled_arm_per_k_degradation": sampled_within,
            "between_arm_gap": {k: value(sampled, k, family) - value(fixed, k, family) for k in sorted(fixed["by_k"])},
        }

    return {}


def arm_record(entry: dict) -> dict:
    """One arm as the record states it: the labels, the numbers, and the conditions the numbers came from."""
    arm = entry["arm"]
    result = entry["result"]
    return {
        "arm": arm.name,
        "variable": arm.variable or "(baseline)",
        "claim": arm.claim,
        "reused": bool(entry.get("reused")),
        "trained_as": entry.get("trained_as"),
        "resumed_from": result["resumed_from"],
        "train_loss": result["train_loss"],
        "by_k": entry["by_k"],
        # What the run's record holds, not what this session computed: on a resumed arm the steps counted
        # include the steps the checkpoint was written at.
        "steps_recorded": result["steps_recorded"],
        "tokens_recorded": result["tokens_recorded"],
        "orders_sha256": result["orders_sha256"],
        "mode": result["mode"],
        "precision": result["precision"],
        "checkpoints_kept": result["checkpoints_kept"],
    }


def write_record(record_dir: str | Path, gate_id: str, conditions: dict, gate_result: dict,
                 measured: dict, verdict_dict: dict) -> Path:
    """
    One gate, written as it is decided.

    Written per gate rather than once at the end because a gate-scale run is the kind of run that ends
    unexpectedly. A single file written at the end loses everything decided before it; a file per gate keeps
    the decisions that were already made, including the conditions they were made under.

    Every record carries the same `conditions` block, so a record read a month later does not have to be
    compared against a command line that no longer exists.
    """
    path = Path(record_dir) / f"{gate_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conditions": conditions,
        "gate": gate_id,
        "claim": gates.gate(gate_id).claim,
        "measured": measured,
        "verdict": verdict_dict,
        "arms": [arm_record(e) for e in gate_result["arms"]],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=tuple(SCALES), default="tiny")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--family", default=None,
                    help=f"the family whose number the gate reads (the task with more than one stage). Defaults "
                         f"to {TWO_STAGE_FAMILY}. The run draws this family, so the gate cannot be read on a "
                         f"family the run does not draw.")
    ap.add_argument("--control-family", default=None,
                    help=f"the family the gate's number is read against, drawn by the same run under the same "
                         f"seed and the same conditions. Defaults to {CONTROL_FAMILY}. A depth gain that also "
                         f"appears on the control is a harness artifact, so the control has to be a task where "
                         f"extra passes have nothing to do. Not applicable to a token corpus, which has one "
                         f"family.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--corpus-dir", default=None,
                    help="train on uint32 token shards instead of the synthetic families (see "
                         "scripts/tokenize_corpus.py)")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override the sequence length; a real-text gate run wants a long window")
    ap.add_argument("--batch", type=int, default=None, help="override the batch size")
    ap.add_argument("--eval-windows", type=int, default=None,
                    help="held-out windows, in place of the spec default of 32. 32 windows cannot resolve a "
                         "0.02-nat threshold: the error bar over 32 documents is larger than the threshold. "
                         f"gates.GATE_EVAL_TOKENS ({gates.GATE_EVAL_TOKENS:,}) is the size the harness states "
                         "as its condition, and the run prints whether the set it was given resolves.")
    ap.add_argument("--eval-tokens", type=int, default=None,
                    help="same setting in tokens rather than windows; cannot be given with --eval-windows")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="write one checkpoint per arm here, so a long run can resume after a crash")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="steps between checkpoints; 0 writes only when an arm finishes")
    ap.add_argument("--checkpoint-keep", type=int, default=0,
                    help="how many checkpoint records survive per arm; 0 keeps every record the run writes. "
                         "The run prints how many records its interval produces, and that number is what the "
                         "storage is: at the ROADMAP budget with a save every 500 steps it is a few hundred "
                         "files per arm (about 244 at 122,070 steps), which at 3.34 GB per record is more than "
                         "any free-tier quota. A bounded N is what a shared box needs - 3 is enough to keep a "
                         "resume point and one earlier record for comparison.")
    ap.add_argument("--record-dir", default=None,
                    help="write one JSON record per gate here, as each gate is decided")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 for fp32 matmul and cuDNN. Off by default because the gate numbers are "
                         "meant to be comparable with the reference on CPU; a run that turns it on for "
                         "throughput produces a different number, and the record says which arithmetic it is.")
    ap.add_argument("--resume", action="store_true",
                    help="continue any checkpoint found in --checkpoint-dir; a missing one starts fresh")
    args = ap.parse_args(argv)
    if args.resume and args.checkpoint_dir is None:
        ap.error("--resume needs --checkpoint-dir")
    if args.checkpoint_every and args.checkpoint_dir is None:
        ap.error("--checkpoint-every needs --checkpoint-dir")
    if args.checkpoint_keep and args.checkpoint_dir is None:
        ap.error("--checkpoint-keep needs --checkpoint-dir (there is nothing to rotate otherwise)")
    if args.eval_windows is not None and args.eval_tokens is not None:
        ap.error("--eval-windows and --eval-tokens are the same setting in two units; give one")

    scale = SCALES[args.scale]()
    overrides = {}
    if args.seq_len is not None:
        overrides["seq_len"] = args.seq_len
    if args.batch is not None:
        overrides["batch"] = args.batch
    spec = RunSpec(steps=args.steps, seed=args.seed, **overrides)
    # The eval size is part of the scale the numeric thresholds are stated at, so it is chosen here rather than
    # left at the spec default, and the window count is what the record reports.
    #
    # Two things the argument cannot be allowed to do silently. A partial window is not an observation, so a
    # token count that is not a whole number of windows is refused rather than rounded down - the difference
    # between 1,953 and 1,954 documents is small, but a run that reports a rounded number is a run whose eval
    # size cannot be checked against the condition it was chosen for. And an eval set of zero windows produces
    # a loss of nan rather than an error, which is a number that looks like a reading.
    if args.eval_tokens is not None:
        if args.eval_tokens < spec.seq_len:
            ap.error(f"--eval-tokens {args.eval_tokens} is shorter than one window of {spec.seq_len}; "
                     f"there is no eval set to report")
        if args.eval_tokens % spec.seq_len:
            whole = args.eval_tokens - args.eval_tokens % spec.seq_len
            ap.error(f"--eval-tokens {args.eval_tokens} is not a whole number of windows at seq_len "
                     f"{spec.seq_len}; a partial window is not an observation. Give {whole} tokens, or use "
                     f"--eval-windows {whole // spec.seq_len}")
        overrides["n_test"] = args.eval_tokens // spec.seq_len
        spec = dataclasses.replace(spec, **overrides)
    elif args.eval_windows is not None:
        if args.eval_windows <= 0:
            ap.error(f"--eval-windows {args.eval_windows} is not an eval set; the loss it produces is nan, "
                     f"which is a number that looks like a reading")
        overrides["n_test"] = args.eval_windows
        spec = dataclasses.replace(spec, **overrides)
    eval_windows = spec.n_test

    family, control_family, drawn, problems = family_roles(args.corpus_dir, args.family, args.control_family)
    for p in problems:
        ap.error(p)
    spec = dataclasses.replace(spec, families=drawn)
    eval_cond = gates.eval_condition(eval_windows, spec.seq_len)
    dev = device()
    # Set before the first matmul, so the record describes the arithmetic that produced the numbers rather than
    # the arithmetic the code intended. The flags are read back from torch: on a CPU build TF32 is not a choice
    # the code makes, and the record should not imply that it was.
    precision = set_tf32(args.tf32)
    tokens_per_arm = spec.steps * spec.batch * spec.seq_len
    # A corpus has a finite number of non-overlapping windows. `TokenStream` wraps after an epoch, so a run
    # that asks for more tokens than this is re-reading text it has already trained on, and the budget line
    # would overstate the data seen. Count only full windows: a shard contributes (tokens - 1) // seq_len.
    corpus_train_tokens = None
    corpus_val_windows = None
    synthetic_supply = None
    if args.corpus_dir:
        manifest = read_manifest(args.corpus_dir)
        corpus_train_tokens = sum(
            ((int(s["tokens"]) - 1) // spec.seq_len) * spec.seq_len
            for s in manifest["shards"] if s["split"] == "train"
        )
        corpus_val_windows = sum(
            (int(s["tokens"]) - 1) // spec.seq_len for s in manifest["shards"] if s["split"] == "val"
        )
        # Checked here rather than inside run_gate, because the alternative is discovering it after a day of
        # training: an eval set larger than the val split cannot be evaluated at all, so the run would spend
        # its compute and produce no number for the gate it was run for.
        if eval_windows > corpus_val_windows:
            ap.error(f"eval set of {eval_windows} windows exceeds the {corpus_val_windows} full windows the "
                     f"val split holds at seq_len {spec.seq_len}; add validation text, lower --eval-windows, "
                     f"or shorten the run")
    else:
        # The synthetic path cannot be handed an eval size it cannot supply, and the reason is not the count but
        # what the count is made of. Each synthetic task admits a fixed number of distinct sequences, so an eval
        # set larger than that space is the same documents counted more than once. The part that stops a run is
        # the consequence of that: once the eval set covers a family's whole space there is no sequence outside
        # it for training to draw, so the first training step fails on a family the run cannot name. Probed
        # before training, for the same reason the val-split capacity is probed before training - discovering it
        # afterwards means the run spent its compute and produced no number.
        synthetic_supply = corpus.eval_supply(spec.seed, spec.seq_len, eval_windows, spec.families)
        # `can_draw` is None when the draw was not run, so `not can_draw` covers both states: a probed run that
        # cannot draw, and an unprobed one the harness cannot promise will. Both are refused before compute.
        if not synthetic_supply["can_draw"]:
            ap.error(synthetic_supply["refusal"])
    # The same condition every gate is read against, computed once (see gate_scale_conditions).
    scale_conditions = gate_scale_conditions(args.scale, tokens_per_arm, corpus_train_tokens, eval_cond)
    at_gate_scale = scale_conditions["at_gate_scale"]

    print("Phase 0 harness")
    print(f"  scale    : {args.scale} (d={scale.d_model}, blocks={scale.n_blocks()}, max_loop={scale.max_loop})")
    print(f"  device   : {dev}")
    print(f"  precision: torch {precision['torch']} (cuda build {precision['torch_cuda_build']}), "
          f"tf32 matmul={'on' if precision['matmul_tf32'] else 'off'}, "
          f"cudnn={'on' if precision['cudnn_tf32'] else 'off'}")
    if precision["matmul_tf32"]:
        print("             TF32 is on. These numbers are not the same numbers the CPU reference produces, so")
        print("             they can only be compared with another run that states the same setting.")
    print(f"  steps    : {spec.steps}, batch {spec.batch}, seq {spec.seq_len}, seed {spec.seed}")
    print(f"  samples  : {spec.steps * spec.batch:,} streamed per arm, {spec.n_test} held out")
    print(f"  tokens   : {tokens_per_arm:,} per arm")
    print(f"  eval     : {eval_windows} windows x {spec.seq_len} = {eval_windows * spec.seq_len:,} tokens")
    print(f"             error bar over {eval_windows} documents, under an assumed window dispersion of "
          f"{eval_cond['assumed_window_loss_std']} nats: {eval_cond['standard_error']:.4f} nats")
    print(f"             {eval_cond['status']} for the smallest numeric threshold "
          f"({eval_cond['smallest_numeric_limit']}, {eval_cond['gate']}); the rule is that the error bar must "
          f"be at most half the threshold ({gates.SE_FACTOR} standard errors).")
    if eval_cond["needed_windows"]:
        print(f"             would need {eval_cond['needed_windows']} windows "
              f"({eval_cond['needed_tokens']:,} tokens at this seq_len)")
    print("             the dispersion is an assumption named in the record, not a measurement: no corpus has")
    print("             been measured yet, so a run that measures its own window dispersion replaces it.")
    if corpus_val_windows is not None:
        print(f"             val split capacity at this seq_len: {corpus_val_windows} windows")
    else:
        print("             the eval windows here come from the synthetic task space, not from documents, so the")
        print("             window-level error bar is the wrong shape for it: task draws are not documents.")
        print(f"             what the task can supply for an eval set, per family. `space` is the number of")
        print(f"             distinct sequences the task admits, which is a property of the task, not of the draw:")
        # Only a probed run reaches this: an unprobed one is refused by ap.error above, so the counts here are
        # measurements rather than None. If the refusal ever moved, this block would have to print the bounds as
        # bounds, which is what the record already distinguishes.
        for f, row in sorted(synthetic_supply["per_family"].items()):
            space = "not a countable pool" if row["space"] is None else row["space"]
            left = "unknown" if row["training_sequences_left"] is None else row["training_sequences_left"]
            print(f"               {f:<10} {row['windows']} windows drawn from a space of {space}: "
                  f"{row['distinct_sequences']} distinct, {row['repeats']} repeats, {left} left for training")
        alt = gates.eval_condition(synthetic_supply["distinct_sequences"], spec.seq_len)
        print(f"             the denominator this run actually has is {synthetic_supply['distinct_sequences']} "
              f"distinct sequences, not {eval_windows} documents. Over that count the same rule gives "
              f"{alt['standard_error']:.4f} nats, which is the number a synthetic run should be read against.")
    if args.corpus_dir:
        print(f"  corpus   : {args.corpus_dir} (real token shards, each window never crosses a shard)")
        print(f"             {corpus_train_tokens:,} non-overlapping train tokens at seq {spec.seq_len}")
    else:
        print(f"  family   : {family} is the number this gate reads; {control_family} is the control; the run "
              f"draws {', '.join(spec.families)}")
        if synthetic_supply["repeated_windows"]:
            print(f"             the eval set is not {eval_windows} documents: it contains "
                  f"{synthetic_supply['distinct_sequences']} distinct sequences, so "
                  f"{synthetic_supply['repeated_windows']} of its windows are the same document counted more than "
                  f"once. The gate's error bar cannot be read against the count the run asked for.")
        if synthetic_supply["max_supply"] is None:
            print("             no bound can be stated for this run: the bound comes from the task spaces, and at")
            print("             least one family here has no countable space. What this run's draw covered is the")
            print("             per-family table above, and the fact that it can draw at all is a property of this")
            print("             draw, not of the task.")
        else:
            print(f"             largest eval set this task can guarantee to leave a sequence for training: "
                  f"{synthetic_supply['max_supply']} windows. Above that a run only works because its draw "
                  f"happened to miss part of a space, so the per-family numbers above are what this run's draw "
                  f"actually covered.")
    if args.corpus_dir and tokens_per_arm > corpus_train_tokens:
        print(f"  budget   : the run draws {tokens_per_arm:,} tokens from {corpus_train_tokens:,} available, so")
        print("             it re-reads text after one epoch. The measured numbers stay readings until the")
        print("             corpus covers the budget (or the run is shortened to one pass over it).")
    if args.scale == "phase0" and tokens_per_arm >= GATE_TOKEN_BUDGET and args.corpus_dir is None:
        print("  budget   : the token count is met by repetition, not by new text. The synthetic families")
        print("             contain a few thousand distinct sequences, so this is not the 1-2B-token gate even")
        print("             though steps x batch x seq reaches it; pass --corpus-dir for a real data budget.")
    if args.checkpoint_dir:
        print(f"  ckpt dir : {args.checkpoint_dir} (every {args.checkpoint_every or 'arm end'} steps, "
              f"keep={args.checkpoint_keep or 'everything'}, "
              f"resume={'on' if args.resume else 'off'})")
        if args.checkpoint_every:
            # The last record is written at the end of the run when the interval does not land on the final
            # step, so the number of records a run writes is ceil(steps / interval), not floor.
            total = (spec.steps + args.checkpoint_every - 1) // args.checkpoint_every
            # What will exist on disk, not what the policy could allow: a run that writes two records under a
            # policy of three has two, and the storage figure has to be the number the quota actually sees.
            kept = min(args.checkpoint_keep, total) if args.checkpoint_keep else total
            print(f"             records written per arm: {total}, of which {kept} survive under keep="
                  f"{args.checkpoint_keep or 'all'}.")
            print(f"             At the 3.34 GB per record tests/preflight.py measures for one PHASE0 arm, "
                  f"that is {kept * 3.34:.1f} GB per arm.")
            print("             Records are counted per unique training run, not per arm label: six runs cover")
            print("             the ten arm entries, because the five gate baselines are one run through")
            print("             training_key. Six is the number the quota sees.")
            if not args.checkpoint_keep:
                print("             keep=everything is not a policy on a shared box: at this interval the storage")
                print("             is reached long before the token budget is, so a gate-scale run needs a")
                print("             bounded --checkpoint-keep.")
    if not args.record_dir:
        print("  record   : nothing is written to disk. A rented session that ends takes its numbers with it,")
        print("             and a gate number that cannot be attributed to the conditions it was measured under")
        print("             (device, torch build, tf32, eval size, corpus) is not evidence later. Pass")
        print("             --record-dir for a run you intend to decide a gate with.")
    if args.record_dir:
        print(f"  record   : one JSON per gate under {args.record_dir}, written as each gate is decided")
    if args.scale == "phase0" and tokens_per_arm < GATE_TOKEN_BUDGET:
        print("  budget   : below the 1-2B tokens/arm ROADMAP line 95 states, so measured thresholds below stay")
        print("             readings rather than decisions even though the model is at the gate size.")

    gates_by_arm = build_arms(scale, spec)
    problems = verify_arms(gates_by_arm)
    print("\nArm verification (single-variable, shared seed/steps/batch/data)")
    if problems:
        for p in problems:
            print(f"  PROBLEM {p}")
    else:
        print("  ok: every arm differs from its baseline in exactly one thing")

    # The conditions block is the part of the record that makes a number attributable. A loss without its
    # conditions is a number that cannot be compared with a later run, so the record carries the device, the
    # torch build, the TF32 state, the eval size it resolved or did not resolve, the corpus it read, and the
    # checkpoint policy that produced it.
    conditions = {
        "scale": args.scale,
        "device": str(dev),
        "precision": precision,
        "tf32_requested": bool(args.tf32),
        "spec": dataclasses.asdict(spec),
        "family_read": family,
        "control_family": control_family,
        "synthetic_supply": synthetic_supply,
        "tokens_per_arm": tokens_per_arm,
        "corpus_dir": args.corpus_dir,
        "corpus_train_tokens": corpus_train_tokens,
        "corpus_val_windows": corpus_val_windows,
        "eval_condition": eval_cond,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_keep": args.checkpoint_keep,
        "at_gate_scale": at_gate_scale,
        # The same decision, kept as its parts. `at_gate_scale` alone would say a run cannot decide without
        # saying which part it is short of, and a run short of the eval size is a different problem from a run
        # short of the token budget.
        "gate_scale_conditions": scale_conditions,
    }

    verdicts, measured_by_gate, gate_results = {}, {}, {}
    # One training run is shared by every gate whose baseline is identical. The cache lives for the driver
    # process; entries are pruned after a gate unless they are the shared baseline, so only one model plus the
    # current gate's models are resident at a time.
    cache: dict[str, dict] = {}
    baseline_entry = None
    baseline_key = None
    for gate_id, arm_list in gates_by_arm.items():
        results = run_gate(gate_id, arm_list, dev,
                           checkpoint_dir=args.checkpoint_dir, checkpoint_every=args.checkpoint_every,
                           checkpoint_keep=args.checkpoint_keep, resume=args.resume,
                           corpus_dir=args.corpus_dir, cache=cache)
        gate_results[gate_id] = results
        measured = measured_for(gate_id, results["arms"], family, control_family)
        measured_by_gate[gate_id] = measured
        verdicts[gate_id] = gates.verdict(gate_id, measured, args.scale, at_gate_scale)

        print(f"\n  {gate_id} - {gates.gate(gate_id).claim}")
        for entry in results["arms"]:
            if entry.get("reused"):
                print(f"    {entry['arm'].name}: reused identical training from {entry['trained_as']}")
        for entry in results["arms"]:
            if entry["result"]["resumed_from"] and not entry.get("reused"):
                print(f"    {entry['arm'].name}: resumed at step {entry['result']['resumed_from']}")
        for entry in results["arms"]:
            arm = entry["arm"]
            row = "  ".join(f"k={k}:{value(entry, k, family):6.3f}" for k in sorted(entry["by_k"]))
            print(f"    {arm.name:<16} {row}")
            print(f"      variable: {arm.variable or '(baseline)'} - {arm.claim or 'baseline arm'}")
        for c in verdicts[gate_id]["checks"]:
            # A rule check (pattern) is decided at any size. A measured number or direction is not, so below
            # the scale it was stated for the status has to be labelled as a reading rather than a decision.
            status = c["status"]
            if not c["decides_at_any_scale"] and args.scale != "phase0":
                status = f"{status} as a reading, not a decision"
            print(f"      {c['metric']} = {c['value']}  [{status}]  ({c['source']})")
        extra = {k: v for k, v in measured.items() if k not in {c['metric'] for c in verdicts[gate_id]['checks']}}
        if extra:
            print(f"      context: {extra}")
        print(f"      verdict: {verdicts[gate_id]['status']}")

        # Written per gate, immediately, rather than once at the end. A gate-scale run is the kind of run that
        # ends unexpectedly, and a single file written after every gate loses the decisions already made. The
        # record is the same information the print shows, plus the conditions block, so the print is the human
        # view and the file is the part that survives the session.
        if args.record_dir:
            path = write_record(args.record_dir, gate_id, conditions, results, measured, verdicts[gate_id])
            print(f"      record : {path}")

        # Keep exactly one model alive across gates: the G0.1 baseline, which the PTQ and QAT sections below
        # use. Every arm keeps its by_k numbers, but a PHASE0 model is ~1.8 GB of weights, and ten of them
        # would not fit on the card even if training one at a time does. The cache is pruned to the same key,
        # because the other unique runs are never reused.
        if gate_id == "G0.1":
            baseline_entry = results["arms"][0]
            baseline_key = training_key(baseline_entry["arm"], baseline_entry["result"]["corpus"])
        for entry in results["arms"]:
            if entry is not baseline_entry:
                entry["result"]["model"] = None
        for key in [k for k in cache if k != baseline_key]:
            del cache[key]

    fresh = sum(1 for group in gate_results.values() for e in group["arms"] if not e.get("reused"))
    total = sum(len(group["arms"]) for group in gate_results.values())
    print(f"\nTraining: {fresh} unique runs cover {total} arm entries; {total - fresh} reused an identical run.")

    print("\nPTQ delta, on the trained G0.1 baseline arm")
    # Reuses the arm already trained above rather than retraining it: the same seed would give the same
    # model, but a second training run is a second measurement of the same thing and would let the PTQ
    # number come from a model that is not the one the gate numbers above describe.
    baseline = gate_results["G0.1"]["arms"][0]["result"]
    delta = ptq_delta(baseline["model"], baseline["corpus_test"], baseline["arm"].spec.train_k[0], dev)
    print(f"  scheme    : {delta['scheme']} ({delta['effective_bits']:.4g} effective bits as simulated)")
    print(f"            ROADMAP quotes 4.9 for Q4_K_M; that includes superblock overhead this simulation does not, so the two numbers are not the same scheme.")
    print(f"  order     : {delta['order']} (the rule the engine must follow; metadata records it as general.arch.heron.weight_norm_order)")
    for order in sorted(delta["by_order"]):
        print(f"    {order:<14} loss {delta['by_order'][order]['loss']:6.3f}")
    print(f"  quantization error      : {delta['delta_nats']:.4f} nats")
    print(f"  normalization order gap : {delta['normalization_order_gap_nats']:.4f} nats")
    if delta["normalization_order_gap_nats"] > delta["delta_nats"]:
        print("            At this size the order is a larger effect than the quantization itself, which is why")
        print("            metadata carries it as a rule the engine must follow rather than as a preference.")
    print(f"  params quantized        : {delta['quantized_params']}, excluded on purpose: {delta['excluded_params']}")
    print(f"  QAT rule                : {delta['qat_proposal']['status']}, adopted if delta > {delta['qat_proposal']['adopt_if_delta_above']} nats")

    # The other half of the QAT decision. A loss delta says nothing about it: a model can lose 0.01 nats and
    # still route every conversation to the same expert, or lose nothing and start firing a different slot.
    # These are decisions a forward pass makes, not rounding on values, so they have to be counted.
    paths = path_stability(baseline["model"], baseline["corpus_test"], baseline["arm"].spec.train_k[0], dev)
    print(f"\nPTQ path decisions, same arm, same samples ({len(baseline['corpus_test'])} conversations)")
    for name in ("router_slot", "memory_gate", "gr_read_branch"):
        p = paths["paths"][name]
        if p.get("undecidable"):
            print(f"    {name:<14} undecidable here: {p['reason']}")
            continue
        size = p.get("samples") or p.get("positions") or p.get("blocks")
        line = f"    {name:<14} {p['changes']} changed of {size}"
        if p.get("cannot_change_by_construction"):
            line += " (protected by the exclusion; a change here means the exclusion was violated)"
        print(line)

    qat_measured = {
        "ptq_does_not_change_paths": paths["ptq_does_not_change_paths"],
        "qat_adopt_if_delta_above": delta["delta_nats"],
    }
    # Read against the same condition as every other gate, not against the scale label alone. The scale is one
    # of four conditions, and a run that satisfies only the scale cannot decide the 0.05-nat proposal this gate
    # is read against.
    verdicts["QAT"] = gates.verdict("QAT", qat_measured, args.scale, at_gate_scale)
    print(f"  path decision rule      : {gates.gate('QAT').claim}")
    if paths["undecidable_paths"]:
        print(f"  QAT verdict             : {verdicts['QAT']['status']} ({paths['undecidable_paths']} not counted)")
    else:
        print(f"  QAT verdict             : {verdicts['QAT']['status']}")

    control_shows_gain = bool(measured_by_gate.get("G0.1", {}).get("depth_gain_on_control_family"))
    g05 = measured_by_gate.get("G0.5", {})
    g05_measured = g05.get("shared_cache_loss_at_k4") is not None

    print("\nWhat this run can and cannot decide")
    print(f"  Measured at {args.scale} scale with {tokens_per_arm:,} tokens per arm.")
    if at_gate_scale:
        print("  Model size and token budget are both at the gate scale ROADMAP line 95 states, so measured")
        print("  thresholds below can decide.")
    else:
        print("  A threshold is stated for the gate scale, which has four parts: 200-250M params, 1-2B tokens")
        print("  per arm, real text rather than a repeating synthetic task, and an eval set large enough to")
        print("  resolve the smallest numeric threshold. This run is short of at least one, so a measured")
        print("  number is a reading, not a pass or a fail against the roadmap.")
        print(f"  short of: {', '.join(scale_conditions['short_of'])}")
    for gate_id, v in verdicts.items():
        print(f"  {gate_id}: {v['status']} ({v['reason']})")
    print("  G0.2 has a second half this harness cannot measure: the mixer, on the S23 build.")
    print("  G0.4 cannot be decided by a PyTorch harness at all; the reference has no GDN path.")
    print("  Only a rule check (a pattern) is decided at any size. A measured number is not decided in either")
    print("  direction below the scale it was stated for, so every status printed above is a reading.")
    if args.corpus_dir:
        print("  The synthetic control family is not present in a real-text corpus, so the 'depth gain on the")
        print("  control' check is not applicable. The depth reading is overall held-out loss, which is what G0.1")
        print("  states; there is no separate family here to rule out a harness artifact.")
    elif control_shows_gain:
        print("  Control family shows a depth gain, so at this scale extra passes cannot be separated from extra")
        print("  compute on an undertrained model. A depth gain measured on lookup is a harness artifact, not a")
        print("  result for G0.1; the gate has to be decided on data where the two can be separated.")
    if bool(g05):
        if g05_measured:
            print("  G0.5: measured at k = 4, the depth the roadmap states the number at.")
        else:
            print("  G0.5: not measured. The roadmap states the number at k = 4, but this config's max_loop is")
            print(f"  {scale.max_loop}, so a pass at 4 would index the read weights out of range. The value is")
            print("  None, so the gate reports as not measured rather than as a pass at the wrong depth.")
        if g05.get("arms_differ_at_deepest_k"):
            print("  The two policies do differ at the deepest depth this config can run, so the mechanism is")
            print("  observable here. But a single delta cannot separate the two causes of that difference: the")
            print("  per-pass K/V the shared cache does not keep, and the entries evicted past the window.")
            print(f"  Eval sequences ({spec.seq_len} tokens) exceed the window ({scale.swa_window}) at this scale,")
            print("  so both causes are in play. Measured inside the window the delta would be only the first.")
        else:
            print("  The two policies produced identical numbers, so there is nothing to price here. At k=1 the")
            print("  shared cache is exact by definition; a difference can only appear at depth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

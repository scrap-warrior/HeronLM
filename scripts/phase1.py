"""
Phase 1 driver: the table gates, measured offline on data the table was not built from.

What this script does, in the order it does it: train one arm so that the model and the memory table are not
all zeros, build the static table from the build split only, then measure everything on the test split. The
order matters because a table built from the eval data would answer its own questions, and a model that has
not been trained cannot agree with anything, so acceptance measured on an untrained core is a property of the
initialization rather than a measurement.

Two things it refuses to do:

  - It does not report a delta loss for the static table. The decode path verifies proposals and never blends
    them, so under the design's own rule the table cannot change the numbers at all. The delta that can be
    attributed to memory is the memory path, a model component, and that is what is measured here. Both are
    printed, with the attribution named.
  - It does not report a number under a name the config cannot produce. ROADMAP states the memory path at 256
    dims; at the tiny config the dim is 64, so the value produced there is printed as context and the gate
    value stays None. Same rule as the depth naming in the Phase 0 driver.

The recall axis is the table's top-1 proposal measured against the answer. That is not the model's loss, and
the two are printed as different numbers on purpose: an average over the whole sequence reports the family's
difficulty, not the retrieval, which is the axis ROADMAP asks for.

Run:
  python scripts/phase1.py                                  # tiny arms, whatever device is available
  python scripts/phase1.py --scale phase0 --steps 2000      # the gate scale, needs a GPU
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heronlm.config import FULL, PHASE0, TINY
from phase0 import gates
from phase0.arms import RunSpec, build_arms
from phase0.corpus import MAX_TABLE_ORDER, MIN_CNT, fact_distance, length_for_window, make_corpus
from phase0.table import (
    STATED_MIN_CNT,
    STATED_MEMORY_DIM,
    STATED_TOP_K,
    build_table,
    decode_path,
    fact_coverage,
    memory_path,
    ngram_length_distribution,
    overlap_check,
    recall_axis,
    table_conditions,
    table_support,
)
from phase0.train import device, train_arm

SCALES = {"tiny": TINY, "phase0": PHASE0, "full": FULL}

# The two families ROADMAP line 107 names as the recall axis. Both are kept in the same run because the point
# of the axis is that a needle beyond the window can only be answered by a path outside the window; a quote
# inside the window can be answered by attention, so the two measure different things and cannot replace each
# other.
RECALL_FAMILIES = ("quote", "needle")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=tuple(SCALES), default="tiny")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=16)
    # Each fact has to appear at least MIN_CNT times in the build corpus or its entry is pruned and the recall
    # axis measures an empty table. The default is the smallest corpus that can cover both pools.
    ap.add_argument("--n-build", type=int, default=MIN_CNT * 16)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args(argv)

    scale = SCALES[args.scale]()
    # The corpus sizes belong in the spec, not only in the argument parser: the driver used to read
    # spec.n_train/n_test while parsing --n-train/--n-test, so the flags silently did nothing.
    spec = RunSpec(steps=args.steps, seed=args.seed, n_train=args.n_train, n_test=args.n_test)
    dev = device()

    print("Phase 1 harness")
    print(f"  scale    : {args.scale} (d={scale.d_model}, blocks={scale.n_blocks()}, swa_window={scale.swa_window}, "
          f"memory dim={scale.ngram_memory_dim}, max_loop={scale.max_loop})")
    print(f"  device   : {dev}")
    print(f"  steps    : {spec.steps}, seed {spec.seed}")
    print(f"  families : {', '.join(RECALL_FAMILIES)}")

    # The length the recall families need is not the Phase 0 default. The needle axis only means something when
    # the gap is longer than the sliding window, and at the tiny config the window is 32 while the default
    # sequence is 48 - the fixed blocks and the minimum fillers do not leave room for both. The length is
    # therefore taken from the window, and the driver says so rather than measuring a gap that quietly shrank.
    wanted = length_for_window(scale.swa_window)
    length = max(spec.seq_len, wanted)
    corpus = make_corpus(spec.seed, length, spec.n_train, spec.n_test,
                         families=RECALL_FAMILIES, n_build=args.n_build, window=scale.swa_window)
    print(f"  length   : {length}" + (f" (raised from {spec.seq_len} so the gap can exceed the window; "
                                      f"needs {wanted})" if length > spec.seq_len else ""))
    for split in ("train", "test", "build"):
        print(f"  {split:<8}: {len(corpus[split])} conversations")

    # The gap the needle axis needs, checked against the window rather than assumed from the family name. Two
    # conditions are printed separately because they are not the same condition:
    #   - the rule the harness states, which is what the length was derived from: the gap is longer than the
    #     window. It is conservative - a gap longer than the window means no window ending inside the second
    #     occurrence can reach the first.
    #   - what is physically true at a recall position: the window reaches back `window` tokens from the position
    #     and sees the first occurrence only if the structure between the position and that occurrence is inside
    #     them. A length where the rule fails but the physical condition holds is uncertified rather than
    #     unmeasurable, so the driver says which one failed.
    print("\nRecall structure")
    for family in RECALL_FAMILIES:
        samples = [s for s in corpus["test"] if s.family == family]
        gaps = [s.gap for s in samples]
        rule = sum(1 for g in gaps if g > scale.swa_window)
        reach = sum(1 for s in samples if scale.swa_window < fact_distance(s.family, s.gap))
        positions = sum(len(s.recall_positions) for s in samples)
        print(f"  {family:<8} gap {min(gaps)}-{max(gaps)} vs window {scale.swa_window}; "
              f"{rule} of {len(samples)} pass the rule (gap > window); "
              f"{reach} of {len(samples)} beyond the window's reach at a recall position "
              f"(needs window < {fact_distance(family, max(gaps))}); {positions} scored positions")
        if rule == 0:
            print("            The rule this axis is stated on does not hold at this length, so the numbers below")
            print("            cannot decide the axis.")
            if reach == 0:
                print("            The window can still reach the first occurrence at a recall position, so the")
                print("            table and attention cannot be told apart here.")
            else:
                print("            The fact is still out of the window's physical reach, so the rule is being")
                print("            conservative rather than the measurement being impossible.")

    if MAX_TABLE_ORDER != max(scale.ngram_orders):
        print(f"  WARNING: recall positions are recorded for order {MAX_TABLE_ORDER}, the config's top order is "
              f"{max(scale.ngram_orders)}. The positions would be computed for a different key length.")

    # One arm, trained so the memory table and the core are not zeros. The arms above are not used because
    # Phase 1 is not a single-variable comparison: there is one table and one model, and the comparison is the
    # same model with and without the path on the same sequences.
    arm = build_arms(scale, spec)["G0.1"][0]
    print(f"\nTraining one arm (seed {spec.seed}, variable: {arm.variable or 'baseline'})")
    result = train_arm(arm, corpus, dev)
    model = result["model"]
    print(f"  train loss at last step : {result['train_loss']:.4f}")
    if (model.memory.table.weight == 0).all():
        print("  memory table still all zeros - the numbers below measure an initialization, not a path.")

    table = build_table(scale, corpus["build"])
    cond = table_conditions(table)
    coverage = fact_coverage(table, corpus["build"], corpus["test"])

    print(f"\nTable conditions, against the values ROADMAP line 105 states")
    print(f"  min_cnt {cond['min_cnt']} (stated >= {STATED_MIN_CNT}) [{'ok' if cond['min_cnt_at_stated'] else 'not the stated value'}]")
    print(f"  top_k   {cond['top_k']} (stated = {STATED_TOP_K}) [{'ok' if cond['top_k_at_stated'] else 'not the stated value'}]")
    print(f"  orders  {cond['orders']}, top order {cond['top_order']}, entries {len(table.entries)}")

    print("\nGuard (ROADMAP line 107): the eval set must be disjoint from the build corpus")
    guard = overlap_check(table, corpus["build"], corpus["test"])
    for key in ("sequences_shared_with_build", "eval_disjoint_from_build", "facts_shared_with_build",
                "ngrams_shared_at_recall_positions", "recall_positions_resolving_to_keys_never_seen_in_build"):
        print(f"  {key:<52} {guard[key]}")
    print("  Sequence overlap is the leak and has to be zero. Fact overlap is expected to be non-zero: the")
    print("  answer has to exist somewhere in what the table was built from, otherwise the probe has nothing")
    print("  to return and the recall axis measures nothing.")

    support = table_support(table, corpus["test"])
    total = coverage["answerable_eval_samples"] + coverage["eval_samples_on_facts_below_min_cnt"]
    print(f"\nCoverage, as two different numbers that have to be read against each other")
    print(f"  corpus support : {coverage['answerable_eval_samples']} of {total} eval samples ask for a fact the build corpus")
    print(f"                 contains at least min_cnt {cond['min_cnt']} times")
    print(f"  table support  : {support['facts_resolving_at_top_order']} of {support['eval_samples_probed']} probes resolve at the")
    print(f"                 top order {support['top_order']}; {support['facts_with_no_entry']} have no entry at all, "
          f"{support['facts_resolving_only_through_a_lower_order']} only through a lower order")
    if coverage["eval_samples_on_facts_below_min_cnt"]:
        print(f"  {coverage['eval_samples_on_facts_below_min_cnt']} eval samples ask for a fact seen fewer than min_cnt "
              f"{cond['min_cnt']} times. Those positions are counted as coverage, not as retrieval failures.")
    if coverage["answerable_eval_samples"] != support["facts_resolving_at_top_order"]:
        print("  These two are not equal, and that is a bug rather than a result. Corpus support says the fact is")
        print("  in the corpus often enough to survive min_cnt; table support is what a probe at a recall position")
        print("  can actually return. A gap means the build did not accumulate counts across conversations -")
        print("  min_cnt applied per conversation instead of per corpus, which deletes exactly the entries the")
        print("  table exists to hold.")

    print("\nN-gram length distribution (reported separately, not a gate number)")
    dist = ngram_length_distribution(table)
    for order in dist["orders"]:
        row = dist["by_order"][order]
        print(f"  order {order}: {row['entries']} entries, {row['mean_continuations_per_entry']:.2f} continuations per entry, "
              f"{row['continuations_above_top_k']} above top_k {dist['top_k']}")
    print(f"  entries truncated by top_k: {dist['entries_truncated_by_top_k']}")

    # One measurement, used twice: the recall axis needs the table, and the loss denominators need the model.
    # Running the model twice - once with the path, once without - is the single-variable comparison this part
    # of the gate is built on, so the same numbers feed the axis below rather than a second run.
    mp = memory_path(model, corpus["test"], arm.spec.train_k[0], dev)
    axis = recall_axis(table, corpus["test"], coverage["counts"])
    print("\nRecall axis, scored at the positions that carry the recalled answer")
    for family in RECALL_FAMILIES:
        f = axis.get(family)
        if f is None:
            print(f"  {family:<8} no scored positions")
            continue
        print(f"  {family:<8} {f['samples']} samples, {f['positions']} positions, "
              f"hit rate {f['hit_rate_at_recall_positions']:.3f}, "
              f"top-1 accuracy {f['accuracy_over_answerable_positions'] if f['accuracy_over_answerable_positions'] is not None else 'no answerable positions'} "
              f"over {f['answerable_positions']} answerable positions")
        print(f"            hit orders: {f['hit_orders']}, samples below min_cnt: {f['samples_below_min_cnt']}")
    print("  Accuracy here is the table's top-1 proposal against the answer, not the model's loss. That is the")
    print("  distinction the axis depends on: a model trained long enough can memorize a small fact pool, and")
    print("  a loss number would then credit the table with something the model did.")

    print("\nTwo denominators, printed against each other")
    with_path = mp["bf16"]["with_path"]
    for family, row in sorted(with_path["by_family"].items()):
        whole = row["loss"]
        recall = row["loss_at_recall_positions"]
        print(f"  {family:<8} whole sequence {whole:.4f} over {row['positions']} positions; "
              f"recall positions {recall:.4f} over {row['recall_positions']} positions")
    print("  These are different numbers by construction. If a harness reported them as the same measurement,")
    print("  it would be reporting the family's difficulty rather than the retrieval.")

    print("\nMemory path, at the stated dims and after int8")
    print(f"  dim used  : {mp['dim_used']} (ROADMAP states {mp['stated_dim']}) - "
          f"{'the stated measurement' if mp['measured_at_stated_dim'] else 'not the stated measurement'}")
    print(f"  int8      : {mp['int8_scheme']}")
    for half in ("bf16", "after_int8"):
        r = mp[half]
        on, off = r["with_path"], r["without_path"]
        d = r["delta_nats"]
        print(f"  {half:<10} with path {on['loss']:.4f}, without {off['loss']:.4f}, "
              f"delta {d if d is None else f'{d:+.4f}'} (positive = the path improves loss)")
    if mp["table_untrained"]:
        print("  The table is all zeros, so every delta here is exactly zero for a reason that has nothing to")
        print("  do with the path. Nothing in this section can be read as evidence.")
    if not mp["measured_at_stated_dim"]:
        print("  The deltas above are printed as context only. A gain at 64 dims is not the gain ROADMAP names,")
        print("  so the gate value stays None and the decision about shipping the memory path is not made here.")

    print("\nDecode path, through the reference's own generate")
    dp = decode_path(model, table, corpus["test"], arm.spec.train_k[0], dev, args.max_new_tokens)
    print(f"  hit rate   : {dp['hit_rate']:.3f} over {dp['lookups']} lookups (attempts, not hits)")
    print(f"  by order   : {dp['hit_rate_by_order']}")
    print(f"  acceptance : {dp['acceptance'] if dp['acceptance'] is not None else 'no proposals were made'} "
          f"over {dp['proposals']} proposals on {dp['decode_samples']} samples")
    print(f"  outputs identical to the model-only run: {dp['outputs_identical_without_table']} of {dp['decode_samples']} "
          f"(must be all, otherwise the path blended)")
    print("  The decode path cannot change loss: it verifies proposals and drops the ones the core disagrees")
    print("  with. A delta loss attributed to the static table can therefore only come from a blend, which the")
    print("  design forbids, so the delta is attributed to the memory path above and the decode-path delta is")
    print("  reported as None.")

    measured = {
        "hit_rate": dp["hit_rate"],
        # The only form of the delta this design can measure, and only at the dim it is stated for.
        "delta_loss": mp["after_int8"]["delta_nats"],
        "speculative_acceptance": dp["acceptance"],
        "eval_disjoint_from_build": guard["eval_disjoint_from_build"],
        "min_cnt_at_stated": cond["min_cnt_at_stated"],
        "top_k_at_stated": cond["top_k_at_stated"],
    }
    v = gates.verdict("P1.1", measured, args.scale, args.scale == "phase0")
    print(f"\nGate P1.1 - {v['claim']}")
    for c in v["checks"]:
        status = c["status"]
        if not c["decides_at_any_scale"] and args.scale != "phase0":
            status = f"{status} as a reading, not a decision"
        print(f"  {c['metric']:<28} {c['value']}  [{status}]  ({c['source']})")
    print(f"  verdict: {v['status']}")

    print("\nWhat this run can and cannot decide")
    print(f"  Measured at {args.scale} scale. ROADMAP states these thresholds for the table at the gate scale")
    print("  with the memory path at 256 dims, so a tiny run can show that each mechanism produces a number in")
    print("  the right shape. It cannot decide any of them, in either direction.")
    print("  Only the rule checks - the guard and the stated min_cnt and top-K - are decided at any size,")
    print("  because they check a rule rather than a measurement. If the harness misimplements the guard, that")
    print("  is a real failure at any size.")
    print("  The whole-corpus hit rate here is dominated by the generator rather than by the mechanism: the")
    print("  filler tokens are drawn uniformly from a 48-token sub-vocab, so nearly every n-gram the fillers")
    print("  produce is a singleton and cannot survive min_cnt. A real corpus has boilerplate and recurring")
    print("  phrasings, which is what makes a hit rate above 40% plausible. The hit rate at the recall")
    print("  positions is the part that reflects the mechanism, which is why the axis reports it separately.")
    if not mp["measured_at_stated_dim"]:
        print(f"  The memory-path half stays not measured at this scale: dim {mp['dim_used']} is not the")
        print(f"  {mp['stated_dim']} dims the threshold is stated for. Run at --scale phase0 to produce the number.")
    if not dp["proposals"]:
        print("  No proposals were made, so acceptance has no denominator. On an untrained or near-untrained")
        print("  model the table rarely fires; the number needs the arm trained on data with facts in it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

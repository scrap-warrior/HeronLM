"""
Phase 1: the table gates, measured offline.

Three things this file is deliberate about.

1. The guard is implemented, not just stated. ROADMAP line 105 requires the table to be built from a corpus
   provably disjoint from every eval set and requires the overlap check to be reported. So the check is a
   function here and its number is printed, and it checks the thing that would actually be a leak: an eval
   sequence appearing in the build corpus. Facts shared between the two splits are not a leak - a lookup
   table is only useful if the fact exists somewhere in what it was built from.

2. The recall axis is measured on its own positions. ROADMAP line 107 asks for verbatim recall as a separate
   axis because that is the table's actual job. An average over the whole sequence reports the family's
   difficulty, not the retrieval, so the positions that carry the recalled answer are scored on their own and
   the two denominators have to be different for the axis to be a separate axis.

3. The static table cannot change loss through the decode path, and that has to be said rather than worked
   around. The decode path verifies proposals and never blends them, so a proposal the core would not have
   chosen is dropped and the generated tokens are identical to the model-only run. A "delta loss" attributed to
   the static table is therefore not measurable under the design's own rule: it can only come from a blend,
   which the design forbids. The measurable form of the delta is the memory path, which is a model component
   and can change the numbers. Both are reported, with the attribution named, because a number produced by a
   path that cannot change loss is not evidence.

The memory path's two conditions - the stated 256 dims and int8 - are measured as a pair, because ROADMAP
states them together and the decision they drive is whether the memory path ships at all. At the tiny config
the dim is 64, so the number produced there is not the stated measurement; it is reported as context and the
gate value stays None. Same rule as the depth naming in the Phase 0 driver: a number cannot be reported under
a name the config cannot produce.

Sign convention for delta loss: positive means the path improves loss, because that is what "survives
quantization" has to mean. The convention is written here because a gate checked against a number whose sign
is not stated can be passed by its opposite.

Run through scripts/phase1.py.
"""

import torch
import torch.nn.functional as F

from heronlm.config import HeronConfig
from heronlm.memory import slot_ids
from heronlm.model import HeronLM
from heronlm.ngram import NGramTable
from phase0.corpus import Sample
from phase0.quant import DENSE_INT4, Q8_0, TABLE_INT8, fake_quantize_model

# ROADMAP line 105 states the memory path measurement at 256 dims, min_cnt >= 3 and top-K 4. A config whose
# values are not those cannot produce the stated measurement, so the harness reports the values it actually
# used rather than calling a reading at other values a measurement of the gate.
STATED_MEMORY_DIM = 256
STATED_MIN_CNT = 3
STATED_TOP_K = 4


def table_conditions(table: NGramTable) -> dict:
    """
    The values the thresholds are stated for, checked against the config the table was built with.

    These are rule checks rather than measurements, so they decide at any scale: if the harness builds a table
    at min_cnt 2 and reports a hit rate against a threshold stated for min_cnt 3, the number is not the gate's
    number. A table built at the wrong min_cnt is not a smaller measurement, it is a different measurement.
    """
    return {
        "min_cnt": table.cfg.ngram_min_cnt,
        "min_cnt_at_stated": table.cfg.ngram_min_cnt >= STATED_MIN_CNT,
        "top_k": table.cfg.ngram_top_k,
        "top_k_at_stated": table.cfg.ngram_top_k == STATED_TOP_K,
        "orders": sorted(table.cfg.ngram_orders, reverse=True),
        "top_order": max(table.cfg.ngram_orders),
        "memory_dim": table.cfg.ngram_memory_dim,
        "memory_dim_at_stated": table.cfg.ngram_memory_dim == STATED_MEMORY_DIM,
    }


def build_table(cfg: HeronConfig, samples: list[Sample]) -> NGramTable:
    """
    Build from the build split only, one sequence at a time, then prune once at the end.

    Building over a concatenation would create n-grams that straddle conversation boundaries, which are not
    facts the model would ever see at decode time, and it would make the overlap check unable to say which
    sequence a counted n-gram came from. Per-sequence building is what keeps the guard meaningful.

    Pruning is corpus-level, so it happens after every sequence has been counted. Pruning after each sequence
    would delete any fact whose support is spread across conversations - a repeated span contributes two counts
    inside one conversation, which is below min_cnt, so the entry would be gone before the next conversation
    could add to it. That is not a smaller version of the same table, it is a different table, and the harness
    measured exactly that difference: a build corpus of 48 conversations produced a table of 9 entries, all at
    order 1, none of which could have answered a recall question.
    """
    table = NGramTable(cfg)
    for s in samples:
        table.build(list(s.tokens))
    table.prune()
    return table


def fact_counts(build: list[Sample]) -> dict[str, dict[tuple[int, ...], int]]:
    """
    How many build samples each fact appears in, per family.

    This is what turns a recall miss into an interpretable event. An eval sample asking for a fact seen fewer
    than min_cnt times has no entry at all, so a wrong answer there is not the table failing at retrieval - it
    is the fact not being in the table. Counting those positions as retrieval failures would make the table
    look worse than the measurement can support.
    """
    out: dict[str, dict[tuple[int, ...], int]] = {}
    for s in build:
        if not s.fact:
            continue
        row = out.setdefault(s.family, {})
        row[s.fact] = row.get(s.fact, 0) + 1
    return out


def fact_coverage(table: NGramTable, build: list[Sample], eval_samples: list[Sample]) -> dict:
    """
    Which eval facts the table can answer at all, as a count rather than as a rate.

    `eval_samples_on_facts_below_min_cnt` is not a failure number; it is the number of eval samples whose
    answer the build corpus never contained. A coverage that low means the recall axis is measuring a table
    that cannot have an answer for most of its questions, and the driver reports that rather than printing an
    accuracy computed over an empty denominator.
    """
    counts = fact_counts(build)
    min_cnt = table.cfg.ngram_min_cnt
    answerable, unanswerable = 0, 0
    for s in eval_samples:
        if not s.fact:
            continue
        if counts.get(s.family, {}).get(s.fact, 0) >= min_cnt:
            answerable += 1
        else:
            unanswerable += 1
    return {
        "build_facts": sum(len(v) for v in counts.values()),
        "answerable_eval_samples": answerable,
        "eval_samples_on_facts_below_min_cnt": unanswerable,
        "counts": counts,
    }


def table_support(table: NGramTable, eval_samples: list[Sample]) -> dict:
    """
    Which eval questions the table can answer, measured by probing the table rather than by counting the corpus.

    This is a different number from `fact_coverage`, and the two have to be printed against each other. Corpus
    support is what min_cnt is stated against; table support is what a probe at a recall position can actually
    return. They agree only when min_cnt is applied at the corpus level. When corpus support says a fact is
    answerable but the probe finds no entry, the build is wrong rather than the table being weak - which is the
    failure mode the per-sequence pruning had, and the reason this is a separate function rather than a note.

    The probe is at the top order, because that is the key the recall positions were recorded for. A lower-order
    hit here is reported as a fallback: it is a hit, but it is not evidence that the fact entered the table.
    """
    top = max(table.cfg.ngram_orders)
    min_cnt = table.cfg.ngram_min_cnt
    probed, top_order, fallback, absent = 0, 0, 0, 0
    for s in eval_samples:
        if not s.fact or not s.recall_positions:
            continue
        probed += 1
        i = s.recall_positions[0]
        key = tuple(s.tokens[max(0, i - top + 1): i + 1])
        entry = table.entries.get(key)
        if entry is None:
            absent += 1
            continue
        if entry.order == top:
            top_order += 1
        else:
            fallback += 1
    return {
        "top_order": top,
        "min_cnt": min_cnt,
        "eval_samples_probed": probed,
        "facts_resolving_at_top_order": top_order,
        "facts_resolving_only_through_a_lower_order": fallback,
        "facts_with_no_entry": absent,
    }


def overlap_check(table: NGramTable, build: list[Sample], eval_samples: list[Sample]) -> dict:
    """
    The guard ROADMAP asks for, reported as numbers.

    `sequences_shared_with_build` is the leak: an eval sequence inside the build corpus. It has to be zero, and
    the test plants one to check that the check can actually see it.

    `facts_shared_with_build` is expected to be non-zero. That is not the leak, it is the mechanism: the answer
    has to exist somewhere in what the table was built from. A harness that reported zero fact overlap would be
    measuring a recall axis on a table that cannot have any answer.

    The per-order check at recall positions is reported too, because the memory path's table is keyed by hash
    and the same key must land in the same slot in the reference and in the engine. If a key at a recall
    position was never seen at that order in the build corpus, the entry it resolves to came from a collision,
    which is the risk the memory path has to judge rather than assume.
    """
    shared_sequences = sum(1 for s in eval_samples if any(tuple(b.tokens) == tuple(s.tokens) for b in build))

    orders = sorted(table.cfg.ngram_orders, reverse=True)
    top = orders[0]
    build_keys: dict[int, set] = {}
    for b in build:
        toks = list(b.tokens)
        for i in range(len(toks)):
            for order in orders:
                if i + 1 - order >= 0:
                    key = tuple(toks[i + 1 - order: i + 1])
                    if key in table.entries:
                        build_keys.setdefault(order, set()).add(key)

    shared_at_recall = 0
    resolves_to_unseen_key = 0
    for s in eval_samples:
        for i in s.recall_positions:
            key = tuple(s.tokens[max(0, i - top + 1): i + 1])
            if key in build_keys.get(top, set()):
                shared_at_recall += 1
            elif key in table.entries:
                # In the table but never counted at this order in the build corpus, so the row the probe reads
                # belongs to a different n-gram that landed in the same hash slot.
                resolves_to_unseen_key += 1

    return {
        "sequences_shared_with_build": shared_sequences,
        "eval_disjoint_from_build": shared_sequences == 0,
        "facts_shared_with_build": len({s.fact for s in eval_samples if s.fact} & {b.fact for b in build if b.fact}),
        "ngrams_shared_at_recall_positions": shared_at_recall,
        "recall_positions_resolving_to_keys_never_seen_in_build": resolves_to_unseen_key,
    }


def ngram_length_distribution(table: NGramTable) -> dict:
    """
    Reported separately, as ROADMAP line 105 requires.

    This is not a gate number. It is the shape of the table: how many entries each order contributes, and
    whether top-K is truncating real mass. A table whose entries are almost all order 1 is a bag of
    single-token continuations, and a hit rate measured over it means something different from a hit rate over
    4-grams - the same threshold, a different mechanism.
    """
    by_order: dict[int, dict[str, float]] = {}
    for entry in table.entries.values():
        row = by_order.setdefault(entry.order, {
            "entries": 0, "distinct_continuations": 0, "count_sum": 0, "continuations_above_top_k": 0,
        })
        row["entries"] += 1
        row["distinct_continuations"] += len(entry.counts)
        row["count_sum"] += entry.total
        row["continuations_above_top_k"] += max(len(entry.counts) - table.cfg.ngram_top_k, 0)
    for row in by_order.values():
        row["mean_continuations_per_entry"] = row["distinct_continuations"] / row["entries"]
    return {
        "orders": sorted(by_order, reverse=True),
        "by_order": by_order,
        "top_k": table.cfg.ngram_top_k,
        "entries_truncated_by_top_k": sum(1 for e in table.entries.values() if len(e.counts) > table.cfg.ngram_top_k),
    }


def recall_axis(table: NGramTable, eval_samples: list[Sample], counts: dict[str, dict[tuple[int, ...], int]]) -> dict:
    """
    Verbatim recall, scored at the positions that carry the recalled answer.

    Two denominators, and they cannot be the same number: hit rate is measured over every position that was
    probed, accuracy over the answerable recall positions only. The axis ROADMAP asks for is the retrieval
    itself, so the whole-sequence average is reported alongside it rather than being replaced by it.

    What accuracy measures is the table's top-1 proposal against the answer, not the model's loss. The
    distinction matters: a model trained long enough can memorize a small fact pool, and a loss number would
    then credit the table with something the model did. The table-only number is the one that can be called
    the table's job.

    A recall position counts as answerable when the fact it asks for survived min_cnt in the build corpus. A
    position whose fact never entered the table is counted as coverage, not as a wrong answer.
    """
    top = max(table.cfg.ngram_orders)
    min_cnt = table.cfg.ngram_min_cnt
    per_family: dict[str, dict] = {}
    for s in eval_samples:
        if not s.recall_positions:
            continue
        f = per_family.setdefault(s.family, {
            "samples": 0, "positions": 0, "hits": 0, "answerable_positions": 0, "top1_correct": 0,
            "hit_orders": {}, "samples_below_min_cnt": 0,
        })
        f["samples"] += 1
        answerable = counts.get(s.family, {}).get(s.fact, 0) >= min_cnt
        if not answerable:
            f["samples_below_min_cnt"] += 1
        for i in s.recall_positions:
            f["positions"] += 1
            hit = table.lookup(list(s.tokens[:i + 1]))
            if hit is None:
                continue
            f["hits"] += 1
            f["hit_orders"][hit[0]] = f["hit_orders"].get(hit[0], 0) + 1
            if not answerable:
                continue
            f["answerable_positions"] += 1
            if hit[1] and hit[1][0][0] == s.targets[i]:
                f["top1_correct"] += 1
    for f in per_family.values():
        f["hit_rate_at_recall_positions"] = f["hits"] / f["positions"] if f["positions"] else None
        f["accuracy_over_answerable_positions"] = f["top1_correct"] / f["answerable_positions"] if f["answerable_positions"] else None
    return per_family


@torch.no_grad()
def path_losses(model: HeronLM, samples: list[Sample], k: int, dev: torch.device, use_memory: bool) -> dict:
    """
    Held-out loss with and without the memory path, on the same weights and the same sequences.

    `use_memory=False` is the no-memory baseline: the same model with the read disabled, which the reference
    expresses as passing no slot ids - the read at block 2 is skipped, so nothing is injected. Same weights,
    same data, so a difference is attributable to the path rather than to a second model or a different order.

    Per-position losses are kept because the recall axis needs its own denominator. A mean over the whole
    sequence is the family's difficulty, not the retrieval, so both are reported and the driver prints the two
    against each other. If the two numbers came out identical, the axis would not be a separate axis.
    """
    cfg = model.cfg
    total, scored = 0.0, 0
    recall_total, recall_scored = 0.0, 0
    by_family: dict[str, dict] = {}
    for s in samples:
        tokens = torch.tensor([list(s.tokens)], device=dev)
        ids = slot_ids(tokens, cfg) if use_memory else None
        logits = model.forward(tokens, k, None, None, ids)

        f = by_family.setdefault(s.family, {
            "samples": 0, "positions": 0, "loss_sum": 0.0,
            "recall_samples": 0, "recall_positions": 0, "recall_loss_sum": 0.0,
        })
        f["samples"] += 1
        if s.recall_positions:
            f["recall_samples"] += 1

        for t in range(len(s.targets)):
            if s.targets[t] < 0:
                continue
            loss = float(F.cross_entropy(logits[0, t].reshape(1, -1), torch.tensor([s.targets[t]], device=dev)))
            total += loss
            scored += 1
            f["positions"] += 1
            f["loss_sum"] += loss
            if t in s.recall_positions:
                recall_total += loss
                recall_scored += 1
                f["recall_positions"] += 1
                f["recall_loss_sum"] += loss

    for row in by_family.values():
        row["loss"] = row["loss_sum"] / row["positions"] if row["positions"] else None
        row["loss_at_recall_positions"] = (row["recall_loss_sum"] / row["recall_positions"]) if row["recall_positions"] else None
        del row["loss_sum"], row["recall_loss_sum"]

    return {
        "use_memory": use_memory,
        "scored_positions": scored,
        "loss": total / scored if scored else None,
        "recall_positions_scored": recall_scored,
        "loss_at_recall_positions": recall_total / recall_scored if recall_scored else None,
        "by_family": by_family,
    }


@torch.no_grad()
def memory_path(model: HeronLM, eval_samples: list[Sample], k: int, dev: torch.device) -> dict:
    """
    The memory path, measured at the stated dims and after int8.

    ROADMAP line 105 pairs the two conditions and the decision they drive: if the memory path does not survive
    reduced dims and quantization, ship the decode path only. Both halves are reported, and the gate values are
    only produced when the measurement is the one ROADMAP states. At tiny scale the dim is 64, so the numbers
    there are the shape of the mechanism at the size the harness can run, and the gate values stay None.

    `table_untrained` is reported because an all-zero table gives exactly the same loss with and without the
    path: a delta of zero there is a property of the initialization, not evidence that the path does not work.
    """
    cfg = model.cfg
    bf16_on = path_losses(model, eval_samples, k, dev, True)
    bf16_off = path_losses(model, eval_samples, k, dev, False)

    q = fake_quantize_model(model, DENSE_INT4, "after_dequant", TABLE_INT8, Q8_0)
    int8_on = path_losses(q, eval_samples, k, dev, True)
    int8_off = path_losses(q, eval_samples, k, dev, False)

    at_stated = bool(cfg.ngram_memory_dim == STATED_MEMORY_DIM)
    # Positive means the path improves loss, which is what "survives" means. ROADMAP states no magnitude for
    # this half, so the direction is the whole claim, and it is only reported at the dim it is stated for.
    def delta(on: dict, off: dict):
        if not at_stated or on["loss"] is None or off["loss"] is None:
            return None
        return off["loss"] - on["loss"]

    return {
        "dim_used": cfg.ngram_memory_dim,
        "stated_dim": STATED_MEMORY_DIM,
        "measured_at_stated_dim": at_stated,
        "int8_scheme": TABLE_INT8.name,
        "bf16": {"with_path": bf16_on, "without_path": bf16_off, "delta_nats": delta(bf16_on, bf16_off)},
        "after_int8": {"with_path": int8_on, "without_path": int8_off, "delta_nats": delta(int8_on, int8_off)},
        # None when the number is not the stated measurement. A gain at 64 dims is not a gain at the dim ROADMAP
        # names, and calling it one would let a tiny config produce a gate result.
        "memory_path_survives_reduced_dims": None,
        "memory_path_survives_quantization": None,
        "table_untrained": bool((model.memory.table.weight == 0).all()),
    }


@torch.no_grad()
def decode_path(model: HeronLM, table: NGramTable, eval_samples: list[Sample], k: int, dev: torch.device,
                max_new_tokens: int = 16) -> dict:
    """
    Hit rate and speculative acceptance, through the reference's own decode path.

    The same samples are decoded twice: once with proposals verified against the model, once with no table at
    all. The outputs have to be identical, because the decode path can only agree or be ignored. If they ever
    differ, the implementation blended, which is the thing the design forbids, and no loss number produced by
    that path could be attributed to the table.

    Acceptance is measured on the proposals, not on the tokens, and lookups count attempts rather than hits -
    the same rule the generate path records, because a hit rate that only counted hits would read 100%.

    The comparison is only meaningful because generate caps its output at the token budget. A verified span can
    cross the budget, since the budget is checked once per loop step rather than once per token inside a span,
    and without the cap the two runs would return different lengths and a mismatch here would be a reporting
    artifact rather than evidence about the path. The cost of that overshoot is still recorded, because it is a
    forward pass the phone really paid for.
    """
    stats: dict[str, int] = {}
    outputs_match = 0
    for s in eval_samples:
        prompt = list(s.tokens)
        with_table, s_stats = model.generate(prompt, max_new_tokens, k, table=table, speculate=True)
        without = model.generate(prompt, max_new_tokens, k)[0]
        # Summed from what generate actually returned, so a stat added on the reference does not silently
        # disappear from the harness. Rates are skipped: they are per-run, not counts, and summing them would
        # produce a number that looks like a total and is not one.
        for key, value in s_stats.items():
            if isinstance(value, float):
                continue
            stats[key] = stats.get(key, 0) + int(value)
        outputs_match += int(with_table[len(prompt):] == without[len(prompt):])

    # hit_stats takes one sequence, so the per-order counts are summed here rather than averaged over a
    # concatenation, which would mix boundaries the table never saw.
    by_order: dict[int, int] = {}
    lookups = 0
    hits = 0
    for s in eval_samples:
        h = table.hit_stats(list(s.tokens))
        lookups += h["lookups"]
        hits += h["hits"]
        for order, n in h["hits_by_order"].items():
            by_order[order] = by_order.get(order, 0) + n

    return {
        # The table's hit rate, measured by probing the eval sequences. This is the number ROADMAP's threshold
        # is about, and it is a different number from what the decode run itself did: the run only proposes on
        # a hit, so its own denominators have to be named separately or the two are read as one measurement.
        "hit_rate": hits / lookups if lookups else None,
        "hit_rate_by_order": by_order,
        "lookups": lookups,
        "hits": hits,
        "entries": len(table.entries),
        "acceptance": stats.get("accepted", 0) / stats["proposed"] if stats.get("proposed") else None,
        "proposals": stats.get("proposed", 0),
        "decode_lookups": stats.get("lookups", 0),
        "decode_hits": stats.get("hits", 0),
        "decode_samples": len(eval_samples),
        # The invariant, as a count: every sample must agree, otherwise the path blended.
        "outputs_identical_without_table": outputs_match,
        # Verification cost paid past the token budget. Recorded separately from forward_passes so the
        # overshoot stays visible: it is a pass that bought tokens which were then truncated.
        "overshoot_tokens": stats.get("overshoot", 0),
        "forward_passes": stats.get("forward_passes", 0),
        # Why this path cannot produce the delta ROADMAP asks for. Reported as None so the gate cannot be
        # decided by a decode-path measurement the design makes impossible.
        "decode_path_can_change_loss": False,
        "decode_path_delta_loss": None,
    }
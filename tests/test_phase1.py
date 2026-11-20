"""
Phase 1 tests: the table gates.

These are not quality tests. A randomly initialized tiny model cannot have a hit rate above 40% by being
good at something, and a 120-step run cannot decide a threshold ROADMAP states at the gate scale. What these
tests check is that the measurement can mean anything:

  - the guard can actually see a leak, so a zero it reports is a real zero rather than a check that cannot
    fire;
  - the recall positions are the positions where the answer comes from: the top-order key there is a piece of
    the fact (for the needle, the key itself rather than a key plus padding), and the answer at that position is
    the continuation of the same piece in the first occurrence;
  - the gap rule and the window's physical reach are two different conditions that can disagree, and the test
    shows the disagreement rather than letting the harness report the conservative rule as if it were the fact;
  - the two denominators the recall axis uses are genuinely different numbers: a hit is not the right answer,
    and a position whose fact never entered the table is coverage, not a retrieval failure;
  - min_cnt at the corpus level and min_cnt per conversation produce different tables, and the difference is
    visible through table_support rather than assumed;
  - the decode path cannot change loss, so the delta ROADMAP asks for is attributed only to the memory path,
    and the sign convention is stated rather than implied;
  - the token budget is a promise about output length, so an accepted span that crosses it is reported as an
    overshoot instead of making two runs of the same model return different lengths.

Run: python tests/test_phase1.py
"""

import sys
from dataclasses import replace
from pathlib import Path

# The project is not installed as a package, so make the project root importable when the tests are run
# directly. Same reason as in test_reference.py and test_phase0.py.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from heronlm.config import TINY
from heronlm.model import HeronLM
from heronlm.ngram import NGramTable
from phase0 import gates
from phase0.corpus import (
    FILLER_PRE,
    KEY_LEN,
    MAX_TABLE_ORDER,
    MIN_CNT,
    MIN_POST,
    PAYLOAD_LEN,
    SPAN_LEN,
    STRUCTURE,
    Sample,
    fact_distance,
    length_for_window,
    make_corpus,
)
from phase0.table import (
    STATED_MEMORY_DIM,
    build_table,
    decode_path,
    fact_counts,
    fact_coverage,
    memory_path,
    ngram_length_distribution,
    overlap_check,
    path_losses,
    recall_axis,
    table_support,
)

DEV = torch.device("cpu")


def threshold(gate_id: str, metric: str):
    """The threshold as ROADMAP states it, taken from gates rather than restated here."""
    for t in gates.GATES[gate_id].thresholds:
        if t.metric == metric:
            return t
    raise AssertionError(f"{gate_id} has no threshold named {metric}")


def targets(seq: list[int]) -> tuple[int, ...]:
    """The same rule the generator uses: the target at position i is the token at position i + 1."""
    return tuple(seq[1:]) + (-1,)


def test_planted_leak_is_visible_to_the_guard():
    """
    ROADMAP line 107 requires the eval set to be provably disjoint from anything used to build the table. A
    guard that reports zero without being able to see a leak is not a guard, so the leak has to be planted.
    """
    cfg = TINY()
    length = length_for_window(cfg.swa_window)
    # A build corpus that cycles the fact pools is what makes fact overlap certain; a smaller one would leave
    # some facts below min_cnt and the assertion below would be about sampling rather than about the guard.
    corpus = make_corpus(11, length, 8, 6, families=("quote", "needle"), n_build=MIN_CNT * 16, window=cfg.swa_window)
    table = build_table(cfg, corpus["build"])

    clean = overlap_check(table, corpus["build"], corpus["test"])
    assert clean["sequences_shared_with_build"] == 0 and clean["eval_disjoint_from_build"], clean

    # Fact overlap is expected to be non-zero and is not the leak: the answer has to exist somewhere in what
    # the table was built from, otherwise the recall axis has no possible answer.
    assert clean["facts_shared_with_build"] > 0, (
        "the guard reported no fact overlap, so the eval set asks for facts the build corpus never contained"
    )

    # The same eval sample as a build sample is the leak, and the guard has to see it.
    leaked = corpus["build"][0]
    dirty = overlap_check(table, corpus["build"], list(corpus["test"]) + [leaked])
    assert dirty["sequences_shared_with_build"] == 1, dirty
    assert not dirty["eval_disjoint_from_build"], dirty

    # The guard is a pattern threshold, so it decides at any scale. This is the harness's own rule: if the
    # guard cannot see a leak, that is a real failure even on a tiny run, whereas a measured number cannot be
    # failed at tiny scale.
    t = threshold("P1.1", "eval_disjoint_from_build")
    assert t.comparison == "pattern", t
    c = gates.check(t, dirty["eval_disjoint_from_build"])
    assert c["status"] == "fail" and c["decides_at_any_scale"], c
    v = gates.verdict("P1.1", {"eval_disjoint_from_build": False}, "tiny", False)
    assert v["status"] == "fail", (
        f"a misimplemented guard must decide at tiny scale, got {v['status']}"
    )
    # The same run with the leak removed cannot be a pass, because the measured thresholds have nothing to
    # decide at tiny scale: the pattern rules pass, the numbers stay pending.
    v_clean = gates.verdict("P1.1", {"eval_disjoint_from_build": True}, "tiny", False)
    assert v_clean["status"] == "pending", v_clean["status"]


def test_recall_positions_record_the_positions_the_answer_comes_from():
    """
    The positions are recorded on the sample rather than recomputed from the family name, which is what makes
    the axis the retrieval rather than the family's difficulty. What has to hold is that the key ending at a
    recorded position is a piece of the first occurrence - the fact - and that the positions before the first
    top-order window are excluded, because their key straddles the gap and the table never saw it.
    """
    cfg = TINY()
    assert max(cfg.ngram_orders) == MAX_TABLE_ORDER, (
        f"the positions assume order {MAX_TABLE_ORDER}, the config's top order is {max(cfg.ngram_orders)}"
    )
    length = length_for_window(cfg.swa_window)
    corpus = make_corpus(5, length, 8, 8, families=("quote", "needle"), n_build=24, window=cfg.swa_window)

    for s in corpus["test"]:
        toks = list(s.tokens)
        if s.family == "quote":
            second_start = FILLER_PRE + len(s.fact) + s.gap
            assert len(s.fact) == SPAN_LEN, s
            pieces = [s.fact[i: i + MAX_TABLE_ORDER] for i in range(len(s.fact) - MAX_TABLE_ORDER + 1)]
            # The recorded positions are exactly those where both things hold at the top order: the key ending
            # there is a piece of the fact, and the answer there is the continuation of that piece inside the
            # fact. Positions before the first top-order window are excluded because their key straddles the gap,
            # which the table never saw; the last span token is excluded because its answer is the trailing
            # filler rather than a fact continuation.
            assert s.recall_positions == tuple(
                range(second_start + MAX_TABLE_ORDER - 1, second_start + len(s.fact) - 1)), s
            for p in s.recall_positions:
                i = p - second_start - MAX_TABLE_ORDER + 1
                key = tuple(toks[p - MAX_TABLE_ORDER + 1: p + 1])
                assert key == pieces[i], (s, p, key)
                assert s.targets[p] == s.fact[i + MAX_TABLE_ORDER], (s, p, key)
            # The key at the first recorded position is the piece that starts the fact, not the whole fact: the
            # fact is SPAN_LEN tokens and the key is MAX_TABLE_ORDER of them. Confusing the two would make the
            # probe look for an entry the table could never have had.
            first = s.recall_positions[0]
            assert first == second_start + MAX_TABLE_ORDER - 1, s
            assert tuple(toks[first - MAX_TABLE_ORDER + 1: first + 1]) == pieces[0], s
            # The positions before it are excluded: their key reaches back across the gap.
            for p in range(second_start, second_start + MAX_TABLE_ORDER - 1):
                key = tuple(toks[p - MAX_TABLE_ORDER + 1: p + 1])
                assert key not in pieces, (s, p, key)
        else:
            # needle: the answer at the last key token is the first payload token, and the key there is the
            # key itself, so the probe lands on the entry built from the first occurrence.
            assert len(s.recall_positions) == 1, s
            p = s.recall_positions[0]
            assert tuple(toks[p - KEY_LEN + 1: p + 1]) == s.fact, s
            assert s.targets[p] != -1, s
            assert len(s.fact) == MAX_TABLE_ORDER, "a key shorter than the top order is only a lower-order fallback"

    # The top order the positions assume has to agree with the config. A config whose top order is longer than
    # the recorded key length measures a different key: the window at the recorded position reaches back to the
    # token before the key, which is padding the table never saw at that order. The driver prints this as a
    # warning; here it is shown as a measurable disagreement rather than as a smaller measurement.
    wider = TINY()
    wider.ngram_orders = (5, 4, 3, 2, 1)
    corpus2 = make_corpus(5, length, 8, 8, families=("needle",), n_build=MIN_CNT * 8, window=wider.swa_window)
    t2 = build_table(wider, corpus2["build"])
    sup = table_support(t2, corpus2["test"])
    assert sup["top_order"] == 5, sup
    assert sup["eval_samples_probed"] == len(corpus2["test"]), sup
    assert sup["facts_resolving_at_top_order"] == 0, sup
    assert sup["facts_with_no_entry"] + sup["facts_resolving_only_through_a_lower_order"] == sup["eval_samples_probed"], sup
    # At the order the positions were recorded for, the same table answers every probe.
    t3 = build_table(TINY(), corpus2["build"])
    sup3 = table_support(t3, corpus2["test"])
    assert sup3["facts_resolving_at_top_order"] == sup3["eval_samples_probed"], sup3


def test_gap_is_clamped_and_the_length_that_lets_it_exceed_the_window_is_derived():
    """
    The needle axis only means something when the payload is out of the window's reach. The gap cannot be
    larger than the room the fixed blocks and the minimum fillers leave, so below the length that fits the
    window the generator has to shrink it - and the harness reports that as not decidable rather than as a
    smaller gap it can measure.

    Two conditions that are easy to collapse into one, and the test keeps them apart:
      - the rule the harness states, which is what length_for_window derives the length from: the gap is longer
        than the window. It is conservative: a gap longer than the window means no window ending inside the
        second occurrence can reach the first occurrence at all.
      - what is physically true at a recall position: the window reaches back `window` tokens from that position,
        and it sees the first occurrence only if the structure between the position and the first occurrence is
        inside those tokens. One token below the derived length, the rule stops holding while the fact is still
        physically out of reach, so the two conditions disagree. The driver prints both, because from the rule
        alone a reader cannot tell whether the axis was unmeasurable or merely uncertified.
    """
    cfg = TINY()
    window = cfg.swa_window
    fit = length_for_window(window)
    assert fit == STRUCTURE + FILLER_PRE + MIN_POST + window + 1, fit

    room = fit - STRUCTURE - FILLER_PRE - MIN_POST
    assert room == window + 1, (room, window)  # the +1 is what makes the gap strictly exceed the window

    corpus = make_corpus(7, fit, 4, 4, families=("quote", "needle"), n_build=16, window=window)
    for s in corpus["test"]:
        assert s.gap > window, f"at {fit} the gap must exceed the window {window}, got {s.gap}"
        assert s.gap == room, "the gap must be the room the fixed blocks leave, not a smaller choice"
        assert window < fact_distance(s.family, s.gap), (s.family, s.gap, fact_distance(s.family, s.gap))

    # One token shorter: the room is exactly the window, so the wanted gap of window + 1 cannot be honoured and
    # the clamp is visible. The rule no longer holds, but the fact is still out of the window's physical reach.
    short = make_corpus(7, fit - 1, 4, 4, families=("quote", "needle"), n_build=16, window=window)
    for s in short["test"]:
        assert s.gap == fit - 1 - STRUCTURE - FILLER_PRE - MIN_POST, s.gap
        assert s.gap == window, (s.gap, window)
        assert not (s.gap > window), "the clamp has to be visible: the rule the axis is stated on no longer holds"
        assert window < fact_distance(s.family, s.gap), (s.family, fact_distance(s.family, s.gap))
        assert len(s.tokens) == fit - 1, "a clamped gap cannot also change the sequence length"

    # At the shortest length that can hold the structure the clamped gap is only the top order, and here the
    # window really can reach the first occurrence: rule and physical reach fail together. This is the case the
    # axis cannot measure at all, and it is the case the harness's rule is meant to catch.
    tight = make_corpus(7, STRUCTURE + FILLER_PRE + MIN_POST + MAX_TABLE_ORDER, 4, 4,
                        families=("quote", "needle"), n_build=16, window=window)
    for s in tight["test"]:
        assert s.gap == MAX_TABLE_ORDER, s.gap
        assert not (s.gap > window), s.gap
        assert window >= fact_distance(s.family, s.gap), (s.family, fact_distance(s.family, s.gap))

    # Below the minimum the structure itself does not fit, which is an error rather than a smaller measurement.
    try:
        make_corpus(7, STRUCTURE + FILLER_PRE + MIN_POST + MAX_TABLE_ORDER - 1, 4, 4,
                    families=("quote",), n_build=8, window=window)
    except ValueError as e:
        assert "cannot hold a top-order gap" in str(e), e
    else:
        raise AssertionError("a length that cannot hold a top-order gap must raise, not shrink the gap")


def test_every_recall_sample_is_the_requested_length():
    """
    Fixed filler counts are what make a batch a tensor. Random filler counts would give sequences of different
    lengths within one family, and the two families could not be scored on the same axis.
    """
    length = 51
    corpus = make_corpus(13, length, 6, 6, families=("quote", "needle"), n_build=16, window=TINY().swa_window)
    for split in ("train", "test", "build"):
        for s in corpus[split]:
            assert len(s.tokens) == length, (s.family, len(s.tokens))
            assert len(s.targets) == len(s.tokens), s
            assert s.targets[-1] == -1, "the last position has no answer and must not be scored"
            for i, t in enumerate(s.targets[:-1]):
                assert t == s.tokens[i + 1], (s.family, i)

    # The trailing filler is not optional: without it the last span token has no target and the axis silently
    # scores one position fewer per sample.
    for s in corpus["test"]:
        if s.family == "quote":
            post = len(s.tokens) - FILLER_PRE - STRUCTURE - s.gap
            assert post >= MIN_POST, (s, post)
        else:
            post = len(s.tokens) - STRUCTURE - s.gap
            assert post >= MIN_POST, (s, post)
        assert s.recall_positions and s.targets[s.recall_positions[0]] != -1, s


def test_decode_path_can_only_agree_or_be_ignored():
    """
    The same sequences decoded twice, with and without the table, have to produce the same tokens. If they
    differ, the implementation blended, which is the thing the design forbids, and no loss number produced by
    that path could be attributed to the table.

    The prompt is cut at the last recall position rather than using the whole sequence, because that is the
    state the table is actually for: everything after that point is the answer, so the proposals are the answer
    tokens and acceptance measures retrieval rather than padding.
    """
    torch.manual_seed(17)
    cfg = TINY()
    length = length_for_window(cfg.swa_window)
    corpus = make_corpus(17, length, 8, 6, families=("quote", "needle"), n_build=MIN_CNT * 16, window=cfg.swa_window)
    model = HeronLM(cfg).eval()
    table = build_table(cfg, corpus["build"])

    prompts = []
    for s in corpus["test"]:
        tokens = tuple(s.tokens[: s.recall_positions[-1] + 1])
        prompts.append(replace(s, tokens=tokens, targets=targets(list(tokens))))
    dp = decode_path(model, table, prompts, 1, DEV, max_new_tokens=12)

    assert dp["decode_samples"] == len(prompts) == 6, dp
    assert dp["outputs_identical_without_table"] == dp["decode_samples"], dp
    assert dp["hits"] > 0, f"the table answered no probe on prompts built to be answerable ({dp})"
    assert dp["proposals"] > 0 and dp["forward_passes"] > 0, dp
    assert dp["lookups"] >= dp["hits"], dp

    # The consequence of the verify-only rule: the static table cannot move loss, so the delta ROADMAP asks
    # for can only come from the memory path, and the decode-path delta has to be reported as None.
    assert dp["decode_path_can_change_loss"] is False, dp
    assert dp["decode_path_delta_loss"] is None, dp

    # A stat added on the reference must show up in the harness rather than be silently dropped by a fixed key
    # list. This is the reporting half of the same invariant: decode_path sums what generate returned.
    assert "overshoot_tokens" in dp, dp


def test_lookups_count_attempts_and_not_only_hits():
    """
    A hit rate that only counted hits would always read 100%, and the miss rate - which is what a table's cost
    profile on a phone depends on - would be invisible. Two prompts, one the table can answer and one it cannot,
    are enough to show that the denominator is attempts.
    """
    torch.manual_seed(41)
    cfg = TINY()
    cfg.ngram_orders = (3, 2, 1)
    cfg.ngram_min_cnt = 1
    cfg.ngram_top_k = 2
    model = HeronLM(cfg).eval()

    table = NGramTable(cfg)
    table.build([10, 11, 12, 20])
    table.prune()

    hit_prompt = [50, 10, 11, 12]
    miss_prompt = [40, 41, 42, 43]
    _, hit_stats = model.generate(hit_prompt, 2, 1, table=table, speculate=True)
    _, miss_stats = model.generate(miss_prompt, 2, 1, table=table, speculate=True)

    assert hit_stats["hits"] == hit_stats["lookups"], hit_stats
    assert miss_stats["lookups"] > 0 and miss_stats["hits"] == 0, miss_stats
    lookups = hit_stats["lookups"] + miss_stats["lookups"]
    hits = hit_stats["hits"] + miss_stats["hits"]
    assert hits < lookups, "lookups were counted as hits, so the miss rate is invisible"

    # Speculation that cannot agree can only be ignored: a proposal the core would not make is dropped, and the
    # output is the model-only output.
    with_table, stats = model.generate(miss_prompt, 4, 1, table=table, speculate=True)
    without = model.generate(miss_prompt, 4, 1)[0]
    assert with_table[len(miss_prompt):] == without[len(miss_prompt):], (with_table, without)
    assert stats["accepted"] == 0, stats


def test_token_budget_is_a_promise_and_an_accepted_span_that_crosses_it_is_recorded():
    """
    The budget is checked once per loop step, not once per token inside a verified span, so a run that accepts
    a long span can end past the budget. The tokens past it are truncated rather than returned, because the
    alternative is that two runs of the same model on the same prompt return different lengths and a harness
    reads a reporting artifact as a blended path. The verification cost for those tokens was still paid, so the
    overshoot is its own number.

    An accepted span has to be longer than one token for the budget to be crossed, and top-k entries are
    distinct tokens, so this test needs a core whose successive continuations are distinct. At tiny scale an
    untrained core's greedy chain collapses onto a single repeated token - the harness checked that over 120
    seeds and none produced two distinct consecutive tokens - so the run samples at temperature 1.0 with a fixed
    seed. The plant is built from the core's own draws in the order it produced them, which is what lets
    verification confirm every proposal; for that to hold the measured run has to start from the same RNG state
    as the run the plant was built from, so the state is captured after the model is constructed and restored
    before each measured run.
    """
    cfg = TINY()
    cfg.ngram_orders = (4, 3, 2, 1)
    cfg.ngram_top_k = 4
    # min_cnt is set to 1 here because the entry is hand-made from the core's own continuations; the corpus
    # rule is tested separately.
    cfg.ngram_min_cnt = 1
    torch.manual_seed(23)
    model = HeronLM(cfg).eval()
    state = torch.get_rng_state()

    prompt = [10, 11, 12, 13, 14]
    plain, _ = model.generate(prompt, 8, 1, temperature=1.0)
    chain = plain[len(prompt):]
    assert len(set(chain[:4])) == 4, f"the plant needs 4 distinct continuations, got {chain}"

    table = NGramTable(cfg)
    key = tuple(prompt[-max(cfg.ngram_orders):])
    # Each continuation is recorded against the same key, which is what makes top-k return four proposals.
    # most_common keeps insertion order for equal counts, so the proposals come out in the order the core
    # produced them and every one is confirmed by verification.
    for tok in chain[:4]:
        table.build(list(key) + [tok])
    table.prune()
    hit = table.lookup(list(prompt))
    assert hit is not None, "the entry the run is measured on must be in the table"
    assert [tok for tok, _ in hit[1]] == list(chain[:4]), hit

    torch.set_rng_state(state)
    budget = 3
    out, stats = model.generate(prompt, budget, 1, table=table, speculate=True, temperature=1.0)
    # All four proposals were confirmed, so the accepted span crossed the budget. If verification had disagreed,
    # the cache-copy path would not match sequential generation, which is the parity invariant already tested.
    assert stats["proposed"] == 4 and stats["accepted"] == 4, (stats, chain)
    assert len(out) == len(prompt) + budget, (len(out), budget, stats)
    assert stats["overshoot"] == 1, stats
    assert out[len(prompt):] == chain[:budget], "the truncation changed which tokens the run returned"

    # A run inside the budget records no overshoot, so the number is a cost rather than a fixed field, and its
    # tokens are the tokens the model-only run produces from the same state: the truncation rule reports, it does
    # not change what was generated.
    torch.set_rng_state(state)
    out2, stats2 = model.generate(prompt, 10, 1, table=table, speculate=True, temperature=1.0)
    torch.set_rng_state(state)
    plain10, _ = model.generate(prompt, 10, 1, temperature=1.0)
    assert stats2["overshoot"] == 0 and len(out2) == len(prompt) + 10, (len(out2), stats2)
    assert out2[len(prompt):] == plain10[len(prompt):], (out2[len(prompt):], plain10[len(prompt):])

    # Without speculation the same prompt cannot overshoot at all, which is what makes the overshoot a cost of
    # verification rather than a cost of the model.
    torch.set_rng_state(state)
    _, stats3 = model.generate(prompt, budget, 1, temperature=1.0)
    assert stats3["lookups"] == 0 and stats3["overshoot"] == 0, stats3


def test_hit_rate_and_accuracy_and_coverage_are_three_different_denominators():
    """
    A hit is not the right answer, and a position whose fact never entered the table is not a retrieval
    failure. The hand-built corpus below makes all three numbers disagree on purpose, because a harness that
    printed one of them under another's name could not tell a weak table from a missing fact.
    """
    cfg = TINY()
    cfg.ngram_orders = (4, 3, 2, 1)
    cfg.ngram_min_cnt = 3
    cfg.ngram_top_k = 4

    def sample(seq, family):
        return Sample(tuple(seq), targets(list(seq)), family, (len(seq) - 2,), 0, tuple(seq[:MAX_TABLE_ORDER]))

    frequent = (10, 11, 12, 13)          # three build samples, so it enters the table
    rare = (20, 21, 22, 23)              # two build samples, so it is pruned
    # The trigram inside the rare fact is present three times through other facts, so a probe at the rare
    # fact's position hits - but through a lower order, which is not evidence that the fact itself entered.
    shared_trigram = [(7, 21, 22, 23, 97), (8, 21, 22, 23, 97), (9, 21, 22, 23, 97)]

    build = [sample(list(frequent) + [99], "quote")] * 3 + [sample(list(rare) + [98], "quote")] * 2
    build = [sample(list(seq), "quote") for seq in shared_trigram] + build

    eval_samples = [
        sample(list(frequent) + [99], "quote"),   # answerable, hit, correct
        sample(list(rare) + [98], "quote"),        # not answerable: coverage, not a failure
        sample(list(frequent) + [71], "quote"),   # answerable, hit, wrong top-1: a real retrieval failure
    ]

    table = build_table(cfg, build)
    cov = fact_coverage(table, build, eval_samples)
    axis = recall_axis(table, eval_samples, cov["counts"])

    assert cov["answerable_eval_samples"] == 2, cov
    assert cov["eval_samples_on_facts_below_min_cnt"] == 1, cov
    assert axis["quote"]["samples_below_min_cnt"] == 1, axis

    # The three denominators.
    assert axis["quote"]["positions"] == 3, axis
    assert axis["quote"]["hits"] == 3, axis
    assert axis["quote"]["answerable_positions"] == 2, axis
    assert axis["quote"]["top1_correct"] == 1, axis
    assert axis["quote"]["hit_rate_at_recall_positions"] == 1.0, axis
    assert axis["quote"]["accuracy_over_answerable_positions"] == 0.5, axis
    assert axis["quote"]["hit_orders"] == {4: 2, 3: 1}, axis
    assert axis["quote"]["hit_rate_at_recall_positions"] != axis["quote"]["accuracy_over_answerable_positions"], (
        "the two denominators came out identical, so the axis is not a separate axis"
    )

    # A fact below min_cnt has no entry at all, so the position that asks for it cannot be scored as wrong.
    # table_support probes the key at the top order directly, which is why it reports the rare fact as absent
    # while the recall axis counts a hit for it through the order-3 fallback.
    sup = table_support(table, eval_samples)
    assert sup["facts_resolving_at_top_order"] == 2, sup
    assert sup["facts_resolving_only_through_a_lower_order"] == 0, sup
    assert sup["facts_with_no_entry"] == 1, sup

    # The lengths are reported as the shape of the table, not as a gate number: a table that is mostly order 1
    # is a bag of single-token continuations, and the same threshold measured over it is a different mechanism.
    dist = ngram_length_distribution(table)
    assert tuple(frequent) in table.entries and tuple(rare) not in table.entries, dist
    assert tuple(rare[1:]) in table.entries, "the lower-order entry is what produced the fallback hit"
    assert dist["by_order"][4]["entries"] == 1, dist
    assert dist["by_order"][3]["entries"] >= 3, dist
    assert dist["entries_truncated_by_top_k"] == 0, dist


def test_min_cnt_at_the_corpus_level_and_per_conversation_are_different_tables():
    """
    The difference the harness depends on: min_cnt is a corpus condition, so support spread across
    conversations counts. Read per conversation, a repeated span contributes two counts inside one conversation
    and is pruned before the next conversation can add to it. The two readings have to differ, otherwise the
    rule is not observable - and this is the bug that made the tiny run measure a nine-entry table that could
    not answer a recall question.
    """
    cfg = TINY()
    length = length_for_window(cfg.swa_window)
    corpus = make_corpus(29, length, 8, 8, families=("quote", "needle"), n_build=48, window=cfg.swa_window)

    correct = build_table(cfg, corpus["build"])
    buggy = NGramTable(cfg)
    for s in corpus["build"]:
        buggy.build(list(s.tokens))
        buggy.prune()

    cov = fact_coverage(correct, corpus["build"], corpus["test"])
    sup_correct = table_support(correct, corpus["test"])
    sup_buggy = table_support(buggy, corpus["build"])

    # The driver prints these two against each other; for the correct build they must agree, and a gap between
    # them is a bug rather than a result.
    assert cov["answerable_eval_samples"] == sup_correct["facts_resolving_at_top_order"], (cov, sup_correct)

    # Per-conversation pruning can never keep a top-order entry: one conversation contributes at most two
    # counts for a fact repeated inside itself, which is below min_cnt.
    assert sup_buggy["facts_resolving_at_top_order"] == 0, sup_buggy
    assert all(e.order < MAX_TABLE_ORDER for e in buggy.entries.values()), (
        "per-conversation pruning kept a top-order entry, so the rule being tested is not observable"
    )
    assert len(buggy.entries) < len(correct.entries), (len(buggy.entries), len(correct.entries))

    # The same disagreement visible as the shape of the table, which is what the driver warns about.
    dist_buggy = ngram_length_distribution(buggy)
    dist_correct = ngram_length_distribution(correct)
    assert MAX_TABLE_ORDER in dist_correct["orders"], dist_correct
    assert MAX_TABLE_ORDER not in dist_buggy["orders"], dist_buggy
    assert all(o < MAX_TABLE_ORDER for o in dist_buggy["orders"]), dist_buggy

    # The counts themselves: every fact is covered at least min_cnt times by cycling, so coverage is a property
    # of the corpus rather than of the draw.
    counts = fact_counts(corpus["build"])
    assert all(c >= cfg.ngram_min_cnt for row in counts.values() for c in row.values()), counts
    assert len(counts["quote"]) == 8 and len(counts["needle"]) == 8, counts


def test_delta_loss_sign_and_the_decode_path_cannot_produce_it():
    """
    ROADMAP states delta loss >= 0.02 nats but not its sign. The convention this harness uses is positive = the
    path improves loss, which is what "survives" means, and it has to be checked rather than implied. The delta
    is only produced at the dim ROADMAP states, so a tiny run prints the two losses as context and the gate
    value stays None.
    """
    t = threshold("P1.1", "delta_loss")
    assert t.proposal, "the sign is a convention this harness states, not a ROADMAP number"
    assert gates.check(t, 0.03)["status"] == "pass" and gates.check(t, -0.03)["status"] == "fail", (
        "positive must mean the path improves loss"
    )
    # A number produced at a config that cannot satisfy the stated condition cannot be reported at all.
    assert gates.check(t, None)["status"] == "not measured", gates.check(t, None)

    torch.manual_seed(31)
    tiny = TINY()
    length = length_for_window(tiny.swa_window)
    corpus = make_corpus(31, length, 4, 4, families=("quote", "needle"), n_build=16, window=tiny.swa_window)
    model = HeronLM(tiny).eval()

    # An untrained table gives exactly the same loss with and without the path, so a delta of zero there is a
    # property of the initialization rather than evidence that the path does not work.
    on = path_losses(model, corpus["test"], 1, DEV, True)
    off = path_losses(model, corpus["test"], 1, DEV, False)
    assert on["loss"] == off["loss"], (on["loss"], off["loss"])
    mp = memory_path(model, corpus["test"], 1, DEV)
    assert mp["table_untrained"] and mp["bf16"]["delta_nats"] is None, mp
    assert mp["measured_at_stated_dim"] is False, mp

    # At the stated dim the delta is produced, and it is the sign convention applied to the two losses the
    # harness already reports. A table that is not zeros is what makes the sign observable.
    stated = TINY()
    stated.ngram_memory_dim = STATED_MEMORY_DIM
    m2 = HeronLM(stated).eval()
    m2.memory.table.weight.data = F.normalize(torch.randn(stated.ngram_memory_entries, STATED_MEMORY_DIM), dim=1)
    mp2 = memory_path(m2, corpus["test"], 1, DEV)
    assert mp2["measured_at_stated_dim"] and not mp2["table_untrained"], mp2
    for half in ("bf16", "after_int8"):
        r = mp2[half]
        on_l, off_l, d = r["with_path"]["loss"], r["without_path"]["loss"], r["delta_nats"]
        assert d is not None, r
        assert abs(d - (off_l - on_l)) < 1e-9, (half, d, off_l - on_l)
        assert (d > 0) == (on_l < off_l), (half, d, on_l, off_l)
        # The per-position denominators are kept, which is the other half of the axis: the whole-sequence loss
        # is the family's difficulty, not the retrieval.
        assert r["with_path"]["loss"] != r["with_path"]["loss_at_recall_positions"], r

    # The gate cannot be decided by a decode-path measurement the design makes impossible.
    v = gates.verdict("P1.1", {"delta_loss": mp["bf16"]["delta_nats"]}, "tiny", False)
    c = {x["metric"]: x for x in v["checks"]}
    assert c["delta_loss"]["status"] == "not measured", c["delta_loss"]
    assert v["status"] == "pending", v["status"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if failures else 0)

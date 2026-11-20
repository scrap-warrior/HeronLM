"""
Generated corpora for the Phase 0 arms and the Phase 1 table gates.

Phase 0 measures mechanisms, not quality, so the data is generated rather than collected. What the data
has to do is make a mechanism observable in loss. A gate cannot be decided on a task where the mechanism
changes nothing, so the corpus is built from tasks that need the mechanism:

  lookup     - one stage. The next token is a fixed permutation of the previous token. A single pass
               solves it, so extra depth should NOT help here. This is the control: if the harness shows
               a depth gain on this family as well, the gain is an artifact of the harness rather than of
               depth, and the gate measurement would be worthless at real scale.
  recurrence - two stages. The next token is the sum modulo the sub-vocab of the previous two tokens.
               Retrieval and the modular wrap are separate computations, so this is the family where the
               extra passes have something to do. It is also the family that rewards a widened residual
               stream: the retrieved value has to stay readable while the second stage runs.
  quote      - verbatim reproduction. A span appears once, then again later in the same conversation. The
               answer at the second occurrence is decided by the first one, which is the table's actual job
               rather than a depth task. ROADMAP requires verbatim recall as its own axis, so the positions
               that carry the answer are recorded on the sample instead of being re-derived from the family
               name.
  needle     - needle-in-haystack. A key, a payload, a gap, the same key again, then the same payload. The
               answer at the last key token is the first payload token, decided entirely by the first
               occurrence. The gap is what makes this axis mean something: when it is longer than the sliding
               window, attention cannot reach the payload, so anything the model gets right there comes from
               a path that is not the window. The table is exactly that path, and this is the only family
               where the two can be told apart.

At the real gate scale the same interface is fed by a real tokenized corpus. What does not change between
the two is the split rule: train and test are disjoint sequences, and the eval set is disjoint from
anything used to build the n-gram table. Measuring a table gain on data the table was built from is the
self-deception the Phase 1 guard exists to prevent.

The Phase 0 arms do not train on the fixed split: `SampleStream` generates sample index i on demand, so a
gate-scale run is not hundreds of passes over the same 64 sequences. `make_corpus` stays as the fixed
split the tests and the Phase 1 table builders need.

The sub-vocab is deliberately small. Modular addition over 151k classes is not learnable from a few
hundred steps, and a task that cannot be learned cannot show a depth gain.

Two things about the recall families that have to be stated rather than assumed:

  - The span pool and the key pool are shared between the build corpus and the eval set. That is not the leak
    the guard is about - it is what a lookup table is for, since the fact has to exist somewhere in the build
    corpus for a probe to return it. What would be a leak is an eval sequence appearing in the build corpus,
    and the harness checks exactly that.
  - A fact seen fewer than min_cnt times in the build corpus is pruned, so an eval sample asking for it has no
    answer available. That is a real state and the harness reports it as one, which is why the fact used by
    each sample is recorded on the sample.

Recall positions are recorded with the table's top order in mind. A key can only be probed when the window
ending at that position is at least one order long and still inside the first occurrence, so the earliest
positions of a repeated span cannot be recalled at the top order even though a lower order can hit them.

For the recall families the sequence length is set by the structure, not by `length`: pre-filler + span + gap +
span + post-filler. `length` is what the gap is chosen to make the total equal, so the gap is only honoured when
the requested length leaves room for a top-order gap. `length` applies as written to lookup and recurrence.

Run through scripts/phase1.py. The Phase 0 driver uses only lookup and recurrence.
"""

import hashlib
import random
from dataclasses import dataclass

SUB_VOCAB = 48

# The pools are small so the same fact appears in more than one conversation. A pool of one would make every
# eval sample a memorized copy of a build sample, which is the leak the guard is meant to catch.
SPAN_POOL = 8
SPAN_LEN = 8
KEY_POOL = 8
KEY_LEN = 4
PAYLOAD_LEN = 4

# The order the recorded positions assume. If a config changes ngram_orders, this has to change with it or the
# positions would be computed for a different key length; the harness asserts the two agree rather than
# quietly measuring a shorter key.
MAX_TABLE_ORDER = 4

# The count ROADMAP names for the table. A fact has to appear in at least this many build samples for its entry
# to survive min_cnt, so a build corpus that repeats each fact fewer than this many times leaves the recall
# axis measuring an empty table.
MIN_CNT = 3

# The bounded retry in SampleStream, named so the refusal can state the number it actually tried. A caller that
# excludes every sequence the task can produce should get an error, not hang; 4096 attempts is far beyond the
# exclusion density of any corpus this harness builds.
SAMPLE_STREAM_ATTEMPTS = 4096

# The size the synthetic eval-supply probe is willing to draw. This is a limit of the harness, not a property of
# any task: the probe reproduces the eval set run_gate would build, which means building it, so a run that asks
# for millions of synthetic windows is refused with the limit named rather than checked by materializing them.
PROBE_MAX_WINDOWS = 200_000

# Fixed filler counts. Both recall families have 16 tokens of structure that is not filler (quote: two spans
# of 8; needle: key+payload twice = 4+4+4+4), one leading filler for the quote, and at least one trailing
# filler so the last repeated token has a target. Every sample is exactly `length` tokens, which is what makes
# a batch a tensor and the two families comparable on the same axis.
FILLER_PRE = 1
MIN_POST = 1
STRUCTURE = 16

# The shortest sequence that can hold both a top-order gap and the minimum fillers.
MIN_LENGTH = STRUCTURE + FILLER_PRE + MIN_POST + MAX_TABLE_ORDER


def length_for_window(window: int) -> int:
    """
    The length that makes the gap longer than the sliding window.

    This is not a nicety. The needle axis only means something when the payload is out of the window's reach,
    because then a correct answer at the second key can only come from a path that is not the window. Below
    this length the generator has to shrink the gap, the window can still reach the first occurrence, and the
    two paths cannot be separated - so a number produced at a shorter length is reported as not decidable
    rather than as a recall result.
    """
    return STRUCTURE + FILLER_PRE + MIN_POST + max(window + 1, MAX_TABLE_ORDER)


def fact_distance(family: str, gap: int) -> int:
    """
    How far back a window ending at a recall position has to reach to see the first occurrence.

    Two conditions that are easy to collapse into one, and the harness has to keep them apart:

      - the rule it states, which is what `length_for_window` derives the length from: the gap is longer than
        the window. That is conservative - if the gap alone exceeds the window, then no window ending inside the
        second occurrence can reach the first occurrence at all.
      - what is physically true at a recall position: the window ends at that position and reaches back `window`
        tokens, so it sees the first occurrence only if it can reach back past the structure between the recall
        position and the first occurrence. For the quote family that is the top-order piece of the second span
        ending at the position; for the needle it is the whole second key plus the payload, because the window
        at the recall position contains the key that ends there.

    The two can disagree, and the test shows it: one token below `length_for_window` the rule stops holding
    while the fact is still physically out of reach. The harness reports the rule, so a measurement at that
    length is reported as not decidable rather than as a recall result - but the physical number is printed as
    well, because from the rule alone a reader cannot tell whether the axis was unmeasurable or merely
    uncertified.
    """
    if family == "quote":
        return gap + MAX_TABLE_ORDER
    if family == "needle":
        return gap + KEY_LEN + PAYLOAD_LEN
    return 0


@dataclass(frozen=True)
class Sample:
    tokens: tuple[int, ...]
    # Target at position i is the token at position i + 1. The last position has no target, and -1 is
    # cross_entropy's ignore_index, so the sequence is scored at every position that has an answer.
    targets: tuple[int, ...]
    family: str
    # Positions where the answer is decided by something earlier in the conversation. Recorded on the sample
    # rather than recomputed from the family name, because the axis ROADMAP asks for is the recall itself: an
    # average over the whole sequence would report the family's difficulty rather than the retrieval.
    recall_positions: tuple[int, ...] = ()
    # Distance between the first occurrence and the second, so the harness can check whether the gap exceeds
    # the sliding window. That is the condition that makes the needle axis mean something at all.
    gap: int = 0
    # The fact the sample asks for. Recorded so the harness can tell a miss from an unavailable answer: a fact
    # that never entered the build corpus cannot be recalled, and counting those positions as failures would
    # make the table look worse than it is.
    fact: tuple[int, ...] = ()


def _lookup_sequence(rng: random.Random, length: int, perm: list[int]) -> list[int]:
    tok = rng.randrange(SUB_VOCAB)
    seq = [tok]
    for _ in range(length - 1):
        tok = perm[tok]
        seq.append(tok)
    return seq


def _recurrence_sequence(rng: random.Random, length: int) -> list[int]:
    seq = [rng.randrange(SUB_VOCAB), rng.randrange(SUB_VOCAB)]
    for _ in range(length - 2):
        seq.append((seq[-1] + seq[-2]) % SUB_VOCAB)
    return seq


def _quote_sequence(rng: random.Random, span: tuple[int, ...], gap: int, length: int) -> tuple[list[int], tuple[int, ...]]:
    """
    filler, span, gap, span, filler.

    The trailing filler gives the last span token a target. Without it the final position has no answer and the
    recall axis would silently score one position fewer per sample.

    The filler counts are fixed rather than random because every sample has to be exactly `length` tokens: a
    batch is a tensor, and a family whose sequences vary in length cannot be trained or scored alongside one
    whose lengths differ. The filler content is still random, so the table does not see the same padding
    n-grams in every sample.
    """
    post = length - FILLER_PRE - SPAN_LEN * 2 - gap
    seq = [rng.randrange(SUB_VOCAB) for _ in range(FILLER_PRE)]
    seq += list(span)
    seq += [rng.randrange(SUB_VOCAB) for _ in range(gap)]
    second_start = len(seq)
    seq += list(span)
    seq += [rng.randrange(SUB_VOCAB) for _ in range(post)]

    # A position counts only when the window ending at it is top-order long and still inside the repeated span.
    # Earlier positions have a key straddling the gap, which the table has never seen.
    positions = tuple(range(second_start + MAX_TABLE_ORDER - 1, second_start + SPAN_LEN - 1))
    return seq, positions


def _needle_sequence(rng: random.Random, key: tuple[int, ...], payload: tuple[int, ...],
                     gap: int, length: int) -> tuple[list[int], tuple[int, ...]]:
    """
    key, payload, gap, key, payload, filler.

    The answer at the last key token is the first payload token. The key window ending there is exactly the
    key, so a probe at the top order lands on the entry built from the first occurrence. With a key shorter
    than the top order, the same answer would only be reachable through a lower-order fallback, which is the
    weakest evidence the table has and not what this axis is testing.

    The sequence starts with the key rather than with a filler, so the probe at the recall position is the key
    itself and not the key plus padding. Same fixed-length rule as the quote family.
    """
    post = length - KEY_LEN * 2 - PAYLOAD_LEN * 2 - gap
    seq = list(key) + list(payload)
    seq += [rng.randrange(SUB_VOCAB) for _ in range(gap)]
    second_key_start = len(seq)
    seq += list(key)
    seq += list(payload)
    seq += [rng.randrange(SUB_VOCAB) for _ in range(post)]
    return seq, (second_key_start + KEY_LEN - 1,)


def _targets(seq: list[int]) -> tuple[int, ...]:
    return tuple(seq[1:]) + (-1,)


def _min_length(family: str) -> int:
    """The shortest sequence that can hold the structure, including a top-order gap and the minimum fillers."""
    return STRUCTURE + FILLER_PRE + MIN_POST + MAX_TABLE_ORDER


def _gap_for(family: str, length: int, window: int | None) -> int:
    """
    The gap, chosen so the second occurrence is out of the window's reach.

    The clamp is the part that matters. `room` is what is left after the fixed blocks and the minimum fillers,
    and the gap cannot be larger than that without making a sample shorter or longer than `length` - which is
    what makes a batch a tensor and the two families comparable on the same axis. So when the window would
    require more than `room`, the gap is clamped to `room` and the axis reports as not decidable, because the
    window can still reach the first occurrence and the two paths cannot be told apart. That is exactly what
    the needle axis is supposed to separate, so a clamped gap has to be visible rather than quietly smaller.
    """
    if family not in ("quote", "needle"):
        return 0
    room = length - STRUCTURE - FILLER_PRE - MIN_POST
    if room < MAX_TABLE_ORDER:
        raise ValueError(
            f"length {length} cannot hold a top-order gap for the {family} family: "
            f"{STRUCTURE} tokens of structure plus {FILLER_PRE} leading and {MIN_POST} trailing filler "
            f"leaves {room} for a gap that has to be at least order {MAX_TABLE_ORDER}. "
            f"Minimum length is {_min_length(family)}, or {length_for_window(window)} for window {window}."
        )
    wanted = max(window + 1, MAX_TABLE_ORDER) if window is not None else room
    return min(room, wanted)


def _pools(seed: int):
    """
    The shared substrate: the lookup permutation and the fact pools for the recall families.

    Kept separate from sample generation because a stream needs the same pools for every sample without
    replaying the sequential draws `make_corpus` uses. Both paths call this with the same seed, so a fixed
    corpus and a stream describe the same task.
    """
    rng = random.Random(seed)
    perm = rng.sample(range(SUB_VOCAB), SUB_VOCAB)
    spans = [tuple(rng.randrange(SUB_VOCAB) for _ in range(SPAN_LEN)) for _ in range(SPAN_POOL)]
    keys = [tuple(rng.randrange(SUB_VOCAB) for _ in range(KEY_LEN)) for _ in range(KEY_POOL)]
    payloads = [tuple(rng.randrange(SUB_VOCAB) for _ in range(PAYLOAD_LEN)) for _ in range(KEY_POOL)]
    return rng, perm, spans, keys, payloads


def make_corpus(seed: int, length: int = 48, n_train: int = 64, n_test: int = 16,
                families: tuple[str, ...] = ("lookup", "recurrence"), n_build: int = 0,
                window: int | None = None, cycle_build: bool = True) -> dict[str, list[Sample]]:
    """
    One fixed split, generated from one seed.

    The seed is fixed per gate rather than per arm, because the arms of a gate have to see the same data in the
    same order. Otherwise a loss difference between two arms could be a data difference, and the single-variable
    comparison the gate is built on would not mean anything.

    `build` is the only split the static table is allowed to see, and it is disjoint from `test` by
    construction: samples are drawn from the same fact pools but never the same sequences. `window` is used
    only to choose the gap; the harness checks separately whether the gap actually exceeds it.

    `cycle_build` covers the fact pools round-robin instead of drawing them at random. That is not a cosmetic
    choice: with a random draw over an 8-fact pool and a build corpus of a few dozen samples, some facts appear
    fewer than min_cnt times and are pruned, so part of the recall axis would measure sampling luck rather than
    retrieval. Cycling guarantees every fact is present at least MIN_CNT times, and coverage then reports as
    what it is - a property of the corpus, not of the draw.
    """
    rng, perm, spans, keys, payloads = _pools(seed)

    def one(family: str, pool_idx: int | None = None) -> Sample:
        if family == "lookup":
            seq, positions, fact, gap = _lookup_sequence(rng, length, perm), (), (), 0
        elif family == "recurrence":
            seq, positions, fact, gap = _recurrence_sequence(rng, length), (), (), 0
        elif family == "quote":
            if pool_idx is None:
                pool_idx = rng.randrange(SPAN_POOL)
            fact = spans[pool_idx]
            gap = _gap_for(family, length, window)
            seq, positions = _quote_sequence(rng, fact, gap, length)
        elif family == "needle":
            if pool_idx is None:
                pool_idx = rng.randrange(KEY_POOL)
            fact, gap = keys[pool_idx], _gap_for(family, length, window)
            seq, positions = _needle_sequence(rng, fact, payloads[pool_idx], gap, length)
        else:
            raise ValueError(f"unknown family: {family}")
        return Sample(tuple(seq), _targets(seq), family, positions, gap, fact)

    # Every family contributes to every split, so a per-family number is a per-family measurement rather than
    # an artifact of which family happened to land in which split.
    def split(count: int, cycle: bool = False) -> list[Sample]:
        out = []
        for f in families:
            pool = SPAN_POOL if f == "quote" else (KEY_POOL if f == "needle" else None)
            for j in range(max(count // len(families), 0)):
                out.append(one(f, (j % pool) if (cycle and pool) else None))
        return out

    return {"train": split(n_train), "test": split(n_test),
            "build": split(n_build, cycle_build and n_build > 0)}


def _sample_seed(seed: int, split: str, index: int, attempt: int) -> int:
    """
    A stable per-sample seed, from the run seed, the split, the sample index and the exclusion attempt.

    blake2b rather than hash() because hash() of a string is randomized per process, and a stream that
    produced different samples in two processes would make a checkpoint resume inexact. The attempt counter
    is part of the seed so that skipping an excluded sample does not shift the meaning of later indices.
    """
    raw = f"{seed}:{split}:{index}:{attempt}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


class SampleStream:
    """
    An endless, deterministic stream of fresh samples for one split.

    `make_corpus` is a fixed pool. At the default 64 train samples, a long run sees the same sequences
    hundreds of times, and "loss still improves with depth" becomes a question about memorizing 64
    sequences. This class generates sample index i on demand from a seed derived from (seed, split, i), so a
    run can draw millions of samples without materializing them and without a fixed pool cap. The same index
    always yields the same sample, which is what keeps the data order comparable between arms and makes a
    checkpoint resume exact.

    The task's input space is still finite - lookup admits 48 sequences and recurrence 48 squared - so a
    stream cannot manufacture new task structure. What it removes is the artificial cap: the arms are no
    longer trained on a 64-sequence fixture. The driver prints the token budget so a short run is not
    mistaken for the real 1-2B-token gate.

    `exclude` holds the sequences reserved for evaluation. A long stream will otherwise draw every sequence
    the task admits, including the held-out ones, and a held-out loss measured on sequences the model trained
    on is not held out. Excluding them is a property of the finite synthetic task, not of a real corpus.
    """

    def __init__(self, seed: int, length: int = 48,
                 families: tuple[str, ...] = ("lookup", "recurrence"),
                 split: str = "train", window: int | None = None,
                 exclude: list[tuple[int, ...]] | None = None):
        if not families:
            raise ValueError("a sample stream needs at least one family")
        if split not in ("train", "test", "build"):
            raise ValueError(f"unknown split: {split}")
        # The pools are shared with make_corpus on purpose: a stream and a fixed split must describe the same
        # task, or a number measured on one would not transfer to the other.
        _, self.perm, self.spans, self.keys, self.payloads = _pools(seed)
        self.seed = seed
        self.length = length
        self.families = tuple(families)
        self.split = split
        self.window = window
        self._exclude = {tuple(s) for s in (exclude or ())}
        # Recorded in a checkpoint so a resume cannot silently switch data sources. The exclusions are not
        # part of the id because they are derived from the same seed, seq_len and n_test that the arm's run
        # key already stores; if any of those changes, the run key changes with them.
        self.corpus_id = f"synthetic:{seed}:{length}:{self.split}:{','.join(self.families)}"

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            raise IndexError("sample indices are non-negative")
        family = self.families[index % len(self.families)]
        round_index = index // len(self.families)
        # A bounded retry rather than an unbounded loop: a caller that excludes every sequence the task can
        # produce should get an error, not hang. See SAMPLE_STREAM_ATTEMPTS.
        for attempt in range(SAMPLE_STREAM_ATTEMPTS):
            rng = random.Random(_sample_seed(self.seed, self.split, index, attempt))
            sample = self._one(rng, family, round_index)
            if sample.tokens not in self._exclude:
                return sample
        raise ValueError(
            f"could not draw a {family} sample outside the excluded set in {SAMPLE_STREAM_ATTEMPTS} attempts; "
            f"the held-out set covers the sequences this task produces, so training has nothing left to draw"
        )

    def _one(self, rng: random.Random, family: str, round_index: int) -> Sample:
        """One sample, with the same family rules make_corpus uses."""
        if family == "lookup":
            seq, positions, fact, gap = _lookup_sequence(rng, self.length, self.perm), (), (), 0
        elif family == "recurrence":
            seq, positions, fact, gap = _recurrence_sequence(rng, self.length), (), (), 0
        elif family == "quote":
            fact = self.spans[round_index % SPAN_POOL]
            gap = _gap_for(family, self.length, self.window)
            seq, positions = _quote_sequence(rng, fact, gap, self.length)
        elif family == "needle":
            fact = self.keys[round_index % KEY_POOL]
            gap = _gap_for(family, self.length, self.window)
            seq, positions = _needle_sequence(rng, fact, self.payloads[round_index % KEY_POOL], gap, self.length)
        else:
            raise ValueError(f"unknown family: {family}")
        return Sample(tuple(seq), _targets(seq), family, positions, gap, fact)

    def sample_batch(self, start: int, count: int) -> list[Sample]:
        """The batch a training step consumes, in the order the step consumes it."""
        return [self[start + j] for j in range(count)]


def sequence_space(family: str) -> int | None:
    """
    The number of distinct sequences a family can produce, where that is a fixed small number.

    lookup is decided by its first token, so it admits SUB_VOCAB sequences; recurrence by its first two, so it
    admits SUB_VOCAB squared. Those are exact counts rather than estimates because a sequence is fully
    determined by its opening tokens: two different openings cannot produce the same sequence at a fixed length.

    The recall families are reported as None. Their sequences depend on random filler as well as on the fact
    pool, so their space is large but not a countable pool, and a number written for it would be invented.
    """
    if family == "lookup":
        return SUB_VOCAB
    if family == "recurrence":
        return SUB_VOCAB * SUB_VOCAB
    return None


def eval_supply(seed: int, length: int, n_test: int, families: tuple[str, ...]) -> dict:
    """
    What the synthetic path can actually supply for an eval set, checked before a run commits to training.

    A gate run asks for N held-out documents. On the synthetic path a document is one draw from a task, and each
    task admits a fixed number of distinct sequences. So an eval set larger than the space it is drawn from is
    not N documents; it is the same documents counted more than once, and the error bar - which is computed over
    documents - is being computed over a denominator the run does not have.

    The second consequence is the one that stops a run. Training draws are made outside the eval set on purpose,
    so if the eval set covers a family's whole space there is no sequence left for the training stream to draw
    from that family, and the run fails at its first step with a message about an exhausted space. Checking this
    before training gives an explanation rather than a traceback, and gives it before the compute is spent.

    `max_supply` is derived from the spaces rather than measured: the largest eval set that is guaranteed to
    leave at least one sequence per family for training. Above the bound a run can still work, but only because
    its draw happened to miss part of the space - so the bound and what this draw covered are both reported, and
    a run above the bound is a run whose ability to run depends on luck rather than on the task. It is only
    stated when every family in the run has a countable space.

    The check draws, so it costs memory before it costs anything else: the eval set is materialized the way
    run_gate would materialize it. Above PROBE_MAX_WINDOWS the draw is not run and the answer is reported as
    not probed. What arithmetic can still say is that a countable family's eval set cannot be N documents - N
    draws from an S-sequence space repeat at least N - S times - and what arithmetic cannot say is whether the
    draw covered the whole space, because N draws from S sequences need not cover all S. Whether training can
    draw is a fact about this draw, so an unprobed draw is reported as unprobed rather than inferred.
    """
    families = tuple(families)
    if not families:
        raise ValueError("a sample stream needs at least one family")
    spaces = [sequence_space(f) for f in families]

    if n_test > PROBE_MAX_WINDOWS:
        per_family = {}
        for i, fam in enumerate(families):
            windows = n_test // len(families)
            per_family[fam] = {
                "windows": windows,
                "space": spaces[i],
                "distinct_sequences": None,
                "repeats": None,
                # The most a family's draws can be, and the fewest documents the run does not have. These are
                # bounds rather than measurements, and they are named as bounds.
                "max_distinct": spaces[i],
                "minimum_repeats": (windows - spaces[i] if spaces[i] is not None and windows > spaces[i] else None),
                "training_sequences_left": None,
                "can_draw": None,
            }
        counted = [f for f in families if sequence_space(f) is not None]
        parts = [f"{f}: {per_family[f]['windows']} draws against a {per_family[f]['space']}-sequence space, "
                 f"so at least {per_family[f]['minimum_repeats']} of them are repeats"
                 for f in counted if per_family[f]["minimum_repeats"] is not None]
        say = (("for " + ", ".join(counted) + ": " + "; ".join(parts)) if parts
               else "no family here has a countable space that this eval set exceeds, so no bound "
                    "can be stated at all")
        refusal = (f"the eval set of {n_test} windows is above the probe's stated limit of "
                   f"{PROBE_MAX_WINDOWS}, so the harness cannot say whether training would be able to draw "
                   f"from it; the probe draws, and drawing an eval set this size means materializing it. "
                   f"What arithmetic does say is {say}. Lower --eval-windows below the limit, or pass "
                   f"--corpus-dir for real text, where the eval windows are documents and the space is the "
                   "val split.")
        return {
            "seed": seed,
            "length": length,
            "windows": n_test,
            "families": list(families),
            "per_family": per_family,
            "distinct_sequences": None,
            "repeated_windows": None,
            "probed": False,
            "probe_limit": PROBE_MAX_WINDOWS,
            "can_draw": None,
            "blocked_families": [],
            "max_supply": (min(spaces) * len(families) - len(families)
                            if all(s is not None for s in spaces) else None),
            "status": "not probed",
            "refusal": refusal,
            "error_bar_denominator": "distinct sequences",
        }

    # The same construction run_gate uses, from the same seed and length, so the numbers here describe the same
    # eval set the run would be measured on.
    corpus = make_corpus(seed, length=length, n_train=0, n_test=n_test, families=families)
    distinct: dict[str, set[tuple[int, ...]]] = {}
    for s in corpus["test"]:
        distinct.setdefault(s.family, set()).add(s.tokens)

    # Attempt the draw training would actually make first: one sample per family, in the order the first batch
    # consumes it. The attempt is the honest check for the recall families, whose space is not countable.
    stream = SampleStream(seed, length=length, families=families, split="train",
                          exclude=[s.tokens for s in corpus["test"]])
    blocked = []
    for i, fam in enumerate(families):
        try:
            stream[i]
        except ValueError:
            blocked.append(fam)

    spaces = [sequence_space(f) for f in families]
    max_supply = min(spaces) * len(families) - len(families) if all(s is not None for s in spaces) else None

    per_family: dict[str, dict] = {}
    for i, fam in enumerate(families):
        windows = max(n_test // len(families), 0)
        seen = len(distinct.get(fam, set()))
        per_family[fam] = {
            "windows": windows,
            "space": spaces[i],
            "distinct_sequences": seen,
            "repeats": windows - seen,
            # A family whose whole space is in the exclusion has nothing left to draw. That is the state the
            # refusal is about, so it is written as a count rather than only as a boolean.
            "training_sequences_left": (None if spaces[i] is None else spaces[i] - seen),
            "can_draw": fam not in blocked,
        }

    refusal = None
    if blocked:
        parts = []
        for f in blocked:
            if per_family[f]["space"] is None:
                # A family with no countable space can only be blocked by the retry limit. Saying "the whole
                # space" would be a stronger claim than what happened: the retry found nothing outside the eval
                # set in SAMPLE_STREAM_ATTEMPTS attempts. That is stated as the weaker claim it is.
                parts.append(f"{f}: its {per_family[f]['windows']} eval draws left nothing the retry could find "
                             f"in {SAMPLE_STREAM_ATTEMPTS} attempts, so training cannot draw from it")
            else:
                parts.append(f"{f}: its {per_family[f]['windows']} eval draws are the whole "
                             f"{per_family[f]['space']}-sequence space, so training has nothing left to draw")
        if max_supply is None:
            bound = ("No bound can be stated for this run: the bound comes from the task spaces, and at least "
                     "one family here has no countable space, so this draw is what the run has. Lower "
                     "--eval-windows.")
        else:
            bound = (f"The largest eval set guaranteed to leave a sequence per family is {max_supply} windows "
                     f"at {length}-token length - a bound from the spaces, not from this draw, which happened "
                     f"to cover the whole space. Lower --eval-windows to {max_supply}.")
        refusal = ("the synthetic path cannot supply this eval set; " + "; ".join(parts) + ". " + bound +
                   " Alternatively pass --corpus-dir for real text, where the eval windows are documents and "
                   "the space is the val split.")

    repeats = sum(v["repeats"] for v in per_family.values())
    status = ("cannot supply an eval set this large" if blocked
              else "supplies repeats" if repeats else "can supply")

    return {
        "seed": seed,
        "length": length,
        "windows": n_test,
        "families": list(families),
        "per_family": per_family,
        "distinct_sequences": sum(len(v) for v in distinct.values()),
        "repeated_windows": repeats,
        "can_draw": not blocked,
        "blocked_families": blocked,
        "max_supply": max_supply,
        "status": status,
        "refusal": refusal,
        "probed": True,
        "probe_limit": PROBE_MAX_WINDOWS,
        # The error bar is over documents. On this path the denominator the run has is the distinct sequences,
        # not the windows it asked for, so both counts are in the record and the difference between them is
        # visible rather than folded away.
        "error_bar_denominator": "distinct sequences",
    }

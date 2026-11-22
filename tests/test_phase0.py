"""
Phase 0 harness tests.

These are not quality tests. A randomly initialized tiny model has no quality to test, and a 20-step run
cannot decide a threshold stated at 200-250M params. What these tests check is the machinery that makes a
gate result mean something:

  - arms differ in exactly one thing and share initialization and data order, so a loss difference is
    attributable to the variable being tested;
  - thresholds come from ROADMAP rather than from the report, and a number that is not in ROADMAP is
    marked as a proposal rather than presented as a gate;
  - the fake-quant path is stable against the stored scale, which is the only version of idempotence the
    engine can actually reproduce;
  - the cache eviction order cannot silently change the numbers. This is a regression test for a real bug
    in this reference: trimming to the window before the window mask is applied produced a fully masked
    row, and a fully masked row is nan rather than an error;
  - the gate-scale data stream is deterministic and never draws a held-out sequence;
  - a checkpoint resume reproduces the uninterrupted run exactly, weights and orders included;
  - the records a long run writes are bounded by the rotation policy, so a gate-scale run does not reach a
    storage quota before it reaches its token budget, and the current record keeps the name `--resume` reads;
  - the recorded orders are checked against their own hash before a run resumes on them, so a truncated or
    edited record cannot quietly become the data order an arm was measured under;
  - the arithmetic that produced a number is read from torch and written into the record rather than assumed,
    because a TF32 reading is not the same number as the CPU reference's reading;
  - the eval size the numeric thresholds need is stated as a condition with its assumption named, and the
    error bar is computed over documents rather than over tokens;
  - the synthetic path is refused before any training when the eval size it is handed is larger than the task's
    sequence space, and the count it can supply is reported as what it is: distinct sequences, not documents. An
    eval size above the probe's stated limit is refused as unprobed rather than checked by materializing it;
  - `--family` drives the families the run draws rather than only the label on the number the gate reads, so a
    run cannot report a number for a family it never drew;
  - identical baselines train once and reuse the same model and numbers, while every single-variable arm
    keeps a distinct training identity;
  - the corpus builder's document rules are pure functions of identity, so the held-out set is the same
    documents at every corpus size; the policy's two problem kinds stay distinguishable and its counts are
    read back from the manifest; and the document index is checked against the shard bytes it claims rather
    than trusted, so a window can be traced to the document it measured.

Run: python tests/test_phase0.py
"""

import contextlib
import io
import json
import sys
import tempfile
from array import array
from pathlib import Path

# The project is not installed as a package, so make the project root importable when the tests are run
# directly. Same reason as in test_reference.py.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from dataclasses import replace

from heronlm.config import PHASE0, TINY
from heronlm.layers import Attention, Cache
from heronlm.memory import slot_ids
from heronlm.model import HeronLM
from phase0 import gates
from phase0.arms import RunSpec, build_arms, verify_arms
from phase0.corpus import (PROBE_MAX_WINDOWS, SampleStream, eval_supply, make_corpus, sequence_space)
import scripts.phase0 as driver
from phase0.tokens import (ShardWriter, TokenizedCorpus, TokenStream, read_manifest, write_manifest)
from phase0 import documents
from heronlm.metadata import SPECIAL_TAGS, placeholder_vocabulary, special_token_ids
from phase0.quant import (
    DENSE_INT4,
    EXCLUSION_REASONS,
    PARITY_RULE,
    Q8_0,
    TABLE_INT8,
    dequantized,
    fake_quantize_model,
    path_stability,
    ptq_delta,
    quantize,
    quantized_stored_weight,
    requantize_is_stable,
    scheme_for,
)
from phase0.train import (
    order_arrays,
    orders_from_arrays,
    orders_from_payload,
    orders_sha256,
    param_groups,
    precision_flags,
    run_gate,
    set_tf32,
    train_arm,
    training_key,
)

DEV = torch.device("cpu")


def test_arms_are_single_variable():
    """Every arm differs from its baseline in exactly one thing, and the label matches the actual field."""
    small = build_arms(TINY(), RunSpec(steps=1))
    assert not verify_arms(small), verify_arms(small)

    # The gate scale uses the same generator, so the same property has to hold there. The only difference is
    # which depth the trained-vs-beyond-trained pair lands on, and that is derived from max_loop rather than
    # hardcoded, which is what keeps the tiny arms the same arms.
    big = build_arms(PHASE0(), RunSpec(steps=1))
    assert not verify_arms(big), verify_arms(big)
    trained = big["G0.1"][1].spec.train_k[0]
    assert trained == 3, f"at the gate scale the trained depth must be 3, not {trained}"
    assert trained < big["G0.1"][1].spec.eval_k[-1], "the comparison must be beyond trained depth"

    # Eval depth is capped by the config. Requesting k=4 on a config whose max_loop is 3 would index read
    # weights past the trained distribution, so the number it produced would measure nothing.
    assert small["G0.1"][0].spec.eval_k == tuple(range(1, TINY().max_loop + 1))

    # G0.4 is deliberately not represented. It is device time on the S23 build and the reference has no
    # GDN path, so an arm for it would produce a loss number where the gate needs a time number.
    assert "G0.4" not in small and "G0.4" not in big
    assert not gates.gate("G0.4").harness_decidable


def test_arms_share_initialization_and_data_order():
    """
    Same seed, same weights, same sequences in the same order.

    Without both, a loss difference between two arms could come from the initialization or from the data
    order rather than from the flag under test. Checked from what the run recorded, not from a re-derived
    formula.
    """
    spec = RunSpec(steps=6)
    arms = build_arms(TINY(), spec)["G0.2"]
    corpus = make_corpus(spec.seed, length=spec.seq_len, n_train=spec.n_train, n_test=spec.n_test,
                         families=spec.families)

    a = train_arm(arms[0], corpus, DEV, log_every=1)
    b = train_arm(arms[1], corpus, DEV, log_every=1)

    assert a["data_order"] == b["data_order"], "arms must see the same sequences in the same order"
    assert a["k_order"] == b["k_order"], "arms that share train_k must sample the same k per batch"

    shared = set(a["initial_state"]) & set(b["initial_state"])
    assert shared, "arms must be comparable on the same parameters"
    for name in shared:
        assert torch.equal(a["initial_state"][name], b["initial_state"][name]), name

    # Param-count neutral, so a loss difference cannot be attributed to size.
    pa = sum(p.numel() for p in a["model"].parameters())
    pb = sum(p.numel() for p in b["model"].parameters())
    assert pa == pb, (pa, pb)

    # G0.6 is the gate where the variable is the distribution, so the sampled arm must actually sample.
    # If it drew one k for every step it would be measuring the fixed arm twice.
    g06 = build_arms(TINY(), RunSpec(steps=12))["G0.6"]
    r = train_arm(g06[1], corpus, DEV, log_every=1)
    assert len(set(r["k_order"])) > 1, f"sampled arm drew only {sorted(set(r['k_order']))}"
    assert sorted(g06[1].spec.train_k) == list(range(1, TINY().max_loop + 1))
    # The fixed arm cannot show the distribution at all, which is why the gate needs two runs.
    assert len(g06[0].spec.train_k) == 1


def test_sample_stream_is_fresh_deterministic_and_excludes_eval():
    """
    The gate-scale data fix, checked rather than asserted in prose.

    A fixed 64-sample pool is fine for a mechanism smoke test but not for a 1-2B-token arm: the model sees
    each sequence hundreds of times and the held-out loss stops being held out. The stream has to satisfy
    three properties at once: the same index yields the same sample (so arms and a resumed run agree), a
    long run is not confined to the old 64 sequences, and no eval sequence appears in training.

    The synthetic task's input space is finite - lookup admits 48 sequences and recurrence 48 squared - so
    the assertion is not "every draw is new". It is that the recurrence half alone exceeds the old cap
    several times over and that the held-out sequences are never drawn.
    """
    families = ("lookup", "recurrence")
    eval_corpus = make_corpus(7, length=48, n_train=0, n_test=32, families=families)["test"]
    excluded = {s.tokens for s in eval_corpus}

    def stream() -> SampleStream:
        return SampleStream(7, length=48, families=families, split="train",
                            exclude=[s.tokens for s in eval_corpus])

    train, same = stream(), stream()
    first = [train[i] for i in range(400)]
    assert [s.tokens for s in first] == [same[i].tokens for i in range(400)], "the stream is not deterministic"
    assert all(len(s.tokens) == 48 for s in first), "every sample must have the batch length"
    assert not ({s.tokens for s in first} & excluded), "a train draw landed on a held-out sequence"

    recurrence = [s.tokens for s in first if s.family == "recurrence"]
    assert len(set(recurrence)) > 64, f"only {len(set(recurrence))} distinct recurrence sequences in 200 draws"

    # The split is part of the per-sample seed, so the eval stream is a different draw even though it shares
    # the fact pools on purpose. Otherwise the exclusion list would be excluding sequences training already
    # generated.
    eval_stream = SampleStream(7, length=48, families=families, split="test")
    assert any(train[i].tokens != eval_stream[i].tokens for i in range(20)), (
        "the train and eval streams produced the same sequences; split is not part of the seed"
    )


def test_checkpoint_resume_is_exact():
    """
    A long arm has to survive a crash without becoming a different run.

    The test is the equality, not the existence of a file: three steps followed by a resume for three more
    must produce the same weights, the same data order, the same k order and the same loss as six
    uninterrupted steps. That requires the checkpoint to carry the optimizer state and both RNG states, not
    just the weights; a weights-only checkpoint would restart Adam and shift the loss even if every sequence
    was the same.

    The sampled-k arm is used on purpose: its k generator advances every step, so a resume that forgot that
    state would draw a different depth sequence and fail the k_order equality before the weights were even
    compared.
    """
    families = ("lookup", "recurrence")
    corpus = make_corpus(7, length=48, n_train=0, n_test=32, families=families)
    exclude = [s.tokens for s in corpus["test"]]

    def stream() -> SampleStream:
        return SampleStream(7, length=48, families=families, split="train", exclude=exclude)

    six = build_arms(TINY(), RunSpec(steps=6, batch=2))["G0.6"][1]
    three = replace(six, spec=replace(six.spec, steps=3))
    full = train_arm(six, corpus, DEV, log_every=1, train_stream=stream())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sampled_k.pt"
        part = train_arm(three, corpus, DEV, log_every=1, train_stream=stream(), checkpoint_path=path)
        assert path.exists(), "the arm did not write a checkpoint"
        assert part["history"] and len(part["history"]) == 3

        resumed = train_arm(six, corpus, DEV, log_every=1, train_stream=stream(),
                            checkpoint_path=path, resume=True)
        assert resumed["resumed_from"] == 3, resumed["resumed_from"]

        assert resumed["data_order"] == full["data_order"], (resumed["data_order"], full["data_order"])
        assert resumed["k_order"] == full["k_order"], (resumed["k_order"], full["k_order"])
        assert resumed["history"] == full["history"], (resumed["history"], full["history"])
        assert resumed["train_loss"] == full["train_loss"], (resumed["train_loss"], full["train_loss"])
        for name, tensor in full["model"].state_dict().items():
            assert torch.equal(tensor, resumed["model"].state_dict()[name]), name

        # A checkpoint from another arm must be refused rather than resumed into a run whose result would not
        # belong to this gate. The failure has to happen before any step is taken.
        other = build_arms(TINY(), RunSpec(steps=6, batch=2))["G0.2"][0]
        try:
            train_arm(other, corpus, DEV, train_stream=stream(), checkpoint_path=path, resume=True)
        except ValueError as e:
            assert "different arm" in str(e), str(e)
        else:
            raise AssertionError("a checkpoint from a different arm must not be resumed")

        # Same arm, different corpus. The arm name and config match, so only the corpus id in the run key can
        # catch this; without it the resumed run would keep the synthetic weights and continue on other data,
        # and its loss would belong to neither gate.
        different_data = SampleStream(8, length=48, families=families, split="train", exclude=exclude)
        try:
            train_arm(six, corpus, DEV, train_stream=different_data, checkpoint_path=path, resume=True)
        except ValueError as e:
            assert "different arm" in str(e), str(e)
        else:
            raise AssertionError("a checkpoint from a different corpus must not be resumed")


def test_identical_baselines_are_trained_once_and_reused_exactly():
    """
    Five of the ten arms are the same training run under different labels.

    G0.1's baseline, G0.2's ngpt arm, G0.3's gr_on arm, G0.5's shared_cache arm and G0.6's fixed_k1 arm all
    use the default config, the default spec and seed 7. Retraining that run per gate is five times the
    compute for five bit-identical models, which at the gate scale is days of GPU time. The cache key is the
    guarantee: it has to omit the labels and include everything that can move a weight.

    The test checks both directions. Identical runs must reuse the same model object and the same measured
    numbers, and every non-baseline arm must have a different key, or a real single-variable difference would
    be silently replaced by the baseline.
    """
    spec = RunSpec(steps=1, batch=1, n_test=2)
    gates_by_arm = build_arms(TINY(), spec)
    baselines = [arms[0] for arms in gates_by_arm.values()]

    # The premise first: apart from the labels, these really are the same run. If a future arm edit gives one
    # baseline a different config, the reuse assertion below should fail rather than quietly share the wrong
    # model.
    assert len({training_key(arm, "corpus") for arm in baselines}) == 1
    for arm in baselines[1:]:
        assert arm.cfg == baselines[0].cfg
        assert arm.spec == baselines[0].spec

    cache, first_by_key, entries = {}, {}, 0
    for gate_id, arm_list in gates_by_arm.items():
        results = run_gate(gate_id, arm_list, DEV, cache=cache)
        for entry in results["arms"]:
            entries += 1
            key = training_key(entry["arm"], entry["result"]["corpus"])
            if key not in first_by_key:
                assert entry["reused"] is False, (gate_id, entry["arm"].name)
                # The gate path must not clone every parameter for the test aid; that clone is 1.8 GB per
                # arm at PHASE0 scale.
                assert entry["result"]["initial_state"] == {}, entry["arm"].name
                first_by_key[key] = entry
            else:
                owner = first_by_key[key]
                assert entry["reused"] is True, (gate_id, entry["arm"].name)
                assert entry["trained_as"] == owner["arm"].name
                assert entry["result"]["model"] is owner["result"]["model"], (
                    "a reused baseline got a different model object, so it was retrained"
                )
                assert entry["by_k"] is owner["by_k"], "a reused baseline was re-evaluated"

    assert entries == 10, entries
    assert len(first_by_key) == 6, sorted(e["arm"].name for e in first_by_key.values())

    # Every second arm differs in exactly the variable it is testing, so none may share the baseline key.
    baseline_key = training_key(baselines[0], "corpus")
    for gate_id in ("G0.1", "G0.2", "G0.3", "G0.5", "G0.6"):
        other = gates_by_arm[gate_id][1]
        assert training_key(other, "corpus") != baseline_key, (gate_id, other.name)


def test_token_shards_stream_deterministically_without_split_leakage():
    """
    The real-text path, exercised without a tokenizer or a download.

    What has to hold for the 1-2B-token budget to mean anything:
      - a window is `seq_len` inputs plus next-token targets, and never crosses a shard boundary;
      - sample i is the same for the same seed, so arms and a resumed run agree;
      - an epoch visits every window exactly once, so the budget is spent on the corpus rather than on a
        biased subset;
      - validation windows come only from val shards, so held-out loss cannot be contaminated by training.

    The shards here are arbitrary uint32 values in a valid vocabulary range. Nothing in the loader depends on
    them being real text; the tokenizer script is the only part that does.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "corpus"
        # 40 tokens per shard and seq_len 8 gives four windows per shard and two shards. Values below 2000
        # are train, values at 5000 and above are val, so a leak is visible in the token values themselves.
        with ShardWriter(root, "train", shard_tokens=40, vocab_size=6000) as writer:
            writer.add(range(0, 40))
            writer.add(range(1000, 1040))
        with ShardWriter(root, "val", shard_tokens=40, vocab_size=6000) as writer:
            writer.add(range(5000, 5020))
        shards = [
            {"path": p.name, "split": "train" if p.name.startswith("train") else "val",
             "tokens": p.stat().st_size // 4}
            for p in sorted(root.glob("*.bin"))
        ]
        write_manifest(root, shards, vocab_size=6000)

        corpus = TokenizedCorpus.load(root)
        try:
            train = corpus.split("train", seq_len=8)
            val = corpus.split("val", seq_len=8)
            stream = TokenStream(train, seed=7)
            same = TokenStream(train, seed=7)

            first = [stream[i] for i in range(32)]
            assert [s.tokens for s in first] == [same[i].tokens for i in range(32)], (
                "the token stream is not deterministic"
            )
            for sample in first:
                assert len(sample.tokens) == 8 and len(sample.targets) == 8
                assert sample.targets == sample.tokens[1:] + (-1,), "targets are not next-token shifted"
                assert sample.family == "text"
                assert max(sample.tokens) < 2000, "a training window came from the validation shard"

            # Every window lies entirely inside one shard: the last target position needs one more token.
            for index in range(len(train)):
                name, start = train.source_window(index)
                tokens_in_shard = next(s["tokens"] for s in shards if s["path"] == name)
                assert start + 8 + 1 <= tokens_in_shard, (name, start, tokens_in_shard)

            # Across one epoch the affine permutation must cover each window exactly once. Repeating the
            # check on the next epoch shows the epoch index is part of the cycle, not ignored.
            count = len(train)
            assert {stream.window_index(i) for i in range(count)} == set(range(count))
            assert {stream.window_index(count + i) for i in range(count)} == set(range(count))

            eval_samples = [val.sample(i) for i in range(len(val))]
            assert eval_samples and all(min(s.tokens) >= 5000 for s in eval_samples), (
                "a validation window contains training tokens"
            )
            assert all(max(s.tokens) < 5020 for s in eval_samples)
        finally:
            corpus.close()


def test_token_corpus_drives_the_gate_without_family_confusion():
    """
    The wiring from `--corpus-dir` through `run_gate` to evaluation.

    A text corpus has one family, and the synthetic control family does not exist in it. If the measurement
    still asked for "recurrence", every loss would be NaN and the gate would print a direction fail that is
    really a missing family. This test runs one TINY arm for one step and checks the recorded corpus id, the
    eval family and the held-out shard.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "corpus"
        # TINY's vocabulary is 512, so the ids have to stay below it. 2 validation windows at seq_len 48 need
        # 98 tokens; 100 leaves one spare token that no window can consume.
        with ShardWriter(root, "train", shard_tokens=64, vocab_size=512) as writer:
            writer.add(range(0, 64))
            writer.add(range(100, 164))
        with ShardWriter(root, "val", shard_tokens=128, vocab_size=512) as writer:
            writer.add(range(300, 400))
        shards = [
            {"path": p.name, "split": "train" if p.name.startswith("train") else "val",
             "tokens": p.stat().st_size // 4}
            for p in sorted(root.glob("*.bin"))
        ]
        write_manifest(root, shards, vocab_size=512)

        spec = RunSpec(steps=1, seq_len=48, n_test=2)
        gate = run_gate("G0.1", build_arms(TINY(), spec)["G0.1"][:1], DEV, corpus_dir=root)
        entry = gate["arms"][0]

        assert entry["result"]["corpus"].startswith("tokens:"), entry["result"]["corpus"]
        assert [s.family for s in entry["result"]["corpus_test"]] == ["text", "text"]
        assert all(min(s.tokens) >= 300 for s in entry["result"]["corpus_test"])
        assert set(entry["by_k"][1]["by_family"]) == {"text"}, entry["by_k"][1]["by_family"]


def test_document_identity_and_split_are_pure_functions_that_survive_a_reorder():
    """
    The held-out set has to survive a rerun, a re-sort, and a corpus that grows.

    Identity is the only randomness in the builder: one hash of identity and seed decides both which documents
    enter the corpus and which split a document belongs to. The consequences checked here are what make a
    held-out number comparable:

      - a walk in a different order decides the same split for the same document, so a corpus read in a
        different order is the same corpus;
      - selection is monotone in the fraction, so a corpus built at a smaller fraction is a strict subset of
        the same corpus at a larger one rather than a different sample of it. That is the only way a corpus can
        grow without changing its eval set, and it is why size has to be reached by a rule rather than by
        stopping at a token count: a stop is a prefix of one walk order, and a prefix has a different eval set;
      - `val_fraction` cannot exceed `select_fraction`, because the eval rule would then name documents the
        selection rule excludes, and a corpus cannot contain its own eval set. Refused as a list before any
        document is read, which is what stops 150M tokens being written under a rule that could not have
        produced a corpus the loader would accept.

    Row identity is positional, and that is stated as what it is: the absolute position in the file, not the
    position inside a read batch. FineWeb-Edu carries no id column, so nothing makes a row's identity stable
    across a re-sort of the same rows. A corpus whose identity is positional can be reproduced from the same
    files, not from a different ordering of them.
    """
    seed, val_fraction = 7, 0.05
    rows = [f"part-{i:02d}.parquet:{row}" for i in range(3) for row in range(400)]

    # The identity shape first, because a hash of the wrong thing decides the wrong split.
    assert documents.document_identity("part-00.parquet", 3) == "part-00.parquet:3"
    assert documents.document_identity("docs/a.txt", None) == "docs/a.txt", "a text file is one document"
    assert documents.document_identity("part-00.parquet", 3) != documents.document_identity("part-01.parquet", 3)
    for call, expected in ((("", 0), "needs a source path"),
                           (("", None), "needs a source path"),
                           (("part-00.parquet", -1), "non-negative")):
        try:
            documents.document_identity(*call)
        except ValueError as e:
            assert expected in str(e), e
        else:
            raise AssertionError(f"identity {call} must be refused")
    try:
        documents.unit("", 7)
    except ValueError as e:
        assert "needs an identity" in str(e), e
    else:
        raise AssertionError("an empty identity must be refused")

    assert all(0.0 <= documents.unit(identity, seed) < 1.0 for identity in rows)
    assert documents.unit("part-00.parquet:3", 7) == documents.unit("part-00.parquet:3", 7)
    assert documents.unit("part-00.parquet:3", 7) != documents.unit("part-00.parquet:3", 8), (
        "the seed is part of the decision, so a different seed is a different eval set"
    )

    # Two walks, different orders, the same decision per document. A rule that used the position in the walk
    # would move documents between splits on a rerun, and a held-out set that moves is not held out.
    first, second = {}, {}
    for identity in rows:
        first[identity] = documents.split_of(identity, seed, val_fraction)
    for identity in sorted(rows, reverse=True):
        second[identity] = documents.split_of(identity, seed, val_fraction)
    assert first == second, "the split depends on walk order"

    # The strict-subset property, stated as sets rather than as a share. Equality is not claimed, because the
    # documents the selection rule excluded cannot be evaluated in a corpus that excluded them; what the rule
    # guarantees is that the eval set at the smaller size is a subset of the eval set at the larger one, and
    # exactly the part of it the smaller corpus contains.
    small = {identity for identity in rows if documents.selected(identity, seed, 0.1)}
    whole = {identity for identity in rows if documents.selected(identity, seed, 1.0)}
    assert small <= whole, "a corpus built at a smaller fraction must be a subset of the same corpus"
    val_small = {identity for identity in small if documents.split_of(identity, seed, val_fraction) == "val"}
    val_whole = {identity for identity in whole if documents.split_of(identity, seed, val_fraction) == "val"}
    assert val_small and val_whole, (len(val_small), len(val_whole))
    assert val_small == {identity for identity in val_whole if identity in small}, (
        "the eval set is the same documents at both corpus sizes"
    )
    assert all(documents.selected(identity, seed, 0.1) <= documents.selected(identity, seed, 0.2)
               for identity in rows), "selection is not monotone in the fraction"

    ids = special_token_ids(placeholder_vocabulary(200))
    eos = ids["endoftext"]

    # The conditions the builder can check before it reads a single document.
    problems = documents.build_conditions("skip", seed, val_fraction, 0.02, 200, ids, eos)
    assert problems and "larger than select_fraction" in problems[0], problems
    assert documents.build_conditions("skip", seed, 0.02, 0.05, 200, ids, eos) == []
    assert documents.build_conditions("skip", seed, 0.0, 1.0, 200, ids, eos), (
        "a val fraction of 0 produces no eval set, which is not a small eval set"
    )
    assert documents.build_conditions("unknown", seed, val_fraction, 1.0, 200, ids, eos), (
        "an unknown policy is a condition, not an exception"
    )
    # The boundary token this builder appends has to be inside the model's embedding. On the pinned file the
    # added tokens sit above the range the model can embed, which is what --vocab-limit is for.
    assert any(str(eos) in p for p in documents.build_conditions("skip", seed, val_fraction, 1.0, eos, ids, eos))


def test_document_policy_states_and_counts_are_read_back_from_the_manifest():
    """
    A policy is a decision about documents, and the record has to show it.

    The two problem kinds stay different because they are different problems. `out_of_range` is a
    representation problem: the id is above the model's embedding, so no model can embed the document.
    `special_in_text` is a measurement problem: the id is inside the text vocabulary but it is a control token
    that came from the page's own text rather than from the builder's document boundary, and a held-out loss
    over a corpus full of turn delimiters measures delimiter prediction rather than what the gate measures. A
    count that merged them would say neither.

    The counts are read back from the manifest rather than from the builder's return value, because the record
    is what a resumed run and a later comparison read. `extra` is inside the digest on purpose: a corpus that
    dropped documents is a different corpus from one that dropped none, so the policy, the fractions and the
    counts belong to the corpus identity rather than beside it.

    The two states that are not a count are checked as states: `refuse` stops at the document it trips on and
    writes no manifest, so the directory is not a corpus; and the val fallback names a document the rule did not
    choose, which is recorded as a deviation rather than as the rule.
    """
    seed, val_fraction = 7, 0.05
    vocabulary = placeholder_vocabulary(200)
    ids = special_token_ids(vocabulary)
    eos = ids["endoftext"]
    position_of = {token: index for index, token in enumerate(vocabulary)}

    def encode(text: str):
        # The fake encoder is the only way these rules can be exercised without a tokenizer, a download, or a
        # parquet reader. It behaves like the pinned one in the one way that matters here: the added tokens are
        # matched even though no special tokens are added, so a tag inside the text produces its id inside the
        # document.
        return [position_of[word] for word in text.split()]

    clean, trips = "tok5 tok6 tok7", f"tok8 {SPECIAL_TAGS['im_end']} tok9"
    found = documents.policy_problems(encode(trips), set(ids.values()), 200)
    assert len(found) == 1, found
    assert found[0]["kind"] == documents.PROBLEM_SPECIAL_IN_TEXT, found
    assert (found[0]["position"], found[0]["id"]) == (1, ids["im_end"]), found
    assert documents.policy_problems(encode(clean), set(ids.values()), 200) == []

    # The same id under a lower limit is a different problem, and the reason it is different is that a model
    # with that embedding cannot represent the document at all. `build_conditions` refuses this case before any
    # document is read, so `skip` cannot quietly drop documents the model could never have trained on.
    lower = documents.policy_problems(encode(trips), set(ids.values()), 150)
    assert lower[0]["kind"] == documents.PROBLEM_OUT_OF_RANGE, lower
    problems = documents.build_conditions("skip", seed, val_fraction, 1.0, 150, ids, eos)
    assert any(str(ids["im_end"]) in p for p in problems), problems

    # Three states on the same document. `refused` returns no ids because the caller stops.
    for policy, state in (("skip", "skipped"), ("strip", "stripped"), ("refuse", "refused")):
        action, kept = documents.apply_policy(encode(trips), found, policy)
        assert action == state and action in documents.ACTIONS, (policy, action)
        assert kept == ([8, 9] if policy == "strip" else None), (policy, kept)
        assert documents.apply_policy(encode(clean), [], policy)[0] == "kept", policy
    try:
        documents.apply_policy(encode(clean), [], "keep")
    except ValueError as e:
        assert "unknown policy" in str(e), e
    else:
        raise AssertionError("an unknown policy must be refused")

    rows = [("part-00.parquet", row) for row in range(600)]
    val_rows = [p for p in rows if documents.split_of(documents.document_identity(*p), seed, val_fraction) == "val"]
    train_rows = [p for p in rows if documents.split_of(documents.document_identity(*p), seed, val_fraction) == "train"]
    assert len(val_rows) >= 2 and len(train_rows) >= 2, (len(val_rows), len(train_rows))

    def read_shard(path: Path) -> list[int]:
        # The check is against the bytes, not against the writer's bookkeeping: an index that agrees with the
        # writer and disagrees with the file is describing a corpus that does not exist.
        buffer = array("I")
        buffer.frombytes(Path(path).read_bytes())
        assert buffer.itemsize == 4, "the shard format is uint32"
        return list(buffer)

    with tempfile.TemporaryDirectory() as tmp:
        # `skip`: one document kept per split, one dropped by the policy, one empty row. The empty row and the
        # policy drop are both counted, because they say how much of the source is not inside the number being
        # reported.
        root = Path(tmp) / "skip"
        stream = [(val_rows[0][0], val_rows[0][1], clean, 11),
                  (val_rows[1][0], val_rows[1][1], trips, 22),
                  (train_rows[0][0], train_rows[0][1], clean, 30),
                  (train_rows[1][0], train_rows[1][1], "   ", 40)]
        result = documents.build_corpus(root, stream, encode, seed=seed, val_fraction=val_fraction,
                                        vocab_size=200, special_ids=ids, eos_id=eos, policy="skip",
                                        shard_tokens=20)
        assert not result["problems"], result["problems"]

        manifest = read_manifest(root)
        counts = manifest["counts"]
        assert manifest["policy"]["name"] == "skip", manifest["policy"]
        assert manifest["policy"]["seed"] == seed and manifest["policy"]["val_fraction"] == val_fraction
        assert manifest["policy"]["special_ids"] == {str(k): int(v) for k, v in sorted(ids.items())}, (
            "the record has to name the ids it resolved, not the numbers it remembered"
        )
        assert manifest["policy"]["end_of_text_id"] == eos, manifest["policy"]
        assert counts["documents_seen"] == 4 and counts["not_selected"] == 0, counts
        assert counts["splits"]["val"]["documents"] == 1 and counts["splits"]["val"]["skipped_policy"] == 1, counts
        assert counts["splits"]["train"]["documents"] == 1 and counts["splits"]["train"]["skipped_empty"] == 1, counts
        assert counts["splits"]["val"]["tokens"] == len(encode(clean)) + 1, counts

        source = manifest["sources"][0]
        assert source["identity_format"] == "path:row" and source["rows"] == 4, source
        # The claim is recorded only for documents this build kept, so it is comparable with the count the
        # manifest reports. A source's own count covers rows this build never stored, and a comparison against a
        # number that excludes them is a comparison against a set the manifest does not describe.
        assert source["claimed_tokens"] == 41, source
        assert source["val_documents"] == 1 and source["train_documents"] == 1, source

        # `extra` is inside the digest, so an edit to the record is visible. A policy record that could be
        # changed without anything noticing is a comment, not a record.
        edited = dict(manifest)
        edited["counts"] = dict(counts, documents_seen=counts["documents_seen"] + 1)
        (root / "manifest.json").write_text(json.dumps(edited, indent=2) + "\n", encoding="utf-8")
        try:
            read_manifest(root)
        except ValueError as e:
            assert "digest does not match" in str(e), e
        else:
            raise AssertionError("the policy record must be inside the corpus digest")

        # A record cannot overwrite a key the loader reads. If it could, the digest would describe a payload the
        # reader does not build.
        try:
            write_manifest(root, manifest["shards"], vocab_size=200, extra={"format": 2})
        except ValueError as e:
            assert "conflicts with a format key" in str(e), e
        else:
            raise AssertionError("an extra manifest key must not be able to overwrite a format key")

        # `strip` keeps the document and removes only the ids that tripped the policy, not the boundary token
        # the builder appends after the decision is made. Read back against the shard bytes.
        stripped = Path(tmp) / "strip"
        result = documents.build_corpus(stripped, [(val_rows[0][0], val_rows[0][1], trips, 3),
                                                   (train_rows[0][0], train_rows[0][1], clean, 3)],
                                        encode, seed=seed, val_fraction=val_fraction, vocab_size=200,
                                        special_ids=ids, eos_id=eos, policy="strip", shard_tokens=20)
        assert not result["problems"], result["problems"]
        assert result["counts"]["splits"]["val"]["ids_stripped"] == 1, result["counts"]
        stripped_manifest = read_manifest(stripped)
        val_shard = [s for s in stripped_manifest["shards"] if s["split"] == "val"][0]
        assert read_shard(stripped / val_shard["path"]) == [8, 9, eos], "strip must keep the boundary token"

        # `refuse` stops at the document it trips on and writes no manifest, so the directory is not a corpus.
        # It cannot report a problem it has not read yet, so the counts it returns are the counts up to that
        # document, and the earlier documents are counted because they were actually read. The refused document
        # comes last here so both splits have a kept document: otherwise the build would report the split it
        # could not write instead of the document it refused, and the two are different findings.
        refused = Path(tmp) / "refuse"
        result = documents.build_corpus(refused, [(val_rows[0][0], val_rows[0][1], clean, 3),
                                                  (train_rows[0][0], train_rows[0][1], clean, 3),
                                                  (val_rows[1][0], val_rows[1][1], trips, 3)],
                                        encode, seed=seed, val_fraction=val_fraction, vocab_size=200,
                                        special_ids=ids, eos_id=eos, policy="refuse", shard_tokens=20)
        assert result["manifest"] is None and len(result["problems"]) == 1, result
        assert result["problems"][0]["identity"] == documents.document_identity(*val_rows[1]), result["problems"]
        assert result["counts"]["splits"]["val"]["documents"] == 1, result["counts"]
        assert result["counts"]["splits"]["train"]["documents"] == 1, result["counts"]
        try:
            read_manifest(refused)
        except FileNotFoundError as e:
            assert "no manifest.json" in str(e), e
        else:
            raise AssertionError("a build that refused must not leave a corpus behind")

        # The val fallback, checked as the deviation it is. The two rules are the same number compared with two
        # thresholds, so a source can be selected and still have no eval set: a val threshold below the number
        # any identity produces selects documents and puts none of them in validation. The rule then cannot
        # produce a corpus, so the caller names one document - and the manifest records that the rule did not
        # choose it.
        val_tiny, select = 1e-6, 0.02
        chosen = [p for p in rows if documents.selected(documents.document_identity(*p), seed, select)]
        landed_in_val = [p for p in chosen if documents.split_of(documents.document_identity(*p), seed, val_tiny) == "val"]
        assert len(chosen) >= 2 and not landed_in_val, (len(chosen), landed_in_val)

        fallback = Path(tmp) / "fallback"
        result = documents.build_corpus(fallback, [(p[0], p[1], clean, 3) for p in chosen], encode,
                                        seed=seed, val_fraction=val_tiny, select_fraction=select,
                                        vocab_size=200, special_ids=ids, eos_id=eos, shard_tokens=20)
        assert result["manifest"] is None, "a rule that produced no val document cannot produce a corpus"
        assert result["problems"] and "no document landed in the val split" in result["problems"][0], result["problems"]

        named = documents.document_identity(*chosen[0])
        result = documents.build_corpus(fallback, [(p[0], p[1], clean, 3) for p in chosen], encode,
                                        seed=seed, val_fraction=val_tiny, select_fraction=select,
                                        vocab_size=200, special_ids=ids, eos_id=eos, shard_tokens=20,
                                        force_val=named)
        assert not result["problems"], result["problems"]
        record = read_manifest(fallback)["policy"]["val_fallback"]
        assert record["requested"] == named and record["applied"] is True, record
        entries = documents.read_document_index(fallback)
        val_entries = [e for e in entries if e["split"] == "val"]
        assert len(val_entries) == 1 and val_entries[0]["identity"] == named, val_entries
        # The document is still one the selection rule admitted; the fallback only says which of them to treat
        # as held out, so the rule itself is unchanged.
        assert documents.split_of(named, seed, val_tiny) == "train", named


def test_the_document_index_is_verified_against_the_shard_bytes_it_claims():
    """
    A window is a slice of a shard, so a number measured on a window belongs to a document only if the
    boundaries were recorded - and the record is a claim, so it is checked at load rather than trusted.

    Three things have to agree: the manifest's shard sizes, the index's positions, and the bytes on disk. The
    reconstruction below reads the shards back and compares them with what the encoder produced, which is the
    only check that does not depend on the code that wrote the file.

    A document can legitimately straddle two shards, because a flush happens at a fixed token count rather than
    at a document boundary, so the check allows that and refuses a position that disagrees with the shard it
    names. That is also why the index records a logical offset: the shard a document lands in is not known until
    `close()` has folded the short remainder into the last shard, and a name captured at write time can name a
    shard that will not exist.

    The crossing flag is the honest half of the attribution. Two documents in a shard are adjacent with the
    boundary token between them, so a fixed-length window can cover the end of one document and the start of the
    next. That is not leakage and not a bug; it is what a window over a run of documents does. A number
    attributed to "the document this window came from" is only true when the window does not cross.
    """
    seed, val_fraction = 7, 0.05
    vocabulary = placeholder_vocabulary(200)
    ids = special_token_ids(vocabulary)
    eos = ids["endoftext"]
    position_of = {token: index for index, token in enumerate(vocabulary)}

    def encode(text: str):
        return [position_of[word] for word in text.split()]

    def read_shard(path: Path) -> list[int]:
        buffer = array("I")
        buffer.frombytes(Path(path).read_bytes())
        return list(buffer)

    rows = [("part-00.parquet", row) for row in range(4000)]
    val_rows = [p for p in rows if documents.split_of(documents.document_identity(*p), seed, val_fraction) == "val"]
    train_rows = [p for p in rows if documents.split_of(documents.document_identity(*p), seed, val_fraction) == "train"]
    assert len(val_rows) >= 4 and len(train_rows) >= 1, (len(val_rows), len(train_rows))

    # Document lengths that are not divisors of the shard size, which is what makes a straddle possible at all:
    # three ids plus the boundary token, four ids plus the boundary token, and one long document.
    groups = [["tok1", "tok2", "tok3"], ["tok4", "tok5", "tok6", "tok7"],
              [f"tok{50 + i}" for i in range(21)]]
    stream = [(p[0], p[1], " ".join(groups[i]), 3) for i, p in enumerate(val_rows[:3])]
    stream.append((train_rows[0][0], train_rows[0][1], " ".join(groups[0]), 3))
    # What each document should read back as: the ids the encoder produced plus the boundary token the builder
    # appends after the policy decision. This is the expectation the index is compared with, so it is written
    # from the encoder rather than from the writer.
    expected = {documents.document_identity(s[0], s[1]): encode(s[2]) + [eos] for s in stream}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "corpus"
        result = documents.build_corpus(root, stream, encode, seed=seed, val_fraction=val_fraction,
                                        vocab_size=200, special_ids=ids, eos_id=eos, shard_tokens=12)
        assert not result["problems"], result["problems"]
        manifest = read_manifest(root)

        # The index and the reconstruction have to describe the same bytes, in the order the writer appended
        # them. Reading the shards back in manifest order is what makes this independent of the writer.
        entries = documents.read_document_index(root, manifest)
        assert len(entries) == 4, entries
        for split in ("train", "val"):
            shards = [s for s in manifest["shards"] if s["split"] == split]
            rebuilt = [token for shard in shards for token in read_shard(root / shard["path"])]
            ordered = [e for e in entries if e["split"] == split]
            assert rebuilt == [token for e in ordered for token in expected[e["identity"]]], (split, rebuilt)
            for entry in ordered:
                ids_for_entry = expected[entry["identity"]]
                shard_tokens_here = int(next(s["tokens"] for s in manifest["shards"] if s["path"] == entry["shard"]))
                inside = min(len(ids_for_entry), shard_tokens_here - int(entry["start"]))
                window = read_shard(root / entry["shard"])[int(entry["start"]):int(entry["start"]) + inside]
                assert window == ids_for_entry[:inside], (entry["identity"], window)

        # The straddle has to exist in this fixture, otherwise the test is not checking the case it names. A
        # document that runs past the end of the shard it starts in is the normal case at corpus scale.
        straddling = [e for e in entries
                      if int(e["start"]) + int(e["tokens"]) >
                      int(next(s["tokens"] for s in manifest["shards"] if s["path"] == e["shard"]))]
        assert straddling, [e["identity"] for e in entries]

        # Five ways the index can disagree with the manifest, each refused rather than reinterpreted. Each case
        # breaks exactly one check the loader makes, and the message it produces names that check, so a failing
        # index says which of its claims is the wrong one. Every case keeps the entry count the manifest states,
        # because otherwise the count check fires first and the case being checked is never reached.
        path = root / documents.INDEX_FILE
        original = path.read_text(encoding="utf-8")
        parsed = [json.loads(line) for line in original.splitlines()]
        first_val = next(e for e in parsed if e["split"] == "val")

        def break_case(entry, **changes):
            copy = [dict(e) for e in parsed]
            copy[parsed.index(entry)] = dict(entry, **changes)
            return copy

        cases = (
            # A position that is not where the entry's own logical offset says it is.
            (break_case(first_val, start=int(first_val["start"]) + 1), "shard and start put it at"),
            # The order the bytes hold the documents in.
            (parsed[::-1], "not contiguous"),
            # The same document named twice, at two different positions.
            (break_case(parsed[-1], identity=parsed[0]["identity"]), "names"),
            # A shard the manifest does not list in this split.
            (break_case(first_val, shard="train-000000.bin"), "does not list in the val split"),
            # A start past the end of the shard the entry names.
            (break_case(first_val, start=500), "inside a shard of"),
        )
        for broken, expected_message in cases:
            path.write_text("\n".join(json.dumps(e, sort_keys=True, separators=(",", ":")) for e in broken) + "\n",
                            encoding="utf-8")
            try:
                documents.read_document_index(root, manifest)
            except ValueError as e:
                assert expected_message in str(e), (expected_message, str(e))
            else:
                raise AssertionError(f"the index must refuse the case: {expected_message}")
        path.write_text(original, encoding="utf-8")
        assert documents.read_document_index(root, manifest) == entries, "the index must read after the checks"

        # A corpus built without an index has shards and no document boundaries, so a held-out loss on it is a
        # number over bytes with no name. That is what `--no-index` means, and it is reported as such.
        silent = Path(tmp) / "no_index"
        documents.build_corpus(silent, stream, encode, seed=seed, val_fraction=val_fraction, vocab_size=200,
                               special_ids=ids, eos_id=eos, shard_tokens=12, write_index=False)
        try:
            documents.read_document_index(silent)
        except FileNotFoundError as e:
            assert "cannot be traced" in str(e), e
        else:
            raise AssertionError("a corpus without an index cannot attribute a window")

        index = documents.DocumentIndex.load(root)
        val_entries = index.entries_for_split("val")
        assert [e["identity"] for e in val_entries] == [documents.document_identity(*p) for p in val_rows[:3]], (
            "the lookup must be ordered by the position the bytes hold them in"
        )
        try:
            index.entries_for_split("test")
        except ValueError as e:
            assert "unknown split" in str(e), e
        else:
            raise AssertionError("an unknown split must be refused")

        # Where each val shard begins in the split's stream, computed here from the manifest rather than from
        # the reader, so the attribution below is checked against a position the test derived itself.
        base, cursor = {}, 0
        for shard in [s for s in manifest["shards"] if s["split"] == "val"]:
            base[shard["path"]] = cursor
            cursor += int(shard["tokens"])
        assert cursor == sum(int(s["tokens"]) for s in manifest["shards"] if s["split"] == "val")

        corpus = TokenizedCorpus.load(root)
        try:
            split = corpus.split("val", seq_len=8)
            attributed, crossing, inside_a_document = 0, [], []
            for window in range(len(split)):
                shard, start = split.source_window(window)
                document, crosses = index.window_document("val", shard, start, 8)
                assert document is not None, (window, shard, start)
                position = base[shard] + int(start)
                offset = position - int(document["logical"])
                window_tokens = read_shard(root / shard)[int(start):int(start) + 8]
                inside_a_document.append(offset > 0)

                if not crosses:
                    # The whole window is one document, so the bytes have to read as that document's ids at the
                    # offset the window starts at. This is the check that makes the attribution evidence rather
                    # than a label.
                    assert window_tokens == expected[document["identity"]][offset:offset + 8], (window, document)
                    attributed += 1
                    continue

                last = index.document_at("val", shard, position + 8 - 1)
                assert last is not None and last["identity"] != document["identity"], (window, document, last)
                crossing.append(window)
                # Two documents in a shard are adjacent, so the window is the tail of the first plus the head of
                # the second, read from the same bytes. A flag that says "crossing" without saying which two is
                # the half of provenance that a per-document error bar needs.
                tail = expected[document["identity"]][offset:]
                assert window_tokens == tail + expected[last["identity"]][:8 - len(tail)], (window, document, last)

            # Both states have to occur in this fixture. A fixed-length window can start inside a document, and
            # `document_at` has to answer with the document containing the position rather than only with the one
            # starting at it; a window that starts at a boundary and runs past the end of its document is the
            # crossing case. Neither is checked if the fixture happens not to contain it.
            assert attributed and crossing, (attributed, crossing)
            assert any(inside_a_document), "a window can start inside a document"

            # A position past the last document returns None rather than raising, because a run reporting
            # provenance has to be able to say "unattributed" instead of crashing on it.
            last_shard = [s for s in manifest["shards"] if s["split"] == "val"][-1]
            assert index.document_at("val", last_shard["path"], int(last_shard["tokens"])) is None
            assert index.document_at("val", "train-000000.bin", 0) is None
            assert index.document_at("val", "not-a-shard.bin", 0) is None
        finally:
            corpus.close()


def test_training_changes_loss_and_arms_differ():
    """
    The precondition for a gate being decidable: a mechanism changes loss at all.

    The only test here that trains. Short on purpose, and it checks the harness rather than the threshold.
    """
    spec = RunSpec(steps=20)
    arms = build_arms(TINY(), spec)["G0.2"]
    corpus = make_corpus(spec.seed, length=spec.seq_len, n_train=spec.n_train, n_test=spec.n_test,
                         families=spec.families)

    losses = []
    for arm in arms:
        r = train_arm(arm, corpus, DEV, log_every=1)
        first, last = r["history"][0][2], r["history"][-1][2]
        assert torch.isfinite(torch.tensor(first)), "untrained loss must be finite"
        assert last < first, f"{arm.name}: loss did not move ({first} -> {last})"
        losses.append(last)

    assert abs(losses[0] - losses[1]) > 1e-6, "the arms produced identical numbers, so nothing was tested"

    # Depth must change the numbers, otherwise the loop is not observable and every depth gate is vacuous.
    model = HeronLM(TINY()).eval()
    sample = corpus["test"][0]
    tokens = torch.tensor([list(sample.tokens)])
    ids = slot_ids(tokens, model.cfg)
    a = model.forward(tokens, 1, None, None, ids)
    b = model.forward(tokens, 3, None, None, ids)
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-6), "loop count does not change the output"


def test_cache_eviction_happens_after_the_mask():
    """
    Regression test for a real bug in this reference.

    Trimming to the last `window` entries inside append evicts entries that earlier rows of the same
    parallel call still need. Those rows become fully masked, and softmax over a fully masked row is nan,
    not an error - so a converter that evicts early would diverge from the reference silently.
    """
    cfg = TINY()
    c = Cache(cfg.kv_heads, cfg.head_dim, cfg.swa_window)
    zero = torch.zeros(1, cfg.kv_heads, 48, cfg.head_dim)
    pos = torch.arange(48)

    c.append(zero, zero, pos)
    assert c.len == 48, "entries the current call wrote cannot be dropped before that call is computed"
    c.evict(newest=47, call_start=0)
    assert c.len == 48, "the earliest row still needs the entry it wrote"

    # A later call that does not own those entries: the entries out of reach are now safely dropped, and
    # the steady state is the window, which is the memory argument.
    c.append(zero[:, :, :1], zero[:, :, :1], torch.tensor([48]))
    c.evict(newest=48, call_start=48)
    assert c.len == cfg.swa_window, c.len
    assert int(c.positions[0]) == 48 - cfg.swa_window + 1, c.positions.tolist()

    # And the state the guard exists for: a call whose rows have no reachable entry must fail loudly
    # rather than produce nan.
    attn = Attention(cfg, "swa")
    stale = Cache(cfg.kv_heads, cfg.head_dim, cfg.swa_window)
    stale.k = torch.zeros(1, cfg.kv_heads, 32, cfg.head_dim)
    stale.v = torch.zeros_like(stale.k)
    stale.positions = torch.arange(16, 48)
    try:
        attn.forward(torch.ones(1, 1, cfg.d_model), stale, torch.tensor([3]), 0, False)
    except ValueError as e:
        assert "fully masked" in str(e), e
    else:
        raise AssertionError("a fully masked row must raise, not produce nan")


def test_prefill_longer_than_the_window_stays_finite():
    """
    The end-to-end form of the same bug: a 48-token prefill through a window of 32 at loop count 3.

    Every loop pass re-reads the segment, so the entries a later pass needs must survive the first pass.
    """
    cfg = TINY()
    assert cfg.swa_window < 48, "this test only means something when the segment exceeds the window"
    torch.manual_seed(0)
    model = HeronLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 48))
    logits = model.forward(tokens, 3, None, None, slot_ids(tokens, cfg))
    assert bool(torch.isfinite(logits).all()), "loop over a segment longer than the window produced nan"

    # At depth 1 there is no approximation to price: both policies read the same entries, so they must agree
    # exactly, inside the window and beyond it. That is the boundary condition for gate G0.5 - the shared cache
    # is exact at k=1 by definition, so the gate's number has to come from a depth where passes exist.
    m1 = HeronLM(cfg).eval()
    m2 = HeronLM(replace(cfg, cache_policy="per_loop")).eval()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        p2.data = p1.data.clone()

    a1 = m1.forward(tokens, 1, None, None, slot_ids(tokens, cfg))
    b1 = m2.forward(tokens, 1, None, None, slot_ids(tokens, cfg))
    assert torch.allclose(a1, b1, atol=1e-6), "at k=1 both policies read the same cache, so they must be identical"

    # At depth they must differ, and the cause has to be named rather than assumed: the per-loop policy keeps
    # every pass's K/V, the shared policy keeps only what pass 0 wrote. That difference exists inside the
    # window too, so a loss delta measured inside the window cannot be attributed to eviction - nothing is
    # evicted there. Keeping the two causes apart is what makes the gate number attributable.
    inside = tokens[:, :cfg.swa_window]
    a2 = m1.forward(inside, 3, None, None, slot_ids(inside, cfg))
    b2 = m2.forward(inside, 3, None, None, slot_ids(inside, cfg))
    assert not torch.allclose(a2, b2, atol=1e-6), "the shared cache must differ from the exact one at depth"

    # Beyond the window the second cause appears: entries from earlier positions are dropped, so the shared
    # cache is no longer exact for a query near the end of a long segment.
    a3 = m1.forward(tokens, 3, None, None, slot_ids(tokens, cfg))
    b3 = m2.forward(tokens, 3, None, None, slot_ids(tokens, cfg))
    assert not torch.allclose(a3, b3, atol=1e-6), "the shared cache must differ from the exact one beyond the window"


def test_fake_quant_is_stable_against_the_stored_scale():
    """
    The invariant the engine can actually reproduce.

    Re-rounding the dequantized values against the stored scale reproduces the same codes. Re-quantizing
    from scratch does not, because the recomputed max-abs is a float product rather than the stored scale.
    A test that asserted the naive form would pass in the harness and fail against a real GGUF file.
    """
    torch.manual_seed(11)
    for trial in range(200):
        w = torch.randn(64, 128) * (10.0 ** (trial % 7))
        assert requantize_is_stable(w, DENSE_INT4), f"unstable at trial {trial}"

    # Table rows are initialized to zero, so the all-zero block path is hit on the first forward pass. A
    # guard that silently used scale 1 would give the same dequantized value while disagreeing with the
    # file about what the scale is.
    row = torch.zeros(4, 64)
    row[0] = torch.randn(64)
    codes, scales = quantize(row, TABLE_INT8)
    assert bool((scales[1:] == 0).all()), "an all-zero block stores scale 0.0, which is what ggml does"
    assert requantize_is_stable(row, TABLE_INT8)

    codes, _ = quantize(torch.randn(32, 128), DENSE_INT4)
    assert bool((codes >= DENSE_INT4.code_min).all() and (codes <= DENSE_INT4.code_max).all())

    # The simulated int4 is 5.0 effective bits, not the roadmap's 4.9, because Q4_K superblock overhead is
    # not simulated. The gap has to be visible rather than hidden behind a scheme name.
    assert abs(DENSE_INT4.effective_bits() - 5.0) < 1e-9, DENSE_INT4.effective_bits()
    assert abs(TABLE_INT8.effective_bits() - 8.0) < 1e-9, TABLE_INT8.effective_bits()


def test_fake_quant_block_divisibility_is_checked():
    """
    ggml requires exact block multiples. The roadmap's expert_ffn_width 20345 is not a multiple of 32, so
    the last block would be partial and the effective bits would change. That has to be an error here
    rather than a silent partial block.
    """
    try:
        quantize(torch.randn(4, 20345), DENSE_INT4)
    except ValueError as e:
        assert "does not divide" in str(e), e
    else:
        raise AssertionError("a width that is not a multiple of the block must not be quantized silently")


def test_excluded_parameters_stay_unquantized():
    """
    Each exclusion has a reason, and the exclusions are the parameters whose values define semantics.

    Quantizing the read mixture would change which branch a block reads, which is the mechanism under test
    rather than a rounding detail.
    """
    torch.manual_seed(3)
    model = HeronLM(TINY()).eval()
    # The reference initializes table rows to zero, and an all-zero row quantizes to all-zero codes, so
    # "the parameter changed" cannot be checked on the initialization as it stands. Filling the rows is the
    # only way to test that the table path is actually quantized rather than passed through.
    model.memory.table.weight.data = torch.randn_like(model.memory.table.weight.data)
    q = fake_quantize_model(model)

    table_params, quantized_count, excluded_count = [], 0, 0
    for name, p in model.named_parameters():
        scheme = scheme_for(name, DENSE_INT4, TABLE_INT8, Q8_0)
        qp = dict(q.named_parameters())[name]
        if scheme is None:
            assert any(e in name for e in EXCLUSION_REASONS), f"{name} is excluded without a reason"
            assert torch.equal(p.data, qp.data), f"{name} was excluded but still changed"
            excluded_count += p.numel()
        else:
            assert not torch.equal(p.data, qp.data), f"{name} was supposed to be quantized"
            assert p.shape == qp.shape, (name, p.shape, qp.shape)
            quantized_count += p.numel()
            if "memory.table" in name:
                table_params.append(p)

    assert quantized_count and excluded_count, "nothing was quantized, so the exclusions are untested"

    # The tied embedding is q8_0 by rule, never Q4, so its codes must reach past the 4-bit range.
    codes, _ = quantize(dict(model.named_parameters())["embed.weight"], Q8_0)
    assert int(codes.abs().max()) > 8, "the embedding is not quantized at 8 bits"

    # The optimizer rule the harness has to honor: table rows are not decayed. Identity, not equality,
    # because the groups hold the same parameter objects the model does, and comparing tensors with `in`
    # would compare values rather than membership.
    groups = param_groups(model, 3e-3, 0.01)
    assert groups[1]["weight_decay"] == 0.0 and groups[0]["weight_decay"] == 0.01
    decayed, nodecay = groups[0]["params"], groups[1]["params"]
    assert all(any(x is p for x in nodecay) for p in table_params), "table rows must not be decayed"
    assert not any(any(x is p for x in decayed) for p in table_params)
    assert table_params, "the table parameter was not found, so the rule is untested"


def test_normalization_order_is_only_a_difference_on_the_normalized_arm():
    """
    The parity risk, stated as a measurement rather than as a preference.

    On the nGPT arm the two orders give different stored weights, because the block scales are computed
    from magnitudes that normalization removes. On the plain arm they are the same operation, so the
    measured gap is zero by construction - worth asserting, because otherwise a zero gap on that arm would
    look like a result rather than a property of the arm.
    """
    cfg = TINY()
    plain = replace(cfg, norm_mode="layernorm")
    w = torch.randn(32, 128)
    a = quantized_stored_weight(w, plain, DENSE_INT4, PARITY_RULE)
    b = quantized_stored_weight(w, plain, DENSE_INT4, "before_dequant")
    assert torch.equal(a, b), "the plain arm has no normalization rule, so the orders must agree"

    ngpt = quantized_stored_weight(w, cfg, DENSE_INT4, PARITY_RULE)
    ngpt_before = quantized_stored_weight(w, cfg, DENSE_INT4, "before_dequant")
    assert not torch.equal(ngpt, ngpt_before), "the normalized arm must distinguish the two orders"

    # Normalize after dequant is the rule recorded in metadata as general.arch.heron.weight_norm_order,
    # and the metadata module has to agree with it.
    assert PARITY_RULE == "after_dequant"
    # The QAT rule has two halves, and gates.py keeps both as proposals: the path rule has no number, and the
    # adoption number is a proposal rather than a ROADMAP threshold. quant.py reads the number from gates.py
    # instead of restating it, so the ROADMAP check has one place to look.
    qat = gates.gate("QAT")
    assert len(qat.thresholds) == 2
    path_rule, adoption = qat.thresholds
    assert path_rule.limit is None and path_rule.comparison == "direction" and path_rule.proposal
    assert adoption.proposal and adoption.limit is not None
    from heronlm.metadata import metadata, placeholder_vocabulary

    # metadata needs the vocabulary it will be written for, including its special tokens, because the ids
    # are resolved by name rather than assumed. This check is about the normalization rule, so a
    # placeholder tokenizer is enough - but it has to be a well-formed one.
    tokens = placeholder_vocabulary(cfg.vocab_size)
    assert metadata(cfg, tokens, ["merge a b"])["general.arch.heron.weight_norm_order"] == PARITY_RULE


def test_paths_are_the_decisions_quantization_cannot_be_allowed_to_move():
    """
    The half of the QAT decision a loss delta cannot answer.

    A model can lose 0.01 nats and still route every conversation to the same expert, or lose nothing and
    start firing a different slot. Paths are decisions rather than rounding on values, so they have to be
    counted separately - and the harness has to say which of them it can actually count.
    """
    torch.manual_seed(5)
    cfg = TINY()
    model = HeronLM(cfg).eval()
    sample = make_corpus(7, length=24, n_train=4, n_test=2)["test"][0]

    # Untrained: the table rows are zero and the gate key is zero, so every gate is exactly 0.5 and no
    # rounding can move it. Reporting "no change" there would look like a result for ROADMAP line 157 when it
    # is a property of an untrained table, so the harness reports it as undecidable instead.
    r = path_stability(model, [sample], 1, DEV)
    assert r["paths"]["memory_gate"]["undecidable"], r["paths"]["memory_gate"]
    assert r["ptq_does_not_change_paths"] is None, "an untrained memory path cannot make the rule decidable"

    # Give the memory path something to decide, and the measurement becomes decidable. The rows have to be
    # written through .data because the module registers them as parameters.
    model.memory.table.weight.data = F.normalize(torch.randn(cfg.ngram_memory_entries, cfg.ngram_memory_dim), dim=1)
    model.memory.gate_key.data = F.normalize(torch.randn(cfg.ngram_memory_dim), dim=0)
    r2 = path_stability(model, [sample], 1, DEV)
    assert not r2["paths"]["memory_gate"]["undecidable"], r2["paths"]["memory_gate"]
    assert isinstance(r2["ptq_does_not_change_paths"], bool), r2

    # The router half is always decidable: the prototypes are data, and the pooled representation reaches the
    # router through quantized weights, so a path change there is a real risk rather than a degenerate case.
    assert not r2["paths"]["router_slot"].get("undecidable")
    adoption = gates.gate("QAT").thresholds[1]
    assert ptq_delta(model, [sample], 1, DEV)["qat_proposal"]["adopt_if_delta_above"] == adoption.limit

    # Why the read mixture is excluded, shown rather than asserted: rounding preserves order except at a tie,
    # and a tie moves the winner. Two values that round to the same code come back equal, so the argmax lands
    # on the first index rather than on the larger original value. Quantizing a read mixture can therefore
    # change which branch a block reads, which is the mechanism gate G0.3 tests rather than a rounding detail.
    row = torch.tensor([[0.019, 0.02]])
    assert int(row[0].argmax()) == 1
    d = dequantized(row, DENSE_INT4)
    assert bool(torch.equal(d[0][0], d[0][1])), "the two values must land on the same code, which is the tie"
    assert int(d[0].argmax()) == 0
    assert "stream.read_weights" in EXCLUSION_REASONS
    assert r2["paths"]["gr_read_branch"]["cannot_change_by_construction"]


def test_verdicts_cannot_be_promoted_by_a_small_run():
    """
    `pending` and `indicative` are real states, not hedges.

    A tiny run can show that a mechanism changes loss at all; it cannot decide a threshold stated at the
    gate scale, so a tiny run must not be able to produce the word `pass`. The same restriction runs the
    other way: it must not produce `fail` either, because the roadmap treats a fail as a design change. A
    tiny reading that disagrees is an indication, not a decision.
    """
    v = gates.verdict("G0.1", {"depth_gain": True, "degradation_beyond_trained": 0.01}, "tiny", False)
    assert v["status"] == "indicative", v["status"]
    assert any(c["status"] == "pass" and c["limit"] is None for c in v["checks"]), "the direction must report"

    # The disagreeing direction at tiny scale: the check reports what it saw, the gate does not decide.
    v_neg = gates.verdict("G0.1", {"depth_gain": False, "degradation_beyond_trained": 0.03}, "tiny", False)
    assert v_neg["status"] == "indicative", v_neg["status"]
    assert any(c["status"] == "fail" for c in v_neg["checks"]), "the disagreement must still be visible"
    assert not any(c["decides_at_any_scale"] for c in v_neg["checks"]), "these are measurements, not rules"

    # A rule check is different: it is decided at any size, so a failure there is a real failure.
    v_rule = gates.verdict("G0.3", {"reserved_branch_written_by_one_block": False, "loss_cost_of_removing": 0.03},
                           "tiny", False)
    assert v_rule["status"] == "fail", v_rule["status"]

    # Same numbers, at the scale the threshold is stated for: now it is a decision.
    v2 = gates.verdict("G0.1", {"depth_gain": True, "degradation_beyond_trained": 0.01}, "phase0", True)
    assert v2["status"] == "pass", v2["status"]
    v3 = gates.verdict("G0.1", {"depth_gain": True, "degradation_beyond_trained": 0.03}, "phase0", True)
    assert v3["status"] == "fail", v3["status"]

    # Nothing measured, nothing decided.
    assert gates.verdict("G0.6", {"fixed_arm_degrades_beyond_trained_depth": True}, "tiny", False)["status"] == "pending"
    assert gates.verdict("G0.4", {"device_time": True}, "tiny", False)["status"] == "not decidable by this harness"

    # A proposal must be labelled as one, so a number cannot become a threshold by appearing in a report.
    for g in gates.GATES.values():
        for t in g.thresholds:
            if t.proposal:
                assert "proposal" in t.source, t.source
            elif t.limit is not None:
                assert "ROADMAP" in t.source, t.source


def test_a_number_cannot_be_reported_at_a_depth_the_config_cannot_run():
    """
    ROADMAP states the cache gate at `k = 4`. On the tiny config max_loop is 3, so k = 4 cannot be evaluated
    at all: the read weights are sized [blocks, max_loop, branches] and a deeper pass indexes past them. The
    harness has to report that number as not measured rather than as a k = 3 reading under a k = 4 name.

    The distinction is what the test is checking: the mechanism is observable at the depth the config can
    reach, and the gate's number is not. Both states have to appear, and the verdict stays pending.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("phase0_driver", ROOT / "scripts" / "phase0.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    def entries(arm, offset: float) -> dict:
        # The losses are stand-ins; what is being tested is how the driver names them, not any measurement.
        return {"arm": arm, "by_k": {k: {"k": k, "loss": 1.0 - 0.01 * k + offset,
                                         "by_family": {"lookup": 1.0 - 0.01 * k + offset}}
                                     for k in arm.spec.eval_k}}

    tiny = TINY()
    tiny_arms = build_arms(tiny, RunSpec())["G0.5"]
    assert tiny.max_loop < 4, "this test only means something on a config that cannot reach the stated depth"
    m_tiny = driver.measured_for("G0.5", [entries(tiny_arms[0], 0.0), entries(tiny_arms[1], -0.02)], "lookup")
    assert m_tiny["shared_cache_loss_at_k4"] is None, m_tiny
    assert m_tiny["deepest_evaluated_k"] == tiny.max_loop, m_tiny
    assert m_tiny["arms_differ_at_deepest_k"], "the difference is observable, it is just not the measurement asked for"
    assert gates.verdict("G0.5", m_tiny, "tiny", False)["status"] == "pending", gates.verdict("G0.5", m_tiny, "tiny", False)

    # The same call at the scale the threshold is stated for produces the measurement, not a gap.
    phase0_arms = build_arms(PHASE0(), RunSpec())["G0.5"]
    m_phase0 = driver.measured_for("G0.5", [entries(phase0_arms[0], 0.0), entries(phase0_arms[1], 0.02)], "lookup")
    assert m_phase0["shared_cache_loss_at_k4"] is not None, m_phase0
    assert gates.verdict("G0.5", m_phase0, "phase0", True)["status"] == "pass", gates.verdict("G0.5", m_phase0, "phase0", True)


def test_gate_numbers_exist_in_the_roadmap():
    """
    Every sourced number has to appear in ROADMAP.md. A number that does not appear has to be marked as a
    proposal. This is what stops a threshold from being invented by the report and then cited as a gate.
    """
    text = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    stated, proposed, unstated = 0, 0, 0
    for g in gates.GATES.values():
        for t in g.thresholds:
            if t.limit is None:
                unstated += 1
                assert t.comparison in ("direction", "pattern"), t
                continue
            value = f"{t.limit:g}"
            # ROADMAP writes some thresholds as percentages. Checking both spellings is what keeps the test
            # honest rather than turning a formatting difference into a false failure.
            found = value in text or f"{t.limit * 100:g}%" in text
            if t.proposal:
                proposed += 1
            else:
                stated += 1
                assert found, f"{g.id}/{t.metric}: {value} is cited as ROADMAP but is not in ROADMAP.md"
    assert stated >= 6, f"only {stated} sourced thresholds"
    assert proposed >= 2, f"only {proposed} proposals; the QAT rule and the G0.3 loss cost must be marked"
    assert unstated >= 4, f"only {unstated} unstated thresholds; G0.1's margin and G0.6 must not have numbers"


def test_checkpoint_rotation_keeps_a_bounded_number_of_records():
    """
    A rotation policy is what a multi-day run actually needs; the test checks what survives on disk.

    At the token budget ROADMAP line 95 states, an arm with a save every 500 steps writes around two thousand
    records of about 3.4 GB each. That is not a policy, it is an accident, and on a shared box the storage
    quota ends the run before the token budget does. With keep = N the newest N records exist, so the run can
    be resumed from any of them and the cost is bounded by N.

    Two things are checked rather than assumed. The current record keeps the name `--resume` reads - a policy
    that renamed it would change what a resumed run finds - and an archive named for a step is the record
    written at that step, which is checked by loading it rather than by trusting its name.
    """
    families = ("lookup", "recurrence")
    spec = RunSpec(steps=5, batch=2, n_test=8, families=families)
    arm = build_arms(TINY(), spec)["G0.2"][0]
    corpus = make_corpus(spec.seed, length=spec.seq_len, n_train=0, n_test=spec.n_test, families=families)
    exclude = [s.tokens for s in corpus["test"]]

    def stream() -> SampleStream:
        return SampleStream(spec.seed, length=spec.seq_len, families=families, split="train", exclude=exclude)

    with tempfile.TemporaryDirectory() as tmp:
        # Five steps, one record per step, keep = 3: the current record plus the two newest archives.
        path = Path(tmp) / "keep3.pt"
        result = train_arm(arm, corpus, DEV, train_stream=stream(), checkpoint_path=path,
                           checkpoint_every=1, checkpoint_keep=3)
        files = sorted(p.name for p in path.parent.glob("keep3*"))
        assert len(files) == 3, files
        assert path.name in files, f"the current record must keep the name --resume reads: {files}"
        assert result["checkpoints_kept"] == files, (result["checkpoints_kept"], files)
        assert result["checkpoint_keep"] == 3
        archived = sorted(int(n[len(path.stem) + 1: n.rindex(".")]) for n in files if n != path.name)
        assert archived == [3, 4], archived

        for name in files:
            payload = torch.load(path.parent / name, map_location="cpu", weights_only=False)
            if name == path.name:
                assert payload["step"] == spec.steps, (name, payload["step"])
            else:
                # The name is the position the record holds, not a counter: the archive is named by the step of
                # the record it replaced, so a run resumed from it is where its name claims.
                named = int(name[len(path.stem) + 1: name.rindex(".")])
                assert payload["step"] == named, (name, payload["step"])

        # A resumed run continues in the same directory and must still leave three records, with the current
        # one at the same name. The pruned archives are gone, which is the whole point of a bounded policy.
        longer = replace(arm, spec=replace(arm.spec, steps=6))
        r6 = train_arm(longer, corpus, DEV, train_stream=stream(), checkpoint_path=path,
                       checkpoint_every=1, checkpoint_keep=3, resume=True)
        files = sorted(p.name for p in path.parent.glob("keep3*"))
        assert len(files) == 3, files
        assert path.name in files, files
        # The record reports what it holds, including the steps it inherited from the record it resumed from.
        assert r6["steps_recorded"] == 6, r6["steps_recorded"]

        # keep = 0 is what the short tests need because they read a record back: every record the run wrote
        # survives. Five records for five steps, not five archives, because the first write created the current
        # file rather than archiving anything.
        all_path = Path(tmp) / "keepall.pt"
        r_all = train_arm(arm, corpus, DEV, train_stream=stream(), checkpoint_path=all_path,
                          checkpoint_every=1, checkpoint_keep=0)
        files_all = sorted(p.name for p in all_path.parent.glob("keepall*"))
        assert len(files_all) == 5, files_all
        assert r_all["checkpoint_keep"] == 0, r_all["checkpoint_keep"]

        # The archive name is built from the stem, so a stem that already contains a dot cannot be ordered by
        # step. Refused rather than written, because a directory of records the pruner cannot read is a run
        # that silently keeps everything it was told to delete.
        dotted = Path(tmp) / "v1.2.pt"
        try:
            train_arm(arm, corpus, DEV, train_stream=stream(), checkpoint_path=dotted,
                      checkpoint_every=1, checkpoint_keep=3)
        except ValueError as e:
            assert "contains a dot" in str(e), str(e)
        else:
            raise AssertionError("a checkpoint name whose stem contains a dot must be refused")

        # The name a gate run actually writes. ROADMAP spells the gate ids with a dot and the arm labels use =
        # and @, so the name has to be sanitized - and sanitized in a way the pruner can still read, because the
        # archive name is built from the stem. Checked through the path run_gate uses rather than through the
        # sanitizer, so the refusal above cannot be handed a name the driver would never produce.
        gate_dir = Path(tmp) / "gate_names"
        small = RunSpec(steps=2, batch=1, n_test=4, families=families)
        r_gate = run_gate("G0.1", build_arms(TINY(), small)["G0.1"], DEV,
                          checkpoint_dir=gate_dir, checkpoint_every=1, checkpoint_keep=2)
        for entry in r_gate["arms"]:
            used = Path(entry["result"]["checkpoint"])
            assert "." not in used.stem, used.name
            assert used.stem.startswith("G0_1_"), used.name
            kept = entry["result"]["checkpoints_kept"]
            assert kept and used.name in kept, kept
            for name in kept:
                if name != used.name:
                    # The step is read back out of the archive name, which is what makes the directory orderable.
                    assert int(name[len(used.stem) + 1: name.rindex(".")]) > 0, name


def test_recorded_orders_survive_the_compact_form_and_refuse_a_truncated_record():
    """
    The compact record carries the same facts as the list form, and its hash is what makes them checkable.

    At the gate budget an arm records about a million steps. As a list of lists of Python ints that is tens of
    millions of objects, and the record costs more than the weights do. As fixed-width arrays the same order
    costs 4 bytes per token position and 1 byte per step, with the row boundaries reconstructed from the batch
    size, which is part of the run key. Nothing is lost, so the only way to know that a resumed record is the
    record that was written is the hash over its canonical bytes.

    The refusal is checked in the resume path, not only in the helper: a truncated record is only dangerous
    because a run is allowed to continue on it.
    """
    # The shape first, without training. A partial row is not a step, and a record whose k count disagrees with
    # its row count is a different order even when its bytes hash correctly.
    data_order = [[1, 2], [3, 4], [5, 6]]
    k_order = [1, 3, 2]
    orders = order_arrays(data_order, k_order, batch=2)
    assert orders_from_arrays(orders, 2) == (data_order, k_order), "the compact form must round trip exactly"
    assert orders_sha256(orders["data"], orders["k"]) == orders["sha256"]

    try:
        orders_from_arrays({"data": orders["data"][:-1], "k": orders["k"], "batch": 2,
                            "sha256": orders["sha256"]}, 2)
    except ValueError as e:
        assert "not a multiple of batch" in str(e), str(e)
    else:
        raise AssertionError("a partial row must be refused")

    # A record that lost rows is refused on its hash rather than resumed with a shorter order: the data order
    # is the reason a between-arm loss difference means anything, so a run cannot continue under an order it
    # cannot verify.
    truncated = {"data": orders["data"][:4], "k": orders["k"][:2], "batch": 2, "sha256": orders["sha256"]}
    try:
        orders_from_arrays(truncated, 2)
    except ValueError as e:
        assert "does not match its own hash" in str(e), str(e)
    else:
        raise AssertionError("a truncated order record must be refused")

    # The case the hash cannot catch on its own: bytes that hash correctly but whose k count disagrees with the
    # number of rows. Both checks have to exist, because a record can be internally consistent and still wrong.
    short_k = orders["k"][:1]
    consistent = {"data": orders["data"][:4], "k": short_k, "batch": 2,
                  "sha256": orders_sha256(orders["data"][:4], short_k)}
    try:
        orders_from_arrays(consistent, 2)
    except ValueError as e:
        assert "expected" in str(e), str(e)
    else:
        raise AssertionError("a record whose k count disagrees with its row count must be refused")

    families = ("lookup", "recurrence")
    spec = RunSpec(steps=3, batch=2, n_test=8, families=families)
    arm = build_arms(TINY(), spec)["G0.6"][1]
    corpus = make_corpus(spec.seed, length=spec.seq_len, n_train=0, n_test=spec.n_test, families=families)
    exclude = [s.tokens for s in corpus["test"]]

    def stream() -> SampleStream:
        return SampleStream(spec.seed, length=spec.seq_len, families=families, split="train", exclude=exclude)

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.pt"
        part = train_arm(arm, corpus, DEV, train_stream=stream(), checkpoint_path=good)
        payload = torch.load(good, map_location="cpu", weights_only=False)
        assert payload["format"] == 2, payload["format"]

        # What the checkpoint holds must be what the run reported, and the hash the run reported must be the
        # hash of those same bytes. Otherwise the record describes an order the run did not use.
        read_data, read_k, read_hash = orders_from_payload(payload)
        assert (read_data, read_k) == (part["data_order"], part["k_order"]), (read_data, part["data_order"])
        assert read_hash == part["orders_sha256"], (read_hash, part["orders_sha256"])

        # Format 1 stored the orders as Python lists and is still read, so a run interrupted before the record
        # form changed stays resumable. The hash is recomputed from the bytes either way, so both forms state
        # the same fact about the same order.
        legacy = dict(payload)
        legacy.pop("orders")
        legacy["format"] = 1
        # A format-1 record stored the orders as lists rather than as arrays, so the keys are different. This is
        # the checkpoint form an interrupted run would have left behind, and it has to stay readable.
        legacy["data_order"] = part["data_order"]
        legacy["k_order"] = part["k_order"]
        legacy_path = Path(tmp) / "legacy.pt"
        torch.save(legacy, legacy_path)
        resumed = train_arm(arm, corpus, DEV, train_stream=stream(), checkpoint_path=legacy_path, resume=True)
        assert resumed["resumed_from"] == 3, resumed["resumed_from"]
        assert resumed["data_order"][:3] == read_data, (resumed["data_order"][:3], read_data)
        # The record reports what it holds, which on a resumed run includes the steps it inherited.
        assert resumed["steps_recorded"] == 3, resumed["steps_recorded"]

        try:
            bad_form = dict(payload)
            bad_form["format"] = 3
            torch.save(bad_form, Path(tmp) / "unknown_form.pt")
            train_arm(arm, corpus, DEV, train_stream=stream(),
                      checkpoint_path=Path(tmp) / "unknown_form.pt", resume=True)
        except ValueError as e:
            assert "unknown checkpoint format" in str(e), str(e)
        else:
            raise AssertionError("a checkpoint in an unknown record form must be refused")

        # A truncated record refused in the resume path itself.
        damaged = dict(payload)
        damaged["orders"] = {"data": payload["orders"]["data"][:4], "k": payload["orders"]["k"][:2],
                             "batch": 2, "sha256": payload["orders"]["sha256"]}
        damaged_path = Path(tmp) / "damaged.pt"
        torch.save(damaged, damaged_path)
        try:
            train_arm(arm, corpus, DEV, train_stream=stream(), checkpoint_path=damaged_path, resume=True)
        except ValueError as e:
            assert "does not match" in str(e), str(e)
        else:
            raise AssertionError("a truncated order record must not be resumed")


def test_precision_is_read_from_torch_and_written_into_the_record():
    """
    The arithmetic that produced a number is a condition of the number, not an assumption about it.

    TF32 is a default the device sets rather than a choice the code makes, and a loss that came from TF32
    matmul is a different number from the same arm on CPU. The reference's parity claim is about one
    environment, so a record has to state which arithmetic produced its reading.

    The flags are restored at the end. A test that left TF32 on would silently change the numbers every later
    test in this file compares, which is exactly the kind of condition a record is supposed to make visible.
    """
    spec = RunSpec(steps=2, batch=1, n_test=4, families=("lookup",))
    arm = build_arms(TINY(), spec)["G0.2"][0]
    corpus = make_corpus(spec.seed, length=spec.seq_len, n_train=0, n_test=spec.n_test, families=spec.families)
    exclude = [s.tokens for s in corpus["test"]]

    def stream() -> SampleStream:
        return SampleStream(spec.seed, length=spec.seq_len, families=spec.families,
                            split="train", exclude=exclude)

    before = precision_flags()
    try:
        on = set_tf32(True)
        assert on["matmul_tf32"] and on["cudnn_tf32"], on
        r_on = train_arm(arm, corpus, DEV, train_stream=stream())
        assert r_on["precision"]["matmul_tf32"], r_on["precision"]
        assert r_on["precision"]["torch"] == torch.__version__, r_on["precision"]

        # The mode is written rather than implied. Nothing in the reference has train-mode behaviour today -
        # there is no dropout anywhere in the reference - so the call changes no number, but the record still
        # has to say which path the loss came from.
        assert r_on["mode"] == "train", r_on["mode"]

        off = set_tf32(False)
        assert not off["matmul_tf32"] and not off["cudnn_tf32"], off
        r_off = train_arm(arm, corpus, DEV, train_stream=stream())
        assert not r_off["precision"]["matmul_tf32"], r_off["precision"]

        # Same arm, same seed, same data order: the only thing the two records are allowed to disagree about is
        # the arithmetic, and that disagreement has to be visible in the record rather than inferred later.
        assert r_off["precision"]["matmul_tf32"] != r_on["precision"]["matmul_tf32"]
        assert r_off["orders_sha256"] == r_on["orders_sha256"], "the same run must record the same order"
    finally:
        set_tf32(before["matmul_tf32"])

    # What the record states about the run, checked against what the run returned. If the hash and the returned
    # orders disagreed, the record would be describing an order the run did not use.
    recomputed = order_arrays(r_off["data_order"], r_off["k_order"], spec.batch)
    assert r_off["orders_sha256"] == recomputed["sha256"], (r_off["orders_sha256"], recomputed["sha256"])
    assert r_off["steps_recorded"] == len(r_off["k_order"]), r_off["steps_recorded"]
    assert r_off["tokens_recorded"] == len(r_off["k_order"]) * spec.seq_len, r_off["tokens_recorded"]
    assert r_off["checkpoints_kept"] == [], r_off["checkpoints_kept"]


def test_eval_size_is_a_condition_the_harness_states_with_its_assumption_visible():
    """
    ROADMAP states the training budget and says nothing about the eval size, so the harness states its own.

    That distinction is why this is a condition and not a result: the value it uses is an assumption named as an
    assumption, and nothing here has measured a corpus. What is not an assumption is the consequence, which the
    test checks - at 32 windows the error bar is larger than the smallest numeric threshold in the table for
    any dispersion that is not far below the assumed one, so G0.1 cannot be decided at that eval size.

    The error bar is over documents, not tokens, and that is the point. A held-out window is one document: the
    losses at its positions are one observation split into seq_len pieces, not seq_len independent
    observations. The careless division is reported alongside the correct one because the difference between
    them is the mistake this condition exists to prevent.
    """
    # The spec default: 32 documents, error bar larger than the 0.02 nats G0.1 states.
    small = gates.eval_condition(32, 2048)
    assert small["status"] == "not resolvable", small["status"]
    assert not small["resolves"], small
    assert small["standard_error"] > small["smallest_numeric_limit"], small
    assert small["per_token_error"] < small["smallest_numeric_limit"] < small["standard_error"], small
    assert small["windows"] == 32, small

    # The smallest size that could resolve it, computed from the rule rather than from a reading: two standard
    # errors must fit inside the threshold.
    needed = gates.eval_condition(small["needed_windows"], 2048)
    assert needed["needed_windows"] == 900, needed["needed_windows"]
    assert needed["resolves"], needed
    assert abs(needed["standard_error"] - needed["smallest_numeric_limit"] / gates.SE_FACTOR) < 1e-12, needed

    # The size the harness states as its condition, met with margin rather than by exactly one window.
    big = gates.eval_condition(2000, 2048)
    assert big["tokens"] >= gates.GATE_EVAL_TOKENS, big["tokens"]
    assert big["matches_gate_eval_tokens"] and big["resolves"], big
    assert big["standard_error"] < small["standard_error"], (big["standard_error"], small["standard_error"])
    assert big["windows"] == 2000 and big["tokens"] == 2000 * 2048, big

    # The assumption stays visible in the record: named, and marked as not measured. A run that measures its own
    # window dispersion replaces it, and the record then states the dispersion the decision was made under.
    assert big["assumed_window_loss_std"] == gates.WINDOW_LOSS_STD, big
    assert not big["measured_from_a_corpus"], big

    # A gate with no numeric threshold has nothing to resolve. That is its own state, not a failure and not a
    # pass-by-default: G0.2 is a direction, so an eval size cannot decide it at any number of windows.
    direction_only = gates.eval_condition(2000, 2048, gate_id="G0.2")
    assert direction_only["smallest_numeric_limit"] is None, direction_only
    assert direction_only["status"] == "no numeric threshold to resolve", direction_only["status"]
    assert not direction_only["resolves"] and direction_only["needed_windows"] is None, direction_only

    # Zero windows is not an eval set, and the condition says so rather than dividing by zero.
    none = gates.eval_condition(0, 2048)
    assert none["standard_error"] == float("inf") and not none["resolves"], none


def test_the_synthetic_path_is_refused_before_training_when_it_cannot_supply_the_eval_size():
    """
    The synthetic eval set is draws from a task, and each task admits a fixed number of distinct sequences:
    lookup 48, recurrence 48 squared. Those are exact counts rather than estimates because a sequence is fully
    determined by its opening tokens, so an eval set larger than the space is not N documents - it is the same
    documents counted more than once. The consequence that stops a run is the second one: once the eval set
    covers a family's whole space there is no sequence outside it for training to draw, and the run fails at its
    first step on a family it cannot name. The driver refuses that before training, which is what this test
    checks by looking at what a trained run would have left behind.

    Two things are kept apart. `max_supply` is a bound derived from the spaces - the largest eval size guaranteed
    to leave a sequence per family - while what this run's draw actually covered is a separate fact. A run above
    the bound can still run because its draw happened to miss part of a space, and that is luck rather than a
    property of the task, so both numbers are in the record.
    """
    assert sequence_space("lookup") == 48, sequence_space("lookup")
    assert sequence_space("recurrence") == 48 * 48, sequence_space("recurrence")
    # The recall families are reported as None rather than as a number: their sequences depend on random filler
    # as well as on the fact pool, so their space is large but not a countable pool.
    assert sequence_space("quote") is None and sequence_space("needle") is None

    # The invariants that hold for any draw, checked as arithmetic rather than as a lucky value: a family's
    # windows split into distinct sequences and repeats, distinct cannot exceed the space, and what training has
    # left is exactly the space minus the distinct sequences the eval set covers.
    for n in (4, 32, 96):
        supply = eval_supply(7, 48, n, ("lookup", "recurrence"))
        assert supply["can_draw"], supply
        assert supply["windows"] == n, supply
        for family, row in supply["per_family"].items():
            assert row["distinct_sequences"] + row["repeats"] == row["windows"], (family, row)
            assert row["distinct_sequences"] <= row["space"], (family, row)
            assert row["training_sequences_left"] == row["space"] - row["distinct_sequences"], (family, row)
            # The eval set is split round-robin across families, so a per-family count is a per-family
            # measurement rather than an artifact of which family landed in which split.
            assert row["windows"] == n // len(supply["families"]), (family, row)
            assert row["can_draw"], (family, row)
        assert supply["max_supply"] == 2 * (48 - 1), supply["max_supply"]
        assert supply["distinct_sequences"] == sum(r["distinct_sequences"] for r in supply["per_family"].values())

    # A draw small enough to be all distinct supplies the document count it claims.
    small = eval_supply(7, 48, 4, ("lookup", "recurrence"))
    assert small["status"] == "can supply" and small["repeated_windows"] == 0, small

    # The spec's default eval size already contains repeats: 16 draws over a 48-sequence space are not 16
    # documents. That is reported as a condition rather than refused, because training can still draw - but the
    # denominator the error bar is computed over is not the count the run asked for.
    default = eval_supply(7, 48, 32, ("lookup", "recurrence"))
    assert default["status"] == "supplies repeats" and default["repeated_windows"] > 0, default
    assert default["distinct_sequences"] < default["windows"], default
    assert default["error_bar_denominator"] == "distinct sequences", default

    # An eval set that covers a family's whole space, by pigeonhole: the draw is the space, so the count it
    # claims is all repeats and training has nothing left. Refused, with the bound and the alternative named.
    big = eval_supply(7, 48, 900, ("lookup", "recurrence"))
    assert big["status"] == "cannot supply an eval set this large", big["status"]
    assert not big["can_draw"] and big["blocked_families"] == ["lookup"], big
    assert big["per_family"]["lookup"]["training_sequences_left"] == 0, big
    # The other family is not blocked; the refusal is about the family whose space was covered, not about the
    # eval size in the abstract.
    assert big["per_family"]["recurrence"]["can_draw"], big
    assert "94" in big["refusal"] and "--corpus-dir" in big["refusal"], big["refusal"]

    # Above the probe's own limit the answer is reported as not probed rather than as inferred. The limit is the
    # harness's, not a property of any task: the probe reproduces the eval set run_gate would build, so checking
    # a request for millions of synthetic windows would mean materializing them. What arithmetic can still say is
    # a bound, and it is named as a bound - a countable family cannot contribute more distinct sequences than its
    # space, so a set of draws larger than the space repeats at least the difference. What arithmetic cannot say
    # is whether the draw covered the whole space, because N draws from S sequences need not cover all S; that is
    # why the probe draws rather than reasoning, and why an unprobed draw is reported as unprobed.
    over = eval_supply(7, 48, PROBE_MAX_WINDOWS + 1, ("lookup", "recurrence"))
    assert over["status"] == "not probed" and not over["probed"], over
    assert over["can_draw"] is None and over["distinct_sequences"] is None, over
    for fam, space in (("lookup", 48), ("recurrence", 48 * 48)):
        row = over["per_family"][fam]
        assert row["windows"] == (PROBE_MAX_WINDOWS + 1) // 2, row
        assert row["space"] == space, row
        assert row["max_distinct"] == space, row
        assert row["minimum_repeats"] == row["windows"] - space, row
        assert row["distinct_sequences"] is None and row["repeats"] is None, row
        assert row["training_sequences_left"] is None and row["can_draw"] is None, row
    assert str(PROBE_MAX_WINDOWS) in over["refusal"] and "--corpus-dir" in over["refusal"], over["refusal"]
    # A family with no countable space contributes no bound at all, so the refusal for a run made only of recall
    # families says that plainly instead of quoting a number.
    over_recall = eval_supply(7, 48, PROBE_MAX_WINDOWS + 1, ("quote",))
    assert over_recall["max_supply"] is None and "no bound can be stated" in over_recall["refusal"], over_recall

    # Both refusals happen before training. A run that trained would have written a record and printed arm
    # numbers; these print nothing and write nothing, which is the difference between a pre-flight and a crash at
    # the first step.
    with tempfile.TemporaryDirectory() as tmp:
        for refused_args, expected in ((["--eval-windows", "900"], "cannot supply"),
                                       (["--eval-windows", str(PROBE_MAX_WINDOWS + 1)], "above the probe")):
            out, err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    driver.main(["--scale", "tiny", "--steps", "1", *refused_args, "--record-dir", tmp])
            except SystemExit as e:
                # The exit code is 2, which is argparse's refusal code, and the message is the explanation rather
                # than a traceback. The driver's own print block is not reached, so a refused run prints nothing.
                assert e.code == 2, e.code
                assert expected in err.getvalue(), err.getvalue()
            else:
                raise AssertionError(f"the driver must refuse {refused_args}")
            assert out.getvalue() == "", out.getvalue()
            assert not list(Path(tmp).glob("*")), "a refused run must not have written a record"

    # --family names which family's number the gate reads, and the run has to draw both it and the control, so
    # the arguments drive the data rather than only the label. The default order is the order RunSpec already
    # had, so a run that passes neither argument is the same run as before with the same run key.
    read, control, drawn, problems = driver.family_roles(None, None, None)
    assert (read, control, drawn, problems) == ("recurrence", "lookup", ("lookup", "recurrence"), []), problems
    assert drawn == RunSpec(steps=1).families, (drawn, RunSpec(steps=1).families)

    # A different target family changes the data, not just the print.
    read, control, drawn, problems = driver.family_roles(None, "quote", "lookup")
    assert (read, control, drawn, problems) == ("quote", "lookup", ("lookup", "quote"), []), problems

    # A control that names the same task as the family is not a control: the comparison would be between a
    # family and itself. Refused as a list rather than raised, so the driver can refuse before spending compute.
    _, _, _, problems = driver.family_roles(None, "lookup", "lookup")
    assert problems and "different task" in problems[0], problems

    # A real corpus has one family and no control task to compare against, which is reported as not applicable
    # rather than as a control that showed no effect. The defaults resolve to that; an explicit synthetic family
    # name handed to a corpus run is refused, because the run would report a number for a family it never drew.
    read, control, drawn, problems = driver.family_roles("runs/corpus", None, None)
    assert (read, control, drawn, problems) == ("text", None, ("text",), []), problems
    _, _, _, problems = driver.family_roles("runs/corpus", "needle", None)
    assert problems and "token corpus" in problems[0], problems
    _, _, _, problems = driver.family_roles("runs/corpus", None, "lookup")
    assert problems and "not applicable" in problems[0], problems


def test_the_gate_scale_condition_is_one_definition_and_names_which_part_a_run_is_short_of():
    """
    "At gate scale" has to mean one thing for every gate, and a run that is not at it has to say which part it
    is short of.

    The QAT verdict call used to read the scale label alone, which is one of the four conditions, and a
    48-token run at the gate size printed a `fail` against the 0.05-nat proposal. ROADMAP treats a fail as a
    design change, so a run that cannot resolve the threshold it is being read against must not be able to
    print one. The two definitions say the same thing at tiny scale, so the only way to tell them apart is to
    run the driver and record what each gates.verdict call was given.
    """
    short = driver.gate_scale_conditions("tiny", 48, None, gates.eval_condition(2, 48))
    assert not short["at_gate_scale"], short
    assert sorted(short["short_of"]) == sorted([
        "model_size_at_gate_scale", "token_budget_met",
        "real_text_corpus_and_the_budget_fits_inside_it",
        "eval_resolves_the_smallest_numeric_threshold"]), short

    # Three conditions satisfied and the run still cannot decide: 1B tokens over real text with 32 held-out
    # windows gives an error bar larger than the 0.02-nat threshold the number is being read against.
    near = driver.gate_scale_conditions("phase0", 1_000_000_000, 1_200_000_000, gates.eval_condition(32, 2048))
    assert near["short_of"] == ["eval_resolves_the_smallest_numeric_threshold"], near
    assert not near["at_gate_scale"], near

    # A run longer than the corpus it is handed is not at the gate scale either: it cannot train on more tokens
    # than the corpus has, so the budget would be a claim about data the run does not have.
    over = driver.gate_scale_conditions("phase0", 1_000_000_000, 500_000_000, gates.eval_condition(2000, 2048))
    assert over["short_of"] == ["real_text_corpus_and_the_budget_fits_inside_it"], over

    full = driver.gate_scale_conditions("phase0", 1_000_000_000, 1_200_000_000, gates.eval_condition(2000, 2048))
    assert full["at_gate_scale"] and full["short_of"] == [], full

    calls, original_verdict, original_conditions = [], gates.verdict, driver.gate_scale_conditions

    def recording(gate_id, measured, scale, at_gate_scale):
        calls.append((gate_id, at_gate_scale))
        return original_verdict(gate_id, measured, scale, at_gate_scale)

    gates.verdict = recording
    # Patched to True at tiny scale, where the scale label alone would still say False. A gate that reads the
    # label instead of the condition is caught here, which is what the QAT call was doing.
    patched = lambda *a, **k: {"at_gate_scale": True, "conditions": {}, "short_of": []}
    driver.gate_scale_conditions = patched
    try:
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            driver.main(["--scale", "tiny", "--steps", "1", "--batch", "1", "--eval-windows", "2",
                         "--record-dir", tmp])
        assert calls, "the driver must verdict at least one gate"
        assert all(flag for _, flag in calls), calls

        # The same run with the condition unpatched is not at the gate scale, so every call is read against
        # False and no numeric threshold can print a decision. `gates.verdict` stays patched across both runs:
        # the two runs are only comparable while the same recording wrapper sees both of them.
        calls.clear()
        driver.gate_scale_conditions = original_conditions
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            driver.main(["--scale", "tiny", "--steps", "1", "--batch", "1", "--eval-windows", "2",
                         "--record-dir", tmp])
        assert calls and all(flag is False for _, flag in calls), calls
    finally:
        gates.verdict, driver.gate_scale_conditions = original_verdict, original_conditions


if __name__ == "__main__":
    from tests import run_tests

    raise SystemExit(1 if run_tests.run(globals()) else 0)

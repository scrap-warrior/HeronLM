"""
Reference-implementation tests.

These are not quality tests; a randomly initialized model has no quality to test. What they check
is that the MECHANISMS behave as described, because every one of these mechanisms has to be
reproduced bit-for-bit by the GGUF fork. If the reference is ambiguous or wrong, the parity test
against the engine is meaningless.

Run: python tests/test_reference.py
"""

import sys
from pathlib import Path

# The project is not installed as a package, so make the project root importable when the tests are
# run directly. Keeping this at the top is what lets `python tests/test_reference.py` work from any
# directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from heronlm.budget import attention_params, device_budget, ffn_params, param_counts
from heronlm.config import FULL, TINY
from heronlm.layers import Cache
from heronlm.memory import slot_ids
from heronlm.metadata import metadata, placeholder_vocabulary, special_token_ids
from heronlm.model import HeronLM
from heronlm.ngram import NGramTable
from heronlm.router import Router


def test_param_counts_match_the_roadmap():
    """
    The roadmap claims ~1.85B compute params at k=1, 2.626B at k=2, 4.170B at k=4.

    Computed from the config arithmetic rather than by instantiating the FULL model: instantiating it
    would allocate 5 experts x 4 slots x 125M params, about 10 GB of random weights, which is not what
    this test is for. If these numbers do not hold, the device budget built on them is wrong.
    """
    cfg = FULL()
    # A plain block: q, k, v, o plus three FFN matrices, no biases.
    attn = attention_params(cfg)
    ffn = ffn_params(cfg, cfg.ffn_width)
    per_block = attn + ffn
    assert abs(per_block - 48.23e6) / 48.23e6 < 0.01, per_block

    # Computed through budget.param_counts, so the test checks that the module reproduces the roadmap
    # rather than checking that the author added up the same sum twice.
    r = param_counts(cfg)
    assert abs(r["core_total"] - 1.854e9) / 1.854e9 < 0.01, r["core_total"]
    assert abs(r["active_at_k"][2] - 2.626e9) / 2.626e9 < 0.01
    assert abs(r["active_at_k"][4] - 4.170e9) / 4.170e9 < 0.01
    assert abs(r["expert_active"] - 125e6 * cfg.expert_slots) / (125e6 * cfg.expert_slots) < 0.02

    # Only the looped blocks multiply, which is the whole point of the loop knob being cheap: a wider
    # model would pay its whole stack on every pass.
    assert r["looped_part"] < r["core_total"] / 2

    # A config small enough to actually build must report the same shape, and the difference between the
    # analytic block and the instantiated block has to be exactly the read weights plus the reinjection
    # gate. If it is anything else, the module layout drifted from the arithmetic the budget is built on.
    t = TINY()
    m = HeronLM(t)
    small = m.param_report()
    analytic = t.vocab_size * t.d_model + t.n_blocks() * small["per_block"]
    assert abs(small["core_total"] - analytic) / analytic < 0.02, (small["core_total"], analytic)
    extras = small["instantiated_block"] - small["per_block"]
    expected = t.n_blocks() * t.max_loop * t.gr_branches + 1
    assert extras == expected, (extras, expected)


def test_cached_decode_matches_full_sequence():
    """
    Parity between incremental decode and a full-sequence pass.

    This is the test that catches the two most common silent bugs in a hybrid-attention
    implementation: positions treated as 0..T-1 instead of absolute positions, and cache appends
    happening on a pass that should only read. Both produce plausible output, so they have to be
    checked numerically rather than by inspection.
    """
    cfg = TINY()
    torch.manual_seed(0)
    m = HeronLM(cfg)
    m.eval()
    seq = torch.randint(0, cfg.vocab_size, (1, 12))

    # Full pass.
    full = m.forward(seq)

    # Incremental: feed tokens one at a time, compare logits at each position. The memory key has to
    # be built from the prefix plus the current token, otherwise the probe is keyed on a shorter
    # n-gram and the comparison is meaningless.
    caches = m.make_caches()
    for t in range(seq.shape[1]):
        logits = m.forward(
            seq[:, t : t + 1],
            1,
            None,
            caches,
            slot_ids(seq[:, t : t + 1], cfg, prefix=[seq[0, :t].tolist()]),
            position_offset=t,
        )
        assert torch.allclose(logits[0, 0], full[0, t], atol=1e-4), (t, float((logits[0, 0] - full[0, t]).abs().max()))
    assert caches[(0, 0)].len == seq.shape[1]


def test_reserved_branch_survives_the_loop():
    """
    Gate G0.3 in miniature, in the form that is actually checkable before training.

    The claim the design depends on is structural: branch 0 is written only by the first sublayer and
    by the memory read, so nothing later can overwrite it. If that holds, the recalled value is part of
    the read at every loop pass, which is what lets a memory lookup feed a looped block.

    Whether the branches are worth the engine work - the actual gate - cannot be decided from a
    randomly initialized model. The dilution argument only shows up as a loss difference after training,
    so this test checks the invariant that makes the gate measurable later, not the gate's conclusion.
    """
    cfg = TINY()
    torch.manual_seed(0)
    m = HeronLM(cfg).eval()

    # Give the table nonzero content; a zero table would make the memory read a no-op.
    with torch.no_grad():
        m.memory.table.weight.normal_(0, 1)

    seq = torch.randint(0, cfg.vocab_size, (1, 8))
    slots = slot_ids(seq, cfg)

    _, state1 = m.forward(seq, 1, None, None, slots, return_state=True)
    _, state3 = m.forward(seq, 3, None, None, slots, return_state=True)

    # The reserved branch must be identical at every loop count. If it changed, the memory value would
    # be overwritten and the branch rule would not be holding.
    assert torch.allclose(state1[:, 0], state3[:, 0], atol=1e-6), "the reserved branch is being overwritten"

    # The other branches must differ, otherwise the loop is doing nothing.
    assert not torch.allclose(state1[:, 1], state3[:, 1], atol=1e-6), "the loop changes nothing"

    # Single-stream arm: every write lands in the same stream, so the injected value is diluted by
    # every later write and the stream differs between loop counts. That is the difference between the
    # two arms, and it is why gate G0.3 exists.
    cfg_off = TINY()
    cfg_off.ngpt_gr_read = False
    m_off = HeronLM(cfg_off).eval()
    with torch.no_grad():
        m_off.memory.table.weight.copy_(m.memory.table.weight)
        m_off.stream.read_weights.copy_(m.stream.read_weights)

    _, s1 = m_off.forward(seq, 1, None, None, slots, return_state=True)
    _, s3 = m_off.forward(seq, 3, None, None, slots, return_state=True)
    assert not torch.allclose(s1[:, 0], s3[:, 0], atol=1e-6), "single-stream arm behaves like a branch"

    # And the memory read must still reach the output, or the whole path is decoration.
    # Comparing against the same model with the table zeroed is the only way to measure the read's
    # influence; a nonzero output change proves the injection is not decoration.
    logits_on = m.forward(seq, 3, None, None, slots)
    saved = m.memory.table.weight.clone()
    with torch.no_grad():
        m.memory.table.weight.zero_()
        logits_off = m.forward(seq, 3, None, None, slots)
        m.memory.table.weight.copy_(saved)
        influence = float((logits_on.detach() - logits_off).norm()) / float(logits_off.norm())
    assert influence > 1e-4, f"memory read does not reach the output: {influence}"


def test_loop_read_weights_are_indexed_by_pass():
    """
    Per-pass read weights have to be real per-pass parameters, not decoration.

    The test: a weight that belongs only to pass 2 must not change the output at k=1, because pass 2
    never happens there, and must change it at k=3. If both changed, the loop index is not actually
    being used, and "elastic depth" would be a wording choice rather than a mechanism.

    The loop is not a no-op even with identical weights: each pass adds to the state, so k=3 differs
    from k=1 by construction. What the per-pass weights add is the ability to mix the branches
    differently depending on how far into the loop you are.
    """
    cfg = TINY()
    torch.manual_seed(0)
    m = HeronLM(cfg).eval()
    seq = torch.randint(0, cfg.vocab_size, (1, 10))

    a1 = m.forward(seq, 1)[0, -1]
    a3 = m.forward(seq, 3)[0, -1]
    assert not torch.allclose(a1, a3, atol=1e-5), "the loop does not change the computation at all"

    # A pass-2-only weight: must be invisible at k=1, visible at k=3.
    with torch.no_grad():
        m.stream.read_weights[cfg.input_blocks, 2, 1] = 3.0
    b1 = m.forward(seq, 1)[0, -1]
    b3 = m.forward(seq, 3)[0, -1]
    assert torch.allclose(a1, b1, atol=1e-6), "a pass-2 weight changed a k=1 forward pass"
    assert not torch.allclose(a3, b3, atol=1e-5), "per-pass read weights are not indexed by pass"


def test_ngram_table_rules():
    """
    The three rules the engine has to agree on: min_cnt prunes after counting, top-k truncates,
    longest order wins on a hit.
    """
    cfg = TINY()
    cfg.ngram_orders = (3, 2, 1)
    cfg.ngram_min_cnt = 2
    cfg.ngram_top_k = 2

    t = NGramTable(cfg)
    # Two occurrences of the trigram, one with continuation 'a' twice and once with 'b' once.
    seq = [10, 11, 12, 20, 10, 11, 12, 20, 10, 11, 12, 30]
    t.build(seq)
    t.prune()

    hit = t.lookup([10, 11, 12])
    assert hit is not None
    assert hit[0] == 3, "longest order must win"
    toks = [tok for tok, _ in hit[1]]
    assert 20 in toks and 30 not in toks, "counts below min_cnt must be pruned before top-k"
    assert len(toks) <= cfg.ngram_top_k, "top-k must truncate"

    # A trigram seen once must not enter the table at all.
    t2 = NGramTable(cfg)
    t2.build([10, 11, 12, 40, 50, 51, 52, 60])
    t2.prune()
    assert t2.lookup([10, 11, 12]) is None, "min_cnt must gate entry into the table"


def test_min_cnt_is_a_corpus_condition_not_a_per_conversation_one():
    """
    The semantic the engine implements and the reference must match: min_cnt is measured over the whole
    build corpus, so support spread across conversations counts.

    The same fact twice inside one conversation contributes two counts. With min_cnt 3 that is below the
    threshold inside that conversation, but three conversations add up to six, and the engine's entry - the
    one the phone actually probes - exists. If the reference pruned at the end of each build() call it would
    delete the entry at the first conversation, and the table the harness measures would not be the table the
    engine ships. This is the bug that made the Phase 1 recall axis measure an almost empty table.

    The test says which is which rather than asserting both produce the same thing: the per-conversation
    reading and the corpus reading have to differ, otherwise the rule being tested is not observable.
    """
    cfg = TINY()
    cfg.ngram_orders = (4, 3, 2, 1)
    cfg.ngram_min_cnt = 3

    fact = [10, 11, 12, 13]
    # Three conversations, each with the fact once and the same continuation, so the support is entirely
    # cross-conversation.
    conversations = [fact + [99] for _ in range(3)]

    corpus_table = NGramTable(cfg)
    for conv in conversations:
        corpus_table.build(conv)
    corpus_table.prune()

    # The same counts, read per conversation: prune after each sequence instead of after the corpus.
    per_conversation = NGramTable(cfg)
    for conv in conversations:
        per_conversation.build(conv)
        per_conversation.prune()

    assert corpus_table.lookup(fact) is not None, "cross-conversation support must enter the table"
    assert per_conversation.lookup(fact) is None, (
        "per-conversation pruning produced the same table, so the corpus-level rule is not being tested"
    )
    entry = corpus_table.entries[tuple(fact)]
    assert entry.counts[99] == 3, (f"corpus support must accumulate across conversations, got {entry.counts[99]}")

    # A fact seen twice in one conversation and once in another reaches min_cnt in the corpus reading, but the
    # continuation count inside the first conversation alone is below it. Same rule, checked from the other side.
    split = NGramTable(cfg)
    split.build([10, 11, 12, 13, 99, 0, 10, 11, 12, 13, 99])   # 2 counts
    split.build([10, 11, 12, 13, 99])                          # 3rd
    split.prune()
    assert split.lookup(fact) is not None, "counts must accumulate across sequences, not restart per sequence"



def test_router_hysteresis_and_pinning():
    """
    Routing is frozen for the conversation. A single out-of-domain turn must not flip it, and
    pinning must override everything. Both are the behaviour Phase 2 gate measures as flapping.
    """
    cfg = TINY()
    torch.manual_seed(0)
    router = Router(cfg)
    g = torch.Generator().manual_seed(1)
    router.prototypes = F.normalize(torch.randn(cfg.n_experts, cfg.d_model, generator=g), dim=1)

    # Turn 1 clearly matches prototype 2.
    p_first = F.normalize(router.prototypes[2] + 0.05 * torch.randn(cfg.d_model, generator=g), dim=-1)
    state, _ = router.decide(p_first)
    assert state.slot == 2

    # Turn 2 is a near-random point: not enough evidence to move the conversation.
    p_noise = F.normalize(torch.randn(cfg.d_model, generator=g), dim=-1)
    state, margin = router.decide(p_noise, state)
    assert state.slot == 2, "one unusual turn must not flip a frozen decision"

    # Sustained evidence for prototype 4, past the hysteresis window, does move it.
    p_fourth = F.normalize(router.prototypes[4 % cfg.n_experts] + 0.01 * torch.randn(cfg.d_model, generator=g), dim=-1)
    for _ in range(cfg.router_hysteresis_turns + 2):
        state, margin = router.decide(p_fourth, state)
    assert state.slot == 4 % cfg.n_experts, "sustained evidence must eventually switch the decision"

    # Pinning overrides the sustained evidence.
    state.pinned_slot = 1
    state2, _ = router.decide(p_fourth, state)
    assert state2.slot == 1, "pinning must override everything"


def test_speculation_can_only_agree_or_be_ignored():
    """
    Speculative output must equal non-speculative output, exactly.

    This is the test that proves the n-gram path verifies rather than blends. A proposal that the core
    would not have produced has to be dropped; the accepted prefix has to be the prefix the core
    already agrees with. If speculation changed a single token, it would mean the table was writing
    tokens the model did not choose, which is how repetition loops start.

    Rejected proposals leaving no trace is covered by the same equality: if a rejected proposal survived
    in the cache, the tokens after it would differ from plain decoding, and the equality would fail. A
    separate cache-content assertion would be vacuous here, because it would compare against caches that
    generate does not use.

    Two arms, so both branches of the verification logic are exercised:
      - a table whose top continuation IS what the model would say, so proposals get accepted;
      - a table whose top continuation is deliberately NOT what the model would say, so proposals get
        rejected. In both arms the emitted tokens must be identical to plain decoding.
    """
    cfg = TINY()
    cfg.ngram_orders = (3, 2)
    cfg.ngram_min_cnt = 2
    torch.manual_seed(0)
    m = HeronLM(cfg).eval()

    prompt = [5, 6, 7]
    plain, _ = m.generate(prompt, 12, 1, None, table=None, speculate=False, temperature=0.0)

    # Arm 1: the table is built from the model's own behaviour repeated, so its top continuation is the
    # token the model would have chosen. Acceptance is the expected outcome here.
    agreeing = NGramTable(cfg)
    agreeing.build((prompt + plain) * 6)
    agreeing.prune()
    out1, stats1 = m.generate(prompt, 12, 1, None, table=agreeing, speculate=True, temperature=0.0)
    assert out1 == plain, f"accepted proposals changed the output: {out1} vs {plain}"
    assert stats1["accepted"] > 0, "the agreeing table produced no acceptance, so the accept path is untested"

    # Arm 2: the top continuation is deliberately the wrong token. It must be dropped, and the model's
    # own choice must be emitted instead. This is the anti-blending guarantee.
    wrong = 200 if plain[0] != 200 else 201
    disagreeing = NGramTable(cfg)
    disagreeing.build((prompt + [wrong]) * 6 + (prompt + plain) * 3)
    disagreeing.prune()
    out2, stats2 = m.generate(prompt, 12, 1, None, table=disagreeing, speculate=True, temperature=0.0)
    assert out2 == plain, f"a rejected proposal changed the output: {out2} vs {plain}"
    assert stats2["accepted"] < stats2["proposed"], "no proposal was rejected, so the reject path is untested"


def test_device_budget_reproduces_the_roadmap_table():
    """
    The roadmap's device table is a claim, not a measurement, so it must be recomputable from the config.
    If the recomputed numbers land outside the roadmap's ranges, the phone budget was built on arithmetic
    that cannot be checked.

    One line is expected to differ, and the test states the direction rather than hiding it: the roadmap
    quotes the core as ~1.15 GB, which assumes the whole core at Q4_K_M. The design rule says the tied
    embedding is never Q4, so the honest figure is higher. The budget still fits, which is what the check
    is for.
    """
    cfg = FULL()
    b = device_budget(cfg)

    assert 0.60 <= b["disk"]["decode_table"] <= 0.90, b["disk"]["decode_table"]
    assert 0.35 <= b["disk"]["memory_table_int8"] <= 0.42, b["disk"]["memory_table_int8"]
    assert 0.25 <= b["disk"]["experts_all_five"] / cfg.n_experts <= 0.32, b["disk"]["experts_all_five"]
    assert 1.90 <= b["resident"]["total"] <= 2.30, b["resident"]["total"]
    assert 3.60 <= b["disk"]["total"] <= 4.40, b["disk"]["total"]

    # Two lines are expected to differ from the roadmap table, and the test states the direction instead
    # of rounding them into agreement. Both are the same kind of error: the roadmap counted something
    # once where the design rule says it must be counted differently.
    roadmap_core_claim = 1.15
    assert b["disk"]["core_total"] > roadmap_core_claim, b["disk"]["core_total"]
    assert b["disk"]["core_total"] - roadmap_core_claim < 0.20, b["disk"]["core_total"]

    # The roadmap quoted 50-100 MB for the SWA cache, which is the figure for the looped blocks only.
    # Every window-bounded block needs its own cache, including the input and output blocks, so the honest
    # figure is one third higher. This is the line that can grow if a longer window is chosen, so it must be
    # recomputable rather than remembered.
    roadmap_cache_claim = 0.10
    assert b["resident"]["swa_cache"] > roadmap_cache_claim, b["resident"]["swa_cache"]
    loop_only = cfg.loop_blocks * cfg.kv_heads * cfg.head_dim * 2 * cfg.swa_window * 2 / 1000.0**3
    assert 0.05 <= loop_only <= roadmap_cache_claim, loop_only
    assert b["resident"]["swa_cache"] < 0.15, b["resident"]["swa_cache"]

    # The flat line covers only the blocks that are bounded by the window. The full-attention blocks cannot be
    # reported as a flat number, because a later query can still reach every entry they hold, so their cache
    # grows with the context and has to be its own line. Without it the resident total is the cost of a
    # conversation no longer than the window, read as though it were the budget at any length.
    growth = b["resident"]["full_attention_cache_growth_per_token"]
    assert growth > 0, growth
    assert growth * cfg.swa_window > 0.003, growth * cfg.swa_window
    bounded = sum(1 for i in range(cfg.n_blocks()) if cfg.mixer_mode(i) == "swa")
    assert bounded < cfg.n_blocks(), "the resident line has to exclude the blocks that grow"

    # Storage is not the constraint: one probe per token at 5-10 tokens/s makes UFS latency irrelevant.
    # The binding limit is the RAM hot cache, so the resident line must show it as its own quantity
    # rather than folding the table into "model size".
    assert b["resident"]["decode_table_hot_cache"] < b["disk"]["decode_table"]


def test_metadata_agrees_with_the_module_layout():
    """
    Metadata is what the fork reads, so a disagreement between metadata and the module is a converter
    loading tensors into the wrong blocks. Both sides derive from the config methods, so this test checks
    the derivation rather than a hand-written table.

    The namespace rule is checked too: every architecture key must live under general.arch.heron.*, which
    is what lets an upstream reader ignore an unknown architecture instead of failing on it.
    """
    cfg = TINY()
    m = HeronLM(cfg)
    tokens = placeholder_vocabulary(cfg.vocab_size)
    md = metadata(cfg, tokens, ["merge a b"])

    assert md["general.arch.heron.block_count"] == m.block_count
    assert md["general.arch.heron.expert_slot_layers"] == m.expert_blocks
    assert md["general.arch.heron.full_attention_layers"] == [
        i for i in range(m.block_count) if cfg.mixer_mode(i) == "full"
    ]
    # Branch assignment is the rule the memory read depends on. If the engine rotated the reserved
    # branch, the recalled value would be overwritten on the second loop pass.
    assert md["general.arch.heron.gr.branch_of"] == m.stream.branch_of
    assert md["general.arch.heron.gr.reserved_branch"] == 0
    assert md["general.arch.heron.table.read_block"] == cfg.ngram_memory_read_block
    assert md["general.arch.heron.loop.layers"] == list(
        range(cfg.input_blocks, cfg.input_blocks + cfg.loop_blocks)
    )

    # Only the architecture namespace and the tokenizer namespace. `general.file_type` and the GGUF version
    # are not in metadata(): they are facts about the file the writer produces, not about the architecture, so
    # they are recorded by the writer at the point where they can be known (see converter.file_type_for).
    allowed = {"general.architecture", "general.name"}
    stray = [
        k for k in md
        if not (k.startswith("general.arch.heron.") or k.startswith("tokenizer.") or k in allowed)
    ]
    assert not stray, f"keys outside the heron namespace: {stray}"

    # The table is keyed by token id, so a vocabulary-length mismatch silently invalidates every entry.
    # The check is in metadata itself; this asserts it actually fires.
    try:
        metadata(cfg, tokens[:-1], ["merge a b"])
    except ValueError:
        pass
    else:
        raise AssertionError("metadata must reject a vocabulary whose length does not match the config")


def test_special_token_ids_and_chat_template_come_from_the_vocabulary():
    """
    The tokenizer is the one hard coupling in the design: the n-gram table is keyed by token id, so the
    ids the metadata records have to be the ids of the vocabulary that was actually written. Qwen's
    published vocabulary puts its special tokens at particular positions, and that is exactly the trap:
    code that remembers those numbers passes every test on the reference tokenizer and silently
    tokenizes a different model on any other one.

    The two vocabularies below place the same tags at deliberately wrong, and mutually different,
    positions. A hardcoded id - or an implementation that resolves the first fixture's positions
    through a constant - fails both. The template is checked against the same strings, because it is
    the other consumer of the tags: an id map and a template that disagree would render a chat turn
    into ids the table was never built with, and nothing in the model output would say so.

    The negative cases matter as much as the positive one. A vocabulary missing a tag cannot render the
    template the model was trained with, and a duplicated tag makes the id ambiguous; both have to be
    hard errors instead of a fallback that picks something plausible.
    """
    cfg = TINY()

    # Independent spellings, concatenated rather than written out, so this test does not borrow the
    # implementation's constants for the fixture it uses to check them.
    im_start_tag = "<|" + "im_start|>"
    im_end_tag = "<|" + "im_end|>"
    end_of_text_tag = "<|" + "endoftext|>"

    def vocab(positions: dict[str, int]) -> list[str]:
        tokens = [f"tok{i}" for i in range(cfg.vocab_size)]
        for tag, index in positions.items():
            tokens[index] = tag
        return tokens

    first = vocab({im_start_tag: 3, im_end_tag: 19, end_of_text_tag: 41})
    ids = special_token_ids(first)
    assert ids == {"im_start": 3, "im_end": 19, "endoftext": 41}, ids

    md = metadata(cfg, first, ["merge a b"])
    assert md["tokenizer.ggml.bos_token_id"] == 41, md["tokenizer.ggml.bos_token_id"]
    assert md["tokenizer.ggml.eos_token_id"] == 19, md["tokenizer.ggml.eos_token_id"]
    assert md["tokenizer.ggml.padding_token_id"] == 41, md["tokenizer.ggml.padding_token_id"]

    # Second placement, so the ids are visibly read from the argument rather than from a table that
    # happens to contain the first fixture's numbers.
    second = vocab({im_start_tag: 40, im_end_tag: 2, end_of_text_tag: 17})
    md2 = metadata(cfg, second, ["merge a b"])
    assert md2["tokenizer.ggml.eos_token_id"] == 2
    assert md2["tokenizer.ggml.padding_token_id"] == 17

    # The template must use the same spelling that is at those ids. The roles and the generation flag
    # are literal text, because they are Jinja variables rather than tokens.
    template = md["tokenizer.chat_template"]
    assert first[3] in template and first[19] in template, template
    assert first[3] + "assistant" in template, template
    assert "message.role" in template and "add_generation_prompt" in template, template
    assert "for message in messages" in template and "{%- endfor %}" in template, template

    # Exactness: the pinned Qwen2.5-1.5B-Instruct config carries the published template. Substituting the
    # tag constants into our placeholders has to reproduce it byte for byte, or we are back to an
    # approximation with a divergence that only shows up on the conversation shapes the subset omitted.
    # The check uses the pinned artifact when it is present; the structural checks above always run, so a
    # checkout without the tokenizer directory still tests the mechanism rather than skipping silently.
    pinned = Path(__file__).resolve().parent.parent / "tokenizer/qwen2.5/tokenizer_config.instruct.json"
    if pinned.exists():
        import json

        published = json.loads(pinned.read_text(encoding="utf-8"))["chat_template"]
        assert template == published, (
            "the chat template is not the published Qwen2.5 template; "
            f"ours is {len(template)} chars, the pinned file is {len(published)}"
        )
        assert len(template) == 2507, len(template)

    # A missing tag is a hard error. The vocabulary keeps its length, so this is the tokenizer check
    # firing rather than the length check.
    for tag, name in [
        (im_start_tag, "im_start"),
        (im_end_tag, "im_end"),
        (end_of_text_tag, "endoftext"),
    ]:
        broken = [token for token in first if token != tag]
        broken.append(f"tok_filler_{name}")
        try:
            metadata(cfg, broken, ["merge a b"])
        except ValueError as e:
            assert name in str(e), (name, str(e))
        else:
            raise AssertionError(f"metadata must reject a vocabulary with no {name} token")

    # A duplicate tag makes the id ambiguous. Picking the first copy would hide a broken export.
    duplicated = list(first)
    duplicated[100] = im_end_tag  # the first copy is still at 19
    try:
        metadata(cfg, duplicated, ["merge a b"])
    except ValueError as e:
        assert "ambiguous" in str(e), str(e)
    else:
        raise AssertionError("metadata must reject an ambiguous special-token id")


def test_fallback_path_always_exists():
    """
    Routing is never allowed to have no effect. If no expert is engaged, the generalist (which is
    the base FFN) runs, so there is always a defined path.
    """
    cfg = TINY()
    torch.manual_seed(0)
    m = HeronLM(cfg).eval()
    seq = torch.randint(0, cfg.vocab_size, (1, 6))

    a = m.forward(seq, 1, None)
    b = m.forward(seq, 1, cfg.generalist_slot)
    assert torch.allclose(a[0, -1], b[0, -1], atol=1e-6), "no-expert path must equal the generalist path"

    # A different expert changes the output, which is the point of having experts at all.
    c = m.forward(seq, 1, 1)
    assert not torch.allclose(a[0, -1], c[0, -1], atol=1e-6)


def test_the_generalist_is_a_module_the_harness_sees_and_a_checkpoint_does_not():
    """
    Slot 0 exists as a module and not as bytes.

    `named_buffers()` covers it, so a run that reports n_experts experts has a module for each of them and the
    param report can name the generalist as a set of slots. Its tensors are non-persistent buffers, the only
    form that is excluded from the saved state in both directions: `state_dict()` skips them, and
    `load_state_dict()` does not expect them. That symmetry is what this asserts, because a write-side exclusion
    on a Parameter would fail an ordinary round trip - `load_state_dict` builds its expectation from
    `_parameters` through `named_modules()`, so the modules nested under slot 0 would still be expecting bytes
    the write side refused to write.

    The cost of keeping the module is asserted too: a buffer never receives a gradient and nothing in forward()
    reads slot 0, so these tensors are the harness's view only. That is why the param report keeps them out of
    `held` and reports `stored_total` as the bytes a checkpoint actually carries.
    """
    cfg = TINY()
    torch.manual_seed(0)
    m = HeronLM(cfg).eval()

    slot = cfg.generalist_slot
    prefix = f"experts.{slot}."
    at_slot = [n for n, _ in m.named_buffers() if n.startswith(prefix)]
    assert at_slot, "the harness has to see the generalist as a module, so the case has to exist"
    # 4 tensors per slot: the three SwiGLU matrices and the slot's own output gate.
    assert len(at_slot) == cfg.expert_slots * 4, (len(at_slot), sorted(at_slot))
    # Not a parameter, so not in a state dict and not in an optimizer's parameter groups.
    assert not any(n.startswith(prefix) for n, _ in m.named_parameters()), sorted(n for n, _ in m.named_parameters())
    assert not any(k.startswith(prefix) for k in m.state_dict()), sorted(m.state_dict())

    # What a training run does: save what state_dict wrote, load it back. This is the round trip a write-side
    # exclusion on a Parameter would break.
    m.load_state_dict(m.state_dict(), strict=True)

    # Bytes at the generalist index have no name, so a dict carrying them is refused. That is the checkpoint from
    # a model where slot 0 was an ordinary expert: there is no key these bytes can be placed at, so they are
    # reported as unexpected rather than half-loaded.
    foreign = dict(m.state_dict())
    key = prefix + "0.ffn.gate.weight"
    foreign[key] = torch.zeros(cfg.expert_ffn_width, cfg.d_model)
    try:
        m.load_state_dict(foreign, strict=True)
    except RuntimeError as e:
        assert "Unexpected key(s)" in str(e) and key in str(e), str(e)
    else:
        raise AssertionError("a state dict that carries the generalist must be refused, not half-loaded")

    # forward() reads slot 0 as the block's base FFN, so the module's tensors are not the tensors that compute.
    # The claim is asserted by changing them: if forward() read slot 0, the logits would change. They cannot,
    # because the generalist that computes for slot 0 is each expert block's base FFN.
    seq = torch.randint(0, cfg.vocab_size, (1, 6))
    before = m.forward(seq, 1, slot)
    assert before.shape == (1, 6, cfg.vocab_size), before.shape
    for _, b in m.named_buffers():
        b.mul_(2.0)
    assert torch.equal(m.forward(seq, 1, slot), before), "forward() read the generalist's tensors"
    assert all(b.grad is None for _, b in m.named_buffers()), "a buffer cannot be trained"

    r = m.param_report()
    assert r["generalist_slot_tensors"] > 0, r
    # `stored_total` is read from state_dict, which is what a checkpoint carries, and it is not the same number as
    # `instantiated_total`: the shared read weight is one Parameter counted once by `named_parameters()` and
    # written as one aliased key per block - the same rows the converter emits per block. The gap between the two
    # numbers is the block keys, so the test computes it from the state dict rather than assuming it.
    stored = sum(t.numel() for t in m.state_dict().values())
    assert r["stored_total"] == stored, r
    aliased = [k for k in m.state_dict() if k.endswith("stream.read_weights") and k != "stream.read_weights"]
    assert len(aliased) == m.block_count, aliased
    assert r["stored_total"] - r["instantiated_total"] == len(aliased) * m.stream.read_weights.numel(), (
        r["stored_total"], r["instantiated_total"], aliased
    )
    # The generalist is not capacity: nothing stores it, so `held_but_not_computed` is built from `parameters()`
    # and does not include it. If the module were counted as capacity, these two would differ by exactly it.
    assert r["held_but_not_computed"] == r["instantiated_total"] - r["core_total"] - r["expert_active"], r


def test_full_attention_caches_grow_and_sliding_window_caches_are_bounded():
    """
    Eviction is only allowed for entries that no later query can reach, and whether that is true is a
    property of the mixer mode, not of the cache: a sliding-window cache is bounded by the window, a
    full-attention cache has no bound because a later query can still reach every entry it holds.

    The check has to be done on a sequence longer than the window, because below the window the two rules
    cannot be told apart. Incremental decode must match a full pass at every position past the window,
    and for the full-attention block that is only true if it kept the entries the window rule would have
    dropped. This is the parity rule the fork has to follow: the bound is per mixer mode, so a converter
    that trims every cache to the window produces a model that diverges from the reference once the
    conversation is longer than the window, and nothing about the output says so.
    """
    cfg = TINY()
    window = cfg.swa_window
    torch.manual_seed(0)
    m = HeronLM(cfg).eval()

    full_blocks = [i for i in range(m.block_count) if cfg.mixer_mode(i) == "full"]
    swa_blocks = [i for i in range(m.block_count) if cfg.mixer_mode(i) == "swa"]
    assert full_blocks and swa_blocks, "this test only means something when both mixer modes exist"

    seq = torch.randint(0, cfg.vocab_size, (1, window + 20))
    full = m.forward(seq)
    caches = m.make_caches()
    for t in range(seq.shape[1]):
        logits = m.forward(
            seq[:, t : t + 1], 1, None, caches,
            slot_ids(seq[:, t : t + 1], cfg, prefix=[seq[0, :t].tolist()]),
            position_offset=t,
        )
        assert torch.allclose(logits[0, 0], full[0, t], atol=1e-4), (
            t, float((logits[0, 0] - full[0, t]).abs().max())
        )

    # The bound, as a count rather than as a comment: the windowed block holds at most `window` entries,
    # the full block holds the whole context.
    assert caches[(swa_blocks[0], 0)].len <= window, caches[(swa_blocks[0], 0)].len
    assert caches[(full_blocks[0], 0)].len == seq.shape[1], caches[(full_blocks[0], 0)].len


def test_verification_cannot_change_what_the_model_has_committed():
    """
    Verification runs against copies, and the promise is that a rejected proposal leaves no trace. Eviction
    is part of the state, so the promise has a second half: eviction is a decision about positions the model
    committed, and the end of a proposed span is not one of them. A verification pass that evicts from the
    span end drops entries the sequential path would still have had, and for a full-attention cache those
    entries are what later tokens attend to - so a guess the model never used would still change the output.

    The equality below is the observable form of the rule. It is not hypothetical: the Phase 1 harness found
    a real run where the speculative and model-only decodes disagreed, and the disagreement was the eviction
    measured from a span end rather than from a committed position. That is the harness doing the job the
    invariant exists for, and it is why the rule is stated for both cache kinds rather than treated as a
    memory detail.
    """
    cfg = TINY()
    window = cfg.swa_window
    assert window < 64, "this test only means something when the decode runs past the window"
    torch.manual_seed(11)
    m = HeronLM(cfg).eval()

    # A prompt longer than the window, so eviction fires during the run.
    prompt = [int(x) for x in torch.randint(0, cfg.vocab_size, (window + 12,))]
    plain, _ = m.generate(prompt, 14, 1, None, table=None, speculate=False, temperature=0.0)

    # A table whose entries are deliberately not what the core would choose, so the accepted prefix is
    # empty at the positions the rule is about. Repeated enough times to survive min_cnt, which is the
    # corpus-level rule the table itself implements.
    top = max(cfg.ngram_orders)
    wrong = [int(x) for x in torch.randint(0, cfg.vocab_size, (4,))]
    disagreeing = NGramTable(cfg)
    for _ in range(cfg.ngram_min_cnt):
        for tok in wrong:
            disagreeing.build(prompt[-top:] + [tok])
    disagreeing.prune()
    assert disagreeing.lookup(prompt) is not None, "the run has to be measured on an entry that exists"

    out, stats = m.generate(prompt, 14, 1, None, table=disagreeing, speculate=True, temperature=0.0)
    assert out == plain, f"the speculative run changed the output: {out[len(prompt):]} vs {plain[len(prompt):]}"
    assert stats["proposed"] > 0 and stats["accepted"] < stats["proposed"], (
        f"no proposal was rejected, so the rule is not exercised: {stats}"
    )

    # The rule on its own, without a model, so the disagreement between the two eviction points is visible
    # rather than inferred from a run that happened to agree. A span that crosses the window is the case
    # where evicting from the span end and evicting from the committed position are not the same operation.
    c = Cache(cfg.kv_heads, cfg.head_dim, window)
    positions = torch.arange(3 * window)
    c.append(torch.zeros(1, cfg.kv_heads, positions.numel(), cfg.head_dim),
            torch.zeros(1, cfg.kv_heads, positions.numel(), cfg.head_dim), positions)
    committed = 3 * window - 5          # the last committed position
    span_end = committed + 4            # the end of a proposed span, four tokens past it
    v = c.clone(allow_evict=False)
    v.evict(span_end, span_end)
    assert v.len == c.len, "a verification copy must not evict: it dropped entries"
    c.evict(committed, committed)
    assert v.len > c.len, (
        "evicting from a position the model never committed has to drop entries that evicting from the "
        f"committed position keeps; got {v.len} vs {c.len}"
    )


if __name__ == "__main__":
    from tests import run_tests

    raise SystemExit(1 if run_tests.run(globals()) else 0)

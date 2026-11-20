"""
Readable end-to-end report: what the design numbers are, and what each mechanism actually does.

Run: python scripts/report.py

Two kinds of numbers appear here and they must not be mixed up:
  - numbers computed from the FULL config. The FULL model is not instantiated: 5 experts x 4 slots x
    125M params is about 10 GB of random weights, so it is arithmetic, not a running model.
  - numbers measured on the TINY model, which has the same mechanisms at miniature scale.

The measured part cannot measure quality, because a randomly initialized model has no quality. What it
can measure is whether each mechanism behaves as described, which is what the GGUF fork has to reproduce
bit-for-bit. Where a number is only plumbing rather than quality, the report says so instead of letting a
plausible-looking figure stand in for a result.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from heronlm.budget import device_budget, param_counts
from heronlm.config import FULL, HeronConfig, TINY
from heronlm.memory import slot_ids
from heronlm.metadata import BASE_ALPHABET_SIZE, base_alphabet, byte_level_rule, metadata, placeholder_vocabulary
from heronlm.model import HeronLM
from heronlm.ngram import NGramTable
from heronlm.router import Router, RouterState


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show_params(cfg) -> None:
    """
    Active, resident and capacity, separately.

    This is the table the whole design rests on. Collapsing it into one "model size" is what makes
    every phone argument wrong: 1.85B is active at k=1, but the phone holds all 32 blocks plus the
    embedding regardless of k, and the tables never compute at all.
    """
    rule("1. Parameter counts (FULL config, arithmetic rather than an instantiated model)")
    c = param_counts(cfg)

    print(f"per block            {c['per_block']:>12,} params")
    print(f"embedding (tied)     {c['embedding']:>12,} params")
    print(f"core total           {c['core_total']:>12,} params")
    print(f"looped part          {c['looped_part']:>12,} params   <- the only part paid k times")
    print(f"paid once            {c['paid_once']:>12,} params   <- embedding + input + output blocks")
    print("\nactive params per token, by loop count:")
    for k, n in c["active_at_k"].items():
        print(f"  k={k}  {n:>12,}")

    print(f"\nexpert, active when engaged   {c['expert_active']:>12,} params "
          f"({cfg.expert_slots} slots x {c['expert_slot']:,})")
    print(f"memory table (read, not computed) {c['memory_table']:>10,} values")

    # Only the looped part scales, which is why the loop is a cheap knob: a wider model would pay its
    # whole stack on every pass. If this ratio is not roughly a third, the loop is not the cheap part.
    print(f"looped share of core: {c['looped_part'] / c['core_total']:.1%}")


def show_device_budget(cfg) -> None:
    rule("2. Device budget (S23: 8 GB RAM, 256 GB storage)")
    b = device_budget(cfg)

    print("on disk:")
    for key, value in b["disk"].items():
        print(f"  {key:<24} {value:6.3f} GB")
    print("\nresident in RAM:")
    for key, value in b["resident"].items():
        if key == "full_attention_cache_growth_per_token":
            # A rate, not a total, so printing it as GB rounds to 0.000 and reads as nothing. It is the
            # line that grows with the conversation, which is why it cannot be folded into the total.
            print(f"  {key:<36} {value * 1000:6.3f} MB per token")
        else:
            print(f"  {key:<24} {value:6.3f} GB")

    print("\nThe resident total is the short-context figure: it includes the cache for the blocks bounded")
    print("by the window, and cannot include the blocks that are not. At the window itself the growing")
    print(f"part is {b['resident']['full_attention_cache_growth_per_token'] * cfg.swa_window * 1000:.1f} MB,")
    print("and it keeps growing with the conversation, which is why the token budget is still an open")
    print("question rather than a number this report can state.")

    print("\nTwo lines here are the reason the budget works at all:")
    print("  - experts are on disk for all five, in RAM for one. Routing loads exactly one, so the")
    print("    resident line is one expert, not the sum. Storage pays for capacity, RAM pays for one.")
    print("  - the memory table is on disk in full, resident only in its hot portion. One probe per")
    print("    token at 5-10 tokens/s makes storage latency irrelevant; RAM is what limits it.")


def show_ngram_table(cfg: HeronConfig) -> NGramTable:
    """
    Table build and lookup, on a synthetic corpus.

    The corpus is a repeating pattern, which is what a real corpus looks like in the parts that make an
    n-gram table worth having: boilerplate, recurring phrasings, the same code idioms. What it is NOT is
    a corpus disjoint from an eval set, so nothing here says anything about quality. That caveat is the
    point: a large table can memorize benchmark answers, and reading that as a table gain is
    self-deception.
    """
    rule("3. N-gram table (decode path): build cost and lookup behaviour")
    table = NGramTable(cfg)

    corpus = [
        10, 11, 12, 20, 10, 11, 12, 20, 10, 11, 12, 30,
        10, 11, 12, 20, 40, 41, 42, 40, 41, 42, 43,
        10, 11, 12, 20, 10, 11, 12, 20, 15, 16,
    ] * 20
    table.build(corpus)
    table.prune()
    stats = table.hit_stats(corpus)

    print(f"entries kept after min_cnt={cfg.ngram_min_cnt} pruning: {stats['entries']}")
    print(f"lookups: {stats['lookups']}  hit rate: {stats['hit_rate']:.1%}")
    print("hits by order (longest hit wins):", stats["hits_by_order"])
    print(f"engine footprint estimate: {stats['bytes_estimate']:,} bytes "
          f"({stats['bytes_estimate'] / max(stats['entries'], 1):.1f} bytes per entry)")

    hit = table.lookup(corpus[:4])
    print(f"\nlookup of {corpus[:4]} -> order {hit[0]}, top continuations: {hit[1]}")
    print("The trigram wins over the bigram because longest-first is the rule: a 4-token span is")
    print("stronger evidence than a 2-token span, so lower orders are only a fallback on a miss.")

    single = NGramTable(cfg)
    single.build([10, 11, 12, 99])
    single.prune()
    print(f"\nsingleton n-gram lookup: {single.lookup([10, 11, 12])} "
          "-> None, because min_cnt gates entry into the table before top-k is taken.")
    return table


def show_routing(cfg: HeronConfig, model: HeronLM) -> RouterState:
    """
    Routing is conversation-level: decided once, then frozen.

    Three properties the Phase 2 gate measures, shown in order: the decision, a single out-of-domain
    turn that must not flip it, and a pin that overrides everything including the margin.
    """
    rule("4. Router: one decision per conversation, then hysteresis")
    torch.manual_seed(0)
    router = Router(cfg)

    # The conversation is scored from the pooled state at the end of the input block, before the loop.
    # It has to be taken there: if routing depended on loop passes, the same conversation would route
    # differently at k=1 and k=4, which is not reproducible.
    conversation = [7, 8, 9, 10, 11, 12, 13, 14]
    _, pooled = model.forward(torch.tensor([conversation]), return_pooled=True)
    state, margin = router.decide(pooled)
    print(f"turn 1: slot {state.slot}, confidence {state.confidence:.3f}, margin {margin:.3f}")

    # The same conversation, then one turn that looks like a different domain. A single turn must not
    # move the conversation, so the decision stays put even though another slot scores higher.
    out_of_domain = [200, 201, 202, 203, 204, 205, 206, 207]
    _, pooled2 = model.forward(torch.tensor([out_of_domain]), return_pooled=True)
    state2, margin2 = router.decide(pooled2, state)
    would_have_been = state2.history[-1][0]
    print(f"turn 2 (out of domain): scored slot {would_have_been}, "
          f"kept slot {state2.slot}, margin {margin2:.3f}")
    print(f"stable turns: {state2.stable_turns} (hysteresis requires {cfg.router_hysteresis_turns})")

    # Pinning overrides the margin and the hysteresis. It exists because a user who knows the domain
    # should not have to trust a router that can flip decisions for reasons invisible in BF16.
    pinned = RouterState(active_slot=state.slot, confidence=state.confidence, pinned_slot=2)
    state3, _ = router.decide(pooled2, pinned)
    print(f"pinned to 2 -> effective slot {state3.slot}. Pinning overrides the margin, as it must.")
    print("\nThese are untrained random prototypes, so which slot wins is arbitrary. What is not")
    print("arbitrary is that the decision is frozen, the margin is enforced, and the pin wins.")
    return state


def show_generation(cfg: HeronConfig, model: HeronLM, table: NGramTable, slot: int) -> None:
    """
    Greedy decoding with and without speculation, in two arms.

    The design rule: proposals are VERIFIED, never blended. So the output must be identical to plain
    greedy decoding, token for token, in both arms. If it ever differs, the n-gram path has become a
    second model overriding the core, which is how repetition loops appear.

    Arm A uses a table built from the model's own first output. That is a degenerate corpus - it is not
    evidence about real acceptance rates, which is Phase 1 gate territory - but it makes the accept path
    observable. Arm B uses a table that contradicts the core, which is what makes the reject path
    observable. Showing only the accept arm would hide half the mechanism.
    """
    rule("5. Generation: verified speculation, accept arm and reject arm")
    prompt = [10, 11, 12, 20, 10, 11, 12, 20, 40, 41, 42, 40]

    plain, plain_stats = model.generate(prompt, max_new_tokens=24, loop=1, active_slot=slot)
    print("plain greedy, no table:", plain[len(prompt):])
    print(f"forward passes: {plain_stats['forward_passes']} (one per token)")
    print("The output is a single repeated token because the model is randomly initialized. Nothing")
    print("here measures quality; a random model has no quality to measure.")

    # Arm A: proposals are the model's own past, so every proposal is confirmed and speculation pays:
    # one forward pass produced the whole span. That is the whole economic case for having it.
    self_table = NGramTable(cfg)
    self_table.build(plain)
    self_table.prune()
    arm_a, stats_a = model.generate(prompt, 24, loop=1, active_slot=slot, table=self_table, speculate=True)
    print("\narm A, table built from the model's own output:")
    print("  output:", arm_a[len(prompt):])
    print(f"  lookups {stats_a['lookups']}, hits {stats_a['hits']} ({stats_a['hit_rate']:.1%}), "
          f"proposed {stats_a['proposed']}, accepted {stats_a['accepted']} ({stats_a['acceptance']:.1%})")
    print(f"  forward passes: {stats_a['forward_passes']} "
          f"({(len(arm_a) - len(prompt)) / max(stats_a['forward_passes'], 1):.2f} tokens per pass)")

    # Arm B: a table that disagrees with the core. The prompt must end on the table key for the lookup to
    # hit, so this arm uses its own prompt; the rejection then has to remove the tail completely.
    prompt_b = [10, 11, 12, 20, 10, 11, 12, 20]
    contradicts = NGramTable(cfg)
    contradicts.build([10, 11, 12, 20, 77] * 4)  # seen 4 times, so it clears min_cnt
    contradicts.prune()
    plain_b, plain_b_stats = model.generate(prompt_b, 12, loop=1, active_slot=slot)
    arm_b, stats_b = model.generate(prompt_b, 12, loop=1, active_slot=slot, table=contradicts, speculate=True)
    print(f"\narm B, table that proposes something the core would not choose:")
    print(f"  lookup of {prompt_b[-4:]} -> {contradicts.lookup(prompt_b)[1]}")
    print(f"  plain greedy:    {plain_b[len(prompt_b):]}")
    print(f"  with speculation: {arm_b[len(prompt_b):]}")
    print(f"  lookups {stats_b['lookups']}, hits {stats_b['hits']} ({stats_b['hit_rate']:.1%}), "
          f"proposed {stats_b['proposed']}, accepted {stats_b['accepted']} ({stats_b['acceptance']:.1%})")
    print(f"  forward passes: {stats_b['forward_passes']} vs {plain_b_stats['forward_passes']} without")
    print("  the table. The extra passes are the price of verification, and they are the only price:")
    print("  a rejected proposal cannot change the text.")
    print("  What this arm shows: when the table is wrong at the first proposed token, speculation is")
    print("  pure overhead. It is never faster than plain decoding in that case, so acceptance rate is")
    print("  not a quality claim, it is a cost model.")

    assert plain == arm_a, "the accept arm changed the output; it may verify, not blend"
    assert plain_b == arm_b, "the reject arm changed the output; a rejected proposal must not survive"
    print("\nBoth arms produce exactly the plain greedy output. That equality is the whole guarantee:")
    print("a wrong guess costs compute time and cannot change the text.")


def show_loop_and_memory(cfg: HeronConfig, model: HeronLM, slot: int) -> None:
    """
    The two mechanisms that can be decoration: the loop and the memory read.

    A loop that changes nothing is a loop that costs compute for nothing, and a memory read that never
    reaches the output is a lookup for nothing. Both are gates for a reason, and both are visible here
    as structure rather than as results, because nothing measured on a random model is a result.
    """
    rule("6. Loop depth and the memory read: are they doing work?")
    prompt = [10, 11, 12, 20, 10, 11, 12, 20, 40, 41, 42, 40]
    seq = torch.tensor([prompt])
    slots = slot_ids(seq, cfg)

    _, state1 = model.forward(seq, 1, slot, None, slots, return_state=True)
    _, state3 = model.forward(seq, 3, slot, None, slots, return_state=True)

    # Branch 0 is the reserved branch. It must be identical at every loop count, otherwise the injected
    # memory value gets overwritten on the second pass and the whole memory path is decoration.
    reserved_same = torch.allclose(state1[:, 0], state3[:, 0], atol=1e-6)
    print(f"reserved branch, k=1 vs k=3: {'identical (rule holds)' if reserved_same else 'CHANGED (rule broken)'}")
    other_differs = not torch.allclose(state1[:, 1], state3[:, 1], atol=1e-6)
    print(f"other branches, k=1 vs k=3:  {'differ, so passes do work' if other_differs else 'identical, so the loop does nothing'}")

    # The outputs below can coincide because a random model collapses to one repeated token. That is not
    # evidence against elastic depth; it is what argmax over random logits does. The branch check above
    # is the one that actually shows the mechanism.
    print(f"\nlogits at k=1 vs k=3 differ: {not torch.allclose(model.forward(seq, 1, slot, None, slots), model.forward(seq, 3, slot, None, slots), atol=1e-6)}")

    # The table is zero at initialization by rule: an untrained table is an empty table. So the memory
    # influence of an untrained model is exactly zero, and that is not a failure, it is the rule. Filling
    # it with random rows does not train anything; it only shows the read path is connected end to end.
    saved = model.memory.table.weight.clone()
    with torch.no_grad():
        model.memory.table.weight.normal_(0, 1)
        logits_on = model.forward(seq, 3, slot, None, slots)
        model.memory.table.weight.zero_()
        logits_off = model.forward(seq, 3, slot, None, slots)
        model.memory.table.weight.copy_(saved)
        influence = float((logits_on - logits_off).norm()) / float(logits_off.norm())
        gate = float(model.memory.read(slots)[1].mean())
    print(f"\nwith the table filled with random content, read influence on the output: {influence:.4f}")
    print(f"mean gate on this prompt: {gate:.4f}")
    print("The gate is what decides whether a recalled row is relevant, because a 1.5M-entry table")
    print("collides. A hit is not automatically a useful hit.")
    print("\nWhat Phase 0 gate G0.4 will decide is whether a trained table carries enough weight to")
    print("justify the mmap'd blob at all. If it does not, the memory path is dropped, and the rest of")
    print("the design does not depend on it.")


def show_metadata(cfg: HeronConfig, model: HeronLM) -> None:
    rule("7. GGUF metadata: what the fork needs that tensor shapes cannot express")
    # A synthetic tokenizer, because the report has no Qwen vocabulary on disk. It is well-formed: the
    # special tags sit at synthetic ids, so this exercises the real lookup rather than a hardcoded number.
    tokens = placeholder_vocabulary(cfg.vocab_size)
    md = metadata(cfg, tokens, ["merge a b"])

    print(f"keys: {len(md)}, all under general.arch.heron.* or tokenizer.*")
    for key in [
        "general.architecture",
        "general.arch.heron.loop.layers",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "general.arch.heron.loop.max_loop",
        "general.arch.heron.full_attention_layers",
        "general.arch.heron.window_size",
        "general.arch.heron.gr.read_weights_shape",
        "general.arch.heron.gr.branch_of",
        "general.arch.heron.expert_slot_layers",
        "general.arch.heron.table.read_block",
        "general.arch.heron.ngram.verify_only",
    ]:
        print(f"  {key:<48} {md[key]}")

    print("\nMetadata against the instantiated module, which is the check that keeps the converter from")
    print("loading tensors into the wrong blocks:")
    for key, value in [
        ("block_count", model.block_count),
        ("expert_slot_layers", model.expert_blocks),
        ("gr.branch_of", model.stream.branch_of),
    ]:
        print(f"  {key:<20} metadata {md['general.arch.heron.' + key]} == module {value}")
        assert md["general.arch.heron." + key] == value

    print("\nThese are the keys that carry real information rather than restating tensor shapes:")
    print("  - loop layers and max loop, because the converted file is one depth but the runtime knob")
    print("    is another, and nothing in the tensor layout says which blocks loop.")
    print("  - window size and full-attention layers, because a sliding-window block and a full block")
    print("    have identical tensors.")
    print("  - read weight shape, because it is indexed by pass, not by block.")
    print("  - branch assignment, because the memory read writes the reserved branch and the engine")
    print("    has to write the same one.")
    print("  - tokenizer tokens, because the table is keyed by token id: a tokenizer change silently")
    print("    invalidates every entry. The length check in metadata() is what catches that.")
    print("  - special-token ids and the chat template, resolved by name from the vocabulary rather than")
    print("    remembered. A literal id would pass on Qwen's file and tokenize a different model on any")
    print("    other. The template is Qwen's published text carried verbatim, with only the tag spellings")
    print("    substituted from the same constants, so the two cannot disagree.")
    print("  - byte fallback, the one tokenizer key the reference cannot decide on its own. See below.")

    print("\nByte fallback, the one tokenizer key the reference cannot decide from the vocabulary alone:")
    value, backed = byte_level_rule(tokens)
    print(f"  value written for this report: {value} <- {backed}")
    print("  gguf has no key constant for it and no writer helper for it, so nothing outside heronlm.metadata")
    print("  can say what the value should be, and a value written as a constant is a claim the reference has")
    print(f"  to be able to back. What a vocabulary can back is only the shape the rule needs: {BASE_ALPHABET_SIZE}")
    print("  distinct single-character symbols at the ids a byte-level reader reads bytes from, 0..255. Whether")
    print("  the tokenizer actually uses those positions is a property of its decoder, not of the symbol set.")
    print("  This report's fixture has fake strings at those ids, so the vocabulary answers for it, and the")
    print("  answer is no: there is no byte this vocabulary has a symbol for.")

    pinned = Path(__file__).resolve().parent.parent / "tokenizer" / "qwen2.5" / "tokenizer.json"
    if pinned.exists():
        model_part = json.loads(pinned.read_text(encoding="utf-8"))["model"]
        first = [None] * BASE_ALPHABET_SIZE
        for symbol, index in model_part["vocab"].items():
            if index < BASE_ALPHABET_SIZE:
                first[index] = symbol
        claim = model_part.get("byte_fallback")
        pinned_value, pinned_backed = byte_level_rule(first, claim)
        byte_table = {chr(256 + b) if (b < 33 or 127 <= b < 161) else chr(b) for b in range(256)}
        print(f"  the pinned file claims {claim} for itself, and its first {BASE_ALPHABET_SIZE} ids are one distinct")
        print(f"  single-character symbol each, so the vocabulary is byte-shaped ({base_alphabet(first) is not None})")
        print(f"  but cannot say the rule is used. The value written for the pinned tokenizer is {pinned_value} <-")
        print(f"  {pinned_backed}, and the divergence between the two is what the CLI prints rather than resolves.")
        print(f"  That shape is not the standard byte table either: {len(set(first) & byte_table)} of these symbols")
        print(f"  are shared with it and {len(set(first) - byte_table)} are extra, so a byte-to-symbol table is")
        print("  something the reference cannot check from a symbol set and therefore does not claim.")
    else:
        print("  the pinned tokenizer file is not on disk here, so its claim cannot be read, and this report can")
        print("  only state what the fixture's vocabulary backs.")

    print("\nNothing here is a ggml tensor. The hash table has no probe op, so it is an mmap'd blob")
    print("probed host-side; the metadata states its size, value width, hash and read block.")


def main() -> int:
    # Nothing here computes gradients; turning them off is what keeps the measured numbers readable and
    # stops torch from warning about converting tensors that carry grad into scalars.
    torch.set_grad_enabled(False)

    full = FULL()
    tiny = TINY()

    show_params(full)
    show_device_budget(full)

    torch.manual_seed(0)
    model = HeronLM(tiny).eval()
    held = sum(p.numel() for p in model.parameters())
    print(f"\nInstantiated TINY model: {held:,} params held. Only the routed expert computes per token;")
    print("the memory table is read rather than computed. Holding everything and computing everything")
    print("are different facts, which is why the report prints them separately.")

    table = show_ngram_table(tiny)
    state = show_routing(tiny, model)
    show_generation(tiny, model, table, state.slot)
    show_loop_and_memory(tiny, model, state.slot)
    show_metadata(tiny, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

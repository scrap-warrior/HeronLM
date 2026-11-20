"""
HeronLM reference implementation: looped core + swappable domain experts + n-gram memory.

This is the PyTorch reference the GGUF fork must match exactly, which is why the file is written
to be readable rather than fast. Every mechanism here has a parity test against the engine, and a
parity test is only possible if the reference defines the semantics unambiguously.

Structure, in the order the forward pass runs:

  1. embedding            token vector, tied with the output head
  2. input block          establishes the GR branches; memory read at block 2
  3. shared looped block  the elastic-depth part, run k times
  4. output block         expert slots live here, never inside the loop
  5. final read + head    logits from the tied embedding

Why experts sit outside the loop: putting them inside would multiply the two compute knobs (loop
count x expert size), so the phone would pay for specialization on every loop pass. Keeping them
in the output block means an expert costs exactly one sublayer per token.

Three param counts must be reported separately, and conflating them breaks every budget argument:
  - active   : what computes per token
  - resident : what the phone holds in RAM
  - capacity : what lives on storage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from heronlm.budget import param_counts
from heronlm.config import HeronConfig
from heronlm.gr import ResidualStream
from heronlm.layers import Block, Cache, ExpertSlot, GeneralistSlot
from heronlm.memory import NGramMemory, slot_ids
from heronlm.norms import Norm, normalized_weight
from heronlm.router import Router, RouterState


class HeronLM(nn.Module):
    def __init__(self, cfg: HeronConfig):
        super().__init__()
        self.cfg = cfg
        self.block_count = cfg.n_blocks()

        # Tied embedding. It is also the output head, which is why it is never quantized to Q4:
        # a quantized BPE table degrades badly, and a tied head degrades with it.
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.stream = ResidualStream(cfg, self.block_count)
        self.memory = NGramMemory(cfg)
        self.router = Router(cfg)
        # The head is the one place that is not a NormalizedLinear, because its weights are the tied
        # embedding. It has to apply the same two rules explicitly, otherwise the head would normalize
        # differently from every other sublayer and the engine would have no single rule to follow.
        self.norm = Norm(cfg)

        # An expert is a SET of slots, one at each expert block. That is why hot-swapping an
        # expert means loading several modules rather than one, and why the per-expert GGUF file
        # is the right storage unit.
        #
        # Slot 0, the generalist, is not a module forward() reads: it is each block's base FFN, which
        # is what makes it always resident and always available as the fallback by construction rather
        # than by convention. It is instantiated as `GeneralistSlot` so the harness sees it as a module -
        # a run that reports n_experts experts has a module for each - and that class holds its tensors
        # outside `state_dict()`, in both directions, because bytes at the generalist index have no name a
        # file can claim. The tensors that compute for slot 0 are the core file's `blk.{bid}.ffn_*`.
        # The block layout comes from config methods rather than from arithmetic repeated here, so the
        # metadata and the module cannot disagree about which block holds which tensors.
        self.expert_blocks = cfg.expert_blocks()
        self.experts = nn.ModuleList(
            [
                nn.ModuleList([
                    (GeneralistSlot(cfg, cfg.expert_ffn_width) if e == cfg.generalist_slot
                     else ExpertSlot(cfg, cfg.expert_ffn_width))
                    for _ in self.expert_blocks
                ])
                for e in range(cfg.n_experts)
            ]
        )

        self.blocks = nn.ModuleList()
        for i in range(self.block_count):
            # Full attention only in the output block, every 4th block. Inside the loop the mixer is
            # sliding window so cache cost stays bounded by the window instead of by context length.
            slots = None
            if i in self.expert_blocks:
                # Map each non-generalist expert to its slot at this block. The generalist is reached
                # through Block.ffn, not through this dict.
                slots = {e: self.experts[e][self.expert_blocks.index(i)] for e in range(1, cfg.n_experts)}
            self.blocks.append(Block(cfg, i, self.stream, mode=cfg.mixer_mode(i), slots=slots))

    def make_caches(self) -> dict:
        """
        Cache dict keyed by (block_index, loop_slot).

        Shared policy: one cache per block, so memory grows once per block regardless of k.
        Per-loop policy: one cache per block per pass, which is exact but is a memory multiplier.
        That difference is exactly what gate G0.5 measures, so both policies have to be
        representable in the reference.

        The bound is not the same for every block. A sliding-window cache is bounded by the window,
        so entries older than the window can be dropped once the model has moved past them. A
        full-attention cache has no bound, because a later query can still reach any earlier entry -
        passing 0 is how this reference says "unbounded", which Cache.evict reads as no eviction.
        """
        caches = {}
        loop_slots = range(self.cfg.max_loop) if self.cfg.cache_policy == "per_loop" else [0]
        for i in range(self.block_count):
            bound = self.cfg.swa_window if self.cfg.mixer_mode(i) == "swa" else 0
            for s in loop_slots:
                caches[(i, s)] = Cache(self.cfg.kv_heads, self.cfg.head_dim, bound, loop_slot=s)
        return caches

    def param_report(self) -> dict:
        """
        The three counts, from the same arithmetic the device budget is built on.

        Active, resident and capacity are reported separately because the design only makes sense when
        they are separate: active is what determines tokens/s, resident is what the phone can hold, and
        capacity is what can live on storage. Collapsing them into "model size" is what makes every
        budget argument wrong.

        This delegates to budget.param_counts so the arithmetic exists in one place. The instantiated
        totals are reported next to it; the difference between the two is exactly the per-block GR read
        weights and the reinjection gates, which the analytic count excludes on purpose.
        """
        counts = param_counts(self.cfg)
        instantiated_block = sum(p.numel() for p in self.blocks[0].parameters())
        instantiated_total = sum(p.numel() for p in self.parameters())
        # Slot 0 is not in `parameters()`: its tensors are non-persistent buffers, so a checkpoint can neither
        # carry them nor expect them. That is why `generalist_slot_tensors` is reported apart from `held` - it is
        # not capacity, nothing stores it and nothing trains it. The generalist that computes is the base FFN,
        # and those bytes are already inside `core_total`.
        generalist_tensors = sum(b.numel() for b in self.experts[self.cfg.generalist_slot].buffers())
        # `stored_total` is what a checkpoint actually carries, so it is read from `state_dict()` rather than
        # from `parameters()`: the two are not the same number. The shared read weight is one Parameter,
        # deduplicated by `named_parameters()` and counted once, but `state_dict()` writes it once per block as
        # eight aliased keys - which is exactly the form the converter writes as one row per block.
        stored_total = sum(t.numel() for t in self.state_dict().values())
        return {
            **counts,
            "instantiated_block": instantiated_block,
            "instantiated_total": instantiated_total,
            "generalist_slot_tensors": generalist_tensors,
            "stored_total": stored_total,
            # Params the model holds but that do not compute per token on the phone: only the routed
            # expert computes, and the memory table is read rather than computed. The generalist's tensors are
            # not in this number because they are not held in the storage sense at all.
            "held_but_not_computed": instantiated_total - counts["core_total"] - counts["expert_active"],
        }

    def forward(
        self,
        tokens: torch.Tensor,
        loop: int = 1,
        active_slot: int | None = None,
        caches: dict | None = None,
        slots_ids: torch.Tensor | None = None,
        position_offset: int = 0,
        return_pooled: bool = False,
        return_state: bool = False,
    ):
        """
        One forward pass over [B, T]. Returns logits, and optionally the pooled representation the
        router scores.

        Positions must be absolute, not 0..T-1, whenever a cache is in use. The sliding-window mask
        is a distance test, so a decode step needs to know where the cached entries sit, not merely
        how many there are. Using 0..T-1 during incremental decode is the most common way this kind
        of implementation quietly produces wrong numbers.

        The router reads pooled state from the last INPUT block, before the loop. It has to be taken
        there because the routing decision is made at conversation start: if the decision depended on
        loop passes, it would change when the loop count changed, which would make routing
        non-reproducible for the same conversation.
        """
        B, T = tokens.shape
        caches = self.make_caches() if caches is None else caches
        positions = torch.arange(position_offset, position_offset + T, device=tokens.device)

        x = self.embed(tokens)
        state = self.stream.init_state(x)

        # --- input block: sets up the branches, then the memory read ---
        for i in range(self.cfg.input_blocks):
            state = self.blocks[i].forward(state, caches, positions, 0, active_slot, reinjection=x)
            if i == self.cfg.ngram_memory_read_block and slots_ids is not None:
                # Branch 0 is the reserved branch: embedding plus memory. Writing here is the
                # exception to "only the first sublayer writes branch 0", and it is the exception
                # that lets the loop see the recalled value on every pass.
                state = self.memory.inject(state, slots_ids, branch=0)

        pooled = F.normalize(self.stream.read(state, self.cfg.input_blocks - 1, 0).mean(dim=1), dim=-1) if return_pooled else None

        # --- shared looped block, the elastic-depth part ---
        for k in range(loop):
            for i in range(self.cfg.input_blocks, self.cfg.input_blocks + self.cfg.loop_blocks):
                # The same weights compute on every pass, but the useful branch mixture differs
                # between pass 1 and pass 3, so the read mixture is indexed by pass.
                state = self.blocks[i].forward(state, caches, positions, k, active_slot, reinjection=x)

        # --- output block, where experts live ---
        for i in range(self.cfg.input_blocks + self.cfg.loop_blocks, self.block_count):
            state = self.blocks[i].forward(state, caches, positions, 0, active_slot, reinjection=x)

        read = self.stream.read(state, self.block_count - 1, 0)
        # The head applies the same two rules as every sublayer: normalize the input, normalize and
        # scale the weights. Writing them out here is what keeps the head from being the one place
        # where the engine has to guess.
        logits = F.linear(self.norm(read), normalized_weight(self.embed.weight, self.cfg))

        if return_state:
            # Exposed for the invariant test: the reserved branch must be identical at every loop
            # count, which is the property the memory read depends on. It cannot be observed from the
            # logits alone, so the state has to be reachable.
            return (logits, state) if not return_pooled else (logits, pooled, state)
        if return_pooled:
            return logits, pooled
        return logits

    def route(self, tokens: torch.Tensor, state: RouterState | None = None) -> RouterState:
        """Conversation-level decision, frozen for the conversation. No per-turn switching."""
        _, pooled = self.forward(tokens, return_pooled=True)
        return self.router.decide(pooled, state)[0]

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        loop: int = 1,
        active_slot: int | None = None,
        table=None,
        speculate: bool = False,
        temperature: float = 0.0,
    ) -> tuple[list[int], dict]:
        """
        Decode, with optional n-gram speculation.

        The rule that matters: n-gram proposals are VERIFIED, never blended. An unverified blend is
        how repetition loops happen, because the table proposes what it saw before and the model
        reinforces it. Here a wrong guess costs one forward pass and cannot change the output.

        Verification runs over the whole proposed span in one pass, which is what makes speculation
        worth having at all. The span is [last committed token] + proposals, so the logits at span
        position 0 are the distribution for the first proposal. Checking against argmax is greedy
        verification; accepting a proposal the core would not have chosen is exactly the blending the
        design forbids.

        Three details that are easy to get silently wrong:
          - the span is verified against a COPY of the caches. Verification must not pollute the real
            conversation state, because rejected proposals have to disappear.
          - the last committed token is rolled back in the copy before the span is replayed. If it
            stayed in the cache, replaying it would duplicate its entry and change the attention
            result, so the verification would no longer match what the core would have produced.
          - the copies do not evict. Eviction depends on which positions the model committed, and a
            span end is not a committed position, so a verification pass that evicts from the end of
            the span drops entries the sequential path would still have had. For a sliding-window
            cache that cannot change a result - an entry more than a window behind the newest position
            is outside the window at every later position as well - but for a full-attention cache it
            does, so the reference states the rule for both: verify without evicting, then evict at
            each committed position. Under that rule the verification logits match the sequential
            logits exactly, which is what makes the decode path able to only agree or be ignored.
        """
        caches = self.make_caches()
        out = list(prompt)
        stats = {"proposed": 0, "accepted": 0, "forward_passes": 0, "lookups": 0, "hits": 0, "overshoot": 0}

        # Prefill: one pass over the prompt, then use the last position's logits.
        logits = self.forward(
            torch.tensor([out]), loop, active_slot, caches, slot_ids(torch.tensor([out]), self.cfg)
        )
        stats["forward_passes"] += 1

        def next_token(position_logits: torch.Tensor) -> int:
            if temperature <= 0:
                return int(position_logits.argmax())
            p = F.softmax(position_logits / temperature, dim=-1)
            return int(torch.multinomial(p, 1).item())

        while len(out) - len(prompt) < max_new_tokens:
            hit = table.lookup(out) if (speculate and table is not None) else None
            if speculate:
                # Counted before the hit test, so lookups means attempts. If it only counted hits, the
                # hit rate would always read 100% and the miss rate - which is what a table's real cost
                # profile depends on - would be invisible.
                stats["lookups"] += 1
            if hit is not None:
                stats["hits"] += 1
                proposals = [tok for tok, _ in hit[1]]
                stats["proposed"] += len(proposals)

                # Verify the whole span in one pass, against copies. The copies are made with
                # eviction disabled, because which entries would be dropped depends on tokens that
                # may be rejected: the end of a span is not a committed position, and evicting from
                # it removes entries the sequential path would still have had. That is not a memory
                # detail - for a full-attention cache it changes what later tokens attend to, so the
                # decode path could change the model's behaviour through a guess it never used,
                # which is the blending the design forbids. Eviction happens below, once the accepted
                # prefix is known, and then at the same positions the sequential path would have.
                verify_caches = {key: c.clone(allow_evict=False) for key, c in caches.items()}
                for c in verify_caches.values():
                    c.pop_last()  # the last committed token is replayed as span position 0
                span = [out[-1]] + proposals
                span_tensor = torch.tensor([span])
                first_committed = len(out) - 1
                lg = self.forward(
                    span_tensor,
                    loop,
                    active_slot,
                    verify_caches,
                    slot_ids(span_tensor, self.cfg, prefix=[out[:-1]]),
                    position_offset=first_committed,
                )

                accepted = 0
                corrected = None
                for j, tok in enumerate(proposals):
                    pred = next_token(lg[0, j])
                    if pred != tok:
                        # Core disagrees: accept the prefix, discard the rest of the proposals, take
                        # the core's token. The corrected token was never replayed through the cache,
                        # so it has to be processed as a new token below.
                        corrected = pred
                        break
                    accepted += 1

                out.extend(proposals[:accepted])
                # Drop the rejected tail from the verified caches. Rejected proposals must not survive,
                # otherwise a wrong guess changes later behaviour, which is the blending the design
                # forbids. Then evict at each committed position, one at a time, which is the same rule
                # the sequential path applies: eviction after the attention for that token, measured
                # from that token's position.
                for c in verify_caches.values():
                    c.truncate(len(out))
                    for pos in range(first_committed, first_committed + accepted + 1):
                        c.evict(pos, pos)
                caches = verify_caches
                stats["accepted"] += accepted

                if corrected is not None:
                    out.append(corrected)
                    logits = self.forward(
                        torch.tensor([out[-1:]]),
                        loop,
                        active_slot,
                        caches,
                        slot_ids(torch.tensor([out[-1:]]), self.cfg, prefix=[out[:-1]]),
                        position_offset=len(out) - 1,
                    )
                    stats["forward_passes"] += 1
                else:
                    # Every proposal was confirmed, so the logits for the last committed position are
                    # already in hand and continuing from them costs nothing. That is where speculation
                    # actually pays for its cost.
                    logits = lg
                continue

            nxt = next_token(logits[0, -1])
            out.append(nxt)
            logits = self.forward(
                torch.tensor([out[-1:]]),
                loop,
                active_slot,
                caches,
                slot_ids(torch.tensor([out[-1:]]), self.cfg, prefix=[out[:-1]]),
                position_offset=len(out) - 1,
            )
            stats["forward_passes"] += 1

        # max_new_tokens is a promise about the length of the returned output, and a verified span can cross
        # it: the budget is checked once per loop step, not once per token inside a span, so a run that accepts
        # a long span can end up past the budget. The tokens past the budget are truncated here rather than
        # returned, because the alternative is that two runs of the same model on the same prompt return
        # different lengths and a harness comparing them reads a reporting artifact as a blended path.
        # The verification cost for those tokens was already paid, so the overshoot is recorded as its own
        # number rather than absorbed: it is a cost the engine really paid on the phone.
        emitted = len(out) - len(prompt)
        if emitted > max_new_tokens:
            stats["overshoot"] = emitted - max_new_tokens
            out = out[:len(prompt) + max_new_tokens]

        stats["hit_rate"] = stats["hits"] / stats["lookups"] if stats["lookups"] else 0.0
        stats["acceptance"] = stats["accepted"] / stats["proposed"] if stats["proposed"] else 0.0
        return out, stats

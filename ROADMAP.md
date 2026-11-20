# HeronLM Roadmap

Design: **looped core (~1.85B compute params) + 5 swappable domain experts (~0.5B each) + n-gram table**, Qwen tokenizer, target runtime GGUF on a Samsung S23.

One framing rule that must stay consistent across the whole project: **active params** (what computes per token), **resident params** (what the phone can hold in RAM), and **capacity** (what lives on storage) are three different numbers. Reporting them as one number makes every budget argument wrong.

---

## 1. Frozen architecture

| Item | Value |
|---|---|
| Tokenizer | Qwen's, unmodified: vocab, merges, byte-fallback, special tokens, chat template |
| `d_model` | 2048 |
| Blocks | 32 total: input 8 / shared looped 16 / output 8 |
| Heads / KV heads | 16 / 4 (GQA, `head_dim` = 128) |
| FFN | SwiGLU, width 6144 (3x `d`), per block ~48.2M params |
| Mixer inside the loop | SWA, window 2048 |
| Full attention | output block only, every 4th block |
| Normalization | nGPT-style: unit-norm activations, weight scaling, per-loop gates |
| Residual | GR-style widened read/write; layer-0 branch must stay readable through every loop |
| Reinjection | input embedding re-added on every loop iteration |
| Embedding | tied, quantized q8_0 (never Q4; doubles as the output head) |
| Cache | single shared cache across loops on device; exact per-loop cache is training-only |
| Loop count | runtime setting, `k = 1..4`, sampled per batch during training |
| Experts | 5, one designated always-resident generalist, conversation-level routing |
| Expert shape | 4 MoE slots in the output block, 125M params per slot, FFN width 20,345 |
| Routing | decided at conversation start from pooled hidden state vs 5 prototype vectors, hysteresis + confidence margin, pinning override |
| Table, decode path | static hash-count table, top-K = 4, `min_cnt >= 3` |
| Table, memory path | learned hash table, 1.5M entries, 256-dim int8 values, read at layer 2, contextual gate, injected through the GR branch |
| Speculation | n-gram proposals must be verified by the core; no unverified blending |

Per-token active params by loop count (looped portion = 16 x 48.2M = 771.8M):

| `k` | Active params | GFLOP/token |
|---|---|---|
| 1 | 1.854B | ~3.7 |
| 2 | 2.626B | ~5.3 |
| 3 | 3.398B | ~6.8 |
| 4 | 4.170B | ~8.3 |

Active expert adds 0.5B params / ~1 GFLOP per token when engaged, so +27% over the loop-1 core.

---

## 2. Device budgets (S23: 8 GB RAM, 256 GB storage, Snapdragon 8 Gen 2)

| Component | On disk (Q4_K_M) | Resident |
|---|---|---|
| Core dense, 1.54B (the 1.85B total minus the tied embedding) | ~0.95 GB | ~0.95 GB |
| Tied embedding/head, q8_0 | ~0.31 GB | part of the read path |
| One active expert | ~0.31 GB | ~0.31 GB |
| Decode table, 30-40M entries | ~600-900 MB | mmap + 128-250 MB LRU hot cache |
| Memory-path table, 1.5M x 256 int8 | ~400 MB | hot portion only |
| SWA cache at window 2048, shared across loops | - | ~130 MB (the window-bounded blocks; ~67 MB if only the looped blocks are cached) |
| Full-attention cache, per token | - | ~4 MB per 1000 tokens (2 blocks). Not a fixed line: it grows with the conversation, so it is reported as a rate and cannot be folded into the total |
| Tokenizer + runtime + app overhead | - | ~150-250 MB |
| **Total** | **~4 GB** | **~2.0-2.1 GB** |

Fits an 8 GB phone with room. Five experts total ~1.4 GB on disk, loaded one at a time. The total is the short-context figure, because the only cache that is bounded is the window-bounded one.

These figures are recomputed from the config by `heronlm/budget.py`, and `tests/test_reference.py` asserts the recomputation lands inside the ranges above. Two lines are counted differently than a single "model size" would suggest, and both are why the table is a table rather than a paragraph: the tied embedding cannot be Q4 (so the core line cannot be quoted as 1.85B at Q4_K_M), and the cache bound is per mixer mode rather than global — a sliding-window block is bounded by the window, a full-attention block is not, because a later query can still reach every entry it holds.

**Why storage is acceptable for the table:** the lookup is one probe per token, and the phone generates ~5-10 tokens/s. Random UFS reads at ~100-200 us therefore cost a fraction of a percent of the token budget. The binding limit is RAM for the hot cache, not storage size or lookup speed. This is the same reason Qwen's PLE lives in host RAM with async prefetch.

**Speed reality:** theoretical bandwidth ceiling ~10 tok/s at `k = 1`, realistic ~5-7. `k = 2` ~3-5, `k = 4` ~1.5-3. An n-gram speculative path at 25-35% acceptance recovers ~6-9 effective tokens/s. The phone is a device where you dial loop count up when you want to think and accept the slowdown, not a fast device.

---

## 3. Engine gap: what the fork must add

Existing, usable in llama.cpp today:

- Qwen tokenizer path and GGUF metadata for vocab/special tokens/chat template.
- Gated DeltaNet / hybrid attention for Qwen3-Next and Qwen3.5 architectures (relevant only if the mixer gate picks GDN).
- Speculative decoding, but with a **draft model** as the proposal source.
- GGUF is extensible via metadata keys, so a new architecture can be expressed without a format bump (follow the project's "adding new model architectures" guide: `llama-arch.cpp`, `constants.py`, `tensor_mapping.py`, `convert_hf_to_gguf.py`).

Missing, and this is the actual fork work:

1. **Hash-table lookup as an execution path.** There is no ggml op for a hash probe. Implement it host-side: probe an mmap'd blob, produce a small `[d]` tensor, feed it into the graph as an input. Do not try to make the table a normal ggml tensor.
2. **Loop count as a runtime knob**, not a fixed converted depth. Dory's support is a draft PR pinned branch, unmerged, and the GGUF encoding for this architecture can still change; base the converter on it rather than writing from scratch, but treat it as unstable.
3. **The GR branch**, if and only if the Phase 0 gate passes.
4. **Per-conversation expert slot with hot-swap**, plus reading expert weights from separate per-expert GGUF files.
5. **n-gram as a proposal source** for verification, which is a new input path rather than an existing speculative flag.
6. **Metadata keys**: loop layer list, window size, full-attention layer list, MoE slot positions, expert file paths, table dims/entry count/hash count, GR branch flag. Keep every key in a `general.arch.heron.*` namespace so upstream readers ignore them instead of failing.
7. **Android build**: CPU and Vulkan backends, ARM-specific kernel risk for whichever mixer wins, and the app-level loop setting plus routing decision.

---

## 4. Ordered build sequence

### Phase 0 - architecture gates at tiny scale (local card, ~2-3 weeks)

200-250M params, `d = 768`, 24 blocks, 1-2B tokens per arm, arms run sequentially. Deliverable is the PyTorch reference implementation that the fork must match exactly.

- **G0.1 Loop generalization.** Does loss still improve when `k` exceeds the trained depth? Pass: `k = 3` beats `k = 1` on held-out by a real margin, and `k` beyond the training distribution degrades by less than 0.02 nats. Fail means elastic depth is not real and the design changes fundamentally.
- **G0.2 nGPT vs plain LayerNorm residual stack under looping.** The normalized stack must beat the standard stack specifically at `k` above trained depth. If it does not, the loop is decorative.
- **G0.3 GR branch.** Measure per-branch write/read magnitudes per layer. Keep only if a branch shows the frozen-write pattern *and* removing it costs measurable loss. Otherwise drop it from v1: it is engine work with no evidence yet.
- **G0.4 Mixer.** Looped-SWA vs looped-GDN, measured **on the S23 build**, not on GPU. A 150M toy model in the fork on the phone is a one-day experiment; ARM GDN fusion is the weak point.
- **G0.5 Cache policy.** Shared-cache vs exact per-loop cache quality cost. Pass: shared cache loses under 0.05 nats at `k = 4`.
- **G0.6 Loop curriculum.** Confirm per-batch `k` sampling produces a model that is good at every `k` individually. A model trained only at `k = 1` will not behave at `k = 4`.

### Phase 1 - table gates (offline, cheap)

Build the static table from a corpus that is **provably disjoint from every eval set**, and report the overlap check. Pass thresholds: hit rate > 40% (`min_cnt >= 3`, top-K = 4), delta loss >= 0.02 nats, speculative acceptance >= 0.25. Also report the n-gram length distribution separately, and measure the memory path at 256-dim/int8. If the memory path does not survive reduced dims and quantization, ship the decode path only.

Guard: report loss on data disjoint from the table-build corpus. A large table can memorize benchmark answers, and measuring that as a table gain is self-deception. Measure verbatim recall (quote reproduction, needle-in-haystack) as a separate axis, because that is the table's actual job.

### Phase 2 - router gates

Derive the five specializations by clustering the teacher's router activations on a domain-tagged sample, rather than guessing domains. That gives the router's training target for free.

Corpus: 500-1,000 domain-tagged conversations. Pass: conversation-level routing accuracy >= 85%, turn-to-turn flapping < 10%, and the generalist is not worse than any expert on out-of-domain text. Fix the hysteresis and confidence-margin values here, not later.

Constraint that is easy to miss: a GDN state at any position is a function of every prior token, so it cannot be rewound. If routing changes mid-conversation, the only correct fix is re-prefilling the conversation under the new expert. Hence routing at conversation start, per conversation freeze, pinning override.

### Phase 3 - the 1.85B run (rental; local card is not enough)

A 2B student needs ~4 GB weights plus ~16 GB optimizer state plus activations. That is over a 12 GB card, so the local card is only good for Phase 0 arms.

Order within the run:

1. Pretrain bulk at `k = 1` (cheap), on a data budget chosen against a loss curve, not a fixed token count.
2. Loop-expansion phase at `k = 2-4` on a smaller, higher-quality slice. This is where elastic depth actually gets trained.
3. KD. Use **top-K logits (K = 64)** on a curated 20-40M-token slice — 384 B/token, so 20M tokens is ~7.7 GB, which is fine on 256 GB — plus response-KD on the rest. Teacher generations from your own hardware are bandwidth-bound at ~1-3 tok/s (Qwen3.6-35B is ~19 GB at Q4; the 3B active makes compute cheap but not reads), which is ~1-2M tokens a month: fine for evals, useless for a large corpus. Rent for generation; it is cheap.
4. Per-expert continued training on the domain slices.
5. Router training.

Do not plan feature-matching KD on your own hardware: it requires teacher and student in the same process, and teacher hidden states are ~300 KB/token, which is hundreds of GB for any meaningful corpus. Top-K logit KD is the only KD that survives being split across machines. If you want white-box work locally, do it at Phase 0 scale where the teacher fits in 12 GB.

### Phase 4 - the fork, in dependency order

1. GGUF metadata extensions plus the converter class, copying what Dory's converter already encodes for loop info.
2. nGPT normalization and loop `k = 1-4`, based on Dory's branch.
3. SWA hybrid, with the shared-cache policy. Two cache rules the fork has to reproduce, both derived from the reference:
   - the bound is per mixer mode, not global. A sliding-window block is bounded by the window; a full-attention block is not, because a later query can still reach every entry it holds. Trimming every cache to the window produces an engine that diverges from the reference once the conversation is longer than the window, and nothing in the output says so.
   - verification never evicts from an uncommitted position. The n-gram span is verified against copies, and eviction is a decision about positions the model committed, so a verification pass must not evict. Eviction happens after the accepted prefix is known, at each committed position, one at a time. Under that rule the verification logits match the sequential logits exactly, which is what makes the proposal path able to agree or be ignored, never to blend.
4. Hash-table tensor/sidecar blob, host-side lookup, gate.
5. Per-conversation expert slot, hot-swappable per-expert files.
6. Speculative path with n-gram as the proposal source.
7. GR, only if Phase 0 passed.

Acceptance test at each step: bit-exact parity with the PyTorch reference on the same prompt at `k = 1`, then per-loop parity, then table parity, then routing parity. Parity tests are the only thing that keeps a fork like this debuggable.

### Phase 5 - on-device eval at Q4

Re-measure **every gate after quantization**, on the phone. Small gains measured in BF16 can vanish at Q4, and embedding-table-like components quantize differently from dense matrices. Report per feature, on the same prompt set, with baselines: no-table, no-loop, no-expert. Metrics: tokens/s at each `k`, RAM peak, cold expert-load time, table hit rate, routing accuracy, and the table's verbatim-recall gain.

### Phase 6 - packaging

Quant tiers for the phone: core Q4_K_M, embedding/head q8_0, table int8/static, cache policy fixed to shared. File layout: one GGUF for the core, one GGUF per expert, table as sidecar or as GGUF tensors. Android integration: build with CPU + Vulkan, expose loop count as a user setting, routing decision at conversation start.

---

## 5. Risks

- **Format churn.** Dory support is an unmerged PR; the GGUF encoding for looped architectures can change before it lands. Anything you build on it is pinned to a branch and a commit.
- **ARM kernels.** The mixer choice depends on performance that has not been measured on the actual device.
- **Table quantization.** A table gain proven at BF16 on GPU is not automatically a gain at int8 on a phone.
- **Quantized routing degradation.** A quantized router can lose enough precision to change which expert fires. Measure routing accuracy on the quantized model, not only on BF16.
- **Memory ceiling creep.** Every feature that adds resident memory (hot cache, second expert, larger table) competes with context length. Pick one priority per tier: 4k context plus a big table, or 8k context plus a small table.
- **Data starvation.** A hobby-scale token budget will not reproduce leaderboard numbers. The value of this project is the measured per-feature contribution, so the eval harness has to be able to show it.

---

## 6. Open questions

1. Which domains the five experts serve — derived from teacher router clustering, or chosen by hand first and then verified.
2. Whether the memory path ships at all, or only the decode path. Phase 1 decides it.
3. Token budget for the 1.85B run, which is the main cost driver.
4. Whether the fork targets upstream llama.cpp contribution or stays a personal branch. Contribution upstream is only realistic after Phase 4 step 4 exists, because the hash path is the genuinely new part.

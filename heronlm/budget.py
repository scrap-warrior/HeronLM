"""
Parameter and memory budget, computed from the config rather than copied from the roadmap table.

The reason this is a module and not a paragraph: active, resident and capacity have to stay separate,
and the only way to keep them separate is to make each one a named quantity rather than a single
"model size".

  active   - what computes per token. This is what determines tokens/s on the phone.
  resident - what the phone must hold in RAM at once.
  capacity - what can live on storage, because the access pattern is one probe per token.

Only the looped blocks multiply with k. That is the whole reason the loop is a cheap knob: a wider
model would pay its whole stack on every pass, while a looped model pays only the looped part.

Units are decimal GB (10^9 bytes), because that is what the roadmap table uses. Quantization bits are
effective bits per weight including block overhead, not the nominal name: Q8_0 is 33 bytes per 32
weights, which is 8.03 bits rather than 8. Rounding that away is what makes a budget claim
unverifiable.
"""

from heronlm.config import HeronConfig

# Effective bits per weight, including quantization block overhead.
Q4_K_M = 4.9   # 4.25 nominal + superblock scales
Q8_0 = 8.03    # 33 bytes per 32 weights
INT8 = 8.0


def attention_params(cfg: HeronConfig) -> int:
    """q, k, v, o. No biases: nGPT's parametrization carries the scale, not the matrices."""
    q = cfg.d_model * cfg.heads * cfg.head_dim
    kv = 2 * cfg.d_model * cfg.kv_heads * cfg.head_dim
    o = cfg.heads * cfg.head_dim * cfg.d_model
    return q + kv + o


def ffn_params(cfg: HeronConfig, width: int) -> int:
    """SwiGLU: gate, up, down."""
    return 3 * cfg.d_model * width


def per_block_params(cfg: HeronConfig) -> int:
    """A plain block, i.e. the block the loop re-executes on every pass."""
    return attention_params(cfg) + ffn_params(cfg, cfg.ffn_width)


def embedding_params(cfg: HeronConfig) -> int:
    """Tied embedding, which is also the output head. Counted once, not twice."""
    return cfg.vocab_size * cfg.d_model


def param_counts(cfg: HeronConfig) -> dict:
    """
    The three counts, from config arithmetic.

    Excluded on purpose: the GR read weights and the reinjection gates. They are real parameters but
    tiny, and including them would make the analytic number differ from an instantiated model for
    reasons that have nothing to do with the architecture. The test that compares the two therefore
    asserts the difference is exactly those and nothing else.
    """
    per_block = per_block_params(cfg)
    embed = embedding_params(cfg)
    core = embed + cfg.n_blocks() * per_block

    # Only the looped part is paid k times per token.
    looped = cfg.loop_blocks * per_block
    paid_once = embed + (cfg.input_blocks + cfg.output_blocks) * per_block

    return {
        "per_block": per_block,
        "embedding": embed,
        "core_total": core,
        "looped_part": looped,
        "paid_once": paid_once,
        "active_at_k": {k: paid_once + k * looped for k in range(1, cfg.max_loop + 1)},
        "expert_slot": ffn_params(cfg, cfg.expert_ffn_width),
        "expert_active": cfg.expert_slots * ffn_params(cfg, cfg.expert_ffn_width),
        "memory_table": cfg.ngram_memory_entries * cfg.ngram_memory_dim,
    }


def device_budget(cfg: HeronConfig) -> dict:
    """
    What the phone actually holds, tier by tier.

    Two things a budget table usually hides, stated here because they are load-bearing:

      - experts are on disk for all five and in RAM for one. Routing loads exactly one, so the resident
        line is one expert, not the sum.
      - the memory table is on disk in full and resident only in its hot portion. The access pattern is
        one probe per token at 5-10 tokens/s, so storage latency is irrelevant; RAM is what limits it.
    """
    counts = param_counts(cfg)
    gbit = 1.0 / 8.0 / 1000.0**3  # effective bits per weight -> decimal GB

    core_dense = counts["core_total"] - counts["embedding"]
    core_dense_bytes = core_dense * Q4_K_M * gbit
    embedding_bytes = counts["embedding"] * Q8_0 * gbit

    # The embedding is q8_0 by rule, never Q4, so the core file cannot be quoted as though the whole
    # core were quantized the same way. That is why this module reports dense and embedding separately.
    core_total_bytes = core_dense_bytes + embedding_bytes
    one_expert = counts["expert_active"] * Q4_K_M * gbit
    all_experts = cfg.n_experts * one_expert

    # Decode table: a 4-byte key hash plus top_k continuations, each a 3-byte token id plus a confidence
    # byte. Entry count is the cap, so this is the worst case rather than an average.
    decode_table = cfg.ngram_max_entries * (4 + cfg.ngram_top_k * 4) / 1000.0**3
    memory_table = counts["memory_table"] * INT8 * gbit

    # Cache, split by mixer mode, because the two kinds have different memory behaviour and that difference
    # is what makes the phone budget work at all. The sliding-window blocks are bounded by the window, so
    # this line does not grow with conversation length. The full-attention blocks are not bounded: a later
    # query can still reach every entry they hold, so their cache grows with the context and cannot be
    # reported as a flat line. The roadmap's 50-100 MB figure is the window-bounded part, which is the
    # honest figure for the looped blocks; the growing part is reported as a rate, because the token budget
    # for the 1.85B run is still an open question and a total would be a chosen context presented as a fact.
    bounded_blocks = sum(1 for i in range(cfg.n_blocks()) if cfg.mixer_mode(i) == "swa")
    unbounded_blocks = cfg.n_blocks() - bounded_blocks
    bytes_per_cached_position = cfg.kv_heads * cfg.head_dim * 2 * 2
    swa_cache = bounded_blocks * bytes_per_cached_position * cfg.swa_window / 1000.0**3
    full_attention_growth = unbounded_blocks * bytes_per_cached_position / 1000.0**3

    hot_cache = 0.20  # 128-250 MB LRU; the upper end is the honest worst case
    overhead = 0.20   # tokenizer, runtime, app

    return {
        "disk": {
            "core_dense_q4_k_m": core_dense_bytes,
            "embedding_head_q8_0": embedding_bytes,
            "core_total": core_total_bytes,
            "decode_table": decode_table,
            "memory_table_int8": memory_table,
            "experts_all_five": all_experts,
            "total": core_total_bytes + decode_table + memory_table + all_experts,
        },
        "resident": {
            "core": core_total_bytes,
            "one_active_expert": one_expert,
            "decode_table_hot_cache": hot_cache,
            # The flat, window-bounded part. The growing part is not a fixed number, so it is reported as a
            # rate next to the total rather than folded into it; the total is therefore the resident cost at
            # a conversation no longer than the window, which is the case the roadmap's table describes.
            "swa_cache": swa_cache,
            "full_attention_cache_growth_per_token": full_attention_growth,
            "runtime_overhead": overhead,
            "total": core_total_bytes + one_expert + hot_cache + swa_cache + overhead,
        },
    }

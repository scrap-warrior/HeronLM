"""
HeronLM configuration.

Every number here is a design decision, not a default. The comments say why, so
this file reads like the architecture note rather than like a settings file.

Three configs exist:
  FULL   - the real target model, about 1.85B compute params
  PHASE0 - the gate scale named in the roadmap: d=768, 24 blocks, 200-250M params. Needs a GPU.
  TINY   - a miniature with the same structure, used for tests and for running the Phase 0 arms
           on a laptop. Shapes scale down, mechanisms do not change.
"""

from dataclasses import dataclass

# RoPE frequency base. It is a module constant rather than a config field because the value is written
# twice: `layers.build_rope` uses it to build the angles, and the GGUF metadata records it for the fork.
# A config field with the same number written inside build_rope would be a second statement of the same
# fact, which is exactly the kind of drift a parity test cannot catch until the numbers disagree.
ROPE_FREQ_BASE = 10000.0


@dataclass
class HeronConfig:
    # --- Tokenizer -------------------------------------------------------------
    # Qwen's tokenizer, taken as-is. Vocab size must match exactly because the
    # n-gram table is keyed by token id: if the tokenizer ever changes, every
    # table built with it has to be rebuilt. That is the one hard coupling in
    # the whole design, and it is why the tokenizer is not tuned or shrunk.
    vocab_size: int = 151646

    # --- Width and depth -----------------------------------------------------
    # d = 2048 with depth reached through the loop knob, not through stacking.
    # On a phone the bottleneck is memory bandwidth plus a long chain of
    # sequential dependent ops per token. Fewer, wider blocks means a shorter
    # chain for the same parameter count, so width wins for the mobile target.
    d_model: int = 2048

    # Three-part split. The input block establishes the residual branches, the
    # shared block is the part that loops, the output block is where experts and
    # full attention live. Keeping experts outside the loop stops the two compute
    # knobs (loop count, expert choice) from multiplying each other.
    input_blocks: int = 8
    loop_blocks: int = 16
    output_blocks: int = 8

    # GQA: 16 query heads reading from 4 KV heads. This is what keeps the cache
    # small enough for a phone at 4-8k context.
    heads: int = 16
    kv_heads: int = 4
    head_dim: int = 128

    # SwiGLU width. 6144 gives about 48.2M params per block, which is what makes
    # the 32-block core land near 1.85B once the tied embedding is added.
    ffn_width: int = 6144

    # --- Attention -------------------------------------------------------------
    # Sliding window inside the loop, full attention only in the output block.
    # SWA keeps cache growth bounded by the window instead of by context length,
    # and unlike a recurrent state it is append-only, so the cache is reusable
    # across turns. A GDN state is a function of every prior token and cannot be
    # rewound, which is what makes mid-conversation expert switching expensive.
    swa_window: int = 2048
    full_attention_every: int = 4  # in the output block

    # --- Normalization ---------------------------------------------------------
    # nGPT-style: activations are normalized to unit norm and the weights carry a
    # scale. Plain LayerNorm residual stacks degrade when looped beyond the depth
    # they were trained at; the normalized parametrization is what makes looping
    # past trained depth work at all. This is not an optional garnish on top of
    # the loop, it is the reason the loop generalizes.
    ngpt: bool = True
    ngpt_weight_scale: float = 0.5  # nGPT alpha, applied at forward time

    # Normalization arm, for gate G0.2. "layernorm" is the plain standardized stack that the nGPT
    # stack is compared against under looping.
    #
    # Both arms are parameter-count-neutral: the LayerNorm arm has no affine gain, so the gain stays in
    # ngpt_weight_scale for both. If one arm carried extra parameters, the two would differ in size as
    # well as in the rule being tested, and the param budget that the device numbers are built on would
    # no longer apply to both arms. This is why the arm is a mode and not a module with a gain.
    norm_mode: str = "ngpt"  # "ngpt" or "layernorm"

    # GR-style widened residual: multiple branches, learned read weights, and each
    # sublayer writes to one branch. The point for this design is that the branch
    # written at layer 0 is barely updated afterwards, so the n-gram vector
    # injected at layer 2 is still readable at every loop iteration.
    gr_branches: int = 4

    # GR read on/off. Unit normalization already makes sublayer inputs comparable, so a
    # GR read on top of it can be redundant. Phase 0 gate G0.3 decides: keep only if the
    # branches carry information that a plain stream loses. When False the branches
    # collapse to one stream and the model is a plain nGPT looped stack.
    ngpt_gr_read: bool = True

    # --- Loop ------------------------------------------------------------------
    # Trained depth is 32 blocks (k=1). The loop knob is runtime elastic depth.
    # Loop count is sampled per batch during training; without that sampling the
    # model only knows the depth it saw and elastic depth does not exist.
    max_loop: int = 4

    # Cache policy. Shared cache across loops is the mobile default because exact
    # per-loop cache is a memory multiplier. It is an approximation, and Phase 0
    # gate G0.5 measures what that approximation costs in loss.
    cache_policy: str = "shared"  # "shared" or "per_loop"

    # --- Experts ---------------------------------------------------------------
    # 5 experts, one of which is the generalist and is always resident. An expert
    # is 4 MoE slots in the output block, so its params are spread across layers
    # instead of dumped in one place.
    n_experts: int = 5
    generalist_slot: int = 0  # slot index that is always loaded
    expert_slots: int = 4
    expert_ffn_width: int = 20345  # 125M params per slot at d_model = 2048

    # Routing happens per conversation, at conversation start. A quantized router
    # can flip which expert fires, so routing accuracy is measured on the quantized
    # model, not only in BF16.
    router_temperature: float = 0.1
    router_margin: float = 0.08       # hysteresis margin before switching
    router_hysteresis_turns: int = 2  # turns a switch must be confident for

    # --- N-gram table ----------------------------------------------------------
    # Decode path: static hash-count table, storage-backed. On a phone the lookup
    # is one probe per token and generation is ~5-10 tokens/s, so UFS random reads
    # at 100-200 us cost essentially nothing. Storage is not the constraint; the
    # RAM hot cache is.
    ngram_orders: tuple = (4, 3, 2, 1)
    ngram_top_k: int = 4
    ngram_min_cnt: int = 3
    ngram_max_entries: int = 40_000_000

    # Memory path: the reduced version of the per-layer-embeddings idea. Qwen's
    # full table is parametric and sized V x d, which is tens of GB. The phone
    # version is 1.5M entries x 256 dims in int8, read at one shallow layer and
    # gated into the residual stream. Shallow placement is deliberate: the read can
    # overlap with the previous block, and the value survives the loop through the
    # GR branch.
    ngram_memory_entries: int = 1_500_000
    ngram_memory_dim: int = 256
    ngram_memory_read_block: int = 2

    # The table itself is not weight-decayed and not quantized with imatrix; only
    # its key/value projections go through Muon. Sparse row updates need different
    # optimizer treatment than dense matrices.
    ngram_optimizer: str = "adam_no_wd"  # table rows are not decayed; only their projections are

    def n_blocks(self) -> int:
        """
        Total block count.

        A method rather than an addition written at each use site, because the same sum appears in the
        model layout, the parameter budget, and the GGUF metadata. When the arithmetic is written three
        times, one of them drifts, and the drift shows up as a converter reading tensors from the wrong
        block.
        """
        return self.input_blocks + self.loop_blocks + self.output_blocks

    def output_start_block(self) -> int:
        """
        The first block of the output block.

        Two behaviour facts are measured from this index rather than from block 0: where an expert slot can be, and
        where the full-attention interval starts counting. Both are written against the output block, not against
        the model, so the same number is needed in both places, and it is the number a loader reading
        `layer_pattern` needs as well.
        """
        return self.input_blocks + self.loop_blocks

    def expert_blocks(self) -> list[int]:
        """
        Absolute block indices that hold expert slots.

        Only the output block can hold them, so the count is capped by output_blocks rather than by
        expert_slots alone. If the two disagreed, the engine would look for slots in blocks that never
        compute through an expert.
        """
        start = self.output_start_block()
        return list(range(start, start + min(self.expert_slots, self.output_blocks)))

    def mixer_mode(self, block: int) -> str:
        """
        Full or sliding window, per block.

        Not inferable from tensor shapes: a full-attention block and a sliding-window block have the same
        tensors. The behaviour is data, so it has to be stated somewhere, and stating it once here is what
        keeps the reference and the metadata describing the same model.
        """
        start = self.output_start_block()
        if block >= start and (block - start) % self.full_attention_every == 0:
            return "full"
        return "swa"

    # --- Training --------------------------------------------------------------
    # Bulk pretraining runs at k=1 because uniform k in 1..4 costs about 2.5x.
    # Elastic depth is trained in a separate loop-expansion phase on a smaller,
    # higher-quality slice.
    loop_curriculum_bulk: int = 1
    loop_curriculum_expand: tuple = (2, 3, 4)


def FULL() -> HeronConfig:
    """The target model. Every value is the frozen architecture, not a default."""
    return HeronConfig()


def PHASE0() -> HeronConfig:
    """
    The gate scale named in the roadmap: d = 768, 24 blocks, 200-250M params, 1-2B tokens per arm.

    Head count scales down with width, because head_dim 128 at d = 768 would make the q projection
    wider than the model itself. Eight query heads over two KV heads keeps the GQA shape of the target
    model at gate scale.

    The block split keeps the same three-part shape as FULL, because the gates are about the loop, and
    the loop is a third of the model. If the split changed shape between the gate scale and the target,
    a pass at gate scale would not say anything about the target.

    This is the scale the gates are decided at, and it needs a GPU. The torch build in this environment
    is CPU-only, so the harness runs at TINY scale here and at this scale on a card. Same arms, same
    gates, same thresholds; only the size changes.
    """
    return HeronConfig(
        d_model=768,
        input_blocks=6,
        loop_blocks=12,
        output_blocks=6,
        heads=8,
        kv_heads=2,
        head_dim=64,
        ffn_width=2048,
        swa_window=256,
        full_attention_every=2,
        expert_ffn_width=3072,
        ngram_memory_entries=200_000,
        ngram_memory_read_block=2,
    )


def TINY() -> HeronConfig:
    """Same mechanisms, small enough to run on a laptop in a few minutes."""
    return HeronConfig(
        vocab_size=512,
        d_model=128,
        input_blocks=2,
        loop_blocks=4,
        output_blocks=2,
        heads=4,
        kv_heads=1,
        head_dim=32,
        ffn_width=384,
        swa_window=32,
        full_attention_every=2,
        gr_branches=3,
        max_loop=3,
        n_experts=3,
        expert_slots=2,
        expert_ffn_width=1024,
        ngram_memory_entries=1024,
        ngram_memory_dim=64,
        ngram_memory_read_block=1,
    )

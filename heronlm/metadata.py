"""
GGUF metadata for the HeronLM architecture.

The fork does not need a format bump: GGUF is typed key-value metadata, so a new architecture is
expressed by adding keys. Everything here lives under `general.arch.heron.*` so that an upstream
reader that does not know the architecture ignores these keys instead of failing on them. That is the
difference between a personal branch and something that can eventually be contributed upstream.

What has to be encoded, and why each item is not derivable from the weights alone:

  - the loop layer list: which blocks are shared across passes. The converter could not infer this
    from tensor shapes, because a looped block has the same tensors as an unlooped one.
  - the sliding-window size and the full-attention layer list: the same shapes, different behaviour.
  - the per-loop read weights: a [n_blocks, max_loop, branches] tensor, which the engine must know is a
    per-pass mixture rather than a normal parameter.
  - the expert slot positions and the per-expert file paths, because experts are loaded one at a time
    from separate files rather than from the core file.
  - the table dimensions, entry count and hash count, plus the read block, because the table is not a
    ggml tensor at all: there is no hash-probe op. It is an mmap'd blob probed host-side, and the graph
    receives the resulting vector as an input. The metadata is what tells the engine where to put it.
    The router prototypes are not like the table: they are ordinary weights, so they can travel as a
    tensor. The distinction is which component has an op that can consume it, not which one is read
    host-side - both are.
  - the tensor names: the mapping from the reference's module paths to the names a loader looks up. A
    shape cannot say which block a tensor belongs to, so the mapping is recorded as data, generated from
    the same dict the converter looks up. Names that the pinned Dory branch already has are spelled as
    that branch spells them; names for mechanisms it does not have are spelled here, and say so.
  - the plain shape facts a generic reader looks for (feed-forward width, context length, norm epsilon,
    the RoPE base). They are recorded even though they are derivable from this module's config, because a
    reader that ignores the `heron` keys still has to allocate the right shapes, and a reader that cannot
    allocate them cannot ignore anything.
  - the special-token ids and the chat template, because both are properties of the tokenizer file rather
    than of the architecture. The ids are resolved from the vocabulary passed in, not remembered. The
    template is Qwen's published text carried verbatim, with only the two tag spellings substituted from the
    same constants the ids use, so template and vocabulary cannot describe different tokenizers.
  - byte fallback, which is the one tokenizer key that is not a fact the reference can derive: see
    `byte_level_rule`. It is recorded from the tokenizer file's claim, and only when the vocabulary can back
    it. The pinned Qwen file states `byte_fallback: false`, so the file that is written for it states false.

    Two things a reader might expect to find here are not: the file type and the GGUF version. Both are
    facts about the file the writer produces, not about the architecture, and the writer records them from
    what it actually did. A metadata dict that states a file type is a claim about a file it has not
    written yet, and claims of that kind are exactly what drift.

The GR branch flag is included even though gate G0.3 may drop it, so the metadata stays honest about
what the checkpoint actually contains rather than what the design hoped it would contain.
"""

from heronlm.config import HeronConfig, ROPE_FREQ_BASE
from heronlm.gr import branch_assignment
from heronlm.norms import EPS, PARITY_RULE


# Qwen's ChatML tags, spelled from their two halves. The literal never appears in this file, because a
# literal is indistinguishable from an assumption: the vocabulary is the authority on what a tag is and
# where it sits, and both the ids and the chat template below are derived from these two strings. Keeping
# one spelling for both is what makes it impossible for the id map and the template to disagree.
IM_START_TAG = "<|" + "im_start|>"
IM_END_TAG = "<|" + "im_end|>"
END_OF_TEXT_TAG = "<|" + "endoftext|>"

# The tags the metadata cannot be written without. The order is irrelevant; the names are what error
# messages use. The extra Qwen tags above the vocabulary (repo/file separators) are not required, because
# a chat turn never produces them and their ids are not recorded in GGUF.
SPECIAL_TAGS = {
    "im_start": IM_START_TAG,
    "im_end": IM_END_TAG,
    "endoftext": END_OF_TEXT_TAG,
}


def special_token_ids(tokens: list[str]) -> dict[str, int]:
    """
    Resolve the special-token ids by name from the vocabulary that will actually be written.

    The ids are data, not constants. Qwen's published vocabulary happens to place these tokens at the tail,
    but the reason to read them from the vocabulary rather than remember a number is that a re-export, a
    trimmed vocabulary, or a test fixture moves them. A hardcoded id passes every check on the reference
    vocabulary and silently tokenizes a different model on any other one - and because the n-gram table is
    keyed by token id, that failure is invisible in the output rather than loud.

    Duplicates are rejected instead of resolved to the first match: if the same tag appears twice, the id is
    ambiguous, and choosing one would hide a broken export behind plausible behaviour.
    """
    ids: dict[str, int] = {}
    for name, tag in SPECIAL_TAGS.items():
        found = [i for i, token in enumerate(tokens) if token == tag]
        if not found:
            raise ValueError(
                f"vocabulary has no {name} token ({tag!r}); it is not the tokenizer this metadata is for"
            )
        if len(found) > 1:
            raise ValueError(
                f"vocabulary has {len(found)} copies of the {name} token at {found}; its id would be ambiguous"
            )
        ids[name] = found[0]
    return ids


# Qwen2.5-1.5B-Instruct's published chat template, carried verbatim.
#
# This is data from the pinned tokenizer_config.json, not a reimplementation. The only edit is that the two
# tag spellings are replaced by placeholders and substituted from IM_START_TAG/IM_END_TAG at call time, so the
# tag text still has exactly one source in this module. tests/test_reference.py substitutes them back and
# asserts that the result equals the published 2,507-character template byte for byte.
#
# Why the whole template rather than the ChatML subset that was here before: ROADMAP line 13 requires Qwen's
# chat template unmodified. The subset covered the ordinary turn shape but silently differed for a missing
# system message, a non-first system message, tool calls, tool responses and grouped tool results - all of
# which the published template defines. A template that is "almost" Qwen's renders some conversations into a
# different token stream than the parity target, and nothing in the output would say so.
#
# The string starts at column zero because its indentation is part of the template. The \n spellings inside
# the Jinja string literals are two-character escapes (raw string), while the line breaks between Jinja
# statements are real newlines; both are as published. Do not reformat this block.
_QWEN_CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '@IM_START@system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>@IM_END@\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '@IM_START@system\n' + messages[0]['content'] + '@IM_END@\n' }}
    {%- else %}
        {{- '@IM_START@system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.@IM_END@\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '@IM_START@' + message.role + '\n' + message.content + '@IM_END@' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '@IM_START@' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '@IM_END@\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '@IM_START@user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '@IM_END@\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '@IM_START@assistant\n' }}
{%- endif %}
"""


def chat_template() -> str:
    """
    Qwen2.5's published chat template, byte for byte.

    The text above comes from the pinned `tokenizer/qwen2.5/tokenizer_config.instruct.json` (Qwen2.5-1.5B-
    Instruct; its SHA-256 is recorded in that directory's README). It is not a hand-written approximation:
    the test suite substitutes the two placeholders back and compares the result with the pinned file.

    The placeholders exist so the tag text still comes from IM_START_TAG/IM_END_TAG. Without them the template
    would be a second, independent statement of what the special tokens are, and a template that disagreed
    with the vocabulary would render a chat turn into ids the n-gram table was never built with.

    This is the full template, so the engine's renderer has to support what it uses: Jinja whitespace
    control, conditionals, loops, `set`, attribute access and the `tojson` filter. The previous single-line
    ChatML subset was easier to parse, but it was not Qwen's template, and ROADMAP line 13 requires the real
    one. The default system message is Qwen's and is used only when the caller supplies no system message.

    The template names `<tool_call>` and `<tool_response>` as text. The frozen vocabulary ends before Qwen's
    added tool-call tokens, so those are encoded as ordinary text rather than as single token ids; ordinary
    system/user/assistant turns use only the two special tags the vocabulary does contain.
    """
    return _QWEN_CHAT_TEMPLATE.replace("@IM_START@", IM_START_TAG).replace("@IM_END@", IM_END_TAG)


def placeholder_vocabulary(vocab_size: int) -> list[str]:
    """
    A synthetic vocabulary for the drivers and for tests that are not about the tokenizer.

    This is NOT Qwen's vocabulary and must never be written into a checkpoint. The fillers are fake strings
    and the special tokens sit at synthetic ids, one at the tail of the list to mirror Qwen's layout. It
    exists so that every caller of metadata() supplies a vocabulary whose special tokens can be resolved,
    exactly as a real tokenizer would - which is what stops hardcoded ids from creeping back in through test
    fixtures that never exercise the lookup at all.

    Its entries are not single characters, so it is not byte-shaped and cannot back a claim of byte fallback.
    That is the honest answer for a fixture whose tokens are fake strings: `byte_level_rule` reports False for
    it, and a test that wanted a byte-shaped fixture would have to build one from real single-character symbols.
    """
    if vocab_size < len(SPECIAL_TAGS) + 1:
        raise ValueError(f"a placeholder vocabulary needs at least {len(SPECIAL_TAGS) + 1} entries, got {vocab_size}")
    tokens = [f"tok{i}" for i in range(vocab_size)]
    # Replace the last len(SPECIAL_TAGS) fillers, walking the tags backwards, so the layout resembles a
    # real tokenizer that appends its special tokens after the learned ones.
    for offset, tag in enumerate(reversed(list(SPECIAL_TAGS.values())), start=1):
        tokens[-offset] = tag
    return tokens


def block_map(cfg: HeronConfig) -> dict:
    """
    Absolute block indices, so the engine does not have to reimplement the three-part split.

    The split itself lives in config.py as methods rather than being recomputed here, because the
    reference implementation builds its blocks from the same definition. If metadata computed the split
    independently, the two could disagree and the converter would read tensors from the wrong blocks.
    """
    start = cfg.input_blocks
    loop_end = cfg.input_blocks + cfg.loop_blocks
    return {
        "input": list(range(0, start)),
        "loop": list(range(start, loop_end)),
        "output": list(range(loop_end, cfg.n_blocks())),
    }


def full_attention_blocks(cfg: HeronConfig) -> list[int]:
    """Blocks that use full attention, derived from the config rule rather than restated here."""
    return [i for i in range(cfg.n_blocks()) if cfg.mixer_mode(i) == "full"]


# Module path, as the reference names it, -> the GGUF tensor name a loader looks up.
#
# This dict is the only place the two spellings exist. metadata() records the mapping as data and the
# converter builds names by looking it up, so a name cannot be written into a file that the metadata
# does not describe - which is the failure Phase 4.1 exists to prevent. A converter that reimplemented
# the arithmetic would load tensors into the wrong blocks while producing a file that passes every
# shape check, because the shape is right and only the block is wrong.
#
# These are patterns, not names: {bid} is a block index, {eid} an expert index, {slot} the slot index
# inside that expert. A routed slot's block is not a placeholder - it is cfg.expert_blocks()[slot], so
# the name for a slot cannot be written for a block the config does not put experts in.
#
# Where the pinned Dory branch has a name for the thing, the thing is named as that branch names it:
# the pin is ggml-org/llama.cpp PR #30306 at commit 98a74c97ea09ac72f6ceda1559808fc0a2f447ce
# (brianchmielNV/llama.cpp). token_embd, blk.{bid}.attn_q/k/v/output, blk.{bid}.ffn_gate/up/down,
# blk.{bid}.ffn_gate_exps/up_exps/down_exps and blk.{bid}.residual_alpha are that branch's spellings.
# Where HeronLM has a mechanism the branch does not have, the name is HeronLM's own and says so: an
# existing name reused for a different thing makes a reader compute the wrong thing from the right
# bytes, which is worse than a name it does not recognise.
TENSOR_NAMES = {
    # Tied embedding. There is no output tensor: the head is this one, which is what
    # general.arch.heron.tied_embeddings is for.
    "embed.weight": "token_embd",

    # Per-block matrices.
    "blocks.{bid}.attn.q.weight": "blk.{bid}.attn_q",
    "blocks.{bid}.attn.k.weight": "blk.{bid}.attn_k",
    "blocks.{bid}.attn.v.weight": "blk.{bid}.attn_v",
    "blocks.{bid}.attn.o.weight": "blk.{bid}.attn_output",
    "blocks.{bid}.ffn.gate.weight": "blk.{bid}.ffn_gate",
    "blocks.{bid}.ffn.up.weight": "blk.{bid}.ffn_up",
    "blocks.{bid}.ffn.down.weight": "blk.{bid}.ffn_down",

    # Scalars. nGPT has no norm weights, so there is no per-block norm tensor to name: the rule is
    # applied at forward time, from the metadata, not from bytes in the file.
    "blocks.{bid}.reinject_gate": "blk.{bid}.reinject_gate",

    # GR read weights. The module holds ONE tensor for the whole model, [block_count, max_loop,
    # branches], and the file carries it as one row per block. That is the pinned branch's
    # residual_alpha role; the reference's shape is what makes it a row rather than a vector.
    "stream.read_weights": "blk.{bid}.residual_alpha",

    # Routed expert slots, in the per-expert file. Slot 0 is absent from this dict on purpose: the
    # generalist is the block's base FFN, so its bytes are already named by the rows above.
    "experts.{eid}.{slot}.ffn.gate.weight": "blk.{bid}.ffn_gate_exps",
    "experts.{eid}.{slot}.ffn.up.weight": "blk.{bid}.ffn_up_exps",
    "experts.{eid}.{slot}.ffn.down.weight": "blk.{bid}.ffn_down_exps",
    # HeronLM's own name, not the branch's ffn_gate_scale: in the pinned branch that name is the
    # gate projection's scale, and here this is the slot's output multiplier. Same file, two
    # meanings, is how a loader quietly produces different numbers.
    "experts.{eid}.{slot}.gate_scale": "blk.{bid}.expert_gate_scale",

    # Router. The prototypes are data rather than parameters - Router stores them as a plain tensor,
    # so state_dict does not carry them and the converter has to be told to write them. Being decided
    # host-side does not make them a blob: five rows of d floats is not what a device cannot hold.
    "router.prototypes": "router_prototypes",

    # Memory path. These are model-level rather than per-block, so they carry no blk prefix: the block
    # they are applied at is a metadata fact (table.read_block), not part of the name.
    "memory.value_proj.weight": "memory_value_proj",
    "memory.gate_key": "memory_gate_key",
    "memory.gate_bias": "memory_gate_bias",

    # The table is not a tensor at all: there is no ggml probe op, so it is an mmap'd blob named by
    # general.arch.heron.table.storage_path. It stays in this dict as None so that "not a tensor" is a
    # decision recorded in the same place the tensor names are, rather than an absence somewhere else.
    "memory.table.weight": None,
}


def tensor_name_patterns() -> list[str]:
    """
    The mapping as data, for the file itself.

    Written as `module_pattern -> gguf_name` strings. Only tensor families are listed: the blob has no
    name to record here, and its shape is already stated by the table keys.
    """
    return [f"{path} -> {name}" for path, name in sorted(TENSOR_NAMES.items()) if name is not None]


# GGUF's token type codes. A control token is what a chat turn is made of; a normal token is text.
# These are gguf.TokenType.CONTROL and gguf.TokenType.NORMAL. They live here as numbers because the
# reference should not depend on the gguf package to describe itself, and tests/test_converter.py
# asserts these still equal the library's members, so the two cannot drift.
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_CONTROL = 3


def token_types(tokens: list[str]) -> list[int]:
    """
    The type of every vocabulary entry, resolved from the vocabulary rather than remembered.

    Same reasoning as the ids: a type code written as a fixed position is a guess about where the tags
    sit, and a reader uses this array to decide whether a token can appear inside a turn.
    """
    special = set(special_token_ids(tokens).values())
    return [TOKEN_TYPE_CONTROL if i in special else TOKEN_TYPE_NORMAL for i in range(len(tokens))]


# The base alphabet: the position a byte-level BPE reads a byte from.
#
# `tokenizer.ggml.byte_fallback` is a rule a reader needs when a byte sequence is not in the vocabulary. It
# is HeronLM's own key - gguf.constants has no name for it and its writer has no helper for it - so nothing
# outside this module can tell the reference what the value should be, and a value written as a constant is a
# claim the reference would have to be able to back.
#
# What a vocabulary CAN back is the shape the rule needs: 256 distinct single-character symbols at the ids a
# reader would read bytes from, 0..255. That is checkable from the vocabulary alone. What it cannot back is
# whether the tokenizer actually uses those positions for arbitrary bytes, because that is a property of the
# file's decoder table, not of the symbol set: the pinned Qwen file's single-character symbols are not the
# standard GPT-2 byte table, and the reference does not reimplement a byte table it cannot check.
#
# So the value comes from the tokenizer file when the file states it, and from the vocabulary when the
# vocabulary is the only thing that can answer. A claim of True that the vocabulary cannot back is refused
# rather than written. A claim of False is written as the file states it, even when the vocabulary looks
# byte-shaped, because the file is the authority on what its own tokenizer does - and the divergence is
# printed by `scripts/convert_to_gguf.py`, which is where a run's provenance is stated.
BASE_ALPHABET_SIZE = 256


def base_alphabet(tokens: list[str]) -> list[str] | None:
    """
    The vocabulary's base alphabet, as a byte-level reader would use it, or None when it is not one.

    None means the vocabulary is not byte-shaped: its first `BASE_ALPHABET_SIZE` entries are not one distinct
    single-character symbol per byte position. That is the only judgment this module can make about bytes
    without assuming a byte table, and it is enough to refuse a claim of byte fallback.
    """
    if len(tokens) < BASE_ALPHABET_SIZE:
        return None
    first = tokens[:BASE_ALPHABET_SIZE]
    if any(len(symbol) != 1 for symbol in first):
        return None
    if len(set(first)) != BASE_ALPHABET_SIZE:
        return None
    return first


def byte_level_rule(tokens: list[str], claim: bool | None = None) -> tuple[bool, str]:
    """
    The value to write, and what backs it.

    Two cases, and each has one authority:

      - The vocabulary is not byte-shaped (the base alphabet is not one symbol per byte). Then no rule can be
        claimed: the value is False, and the authority is the vocabulary. A caller claiming True is refused,
        because writing True would tell a reader there is a rule for bytes this vocabulary has no symbols for.
      - The vocabulary IS byte-shaped. Then the vocabulary can hold the rule but cannot say whether the
        tokenizer uses it, which is a decoder fact. The caller has to supply the file's claim, and the value
        written is that claim.

    `scripts/convert_to_gguf.py` reads the claim from the tokenizer file's `model.byte_fallback` and prints
    which of the two backed the value. tests/test_converter.py checks both paths: the pinned file's claim
    against its own vocabulary, and a fixture that claims more than its vocabulary can back.
    """
    byte_shaped = base_alphabet(tokens) is not None
    if claim is None:
        if byte_shaped:
            raise ValueError(
                "the vocabulary is byte-shaped, but that does not say whether the tokenizer uses it for bytes "
                "it cannot find: the byte-to-symbol table is a property of the tokenizer file's decoder, not of "
                "the symbol set. Supply the file's claim, from `model.byte_fallback`."
            )
        return False, "the vocabulary's base alphabet is not one symbol per byte"
    claim = bool(claim)
    if claim and not byte_shaped:
        raise ValueError(
            f"byte_fallback is claimed as True but the vocabulary's first {BASE_ALPHABET_SIZE} ids are not "
            "one symbol per byte, so there is no byte this vocabulary can represent; the claim cannot be written"
        )
    return claim, "the tokenizer file's claim"


def metadata(cfg: HeronConfig, tokens: list[str], merges: list[str], *, claim: bool | None = None) -> dict:
    """
    tokens and merges must be Qwen's, taken as-is.

    They are arguments rather than constants here because the vocabulary is data: the metadata cannot
    be correct without it, and a hardcoded placeholder would produce a GGUF file that loads but
    tokenizes differently from the model the n-gram table was built with. Since the table is keyed by
    token id, a tokenizer mismatch silently invalidates every entry, so both the length and the presence
    of the special tokens are checked before anything is written.

    `claim` is the tokenizer file's own `model.byte_fallback`. It is an argument for the same reason the
    vocabulary is one: the metadata cannot say what a tokenizer does about bytes without being told, and this
    is the only tokenizer key here that is not a fact about the supplied vocabulary. `byte_level_rule` decides
    the value and refuses the case where the file claims more than the vocabulary can back; when no claim is
    supplied the value is whatever the vocabulary itself entails.
    """
    if len(tokens) != cfg.vocab_size:
        raise ValueError(f"tokenizer has {len(tokens)} tokens, config expects {cfg.vocab_size}")

    # Resolve the ids by name and render the template from the same constants. Reading these from the
    # vocabulary is the whole point: the numbers are not part of the architecture, they are part of the
    # tokenizer file the caller passed in.
    special = special_token_ids(tokens)
    blocks = block_map(cfg)
    byte_fallback, _ = byte_level_rule(tokens, claim)
    return {
        # Upstream readers look at general.architecture. Keeping the value unrecognized means an
        # upstream reader ignores the file rather than crashing on it.
        #
        # general.file_type and the GGUF version are deliberately not in this dict: both are facts about
        # the file the writer produces, not facts about the architecture, and a dict that states them can
        # disagree with the file it is describing. The GGUF version is a header field, not a metadata key
        # at all - gguf.constants has no key for it - and the writer records the file type it actually
        # used. general.architecture stays because it is what the namespace below is named after.
        "general.architecture": "heron",
        "general.name": "HeronLM",

        # Tokenizer: Qwen's, unmodified. The n-gram table is keyed by token id, so a tokenizer change
        # invalidates every table built with it. The engine has to be able to see this coupling.
        "tokenizer.ggml.model": "gpt2",  # BPE
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.add_bos_token": False,

        # Byte-level BPE, decided by `byte_level_rule` above rather than asserted here. The value is either the
        # tokenizer file's claim or what the vocabulary entails, and the two are not the same fact: the pinned
        # Qwen file states `model.byte_fallback: false` while its base alphabet at ids 0..255 is 256
        # single-character symbols, so its byte-level behaviour comes from its ByteLevel decoder rather than
        # from a fallback rule. gguf has no key and no writer helper for this, which is why the reference can
        # only write what it can back: the value below is the file's claim for the pinned tokenizer (false),
        # and the CLI prints which of the two backed it.
        "tokenizer.ggml.byte_fallback": byte_fallback,

        # Which vocabulary entries are chat tags. Read from the same lookup as the ids, so the file
        # cannot say a token is ordinary text while also calling it a turn boundary.
        #
        # The array is ordered by vocabulary id rather than by name, for the same reason the vocabulary itself is
        # ordered: a reader pairs `added_tokens` with `token_type` position by position, and `token_type` is by id.
        # gguf's own vocab writer sorts added tokens by id for the same reason. Name order and id order coincide
        # for a placeholder fixture (its tags sit at the tail in name order) and do not coincide for Qwen, so the
        # tests assert the order as an order and not as a set.
        "tokenizer.ggml.added_tokens": [SPECIAL_TAGS[name] for name in sorted(special, key=special.get)],
        "tokenizer.ggml.token_type": token_types(tokens),

        # The ids are resolved from `tokens` above, never written as numbers here. Qwen uses the
        # end-of-text token for both BOS and padding (and does not add BOS at encode time, hence the
        # flag above), and the end-of-turn token as EOS. Those are policy facts; the ids themselves are
        # facts about the supplied vocabulary, which is why they come from the lookup.
        "tokenizer.ggml.bos_token_id": special["endoftext"],
        "tokenizer.ggml.eos_token_id": special["im_end"],
        "tokenizer.ggml.padding_token_id": special["endoftext"],

        # Rendered from the same tag constants as the ids, so a chat turn and the id map cannot drift.
        "tokenizer.chat_template": chat_template(),

        # Architecture core
        "general.arch.heron.vocab_size": cfg.vocab_size,
        "general.arch.heron.embedding_length": cfg.d_model,
        "general.arch.heron.block_count": cfg.n_blocks(),
        "general.arch.heron.attention.head_count": cfg.heads,
        "general.arch.heron.attention.head_count_kv": cfg.kv_heads,
        "general.arch.heron.attention.head_dim": cfg.head_dim,
        "general.arch.heron.expert_ffn_width": cfg.expert_ffn_width,

        # The base FFN width, which is a different number from the expert slot width. A generic reader can
        # only allocate the shape it is told: expert_ffn_width describes the routed slots, while this is the
        # FFN every block computes through when no expert fires. Both are recorded so a reader that knows
        # nothing about experts can still load the core.
        "general.arch.heron.feed_forward_length": cfg.ffn_width,

        # Context length, which is not the same fact as the window even though the two numbers are equal at
        # the design's scale. The window bounds a sliding-window cache; context length is what a reader
        # reserves memory for. They part company at a full-attention block, which can read past the window,
        # so `full_attention_layers` is the cache rule and this key is the reservation.
        "general.arch.heron.context_length": cfg.swa_window,

        # The head IS the embedding. Recorded as a key because an engine that expects an untied head looks
        # for `output.weight` and has to be told, in the file itself, not to look.
        "general.arch.heron.tied_embeddings": 1,

        # Loop: the elastic-depth knob. max_loop is the runtime range, and the default is the trained
        # bulk value, not the maximum, because a model trained mostly at k=1 behaves worst at k=4.
        "general.arch.heron.loop.layers": blocks["loop"],
        "general.arch.heron.loop.input_layers": blocks["input"],
        "general.arch.heron.loop.output_layers": blocks["output"],
        "general.arch.heron.loop.max_loop": cfg.max_loop,
        "general.arch.heron.loop.default_loop": cfg.loop_curriculum_bulk,

        # The same facts in the spellings the pinned Dory fork reads. The pinned reference is
        # ggml-org/llama.cpp PR #30306 at commit 98a74c97ea09ac72f6ceda1559808fc0a2f447ce
        # (brianchmielNV/llama.cpp); the key names below are that branch's, character for character. They
        # are generated from `blocks` above rather than restated, so an alias cannot disagree with the fact
        # it aliases - which is the only reason two spellings of one fact are acceptable in one file.
        # The counts are recorded as well as the lists because a reader that only wants the loop length
        # should not have to reconstruct it from a list, and because the counts are what a Dory reader
        # looks for first.
        "general.arch.heron.num_loops": cfg.max_loop,
        "general.arch.heron.input_layer_count": len(blocks["input"]),
        "general.arch.heron.recurrent_layer_count": len(blocks["loop"]),
        "general.arch.heron.output_layer_count": len(blocks["output"]),
        "general.arch.heron.recurrent_kv_cache_mode": cfg.cache_policy,

        # Attention policy, which is not inferable from tensor shapes.
        #
        # window_size applies to the blocks NOT listed in full_attention_layers. For the listed blocks the
        # cache has no bound, because a later query can still reach every entry it holds; an engine that
        # trims every cache to the window matches the reference below the window and diverges above it,
        # silently. So this pair of keys is the per-block cache rule, not two independent facts.
        "general.arch.heron.window_size": cfg.swa_window,
        "general.arch.heron.full_attention_layers": full_attention_blocks(cfg),
        "general.arch.heron.cache_policy": cfg.cache_policy,

        # The per-block cache rule as an array, in the shape a loader wants: 0 = sliding window, 1 = full.
        # The codes are not behaviour, they are the reader's instruction, and this is the only place in the
        # file where the rule is stated per block rather than as a window plus an exception list. Both forms
        # come from the same `mixer_mode` call, so the array and the list cannot disagree.
        "general.arch.heron.layer_pattern": [0 if cfg.mixer_mode(i) == "swa" else 1 for i in range(cfg.n_blocks())],

        # Positional encoding. The reference rotates the whole head vector with the module constant base,
        # and applies no scaling. A reader that assumes a default rope would apply a scaling it was never
        # trained with, so "none" is stated rather than left as an absence.
        "general.arch.heron.rope.freq_base": ROPE_FREQ_BASE,
        "general.arch.heron.rope.dimension_count": cfg.head_dim,
        "general.arch.heron.rope.scaling_type": "none",

        # Normalization. nGPT normalizes weights at forward time, so the quantized weights must be read
        # with the same rule or the engine will not match the reference.
        #
        # The order is recorded as data because it is load-bearing: the engine dequantizes and THEN
        # normalizes. A converter that normalized before quantizing would store weights that no longer
        # reproduce the reference, because per-block scales are computed from the magnitudes that
        # normalization removes. The mode is the authority; ngpt is kept as the boolean form for readers
        # that only look at the flag.
        "general.arch.heron.norm_mode": cfg.norm_mode,
        "general.arch.heron.ngpt": int(cfg.norm_mode == "ngpt"),
        "general.arch.heron.weight_norm_order": PARITY_RULE,
        "general.arch.heron.ngpt_weight_scale": cfg.ngpt_weight_scale,
        # The denominator floor, recorded because it is part of the rule and not a rounding detail: a
        # near-zero residual reads differently at different floors, and this is the only value the engine can
        # reproduce. It is the same constant `norms.unit_norm` clamps at.
        "general.arch.heron.norm_eps": EPS,

        # GR branch. Flagged even when off, so the checkpoint describes what it contains.
        "general.arch.heron.gr.branches": cfg.gr_branches,
        "general.arch.heron.gr.read_enabled": int(cfg.ngpt_gr_read),

        # Per-loop read weights. The engine must know this tensor is [block_count, max_loop, branches]
        # and that it is indexed by pass, not by block alone. Without the max_loop dimension the
        # converter would read it as a normal per-block parameter and every loop pass after the first
        # would use pass 0's mixture, which is exactly the difference elastic depth is made of.
        "general.arch.heron.gr.read_weights_shape": [cfg.n_blocks(), cfg.max_loop, cfg.gr_branches],

        # How that one tensor is carried in the file: one tensor per block, name pattern below, each carrying
        # the logical row [max_loop, branches] for its block. The module holds one tensor for the whole model,
        # so the file has to say which name carries which slice. Without this, a loader would read a per-block
        # tensor as a [branches] vector and use pass 0's mixture on every pass, which is the exact failure
        # elastic depth is made of. It is recorded as a pattern rather than as a list of names because the
        # block index is what the loader already has.
        #
        # The dims in the file are the reverse of the logical row: GGUF records dims fastest-first, so a
        # logical [max_loop, branches] row appears as [branches, max_loop]. Rows past the first exist for
        # blocks that never loop; the reference has those values, so the converter writes them. A loader that
        # reads only row 0 for a fixed block still matches the reference, because the reference indexes those
        # blocks at pass 0.
        "general.arch.heron.gr.read_weights_name": TENSOR_NAMES["stream.read_weights"],

        # Branch assignment rule, encoded as data so the engine and the reference cannot disagree. The
        # reserved branch is the one the memory read writes; if the engine rotated it, the recalled
        # value would be overwritten on the second loop pass.
        "general.arch.heron.gr.reserved_branch": 0,
        "general.arch.heron.gr.branch_of": branch_assignment(cfg, cfg.n_blocks()),

        # Experts: slots live in the output block only, so the two compute knobs do not multiply.
        "general.arch.heron.experts.count": cfg.n_experts,
        "general.arch.heron.experts.generalist_slot": cfg.generalist_slot,
        "general.arch.heron.expert_slot_layers": cfg.expert_blocks(),
        # One file per expert, because an expert is a set of slots spread across blocks and hot-swapping
        # loads the set as a unit rather than one module. Index 0, the generalist, has no file: its tensors
        # ARE the core file's base FFN, which is the construction that makes it always resident. The name
        # stays in the list so that slot indices and names line up for a loader resolving a routed slot to a
        # path; the file named for slot 0 is simply never opened.
        "general.arch.heron.expert_files": [f"expert_{i}.gguf" for i in range(cfg.n_experts)],

        # The core file's name, recorded next to the expert files and `table.storage_path` for one reason:
        # these three keys are the set of artifacts a loader has to open, and a name in that set cannot be
        # derived from the architecture. The expert names are there because hot-swapping resolves a routed
        # slot to a path; the blob is there because a host-side probe needs one; the core is there because a
        # loader still has to find it. Recording the core's name and not the others would make the CLI the only
        # place the core name exists, and a name that only exists in the CLI is what a test cannot check - the
        # files it writes could be anything. `scripts/convert_to_gguf.py` takes every output name from these
        # keys, and tests/test_converter.py asserts the names it writes are the names recorded here.
        #
        # This is not the same kind of fact as `general.file_type`, which metadata() deliberately does not
        # state: a file type is what the writer did, so it can only be known at write time. A name is a
        # convention, knowable before anything is written, which is why it can live here.
        "general.arch.heron.core_file": "heron_core.gguf",

        # Memory path. The table is not a ggml tensor: it is an mmap'd blob probed host-side. These keys
        # tell the engine the entry count, the value width, the storage path, and which block injects
        # the value into the reserved branch.
        #
        # The blob is not self-describing, so its shape and dtype are recorded here. Row-major is load
        # bearing: a probe reads exactly one row, so a column-major blob would need a stride-256 gather per
        # probe instead of one contiguous read. int8 is the storage dtype, not the compute dtype - the engine
        # dequantizes the row before the gate sees it.
        "general.arch.heron.table.entries": cfg.ngram_memory_entries,
        "general.arch.heron.table.value_dim": cfg.ngram_memory_dim,
        "general.arch.heron.table.read_block": cfg.ngram_memory_read_block,
        "general.arch.heron.table.hash": "blake2b-64-mod-entries",
        "general.arch.heron.table.storage_path": "heron_table.bin",
        "general.arch.heron.table.layout": "row_major",
        "general.arch.heron.table.dtype": "int8",

        # A row is not reproducible from the code dtype alone. The scheme is one scale per row, so a probe
        # needs the code and its scale: code x scale. Recording only int8 would describe half of every row.
        #
        # The order is recorded because a blob has no header: codes then scales is what keeps a probe a
        # single contiguous read of value_dim bytes plus one half-precision load, which is the property the
        # row_major decision above was made for. If the scales were interleaved with the codes, the probe
        # would read a strided row.
        "general.arch.heron.table.scale_dtype": "float16",
        "general.arch.heron.table.blob_order": ["codes", "scales"],

        # Decode path: static count table, storage-backed, used only as a proposal source.
        "general.arch.heron.ngram.max_entries": cfg.ngram_max_entries,
        "general.arch.heron.ngram.top_k": cfg.ngram_top_k,
        "general.arch.heron.ngram.min_cnt": cfg.ngram_min_cnt,
        "general.arch.heron.ngram.orders": list(cfg.ngram_orders),
        "general.arch.heron.ngram.verify_only": 1,

        # Routing, which is conversation-level. The engine needs the margin and hysteresis values
        # because they are what prevents turn-to-turn flapping.
        "general.arch.heron.router.temperature": cfg.router_temperature,
        "general.arch.heron.router.margin": cfg.router_margin,
        "general.arch.heron.router.hysteresis_turns": cfg.router_hysteresis_turns,
        # The prototypes are carried as a tensor, not as a blob. Being decided host-side does not make
        # weights a blob - a blob is what the device cannot hold in RAM, which is the table's case and not
        # the router's: five rows of d floats. The tensor name is where the bytes live for both a ggml graph
        # and a host-side cosine, so one name serves both.
        "general.arch.heron.router.prototypes_tensor": TENSOR_NAMES["router.prototypes"],

        # Every name the file uses, as data. The keys above tell a reader what a tensor is for; this tells
        # it where the bytes are. It is generated from the same dict the converter looks up, so a file
        # cannot contain a tensor whose name this file does not claim - and a name this file claims cannot
        # be absent from the file, which is what tests/test_converter.py checks in the other direction.
        "general.arch.heron.tensor_names": tensor_name_patterns(),
    }

# Qwen2.5 tokenizer provenance

Files:

- `tokenizer.json` — from `Qwen/Qwen2.5-1.5B` (identical, by SHA-256, to the file in
  `Qwen/Qwen2.5-1.5B-Instruct`). SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
- `tokenizer_config.instruct.json` — from `Qwen/Qwen2.5-1.5B-Instruct`.
  SHA-256 `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`.

Retrieved 2026-11-14. Tested with `tokenizers` 0.23.3.

## Vocabulary limit

The tokenizer file has 151,665 entries, but the model's frozen vocabulary is 151,646 (the three text
special tokens are the last entries the model embeds). The remaining entries are added tags for vision,
tool calls and FIM. `scripts/tokenize_corpus.py` imports the limit from the model config and rejects any
document that produces an id at or above it, so a corpus cannot contain an id the embedding cannot index.
Plain document text does not emit them.

The three text special-token ids resolve by name from this file through
`heronlm.metadata.special_token_ids`, rather than being remembered as literals.

## Chat template

`tokenizer_config.instruct.json` carries Qwen2.5's published chat template (2,507 characters).
`heronlm.metadata.chat_template()` returns that template byte for byte, which is what ROADMAP line 13
("chat template, unmodified") requires. The only source-level deviation is that the two tag spellings are
stored as placeholders and substituted from `IM_START_TAG`/`IM_END_TAG`, so the tag text still has one
source; `tests/test_reference.py` substitutes them back and asserts equality with this file.

Consequences of using the real template:

- It uses Jinja whitespace control, conditionals, loops, `set`, attribute access and the `tojson` filter.
  The engine's renderer has to support those constructs. The earlier 342-character subset did not, but it
  was also not Qwen's template.
- The default system message is Qwen's ("You are Qwen...") and is inserted only when the caller supplies
  no system message. Passing a system message avoids it.
- The template names Qwen's tool-call and tool-response tags as text. The frozen vocabulary ends before the
  added tokens for them (151,657/151,658), so those render as ordinary text rather than single token ids.
  Ordinary system/user/assistant turns use only `im_start`/`im_end`, which are in the vocabulary.

## Byte fallback

`tokenizer.json` states `model.byte_fallback: false` and `decoder.type: ByteLevel`. Its first 256 ids are
256 distinct single-character symbols, but they are not the standard GPT-2 byte table: 221 of them are shared
with it and 35 are extra, so ids 0..255 are not byte positions. The file's byte-level behaviour comes from its
decoder, not from a fallback rule.

The value `heronlm.metadata` writes for `tokenizer.ggml.byte_fallback` is this file's claim, and gguf has no key
constant and no writer helper for it — the key is HeronLM's own, so the reference can only state what it can
back. What a vocabulary can back is the shape (one symbol per byte position); whether the tokenizer uses byte
positions is the file's claim. `byte_level_rule` takes the file's claim when the file states it, derives the
value from the vocabulary when it does not, and refuses a claim the vocabulary cannot back. For this file the
value written is `false`. `scripts/convert_to_gguf.py` prints which of the two backed the value, and
`tests/test_converter.py` asserts both paths and that the numbers above (221 shared, 35 extra) still hold for
the pinned file.

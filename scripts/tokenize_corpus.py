"""
Tokenize a directory of UTF-8 text into the uint32 shards the Phase 0 gate trains on.

The gate budget in ROADMAP line 95 is 1-2B tokens per arm. The synthetic task families cannot fill that
honestly - they contain a few thousand distinct sequences - so a gate-scale run needs real text. This script
is the bridge for whatever text the run is allowed to use: it walks a directory, tokenizes each document with
the project's Qwen tokenizer file, appends the end-of-text token, and writes flat little-endian uint32 shards
plus a manifest that `phase0.tokens` validates.

Split rule: files (documents), not windows. A document hashed into the validation fraction goes entirely to
`val-*.bin`; every other document goes to `train-*.bin`. A window never crosses a shard boundary, so no
validation token can share a window with a training token. That is the property the Phase 0 held-out loss
needs, and it is why the split is not a random per-token mask.

Special tokens are looked up by name through `heronlm.metadata.END_OF_TEXT_TAG`, not written as a literal.
The end-of-text id is the only special token this script emits; a chat template is not applied because the
gate measures next-token loss on documents, not instruction following. The tokenizer file carries added
tokens above the model's text vocabulary (vision, tool-call and FIM tags); their ids are rejected rather than
stored, so every id in a shard is one the model can embed.

Run:
  uv pip install tokenizers
  python scripts/tokenize_corpus.py --tokenizer tokenizer/qwen2.5/tokenizer.json --text-dir <docs/> --out <corpus/>
  python scripts/phase0.py --scale phase0 --corpus-dir <corpus/> --seq-len 2048 --batch 8 ...
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from heronlm.config import FULL
from heronlm.metadata import END_OF_TEXT_TAG
from phase0.tokens import ShardWriter, write_manifest

SUFFIXES = (".txt", ".md")


def sha256_file(path: Path) -> str:
    """Hash the tokenizer file, so the manifest records which vocabulary produced these ids."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def in_validation(relative_path: str, seed: int, fraction: float) -> bool:
    """
    Put a whole document in validation or training, deterministically.

    The hash is of the seed and the relative path, so the split survives reordering, added files and a rerun
    on another machine. It does not depend on file size or on where in the walk the file appeared. Roughly
    `fraction` of documents land in validation, which is enough for a few dozen eval windows at any corpus
    scale; the exact count is printed.
    """
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    raw = hashlib.blake2b(f"{seed}:{relative_path}".encode(), digest_size=8).digest()
    return int.from_bytes(raw, "little") / float(1 << 64) < fraction


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to the Qwen tokenizer.json")
    ap.add_argument("--text-dir", required=True, help="directory of UTF-8 documents")
    ap.add_argument("--out", required=True, help="corpus directory to write")
    ap.add_argument("--shard-tokens", type=int, default=50_000_000,
                    help="tokens per shard file; 50M uint32 tokens is 200 MB")
    ap.add_argument("--val-fraction", type=float, default=0.005,
                    help="fraction of documents held out for eval; whole files, never windows")
    ap.add_argument("--seed", type=int, default=7, help="split seed; the gate data seed by default")
    ap.add_argument("--suffixes", default=".txt,.md", help="comma-separated file extensions to read")
    ap.add_argument("--vocab-limit", type=int, default=None,
                    help="highest token id the model can embed; defaults to the FULL config vocabulary. "
                         "The tokenizer file has extra added tokens above this range (vision, FIM, tool "
                         "tags); a document containing one literally would otherwise produce an id that "
                         "crashes the model mid-run.")
    args = ap.parse_args(argv)

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("tokenizers is not installed; the harness venv does not include it by default.")
        print("install it with:  uv pip install tokenizers")
        return 2

    tokenizer_path = Path(args.tokenizer)
    text_dir = Path(args.text_dir)
    out_dir = Path(args.out)
    if not tokenizer_path.exists():
        print(f"tokenizer file does not exist: {tokenizer_path}")
        return 2
    if not text_dir.is_dir():
        print(f"text directory does not exist: {text_dir}")
        return 2

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Looked up by name. If the vocabulary does not carry the tag, a literal id would silently point at a
    # different token or index past the vocabulary, so this fails instead of guessing.
    eos = tokenizer.token_to_id(END_OF_TEXT_TAG)
    if eos is None:
        print(f"the tokenizer has no {END_OF_TEXT_TAG!r} token; refusing to invent an end-of-text id")
        return 2
    # The tokenizer file's vocabulary can be larger than the model's embedding. Qwen2.5's file carries 22
    # added tokens, 19 of them above the text range (vision, tool-call and FIM tags); the model config stops
    # at the three text specials. The limit is imported from the config rather than remembered, and every id
    # is checked against it, so a corpus cannot contain an id the model would index out of range.
    limit = args.vocab_limit if args.vocab_limit is not None else FULL().vocab_size
    if limit <= 0:
        print(f"vocab limit must be positive, got {limit}")
        return 2
    if eos >= limit:
        print(f"end-of-text id {eos} is outside the model vocabulary limit of {limit}")
        return 2
    vocab_size = limit
    tokenizer_entries = tokenizer.get_vocab_size()
    if tokenizer_entries > limit:
        print(f"note: tokenizer file has {tokenizer_entries:,} entries; the corpus is limited to id < "
              f"{limit:,}, the model's embedding. Added tokens above the limit will be rejected, not stored.")

    suffixes = tuple(s.strip().lower() for s in args.suffixes.split(",") if s.strip())
    files = sorted(p for p in text_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    if not files:
        print(f"no files with suffixes {suffixes} under {text_dir}")
        return 2

    relative = {p: p.relative_to(text_dir).as_posix() for p in files}
    val_files = [p for p in files if in_validation(relative[p], args.seed, args.val_fraction)]
    if args.val_fraction > 0 and not val_files:
        # A tiny corpus can hash every document into train. Without this, the run would fail at the first
        # validation window after the tokenization, possibly hours later. The last document moves instead.
        val_files = [files[-1]]
        print(f"warning: no document hashed into validation; holding out {relative[val_files[0]]}")
    val_set = set(val_files)
    train_files = [p for p in files if p not in val_set]
    if not train_files:
        print("every document hashed into validation; lower --val-fraction")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": ShardWriter(out_dir, "train", args.shard_tokens, vocab_size=vocab_size),
        "val": ShardWriter(out_dir, "val", args.shard_tokens, vocab_size=vocab_size),
    }
    totals = {"train": 0, "val": 0}
    chars = {"train": 0, "val": 0}
    try:
        for split, group in (("train", train_files), ("val", val_files)):
            for path in group:
                # Each file is one document. Reading it whole keeps the end-of-text token at the document
                # boundary; streaming by line would make every line a document and teach the model that
                # newlines end texts.
                text = path.read_text(encoding="utf-8", errors="replace")
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if ids and max(ids) >= vocab_size:
                    highest = max(ids)
                    print(
                        f"{relative[path]}: produced token id {highest}, outside the model vocabulary limit "
                        f"of {vocab_size}. The document contains one of the tokenizer's added tags "
                        f"literally; remove it or exclude the file rather than storing an id the model "
                        f"cannot embed."
                    )
                    return 2
                ids.append(eos)
                writers[split].add(ids)
                totals[split] += len(ids)
                chars[split] += len(text)
            print(f"  {split}: {len(group)} documents, {totals[split]:,} tokens")
    finally:
        shards = []
        for split in ("train", "val"):
            shards.extend(writers[split].close())

    manifest = write_manifest(
        out_dir, shards, vocab_size=vocab_size,
        tokenizer={"file": tokenizer_path.name, "sha256": sha256_file(tokenizer_path)},
        created_by="scripts/tokenize_corpus.py",
    )

    total = sum(totals.values())
    print(f"\nwrote {out_dir}")
    print(f"  shards      : {len(shards)} ({sum(1 for s in shards if s['split'] == 'train')} train, "
          f"{sum(1 for s in shards if s['split'] == 'val')} val)")
    print(f"  tokens      : {total:,} (train {totals['train']:,}, val {totals['val']:,})")
    print(f"  chars/token : {(chars['train'] + chars['val']) / total:.2f}")
    print(f"  vocab       : limited to id < {vocab_size:,}; end-of-text id resolved by name")
    print(f"  digest      : {manifest['digest']}")
    print(f"  bytes       : {sum(s['tokens'] for s in shards) * 4 / 1e9:.2f} GB of token shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

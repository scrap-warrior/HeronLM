"""
Build a corpus directory from real text: shards, manifest, and the document index.

The gate budget in ROADMAP line 95 is 1-2B tokens per arm. The synthetic task families cannot fill that
honestly - they contain a few thousand distinct sequences - so a gate-scale run needs real text. This script is
the bridge for whatever text the run is allowed to use. It reads documents from a directory of text files or from
parquet files, tokenizes each document with the project's pinned Qwen tokenizer file, and writes flat
little-endian uint32 shards plus a manifest that `phase0.tokens` validates.

The rules themselves are not in this file. Identity, selection, the split and the special-token policy live in
`phase0.documents`, which takes the encoder as an argument rather than importing a tokenizer. That separation is
what lets the rules be tested without a tokenizer, a download, or a parquet reader; this file is the only place
that reads the pinned tokenizer file, and it is a thin loop that hands those rules a stream of documents.

Two sources, two identity shapes:
  - a text file is one document, so its path relative to the source directory is its identity;
  - a parquet row is one document, and FineWeb-Edu rows carry no id column - text, score, url, date and
    token_count, none of which says which row a document is - so a row's identity is (file, row index).

Size is reached by a rule, not by stopping. `--select-fraction` compares a hash of the identity with a fraction,
so a corpus built at 0.1 is a strict subset of the same corpus at 1.0: every document already inside it stays
inside it, and the held-out set is the same documents at both sizes. A `--max-tokens` stop is only a prefix if the
walk order and the file set are unchanged, and a corpus that is a prefix rather than a subset cannot be compared
with a later larger corpus, because its eval set is a different set of documents. Since the corpus size is decided
before the first run and `corpus_id` is inside the checkpoint's run key, the size has to be reached by the stable
rule.

Special tokens. The pinned tokenizer matches its added tokens even when asked not to add special tokens, so a
crawled page that contains a chat tag literally produces an added-token id inside the document rather than at its
boundary. Two different problems follow, and `phase0.documents` names both:
  - `out_of_range`: the id is above the model's embedding, so no model can embed the document;
  - `special_in_text`: the id is inside the text vocabulary but it is a control token that came from the page's
    text rather than from the builder's own document boundary.
The ids are resolved from the vocabulary by name through `heronlm.metadata.SPECIAL_TAGS`, never written as
literals here. The end-of-text id is the only special id a stored document contains, and only because this script
appends it at the document boundary.

Run:
  uv pip install tokenizers
  python scripts/tokenize_corpus.py --tokenizer tokenizer/qwen2.5/tokenizer.json --text-dir <docs/> --out <corpus/>
  python scripts/tokenize_corpus.py --tokenizer tokenizer/qwen2.5/tokenizer.json --parquet-dir <parquet/> --out <corpus/>
  python scripts/phase0.py --scale phase0 --corpus-dir <corpus/> --seq-len 512 --batch 1 ...
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from heronlm.config import FULL
from heronlm.metadata import SPECIAL_TAGS
from phase0.documents import (
    DEFAULT_POLICY,
    POLICIES,
    build_conditions,
    build_corpus,
    document_identity,
    selected,
    split_of,
)

SUFFIXES = (".txt", ".md")
PARQUET_SUFFIXES = (".parquet",)


def sha256_file(path: Path) -> str:
    """Hash the tokenizer file, so the manifest records which vocabulary produced these ids."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_documents(source_dir: Path, suffixes: tuple[str, ...]):
    """
    One document per file.

    Reading a file whole is what keeps the end-of-text token at the document boundary; streaming by line would
    make every line a document and teach the model that newlines end texts. The walk order does not matter to the
    split, because the split is a hash of the identity rather than of the position in the walk - but the order is
    sorted anyway, so the shard bytes for the same source directory are the same bytes on a rerun.
    """
    for path in text_file_list(source_dir, suffixes):
        # A text file makes no claim about its token count, so the fourth field is None rather than a guess. The
        # contract is the same shape for both sources so the builder does not need to know which kind it was given.
        yield path.relative_to(source_dir).as_posix(), None, path.read_text(encoding="utf-8", errors="replace"), None


def text_file_list(source_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    return sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def parquet_file_list(source_dir: Path) -> list[Path]:
    return sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in PARQUET_SUFFIXES)


def parquet_row_count(path: Path) -> int:
    """
    Rows in a parquet file, from its footer metadata.

    This is the only way to know a row's identity without reading the row, and the identity is what the split is a
    hash of. Reading the footer rather than the data is what makes the pre-check below cheap: deciding whether any
    document hashes into validation should not require decoding 2 GB of text.
    """
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def parquet_documents(source_dir: Path, text_column: str, batch_rows: int, count_column: str = ""):
    """
    One document per row.

    Row order within a file and file order on disk are the identity, which is stated as what it is: positional.
    FineWeb-Edu has no id column, so nothing makes a row's identity stable across a re-sort of the same rows. The
    row counter is the absolute position in the file, not the offset inside a batch, because an identity that
    restarted at every batch would name the same document twice.
    """
    import pyarrow.parquet as pq

    columns = [text_column] + ([count_column] if count_column else [])
    for path in parquet_file_list(source_dir):
        source = path.relative_to(source_dir).as_posix()
        row = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
            texts = batch.column(0).to_pylist()
            claims = batch.column(1).to_pylist() if count_column else [None] * len(texts)
            for text, claim in zip(texts, claims):
                yield source, row, text, claim
                row += 1


def resolve_special_ids(tokenizer) -> dict[str, int]:
    """
    The special ids, resolved by name from the vocabulary that produced them.

    `heronlm.metadata.special_token_ids` is the same lookup the converter uses, and the reason it is a lookup here
    as well is the same: a remembered number passes every check on the pinned vocabulary and points at a
    different token on any other one. The names come from `SPECIAL_TAGS`, which is the only sanctioned spelling of
    a tag in this project.
    """
    ids = {}
    for name, tag in SPECIAL_TAGS.items():
        value = tokenizer.token_to_id(tag)
        if value is None:
            raise SystemExit(
                f"the tokenizer has no {name} token ({tag!r}); refusing to invent an id for it. The policy needs "
                "the ids the vocabulary actually assigns, not the ids this file remembers."
            )
        ids[name] = int(value)
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to the pinned Qwen tokenizer.json")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-dir", help="directory of UTF-8 documents, one document per file")
    source.add_argument("--parquet-dir", help="directory of parquet files, one document per row")
    ap.add_argument("--out", required=True, help="corpus directory to write")
    ap.add_argument("--text-column", default="text", help="parquet column holding the document text")
    ap.add_argument("--batch-rows", type=int, default=2000,
                    help="rows read per parquet batch; this is a read size, not a document size")
    ap.add_argument("--shard-tokens", type=int, default=50_000_000,
                    help="tokens per shard file; 50M uint32 tokens is 200 MB")
    ap.add_argument("--val-fraction", type=float, default=0.005,
                    help="fraction of documents held out for eval; whole documents, never windows")
    ap.add_argument("--select-fraction", type=float, default=1.0,
                    help="fraction of the source that enters the corpus, by identity hash. A corpus built at a "
                         "smaller fraction is a strict subset of the same corpus at a larger fraction, which is "
                         "the only way a corpus can grow without changing its eval set. Use this rather than "
                         "stopping at a token count, because a stop is a prefix of one walk order and a fraction "
                         "is a rule that survives the source being re-ordered.")
    ap.add_argument("--seed", type=int, default=7, help="split seed; the gate data seed by default")
    ap.add_argument("--suffixes", default=".txt,.md", help="comma-separated text file extensions to read")
    ap.add_argument("--special-policy", default=DEFAULT_POLICY, choices=POLICIES,
                    help="what to do with a document that contains a special id in its text: skip drops it, "
                         "strip removes the offending ids, refuse stops the build. Default skip, because one bad "
                         "document must not end a run that has already read 150M tokens; the count is recorded in "
                         "the manifest, so a reader can see how much was dropped.")
    ap.add_argument("--vocab-limit", type=int, default=None,
                    help="highest token id the model can embed; defaults to the FULL config vocabulary. "
                         "The tokenizer file has extra added tokens above this range (vision, FIM, tool tags); a "
                         "document containing one literally would otherwise produce an id that crashes the model "
                         "mid-run.")
    ap.add_argument("--count-column", default="token_count",
                    help="parquet column carrying the source's own claim about its token count, for the "
                         "cross-check printed at the end. Empty string skips it. The claim is a diagnostic, not "
                         "authority: it was produced by a different tokenizer, and the number the run trains on "
                         "is the one this script produced.")
    ap.add_argument("--no-index", action="store_true",
                    help="do not write documents.jsonl. A corpus without an index cannot attribute a window to a "
                         "document, so a held-out loss on it is a number over bytes with no name.")
    args = ap.parse_args(argv)

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("tokenizers is not installed; the harness venv does not include it by default.")
        print("install it with:  uv pip install tokenizers")
        return 2

    is_parquet = bool(args.parquet_dir)
    if is_parquet:
        try:
            import pyarrow.parquet  # noqa: F401 - only to report the missing reader before the first read
        except ImportError:
            print("pyarrow is not installed; the harness venv does not include it by default.")
            print("It is not declared in pyproject.toml for the same reason tokenizers is not: the reference "
                  "itself does not read parquet, only this builder does.")
            print("install it with:  uv pip install pyarrow")
            return 2

    tokenizer_path = Path(args.tokenizer)
    out_dir = Path(args.out)
    if not tokenizer_path.exists():
        print(f"tokenizer file does not exist: {tokenizer_path}")
        return 2

    source_dir = Path(args.parquet_dir if is_parquet else args.text_dir)
    if not source_dir.is_dir():
        print(f"{source_dir} is not a directory")
        return 2

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    special_ids = resolve_special_ids(tokenizer)
    eos = special_ids["endoftext"]

    limit = args.vocab_limit if args.vocab_limit is not None else FULL().vocab_size
    problems = build_conditions(args.special_policy, args.seed, args.val_fraction, args.select_fraction,
                                limit, special_ids, eos)
    if problems:
        for problem in problems:
            print(problem)
        return 2

    tokenizer_entries = tokenizer.get_vocab_size()
    if tokenizer_entries > limit:
        print(f"note: tokenizer file has {tokenizer_entries:,} entries; the corpus is limited to id < "
              f"{limit:,}, the model's embedding. Added tokens above the limit are policy cases, not stored ids.")

    suffixes = tuple(s.strip().lower() for s in args.suffixes.split(",") if s.strip())
    count_column = args.count_column if is_parquet else ""

    # The encoder is passed to the builder rather than imported by it, so the builder's rules can be exercised
    # without a tokenizer. `add_special_tokens=False` is the point: the only special id a stored document is
    # allowed to contain is the boundary token this script appends, so a control token found inside the text is
    # the document's own text, which is what the policy is about.
    def encode(text: str):
        return tokenizer.encode(text, add_special_tokens=False).ids

    def documents():
        if is_parquet:
            for source, row, text, claim in parquet_documents(source_dir, args.text_column, args.batch_rows,
                                                              count_column):
                yield source, row, text, claim
        else:
            for source, row, text, claim in text_documents(source_dir, suffixes):
                yield source, row, text, claim

    def identities():
        # The identities a source contains, without reading any document. For text this is the file list; for
        # parquet it is the row count from the footer. The split is a hash of identity, so the fallback decision
        # below can be made before 150M tokens are written rather than at the first eval window.
        if is_parquet:
            for path in parquet_file_list(source_dir):
                name = path.relative_to(source_dir).as_posix()
                for row in range(parquet_row_count(path)):
                    yield document_identity(name, row)
        else:
            for path in text_file_list(source_dir, suffixes):
                yield document_identity(path.relative_to(source_dir).as_posix(), None)

    force_val = None
    if 0.0 < args.val_fraction < 1.0:
        chosen = [identity for identity in identities() if selected(identity, args.seed, args.select_fraction)]
        if not chosen:
            print("no document would be selected at this fraction; the corpus cannot be built")
            return 2
        if not any(split_of(identity, args.seed, args.val_fraction) == "val" for identity in chosen):
            force_val = chosen[-1]
            print(f"warning: no document hashed into validation at --val-fraction {args.val_fraction}; the last "
                  f"selected document {force_val} is held out instead, and the manifest records that as a "
                  "deviation from the rule rather than as the rule. Raise --val-fraction so the rule itself "
                  "produces the eval set.")

    result = build_corpus(out_dir, documents(), encode,
                          seed=args.seed, val_fraction=args.val_fraction, vocab_size=limit,
                          special_ids=special_ids, eos_id=eos, policy=args.special_policy,
                          select_fraction=args.select_fraction, shard_tokens=args.shard_tokens,
                          write_index=not args.no_index, force_val=force_val,
                          tokenizer={"file": tokenizer_path.name, "sha256": sha256_file(tokenizer_path),
                                     "entries": tokenizer_entries})

    for problem in result["problems"]:
        if isinstance(problem, str):
            print(problem)
        else:
            print(f"{problem['identity']}: {[p['kind'] for p in problem['problems']]}")
    if result["manifest"] is None:
        return 2

    manifest, counts = result["manifest"], result["counts"]
    total = sum(c["tokens"] for c in counts["splits"].values())
    print(f"\nwrote {out_dir}")
    print(f"  documents   : {counts['documents_seen']:,} seen, {counts['not_selected']:,} not selected")
    for split in ("train", "val"):
        record = counts["splits"][split]
        print(f"  {split:<9}: {record['documents']:,} documents, {record['tokens']:,} tokens, "
              f"empty {record['skipped_empty']:,}, by policy {record['skipped_policy']:,}, "
              f"ids stripped {record['ids_stripped']:,}")
    print(f"  policy      : {args.special_policy}; special ids resolved by name "
          f"({', '.join(f'{name}={value}' for name, value in sorted(special_ids.items()))})")
    print(f"  selection   : {args.select_fraction} of the source, by identity hash")
    if manifest["policy"]["val_fallback"]["requested"]:
        print(f"  val fallback: {manifest['policy']['val_fallback']['requested']} "
              f"({'held out' if manifest['policy']['val_fallback']['applied'] else 'not applied'})")
    print(f"  eval set    : {counts['splits']['val']['documents']:,} documents, named in documents.jsonl")
    print(f"  chars/token : {(counts['splits']['train']['chars'] + counts['splits']['val']['chars']) / total:.2f}")
    print(f"  digest      : {manifest['digest']}")
    print(f"  bytes       : {total * 4 / 1e9:.2f} GB of token shards")

    if count_column:
        claimed = 0
        for record in manifest["sources"]:
            claimed += int(record.get("claimed_tokens") or 0)
        # The source's claim is per document; this script's count includes the boundary token it appends, so the
        # two are only comparable after that difference is taken out. Stating the difference is what makes the
        # comparison an observation rather than a mismatch report.
        produced = total - sum(c["documents"] for c in counts["splits"].values())
        print(f"\n  cross-check : source claims {claimed:,} tokens, this tokenizer produced {produced:,} "
              f"(boundary tokens excluded). Diagnostic only: the claim comes from a different tokenizer, and the "
              "number a run trains on is the one produced here.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

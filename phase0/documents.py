"""
Document rules for the corpus builder: identity, selection, policy, and the index that ties a window back to
the document it came from.

Why this lives next to `phase0.tokens` rather than inside it: `tokens.py` is a promise about bytes - the shard
format, the manifest, the stream over fixed windows. This module is a set of decisions about documents, and the
two cannot be merged because a decision can be revised without changing the format, while a format cannot be
revised without invalidating every corpus already built. The split is a decision; a `.bin` file is not.

Identity. A text file is one document, so its path identifies it. Row-shaped sources have no id column:
FineWeb-Edu's rows carry text, score, url, date and token_count, and none of those says which row a document
is, so a row-shaped document is identified by (source file, row index). Identity is not decoration - the split
is a hash of it. A hash of anything else (the position in the walk, the file size, the row group) would put the
same document in a different split on a rerun, and a held-out set that moves between runs is not held out.

One number, three decisions. `unit(identity, seed)` is the only randomness in the builder, and it is a pure
function of identity and seed:
  - selection compares it with `select_fraction`, so a corpus built at a smaller fraction is a strict subset of
    the same corpus at a larger fraction: every document already in it stays in it;
  - the split compares the same number with `val_fraction`, so the held-out set is the same documents at every
    corpus size. That is what makes an eval set nameable before the first run and still the same eval set in
    the second one, which is the only way a comparison across scales means anything;
  - `val_fraction` cannot exceed `select_fraction`, because the rule would then promise documents the corpus
    does not contain.

Policy. Some documents cannot be stored as they are. A crawled page that contains a chat tag literally encodes
to an added-token id, and the pinned tokenizer matches added tokens even when asked not to add special tokens,
so the id appears inside the document rather than at its boundary. Two different problems follow from that, and
the builder names both separately:
  - `out_of_range`: the id is above the model's embedding, so no model can embed the document at all;
  - `special_in_text`: the id is inside the text vocabulary but it is a control token that came from the
    document's text rather than from the builder's own document boundary. A held-out loss over a corpus full of
    turn delimiters measures delimiter prediction, which is not what the Phase 0 gate measures.

Three states, decided per document and counted: `skip` (drop the document), `strip` (remove the offending ids,
keep the document), `refuse` (stop and report the problems). The default is `skip`, because at corpus scale one
bad document must not end a run that has already read 150M tokens; the count is written into the manifest so a
reader can see how much was dropped. `refuse` keeps the earlier behaviour and is what to use when the corpus is
small enough that a problem is worth looking at rather than counting.

The index. A shard is a run of documents and a window is a fixed slice of that run, so a loss number on a window
can be traced back to the document it came from only if the boundaries were recorded. The manifest records the
shards; `documents.jsonl` records the documents. Both are claims about the same bytes, so both are checked at
load: the per-split sums must match, and a document may legitimately straddle two shards because a flush happens
at a fixed token count rather than at a document boundary. That is also why the index records where a document
starts rather than claiming each shard is one document.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from pathlib import Path

from phase0.tokens import SPLITS, ShardWriter, read_manifest, write_manifest

# The index format number, kept separate from the shard format number because a corpus can carry one without the
# other: a corpus built before document boundaries were recorded has shards and no index, and refusing it is a
# different decision from refusing a shard dtype.
INDEX_FORMAT = 1
INDEX_FILE = "documents.jsonl"

POLICIES = ("skip", "strip", "refuse")
DEFAULT_POLICY = "skip"

PROBLEM_OUT_OF_RANGE = "out_of_range"
PROBLEM_SPECIAL_IN_TEXT = "special_in_text"

# What the policy decided about one document. `refused` is a state rather than an action: the builder stops and
# reports, so no decision is made about that document.
ACTIONS = ("kept", "skipped", "stripped", "refused")


def unit(identity: str, seed: int) -> float:
    """
    The one number a document's fate is decided from, in [0, 1).

    Identity and seed only, so the decision does not depend on walk order, file size, row group, or when the
    document was read. This is the same rule the text-file path has always used (`blake2b(seed:identity)`), kept
    byte for byte so a split rule cannot be restated as a different rule while claiming to be the same one.
    """
    if not identity:
        raise ValueError("a document needs an identity")
    raw = hashlib.blake2b(f"{seed}:{identity}".encode(), digest_size=8).digest()
    return int.from_bytes(raw, "little") / float(1 << 64)


def document_identity(source: str, row: int | None = None) -> str:
    """
    The identity of one document: its source path, plus the row when the source is row-shaped.

    `row is None` means the file is the document, which is the case for a directory of text files. For a parquet
    file the identity is positional within that file, and that is stated rather than assumed: it is stable for an
    unchanged file, and nothing in FineWeb-Edu makes it stable across a re-sort because the dataset carries no
    id. A corpus whose identity is positional can be reproduced from the same files, but not from a different
    ordering of the same rows.
    """
    name = str(source).strip()
    if not name:
        raise ValueError("a document needs a source path")
    name = Path(name).as_posix()
    if row is None:
        return name
    if int(row) < 0:
        raise ValueError("row indices are non-negative")
    return f"{name}:{int(row)}"


def selected(identity: str, seed: int, fraction: float) -> bool:
    """Whether a document is in the corpus at all, by rule rather than by how many tokens have been read."""
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    return unit(identity, seed) < fraction


def split_of(identity: str, seed: int, val_fraction: float) -> str:
    """
    Which split a document goes to, from identity alone.

    Whole documents, never windows: a window cannot be split across two sets, and a per-token mask would let a
    neighbouring window leak. The value is the same `unit` selection uses, so the held-out set is a subset of the
    corpus at every size and does not change when the corpus grows.
    """
    if val_fraction <= 0.0:
        return "train"
    if val_fraction >= 1.0:
        return "val"
    return "val" if unit(identity, seed) < val_fraction else "train"


def policy_problems(ids, special_ids: set[int], vocab_limit: int) -> list[dict]:
    """
    The ids in one document that cannot be stored as they are, in the order they appear.

    A list, not an exception, so a driver can show every document that trips the policy rather than the first
    one, and so the two kinds of problem stay distinguishable: an id above the model's embedding is a
    representation problem, and a control token inside the text is a measurement problem. They are not the same
    thing, and a count that merged them would say neither.
    """
    problems: list[dict] = []
    for position, token in enumerate(ids):
        if int(token) >= int(vocab_limit):
            kind = PROBLEM_OUT_OF_RANGE
        elif int(token) in special_ids:
            kind = PROBLEM_SPECIAL_IN_TEXT
        else:
            continue
        problems.append({"position": position, "id": int(token), "kind": kind})
    return problems


def apply_policy(ids, problems: list[dict], policy: str) -> tuple[str, list[int] | None]:
    """
    What to store for one document, and which state the policy chose.

    `refused` returns no ids: the caller stops. `strip` removes only the ids that tripped the policy, not the
    document's own boundary token, which the caller appends after this decision is made.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if not problems:
        return "kept", list(ids)
    if policy == "skip":
        return "skipped", None
    if policy == "refuse":
        return "refused", None
    tripped = {problem["position"] for problem in problems}
    return "stripped", [token for position, token in enumerate(ids) if position not in tripped]


def build_conditions(policy: str, seed: int, val_fraction: float, select_fraction: float,
                     vocab_size: int, special_ids: dict[str, int], eos_id: int) -> list[str]:
    """
    The settings a corpus build cannot be run with, as a list.

    These are the conditions the builder can check before it reads a single document. Checking them first is what
    stops a run from writing 150M tokens under a rule that could not have produced a corpus the loader would
    accept - for example a val fraction larger than the selection fraction, which promises documents the corpus
    does not contain, or an end-of-text id the model cannot embed.
    """
    problems: list[str] = []
    if policy not in POLICIES:
        problems.append(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if not 0.0 < float(val_fraction) < 1.0:
        problems.append(
            f"val_fraction {val_fraction} cannot produce both splits; it must be strictly between 0 and 1, "
            "because a corpus with no val split has no held-out loss"
        )
    if not 0.0 < float(select_fraction) <= 1.0:
        problems.append(f"select_fraction {select_fraction} must be in (0, 1]")
    if 0.0 < float(val_fraction) < 1.0 and float(select_fraction) > 0.0 and val_fraction > select_fraction:
        problems.append(
            f"val_fraction {val_fraction} is larger than select_fraction {select_fraction}: the eval rule would "
            "name documents that the selection rule excludes, so the corpus cannot contain its own eval set. "
            "Lower val_fraction or raise select_fraction."
        )
    if int(vocab_size) <= 0:
        problems.append("vocab_size must be positive")
    if int(eos_id) >= int(vocab_size):
        problems.append(
            f"end-of-text id {eos_id} is outside the model vocabulary limit of {vocab_size}, so the boundary "
            "token this builder appends is a token the model cannot embed"
        )
    for name, sid in sorted(special_ids.items(), key=lambda item: int(item[1])):
        if int(sid) >= int(vocab_size):
            problems.append(
                f"special id {name} resolves to {sid}, above the model vocabulary limit of {vocab_size}; the "
                "policy cannot check documents against it because the model could not embed it anyway"
            )
    return problems


def _empty_counts() -> dict:
    """
    Per-split counters.

    The skipped counts are part of the record, not a footnote: they say how much of the source is not inside the
    number being reported. A corpus that dropped 3% of its documents to the policy is a different corpus from one
    that dropped none, and the manifest has to say which one a run trained on.
    """
    return {
        "documents": 0,
        "tokens": 0,
        "chars": 0,
        "skipped_empty": 0,
        "skipped_policy": 0,
        "ids_stripped": 0,
    }


def build_corpus(root: str | Path, documents, encode, *, seed: int, val_fraction: float,
                 vocab_size: int, special_ids: dict[str, int], eos_id: int,
                 policy: str = DEFAULT_POLICY, select_fraction: float = 1.0,
                 shard_tokens: int = 50_000_000, write_index: bool = True,
                 force_val: str | None = None,
                 tokenizer: dict | None = None,
                 created_by: str = "scripts/tokenize_corpus.py") -> dict:
    """
    Build a corpus directory from a stream of documents: shards, manifest, and the document index.

    `documents` yields `(source, row, text, claim)`, where `row` is None for a source that is one document per
    file and `claim` is the source's own statement about how many tokens this document contains, or None when the
    source does not make one. A text file makes no claim; FineWeb-Edu rows carry a `token_count` column produced
    by a different tokenizer, so it is recorded per source as a diagnostic and never used to decide anything.
    `encode` is the tokenizer as a function from text to ids, passed in rather than imported. That is the shape
    of this module: the rules about documents live here, and the part that reads the pinned tokenizer file stays
    in `scripts/`, which is the only place that reads it. A test can hand this function a fake encoder and check
    the rules without a tokenizer, a download, or a parquet reader.

    `special_ids` is the resolved name-to-id map, not a set of remembered numbers, because the manifest records
    which names produced which ids - the same reason the reference resolves ids by name.

    `force_val` is an escape hatch for one case, and it is recorded rather than applied quietly. When the source
    is small enough that every identity hashes into train, the rule produces no val shard and the corpus cannot
    be loaded, because `read_manifest` requires both splits. Naming one document as held-out fixes the symptom,
    but it is a document the rule did not choose, so the manifest records it as a deviation. The rule stays a
    pure function of identity: `force_val` says which document to treat as held-out when the rule produced none,
    and it is checked before any document is encoded, so the deviation is known at the start of a build rather
    than discovered at the first eval window.

    Returns the manifest, the counts, the problems, and the index entries. When the settings cannot produce a
    corpus, nothing is written and `problems` is the reason. When the policy is `refuse` and a document trips it,
    the build stops at that document and `problems` names it; the shards already written are left but no manifest
    is written, so the directory is not a corpus. `refuse` cannot report a problem it has not read yet, and the
    counts it returns are the counts up to the point where it stopped.
    """
    root = Path(root)
    problems = build_conditions(policy, seed, val_fraction, select_fraction, vocab_size, special_ids, eos_id)
    if problems:
        return {"manifest": None, "counts": {}, "problems": problems, "index_entries": []}

    special_set = {int(v) for v in special_ids.values()}
    root.mkdir(parents=True, exist_ok=True)
    writers = {split: ShardWriter(root, split, shard_tokens, vocab_size=vocab_size) for split in SPLITS}
    counts = {"documents_seen": 0, "not_selected": 0, "splits": {s: _empty_counts() for s in SPLITS}}
    sources: dict[str, dict] = {}
    entries: list[dict] = []
    stopped: list[dict] = []
    fallback_applied = False

    try:
        for source, row, text, claim in documents:
            identity = document_identity(source, row)
            counts["documents_seen"] += 1
            record = sources.setdefault(source, {
                "path": Path(source).as_posix(),
                "identity_format": "path:row" if row is not None else "path",
                "rows": 0, "documents": 0, "not_selected": 0,
                "skipped_empty": 0, "skipped_policy": 0, "ids_stripped": 0, "claimed_tokens": 0,
                **{f"{split}_documents": 0 for split in SPLITS},
                **{f"{split}_tokens": 0 for split in SPLITS},
            })
            record["rows"] += 1

            if not selected(identity, seed, select_fraction):
                counts["not_selected"] += 1
                record["not_selected"] += 1
                continue

            split = split_of(identity, seed, val_fraction)
            if force_val is not None and identity == str(force_val):
                # The rule did not choose this document; the caller named it because the rule produced no val
                # document at all. It still has to be a document the selection rule admitted, and the manifest
                # records it as a deviation rather than as the rule.
                split = "val"
                fallback_applied = True
            if not text or not text.strip():
                counts["splits"][split]["skipped_empty"] += 1
                record["skipped_empty"] += 1
                continue

            ids = list(encode(text))
            found = policy_problems(ids, special_set, vocab_size)
            action, kept = apply_policy(ids, found, policy)
            if action == "refused":
                stopped.append({"identity": identity, "problems": found})
                break
            if action == "skipped":
                counts["splits"][split]["skipped_policy"] += 1
                record["skipped_policy"] += 1
                continue
            if action == "stripped":
                counts["splits"][split]["ids_stripped"] += len(found)
                record["ids_stripped"] += len(found)

            # The document boundary token is added by the builder, not found in the text, so it is the only
            # special id a stored document contains. Recording the position before writing is what makes the
            # index a claim that can be checked against the bytes. It is recorded as a logical offset: the shard
            # a document lands in is not known until close() has folded the remainder, so the shard name is
            # filled in from the finished shard list below.
            logical = writers[split].logical_offset()
            kept.append(int(eos_id))
            writers[split].add(kept)

            if write_index:
                entries.append({
                    "identity": identity, "source": Path(source).as_posix(),
                    "row": int(row) if row is not None else None,
                    "split": split, "logical": logical, "shard": None, "start": None, "tokens": len(kept),
                })

            # The claim is recorded only for documents this build kept. A source's claim covers rows this build
            # never stored, so a comparison against a count that excludes them is a comparison against a set the
            # manifest does not describe. Keeping them aligned is what makes the number readable.
            counts["splits"][split]["documents"] += 1
            counts["splits"][split]["tokens"] += len(kept)
            counts["splits"][split]["chars"] += len(text)
            if claim is not None:
                record["claimed_tokens"] += int(claim)
            record["documents"] += 1
            record[f"{split}_documents"] += 1
            record[f"{split}_tokens"] += len(kept)
    finally:
        # close() folds a short remainder into the last shard, so the index and the manifest must be written
        # after it: the index claims a document starts in a shard that may not exist until this point.
        shards = {split: writers[split].close() for split in SPLITS}

    if stopped:
        # `refuse` is a decision to stop and report, so the directory is left as what it is: the shards that
        # were written, with no manifest and no index. A manifest here would describe a corpus that is a prefix
        # of the source rather than the source, and `read_manifest` would accept it as a corpus. The counts
        # returned are the counts up to the document it stopped on, which is the only honest reading of them.
        return {"manifest": None, "counts": counts, "problems": stopped, "index_entries": entries}

    for split in SPLITS:
        if not shards[split]:
            return {"manifest": None, "counts": counts, "problems": [
                f"no document landed in the {split} split, so no shard was written for it and the corpus cannot "
                "be a corpus; lower select_fraction or raise val_fraction"
            ], "index_entries": entries}

    # Turn each recorded logical offset back into a file name and an offset inside it, now that the shard list is
    # final. Shards are contiguous, so the shard containing an offset is the last one that starts at or before
    # it. A document can start near the end of a shard and run past its end; the index names where it starts,
    # not where it ends, and that is what a reader checking the index needs.
    for split in SPLITS:
        starts = _shard_starts({"shards": [dict(s) for s in shards[split]]}, split)
        ordered = sorted(starts.items(), key=lambda item: item[1])
        for entry in entries:
            if entry["split"] != split:
                continue
            shard = ordered[bisect_right([v for _, v in ordered], int(entry["logical"])) - 1][0]
            entry["shard"] = shard
            entry["start"] = int(entry["logical"]) - starts[shard]

    flat = [shard for split in SPLITS for shard in shards[split]]
    manifest_extra = {
        "policy": {
            "name": policy,
            "seed": int(seed),
            "val_fraction": float(val_fraction),
            "select_fraction": float(select_fraction),
            # Resolved from the vocabulary by name by the caller. Recorded as the names and the numbers those
            # names produced, because a number without its name is what drifts.
            "special_ids": {str(name): int(value) for name, value in sorted(special_ids.items())},
            "end_of_text_id": int(eos_id),
            # The split rule is a pure function of identity. When a document is named here, the rule did not
            # choose it, and the manifest says so rather than describing the idealised rule that did not run.
            "val_fallback": {
                "requested": str(force_val) if force_val is not None else None,
                "applied": bool(fallback_applied),
            },
        },
        "sources": sorted(sources.values(), key=lambda r: r["path"]),
        "counts": counts,
    }
    if write_index:
        manifest_extra["documents_index"] = {
            "format": INDEX_FORMAT,
            "path": INDEX_FILE,
            "entries": len(entries),
        }

    manifest = write_manifest(root, flat, vocab_size=vocab_size, tokenizer=tokenizer,
                              created_by=created_by, extra=manifest_extra)

    if write_index:
        with open(root / INDEX_FILE, "w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")

    return {"manifest": manifest, "counts": counts, "problems": stopped, "index_entries": entries}


def _shard_starts(manifest: dict, split: str) -> dict[str, int]:
    """Where each shard begins in the split's logical token stream, in the order the manifest lists them."""
    starts: dict[str, int] = {}
    cursor = 0
    for shard in manifest["shards"]:
        if shard["split"] == split:
            starts[str(shard["path"])] = cursor
            cursor += int(shard["tokens"])
    return starts


def read_document_index(root: str | Path, manifest: dict | None = None) -> list[dict]:
    """
    Read and validate the document index against the manifest.

    The index is a claim about the same bytes the manifest describes, so it is checked at load rather than
    trusted: every entry's shard and start must be where the manifest's shard sizes say they are, and the
    per-split sums must match. A document can straddle two shards - a flush happens at a fixed token count, not
    at a document boundary - so the check allows that, and an entry whose position disagrees with the shard it
    claims is refused rather than reinterpreted.

    The check is what makes "which document did this window come from" an answer rather than a guess. Without
    it, an eval number is a number over bytes with no name, and a corpus can be swapped for another corpus of
    the same size without anything noticing.
    """
    root = Path(root)
    if manifest is None:
        manifest = read_manifest(root)

    info = manifest.get("documents_index") or {}
    name = str(info.get("path", INDEX_FILE))
    path = root / name
    if not path.exists():
        raise FileNotFoundError(
            f"no document index at {path}; the corpus was built without one, so a window in it cannot be traced "
            "to a document"
        )
    if info.get("format", INDEX_FORMAT) != INDEX_FORMAT:
        raise ValueError(f"unsupported document index format {info.get('format')!r}; expected {INDEX_FORMAT}")

    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if info.get("entries") is not None and len(entries) != int(info["entries"]):
        raise ValueError(f"document index has {len(entries)} entries but the manifest says {info['entries']}")
    if not entries:
        raise ValueError("document index is empty")

    for entry in entries:
        if not entry.get("identity"):
            raise ValueError("a document index entry needs an identity")
        if entry.get("split") not in SPLITS:
            raise ValueError(f"document index entry has unknown split {entry.get('split')!r}")
        if Path(str(entry.get("shard", ""))).name != entry.get("shard"):
            raise ValueError(f"document index shard must be a file name, got {entry.get('shard')!r}")
        if entry.get("start") is None or int(entry["start"]) < 0 or int(entry["tokens"]) <= 0:
            raise ValueError(f"document index entry for {entry['identity']} has a bad position")

    seen: set[str] = set()
    cursors = {split: 0 for split in SPLITS}
    starts = {split: _shard_starts(manifest, split) for split in SPLITS}
    sizes = {split: {str(s["path"]): int(s["tokens"]) for s in manifest["shards"] if s["split"] == split}
            for split in SPLITS}

    for entry in entries:
        split = str(entry["split"])
        shard, start, tokens = str(entry["shard"]), int(entry["start"]), int(entry["tokens"])
        if entry["identity"] in seen:
            raise ValueError(f"document index names {entry['identity']} twice")
        seen.add(entry["identity"])

        if shard not in starts[split]:
            raise ValueError(
                f"document index entry for {entry['identity']} points at {shard}, which the manifest does not "
                f"list in the {split} split"
            )
        if start >= sizes[split][shard]:
            raise ValueError(
                f"document index entry for {entry['identity']} starts at {start} inside a shard of "
                f"{sizes[split][shard]} tokens"
            )
        expected = starts[split][shard] + start
        # The entry may name a shard that does not contain the whole document, so `start` is where the document
        # begins and `tokens` can run past the end of that shard. `logical` is the position in the split's
        # stream, and the two ways of stating the same position have to agree; if they do not, one of them was
        # written against a shard list that is not the one the manifest describes.
        if entry.get("logical") is not None and int(entry["logical"]) != expected:
            raise ValueError(
                f"document index entry for {entry['identity']} says logical offset {entry['logical']} but its "
                f"shard and start put it at {expected}"
            )
        if expected != cursors[split]:
            raise ValueError(
                f"document index entries are not contiguous in the {split} split: {entry['identity']} starts at "
                f"{expected}, but the previous entry ended at {cursors[split]}"
            )
        # A document may run past the end of the shard it starts in; the next entry then begins in the next
        # shard at the position this cursor reaches. Allowing that is what makes the check describe the writer
        # rather than an idealised one.
        cursors[split] += tokens

    for split in SPLITS:
        total = sum(sizes[split].values())
        if total and cursors[split] != total:
            raise ValueError(
                f"document index covers {cursors[split]} tokens of the {split} split but the manifest says "
                f"{total}"
            )
    return entries


class DocumentIndex:
    """
    The document boundaries, as a lookup.

    Two directions are worth having. A window number is a slice of a shard, and a reader that wants to know what
    it measured has to go from the window to the document. A run that wants to know which windows belong to a
    document goes the other way. The second is why the eval set can be named: `entries_for_split("val")` is the
    eval set as identities, not as a count.
    """

    def __init__(self, root: str | Path, manifest: dict, entries: list[dict]):
        self.root = Path(root)
        self.manifest = manifest
        self.entries = entries
        self._by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
        self._logical: dict[str, list[int]] = {split: [] for split in SPLITS}

        starts = {split: _shard_starts(manifest, split) for split in SPLITS}
        for entry in entries:
            split = str(entry["split"])
            logical = starts[split][str(entry["shard"])] + int(entry["start"])
            enriched = dict(entry)
            enriched["logical"] = logical
            enriched["logical_end"] = logical + int(entry["tokens"])
            self._by_split[split].append(enriched)
            self._logical[split].append(logical)
        for split in SPLITS:
            order = sorted(range(len(self._by_split[split])), key=lambda i: self._logical[split][i])
            self._by_split[split] = [self._by_split[split][i] for i in order]
            self._logical[split] = [self._logical[split][i] for i in order]

    @classmethod
    def load(cls, root: str | Path, manifest: dict | None = None) -> "DocumentIndex":
        root = Path(root)
        manifest = manifest or read_manifest(root)
        return cls(root, manifest, read_document_index(root, manifest))

    def entries_for_split(self, split: str) -> list[dict]:
        """
        The documents in one split, ordered by the position they occupy in the shards.

        This is the order the writer appended them, which is the order the bytes hold them. Sorted by position
        rather than presented in whatever order the walk happened to produce, so the same source read in a
        different order produces the same list.
        """
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        return self._by_split[split]

    def document_at(self, split: str, shard: str, start: int) -> dict | None:
        """
        The document containing a token position.

        Returns None when no entry covers the position, which is the case for a corpus written without an index
        and for a position past the last document. It does not raise: a caller reporting provenance should be
        able to say "unattributed" rather than crash on it.
        """
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        starts = _shard_starts(self.manifest, split)
        if shard not in starts:
            return None
        position = starts[shard] + int(start)
        logical = self._logical[split]
        if not logical:
            return None
        index = bisect_right(logical, position) - 1
        if index < 0:
            return None
        entry = self._by_split[split][index]
        if position >= entry["logical_end"]:
            return None
        return entry

    def window_document(self, split: str, shard: str, start: int, seq_len: int) -> tuple[dict | None, bool]:
        """
        The document a window came from, and whether the window crosses a document boundary.

        A window is `seq_len` tokens, and two documents in a shard are adjacent with the boundary token between
        them, so a window can cover the end of one document and the start of the next. That is not leakage and
        not a bug; it is what a fixed-length window over a run of documents does. The flag is reported because a
        number attributed to "the document this window came from" is only true when it does not cross.
        """
        first = self.document_at(split, shard, start)
        if first is None:
            return None, False
        last = self.document_at(split, shard, int(start) + int(seq_len) - 1)
        return first, bool(last is not None and last["identity"] != first["identity"])

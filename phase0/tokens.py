"""
Real-text corpus path: uint32 token shards and a deterministic stream over them.

Why this exists next to `corpus.py`: the synthetic families are built to make a mechanism observable, but
their task space is finite - lookup admits 48 sequences and recurrence 48 squared. A stream over them stops
containing new sequences long before 1B tokens, and a gate that trains for 1-2B tokens on a repeating task
measures memorization, not depth. This module carries the other half of the data budget: tokenized natural
text, stored as flat uint32 shards, streamed as fixed-length windows.

Format, kept deliberately boring so any tokenizer can produce it:

  corpus/
    manifest.json
    train-000000.bin
    train-000001.bin
    val-000000.bin

Each .bin is little-endian uint32 token ids, no header. Each manifest shard records its token count, and the
loader checks the file size against it. A window never crosses a shard boundary: a shard is a document
boundary, and a target at the last token of one document must not be the first token of an unrelated next
document.

Splits are by shard, not by window. `train` and `val` therefore cannot overlap by construction, which is the
property the Phase 0 held-out loss needs. The split is not a random per-window mask that could leak a
neighbouring window into training.

Determinism: sample i is a pure function of i and the seed. `TokenStream` permutes the windows of each epoch
with an affine map whose multiplier is coprime to the window count, so an epoch visits every window exactly
once. A checkpoint resume needs no cursor: the step number says which sample comes next.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from array import array
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from phase0.corpus import Sample

# One format number, one dtype. A reader that does not recognise either refuses the corpus rather than
# interpreting the bytes as something they are not.
FORMAT = 1
DTYPE = "uint32"
SPLITS = ("train", "val")


def _canonical_digest(payload: dict) -> str:
    """
    A short digest of the corpus contents, used to tie a checkpoint to the data it trained on.

    Key order and whitespace are normalised first, so reformatting the manifest does not change the digest.
    The digest does not try to be a checksum of every token: it covers the shard names, splits, sizes and the
    tokenizer identity, which is what makes one corpus different from another in a way a resume cares about.
    A corrupted file of the same size would pass this check; the loader's size check catches truncation, and
    the tokenizer hash catches a different vocabulary.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def write_manifest(root: str | Path, shards: list[dict], vocab_size: int,
                   tokenizer: dict | None = None, created_by: str | None = None) -> dict:
    """
    Write manifest.json and return the payload that was written.

    `tokenizer` is free-form metadata (name, file hash, vocab size) recorded for provenance. The loader only
    requires the corpus-wide `vocab_size`, because that is the bound a token id has to respect.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not shards:
        raise ValueError("a corpus needs at least one shard")
    seen: set[str] = set()
    for shard in shards:
        name = str(shard["path"])
        if Path(name).name != name:
            raise ValueError(f"shard paths must be file names inside the corpus directory, got {name!r}")
        if name in seen:
            raise ValueError(f"duplicate shard: {name}")
        seen.add(name)
        if shard["split"] not in SPLITS:
            raise ValueError(f"unknown split {shard['split']!r}; expected one of {SPLITS}")
        if int(shard["tokens"]) <= 0:
            raise ValueError(f"shard {name} has no tokens")
        # The manifest is a promise about the bytes on disk. Checking it here means a writer bug is caught at
        # build time rather than hours into a run, and the reader repeats the check because a corpus can be
        # moved or truncated after it is built.
        data = root / name
        if not data.exists():
            raise FileNotFoundError(f"shard does not exist: {data}")
        expected_bytes = int(shard["tokens"]) * 4
        if data.stat().st_size != expected_bytes:
            raise ValueError(
                f"shard {name} is {data.stat().st_size} bytes but {shard['tokens']} tokens need "
                f"{expected_bytes}"
            )
    payload = {
        "format": FORMAT,
        "dtype": DTYPE,
        "byte_order": "little",
        "vocab_size": int(vocab_size),
        "tokenizer": tokenizer or {},
        "created_by": created_by or "",
        "shards": [
            {"path": str(s["path"]), "split": str(s["split"]), "tokens": int(s["tokens"])}
            for s in shards
        ],
    }
    payload["digest"] = _canonical_digest(payload)
    (root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


@dataclass(frozen=True)
class ShardRecord:
    path: str
    split: str
    tokens: int


def read_manifest(root: str | Path) -> dict:
    """
    Read and validate the manifest, including every shard's size.

    Validation happens at load rather than on first window access. A corpus that ends halfway through a
    shard produces plausible losses until the last window is reached, and a run that discovers its data is
    truncated at hour ten has wasted the run. The size check is O(number of shards), not O(tokens).
    """
    root = Path(root)
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no manifest.json in {root}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    if int(raw.get("format", -1)) != FORMAT:
        raise ValueError(f"unsupported corpus format {raw.get('format')!r}; expected {FORMAT}")
    if raw.get("dtype") != DTYPE:
        raise ValueError(f"unsupported dtype {raw.get('dtype')!r}; expected {DTYPE}")
    if raw.get("byte_order", "little") != "little":
        raise ValueError("only little-endian token shards are supported")
    if sys.byteorder != "little":
        # The writer stores native-endian uint32; both ends are checked so a big-endian machine fails loudly
        # instead of silently reading byte-swapped token ids.
        raise RuntimeError("token shards are written little-endian; this machine is big-endian")

    vocab_size = int(raw.get("vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError("manifest vocab_size must be positive")

    shards = raw.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest has no shards")
    records, seen = [], set()
    for shard in shards:
        record = ShardRecord(str(shard["path"]), str(shard["split"]), int(shard["tokens"]))
        if Path(record.path).name != record.path:
            raise ValueError(f"shard path must be a file name, got {record.path!r}")
        if record.path in seen:
            raise ValueError(f"duplicate shard: {record.path}")
        seen.add(record.path)
        if record.split not in SPLITS:
            raise ValueError(f"unknown split {record.split!r}")
        if record.tokens <= 0:
            raise ValueError(f"shard {record.path} has no tokens")
        data = root / record.path
        if not data.exists():
            raise FileNotFoundError(f"shard listed in the manifest is missing: {data}")
        size = data.stat().st_size
        if size != record.tokens * 4:
            raise ValueError(
                f"shard {record.path} is {size} bytes but the manifest says {record.tokens} uint32 tokens "
                f"({record.tokens * 4} bytes); the corpus is truncated or the manifest is stale"
            )
        records.append(record)

    missing = [s for s in SPLITS if not any(r.split == s for r in records)]
    if missing:
        raise ValueError(f"corpus is missing split(s): {', '.join(missing)}")

    raw["shards"] = [
        {"path": r.path, "split": r.split, "tokens": r.tokens} for r in records
    ]
    # Recompute the digest and compare, so an edit to the shard list after writing is visible rather than
    # silently changing the corpus identity a checkpoint was validated against.
    expected = raw.get("digest")
    if expected is None:
        raise ValueError("manifest has no digest; rebuild it with write_manifest")
    unsigned = {k: v for k, v in raw.items() if k != "digest"}
    if _canonical_digest(unsigned) != expected:
        raise ValueError("manifest digest does not match its contents; the manifest was edited")
    return raw


class ShardWriter:
    """
    Append token ids and flush them into fixed-size shard files.

    The buffer is `array('I')` rather than a Python list: 50M ids are 200 MB as uint32 instead of ~1.4 GB as
    boxed integers, and `tofile` writes the buffer without a per-token Python loop. `close()` folds a short
    remainder into the previous shard so the corpus does not end with a one-token file that no window can use.
    """

    def __init__(self, root: str | Path, split: str, shard_tokens: int = 50_000_000,
                 vocab_size: int | None = None):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        if shard_tokens < 2:
            raise ValueError("a shard must hold at least two tokens (one input plus one target)")
        if sys.byteorder != "little":
            raise RuntimeError("token shards are written little-endian; this machine is big-endian")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_tokens = int(shard_tokens)
        self.vocab_size = vocab_size
        self._buffer = array("I")
        self._shards: list[dict] = []
        self._next_index = 0

    def add(self, ids) -> None:
        """Append one document's token ids. Ids are validated against vocab_size when it is known."""
        values = array("I", ids)
        if self.vocab_size is not None and len(values):
            # max() over the array is one C loop rather than a Python loop over every id, which matters at
            # corpus scale. A tokenizer that produced an out-of-range id means the manifest's vocabulary and
            # the tokenizer disagree, and the corpus is not usable by a model with that vocabulary.
            highest = max(values)
            if highest >= self.vocab_size:
                raise ValueError(
                    f"token id {highest} is outside the declared vocabulary of {self.vocab_size}"
                )
        self._buffer.extend(values)
        while len(self._buffer) >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, count: int | None = None) -> None:
        if not self._buffer:
            return
        take = self._buffer if count is None else self._buffer[:count]
        name = f"{self.split}-{self._next_index:06d}.bin"
        with open(self.root / name, "wb") as handle:
            take.tofile(handle)
        self._shards.append({"path": name, "split": self.split, "tokens": len(take)})
        del self._buffer[: len(take)]
        self._next_index += 1

    def close(self) -> list[dict]:
        """Flush the remainder and return the shard records for `write_manifest`."""
        if self._buffer:
            if self._shards:
                # Append to the last shard rather than creating a runt. Appending is safe because the file is
                # a flat token array with no header, and the manifest is written only after this point.
                last = self._shards[-1]
                with open(self.root / last["path"], "ab") as handle:
                    self._buffer.tofile(handle)
                last["tokens"] += len(self._buffer)
                self._buffer = array("I")
            else:
                self._flush()
        return list(self._shards)

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TokenSplit:
    """
    One split, windowed at a fixed length.

    Windows are non-overlapping and sit entirely inside one shard. The window count per shard is
    `(tokens - 1) // seq_len`: each window consumes `seq_len` input tokens and needs one more token for the
    final target. A shard shorter than `seq_len + 1` contributes no windows and is reported rather than
    silently ignored.
    """

    def __init__(self, corpus: "TokenizedCorpus", name: str, seq_len: int, records: list[ShardRecord]):
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.corpus = corpus
        self.name = name
        self.seq_len = int(seq_len)
        self._records = records
        self._arrays = [corpus.array(r.path) for r in records]
        self._starts: list[int] = []
        total = 0
        for record in records:
            self._starts.append(total)
            total += max(0, (record.tokens - 1) // self.seq_len)
        self._count = total
        if not total:
            shortest = min(r.tokens for r in records)
            raise ValueError(
                f"split {name!r} has no window of {seq_len + 1} tokens; its shortest shard holds "
                f"{shortest}. Use shorter sequences or larger shards."
            )

    def __len__(self) -> int:
        return self._count

    def _locate(self, index: int) -> tuple[int, int]:
        """Shard number and token offset for a global window index."""
        if not 0 <= index < self._count:
            raise IndexError(f"window index {index} outside [0, {self._count})")
        shard_index = bisect_right(self._starts, index) - 1
        local = index - self._starts[shard_index]
        return shard_index, local * self.seq_len

    def source_window(self, index: int) -> tuple[str, int]:
        """The shard file and token offset a window comes from, for tests and for diagnostics."""
        shard_index, start = self._locate(index)
        return self._records[shard_index].path, start

    def sample(self, index: int) -> Sample:
        """
        One training sample: `seq_len` input tokens and the same number of next-token targets.

        The final target is -1, the ignore index every loss in the harness already handles. It is not a
        prediction of a sentinel token; the position simply has no in-corpus successor.
        """
        shard_index, start = self._locate(index)
        data = self._arrays[shard_index]
        tokens = data[start: start + self.seq_len].tolist()
        targets = data[start + 1: start + self.seq_len + 1].tolist()
        targets[-1] = -1
        return Sample(tuple(tokens), tuple(targets), "text", (), 0, ())

    def close(self) -> None:
        """Release the mapped files. Required before deleting the directory on Windows."""
        self._arrays = []
        self._records = []
        self.corpus = None


@lru_cache(maxsize=256)
def _cycle(seed: int, epoch: int, count: int) -> tuple[int, int]:
    """
    An affine permutation of the windows in one epoch.

    The multiplier is coprime to `count`, so `a * pos + b mod count` is a bijection: an epoch visits every
    window exactly once. Drawing a fresh (a, b) from the epoch keeps epoch 0 and epoch 1 in different orders
    without materialising a shuffled list of every window - at 1B tokens that list is millions of entries and
    the point of the stream is not to hold one.
    """
    if count <= 0:
        raise ValueError("a cycle needs at least one window")
    if count == 1:
        return 1, 0
    raw = hashlib.blake2b(f"{seed}:{epoch}".encode(), digest_size=16).digest()
    x, y = int.from_bytes(raw[:8], "little"), int.from_bytes(raw[8:], "little")
    a = 1 + x % (count - 1)
    while math.gcd(a, count) != 1:
        # 1 is always coprime, and the step moves through every non-zero residue, so this terminates.
        a = 1 + a % (count - 1)
    return a, y % count


class TokenStream:
    """
    The training stream over one split: sample i is a pure function of i and the seed.

    The interface matches `corpus.SampleStream`, so `train_arm` does not need to know which kind of data it
    is training on. `corpus_id` is recorded in a checkpoint, because resuming a real-text arm into synthetic
    data would produce a run whose loss belongs to neither gate.
    """

    def __init__(self, split: TokenSplit, seed: int = 0):
        self.split = split
        self.seed = int(seed)
        self.count = len(split)
        manifest = split.corpus.manifest
        self.corpus_id = (
            f"tokens:{manifest['digest']}:{split.name}:{split.seq_len}:{manifest.get('vocab_size')}"
        )

    def window_index(self, index: int) -> int:
        if index < 0:
            raise IndexError("sample indices are non-negative")
        epoch, position = divmod(index, self.count)
        a, b = _cycle(self.seed, epoch, self.count)
        return (a * position + b) % self.count

    def __getitem__(self, index: int) -> Sample:
        return self.split.sample(self.window_index(index))

    def sample_batch(self, start: int, count: int) -> list[Sample]:
        """The batch one training step consumes, in the order the step consumes it."""
        return [self[start + j] for j in range(count)]


class TokenizedCorpus:
    """
    A loaded corpus directory: manifest plus read-only memory maps.

    Memory maps rather than arrays in RAM because the gate budget is 1-2B tokens. At uint32 that is 4-8 GB of
    text; the operating system can page it, and the model's weights and optimizer state get the RAM instead.
    """

    def __init__(self, root: str | Path, manifest: dict):
        self.root = Path(root)
        self.manifest = manifest
        self.vocab_size = int(manifest["vocab_size"])
        self._arrays: dict[str, np.memmap] = {}
        self._splits: dict[tuple[str, int], TokenSplit] = {}

    @classmethod
    def load(cls, root: str | Path) -> "TokenizedCorpus":
        root = Path(root)
        return cls(root, read_manifest(root))

    def array(self, name: str) -> np.memmap:
        if name not in self._arrays:
            self._arrays[name] = np.memmap(self.root / name, dtype=np.uint32, mode="r")
        return self._arrays[name]

    def split(self, name: str, seq_len: int) -> TokenSplit:
        key = (name, int(seq_len))
        if key not in self._splits:
            records = [ShardRecord(s["path"], s["split"], s["tokens"])
                       for s in self.manifest["shards"] if s["split"] == name]
            if not records:
                raise ValueError(f"corpus has no {name!r} split")
            self._splits[key] = TokenSplit(self, name, seq_len, records)
        return self._splits[key]

    @property
    def digest(self) -> str:
        return str(self.manifest["digest"])

    def close(self) -> None:
        for split in self._splits.values():
            split.close()
        self._splits.clear()
        self._arrays.clear()
        # Windows keeps a mapped file open until the mapping object is collected; without this a temporary
        # corpus directory cannot be deleted in the same process.
        gc.collect()

    def __enter__(self) -> "TokenizedCorpus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

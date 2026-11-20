"""
N-gram table, decode path.

This is the static, count-based table. It is memory, not compute: building it
costs one pass over a corpus, and using it costs one hash probe per token. It is
the part that can live on storage.

Why storage is fine on a phone: the lookup is one probe per token, and the phone
generates about 5-10 tokens/s. Random UFS reads at 100-200 microseconds therefore
cost a fraction of a percent of the token budget. The binding limit is RAM for the
hot cache, not table size or lookup speed.

Reference implementation uses a dict. The engine version is a fixed-size hash table
with open addressing and eviction; the semantics below are what both must agree on,
so they are written out explicitly:

  - min_cnt filter: an n-gram enters the table only after it is seen min_cnt times.
    Without this the table fills with one-off memorized strings and the hit rate
    drops while the memory cost explodes. This is a corpus-level condition, not a
    per-conversation one: a fact's support is usually spread across conversations, so
    a repeated span appearing twice in one conversation contributes two counts and is
    pruned there, but reaches min_cnt across the corpus. prune() is therefore a
    separate step called after the whole build corpus is counted. Applying it at the
    end of each build() call would delete entries the engine keeps, and the reference
    and the engine would no longer describe the same table.
  - top_k: each entry stores only the k most frequent continuations. A full
    distribution per entry is 151k floats per entry, which is not a table, it is a
    second model.
  - order priority: longest n-gram wins. If a 4-gram hits, its continuation is more
    trustworthy than a 2-gram continuation, so fallback happens only on a miss.
  - disjointness: the build corpus must not overlap the eval sets. A large table can
    memorize benchmark answers, and measuring that as a table gain is self-deception.
"""

from collections import Counter, defaultdict

from heronlm.config import HeronConfig


class NGramEntry:
    __slots__ = ("ngram", "order", "counts", "total")

    def __init__(self, ngram: tuple, order: int):
        self.ngram = ngram
        self.order = order
        self.counts = Counter()
        self.total = 0

    def add(self, token: int) -> None:
        self.counts[token] += 1
        self.total += 1

    def top(self, k: int) -> list[tuple[int, float]]:
        """Most frequent continuations, as (token, probability)."""
        return [(tok, cnt / self.total) for tok, cnt in self.counts.most_common(k)]

    def size_bytes(self, k: int) -> int:
        """
        Engine footprint estimate. Key is a 4-byte hash; each stored continuation is
        a 3-byte token id plus a confidence byte, which is roughly 4 bytes per slot.
        """
        return 4 + len(self.counts) * 4


class NGramTable:
    def __init__(self, cfg: HeronConfig):
        self.cfg = cfg
        self.entries: dict[tuple, NGramEntry] = {}
        self.stats = Counter()

    def build(self, tokens: list[int]) -> None:
        """
        Single pass over one token stream. O(tokens * orders). Counts only; min_cnt is applied by
        prune(), once, after the whole build corpus has been counted.

        The key is the n-gram ending at position i, the continuation is tokens[i + 1]. Getting this
        off by one is the classic n-gram table bug: if the continuation is taken from inside the key,
        the table counts a token as its own continuation and the hit statistics measure nothing.
        """
        orders = sorted(self.cfg.ngram_orders, reverse=True)
        n = len(tokens)
        for i in range(n - 1):
            for order in orders:
                if i + 1 - order >= 0:
                    ngram = tuple(tokens[i + 1 - order : i + 1])
                    entry = self.entries.get(ngram)
                    if entry is None:
                        if len(self.entries) >= self.cfg.ngram_max_entries:
                            # Entry cap reached. The real engine evicts here; the reference stops
                            # adding entries so the memory budget stays visible rather than implicit.
                            self.stats["evicted"] += 1
                            continue
                        entry = NGramEntry(ngram, order)
                        self.entries[ngram] = entry
                    entry.add(tokens[i + 1])

    def prune(self) -> None:
        """
        Apply min_cnt to the accumulated counts. This is a corpus-level step, not a per-sequence one.

        Pruning during counting would change which continuations are top-k, because early rare
        continuations would be dropped before their frequency is known. Pruning at the end of each
        build() call is a different bug with the same effect: a fact repeated twice inside one
        conversation contributes two counts, which is below min_cnt, so the entry is deleted before the
        next conversation can add to it. Cross-conversation support is what the table exists to hold, so
        per-call pruning produces a table that is not the engine's table - and a harness measuring a hit
        rate over it is measuring nothing, because the facts it is asked about were never allowed in.

        lookup() and hit_stats() assume a pruned table. Build the corpus, then prune, then measure.
        """
        min_cnt = self.cfg.ngram_min_cnt
        for ngram, entry in list(self.entries.items()):
            entry.counts = Counter({t: c for t, c in entry.counts.items() if c >= min_cnt})
            if not entry.counts:
                del self.entries[ngram]
            else:
                entry.total = sum(entry.counts.values())

    def lookup(self, context: list[int]) -> tuple[tuple, list[tuple[int, float]]] | None:
        """
        Longest-hit lookup. Returns (order_hit, top_continuations) or None on miss.

        Longest-first is the rule that matters for quality: a 4-gram continuation is
        evidence from a longer span, so it should win whenever it exists.
        """
        for order in sorted(self.cfg.ngram_orders, reverse=True):
            if len(context) < order:
                continue
            ngram = tuple(context[-order:])
            entry = self.entries.get(ngram)
            if entry is not None:
                return order, entry.top(self.cfg.ngram_top_k)
        return None

    def hit_stats(self, tokens: list[int]) -> dict:
        """Hit rate per order, measured on a corpus disjoint from the build corpus."""
        per_order = defaultdict(int)
        hits = 0
        total = 0
        for i in range(len(tokens)):
            for order in sorted(self.cfg.ngram_orders, reverse=True):
                if i >= order:
                    total += 1
                    if self.entries.get(tuple(tokens[i - order : i])) is not None:
                        hits += 1
                        per_order[order] += 1
                    break
        return {
            "entries": len(self.entries),
            "lookups": total,
            "hits": hits,
            "hit_rate": hits / total if total else 0.0,
            "hits_by_order": dict(per_order),
            "bytes_estimate": sum(e.size_bytes(self.cfg.ngram_top_k) for e in self.entries.values()),
        }

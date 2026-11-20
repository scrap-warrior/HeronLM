"""
N-gram memory path: the learned hash table, which is the reduced version of the
per-layer-embeddings idea.

The full version in a large model is a parametric table sized vocab x d, which is tens of
gigabytes. The phone version is `ngram_memory_entries` x `ngram_memory_dim` in int8, about
400 MB. That reduction is not a minor detail, it is the whole feasibility argument: at full
size this component cannot exist on the device at all.

Two consequences of shrinking it, stated plainly because they are the cost rather than the win:

  - Collisions. A 1.5M-slot table over a vocab-scale key space will collide. A recalled value
    can belong to a different n-gram, so the gate has to judge whether the recall is relevant
    instead of assuming it is.
  - Quantization. A gain measured in BF16 on a GPU is not automatically a gain at int8 on a
    phone. Table-like components quantize differently from dense matrices, so Phase 5
    re-measures this after quantization.

Placement is deliberate. The read happens at block 2, near the bottom of the input block:
  - a shallow read can overlap with the previous block's compute, which is what makes the
    storage lookup free in practice;
  - the injected value survives every loop pass only because the GR branch carrying it is
    written once and rarely updated afterwards. On a single residual stream it would be
    diluted by every later sublayer.

Optimizer rule: the table rows are not weight-decayed. Sparse row updates need different
treatment than dense matrices, so only the projections go through Muon and the rows go through
Adam without weight decay.

Engine note: there is no ggml op for a hash probe, so this cannot be a normal ggml tensor. The
engine implements it host-side: probe an mmap'd blob, produce a small [d] tensor, feed it into
the graph as an input. The reference path below is the same operation in Python, which is what
makes parity testing between the two possible.
"""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from heronlm.config import HeronConfig
from heronlm.norms import NormalizedLinear


def hash_ngram(ngram: tuple, entries: int) -> int:
    """
    Map an n-gram to a table slot.

    This hash is stable so the reference implementation and the engine agree: the engine uses open
    addressing with eviction, and the same n-gram must land in the same slot in both. Collisions are
    expected and handled by the gate, not avoided.
    """
    raw = b"".join(t.to_bytes(4, "little") for t in ngram)
    h = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(h, "little") % entries


def slot_ids(tokens: torch.Tensor, cfg: HeronConfig, prefix: list[list[int]] | None = None) -> torch.Tensor:
    """
    Table slot for every position, as [B, T].

    `prefix` is what came before this span. Incremental decode processes one token at a time, but the
    n-gram key is still the rolling window ending at that token, so the key has to be built from the
    prefix plus the current span rather than from the span alone. Getting this wrong is a common
    silent bug: the numbers look plausible but the probe is keyed on a shorter n-gram.

    The engine computes the same value incrementally per token. Parity between the two is a test
    against exactly this kind of mistake.
    """
    order = max(cfg.ngram_orders)
    B, T = tokens.shape
    out = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        history = list(prefix[b]) if prefix is not None else []
        for t in range(T):
            key = (history + tokens[b, : t + 1].tolist())[-order:]
            out[b, t] = hash_ngram(tuple(key), cfg.ngram_memory_entries)
    return out


class NGramMemory(nn.Module):
    """Learned hash table, read at one shallow layer, gated into the residual stream."""

    def __init__(self, cfg: HeronConfig):
        super().__init__()
        self.cfg = cfg
        # Row-major layout mirrors what the engine will mmap: a probe reads exactly one row,
        # so the layout must be row-contiguous for the host-side lookup to be one read.
        self.table = nn.Embedding(cfg.ngram_memory_entries, cfg.ngram_memory_dim)
        nn.init.zeros_(self.table.weight)

        # Value -> residual width. Only this projection is weight-decayed, matching the rule
        # that table rows are not decayed but their projections are.
        self.value_proj = NormalizedLinear(cfg, cfg.ngram_memory_dim, cfg.d_model)

        # Contextual gate. The table is small enough to collide, so a recalled value is not
        # automatically relevant. The gate is the cosine between the recalled row and a learned
        # key direction, judged BEFORE projection, which keeps it cheap enough to compute during
        # the probe itself.
        self.gate_key = nn.Parameter(torch.zeros(cfg.ngram_memory_dim))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

    def read(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (value [B,T,dim], gate [B,T,1])."""
        value = self.table(slots)
        key = F.normalize(self.gate_key, dim=0)
        relevance = (value * key).sum(dim=-1)
        gate = torch.sigmoid(relevance + self.gate_bias).unsqueeze(-1)
        return value, gate

    def inject(self, state: torch.Tensor, slots: torch.Tensor, branch: int) -> torch.Tensor:
        """
        Add the gated value into the reserved branch.

        This is why the branch matters: a memory lookup that lands in a shared residual stream
        is diluted by every later sublayer and every loop pass. Landing in a branch that is
        written only here keeps it readable for the whole loop.
        """
        value, gate = self.read(slots)
        out = state.clone()
        out[:, branch] = out[:, branch] + gate * self.value_proj(value)
        return out

"""
HeronLM reference implementation.

Modules:
  config  - the frozen architecture, with the reason for each number
  norms   - nGPT unit normalization and weight scaling
  gr      - the widened residual stream and the branch-assignment rule
  layers  - attention (full / sliding window), SwiGLU, expert slot, block
  memory  - the learned hash table read at block 2 and gated into branch 0
  ngram   - the static count table, storage-backed, used for verified proposals
  router  - conversation-level expert choice with hysteresis and pinning
  model   - the assembled forward pass, routing, and generation
"""

from heronlm.config import FULL, TINY, HeronConfig
from heronlm.model import HeronLM

__all__ = ["FULL", "TINY", "HeronConfig", "HeronLM"]

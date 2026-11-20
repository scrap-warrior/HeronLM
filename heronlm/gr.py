"""
GR-style gated residual: the widened residual stream.

Plain residual is one stream: read it, add, write it back. GR makes the stream a set of
branches. Each sublayer reads a learned weighted combination of all branches and writes
into one branch, so early information stays readable at later layers instead of being
averaged into a single running sum.

Why this design cares about it: the n-gram memory vector is injected at block 2, and the
loop re-runs the shared block 1-4 times. With a single residual stream, that injected vector
is diluted by every later sublayer and by every loop pass. With a branch that is written once
and rarely updated afterwards, it stays readable at every iteration. That is the mechanism
that lets a memory lookup feed a looped block, and it is Phase 0 gate G0.3.

Branch assignment rule, which is what actually preserves the property:
  - branch 0 is reserved for the embedding plus the n-gram value, and only the first sublayer
    writes it. Every other block writes into branches 1..N-1.
  - if every block wrote into all branches, the reserved branch would be overwritten by the
    loop on the second pass and the rule would be meaningless.

Read weights are per block AND per loop pass. The same block computes the same way on every
pass, but the useful mixture is different on pass 1 and pass 3, so the mixture has to be a
per-loop parameter. Without that, elastic depth is only a re-execution, not a depth change.

Set `cfg.ngpt_gr_read = False` and the branches collapse to a single stream, which is the
plain nGPT stack. That is the ablation arm for gate G0.3, and it is why the read path is a
function here rather than a fixed formula inside the blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from heronlm.config import HeronConfig


def branch_assignment(cfg: HeronConfig, n_blocks: int) -> list[int]:
    """
    Which branch each block writes to.

    Block 0 owns branch 0, the reserved one; every later block rotates over the remaining branches so
    the reserved branch stays readable through the loop.

    When the GR read is off, every block must also write to branch 0. Otherwise reads take only branch
    0 while writes go to the other branches, so sublayer outputs are silently dropped. That would not
    be a plain nGPT stack, it would be a broken one, and gate G0.3 must compare against a fair arm.

    This is a function rather than a line inside ResidualStream so the GGUF metadata can encode the
    same rule. The engine has to write to the same branches the reference writes to, and a rule written
    in two places drifts.
    """
    if not cfg.ngpt_gr_read:
        return [0] * n_blocks
    return [0] + [1 + (i % (cfg.gr_branches - 1)) for i in range(n_blocks - 1)]


class ResidualStream(nn.Module):
    """The widened residual stream shared by every block in the model."""

    def __init__(self, cfg: HeronConfig, n_blocks: int):
        super().__init__()
        self.cfg = cfg
        self.branches = cfg.gr_branches
        self.read_weights = nn.Parameter(torch.ones(n_blocks, cfg.max_loop, cfg.gr_branches) / cfg.gr_branches)
        self.branch_of = branch_assignment(cfg, n_blocks)

    def init_state(self, token_vector: torch.Tensor) -> torch.Tensor:
        """
        State shape is [B, branches, T, d]: the branches exist at every position, not once per
        sequence. Branch 0 carries the token vector, the others start empty.
        """
        B, T, d = token_vector.shape
        state = torch.zeros(B, self.branches, T, d, device=token_vector.device, dtype=token_vector.dtype)
        state[:, 0] = token_vector
        return state

    def read(self, state: torch.Tensor, block: int, loop: int) -> torch.Tensor:
        if not self.cfg.ngpt_gr_read:
            # G0.3 off: single-stream behaviour.
            return state[:, 0]
        w = F.softmax(self.read_weights[block, min(loop, self.read_weights.shape[1] - 1)], dim=0)
        return (state * w.view(1, -1, 1, 1)).sum(dim=1)

    def write(self, state: torch.Tensor, block: int, value: torch.Tensor) -> torch.Tensor:
        """value is [B, T, d]; the write goes into one branch and leaves the others alone."""
        out = state.clone()
        out[:, self.branch_of[block]] = out[:, self.branch_of[block]] + value
        return out

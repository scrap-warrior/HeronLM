"""
Conversation-level expert routing.

The router does not decide per token. It decides once, at conversation start, and the
decision is frozen for the conversation. Two reasons, and the second one is a hard
constraint rather than a preference:

  1. Expert choice is a specialization judgment about the whole conversation, not about
     the current sentence. A single turn can be out of domain while the conversation is
     not, so per-turn routing produces flapping.
  2. If the mixer is GDN, its state at any position is a function of every prior token,
     so it cannot be rewound. Changing experts mid-conversation means the state built
     under the old expert is stale, and the only correct fix is to re-prefill the
     conversation under the new expert. On a phone that is a full context re-read, so the
     design avoids mid-conversation switching entirely.

So: score the pooled conversation representation against five prototype vectors, require
a confidence margin, require the decision to hold for a number of turns before switching,
and allow pinning to override everything. The generalist is slot 0 and is always resident,
which is what makes the fallback path always available.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from heronlm.config import HeronConfig


@dataclass
class RouterState:
    """The frozen routing decision for one conversation."""

    active_slot: int
    confidence: float
    # Turns the current decision must survive before a switch is allowed. This is the
    # hysteresis: a single out-of-domain turn must not move the whole conversation.
    stable_turns: int = 0
    pinned_slot: int | None = None
    history: list[tuple[int, float]] = field(default_factory=list)

    @property
    def slot(self) -> int:
        # Pinning overrides everything, including the margin and the hysteresis. It exists
        # because a user who knows the domain should be able to stop the router from
        # guessing, and because a quantized router can flip decisions for reasons that are
        # not visible from BF16 measurements.
        return self.pinned_slot if self.pinned_slot is not None else self.active_slot


class Router:
    def __init__(self, cfg: HeronConfig, prototypes: torch.Tensor | None = None):
        """
        prototypes: [n_experts, d_model] prototype vectors, one per specialization.

        These come from clustering the teacher's router activations on a domain-tagged
        sample, not from hand-picked domains. That is what gives the router its training
        target for free, and it is why the prototypes are data rather than a list of
        labels we invented.
        """
        self.cfg = cfg
        if prototypes is None:
            # Untrained placeholder. Distinct random directions so the router can be
            # exercised before clustering produces real prototypes.
            g = torch.Generator().manual_seed(0)
            prototypes = F.normalize(torch.randn(cfg.n_experts, cfg.d_model, generator=g), dim=1)
        self.prototypes = prototypes

    def score(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        cosine similarity against every prototype, then sharpened.

        pooled is the mean-normalized conversation representation. Cosine rather than a
        learned classifier head, because the prototypes ARE the classifier: keeping them as
        explicit vectors is what makes routing auditable, which matters because routing
        accuracy has to be measured on the quantized model, not only in BF16.
        """
        p = F.normalize(pooled, dim=-1)
        # [B, n_experts]. The prototypes ARE the classifier, which is what keeps routing auditable:
        # an auditable router is required because routing accuracy must be measurable, on the
        # quantized model as well as in BF16.
        sim = p @ self.prototypes.transpose(-1, -2)
        # Temperature sharpens the winner-take-most. At temperature 0.1 the argmax is
        # dominated by the top prototype, which is what makes a margin meaningful.
        return F.softmax(sim / self.cfg.router_temperature, dim=-1)

    def decide(self, pooled: torch.Tensor, state: RouterState | None = None) -> tuple[RouterState, float]:
        """Returns (state, margin). The margin is reported so flapping can be measured."""
        probs = self.score(pooled if pooled.dim() == 2 else pooled.unsqueeze(0))
        # Routing is a per-conversation decision, so the decision-time batch is one conversation.
        assert probs.shape[0] == 1, "routing decides per conversation, not per batch"
        best = int(probs.argmax(dim=-1)[0])
        conf = float(probs[0, best])
        margin = conf - float(probs[0].sort(descending=True).values[1])

        if state is None:
            # First turn of the conversation: this is the decision that gets frozen.
            return RouterState(active_slot=best, confidence=conf, history=[(best, conf)]), margin

        if state.pinned_slot is not None:
            return state, margin

        # Switch rule. Both conditions must hold: the new best must beat the current best
        # by the margin, and the decision must be stable for `router_hysteresis_turns`.
        # Without the turn requirement, a single unusual turn flips the whole conversation;
        # that is the flapping Phase 2 gate measures, and it is why the gate exists.
        current = state.active_slot
        if best == current:
            state.stable_turns += 1
        elif margin >= self.cfg.router_margin and state.stable_turns >= self.cfg.router_hysteresis_turns:
            state.active_slot = best
            state.confidence = conf
            state.stable_turns = 0
        else:
            # Not enough evidence to switch. Keep the frozen decision, but record what
            # would have happened, because the difference between these two paths is what
            # flapping looks like in the Phase 2 measurement.
            state.stable_turns += 1

        state.history.append((best, conf))
        return state, margin

"""
Sublayer building blocks: attention (full or sliding window), SwiGLU, and an MoE
expert slot.

These modules do NOT contain residual connections. The residual structure lives in
gr.py and model.py. That separation is deliberate: the loop re-enters the same
sublayers, and if a sublayer held its own residual add, the loop would double it.

Cache policy, the part that decides whether this runs on a phone:
  - full attention: cache grows with context length, so per-loop cache is a memory
    multiplier you cannot afford on a phone. Nothing is evicted from it, because a later
    query can still reach every entry it holds.
  - sliding window: growth is bounded by the window, not by context, and it is
    append-only, so it stays reusable across conversation turns.
  - a GDN state is a function of every prior token and cannot be rewound, which is
    why changing experts mid-conversation forces a re-prefill. That is why routing
    is decided at conversation start.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from heronlm.config import HeronConfig, ROPE_FREQ_BASE
from heronlm.gr import ResidualStream
from heronlm.norms import Norm, NormalizedLinear


class Cache:
    """
    Attention cache for one block.

    `positions` records the absolute position of every cached key, because the sliding
    window mask is about distance, and distance requires knowing where each cached
    entry sits, not merely how many entries there are.

    `window` is the eviction bound, not the attention mask: a window of 0 means the cache is
    unbounded, which is what a full-attention cache has to be, because a later query can still
    reach every earlier entry. So the mask is decided by the mixer mode and the bound is decided
    by whether the block is bounded by the window or grows with the context.
    """

    def __init__(self, kv_heads: int, head_dim: int, window: int, loop_slot: int = 0,
                 allow_evict: bool = True):
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.window = window
        self.loop_slot = loop_slot
        # Verification runs against copies, and eviction is a decision about tokens that may be
        # rejected, so a verification copy must not evict. The committed prefix is evicted after
        # the accepted tokens are known, which is the same state sequential decoding would have
        # reached.
        self.allow_evict = allow_evict
        self.k = None
        self.v = None
        self.positions = None

    @property
    def len(self) -> int:
        return 0 if self.positions is None else self.positions.numel()

    def append(self, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor):
        if self.k is None:
            self.k, self.v, self.positions = k, v, positions
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
            self.positions = torch.cat([self.positions, positions])

    def evict(self, newest: int, call_start: int) -> None:
        """
        Drop the entries that no later query can reach. This is the point of SWA on a phone: growth is
        bounded by the window, so the loop can pass over the same tokens without paying for the context.

        Eviction happens AFTER the attention for the current call is computed, never inside append. The
        reason is a bug this reference had and the engine will have unless the rule is written down: during
        a parallel prefill over T tokens, the earliest query row still needs the entries written at the
        start of the segment. Trimming to the last `window` entries before the mask is applied evicts
        entries that earlier rows of the same call still need, and those rows become fully masked. Softmax
        over a fully masked row is not an error, it is nan, so a converter that evicts early produces a
        model that diverges from the reference silently.

        Two rules follow from that, and both cost memory, so they are stated rather than glossed:
          - entries written by the current call are always kept, because a loop pass may re-read them.
          - entries older than the window are only safe to drop once the model has moved past them.
        So the steady-state cache is `window` entries, but the peak during a prefill longer than the
        window is `window + segment`. On a phone the peak is what has to fit, not the steady state.

        A third rule, and it is the one speculation depends on: eviction is only allowed for entries
        the model has committed. A verification copy is not committed, so it must not evict - the
        entries it would drop are decided by tokens that may be rejected. For a sliding-window cache
        this cannot change a result anyway, because an entry more than a window behind the newest
        position is outside the window for every later position too; it is not harmless for a
        full-attention cache, which is why the rule is stated for copies rather than left as a
        memory detail.
        """
        if not self.allow_evict or self.window <= 0 or self.len <= self.window:
            return
        # Positions are ascending, so the retained set is the current call's own entries plus the
        # window tail measured from the newest position.
        keep = (self.positions >= call_start) | (newest - self.positions < self.window)
        if bool(keep.all()):
            return
        idx = keep.nonzero(as_tuple=True)[0]
        self.k, self.v, self.positions = self.k[:, :, idx], self.v[:, :, idx], self.positions[idx]

    def clone(self, allow_evict: bool = True) -> "Cache":
        """Copy-on-write for speculative verification. Verification must not pollute the real state."""
        c = Cache(self.kv_heads, self.head_dim, self.window, self.loop_slot, allow_evict)
        c.k, c.v, c.positions = self.k, self.v, self.positions
        return c

    def pop_last(self) -> None:
        """
        Drop the most recent entry.

        Speculative verification needs the logits at the position of the last COMMITTED token, which
        means re-running that token. If it is already in the cache, re-appending it would duplicate it
        and change the attention result. So verification runs against a copy with the last entry
        removed, which is the exact operation rather than an approximation. This is only safe on a
        copy; rolling back a shared cache would corrupt the conversation state.
        """
        if self.positions is None:
            return
        self.k, self.v, self.positions = self.k[:, :, :-1], self.v[:, :, :-1], self.positions[:-1]

    def truncate(self, n: int) -> None:
        """Keep at most n entries, used when speculative proposals are rejected."""
        if self.positions is None:
            return
        keep = min(n, self.len)
        if keep < self.len:
            self.k, self.v, self.positions = self.k[:, :, :keep], self.v[:, :, :keep], self.positions[:keep]


def build_rope(positions: torch.Tensor, head_dim: int) -> torch.Tensor:
    """
    Rotary angles, shaped for broadcasting over [B, heads, T, head_dim].

    The whole head vector is rotated: half is rotated by the angle, the other half by the same angle,
    which is what makes the dot product depend on relative position. The frequency base is the module
    constant so the metadata can record the same number it is built from.
    """
    half = head_dim // 2
    inv = 1.0 / (ROPE_FREQ_BASE ** (torch.arange(half, device=positions.device, dtype=torch.float32) / half))
    angles = positions.to(torch.float32).unsqueeze(-1) * inv  # [T, half]
    return torch.cat([angles, angles], dim=-1).unsqueeze(0).unsqueeze(0)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of each head vector by the same angle, which makes the
    dot product depend on relative position rather than absolute position."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.cat(
        [x1 * cos[..., :half] - x2 * sin[..., :half], x2 * cos[..., half:] + x1 * sin[..., :half]],
        dim=-1,
    )


class Attention(nn.Module):
    """Causal attention, GQA, full or sliding window. Weights are shared across loop passes."""

    def __init__(self, cfg: HeronConfig, mode: str = "swa"):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.q = NormalizedLinear(cfg, cfg.d_model, cfg.heads * cfg.head_dim)
        self.k = NormalizedLinear(cfg, cfg.d_model, cfg.kv_heads * cfg.head_dim)
        self.v = NormalizedLinear(cfg, cfg.d_model, cfg.kv_heads * cfg.head_dim)
        self.o = NormalizedLinear(cfg, cfg.heads * cfg.head_dim, cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        cache: Cache | None = None,
        positions: torch.Tensor | None = None,
        loop: int = 0,
        allow_append: bool = True,
    ) -> torch.Tensor:
        B, T, d = x.shape
        cfg = self.cfg
        nq, nk, hd = cfg.heads, cfg.kv_heads, cfg.head_dim

        if positions is None:
            start = cache.len if cache is not None else 0
            positions = torch.arange(start, start + T, device=x.device)

        # Input normalization happens inside NormalizedLinear, so q, k and v all read the same
        # normalized vector without the call site normalizing it a second time.
        q = self.q(x).view(B, T, nq, hd).transpose(1, 2)
        k = self.k(x).view(B, T, nk, hd).transpose(1, 2)
        v = self.v(x).view(B, T, nk, hd).transpose(1, 2)

        angles = build_rope(positions, hd)
        q, k = apply_rope(q, angles), apply_rope(k, angles)

        if cache is not None:
            # Under the shared-cache policy, a loop pass beyond the first must NOT append:
            # the cache already holds this block's entries for these tokens from pass 0.
            # Re-appending would duplicate them and silently change what "looped" means.
            # This is the shared-cache approximation, and gate G0.5 measures what it costs.
            if allow_append:
                cache.append(k, v, positions)
            k, v, kv_pos = cache.k, cache.v, cache.positions
        else:
            kv_pos = positions

        # GQA: every query head shares one KV head group, so KV is expanded by repeat.
        rep = nq // nk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)

        # Causal mask plus window mask, expressed in absolute positions. A loop pass is
        # allowed to see the whole cached span only when the cache policy is shared:
        # under shared cache the loop attends to the same entries it already wrote, so
        # the window is measured from the current position.
        mask = kv_pos.view(1, 1, 1, -1) <= positions.view(1, 1, T, 1)
        if self.mode == "swa":
            mask = mask & (positions.view(1, 1, T, 1) - kv_pos.view(1, 1, 1, -1) < cfg.swa_window)

        scores = scores.masked_fill(~mask, float("-inf"))
        if not bool(mask.any(dim=-1).all()):
            # A row with no valid entry means an entry that row needed was evicted before the row was
            # computed. With eviction moved after the computation this cannot happen for entries written by
            # the current call, so if it fires the caller is mixing cache policies - for example a loop pass
            # reading a shared cache already trimmed past the span it is re-processing. Failing here is the
            # point: the alternative is nan, and a nan would make the parity test meaningless.
            raise ValueError(
                "fully masked attention row: the cache no longer holds what this call needs. "
                f"window={cfg.swa_window}, cached positions={kv_pos.tolist()}, query positions={positions.tolist()}"
            )
        out = F.softmax(scores, dim=-1) @ v
        if cache is not None:
            # Evict after, for the reason given in Cache.evict. Under the shared-cache policy this is also
            # where the memory cost of the loop is decided: the entries a later pass may re-read are the
            # entries this call wrote, so they cannot be dropped here.
            cache.evict(int(positions[-1]), int(positions[0]))
        out = out.transpose(1, 2).reshape(B, T, nq * hd)
        return self.o(out)


class SwiGLU(nn.Module):
    """Three matrices, no bias. Width 6144 at d_model 2048 is ~37.75M params per block."""

    def __init__(self, cfg: HeronConfig, width: int | None = None, persistent: bool = True):
        super().__init__()
        w = width or cfg.ffn_width
        self.cfg = cfg
        # `persistent` is passed down, not decided here: whether this group's bytes have a name is a fact about
        # the slot that owns it, not about the FFN. See NormalizedLinear.
        self.gate = NormalizedLinear(cfg, cfg.d_model, w, persistent=persistent)
        self.up = NormalizedLinear(cfg, cfg.d_model, w, persistent=persistent)
        self.down = NormalizedLinear(cfg, w, cfg.d_model, persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input rule lives inside NormalizedLinear, so the product between the gate branch and
        # the up branch is normalized too, which is what keeps the down projection from seeing a
        # magnitude that grows with width.
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ExpertSlot(nn.Module):
    """
    One MoE slot: a domain-specialized FFN.

    An expert is `expert_slots` of these, placed in the OUTPUT block only. Putting them
    inside the loop would multiply the two compute knobs (loop count x expert size), so
    expert params stay where they are paid once per token.

    Each slot also owns its own output norm, which is what makes hot-swapping one slot
    safe: swapping the FFN does not change how the residual read is scaled, so the rest
    of the model keeps behaving the same when a different domain expert fires.
    """

    def __init__(self, cfg: HeronConfig, slot_width: int, persistent: bool = True):
        super().__init__()
        self.ffn = SwiGLU(cfg, width=slot_width, persistent=persistent)
        # Learned per-slot output gate. A domain expert can be strong on its domain and
        # harmless elsewhere only if its contribution can be scaled down, which is what
        # this gate is for.
        scale = torch.tensor(1.0)
        if persistent:
            self.gate_scale = nn.Parameter(scale)
        else:
            self.register_buffer("gate_scale", scale, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) * self.gate_scale


class GeneralistSlot(ExpertSlot):
    """
    The generalist as the training harness sees it: a module, and not a file.

    Slot 0 is never read by `forward()`. The generalist is each expert block's base FFN - `Block.ffn`, which
    `Block.forward` falls back to when the router picks slot 0 - so it is always resident by construction
    rather than by a loading rule. This class exists so the harness can still see the generalist as a module:
    a run that reports `n_experts` experts has a module for each of them, and the parameter count the device
    budget is built on can name the generalist as a set of slots.

    Its tensors are the same shapes a routed slot has, declared non-persistent, which is the only form that is
    excluded from the saved state in both directions. `state_dict()` skips a non-persistent buffer, and
    `load_state_dict()` does not expect one. Those two have to be the same statement, because `load_state_dict`
    builds its expectation from `_parameters` through `named_modules()` rather than from `state_dict()`: an
    exclusion on the write side alone - a Parameter that `state_dict()` hides - leaves the nested modules under
    slot 0 expecting bytes the write side refused to write, so an ordinary checkpoint round trip fails with
    missing keys. Here nothing is expected at slot 0, so a key that arrives for it is recorded as unexpected
    and the load is refused: that is the checkpoint from a model where slot 0 was an ordinary expert, and it is
    refused rather than quietly half-loaded.

    Two consequences are stated plainly, and both follow from the same fact. These tensors are never trained,
    because a buffer never receives a gradient and nothing in forward() reads them; and they are never stored,
    because the converter refuses a name for slot 0 - `blk.{bid}.ffn_*` already names the same thing, and a
    second name for it would be a file with two names for one tensor. The harness counts five experts, four of
    which compute. What the harness sees at slot 0 is its own view of the generalist; the bytes that compute for
    slot 0 are the core file's base FFN.
    """

    def __init__(self, cfg: HeronConfig, slot_width: int):
        super().__init__(cfg, slot_width, persistent=False)


class Block(nn.Module):
    """
    One transformer block: attention, then an FFN or an expert slot.

    `use_gr_read` decides whether the block reads the full widened residual mixture or
    only branch 0. nGPT's own normalization already keeps sublayer inputs comparable, so
    a GR read on top of unit normalization is often redundant; Phase 0 gate G0.3 measures
    whether the branches carry information a plain stream loses. If the gate fails, this
    flag is turned off and the branches collapse to a single stream.
    """

    def __init__(
        self,
        cfg: HeronConfig,
        index: int,
        stream: ResidualStream,
        mode: str = "swa",
        slots: dict[int, ExpertSlot] | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.index = index
        self.stream = stream
        self.mode = mode
        # slots is None for non-expert blocks. When present, the block computes through
        # whichever expert the router picked. self.ffn IS the generalist's slot: the
        # generalist is not an extra module, it is the base FFN, which is what makes it
        # literally always resident and always available as the fallback.
        self.slots = slots
        self.attn = Attention(cfg, mode)
        self.ffn = SwiGLU(cfg)
        self.norm = Norm(cfg)
        # Learned reinjection gate. Reinjection is the input embedding added again on every
        # loop pass; without a gate the loop has no way to decide how much of the original
        # token identity to keep on pass 3 versus pass 1.
        #
        # The read is normalized BEFORE the reinjection is added, so the gate value is a ratio
        # between the current state and the original token identity rather than a function of
        # stream magnitude. The engine has to normalize at the same point or the same gate value
        # means something different.
        self.reinject_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, state, caches, positions, loop, active_slot, reinjection=None):
        # Attention. Each sublayer reads the residual mixture, computes, and writes into its
        # own branch rather than adding into a single running stream.
        x = self.norm(self.stream.read(state, self.index, loop))
        if reinjection is not None:
            x = x + self.reinject_gate * reinjection
        # Shared cache: only the first pass appends. Later passes read what pass 0 wrote, which
        # is the shared-cache approximation gate G0.5 measures. Re-appending would duplicate
        # entries and silently change what "looped" means.
        cache_key = (self.index, loop) if self.cfg.cache_policy == "per_loop" else (self.index, 0)
        allow_append = self.cfg.cache_policy == "per_loop" or loop == 0
        state = self.stream.write(
            state, self.index, self.attn(x, caches[cache_key], positions, loop, allow_append)
        )

        # FFN or expert slot.
        x = self.stream.read(state, self.index, loop)
        if self.slots is not None:
            # active_slot None or equal to the generalist means the base FFN runs. Routing is
            # never allowed to have no effect, so the generalist path must always exist.
            slot = self.slots.get(active_slot, self.ffn)
            value = slot(x)
        else:
            value = self.ffn(x)
        return self.stream.write(state, self.index, value)

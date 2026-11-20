"""
nGPT normalization, plus the arm it is compared against.

Two parts, and both matter for the loop.

1. Unit normalization. Sublayer inputs and the residual stream are normalized to
   unit L2 norm, so the magnitude of the stream does not grow with depth.
2. Weight scaling. Normalizing away the magnitude removes the role a LayerNorm
   scale used to play, so the weights carry a scale instead. That scale is the
   effective learning-rate knob, and it is why nGPT layers need a much larger
   multiplier than a LayerNorm layer.

Why this is not cosmetic here: a plain LayerNorm residual stack degrades when it
is run deeper than the depth it was trained at. A looped block is exactly that
situation. So the normalized parametrization is what makes elastic depth work at
all, not a garnish added on top of the loop.

The two arms gate G0.2 compares are expressed here, in one code path:
  - "ngpt": unit L2 norm, weights normalized at forward time
  - "layernorm": the plain standardized stack, weights used as stored

Both arms are deliberately parameter-free. An affine gain on the LayerNorm arm
would make the two arms differ in size as well as in the rule being tested, and
the analytic param count that the whole device budget is built on would stop
applying to both. The gain stays in the scale instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_MODES = ("ngpt", "layernorm")

# The denominator floor for unit normalization. A zero vector has no direction, so the clamp is what keeps
# the first token of a sequence from producing nan. It is a module constant rather than a default argument
# because the metadata records it for the fork: the engine has to clamp at the same value or a near-zero
# residual reads differently, and "near zero" is the only case where that shows up.
EPS = 1e-6

# The rule the engine must follow, in one place. It lives in the reference rather than in the harness
# because it is a semantic rule, not a measurement: metadata records it, and the harness compares against
# it. If the harness owned it, the parity test would depend on harness details.
PARITY_RULE = "after_dequant"


def unit_norm(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Normalize each token vector to unit length, ignoring batch and position."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class Norm(nn.Module):
    """
    Sublayer input normalization.

    No parameters, in either mode, for the reason stated in the module docstring: the arms must differ
    only in the normalization rule. The magnitude is carried by the weight scale, which is shared by
    both arms, so a loss difference between them is attributable to normalization and not to size.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.norm_mode == "layernorm":
            return F.layer_norm(x, (x.shape[-1],))
        return unit_norm(x)


def normalized_weight(weight: torch.Tensor, cfg) -> torch.Tensor:
    """
    The weight as the forward pass sees it.

    nGPT normalizes rows at forward time rather than baking them into the stored weights, which is what
    keeps training and inference identical. The GGUF converter has to reproduce the same rule relative to
    dequantization, so the order matters: quantized weights are read, dequantized, then normalized. The
    engine has to apply normalization after dequant or it will not match this reference.

    The layernorm arm uses weights as stored, which is the point of that arm.
    """
    if cfg.norm_mode == "layernorm":
        return weight * cfg.ngpt_weight_scale
    return F.normalize(weight, dim=1) * cfg.ngpt_weight_scale


class NormalizedLinear(nn.Module):
    """
    A linear layer whose weights are normalized at forward time, in the nGPT arm.

    Keeping normalization at forward time (instead of baking it into the stored weights) keeps training
    and inference identical, which matters because the GGUF converter has to reproduce the same numbers
    from quantized weights.

    Normalization belongs to the linear, not to the call site, so there is exactly one normalization
    point per sublayer use: the one inside this module. That is deliberate. The engine has to reproduce
    the same single point, and if both the sublayer and the linear normalized, the engine would have to
    guess which one the reference meant. Every sublayer use in the reference is a NormalizedLinear, so
    every sublayer input is normalized, including the inner product inside SwiGLU. The output head is
    the only place that is not a NormalizedLinear, so it normalizes explicitly.

    `persistent` is torch's word for "part of the saved state", and it is the only knob that works in both
    directions: `state_dict()` skips a non-persistent buffer, and `load_state_dict()` does not expect one.
    An `nn.Parameter` cannot be excluded from either side, because `load_state_dict` builds its expectation
    from `_parameters` rather than from `state_dict()` - a module that only overrides `state_dict()` fails a
    normal round trip with missing keys. So a non-persistent buffer is the one form that can say "this module
    holds bytes no file can name" without breaking a checkpoint round trip, which is the generalist's case.
    """

    def __init__(self, cfg, in_features: int, out_features: int, persistent: bool = True):
        super().__init__()
        self.cfg = cfg
        self.norm = Norm(cfg)
        weight = torch.empty(out_features, in_features)
        nn.init.normal_(weight, std=1.0 / in_features ** 0.5)
        if persistent:
            self.weight = nn.Parameter(weight)
        else:
            # Held by the module, outside the saved state in both directions. It also cannot be trained: a
            # buffer never receives a gradient, which is consistent with the reason it has no name - nothing in
            # forward() reads the generalist's own weights.
            self.register_buffer("weight", weight, persistent=False)
        # The scale is the sublayer's effective output magnitude. In nGPT this is the knob that
        # replaces the LayerNorm gain, and it is shared by both arms so the arms stay comparable.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.norm(x), normalized_weight(self.weight, self.cfg))

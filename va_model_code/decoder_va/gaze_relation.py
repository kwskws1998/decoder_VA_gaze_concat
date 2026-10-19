"""TRT-dependent relative rotations of mapped text attention queries and keys.

The caller supplies queries and keys after the backbone's native position
rotation. Each query head learns one zero-initialized scalar, shared across the
split-half two-dimensional planes. A token's extra phase is alpha * TRT / scale.
The returned tensor is an additive attention-logit correction, not an attention
mask or a replacement for the backbone's position encoding.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch
from torch import nn


GAZE_RELATION_METHODS = ("none", "trt-relative-rotation")
GAZE_RELATION_INPUTS = ("aligned", "shuffled")
_ENABLED_METADATA = {
    "geometry": "native_rope_then_split_half",
    "scope": "mapped_text_pairs",
    "alpha_init": 0.0,
    "compute_dtype": "float32",
}


def _positive_float(value: Real, name: str) -> float:
    """Accept only finite, positive scalar configuration values usable in FP32."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    fp32_value = torch.tensor(normalized, dtype=torch.float32)
    if not bool(torch.isfinite(fp32_value)) or not bool(fp32_value > 0):
        raise ValueError(f"{name} must be representable as a finite positive FP32 value.")
    return normalized


def gaze_relation_contract(
    method: str = "none", *, scale: float = 1.0,
    input_mode: str = "aligned", shuffle_seed: int | None = None,
) -> dict:
    """Build the explicit versioned contract for a disabled or TRT relation branch."""

    if method not in GAZE_RELATION_METHODS:
        raise ValueError(f"Unknown gaze relation method: {method!r}.")
    if input_mode not in GAZE_RELATION_INPUTS:
        raise ValueError(f"Unknown gaze relation input: {input_mode!r}.")
    if input_mode == "aligned" and shuffle_seed is not None:
        raise ValueError("shuffle_seed requires shuffled gaze relation input.")
    if input_mode == "shuffled" and (
        type(shuffle_seed) is not int or not 0 <= shuffle_seed <= 2**63 - 1
    ):
        raise ValueError("Shuffled gaze relation input requires an integer shuffle_seed in [0, 2**63 - 1].")
    normalized_scale = _positive_float(scale, "scale")
    if method == "none":
        if input_mode != "aligned":
            raise ValueError("Shuffled gaze relation input requires trt-relative-rotation.")
        if normalized_scale != 1.0:
            raise ValueError("A gaze relation scale requires trt-relative-rotation.")
        return {"version": 1, "method": "none"}
    contract = {
        "version": 1,
        "method": method,
        "scale": normalized_scale,
        **_ENABLED_METADATA,
    }
    if input_mode == "shuffled":
        from .gaze_relation_inputs import SHUFFLE_ALGORITHM

        contract.update({
            "version": 2,
            "input_mode": "shuffled",
            "shuffle_seed": shuffle_seed,
            "shuffle_algorithm": SHUFFLE_ALGORITHM,
        })
    return contract


def validate_gaze_relation_contract(contract: Mapping) -> dict:
    """Reject incomplete, extra, or semantically incompatible saved contracts."""

    if not isinstance(contract, Mapping):
        raise ValueError("gaze_relation must be a mapping.")
    method = contract.get("method")
    if method not in GAZE_RELATION_METHODS:
        raise ValueError(f"Unknown gaze relation method: {method!r}.")
    version = contract.get("version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("gaze relation version must be integer 1 or 2.")
    if version == 2 and method == "none":
        raise ValueError("gaze relation version 2 requires trt-relative-rotation.")
    expected_keys = {"version", "method"}
    if method != "none":
        expected_keys |= {"scale", *_ENABLED_METADATA}
    if version == 2:
        expected_keys |= {"input_mode", "shuffle_seed", "shuffle_algorithm"}
    if set(contract) != expected_keys:
        raise ValueError(
            "Invalid gaze relation contract fields: "
            f"missing={sorted(expected_keys.difference(contract))}, "
            f"extra={sorted(set(contract).difference(expected_keys), key=str)}."
        )
    if method == "none":
        return gaze_relation_contract()
    for name, expected in _ENABLED_METADATA.items():
        if type(contract[name]) is not type(expected) or contract[name] != expected:
            raise ValueError(f"Invalid gaze relation {name}: expected {expected!r}.")
    if version == 2:
        from .gaze_relation_inputs import SHUFFLE_ALGORITHM

        if contract["input_mode"] != "shuffled" or contract["shuffle_algorithm"] != SHUFFLE_ALGORITHM:
            raise ValueError("Version 2 requires shuffled input and the supported shuffle_algorithm.")
        return gaze_relation_contract(
            method, scale=contract["scale"], input_mode="shuffled",
            shuffle_seed=contract["shuffle_seed"],
        )
    return gaze_relation_contract(method, scale=contract["scale"])


class TRTRelativeRotation(nn.Module):
    """Return FP32-computed TRT-relative score corrections in the query dtype.

For eligible text pairs, the correction is
``scale_attn * ((R_i q_i)^T (R_j k_j) - q_i^T k_j)``. ``R_i`` rotates
coordinate pairs ``(d, d + head_dim / 2)`` by ``alpha[head] * TRT[i] / scale``.
The same phase is used in every plane. Query-head-specific rotations are applied
after repeating grouped keys, so grouped query attention needs no new key heads.

Both endpoints must be marked by the explicit text gaze mask. Invalid gaze and
query/key values are zeroed before arithmetic, preventing padding NaNs from
entering the correction. The equivalent relative-phase bilinear equation is
evaluated directly, with ``cos(phase) - 1 = -2 * sin(phase / 2)^2``, avoiding
subtraction of nearly equal dot products at initialization. Equal TRT values
and the diagonal have an exact zero correction. The caller retains
responsibility for causal and other attention masks.
"""

    def __init__(self, num_heads: int, *, scale: float = 1.0) -> None:
        """Initialize one FP32 zero phase coefficient per query attention head."""

        super().__init__()
        if isinstance(num_heads, bool) or not isinstance(num_heads, Integral) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        self.num_heads = int(num_heads)
        self.scale = _positive_float(scale, "scale")
        self.alpha = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))

    def _apply(self, fn, recurse: bool = True):
        """Keep original FP32 values and gradients while following the target device."""

        original_alpha = self.alpha.detach().float().clone()
        original_gradient = (
            None if self.alpha.grad is None else self.alpha.grad.detach().float().clone()
        )
        super()._apply(fn, recurse=recurse)
        self.alpha.data = (
            self.alpha.data.float()
            if original_alpha.is_meta
            else original_alpha.to(device=self.alpha.device, dtype=torch.float32)
        )
        if original_gradient is not None:
            self.alpha.grad = (
                self.alpha.grad.float()
                if original_gradient.is_meta
                else original_gradient.to(device=self.alpha.device, dtype=torch.float32)
            )
        return self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        trt_values: torch.Tensor,
        text_gaze_mask: torch.Tensor,
        *,
        attention_scale: float | None = None,
    ) -> torch.Tensor:
        """Evaluate stable relative-phase logits without bypassing the zero-alpha graph."""

        if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
            raise ValueError("query and key must be floating-point tensors.")
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must have shape [batch, heads, length, head_dim].")
        if not query.is_floating_point() or not key.is_floating_point():
            raise ValueError("query and key must be floating-point tensors.")
        if key.device != query.device or key.dtype != query.dtype:
            raise ValueError("query and key must have the same device and dtype.")
        batch_size, heads, length, head_dim = query.shape
        if heads != self.num_heads:
            raise ValueError("query head count must match num_heads.")
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer.")
        if key.shape[0] != batch_size or key.shape[2:] != (length, head_dim):
            raise ValueError("query and key must have matching batch, length, and head_dim.")
        key_heads = key.shape[1]
        if key_heads <= 0 or heads % key_heads:
            raise ValueError("query heads must be divisible by the positive key head count.")
        if self.alpha.device != query.device:
            raise ValueError("alpha and query must be on the same device.")
        if not isinstance(trt_values, torch.Tensor) or not trt_values.is_floating_point():
            raise ValueError("trt_values must be a floating-point tensor.")
        if trt_values.shape != (batch_size, length) or trt_values.device != query.device:
            raise ValueError("trt_values must match query batch, length, and device.")
        if not isinstance(text_gaze_mask, torch.Tensor):
            raise ValueError("text_gaze_mask must be an explicit binary tensor.")
        if text_gaze_mask.shape != (batch_size, length) or text_gaze_mask.device != query.device:
            raise ValueError("text_gaze_mask must match query batch, length, and device.")
        if text_gaze_mask.is_complex() or not bool(
            ((text_gaze_mask == 0) | (text_gaze_mask == 1)).all()
        ):
            raise ValueError("text_gaze_mask must contain only binary zero or one values.")
        score_scale = (
            1.0 / math.sqrt(head_dim)
            if attention_scale is None
            else _positive_float(attention_scale, "attention_scale")
        )
        valid = text_gaze_mask.bool()
        token_mask = valid[:, None, :, None]
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = torch.where(token_mask, query.float(), 0.0)
            k = torch.where(token_mask, key.float(), 0.0)
            gaze = torch.where(valid, trt_values.float(), 0.0)
            alpha = self.alpha.float()
            if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(k).all()):
                raise ValueError("query and key must be finite FP32 values at eligible text anchors.")
            if not bool(torch.isfinite(gaze).all()) or not bool(torch.isfinite(alpha).all()):
                raise ValueError("eligible TRT values and alpha must be finite FP32 values.")
            k = k.repeat_interleave(heads // key_heads, dim=1)
            pair_mask = valid[:, None, :, None] & valid[:, None, None, :]
            pair_mask = pair_mask & ~torch.eye(length, dtype=torch.bool, device=query.device)[None, None]
            relative_gaze = torch.where(
                pair_mask, gaze[:, None, None, :] - gaze[:, None, :, None], 0.0,
            ) / self.scale
            phase = alpha[None, :, None, None] * relative_gaze
            if not bool(torch.isfinite(phase).all()):
                raise ValueError("TRT rotation phase must remain finite in FP32.")
            k_half = torch.cat((-k[..., head_dim // 2 :], k[..., : head_dim // 2]), dim=-1)
            original_scores = torch.matmul(q, k.transpose(-2, -1))
            cross_scores = torch.matmul(q, k_half.transpose(-2, -1))
            cosine_minus_one = -2.0 * (phase * 0.5).sin().square()
            delta = torch.where(
                pair_mask,
                (cosine_minus_one * original_scores + phase.sin() * cross_scores) * score_scale,
                0.0,
            )
            if not bool(torch.isfinite(delta).all()):
                raise ValueError("TRT attention correction must remain finite in FP32.")
            return delta.to(query.dtype)

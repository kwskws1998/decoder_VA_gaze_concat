"""TRT-dependent relative rotations of mapped text and optional readout queries/keys.

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


GAZE_RELATION_METHODS = ("none", "trt-relative-rotation", "trt-readout-rotation")
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
    if method == "trt-readout-rotation":
        contract.update(version=3, scope="mapped_text_pairs_and_readout", readout_reference_trt=0.0)
    if input_mode == "shuffled":
        from .gaze_relation_inputs import SHUFFLE_ALGORITHM

        contract.update({
            "version": 3 if method == "trt-readout-rotation" else 2,
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
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("gaze relation version must be integer 1, 2 or 3.")
    if (method == "trt-readout-rotation") != (version == 3):
        raise ValueError("gaze relation version 3 requires trt-readout-rotation and vice versa.")
    if version == 2 and method == "none":
        raise ValueError("gaze relation version 2 requires trt-relative-rotation.")
    shuffled = version == 2 or "input_mode" in contract
    expected = gaze_relation_contract(
        method, scale=contract.get("scale", 1.0),
        input_mode="shuffled" if shuffled else "aligned",
        shuffle_seed=contract.get("shuffle_seed") if shuffled else None,
    )
    if set(contract) != set(expected):
        raise ValueError(
            "Invalid gaze relation contract fields: "
            f"missing={sorted(set(expected).difference(contract))}, "
            f"extra={sorted(set(contract).difference(expected), key=str)}."
        )
    for name, expected_value in expected.items():
        if name == "scale":
            continue
        if type(contract[name]) is not type(expected_value) or contract[name] != expected_value:
            raise ValueError(f"Invalid gaze relation {name}: expected {expected_value!r}.")
    return expected


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
        anchor_indices: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate stable logits on either all positions or a compact anchor layout.

        Compact layouts contain each eligible position exactly once, with inactive
        slots pointing to position zero. They may be reused across attention layers.
        Gathering precedes FP32 conversion, and only anchor pairs participate in the
        expensive matrix products. The compact correction is then scattered into
        the original attention shape. All tensor-value checks share one aggregate
        device-to-host synchronization on the successful path.
        """

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
        if text_gaze_mask.is_complex():
            raise ValueError("text_gaze_mask must contain only binary zero or one values.")
        checks = []
        if text_gaze_mask.dtype != torch.bool:
            checks.append((
                "text_gaze_mask must contain only binary zero or one values.",
                ((text_gaze_mask == 0) | (text_gaze_mask == 1)).all(),
            ))
        score_scale = (
            1.0 / math.sqrt(head_dim)
            if attention_scale is None
            else _positive_float(attention_scale, "attention_scale")
        )
        valid = text_gaze_mask.bool()
        compact = anchor_indices is not None or anchor_mask is not None
        indices = None
        if compact:
            if not isinstance(anchor_indices, torch.Tensor) or not isinstance(anchor_mask, torch.Tensor):
                raise ValueError("anchor_indices and anchor_mask must both be tensors.")
            if anchor_indices.dtype != torch.long or anchor_indices.ndim != 2:
                raise ValueError("anchor_indices must have dtype int64 and shape [batch, anchors].")
            if anchor_indices.shape[0] != batch_size or anchor_indices.device != query.device:
                raise ValueError("anchor_indices must match query batch and device.")
            if anchor_mask.shape != anchor_indices.shape or anchor_mask.device != query.device:
                raise ValueError("anchor_mask must match anchor_indices shape and device.")
            if anchor_mask.dtype != torch.bool:
                raise ValueError("anchor_mask must be a boolean tensor.")
            if length == 0 and anchor_indices.shape[1] != 0:
                raise ValueError("A zero-length sequence requires zero anchor slots.")
            checks.append((
                "anchor_indices must be in range and inactive anchor slots must use position zero.",
                (((anchor_indices >= 0) & (anchor_indices < length)) | ~anchor_mask).all()
                & ((anchor_indices == 0) | anchor_mask).all(),
            ))
            indices = anchor_indices.clamp(min=0, max=max(length - 1, 0))
            counts = torch.zeros_like(valid, dtype=torch.long)
            counts.scatter_add_(1, indices, anchor_mask.long())
            checks.append((
                "anchor layout must contain every eligible text position exactly once.",
                (counts == valid.long()).all(),
            ))
            query = query.gather(2, indices[:, None, :, None].expand(-1, heads, -1, head_dim))
            key = key.gather(2, indices[:, None, :, None].expand(-1, key_heads, -1, head_dim))
            trt_values = trt_values.gather(1, indices)
            valid = anchor_mask
        active_length = valid.shape[1]
        token_mask = valid[:, None, :, None]
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = torch.where(token_mask, query.float(), 0.0)
            k = torch.where(token_mask, key.float(), 0.0)
            gaze = torch.where(valid, trt_values.float(), 0.0)
            alpha = self.alpha.float()
            checks.extend((
                ("query and key must be finite FP32 values at eligible text anchors.",
                 torch.isfinite(q).all() & torch.isfinite(k).all()),
                ("eligible TRT values and alpha must be finite FP32 values.",
                 torch.isfinite(gaze).all() & torch.isfinite(alpha).all()),
            ))
            if compact and active_length <= 1:
                zero = (
                    q.reshape(-1)[:0].sum() + k.reshape(-1)[:0].sum()
                    + gaze.reshape(-1)[:0].sum() + alpha[:0].sum()
                )
                _validate_rotation_checks(checks)
                return zero.to(query.dtype).expand(batch_size, heads, length, length)
            k = k.repeat_interleave(heads // key_heads, dim=1)
            pair_mask = valid[:, None, :, None] & valid[:, None, None, :]
            pair_mask = pair_mask & ~torch.eye(active_length, dtype=torch.bool, device=query.device)[None, None]
            relative_gaze = torch.where(
                pair_mask, gaze[:, None, None, :] - gaze[:, None, :, None], 0.0,
            ) / self.scale
            phase = alpha[None, :, None, None] * relative_gaze
            checks.append((
                "TRT rotation phase must remain finite in FP32.", torch.isfinite(phase).all(),
            ))
            k_half = torch.cat((-k[..., head_dim // 2 :], k[..., : head_dim // 2]), dim=-1)
            original_scores = torch.matmul(q, k.transpose(-2, -1))
            cross_scores = torch.matmul(q, k_half.transpose(-2, -1))
            cosine_minus_one = -2.0 * (phase * 0.5).sin().square()
            delta = torch.where(
                pair_mask,
                (cosine_minus_one * original_scores + phase.sin() * cross_scores) * score_scale,
                0.0,
            )
            checks.append((
                "TRT attention correction must remain finite in FP32.", torch.isfinite(delta).all(),
            ))
            _validate_rotation_checks(checks)
            if indices is None:
                return delta.to(query.dtype)
            output = query.new_zeros(batch_size, heads, length * length)
            pair_indices = (
                indices[:, :, None] * length + indices[:, None, :]
            ).reshape(batch_size, 1, active_length * active_length).expand(-1, heads, -1)
            output.scatter_add_(2, pair_indices, delta.to(query.dtype).reshape(batch_size, heads, active_length * active_length))
            return output.reshape(batch_size, heads, length, length)


def _validate_rotation_checks(checks: list[tuple[str, torch.Tensor]]) -> None:
    """Synchronize tensor checks once; report individual causes only on invalid input."""

    flags = torch.stack([condition for _, condition in checks])
    if not bool(flags.all()):
        failures = [message for (message, _), passed in zip(checks, flags.tolist()) if not passed]
        raise ValueError(" ".join(failures))

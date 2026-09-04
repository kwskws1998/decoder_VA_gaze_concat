"""Mask-aware, trainable TRT redistribution before independent gaze projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from pathlib import Path

import torch
from torch import nn

from .gaze import normalize_et2_feature_indices


GAZE_REDISTRIBUTION_METHODS = ("none", "asym-gaussian")
_NUMERIC_FIELDS = ("init_sigma_left", "init_sigma_right", "min_sigma")
_DEFAULTS = (1.0, 1.0, 1e-6)
_ENABLED_METADATA = {
    "position_space": "aligned_qwen_tokens",
    "mask_policy": "valid_gaze_sources_and_targets",
    "normalization": "per_source_unit_mass",
    "feature": "TRT",
    "trainable": True,
    "compute_dtype": "float32",
}


def _positive_float(value: Real, name: str) -> float:
    """Reject ambiguous, nonfinite, and nonpositive Gaussian configuration values."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return normalized


def redistribution_contract(
    method: str = "none",
    *,
    init_sigma_left: float = 1.0,
    init_sigma_right: float = 1.0,
    min_sigma: float = 1e-6,
    gaze_fusion: str = "prefix-concat",
    feature_indices: Sequence[int] = (3,),
) -> dict:
    """Return a fully specified, serializable redistribution configuration."""

    if method not in GAZE_REDISTRIBUTION_METHODS:
        raise ValueError(f"Unknown gaze redistribution method: {method!r}.")
    values = tuple(
        _positive_float(value, name)
        for name, value in zip(
            _NUMERIC_FIELDS, (init_sigma_left, init_sigma_right, min_sigma)
        )
    )
    if method == "none":
        if values != _DEFAULTS:
            raise ValueError("Redistribution sigma settings require asym-gaussian.")
        return {"method": "none"}
    if gaze_fusion != "prefix-concat":
        raise ValueError("asym-gaussian redistribution requires prefix-concat gaze fusion.")
    if 3 not in normalize_et2_feature_indices(feature_indices):
        raise ValueError("asym-gaussian redistribution requires the TRT feature.")
    return {
        "method": method,
        **dict(zip(_NUMERIC_FIELDS, values)),
        **_ENABLED_METADATA,
    }


def validate_redistribution_contract(
    contract: Mapping,
    *,
    gaze_fusion: str,
    feature_indices: Sequence[int],
) -> dict:
    """Validate exact saved contract fields without silently repairing metadata."""

    if not isinstance(contract, Mapping):
        raise ValueError("gaze_redistribution must be a mapping.")
    method = contract.get("method")
    if method not in GAZE_REDISTRIBUTION_METHODS:
        raise ValueError(f"Unknown gaze redistribution method: {method!r}.")
    expected_keys = (
        {"method"}
        if method == "none"
        else {"method", *_NUMERIC_FIELDS, *_ENABLED_METADATA}
    )
    if set(contract) != expected_keys:
        raise ValueError(
            "Invalid gaze redistribution contract fields: "
            f"missing={sorted(expected_keys.difference(contract))}, "
            f"extra={sorted(set(contract).difference(expected_keys), key=str)}."
        )
    if method == "none":
        return {"method": "none"}
    for name, expected in _ENABLED_METADATA.items():
        actual = contract[name]
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"Invalid redistribution {name}: expected {expected!r}.")
    return redistribution_contract(
        method,
        **{name: contract[name] for name in _NUMERIC_FIELDS},
        gaze_fusion=gaze_fusion,
        feature_indices=feature_indices,
    )


def validate_redistribution_state_file(weights_path: str | Path) -> None:
    """Lazily require exactly two finite FP32 scalar widths in safe model weights.

    Validate the complete redistribution subtree and scalar shapes before reading
    the two scalar tensors. Backbone tensors are never materialized by this
    inspection, and incompatible dtypes cannot be silently cast during loading.
    """

    from safetensors import SafetensorError, safe_open

    expected_keys = {
        "gaze_redistributor.kernel.log_sigma_left",
        "gaze_redistributor.kernel.log_sigma_right",
    }
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as weights:
            actual_keys = {
                key for key in weights.keys() if key.startswith("gaze_redistributor.")
            }
            if actual_keys != expected_keys:
                raise ValueError(
                    "gaze_redistribution state keys mismatch: "
                    f"missing={sorted(expected_keys - actual_keys)}, "
                    f"unexpected={sorted(actual_keys - expected_keys)}."
                )
            for key in sorted(expected_keys):
                if weights.get_slice(key).get_shape() != []:
                    raise ValueError(f"gaze_redistribution parameter {key} must be scalar.")
            for key in sorted(expected_keys):
                parameter = weights.get_tensor(key)
                if parameter.dtype != torch.float32:
                    raise ValueError(f"gaze_redistribution parameter {key} must be FP32.")
                if not bool(torch.isfinite(parameter).item()):
                    raise ValueError(f"gaze_redistribution parameter {key} must be finite.")
    except (SafetensorError, OSError) as exc:
        raise ValueError(
            f"Cannot safely inspect gaze_redistribution checkpoint: {weights_path}."
        ) from exc


class AsymGaussianRedistributor(nn.Module):
    """Redistribute each valid source's signed TRT over valid aligned destinations.

    The Gaussian orientation and source-wise mass normalization follow the
    supplied supplementary implementation. An explicit gaze mask fixes both
    padded and unmapped tokens as invalid sources and destinations. Valid zero
    predictions remain destinations; the mask never comes from feature values.
    Aligned Qwen token offsets, including unmapped interior gaps, are retained.
    """

    def __init__(
        self,
        init_sigma_left: float = 1.0,
        init_sigma_right: float = 1.0,
        min_sigma: float = 1e-6,
    ):
        super().__init__()
        self.log_sigma_left = nn.Parameter(
            torch.tensor(
                math.log(_positive_float(init_sigma_left, "init_sigma_left")),
                dtype=torch.float32,
            )
        )
        self.log_sigma_right = nn.Parameter(
            torch.tensor(
                math.log(_positive_float(init_sigma_right, "init_sigma_right")),
                dtype=torch.float32,
            )
        )
        self.min_sigma = _positive_float(min_sigma, "min_sigma")

    @property
    def sigma_left(self) -> torch.Tensor:
        """Return the source implementation's learned left Gaussian width."""

        return self.log_sigma_left.float().exp() + self.min_sigma

    @property
    def sigma_right(self) -> torch.Tensor:
        """Return the source implementation's learned right Gaussian width."""

        return self.log_sigma_right.float().exp() + self.min_sigma

    def forward(
        self, trt_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute in FP32 and return the input dtype, with exact zero masked output.

        Log-space Gaussian arithmetic is algebraically equivalent to
        sigma=exp(log_sigma)+min_sigma. Saturating only the log squared distance
        beyond 80 avoids overflow where the FP32 kernel is already exactly zero.
        """

        if trt_values.ndim != 2 or not trt_values.is_floating_point():
            raise ValueError("trt_values must be a floating-point [B, T] tensor.")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("An explicit binary gaze attention_mask is required.")
        if attention_mask.shape != trt_values.shape:
            raise ValueError("attention_mask must have the same [B, T] shape as TRT.")
        if attention_mask.device != trt_values.device:
            raise ValueError("attention_mask and TRT must be on the same device.")
        if attention_mask.is_complex() or not bool(
            ((attention_mask == 0) | (attention_mask == 1)).all()
        ):
            raise ValueError("attention_mask must contain only binary zero/one values.")
        valid = attention_mask.bool()
        if trt_values.numel() == 0:
            return trt_values.clone()
        with torch.autocast(device_type=trt_values.device.type, enabled=False):
            x = torch.where(valid, trt_values.float(), 0.0)
            if not bool(torch.isfinite(x).all()):
                raise ValueError(
                    "TRT must be finite and representable in float32 at every valid gaze position."
                )
            length = x.shape[1]
            positions = torch.arange(length, dtype=torch.float32, device=x.device)
            difference = positions.view(1, length, 1) - positions.view(1, 1, length)
            distance = difference.abs()
            log_sigma = torch.where(
                difference < 0,
                self.log_sigma_left.float(),
                self.log_sigma_right.float(),
            )
            log_floor = log_sigma.new_tensor(math.log(self.min_sigma))
            effective_log_sigma = torch.logaddexp(log_sigma, log_floor)
            log_squared_distance = 2 * (distance.clamp_min(1).log() - effective_log_sigma)
            squared_distance = log_squared_distance.clamp_max(80).exp()
            squared_distance = torch.where(distance == 0, 0.0, squared_distance)
            weights = (-0.5 * squared_distance).exp()
            pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
            weights = torch.where(pair_valid, weights, 0.0)
            denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = weights / denominator
            redistributed = torch.bmm(weights, x.unsqueeze(-1)).squeeze(-1)
            return torch.where(valid, redistributed, 0.0).to(dtype=trt_values.dtype)


class GazeRedistributor(nn.Module):
    """Apply optional TRT-only redistribution while preserving all other channels."""

    def __init__(self, config: Mapping, feature_indices: Sequence[int]):
        super().__init__()
        self.feature_indices = tuple(feature_indices)
        normalize_et2_feature_indices(self.feature_indices)
        self.config = validate_redistribution_contract(
            config, gaze_fusion="prefix-concat", feature_indices=self.feature_indices
        )
        self.kernel = (
            None
            if self.config["method"] == "none"
            else AsymGaussianRedistributor(
                **{name: self.config[name] for name in _NUMERIC_FIELDS}
            )
        )

    def forward(self, raw_gaze: torch.Tensor, gaze_mask: torch.Tensor) -> torch.Tensor:
        """Redistribute raw TRT before projection, independently of frozen ET caching."""

        if self.kernel is None:
            return raw_gaze
        if raw_gaze.ndim != 3 or raw_gaze.shape[2] != len(self.feature_indices):
            raise ValueError("raw_gaze must be [B, T, selected_ET2_features].")
        trt_channel = self.feature_indices.index(3)
        redistributed = self.kernel(raw_gaze[:, :, trt_channel], gaze_mask)
        output = raw_gaze.clone()
        output[:, :, trt_channel] = redistributed
        return output

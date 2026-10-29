"""Archived asymmetric Gaussian TRT operator, unchanged apart from module isolation.

Original source: legacy_code/va_model_code/decoder_va/redistribution.py
The exact source SHA is recorded for comparison with the preserved archive.
"""
import math
from numbers import Real
import torch
from torch import nn

ARCHIVED_SOURCE_SHA256 = '658e7063ff6402691046c075c4394b5f6d6ae6df22d5149a0d496f73147ac0b5'

def _positive_float(value: Real, name: str) -> float:
    """Reject ambiguous, nonfinite, and nonpositive Gaussian configuration values."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return normalized


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


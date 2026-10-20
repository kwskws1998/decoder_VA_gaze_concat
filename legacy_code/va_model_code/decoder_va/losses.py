"""Loss functions for two-dimensional valence/arousal regression."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import LOSS_CHOICES
from .uncertainty import HETEROSCEDASTIC_LOSS, MIN_VARIANCE, MAX_LOG_VARIANCE


@dataclass(frozen=True)
class LossBreakdown:
    """Named scalar components returned by :func:`va_regression_loss`."""

    total: Tensor
    point: Tensor
    ccc: Tensor


def concordance_correlation_coefficient(
    predictions: Tensor,
    targets: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute one CCC value per output dimension using population moments."""

    if predictions.shape != targets.shape:
        raise ValueError(
            "CCC inputs must have identical shapes; "
            f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
        )
    if predictions.ndim != 2:
        raise ValueError(
            f"CCC expects [batch, dimensions] tensors, got {predictions.ndim}D."
        )

    predictions = predictions.float()
    targets = targets.float()
    pred_mean = predictions.mean(dim=0)
    target_mean = targets.mean(dim=0)
    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean
    covariance = (pred_centered * target_centered).mean(dim=0)
    pred_variance = pred_centered.square().mean(dim=0)
    target_variance = target_centered.square().mean(dim=0)
    denominator = (
        pred_variance
        + target_variance
        + (pred_mean - target_mean).square()
    )
    coefficient = (2.0 * covariance) / denominator.clamp_min(eps)
    return coefficient.clamp(min=-1.0, max=1.0)


def va_regression_loss(
    logits: Tensor,
    labels: Tensor,
    loss_name: str,
) -> LossBreakdown:
    """Compute FP32 VA objectives, with Gaussian NLL over two means and two log variances."""

    if labels.ndim != 2 or labels.shape[-1] != 2:
        raise ValueError(f"labels must have shape [batch, 2], got {tuple(labels.shape)}.")
    normalized_name = loss_name.lower()
    if normalized_name not in LOSS_CHOICES:
        raise ValueError(f"Unknown loss {loss_name!r}; choose one of: {', '.join(LOSS_CHOICES)}.")
    width = 4 if normalized_name == HETEROSCEDASTIC_LOSS else 2
    if logits.ndim != 2 or logits.shape[-1] != width:
        raise ValueError(f"logits must have shape [batch, {width}], got {tuple(logits.shape)}.")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must contain the same number of examples.")
    if not labels.shape[0]:
        raise ValueError("Cannot compute VA loss for an empty batch.")

    predictions = logits[:, :2].float()
    labels = labels.float()
    mse = F.mse_loss(predictions, labels)
    ccc_loss = (
        1.0
        - concordance_correlation_coefficient(predictions, labels).mean()
    )

    if normalized_name == HETEROSCEDASTIC_LOSS:
        log_variance = logits[:, 2:].float()
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(labels).all()):
            raise ValueError("Heteroscedastic loss requires finite predictions and labels.")
        if bool((log_variance < math.log(MIN_VARIANCE) - 1e-5).any()):
            raise ValueError("Predicted log variance is below the declared variance floor.")
        if bool((log_variance > MAX_LOG_VARIANCE).any()):
            raise ValueError("Predicted log variance exceeds the finite FP32 variance range.")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            total = 0.5 * (
                log_variance + (predictions - labels).square() * torch.exp(-log_variance)
                + math.log(2.0 * math.pi)
            ).mean()
        return LossBreakdown(total=total, point=mse, ccc=ccc_loss)

    if normalized_name == "mse":
        return LossBreakdown(total=mse, point=mse, ccc=ccc_loss)
    if normalized_name == "ccc":
        return LossBreakdown(total=ccc_loss, point=mse, ccc=ccc_loss)
    if normalized_name == "mse+ccc":
        total = 0.5 * (mse + ccc_loss)
        return LossBreakdown(total=total, point=mse, ccc=ccc_loss)
    raise AssertionError("Validated VA loss has no implementation.")

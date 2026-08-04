"""Loss functions for two-dimensional valence/arousal regression."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F


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
    """Compute one retained two-output VA regression objective."""

    if labels.ndim != 2 or labels.shape[-1] != 2:
        raise ValueError(f"labels must have shape [batch, 2], got {tuple(labels.shape)}.")
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError(f"logits must have shape [batch, 2], got {tuple(logits.shape)}.")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must contain the same number of examples.")

    normalized_name = loss_name.lower()
    predictions = logits.float()
    labels = labels.float()
    mse = F.mse_loss(predictions, labels)
    ccc_loss = (
        1.0
        - concordance_correlation_coefficient(predictions, labels).mean()
    )

    if normalized_name == "mse":
        return LossBreakdown(total=mse, point=mse, ccc=ccc_loss)
    if normalized_name == "ccc":
        return LossBreakdown(total=ccc_loss, point=mse, ccc=ccc_loss)
    if normalized_name == "mse+ccc":
        total = 0.5 * (mse + ccc_loss)
        return LossBreakdown(total=total, point=mse, ccc=ccc_loss)
    choices = "mse, ccc, mse+ccc"
    raise ValueError(f"Unknown loss {loss_name!r}; choose one of: {choices}.")

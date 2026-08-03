"""Loss functions for two-dimensional valence/arousal regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class LossBreakdown:
    """Named scalar components returned by :func:`va_regression_loss`."""

    total: Tensor
    point: Tensor
    ccc: Tensor
    nll: Tensor


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
    loss_name: str = "heteroscedastic+ccc",
    *,
    ccc_weight: float = 0.1,
    mse_weight: float = 0.1,
    min_log_variance: float = -5.0,
    max_log_variance: float = 3.0,
) -> LossBreakdown:
    """Compute a stable VA objective from point or distributional predictions."""

    if labels.ndim != 2 or labels.shape[-1] != 2:
        raise ValueError(f"labels must have shape [batch, 2], got {tuple(labels.shape)}.")
    if logits.ndim != 2 or logits.shape[-1] not in (2, 4):
        raise ValueError(f"logits must have shape [batch, 2 or 4], got {tuple(logits.shape)}.")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must contain the same number of examples.")

    aliases = {
        "hetero": "heteroscedastic+ccc",
        "hetero+ccc": "heteroscedastic+ccc",
        "nll+ccc": "heteroscedastic+ccc",
    }
    normalized_name = aliases.get(loss_name.lower(), loss_name.lower())
    means = logits[:, :2].float()
    labels = labels.float()
    mse = F.mse_loss(means, labels)
    ccc_loss = 1.0 - concordance_correlation_coefficient(means, labels).mean()
    zero = means.new_zeros(())

    if normalized_name == "mse":
        return LossBreakdown(total=mse, point=mse, ccc=ccc_loss, nll=zero)
    if normalized_name == "ccc":
        return LossBreakdown(total=ccc_loss, point=mse, ccc=ccc_loss, nll=zero)
    if normalized_name == "mse+ccc":
        total = 0.5 * (mse + ccc_loss)
        return LossBreakdown(total=total, point=mse, ccc=ccc_loss, nll=zero)
    if normalized_name != "heteroscedastic+ccc":
        choices = "mse, ccc, mse+ccc, heteroscedastic+ccc"
        raise ValueError(f"Unknown loss {loss_name!r}; choose one of: {choices}.")
    if logits.shape[-1] != 4:
        raise ValueError("heteroscedastic+ccc requires four logits: two means and two log variances.")

    log_variance = logits[:, 2:].float().clamp(
        min=min_log_variance,
        max=max_log_variance,
    )
    squared_error = (means - labels).square()
    nll = 0.5 * (log_variance + squared_error * torch.exp(-log_variance))
    nll = nll.mean()
    total = nll + mse_weight * mse + ccc_weight * ccc_loss
    return LossBreakdown(total=total, point=mse, ccc=ccc_loss, nll=nll)

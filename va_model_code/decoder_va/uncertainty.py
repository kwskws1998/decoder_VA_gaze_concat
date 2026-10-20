"""Versioned Gaussian VA outputs and stable, input-dependent observation variance."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


HETEROSCEDASTIC_LOSS = "heteroscedastic"
MIN_VARIANCE = 1e-6
INITIAL_VARIANCE = 0.01
MAX_LOG_VARIANCE = math.log(torch.finfo(torch.float32).max) + 1e-5
UNCERTAINTY_COLUMNS = ("log_variance_valence", "log_variance_arousal")
UNCERTAINTY_TABLE_COLUMNS = UNCERTAINTY_COLUMNS + (
    "pred_variance_valence", "pred_variance_arousal", "pred_std_valence", "pred_std_arousal",
)


def prediction_contract(loss: str | None = None) -> dict:
    """Describe VA means and optional diagonal Gaussian variance in normalized label units."""

    result = {"version": 1, "family": "point", "output_columns": ["pred_valence", "pred_arousal"]}
    if loss == HETEROSCEDASTIC_LOSS:
        result.update(
            family="diagonal_gaussian",
            output_columns=result["output_columns"] + list(UNCERTAINTY_COLUMNS),
            variance_parameterization="softplus_raw_plus_floor",
            minimum_variance=MIN_VARIANCE,
            initial_variance=INITIAL_VARIANCE,
            variance_units="squared_normalized_va_units",
            nll="mean_over_examples_and_va_dimensions_including_log_2pi",
        )
    elif loss not in (None, "mse", "ccc", "mse+ccc"):
        raise ValueError(f"Unknown loss for prediction distribution: {loss!r}.")
    return result


def validate_prediction_contract(value: Mapping) -> dict:
    """Reject ambiguous output order, variance units, or numeric stabilization semantics."""

    if not isinstance(value, Mapping) or type(value.get("version")) is not int:
        raise ValueError("prediction_distribution must be a versioned mapping.")
    family = value.get("family")
    if family not in ("point", "diagonal_gaussian"):
        raise ValueError("Unsupported prediction_distribution family.")
    expected = prediction_contract(HETEROSCEDASTIC_LOSS if family == "diagonal_gaussian" else None)
    if dict(value) != expected:
        raise ValueError("Incompatible prediction_distribution contract.")
    return expected


def saved_prediction_contract(metadata: Mapping, *, schema_version: int, context: str) -> dict:
    """Load legacy point predictions and require explicit schema-11 probabilistic metadata."""

    if type(schema_version) is not int or schema_version not in (6, 7, 8, 9, 10, 11):
        raise ValueError(f"{context}: unsupported prediction schema {schema_version!r}.")
    value = metadata.get("prediction_distribution")
    if value is None:
        if schema_version >= 11:
            raise ValueError(f"{context}: missing prediction_distribution.")
        value = prediction_contract()
    contract = validate_prediction_contract(value)
    if schema_version < 11 and contract["family"] != "point":
        raise ValueError(f"{context}: legacy schema cannot enable heteroscedastic predictions.")
    if "loss" in metadata and prediction_contract(metadata["loss"]) != contract:
        raise ValueError(f"{context}: loss disagrees with prediction_distribution.")
    return contract


def make_variance_head(hidden_size: int) -> nn.Linear:
    """Initialize variance to a shared 0.1 standard deviation without advancing the CPU RNG."""

    with torch.random.fork_rng(devices=[]):
        head = nn.Linear(hidden_size, 2, dtype=torch.float32, device="cpu")
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, math.log(math.expm1(INITIAL_VARIANCE - MIN_VARIANCE)))
    return head


def log_variance_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """Use stable softplus and a positive floor, without squashing variance with the VA activation."""

    return (F.softplus(raw.float()) + MIN_VARIANCE).log()


def validate_saved_variance_head(model: nn.Module, weights_path) -> None:
    """Fail closed on missing, unexpected, nonfinite or incompatible variance-head weights."""

    from safetensors import safe_open

    expected = {name: value for name, value in model.named_parameters() if name.startswith("variance_head.")}
    with safe_open(weights_path, framework="pt", device="cpu") as saved:
        available = {name for name in saved.keys() if name.startswith("variance_head.")}
        if available != set(expected):
            raise ValueError("Saved variance_head keys disagree with prediction_distribution.")
        for name, parameter in expected.items():
            value = saved.get_tensor(name)
            if value.shape != parameter.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"Saved variance_head parameter {name!r} must have the expected shape and finite FP32 values.")

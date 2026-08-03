"""Metrics and out-of-fold reports for valence/arousal regression."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


VA_NAMES = ("valence", "arousal")


def _as_prediction_array(predictions) -> np.ndarray:
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    array = np.asarray(predictions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 4):
        raise ValueError(
            "Predictions must have shape [examples, 2] or [examples, 4]; "
            f"got {array.shape}."
        )
    return array


def _validate_arrays(labels, predictions) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(labels, dtype=np.float64)
    prediction_array = _as_prediction_array(predictions)
    if label_array.ndim != 2 or label_array.shape[1] != 2:
        raise ValueError(f"Labels must have shape [examples, 2], got {label_array.shape}.")
    if label_array.shape[0] != prediction_array.shape[0]:
        raise ValueError("Labels and predictions must contain the same number of rows.")
    if label_array.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    if not np.isfinite(label_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("Labels and predictions must contain only finite numbers.")
    return label_array, prediction_array


def safe_pearson(first, second, eps: float = 1e-12) -> float:
    """Return Pearson r, or NaN when it is mathematically undefined."""

    first_array = np.asarray(first, dtype=np.float64).reshape(-1)
    second_array = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_array.size < 2 or first_array.size != second_array.size:
        return float("nan")
    if np.std(first_array) <= eps or np.std(second_array) <= eps:
        return float("nan")
    return float(stats.pearsonr(first_array, second_array).statistic)


def concordance_correlation(first, second, eps: float = 1e-12) -> float:
    """Return Lin's concordance correlation coefficient."""

    first_array = np.asarray(first, dtype=np.float64).reshape(-1)
    second_array = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_array.size == 0 or first_array.size != second_array.size:
        return float("nan")
    first_mean = float(np.mean(first_array))
    second_mean = float(np.mean(second_array))
    covariance = float(
        np.mean((first_array - first_mean) * (second_array - second_mean))
    )
    denominator = (
        float(np.var(first_array))
        + float(np.var(second_array))
        + (first_mean - second_mean) ** 2
    )
    if denominator <= eps:
        return float("nan")
    return float(2.0 * covariance / denominator)


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def calculate_va_metrics(
    labels,
    predictions,
    *,
    logvar_min: float = -5.0,
    logvar_max: float = 3.0,
) -> dict[str, float]:
    """Calculate legacy point metrics plus CCC and optional uncertainty metrics."""

    label_array, prediction_array = _validate_arrays(labels, predictions)
    means = prediction_array[:, :2]
    metrics: dict[str, float] = {"n_examples": int(label_array.shape[0])}
    mse_values: list[float] = []
    mae_values: list[float] = []
    pearson_values: list[float] = []
    ccc_values: list[float] = []

    for index, name in enumerate(VA_NAMES):
        targets = label_array[:, index]
        estimates = means[:, index]
        mse = float(mean_squared_error(targets, estimates))
        mae = float(mean_absolute_error(targets, estimates))
        pearson = safe_pearson(targets, estimates)
        ccc = concordance_correlation(targets, estimates)
        metrics[f"mse_{name}"] = mse
        metrics[f"rmse_{name}"] = math.sqrt(mse)
        metrics[f"mae_{name}"] = mae
        metrics[f"pearson_corr_{name}"] = pearson
        metrics[f"ccc_{name}"] = ccc
        mse_values.append(mse)
        mae_values.append(mae)
        pearson_values.append(pearson)
        ccc_values.append(ccc)

    metrics["mse_mean"] = _finite_mean(mse_values)
    metrics["mae_mean"] = _finite_mean(mae_values)
    metrics["pearson_corr_mean"] = _finite_mean(pearson_values)
    metrics["ccc_mean"] = _finite_mean(ccc_values)

    if prediction_array.shape[1] == 4:
        raw_logvars = prediction_array[:, 2:4]
        effective_logvars = np.clip(raw_logvars, logvar_min, logvar_max)
        nll_values: list[float] = []
        for index, name in enumerate(VA_NAMES):
            variance = np.exp(effective_logvars[:, index])
            squared_error = np.square(label_array[:, index] - means[:, index])
            gaussian_nll = 0.5 * (
                math.log(2.0 * math.pi)
                + effective_logvars[:, index]
                + squared_error / variance
            )
            metrics[f"gaussian_nll_{name}"] = float(np.mean(gaussian_nll))
            metrics[f"mean_logvar_{name}"] = float(np.mean(raw_logvars[:, index]))
            metrics[f"mean_variance_{name}"] = float(np.mean(variance))
            metrics[f"logvar_lower_clamp_rate_{name}"] = float(
                np.mean(raw_logvars[:, index] < logvar_min)
            )
            metrics[f"logvar_upper_clamp_rate_{name}"] = float(
                np.mean(raw_logvars[:, index] > logvar_max)
            )
            nll_values.append(metrics[f"gaussian_nll_{name}"])
        metrics["gaussian_nll_mean"] = _finite_mean(nll_values)
    return metrics


def trainer_compute_metrics(
    eval_prediction,
    *,
    logvar_min: float = -5.0,
    logvar_max: float = 3.0,
) -> dict[str, float]:
    """Transformers-compatible metrics callback."""

    return calculate_va_metrics(
        eval_prediction.label_ids,
        eval_prediction.predictions,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )


def prediction_frame(
    metadata: pd.DataFrame,
    labels,
    predictions,
    *,
    fold: int,
) -> pd.DataFrame:
    """Join predictions to the untouched evaluation-row metadata."""

    label_array, prediction_array = _validate_arrays(labels, predictions)
    if len(metadata) != label_array.shape[0]:
        raise ValueError(
            f"Metadata has {len(metadata)} rows but predictions have {label_array.shape[0]}."
        )
    required = {"index", "text", "dataset_of_origin"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}.")

    output = metadata.loc[:, ["index", "text", "dataset_of_origin"]].reset_index(drop=True)
    output.insert(1, "held_out_fold", int(fold))
    output["valence"] = label_array[:, 0]
    output["arousal"] = label_array[:, 1]
    output["pred_valence"] = prediction_array[:, 0]
    output["pred_arousal"] = prediction_array[:, 1]
    if prediction_array.shape[1] == 4:
        output["pred_logvar_valence"] = prediction_array[:, 2]
        output["pred_logvar_arousal"] = prediction_array[:, 3]
    return output


def metrics_from_prediction_frame(
    frame: pd.DataFrame,
    *,
    logvar_min: float = -5.0,
    logvar_max: float = 3.0,
) -> dict[str, float]:
    """Calculate metrics from a persisted prediction table."""

    prediction_columns = ["pred_valence", "pred_arousal"]
    if {"pred_logvar_valence", "pred_logvar_arousal"}.issubset(frame.columns):
        prediction_columns.extend(["pred_logvar_valence", "pred_logvar_arousal"])
    return calculate_va_metrics(
        frame.loc[:, ["valence", "arousal"]].to_numpy(),
        frame.loc[:, prediction_columns].to_numpy(),
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )


def write_oof_reports(
    fold_frames: Sequence[pd.DataFrame],
    output_dir: str | Path,
    run_parameters: Mapping | None = None,
    *,
    logvar_min: float = -5.0,
    logvar_max: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Write OOF predictions, overall metrics, and present-dataset metrics."""

    if not fold_frames:
        raise ValueError("At least one held-out fold prediction frame is required.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(fold_frames, ignore_index=True)
    combined = combined.sort_values(["index", "held_out_fold"], kind="stable").reset_index(
        drop=True
    )
    overall = metrics_from_prediction_frame(
        combined,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )

    source_rows = []
    for source_name, source_frame in combined.groupby("dataset_of_origin", sort=True):
        source_metrics = metrics_from_prediction_frame(
            source_frame,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        source_rows.append({"dataset_of_origin": source_name, **source_metrics})
    by_source = pd.DataFrame(source_rows)

    combined.to_csv(output_path / "oof_predictions.tsv", sep="\t", index=False)
    by_source.to_csv(output_path / "metrics_by_dataset.tsv", sep="\t", index=False)
    json_metrics = {
        key: (float(value) if np.isfinite(value) else None)
        for key, value in overall.items()
    }
    with open(output_path / "oof_metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(json_metrics, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    if run_parameters is not None:
        with open(output_path / "training_parameters.json", "w", encoding="utf-8") as output_file:
            json.dump(dict(run_parameters), output_file, indent=2, sort_keys=True)
            output_file.write("\n")
    return combined, overall, by_source

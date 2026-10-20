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

from .uncertainty import MIN_VARIANCE, MAX_LOG_VARIANCE, UNCERTAINTY_COLUMNS, UNCERTAINTY_TABLE_COLUMNS


VA_NAMES = ("valence", "arousal")


def _as_prediction_array(predictions) -> np.ndarray:
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    array = np.asarray(predictions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 4):
        raise ValueError(
            "Predictions must have shape [examples, 2] or [examples, 4] (VA means then log variances); "
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
    if prediction_array.shape[1] == 4:
        log_variance = prediction_array[:, 2:]
        if (log_variance < math.log(MIN_VARIANCE) - 1e-5).any() or (log_variance > MAX_LOG_VARIANCE).any():
            raise ValueError("Predicted log variance violates the finite FP32 variance contract.")
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
) -> dict[str, float]:
    """Score VA means separately from optional Gaussian uncertainty in normalized label units."""

    label_array, prediction_array = _validate_arrays(labels, predictions)
    metrics: dict[str, float] = {"n_examples": int(label_array.shape[0])}
    mse_values: list[float] = []
    mae_values: list[float] = []
    pearson_values: list[float] = []
    ccc_values: list[float] = []

    for index, name in enumerate(VA_NAMES):
        targets = label_array[:, index]
        estimates = prediction_array[:, index]
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
        metrics.update(calculate_uncertainty_metrics(label_array, prediction_array))
    return metrics


def calculate_uncertainty_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Report proper NLL, error association and untruncated Gaussian interval calibration."""

    metrics = {}
    residual_squared = (labels - predictions[:, :2]) ** 2
    log_variances = predictions[:, 2:]
    variances = np.exp(log_variances)
    for index, name in enumerate(VA_NAMES):
        error = residual_squared[:, index]
        variance = variances[:, index]
        std = np.sqrt(variance)
        metrics[f"gaussian_nll_{name}"] = float(np.mean(
            0.5 * (math.log(2 * math.pi) + log_variances[:, index] + error / variance)
        ))
        metrics[f"pred_variance_mean_{name}"] = float(np.mean(variance))
        metrics[f"pred_std_mean_{name}"] = float(np.mean(std))
        metrics[f"standardized_mse_{name}"] = float(np.mean(error / variance))
        metrics[f"variance_error_spearman_{name}"] = (
            float(stats.spearmanr(variance, error).statistic)
            if len(error) > 1 and np.ptp(variance) > 0 and np.ptp(error) > 0 else float("nan")
        )
        for coverage, z in ((90, 1.6448536269514722), (95, 1.959963984540054)):
            metrics[f"coverage_{coverage}_{name}"] = float(np.mean(error <= z * z * variance))
            metrics[f"interval_width_{coverage}_{name}"] = float(np.mean(2 * z * std))
    metrics["gaussian_nll_mean"] = _finite_mean([metrics[f"gaussian_nll_{name}"] for name in VA_NAMES])
    return metrics


def trainer_compute_metrics(eval_prediction) -> dict[str, float]:
    """Transformers-compatible metrics callback."""

    return calculate_va_metrics(
        eval_prediction.label_ids,
        eval_prediction.predictions,
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
    present_label_columns = set(VA_NAMES).intersection(metadata.columns)
    if present_label_columns and present_label_columns != set(VA_NAMES):
        raise ValueError("Metadata must contain both valence and arousal when gold labels are present.")
    if present_label_columns:
        expected_labels = metadata.loc[:, list(VA_NAMES)].to_numpy(dtype=np.float64)
        matching_rows = np.isclose(
            expected_labels, label_array, rtol=1e-7, atol=1e-8
        ).all(axis=1)
        if not matching_rows.all():
            positions = np.flatnonzero(~matching_rows)[:10].tolist()
            raise ValueError(
                "Prediction labels do not match evaluation metadata in row order; "
                f"mismatched row positions: {positions}. Use sequential evaluation sampling."
            )

    output = metadata.loc[:, ["index", "text", "dataset_of_origin"]].reset_index(drop=True)
    output.insert(1, "held_out_fold", int(fold))
    output["valence"] = label_array[:, 0]
    output["arousal"] = label_array[:, 1]
    output["pred_valence"] = prediction_array[:, 0]
    output["pred_arousal"] = prediction_array[:, 1]
    if prediction_array.shape[1] == 4:
        for index, name in enumerate(VA_NAMES):
            output[UNCERTAINTY_COLUMNS[index]] = prediction_array[:, index + 2]
        for index, name in enumerate(VA_NAMES):
            output[f"pred_variance_{name}"] = np.exp(prediction_array[:, index + 2])
        for index, name in enumerate(VA_NAMES):
            output[f"pred_std_{name}"] = np.exp(0.5 * prediction_array[:, index + 2])
    return output


def metrics_from_prediction_frame(frame: pd.DataFrame) -> dict[str, float]:
    """Calculate metrics from a persisted prediction table."""

    return calculate_va_metrics(
        frame.loc[:, ["valence", "arousal"]].to_numpy(),
        prediction_array_from_frame(frame),
    )


def prediction_array_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """Recover canonical model outputs and verify every exported variance/std against log variance."""

    columns = ["pred_valence", "pred_arousal"]
    present = set(UNCERTAINTY_TABLE_COLUMNS).intersection(frame.columns)
    if present:
        if present != set(UNCERTAINTY_TABLE_COLUMNS):
            raise ValueError("Incomplete uncertainty columns in prediction table.")
        columns += list(UNCERTAINTY_COLUMNS)
        log_variances = frame.loc[:, list(UNCERTAINTY_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(log_variances).all() or (log_variances > MAX_LOG_VARIANCE).any() or (log_variances < math.log(MIN_VARIANCE) - 1e-5).any():
            raise ValueError("Invalid log variance in prediction table.")
        for index, name in enumerate(VA_NAMES):
            for prefix, exponent in (("pred_variance", 1.0), ("pred_std", 0.5)):
                values = frame[f"{prefix}_{name}"].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all() or not np.allclose(values, np.exp(exponent * log_variances[:, index]), rtol=1e-6, atol=1e-12):
                    raise ValueError(f"{prefix}_{name} disagrees with log variance.")
    return frame.loc[:, columns].to_numpy(dtype=np.float64)


def write_oof_reports(
    fold_frames: Sequence[pd.DataFrame],
    output_dir: str | Path,
    run_parameters: Mapping | None = None,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Write OOF predictions, overall metrics, and present-dataset metrics."""

    if not fold_frames:
        raise ValueError("At least one held-out fold prediction frame is required.")
    if any(tuple(frame.columns) != tuple(fold_frames[0].columns) for frame in fold_frames):
        raise ValueError("Fold prediction schemas must match before writing OOF reports.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(fold_frames, ignore_index=True)
    combined = combined.sort_values(["index", "held_out_fold"], kind="stable").reset_index(
        drop=True
    )
    overall = metrics_from_prediction_frame(combined)

    source_rows = []
    for source_name, source_frame in combined.groupby("dataset_of_origin", sort=True):
        source_metrics = metrics_from_prediction_frame(source_frame)
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

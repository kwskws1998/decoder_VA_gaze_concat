"""Create a verified results-only ZIP for one canonical experiment run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Mapping, Sequence
import uuid
import zipfile

import numpy as np

if __package__:
    from .decoder_va.contracts import LOSS_CHOICES, MODEL_ALIASES
    from .decoder_va.evaluation import calculate_va_metrics
    from .decoder_va.gaze import ET2_FEATURE_NAMES
    from .decoder_va.model import (
        ARCHITECTURE_MANIFEST_VERSION,
        DEFAULT_CLASSIFIER_DROPOUT,
        DEFAULT_GAZE_PROJECTION_DIM,
        DEFAULT_GAZE_PROJECTION_DROPOUT,
        FINETUNING_MODES,
        GAZE_FUSIONS,
        GAZE_PREFIX_ORDER,
        GAZE_PREFIX_POOLING,
        OUTPUT_ACTIVATION,
    )
    from .decoder_va.paths import (
        RESULTS_ROOT,
        condition_slug,
        resolve_run_directory,
        validate_run_name,
    )
else:
    from decoder_va.contracts import LOSS_CHOICES, MODEL_ALIASES
    from decoder_va.evaluation import calculate_va_metrics
    from decoder_va.gaze import ET2_FEATURE_NAMES
    from decoder_va.model import (
        ARCHITECTURE_MANIFEST_VERSION,
        DEFAULT_CLASSIFIER_DROPOUT,
        DEFAULT_GAZE_PROJECTION_DIM,
        DEFAULT_GAZE_PROJECTION_DROPOUT,
        FINETUNING_MODES,
        GAZE_FUSIONS,
        GAZE_PREFIX_ORDER,
        GAZE_PREFIX_POOLING,
        OUTPUT_ACTIVATION,
    )
    from decoder_va.paths import (
        RESULTS_ROOT,
        condition_slug,
        resolve_run_directory,
        validate_run_name,
    )


ROOT_RESULT_FILES = (
    "training_parameters.json",
    "oof_metrics.json",
    "oof_predictions.tsv",
    "metrics_by_dataset.tsv",
)
FOLD_RESULT_FILES = (
    "run_manifest.json",
    "metrics.json",
    "predictions.tsv",
    "gpu_memory.json",
    "final_model/decoder_va_architecture.json",
)
PREDICTION_COLUMNS = (
    "index",
    "held_out_fold",
    "text",
    "dataset_of_origin",
    "valence",
    "arousal",
    "pred_valence",
    "pred_arousal",
)
METRIC_NAMES = (
    "n_examples",
    "mse_valence",
    "rmse_valence",
    "mae_valence",
    "pearson_corr_valence",
    "ccc_valence",
    "mse_arousal",
    "rmse_arousal",
    "mae_arousal",
    "pearson_corr_arousal",
    "ccc_arousal",
    "mse_mean",
    "mae_mean",
    "pearson_corr_mean",
    "ccc_mean",
)
REQUIRED_TRAINING_FIELDS = (
    "architecture_manifest_version",
    "model",
    "model_id",
    "model_revision",
    "loss",
    "finetuning_mode",
    "gaze_fusion",
    "gaze_features",
    "gaze_feature_indices",
    "features_used",
    "et_model_id",
    "et_revision",
    "et_filename",
    "output_activation",
    "output_dim",
    "dtype",
    "data_dir",
    "et_cache_size",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "attn_implementation",
    "gaze_concat_order",
    "pooling_position",
    "fold_sha256",
    "dataset_counts_after_filter",
    "excluded_dataset_names",
    "held_out_folds",
    "max_length",
    "train_batch_size",
    "eval_batch_size",
    "gradient_accumulation_steps",
    "epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "seed",
    "no_iemocap",
    "no_ieomcap",
    "effective_output_dir",
)
IEMOCAP_DATASET_NAME = "IEMOCAP sentences"
IEMOCAP_DATASET_SLUG = "iemocap"
MAX_RESULT_FILE_BYTES = 256 * 1024 * 1024
MAX_PACKAGE_INPUT_BYTES = 512 * 1024 * 1024


def _load_json_object(data: bytes, label: str) -> dict[str, object]:
    """Load one UTF-8 JSON object from an immutable artifact snapshot."""

    def unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate JSON key {key!r} in {label}.")
            output[key] = value
        return output

    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot read valid JSON from {label}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object in {label}.")
    return payload


def _read_tsv_rows(
    data: bytes,
    label: str,
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read one UTF-8 TSV snapshot with a complete rectangular schema."""

    try:
        input_file = io.StringIO(data.decode("utf-8"), newline="")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Cannot decode UTF-8 TSV data from {label}.") from exc
    reader = csv.DictReader(input_file, delimiter="\t")
    if not reader.fieldnames:
        raise ValueError(f"TSV result has no header: {label}")
    headers = tuple(reader.fieldnames)
    if len(set(headers)) != len(headers):
        raise ValueError(f"TSV result has duplicate columns: {label}")
    rows = list(reader)
    malformed = any(
        any(key is None or value is None for key, value in row.items())
        for row in rows
    )
    if malformed:
        raise ValueError(f"TSV result has non-rectangular rows: {label}")
    return headers, rows


def _require_fields(
    payload: Mapping[str, object],
    fields: Sequence[str],
    *,
    label: str,
) -> None:
    """Require a stable set of provenance or result fields."""

    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}.")


def _strict_bool(value: object, *, label: str) -> bool:
    """Accept only an actual JSON boolean."""

    if type(value) is not bool:
        raise ValueError(f"{label} must be a JSON boolean.")
    return value


def _strict_int(value: object, *, label: str, positive: bool = False) -> int:
    """Accept only an actual JSON integer and optionally require positivity."""

    if type(value) is not int:
        raise ValueError(f"{label} must be a JSON integer.")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


def _integer_number(value: object, *, label: str) -> int:
    """Accept an int or an exactly integer-valued finite float, but never bool."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an integer-valued number.")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be an integer-valued number.")
    return int(numeric)


def _finite_number(value: object, *, label: str) -> float:
    """Accept one finite JSON number, excluding booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite.")
    return numeric


def _held_out_folds(parameters: Mapping[str, object]) -> tuple[int, ...]:
    """Require the complete two-fold OOF protocol used by result packages."""

    raw_folds = parameters.get("held_out_folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ValueError("training_parameters.json must record held_out_folds.")
    folds = tuple(
        _strict_int(value, label="held_out_folds entry")
        for value in raw_folds
    )
    if len(folds) != 2 or set(folds) != {1, 2}:
        raise ValueError(
            "Results-only packages require the complete two-fold OOF protocol: "
            "held_out_folds must contain exactly 1 and 2."
        )
    return folds


def _effective_no_iemocap(parameters: Mapping[str, object]) -> bool:
    """Return the actual IEMOCAP exclusion from flags or resolved source names."""

    no_iemocap = _strict_bool(
        parameters.get("no_iemocap"),
        label="no_iemocap",
    )
    no_ieomcap = _strict_bool(
        parameters.get("no_ieomcap"),
        label="no_ieomcap",
    )
    excluded = parameters.get("excluded_dataset_names")
    resolved_exclusion = isinstance(excluded, list) and any(
        IEMOCAP_DATASET_SLUG in _dataset_slug(value)
        for value in excluded
    )
    return no_iemocap or no_ieomcap or resolved_exclusion


def _dataset_slug(value: object) -> str:
    """Normalize dataset names exactly like the active exclusion filter."""

    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _validate_training_parameters(parameters: Mapping[str, object]) -> None:
    """Validate the minimum reproducibility and condition identity contract."""

    _require_fields(
        parameters,
        REQUIRED_TRAINING_FIELDS,
        label="training_parameters.json",
    )
    _held_out_folds(parameters)
    _effective_no_iemocap(parameters)
    _strict_int(parameters["seed"], label="seed")
    for field in (
        "architecture_manifest_version",
        "output_dim",
        "max_length",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
    ):
        _strict_int(parameters[field], label=field, positive=True)
    if parameters["architecture_manifest_version"] != ARCHITECTURE_MANIFEST_VERSION:
        raise ValueError(
            "architecture_manifest_version must equal the active supported "
            f"schema version {ARCHITECTURE_MANIFEST_VERSION}."
        )
    for field in ("epochs", "learning_rate"):
        if _finite_number(parameters[field], label=field) <= 0:
            raise ValueError(f"{field} must be positive.")
    for field in ("weight_decay", "warmup_ratio"):
        if _finite_number(parameters[field], label=field) < 0:
            raise ValueError(f"{field} cannot be negative.")

    for field in (
        "model",
        "model_id",
        "model_revision",
        "loss",
        "finetuning_mode",
        "gaze_fusion",
        "et_model_id",
        "et_revision",
        "et_filename",
        "output_activation",
        "dtype",
        "data_dir",
        "effective_output_dir",
    ):
        if not isinstance(parameters[field], str) or not parameters[field]:
            raise ValueError(f"{field} must be a non-empty string.")
    if parameters["model"] not in MODEL_ALIASES:
        raise ValueError(f"model must be one of {MODEL_ALIASES}.")
    if parameters["loss"] not in LOSS_CHOICES:
        raise ValueError(f"loss must be one of {LOSS_CHOICES}.")
    if parameters["finetuning_mode"] not in FINETUNING_MODES:
        raise ValueError(f"finetuning_mode must be one of {FINETUNING_MODES}.")
    if parameters["gaze_fusion"] not in GAZE_FUSIONS:
        raise ValueError(f"gaze_fusion must be one of {GAZE_FUSIONS}.")
    if parameters["output_dim"] != 2:
        raise ValueError("output_dim must be 2 for valence/arousal regression.")
    if parameters["output_activation"] != OUTPUT_ACTIVATION:
        raise ValueError(f"output_activation must be {OUTPUT_ACTIVATION}.")
    for field in ("model_revision", "et_revision"):
        if re.fullmatch(r"[0-9a-f]{40}", str(parameters[field])) is None:
            raise ValueError(
                f"{field} must be an immutable lowercase 40-character commit."
            )
    run_name = parameters.get("run_name")
    if run_name is not None:
        if not isinstance(run_name, str):
            raise ValueError("run_name must be a string when recorded.")
        validate_run_name(run_name)
    _strict_int(parameters["et_cache_size"], label="et_cache_size")
    if parameters["et_cache_size"] < 0:
        raise ValueError("et_cache_size cannot be negative.")
    attention = parameters["attn_implementation"]
    if attention is not None and (
        not isinstance(attention, str) or not attention
    ):
        raise ValueError("attn_implementation must be null or a non-empty string.")

    gaze_features = parameters["gaze_features"]
    gaze_indices = parameters["gaze_feature_indices"]
    features_used = parameters["features_used"]
    if not isinstance(gaze_features, list) or any(
        not isinstance(value, str) for value in gaze_features
    ):
        raise ValueError("gaze_features must be a list of strings.")
    if not isinstance(gaze_indices, list):
        raise ValueError("gaze_feature_indices must be a list.")
    indices = tuple(
        _strict_int(value, label="gaze_feature_indices entry")
        for value in gaze_indices
    )
    if len(set(indices)) != len(indices) or any(
        value < 0 or value >= len(ET2_FEATURE_NAMES) for value in indices
    ):
        raise ValueError("gaze_feature_indices contains duplicates or invalid indices.")
    expected_names = [ET2_FEATURE_NAMES[index] for index in indices]
    if gaze_features != expected_names:
        raise ValueError("gaze_features does not match gaze_feature_indices.")
    if not isinstance(features_used, list) or len(features_used) != len(
        ET2_FEATURE_NAMES
    ):
        raise ValueError("features_used must contain five binary integers.")
    mask = tuple(
        _strict_int(value, label="features_used entry")
        for value in features_used
    )
    if any(value not in {0, 1} for value in mask):
        raise ValueError("features_used must contain only 0 or 1.")
    expected_mask = tuple(
        1 if index in indices else 0
        for index in range(len(ET2_FEATURE_NAMES))
    )
    if mask != expected_mask:
        raise ValueError("features_used does not match gaze_feature_indices.")
    gaze_fusion = parameters["gaze_fusion"]
    if gaze_fusion == "none" and (gaze_features or any(mask)):
        raise ValueError("A baseline run must record no active gaze features.")
    if gaze_fusion == "prefix-concat" and not gaze_features:
        raise ValueError("A gaze run must record at least one gaze feature.")
    expected_order = GAZE_PREFIX_ORDER if gaze_fusion == "prefix-concat" else None
    expected_pooling = (
        GAZE_PREFIX_POOLING
        if gaze_fusion == "prefix-concat"
        else "last_valid_text_token"
    )
    if parameters["gaze_concat_order"] != expected_order:
        raise ValueError("gaze_concat_order does not match gaze_fusion.")
    if parameters["pooling_position"] != expected_pooling:
        raise ValueError("pooling_position does not match gaze_fusion.")

    if parameters["finetuning_mode"] == "full":
        if any(
            parameters[field] is not None
            for field in ("lora_rank", "lora_alpha", "lora_dropout")
        ):
            raise ValueError(
                "Full fine-tuning must record null LoRA hyperparameters."
            )
    else:
        for field in ("lora_rank", "lora_alpha"):
            _strict_int(parameters[field], label=field, positive=True)
        dropout = _finite_number(parameters["lora_dropout"], label="lora_dropout")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1).")

    fold_hashes = parameters["fold_sha256"]
    if not isinstance(fold_hashes, dict) or set(fold_hashes) != {
        "full_dataset_fold1.csv",
        "full_dataset_fold2.csv",
    }:
        raise ValueError("fold_sha256 must record exactly the two fold files.")
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in fold_hashes.values()
    ):
        raise ValueError("fold_sha256 values must be lowercase SHA-256 strings.")
    dataset_counts = parameters["dataset_counts_after_filter"]
    if not isinstance(dataset_counts, dict) or not dataset_counts:
        raise ValueError("dataset_counts_after_filter must be a non-empty object.")
    for source, count in dataset_counts.items():
        if not isinstance(source, str) or not source:
            raise ValueError("dataset_counts_after_filter has an invalid source name.")
        _strict_int(count, label=f"dataset count for {source}", positive=True)
    excluded = parameters["excluded_dataset_names"]
    if not isinstance(excluded, list) or any(
        not isinstance(value, str) or not value for value in excluded
    ):
        raise ValueError(
            "excluded_dataset_names must be a list of non-empty strings."
        )
    if len(set(excluded)) != len(excluded):
        raise ValueError("excluded_dataset_names cannot contain duplicates.")
    if _effective_no_iemocap(parameters) and not any(
        IEMOCAP_DATASET_SLUG in _dataset_slug(value)
        for value in excluded
    ):
        raise ValueError(
            f"A no-IEMOCAP run must record {IEMOCAP_DATASET_NAME} in "
            "excluded_dataset_names."
        )


def _safe_read_file(path: Path, run_root: Path) -> bytes:
    """Read one regular, single-link file inside the run into an immutable snapshot."""

    try:
        relative_path = path.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"Result file is outside the run directory: {path}") from exc
    current_path = run_root
    for component in relative_path.parts:
        current_path = current_path / component
        if current_path.is_symlink():
            raise ValueError(
                f"Results package refuses symbolic links in artifact paths: "
                f"{current_path}"
            )
    if not path.is_file():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    resolved_root = run_root.resolve()
    try:
        path.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Result file escapes the run directory: {path}") from exc
    with path.open("rb") as input_file:
        file_stat = os.fstat(input_file.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"Result artifact is not a regular file: {path}")
        if file_stat.st_nlink != 1:
            raise ValueError(f"Results package refuses hard-linked files: {path}")
        if file_stat.st_size > MAX_RESULT_FILE_BYTES:
            raise ValueError(
                f"Result file exceeds the {MAX_RESULT_FILE_BYTES}-byte limit: {path}"
            )
        chunks = []
        total_size = 0
        while True:
            chunk = input_file.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_RESULT_FILE_BYTES:
                raise ValueError(
                    f"Result file grew beyond the size limit while reading: {path}"
                )
            chunks.append(chunk)
    return b"".join(chunks)


def _select_result_paths(
    run_dir: Path,
    parameters: Mapping[str, object],
) -> tuple[Path, ...]:
    """Select the fixed evidence allowlist and reject stale fold directories."""

    recorded_folds = set(_held_out_folds(parameters))
    existing_folds = set()
    for child in run_dir.iterdir():
        if not child.name.startswith("heldout_fold"):
            continue
        match = re.fullmatch(r"heldout_fold([12])", child.name)
        if match is None or child.is_symlink() or not child.is_dir():
            raise ValueError(f"Unexpected held-out fold artifact: {child}")
        existing_folds.add(int(match.group(1)))
    if existing_folds != recorded_folds:
        raise ValueError(
            "Held-out fold directories do not match training_parameters.json: "
            f"recorded {sorted(recorded_folds)}, found {sorted(existing_folds)}."
        )

    relative_paths = [Path(name) for name in ROOT_RESULT_FILES]
    for fold in sorted(recorded_folds):
        fold_root = Path(f"heldout_fold{fold}")
        relative_paths.extend(fold_root / name for name in FOLD_RESULT_FILES)
        checkpoint_root = run_dir / fold_root / "checkpoints"
        if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory is missing or unsafe: {checkpoint_root}"
            )
        trainer_states = sorted(
            checkpoint_root.glob("checkpoint-*/trainer_state.json")
        )
        if not trainer_states:
            raise FileNotFoundError(
                f"No checkpoint trainer_state.json found below {checkpoint_root}."
            )
        relative_paths.extend(
            path.relative_to(run_dir)
            for path in trainer_states
        )
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("Result allowlist contains duplicate paths.")
    return tuple(run_dir / relative_path for relative_path in relative_paths)


def _snapshot_result_files(
    run_dir: Path,
    selected_paths: Sequence[Path],
    *,
    training_parameters: bytes,
) -> dict[str, bytes]:
    """Snapshot every selected file once so validation and ZIP bytes cannot diverge."""

    snapshots: dict[str, bytes] = {}
    total_size = 0
    for path in selected_paths:
        relative_name = path.relative_to(run_dir).as_posix()
        data = (
            training_parameters
            if relative_name == "training_parameters.json"
            else _safe_read_file(path, run_dir)
        )
        total_size += len(data)
        if total_size > MAX_PACKAGE_INPUT_BYTES:
            raise ValueError(
                "Selected result files exceed the package input-size limit of "
                f"{MAX_PACKAGE_INPUT_BYTES} bytes."
            )
        snapshots[relative_name] = data
    return snapshots


def _prediction_numeric_values(
    rows: Sequence[Mapping[str, str]],
    *,
    label: str,
) -> None:
    """Validate prediction identity, source, and bounded finite VA values."""

    for row_number, row in enumerate(rows, start=2):
        index = row["index"]
        if re.fullmatch(r"[0-9]+", index) is None:
            raise ValueError(f"{label}:{row_number} has an invalid index.")
        dataset_source = row["dataset_of_origin"]
        if not dataset_source or dataset_source != dataset_source.strip():
            raise ValueError(
                f"{label}:{row_number} has an empty or untrimmed dataset source."
            )
        for field in ("valence", "arousal", "pred_valence", "pred_arousal"):
            try:
                value = float(row[field])
            except ValueError as exc:
                raise ValueError(
                    f"{label}:{row_number} has non-numeric {field}."
                ) from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{label}:{row_number} has out-of-range {field}: {value}."
                )


def _calculate_metrics(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, float | int | None]:
    """Recompute metrics with the exact production evaluator semantics."""

    if not rows:
        raise ValueError("Cannot calculate metrics for an empty prediction table.")
    labels = [
        [float(row["valence"]), float(row["arousal"])]
        for row in rows
    ]
    predictions = [
        [float(row["pred_valence"]), float(row["pred_arousal"])]
        for row in rows
    ]
    computed = calculate_va_metrics(labels, np.asarray(predictions, dtype=np.float64))
    result: dict[str, float | int | None] = {}
    for name, value in computed.items():
        if name == "n_examples":
            result[name] = int(value)
            continue
        numeric = float(value)
        result[name] = numeric if math.isfinite(numeric) else None
    return result


def _assert_metrics_match(
    recorded: Mapping[str, object],
    computed: Mapping[str, float | int | None],
    *,
    prefix: str,
    label: str,
) -> None:
    """Require recorded metrics to match recomputation from packaged predictions."""

    for name in METRIC_NAMES:
        recorded_name = f"{prefix}{name}"
        if recorded_name not in recorded:
            raise ValueError(f"{label} is missing {recorded_name}.")
        if name == "n_examples":
            actual_count = _integer_number(
                recorded[recorded_name],
                label=f"{label}.{recorded_name}",
            )
            if actual_count != computed[name]:
                raise ValueError(
                    f"{label}.{recorded_name} does not match predictions."
                )
            continue
        expected_value = computed[name]
        if expected_value is None:
            if recorded[recorded_name] is not None:
                raise ValueError(
                    f"{label}.{recorded_name} must be null because the metric "
                    "is mathematically undefined."
                )
            continue
        actual = _finite_number(
            recorded[recorded_name],
            label=f"{label}.{recorded_name}",
        )
        expected = float(expected_value)
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{label}.{recorded_name} does not match prediction recomputation."
            )


def _validate_architecture(
    architecture: Mapping[str, object],
    parameters: Mapping[str, object],
    *,
    fold: int,
) -> None:
    """Cross-check the saved-model architecture identity against the run contract."""

    required = (
        "schema_version",
        "decoder_model_id",
        "decoder_commit",
        "finetuning_mode",
        "gaze_fusion",
        "gaze_features",
        "gaze_feature_indices",
        "features_used",
        "et_model",
        "gaze_concat_order",
        "pooling_position",
        "output_activation",
        "output_names",
        "reconstruction",
        "state_dict",
    )
    _require_fields(
        architecture,
        required,
        label=f"fold {fold} architecture manifest",
    )
    schema_version = _strict_int(
        architecture["schema_version"],
        label=f"fold {fold} architecture schema_version",
        positive=True,
    )
    if schema_version != parameters["architecture_manifest_version"]:
        raise ValueError(
            f"Fold {fold} architecture schema_version disagrees with "
            "architecture_manifest_version."
        )
    expected = {
        "decoder_model_id": parameters["model_id"],
        "decoder_commit": parameters["model_revision"],
        "finetuning_mode": parameters["finetuning_mode"],
        "gaze_fusion": parameters["gaze_fusion"],
        "gaze_features": parameters["gaze_features"],
        "gaze_feature_indices": parameters["gaze_feature_indices"],
        "features_used": parameters["features_used"],
        "gaze_concat_order": parameters["gaze_concat_order"],
        "pooling_position": parameters["pooling_position"],
        "output_activation": parameters["output_activation"],
        "output_names": ["valence", "arousal"],
    }
    for field, expected_value in expected.items():
        if architecture[field] != expected_value:
            raise ValueError(
                f"Fold {fold} architecture disagrees with {field}."
            )

    reconstruction = architecture["reconstruction"]
    if not isinstance(reconstruction, dict):
        raise ValueError(
            f"Fold {fold} architecture reconstruction must be an object."
        )
    reconstruction_expected = {
        "decoder_model_id": parameters["model_id"],
        "decoder_revision": parameters["model_revision"],
        "finetuning_mode": parameters["finetuning_mode"],
        "gaze_fusion": parameters["gaze_fusion"],
        "et_repo_id": parameters["et_model_id"],
        "et_revision": parameters["et_revision"],
        "et_filename": parameters["et_filename"],
        "et_feature_names": parameters["gaze_features"],
        "et_feature_indices": parameters["gaze_feature_indices"],
        "features_used": parameters["features_used"],
        "et_cache_size": parameters["et_cache_size"],
        "output_dim": parameters["output_dim"],
        "output_activation": parameters["output_activation"],
        "lora_rank": parameters["lora_rank"],
        "lora_alpha": parameters["lora_alpha"],
        "lora_dropout": parameters["lora_dropout"],
        "lora_target_modules": (
            "all-linear"
            if parameters["finetuning_mode"] == "lora"
            else None
        ),
        "lora_task_type": (
            "FEATURE_EXTRACTION"
            if parameters["finetuning_mode"] == "lora"
            else None
        ),
        "gaze_projection_dim": DEFAULT_GAZE_PROJECTION_DIM,
        "gaze_projection_dropout": list(DEFAULT_GAZE_PROJECTION_DROPOUT),
        "classifier_dropout": DEFAULT_CLASSIFIER_DROPOUT,
        "attn_implementation": parameters["attn_implementation"],
        "backbone_dtype_at_construction": parameters["dtype"],
    }
    _require_fields(
        reconstruction,
        tuple(reconstruction_expected),
        label=f"fold {fold} architecture reconstruction",
    )
    for field, expected_value in reconstruction_expected.items():
        if reconstruction[field] != expected_value:
            raise ValueError(
                f"Fold {fold} architecture reconstruction disagrees with {field}."
            )

    expected_et_model: dict[str, object] | None = None
    if parameters["gaze_fusion"] == "prefix-concat":
        expected_et_model = {
            "repo_id": parameters["et_model_id"],
            "revision": parameters["et_revision"],
            "filename": parameters["et_filename"],
            "feature_names": parameters["gaze_features"],
            "feature_indices": parameters["gaze_feature_indices"],
            "features_used": parameters["features_used"],
        }
    if architecture["et_model"] != expected_et_model:
        raise ValueError(f"Fold {fold} architecture disagrees with et_model.")
    expected_state_dict = {
        "filename": "model.safetensors",
        "format": "safetensors",
        "scope": (
            "complete DecoderVARegressor state_dict; "
            "ET2 remains external and frozen"
        ),
        "strict_loading": True,
    }
    if architecture["state_dict"] != expected_state_dict:
        raise ValueError(f"Fold {fold} architecture has an invalid state_dict contract.")


def _validate_result_contract(
    run_dir: Path,
    parameters: Mapping[str, object],
    snapshots: Mapping[str, bytes],
) -> None:
    """Reject mixed, malformed, or numerically inconsistent result artifacts."""

    recorded_run_name = parameters.get("run_name")
    if recorded_run_name is not None and recorded_run_name != run_dir.name:
        raise ValueError(
            "Run directory name does not match training_parameters.json: "
            f"{run_dir.name!r} != {recorded_run_name!r}."
        )
    effective_output = parameters["effective_output_dir"]
    if Path(str(effective_output)).name != run_dir.name:
        raise ValueError(
            "Run directory name does not match effective_output_dir in "
            "training_parameters.json."
        )

    prediction_headers: tuple[str, ...] | None = None
    fold_prediction_records: list[tuple[str, ...]] = []
    seen_indices: set[str] = set()
    for fold in _held_out_folds(parameters):
        fold_prefix = f"heldout_fold{fold}"
        manifest_label = f"{fold_prefix}/run_manifest.json"
        manifest = _load_json_object(snapshots[manifest_label], manifest_label)
        held_out_fold = _strict_int(
            manifest.get("held_out_fold"),
            label=f"fold {fold} held_out_fold",
        )
        if held_out_fold != fold:
            raise ValueError(f"{manifest_label} records the wrong held-out fold.")
        for field, expected_value in parameters.items():
            if field not in manifest:
                raise ValueError(
                    f"Fold {fold} manifest is missing training parameter {field}."
                )
            if manifest[field] != expected_value:
                raise ValueError(
                    f"Fold {fold} manifest disagrees with training parameters "
                    f"for {field}."
                )
        evaluation_rows = _strict_int(
            manifest.get("evaluation_rows"),
            label=f"fold {fold} evaluation_rows",
            positive=True,
        )
        training_fold = _strict_int(
            manifest.get("training_fold"),
            label=f"fold {fold} training_fold",
        )
        if training_fold != 3 - fold:
            raise ValueError(
                f"{manifest_label} must train the fold opposite held-out fold {fold}."
            )
        fold_seed = _strict_int(
            manifest.get("fold_seed"),
            label=f"fold {fold} fold_seed",
        )
        expected_fold_seed = int(parameters["seed"]) + fold - 1
        if fold_seed != expected_fold_seed:
            raise ValueError(
                f"{manifest_label} fold_seed does not match the paired seed policy."
            )
        training_rows = _strict_int(
            manifest.get("training_rows"),
            label=f"fold {fold} training_rows",
            positive=True,
        )
        total_rows = sum(parameters["dataset_counts_after_filter"].values())
        if training_rows != total_rows - evaluation_rows:
            raise ValueError(
                f"{manifest_label} training_rows does not match the opposite fold."
            )

        prediction_label = f"{fold_prefix}/predictions.tsv"
        headers, prediction_rows = _read_tsv_rows(
            snapshots[prediction_label],
            prediction_label,
        )
        if headers != PREDICTION_COLUMNS:
            raise ValueError(
                f"{prediction_label} must contain exactly the eight prediction "
                "columns in canonical order."
            )
        if prediction_headers is None:
            prediction_headers = headers
        elif headers != prediction_headers:
            raise ValueError("Fold prediction TSV schemas do not match.")
        if len(prediction_rows) != evaluation_rows:
            raise ValueError(
                f"Fold {fold} predictions contain {len(prediction_rows)} rows; "
                f"manifest records {evaluation_rows}."
            )
        _prediction_numeric_values(prediction_rows, label=prediction_label)
        for row in prediction_rows:
            if row["held_out_fold"] != str(fold):
                raise ValueError(
                    f"Fold {fold} predictions contain a different held_out_fold value."
                )
            index = row["index"]
            if index in seen_indices:
                raise ValueError(
                    f"Prediction index is duplicated across folds: {index}."
                )
            seen_indices.add(index)
            fold_prediction_records.append(tuple(row[name] for name in headers))

        metrics_label = f"{fold_prefix}/metrics.json"
        fold_metrics = _load_json_object(snapshots[metrics_label], metrics_label)
        _assert_metrics_match(
            fold_metrics,
            _calculate_metrics(prediction_rows),
            prefix="test_",
            label=metrics_label,
        )
        gpu_label = f"{fold_prefix}/gpu_memory.json"
        gpu_memory = _load_json_object(snapshots[gpu_label], gpu_label)
        cuda_enabled = _strict_bool(
            gpu_memory.get("cuda_enabled"),
            label=f"{gpu_label}.cuda_enabled",
        )
        if cuda_enabled:
            for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
                if _integer_number(
                    gpu_memory.get(field),
                    label=f"{gpu_label}.{field}",
                ) < 0:
                    raise ValueError(f"{gpu_label}.{field} cannot be negative.")
        architecture_label = (
            f"{fold_prefix}/final_model/decoder_va_architecture.json"
        )
        architecture = _load_json_object(
            snapshots[architecture_label],
            architecture_label,
        )
        _validate_architecture(architecture, parameters, fold=fold)
        trainer_states = [
            name
            for name in snapshots
            if name.startswith(f"{fold_prefix}/checkpoints/checkpoint-")
            and name.endswith("/trainer_state.json")
        ]
        if not trainer_states:
            raise ValueError(f"Fold {fold} has no packaged trainer state.")
        checkpoint_contracts: dict[str, tuple[str, int]] = {}
        for trainer_label in trainer_states:
            match = re.fullmatch(
                rf"{fold_prefix}/checkpoints/(checkpoint-([1-9][0-9]*))/"
                r"trainer_state\.json",
                trainer_label,
            )
            if match is None:
                raise ValueError(
                    f"Fold {fold} has a malformed checkpoint path: {trainer_label}."
                )
            checkpoint_contracts[trainer_label] = (
                match.group(1),
                int(match.group(2)),
            )
        checkpoint_names = {
            checkpoint_name
            for checkpoint_name, _ in checkpoint_contracts.values()
        }
        for trainer_label in trainer_states:
            trainer_state = _load_json_object(
                snapshots[trainer_label],
                trainer_label,
            )
            _require_fields(
                trainer_state,
                ("global_step", "best_metric", "best_model_checkpoint"),
                label=trainer_label,
            )
            global_step = _strict_int(
                trainer_state["global_step"],
                label=f"{trainer_label}.global_step",
                positive=True,
            )
            checkpoint_name, checkpoint_step = checkpoint_contracts[trainer_label]
            if global_step != checkpoint_step:
                raise ValueError(
                    f"{trainer_label}.global_step does not match {checkpoint_name}."
                )
            _finite_number(
                trainer_state["best_metric"],
                label=f"{trainer_label}.best_metric",
            )
            best_checkpoint = trainer_state["best_model_checkpoint"]
            if not isinstance(best_checkpoint, str) or not best_checkpoint:
                raise ValueError(
                    f"{trainer_label}.best_model_checkpoint must be a non-empty path."
                )
            best_parts = Path(best_checkpoint).parts
            best_name = Path(best_checkpoint).name
            expected_suffix = (
                run_dir.name,
                fold_prefix,
                "checkpoints",
                best_name,
            )
            if (
                best_name not in checkpoint_names
                or len(best_parts) < len(expected_suffix)
                or tuple(best_parts[-4:]) != expected_suffix
            ):
                raise ValueError(
                    f"{trainer_label}.best_model_checkpoint does not identify "
                    "a packaged checkpoint for this held-out fold."
                )

    assert prediction_headers is not None
    oof_label = "oof_predictions.tsv"
    oof_headers, oof_rows = _read_tsv_rows(snapshots[oof_label], oof_label)
    if oof_headers != prediction_headers:
        raise ValueError("OOF and fold prediction TSV schemas do not match.")
    _prediction_numeric_values(oof_rows, label=oof_label)
    oof_records = [
        tuple(row[name] for name in oof_headers)
        for row in oof_rows
    ]
    if Counter(oof_records) != Counter(fold_prediction_records):
        raise ValueError(
            "OOF predictions are not the exact union of the fold predictions."
        )

    oof_metrics = _load_json_object(
        snapshots["oof_metrics.json"],
        "oof_metrics.json",
    )
    _assert_metrics_match(
        oof_metrics,
        _calculate_metrics(oof_rows),
        prefix="",
        label="oof_metrics.json",
    )

    dataset_label = "metrics_by_dataset.tsv"
    dataset_headers, dataset_rows = _read_tsv_rows(
        snapshots[dataset_label],
        dataset_label,
    )
    expected_dataset_columns = ("dataset_of_origin", *METRIC_NAMES)
    if dataset_headers != expected_dataset_columns or not dataset_rows:
        raise ValueError(
            "metrics_by_dataset.tsv must contain exactly the canonical metric "
            "columns and at least one row."
        )
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in oof_rows:
        grouped_rows[row["dataset_of_origin"]].append(row)
    recorded_sources = [row["dataset_of_origin"] for row in dataset_rows]
    if (
        len(set(recorded_sources)) != len(recorded_sources)
        or set(recorded_sources) != set(grouped_rows)
    ):
        raise ValueError(
            "metrics_by_dataset.tsv sources do not match OOF predictions."
        )
    observed_counts = {
        source: len(source_rows)
        for source, source_rows in grouped_rows.items()
    }
    if observed_counts != parameters["dataset_counts_after_filter"]:
        raise ValueError(
            "dataset_counts_after_filter does not match the complete two-fold "
            "OOF predictions."
        )
    excluded_sources = set(parameters["excluded_dataset_names"])
    leaked_sources = sorted(excluded_sources.intersection(grouped_rows))
    if leaked_sources:
        raise ValueError(
            "OOF predictions contain excluded dataset source(s): "
            + ", ".join(leaked_sources)
            + "."
        )
    if _effective_no_iemocap(parameters):
        iemocap_sources = sorted(
            source
            for source in grouped_rows
            if IEMOCAP_DATASET_SLUG in _dataset_slug(source)
        )
        if iemocap_sources:
            raise ValueError(
                "OOF predictions violate the no-IEMOCAP condition: "
                + ", ".join(iemocap_sources)
                + "."
            )
    for row in dataset_rows:
        source = row["dataset_of_origin"]
        recorded_metrics: dict[str, object] = {}
        for name in METRIC_NAMES:
            raw_value = row[name]
            if name == "n_examples":
                if re.fullmatch(r"[0-9]+", raw_value) is None:
                    raise ValueError(
                        f"metrics_by_dataset.tsv[{source}].n_examples "
                        "must be an integer."
                    )
                recorded_metrics[name] = int(raw_value)
            else:
                if raw_value == "":
                    recorded_metrics[name] = None
                else:
                    try:
                        recorded_metrics[name] = float(raw_value)
                    except ValueError as exc:
                        raise ValueError(
                            f"metrics_by_dataset.tsv[{source}].{name} "
                            "must be numeric or blank when undefined."
                        ) from exc
        _assert_metrics_match(
            recorded_metrics,
            _calculate_metrics(grouped_rows[source]),
            prefix="",
            label=f"metrics_by_dataset.tsv[{source}]",
        )


def _archive_stem(parameters: Mapping[str, object]) -> str:
    """Derive the archive label from strict recorded condition fields."""

    seed = _strict_int(parameters["seed"], label="seed")
    gaze_features = parameters["gaze_features"]
    assert isinstance(gaze_features, list)
    return condition_slug(
        model=str(parameters["model"]),
        finetuning_mode=str(parameters["finetuning_mode"]),
        gaze_fusion=str(parameters["gaze_fusion"]),
        gaze_features=tuple(gaze_features),
        seed=seed,
        no_iemocap=_effective_no_iemocap(parameters),
    )


def _build_package_manifest(
    run_path: Path,
    parameters: Mapping[str, object],
    snapshots: Mapping[str, bytes],
) -> dict[str, object]:
    """Build hashes from the exact immutable bytes written to the ZIP."""

    file_manifest = {
        name: {
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in snapshots.items()
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_path.name,
        "condition": {
            "model": parameters["model"],
            "finetuning_mode": parameters["finetuning_mode"],
            "gaze_fusion": parameters["gaze_fusion"],
            "gaze_features": parameters["gaze_features"],
            "seed": parameters["seed"],
            "no_iemocap": _effective_no_iemocap(parameters),
        },
        "files": file_manifest,
    }


def _verify_archive(
    archive_path: Path,
    *,
    archive_root: str,
    package_manifest: Mapping[str, object],
) -> None:
    """Verify exact members, CRC, sizes, and hashes from the finished ZIP."""

    manifest_name = f"{archive_root}/results_manifest.json"
    files = package_manifest["files"]
    assert isinstance(files, dict)
    expected_names = {
        *(f"{archive_root}/{name}" for name in files),
        manifest_name,
    }
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise ValueError("ZIP member list does not match the result allowlist.")
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"ZIP verification failed at {corrupt_member}.")
        embedded_manifest = json.loads(archive.read(manifest_name))
        if embedded_manifest != package_manifest:
            raise ValueError("Embedded results manifest changed during ZIP creation.")
        for relative_name, expected in files.items():
            member_name = f"{archive_root}/{relative_name}"
            digest = hashlib.sha256()
            size = 0
            with archive.open(member_name) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(
                    f"ZIP member does not match its manifest hash: {member_name}"
                )


def package_results(
    run_dir: str | Path,
    *,
    overwrite: bool = False,
    results_root: str | Path = RESULTS_ROOT,
) -> Path:
    """Snapshot, validate, package, and atomically publish one completed run."""

    requested_run_path = Path(run_dir).expanduser()
    if requested_run_path.is_symlink():
        raise ValueError(f"Run directory cannot be a symbolic link: {requested_run_path}")
    run_path = requested_run_path.resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    requested_results_root = Path(results_root).expanduser()
    if requested_results_root.is_symlink():
        raise ValueError(
            f"Results root cannot be a symbolic link: {requested_results_root}"
        )
    canonical_results_root = requested_results_root.resolve()
    if run_path.parent != canonical_results_root:
        raise ValueError(
            "Run directory must be one direct child of the canonical results root: "
            f"{canonical_results_root}."
        )
    validate_run_name(run_path.name)

    parameters_path = run_path / "training_parameters.json"
    parameters_bytes = _safe_read_file(parameters_path, run_path)
    parameters = _load_json_object(
        parameters_bytes,
        "training_parameters.json",
    )
    _validate_training_parameters(parameters)
    selected_paths = _select_result_paths(run_path, parameters)
    snapshots = _snapshot_result_files(
        run_path,
        selected_paths,
        training_parameters=parameters_bytes,
    )
    _validate_result_contract(run_path, parameters, snapshots)

    archive_path = run_path / f"{_archive_stem(parameters)}_results_only.zip"
    if archive_path.exists() and not overwrite:
        raise FileExistsError(
            f"Results archive already exists: {archive_path}. Use --overwrite to replace it."
        )
    package_manifest = _build_package_manifest(
        run_path,
        parameters,
        snapshots,
    )
    temporary_path = run_path / f".{archive_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for relative_name, data in snapshots.items():
                archive.writestr(f"{run_path.name}/{relative_name}", data)
            archive.writestr(
                f"{run_path.name}/results_manifest.json",
                json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
            )
        _verify_archive(
            temporary_path,
            archive_root=run_path.name,
            package_manifest=package_manifest,
        )
        if overwrite:
            os.replace(temporary_path, archive_path)
        else:
            try:
                os.link(temporary_path, archive_path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Results archive appeared during packaging: {archive_path}."
                ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive_path


def main(argv: Sequence[str] | None = None) -> int:
    """Package one run selected by its single canonical directory name."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run_dir = resolve_run_directory(args.run_name)
    archive = package_results(run_dir, overwrite=args.overwrite)
    print(f"Results-only ZIP: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

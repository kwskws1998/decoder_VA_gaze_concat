from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from va_model_code.decoder_va.paths import (
    condition_slug,
    default_run_name,
    resolve_run_directory,
    validate_run_name,
)
from va_model_code.package_results import (
    METRIC_NAMES,
    PREDICTION_COLUMNS,
    _assert_metrics_match,
    _calculate_metrics,
    package_results,
)

PREDICTION_HEADER = "\t".join(PREDICTION_COLUMNS) + "\n"


def _perfect_metrics(count: int, *, prefix: str = "") -> dict[str, int | float]:
    """Build exact metrics for perfectly predicted, non-constant VA rows."""

    values: dict[str, int | float] = {
        "n_examples": count,
        "mse_valence": 0.0,
        "rmse_valence": 0.0,
        "mae_valence": 0.0,
        "pearson_corr_valence": 1.0,
        "ccc_valence": 1.0,
        "mse_arousal": 0.0,
        "rmse_arousal": 0.0,
        "mae_arousal": 0.0,
        "pearson_corr_arousal": 1.0,
        "ccc_arousal": 1.0,
        "mse_mean": 0.0,
        "mae_mean": 0.0,
        "pearson_corr_mean": 1.0,
        "ccc_mean": 1.0,
    }
    return {f"{prefix}{name}": value for name, value in values.items()}


def _write_completed_run(run_dir: Path, *, gaze_fusion: str = "none") -> None:
    """Create the smallest complete two-fold artifact tree used by packaging tests."""

    gaze_features = [] if gaze_fusion == "none" else ["TRT"]
    parameters = {
        "architecture_manifest_version": 6,
        "model": "qwen3.5-0.8b",
        "model_id": "Qwen/fake",
        "model_revision": "a" * 40,
        "loss": "mse",
        "finetuning_mode": "full",
        "gaze_fusion": gaze_fusion,
        "gaze_features": gaze_features,
        "gaze_feature_indices": [3] if gaze_features else [],
        "features_used": [0, 0, 0, 1, 0] if gaze_features else [0, 0, 0, 0, 0],
        "et_model_id": "ET/fake",
        "et_revision": "b" * 40,
        "et_filename": "et.safetensors",
        "output_activation": "hard_sigmoid",
        "output_dim": 2,
        "dtype": "bfloat16",
        "data_dir": "data/fake",
        "et_cache_size": 70000,
        "lora_rank": None,
        "lora_alpha": None,
        "lora_dropout": None,
        "attn_implementation": None,
        "gaze_concat_order": (
            "eye_start, compact_selected_gaze, eye_end, text"
            if gaze_fusion == "prefix-concat"
            else None
        ),
        "pooling_position": (
            "last_valid_text_token_after_gaze_prefix"
            if gaze_fusion == "prefix-concat"
            else "last_valid_text_token"
        ),
        "dataset_counts_after_filter": {"fake": 4},
        "excluded_dataset_names": ["IEMOCAP sentences"],
        "seed": 42,
        "no_iemocap": True,
        "no_ieomcap": False,
        "held_out_folds": [1, 2],
        "fold_sha256": {
            "full_dataset_fold1.csv": "c" * 64,
            "full_dataset_fold2.csv": "d" * 64,
        },
        "max_length": 200,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "epochs": 1.0,
        "learning_rate": 6e-6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "run_name": run_dir.name,
        "effective_output_dir": str(run_dir),
    }
    run_dir.mkdir(parents=True)
    (run_dir / "training_parameters.json").write_text(
        json.dumps(parameters),
        encoding="utf-8",
    )
    (run_dir / "oof_metrics.json").write_text(
        json.dumps(_perfect_metrics(4)),
        encoding="utf-8",
    )
    (run_dir / "oof_predictions.tsv").write_text(
        PREDICTION_HEADER
        + "0\t1\ta\tfake\t0.1\t0.2\t0.1\t0.2\n"
        "1\t1\tb\tfake\t0.9\t0.8\t0.9\t0.8\n"
        "2\t2\tc\tfake\t0.2\t0.3\t0.2\t0.3\n"
        "3\t2\td\tfake\t0.8\t0.7\t0.8\t0.7\n",
        encoding="utf-8",
    )
    dataset_values = _perfect_metrics(4)
    dataset_header = "\t".join(("dataset_of_origin", *METRIC_NAMES))
    dataset_row = "\t".join(
        ("fake", *(str(dataset_values[name]) for name in METRIC_NAMES))
    )
    (run_dir / "metrics_by_dataset.tsv").write_text(
        f"{dataset_header}\n{dataset_row}\n",
        encoding="utf-8",
    )
    for fold in (1, 2):
        fold_dir = run_dir / f"heldout_fold{fold}"
        (fold_dir / "final_model").mkdir(parents=True)
        checkpoint = fold_dir / "checkpoints" / "checkpoint-10"
        checkpoint.mkdir(parents=True)
        fold_manifest = {
            **parameters,
            "held_out_fold": fold,
            "training_fold": 3 - fold,
            "fold_seed": 42 + fold - 1,
            "training_rows": 2,
            "evaluation_rows": 2,
        }
        (fold_dir / "run_manifest.json").write_text(
            json.dumps(fold_manifest),
            encoding="utf-8",
        )
        (fold_dir / "metrics.json").write_text(
            json.dumps(_perfect_metrics(2, prefix="test_")),
            encoding="utf-8",
        )
        fold_rows = (
            (
                "0\t1\ta\tfake\t0.1\t0.2\t0.1\t0.2\n"
                "1\t1\tb\tfake\t0.9\t0.8\t0.9\t0.8\n"
            )
            if fold == 1
            else (
                "2\t2\tc\tfake\t0.2\t0.3\t0.2\t0.3\n"
                "3\t2\td\tfake\t0.8\t0.7\t0.8\t0.7\n"
            )
        )
        (fold_dir / "predictions.tsv").write_text(
            PREDICTION_HEADER
            + fold_rows,
            encoding="utf-8",
        )
        (fold_dir / "gpu_memory.json").write_text(
            json.dumps({"cuda_enabled": False}),
            encoding="utf-8",
        )
        (fold_dir / "final_model" / "decoder_va_architecture.json").write_text(
            json.dumps(
                {
                    "schema_version": 6,
                    "decoder_model_id": parameters["model_id"],
                    "decoder_commit": parameters["model_revision"],
                    "finetuning_mode": parameters["finetuning_mode"],
                    "gaze_fusion": parameters["gaze_fusion"],
                    "gaze_features": parameters["gaze_features"],
                    "gaze_feature_indices": parameters["gaze_feature_indices"],
                    "features_used": parameters["features_used"],
                    "et_model": (
                        {
                            "repo_id": parameters["et_model_id"],
                            "revision": parameters["et_revision"],
                            "filename": parameters["et_filename"],
                            "feature_names": parameters["gaze_features"],
                            "feature_indices": parameters["gaze_feature_indices"],
                            "features_used": parameters["features_used"],
                        }
                        if gaze_fusion == "prefix-concat"
                        else None
                    ),
                    "gaze_concat_order": parameters["gaze_concat_order"],
                    "pooling_position": parameters["pooling_position"],
                    "output_activation": parameters["output_activation"],
                    "output_names": ["valence", "arousal"],
                    "reconstruction": {
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
                        "lora_target_modules": None,
                        "lora_task_type": None,
                        "gaze_projection_dim": 128,
                        "gaze_projection_dropout": [0.1, 0.3],
                        "classifier_dropout": 0.1,
                        "attn_implementation": parameters["attn_implementation"],
                        "backbone_dtype_at_construction": parameters["dtype"],
                    },
                    "state_dict": {
                        "filename": "model.safetensors",
                        "format": "safetensors",
                        "scope": (
                            "complete DecoderVARegressor state_dict; "
                            "ET2 remains external and frozen"
                        ),
                        "strict_loading": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": 10,
                    "best_metric": 0.0,
                    "best_model_checkpoint": str(checkpoint),
                }
            ),
            encoding="utf-8",
        )
        (checkpoint / "optimizer.pt").write_bytes(b"large optimizer placeholder")
        (fold_dir / "final_model" / "model.safetensors").write_bytes(
            b"large model placeholder"
        )


def _rewrite_parameter_contract(
    run_dir: Path,
    field: str,
    value: object,
) -> None:
    """Rewrite one root parameter and its identical per-fold copies."""

    paths = [
        run_dir / "training_parameters.json",
        run_dir / "heldout_fold1" / "run_manifest.json",
        run_dir / "heldout_fold2" / "run_manifest.json",
    ]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = value
        path.write_text(json.dumps(payload), encoding="utf-8")


def _prediction_rows(path: Path) -> list[dict[str, str]]:
    """Read one test prediction table after a controlled rewrite."""

    return list(
        csv.DictReader(
            io.StringIO(path.read_text(encoding="utf-8")),
            delimiter="\t",
        )
    )


def _write_constant_prediction_metrics(run_dir: Path) -> None:
    """Make all predictions constant and persist production-compatible null metrics."""

    prediction_paths = [
        run_dir / "heldout_fold1" / "predictions.tsv",
        run_dir / "heldout_fold2" / "predictions.tsv",
        run_dir / "oof_predictions.tsv",
    ]
    for path in prediction_paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        rewritten = [lines[0]]
        for line in lines[1:]:
            fields = line.split("\t")
            fields[4:] = ["0.5", "0.5", "0.5", "0.5"]
            rewritten.append("\t".join(fields))
        path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    for fold in (1, 2):
        prediction_path = run_dir / f"heldout_fold{fold}" / "predictions.tsv"
        metrics = _calculate_metrics(_prediction_rows(prediction_path))
        fold_metrics = {f"test_{name}": value for name, value in metrics.items()}
        (prediction_path.parent / "metrics.json").write_text(
            json.dumps(fold_metrics),
            encoding="utf-8",
        )

    oof_metrics = _calculate_metrics(
        _prediction_rows(run_dir / "oof_predictions.tsv")
    )
    (run_dir / "oof_metrics.json").write_text(
        json.dumps(oof_metrics),
        encoding="utf-8",
    )
    dataset_header = "\t".join(("dataset_of_origin", *METRIC_NAMES))
    dataset_values = [
        "" if oof_metrics[name] is None else str(oof_metrics[name])
        for name in METRIC_NAMES
    ]
    (run_dir / "metrics_by_dataset.tsv").write_text(
        dataset_header + "\n" + "\t".join(("fake", *dataset_values)) + "\n",
        encoding="utf-8",
    )


def test_run_names_resolve_only_below_one_results_root(tmp_path: Path) -> None:
    assert validate_run_name("paper7_full_baseline_seed42") == (
        "paper7_full_baseline_seed42"
    )
    assert resolve_run_directory("run-1", results_root=tmp_path) == (
        tmp_path / "run-1"
    ).resolve()
    for invalid in ("", "../run", "nested/run", "/absolute", "has space"):
        with pytest.raises(ValueError):
            validate_run_name(invalid)
    real_run = tmp_path / "real_run"
    real_run.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_run, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        resolve_run_directory("alias", results_root=tmp_path)


def test_condition_aware_default_names_distinguish_baseline_and_gaze() -> None:
    timestamp = datetime(2026, 8, 6, 12, 34, 56, 123456)
    baseline = default_run_name(
        model="qwen3.5-0.8b",
        finetuning_mode="full",
        gaze_fusion="none",
        gaze_features=(),
        seed=42,
        no_iemocap=True,
        timestamp=timestamp,
    )
    gaze = default_run_name(
        model="qwen3.5-0.8b",
        finetuning_mode="full",
        gaze_fusion="prefix-concat",
        gaze_features=("TRT",),
        seed=42,
        no_iemocap=True,
        timestamp=timestamp,
    )

    assert baseline == (
        "20260806_123456_123456_qwen3.5-0.8b_full_baseline_no_iemocap_seed42"
    )
    assert gaze == (
        "20260806_123456_123456_qwen3.5-0.8b_full_gaze_TRT_no_iemocap_seed42"
    )
    assert condition_slug(
        model="qwen3.5-0.8b",
        finetuning_mode="lora",
        gaze_fusion="prefix-concat",
        gaze_features=("nFix", "TRT"),
        seed=7,
    ) == "qwen3.5-0.8b_lora_gaze_nFix-TRT_seed7"


def test_results_only_package_is_condition_named_and_excludes_weights(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "misleading_folder_name"
    _write_completed_run(run_dir, gaze_fusion="none")

    archive_path = package_results(run_dir, results_root=tmp_path)

    assert archive_path.parent == run_dir
    assert archive_path.name == (
        "qwen3.5-0.8b_full_baseline_no_iemocap_seed42_results_only.zip"
    )
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert f"{run_dir.name}/results_manifest.json" in names
        assert all(not name.endswith("model.safetensors") for name in names)
        assert all(not name.endswith("optimizer.pt") for name in names)
        manifest = json.loads(
            archive.read(f"{run_dir.name}/results_manifest.json")
        )
        training_name = f"{run_dir.name}/training_parameters.json"
        training_bytes = archive.read(training_name)
        assert manifest["condition"]["gaze_fusion"] == "none"
        assert manifest["files"]["training_parameters.json"]["sha256"] == (
            hashlib.sha256(training_bytes).hexdigest()
        )

    with pytest.raises(FileExistsError):
        package_results(run_dir, results_root=tmp_path)
    assert package_results(
        run_dir,
        overwrite=True,
        results_root=tmp_path,
    ) == archive_path


def test_results_only_package_rejects_incomplete_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "incomplete"
    _write_completed_run(run_dir)
    (run_dir / "heldout_fold2" / "metrics.json").unlink()

    with pytest.raises(FileNotFoundError, match="metrics.json"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_mixed_fold_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "mixed"
    _write_completed_run(run_dir)
    manifest_path = run_dir / "heldout_fold2" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gaze_fusion"] = "prefix-concat"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="gaze_fusion"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_oof_count_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "wrong_count"
    _write_completed_run(run_dir)
    (run_dir / "oof_metrics.json").write_text(
        json.dumps(_perfect_metrics(3)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="n_examples"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("no_iemocap", "false", "JSON boolean"),
        ("seed", 42.9, "JSON integer"),
        ("held_out_folds", [1.0, 2], "JSON integer"),
    ),
)
def test_results_only_package_rejects_coerced_parameter_types(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    run_dir = tmp_path / f"invalid_{field}"
    _write_completed_run(run_dir)
    parameters_path = run_dir / "training_parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    parameters[field] = value
    parameters_path.write_text(json.dumps(parameters), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_malformed_prediction_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "malformed_tsv"
    _write_completed_run(run_dir)
    predictions = run_dir / "heldout_fold1" / "predictions.tsv"
    predictions.write_text(
        PREDICTION_HEADER
        + "0\t1\tshort\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-rectangular"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_wrong_dataset_metrics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "wrong_dataset_metrics"
    _write_completed_run(run_dir)
    metrics_path = run_dir / "metrics_by_dataset.tsv"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8").replace("fake", "other"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sources do not match"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_external_parameter_symlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "parameter_symlink"
    _write_completed_run(run_dir)
    parameters_path = run_dir / "training_parameters.json"
    external = tmp_path / "external.json"
    external.write_bytes(parameters_path.read_bytes())
    parameters_path.unlink()
    parameters_path.symlink_to(external)

    with pytest.raises(ValueError, match="symbolic links"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_stale_unrecorded_fold(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "stale_fold"
    _write_completed_run(run_dir)
    parameters_path = run_dir / "training_parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    parameters["held_out_folds"] = [1]
    parameters_path.write_text(json.dumps(parameters), encoding="utf-8")

    with pytest.raises(ValueError, match="complete two-fold OOF protocol"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_accepts_undefined_correlations(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "constant_predictions"
    _write_completed_run(run_dir)
    _write_constant_prediction_metrics(run_dir)

    archive_path = package_results(run_dir, results_root=tmp_path)

    assert archive_path.is_file()
    computed = _calculate_metrics(
        _prediction_rows(run_dir / "heldout_fold1" / "predictions.tsv")
    )
    assert computed["pearson_corr_valence"] is None
    assert computed["ccc_valence"] is None
    _assert_metrics_match(computed, computed, prefix="", label="constant metrics")


def test_results_only_package_rejects_extra_prediction_column(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "extra_prediction_column"
    _write_completed_run(run_dir)
    predictions = run_dir / "heldout_fold1" / "predictions.tsv"
    lines = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text(
        "\n".join(f"{line}\textra_secret" for line in lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly the eight prediction columns"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_extra_dataset_metric_column(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "extra_dataset_column"
    _write_completed_run(run_dir)
    metrics = run_dir / "metrics_by_dataset.tsv"
    lines = metrics.read_text(encoding="utf-8").splitlines()
    metrics.write_text(
        "\n".join(f"{line}\textra_secret" for line in lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly the canonical metric columns"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_dataset_count_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "wrong_dataset_count"
    _write_completed_run(run_dir)
    _rewrite_parameter_contract(
        run_dir,
        "dataset_counts_after_filter",
        {"fake": 999},
    )
    for fold in (1, 2):
        manifest_path = run_dir / f"heldout_fold{fold}" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["training_rows"] = 997
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the complete two-fold"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_excluded_source_leakage(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "iemocap_leak"
    _write_completed_run(run_dir)
    _rewrite_parameter_contract(
        run_dir,
        "dataset_counts_after_filter",
        {"IEMOCAP sentences": 4},
    )
    for path in (
        run_dir / "heldout_fold1" / "predictions.tsv",
        run_dir / "heldout_fold2" / "predictions.tsv",
        run_dir / "oof_predictions.tsv",
        run_dir / "metrics_by_dataset.tsv",
    ):
        path.write_text(
            path.read_text(encoding="utf-8").replace("fake", "IEMOCAP sentences"),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="contain excluded dataset source"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("global_step", 999, "does not match checkpoint-10"),
        ("best_model_checkpoint", None, "must be a non-empty path"),
    ),
)
def test_results_only_package_rejects_invalid_trainer_state(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    run_dir = tmp_path / f"invalid_trainer_{field}"
    _write_completed_run(run_dir)
    state_path = (
        run_dir
        / "heldout_fold1"
        / "checkpoints"
        / "checkpoint-10"
        / "trainer_state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize(
    ("target", "value", "message"),
    (
        ("schema_version", 5, "architecture_manifest_version"),
        ("decoder_revision", "e" * 40, "decoder_revision"),
        ("et_revision", "f" * 40, "et_revision"),
        ("gaze_projection_dim", 999, "gaze_projection_dim"),
        ("lora_target_modules", "all-linear", "lora_target_modules"),
    ),
)
def test_results_only_package_rejects_architecture_provenance_mismatch(
    tmp_path: Path,
    target: str,
    value: object,
    message: str,
) -> None:
    run_dir = tmp_path / f"invalid_architecture_{target}"
    _write_completed_run(run_dir, gaze_fusion="prefix-concat")
    architecture_path = (
        run_dir
        / "heldout_fold1"
        / "final_model"
        / "decoder_va_architecture.json"
    )
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    if target == "schema_version":
        architecture[target] = value
    else:
        architecture["reconstruction"][target] = value
    architecture_path.write_text(json.dumps(architecture), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_incomplete_architecture_reconstruction(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "incomplete_architecture"
    _write_completed_run(run_dir)
    architecture_path = (
        run_dir
        / "heldout_fold1"
        / "final_model"
        / "decoder_va_architecture.json"
    )
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    architecture["reconstruction"].pop("classifier_dropout")
    architecture_path.write_text(json.dumps(architecture), encoding="utf-8")

    with pytest.raises(ValueError, match="classifier_dropout"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("training_fold", 1, "opposite held-out fold"),
        ("fold_seed", 999, "paired seed policy"),
        ("training_rows", 999, "opposite fold"),
    ),
)
def test_results_only_package_rejects_fold_protocol_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    run_dir = tmp_path / f"invalid_fold_{field}"
    _write_completed_run(run_dir)
    manifest_path = run_dir / "heldout_fold1" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_unrelated_best_checkpoint_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "unrelated_best_checkpoint"
    _write_completed_run(run_dir)
    state_path = (
        run_dir
        / "heldout_fold1"
        / "checkpoints"
        / "checkpoint-10"
        / "trainer_state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["best_model_checkpoint"] = (
        "/tmp/unrelated-run/heldout_fold1/checkpoints/checkpoint-10"
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="this held-out fold"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_checkpoint_parent_symlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "checkpoint_parent_symlink"
    _write_completed_run(run_dir)
    fold1_checkpoint = (
        run_dir / "heldout_fold1" / "checkpoints" / "checkpoint-10"
    )
    fold2_checkpoint = (
        run_dir / "heldout_fold2" / "checkpoints" / "checkpoint-10"
    )
    shutil.rmtree(fold1_checkpoint)
    fold1_checkpoint.symlink_to(fold2_checkpoint, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic links in artifact paths"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_normalized_iemocap_source(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "normalized_iemocap_leak"
    _write_completed_run(run_dir)
    _rewrite_parameter_contract(
        run_dir,
        "dataset_counts_after_filter",
        {"iemocap SENTENCES": 4},
    )
    for path in (
        run_dir / "heldout_fold1" / "predictions.tsv",
        run_dir / "heldout_fold2" / "predictions.tsv",
        run_dir / "oof_predictions.tsv",
        run_dir / "metrics_by_dataset.tsv",
    ):
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "fake",
                "iemocap SENTENCES",
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="no-IEMOCAP condition"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("architecture_manifest_version", 999, "supported schema version"),
        ("model_revision", "main", "immutable lowercase"),
        ("et_revision", "main", "immutable lowercase"),
        ("model", "roberta-large", "model must be one of"),
        ("loss", "heteroscedastic+ccc", "loss must be one of"),
    ),
)
def test_results_only_package_rejects_unsupported_root_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    run_dir = tmp_path / f"unsupported_{field}"
    _write_completed_run(run_dir)
    _rewrite_parameter_contract(run_dir, field, value)

    with pytest.raises(ValueError, match=message):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "duplicate_json_key"
    _write_completed_run(run_dir)
    parameters_path = run_dir / "training_parameters.json"
    payload = parameters_path.read_text(encoding="utf-8")
    parameters_path.write_text(
        payload.replace("{", '{"no_iemocap": false,', 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="valid JSON"):
        package_results(run_dir, results_root=tmp_path)


def test_results_only_package_labels_explicit_iemocap_exclusion(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "explicit_iemocap_exclusion"
    _write_completed_run(run_dir)
    _rewrite_parameter_contract(run_dir, "no_iemocap", False)

    archive = package_results(run_dir, results_root=tmp_path)

    assert "_no_iemocap_" in archive.name

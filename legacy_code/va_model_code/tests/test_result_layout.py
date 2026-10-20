from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import zipfile

import pytest
import pandas as pd

from va_model_code.decoder_va.alignment import GAZE_ALIGNMENT_CONTRACT
from va_model_code.decoder_va.filters import (
    TRAIN_EVAL_TEXT_OVERLAP_CONTRACT,
    apply_training_text_overlap_policy,
    exclude_eval_text_overlaps,
    training_text_overlap_contract,
)
from va_model_code.decoder_va.model import PRECISION_CONTRACT
from va_model_code.decoder_va.gaze_relation import gaze_relation_contract
from va_model_code.decoder_va.readout import readout_contract, readout_concat_order, readout_pooling_position

from va_model_code.decoder_va.redistribution import redistribution_contract
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


def _write_completed_run(
    run_dir: Path,
    *,
    gaze_fusion: str = "none",
    schema_version: int = 6,
    gaze_redistribution: dict | None = None,
    gaze_relation: dict | None = None,
    readout: dict | None = None,
    dataset_sources: tuple[str, str, str, str] = ("fake", "fake", "fake", "fake"),
) -> None:
    """Create the smallest complete two-fold artifact tree used by packaging tests."""

    if len(dataset_sources) != 4:
        raise ValueError("The packaging fixture requires exactly four dataset sources.")
    gaze_features = [] if gaze_fusion == "none" else ["TRT"]
    parameters = {
        "architecture_manifest_version": schema_version,
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
        "dataset_counts_after_filter": dict(Counter(dataset_sources)),
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
    if schema_version >= 7 or gaze_redistribution is not None:
        parameters["gaze_redistribution"] = (
            {"method": "none"}
            if gaze_redistribution is None
            else gaze_redistribution
        )
    relation_metadata = {}
    if schema_version >= 9 or gaze_relation is not None:
        relation_metadata = {
            "gaze_relation": gaze_relation_contract() if gaze_relation is None else gaze_relation
        }
        parameters.update(relation_metadata)
        if parameters["gaze_relation"]["method"] != "none":
            parameters["attn_implementation"] = "eager"
    readout_metadata = {}
    if schema_version >= 10 or readout is not None:
        readout_metadata = {"readout": readout_contract() if readout is None else readout}
        parameters.update(readout_metadata)
        parameters["gaze_concat_order"] = readout_concat_order(gaze_fusion, parameters["readout"])
        parameters["pooling_position"] = readout_pooling_position(gaze_fusion, parameters["readout"])
    current_metadata = {}
    if schema_version >= 11:
        from va_model_code.decoder_va.uncertainty import prediction_contract

        readout_metadata["prediction_distribution"] = prediction_contract()
        parameters["prediction_distribution"] = prediction_contract()
    if schema_version >= 8:
        current_metadata = {
            "gaze_alignment": GAZE_ALIGNMENT_CONTRACT if gaze_features else None,
            "precision_contract": PRECISION_CONTRACT,
        }
        parameters.update({
            **current_metadata,
            "dtype": "float32",
            "backbone_parameter_dtype": "float32",
            "trainable_parameter_dtype": "float32",
            "compute_dtype": "bfloat16",
            "training_text_overlap_contract": TRAIN_EVAL_TEXT_OVERLAP_CONTRACT,
        })
    redistribution_metadata = (
        {"gaze_redistribution": parameters["gaze_redistribution"]}
        if "gaze_redistribution" in parameters
        else {}
    )
    run_dir.mkdir(parents=True)
    (run_dir / "training_parameters.json").write_text(
        json.dumps(parameters),
        encoding="utf-8",
    )
    (run_dir / "oof_metrics.json").write_text(
        json.dumps(_perfect_metrics(4)),
        encoding="utf-8",
    )
    prediction_rows = (
        f"0\t1\ta\t{dataset_sources[0]}\t0.1\t0.2\t0.1\t0.2\n",
        f"1\t1\tb\t{dataset_sources[1]}\t0.9\t0.8\t0.9\t0.8\n",
        f"2\t2\tc\t{dataset_sources[2]}\t0.2\t0.3\t0.2\t0.3\n",
        f"3\t2\td\t{dataset_sources[3]}\t0.8\t0.7\t0.8\t0.7\n",
    )
    (run_dir / "oof_predictions.tsv").write_text(
        PREDICTION_HEADER + "".join(prediction_rows),
        encoding="utf-8",
    )
    dataset_header = "\t".join(("dataset_of_origin", *METRIC_NAMES))
    parsed_rows = _prediction_rows(run_dir / "oof_predictions.tsv")
    dataset_rows = []
    for source in sorted(set(dataset_sources), key=str.casefold):
        source_metrics = _calculate_metrics(
            [row for row in parsed_rows if row["dataset_of_origin"] == source]
        )
        dataset_rows.append(
            "\t".join(
                (
                    source,
                    *(
                        "" if source_metrics[name] is None else str(source_metrics[name])
                        for name in METRIC_NAMES
                    ),
                )
            )
        )
    (run_dir / "metrics_by_dataset.tsv").write_text(
        f"{dataset_header}\n" + "\n".join(dataset_rows) + "\n",
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
        if schema_version >= 8:
            _, overlap = exclude_eval_text_overlaps(
                pd.DataFrame({"text": ["c", "d"] if fold == 1 else ["a", "b"]}),
                pd.DataFrame({"text": ["a", "b"] if fold == 1 else ["c", "d"]}),
            )
            fold_manifest["training_text_overlap"] = overlap
        (fold_dir / "run_manifest.json").write_text(
            json.dumps(fold_manifest),
            encoding="utf-8",
        )
        (fold_dir / "metrics.json").write_text(
            json.dumps(_perfect_metrics(2, prefix="test_")),
            encoding="utf-8",
        )
        fold_rows = "".join(prediction_rows[:2] if fold == 1 else prediction_rows[2:])
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
                    "schema_version": schema_version,
                    **redistribution_metadata,
                    **relation_metadata,
                    **readout_metadata,
                    **current_metadata,
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
                        **redistribution_metadata,
                        **relation_metadata,
                        **readout_metadata,
                        **current_metadata,
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
    assert condition_slug(
        model="qwen3.5-0.8b",
        finetuning_mode="full",
        gaze_fusion="prefix-concat",
        gaze_features=("TRT",),
        seed=42,
        sentence_only=True,
        no_iemocap=True,
    ) == "qwen3.5-0.8b_full_gaze_TRT_sentence_only_no_iemocap_seed42"


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

    with pytest.raises(ValueError, match="exactly the prediction columns"):
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
        ("sentence_only", "true", "must be a JSON boolean"),
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


def test_results_only_package_records_sentence_only_scope(tmp_path: Path) -> None:
    run_dir = tmp_path / "sentence_only_run"
    _write_completed_run(
        run_dir,
        dataset_sources=("EmoTales sentences", "Emobank", "fb", "fb"),
    )
    _rewrite_parameter_contract(run_dir, "sentence_only", True)

    archive = package_results(run_dir, results_root=tmp_path)

    assert "_sentence_only_" in archive.name
    with zipfile.ZipFile(archive) as handle:
        manifest = json.loads(handle.read(f"{run_dir.name}/results_manifest.json"))
    assert manifest["condition"]["sentence_only"] is True


def test_results_only_package_rejects_false_sentence_only_provenance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "mislabeled_sentence_only_run"
    _write_completed_run(run_dir)
    _rewrite_parameter_contract(run_dir, "sentence_only", True)

    with pytest.raises(ValueError, match="exactly these retained datasets"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("version,method", [(6, None), (6, "none"), (7, "none"), (7, "fixed-gaussian"), (7, "asym-gaussian"), (8, "none"), (8, "fixed-gaussian"), (8, "asym-gaussian"), (9, "none"), (9, "fixed-gaussian"), (9, "asym-gaussian")])
def test_package_records_distinct_redistribution_condition(tmp_path, version, method):
    """Keep legacy archive names while distinguishing the learned redistribution arm."""

    run_dir = tmp_path / "run"
    contract = None if method is None else redistribution_contract(method)
    _write_completed_run(
        run_dir,
        gaze_fusion="prefix-concat",
        schema_version=version,
        gaze_redistribution=contract,
    )

    archive_path = package_results(run_dir, results_root=tmp_path)

    assert ("redistribution_asym-gaussian" in archive_path.name) == (method == "asym-gaussian")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(f"{run_dir.name}/results_manifest.json"))
        assert manifest["condition"]["gaze_fusion"] == "prefix-concat"
        assert manifest["condition"]["gaze_redistribution"] == (
            {"method": "none"} if contract is None else contract
        )


def test_package_preserves_sigma_diagnostics_and_complete_history(tmp_path):
    """Include diagnostic evidence while continuing to exclude model weights."""

    run_dir = tmp_path / "diagnostic_run"
    _write_completed_run(run_dir, gaze_fusion="prefix-concat", schema_version=7, gaze_redistribution=redistribution_contract("fixed-gaussian"))
    _rewrite_parameter_contract(run_dir, "sigma_diagnostics_steps", 50)
    expected = {}
    for fold in (1, 2):
        for name in ("sigma_updates.jsonl", "sigma_probes.jsonl", "checkpoints/trainer_state.json"):
            relative = f"heldout_fold{fold}/{name}"
            content = json.dumps({"test_evidence": name}) + "\n"
            (run_dir / relative).write_text(content, encoding="utf-8")
            expected[relative] = content.encode()
    archive_path = package_results(run_dir, results_root=tmp_path)
    with zipfile.ZipFile(archive_path) as archive:
        for relative, content in expected.items():
            assert archive.read(f"{run_dir.name}/{relative}") == content
        assert not any(name.endswith(".safetensors") for name in archive.namelist())
    (run_dir / "heldout_fold1/sigma_updates.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        package_results(run_dir, results_root=tmp_path)


def _tamper_redistribution(run_dir, target, value, *, remove=False):
    """Modify one persisted contract copy to test rejection before ZIP publication."""

    if target == "root":
        path = run_dir / "training_parameters.json"
    elif target.startswith("fold"):
        path = run_dir / f"heldout_{target}" / "run_manifest.json"
    else:
        path = run_dir / "heldout_fold2" / "final_model" / "decoder_va_architecture.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    container = payload["reconstruction"] if target == "reconstruction" else payload
    if remove:
        container.pop("gaze_redistribution")
    else:
        container["gaze_redistribution"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("target", ["root", "fold1", "fold2", "architecture", "reconstruction"])
@pytest.mark.parametrize("method", ["none", "asym-gaussian"])
def test_package_schema7_rejects_missing_redistribution(tmp_path, target, method):
    """Require explicit disabled metadata too, preventing silent condition downgrades."""

    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        gaze_fusion="prefix-concat",
        schema_version=7,
        gaze_redistribution=redistribution_contract(method),
    )
    _tamper_redistribution(run_dir, target, None, remove=True)

    with pytest.raises(ValueError, match="gaze_redistribution"):
        package_results(run_dir, results_root=tmp_path)
    assert not list(run_dir.glob("*.zip"))


@pytest.mark.parametrize("target", ["root", "fold1", "fold2", "architecture", "reconstruction"])
def test_package_rejects_redistribution_disagreement(tmp_path, target):
    """Prevent a concat result from being relabeled as a redistribution result."""

    run_dir = tmp_path / "run"
    _write_completed_run(
        run_dir,
        gaze_fusion="prefix-concat",
        schema_version=7,
        gaze_redistribution=redistribution_contract("asym-gaussian"),
    )
    _tamper_redistribution(run_dir, target, {"method": "none"})

    with pytest.raises(ValueError, match="gaze_redistribution"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("target", ["root", "fold1", "fold2", "architecture", "reconstruction"])
def test_package_rejects_redistribution_in_schema6(tmp_path, target):
    """Legacy files cannot claim a trainable component they could not have saved."""

    run_dir = tmp_path / "run"
    _write_completed_run(run_dir, gaze_fusion="prefix-concat")
    _tamper_redistribution(run_dir, target, redistribution_contract("asym-gaussian"))

    with pytest.raises(ValueError, match="schema 6 cannot enable"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [("normalization", "none"), ("compute_dtype", "float16"), ("min_sigma", 0), ("trainable", 1)],
)
def test_package_rejects_malformed_redistribution(tmp_path, field, value):
    """Reject invalid semantics even when every provenance copy contains the same edit."""

    run_dir = tmp_path / "run"
    contract = redistribution_contract("asym-gaussian")
    contract[field] = value
    _write_completed_run(
        run_dir, gaze_fusion="prefix-concat", schema_version=7, gaze_redistribution=contract
    )

    with pytest.raises(ValueError, match="redistribution|sigma"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("target", ("root", "fold", "architecture", "reconstruction"))
@pytest.mark.parametrize("field", ("precision_contract", "gaze_alignment"))
@pytest.mark.parametrize("version", (8, 9))
def test_package_schema8_rejects_missing_current_contract(tmp_path, target, field, version):
    """Reject incomplete current metadata without losing historical packaging support."""

    run_dir = tmp_path / "current_run"
    _write_completed_run(run_dir, gaze_fusion="prefix-concat", schema_version=version)
    relative = {
        "root": "training_parameters.json",
        "fold": "heldout_fold1/run_manifest.json",
        "architecture": "heldout_fold1/final_model/decoder_va_architecture.json",
        "reconstruction": "heldout_fold1/final_model/decoder_va_architecture.json",
    }[target]
    path = run_dir / relative
    payload = json.loads(path.read_text())
    container = payload["reconstruction"] if target == "reconstruction" else payload
    del container[field]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=field):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("field,value", (
    ("compute_dtype", "auto"),
    ("backbone_parameter_dtype", "bfloat16"),
    ("trainable_parameter_dtype", "bfloat16"),
    ("dtype", "bfloat16"),
))
@pytest.mark.parametrize("version", (8, 9))
def test_package_schema8_rejects_invalid_precision_identity(tmp_path, field, value, version):
    """Keep compute precision separate from storage instead of trusting ambiguous dtype."""

    run_dir = tmp_path / "current_run"
    _write_completed_run(run_dir, schema_version=version)
    _rewrite_parameter_contract(run_dir, field, value)
    with pytest.raises(ValueError, match="dtype"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("version", (8, 9))
def test_package_schema8_verifies_overlap_counts_from_actual_oof_texts(tmp_path, version):
    """Package reduced training counts only when held-out texts justify exclusion."""

    run_dir = tmp_path / "current_run"
    _write_completed_run(run_dir, schema_version=version)
    for relative in ("oof_predictions.tsv", "heldout_fold2/predictions.tsv"):
        path = run_dir / relative
        path.write_text(path.read_text().replace("2\t2\tc\t", "2\t2\t A \t"))
    frames = {
        1: pd.DataFrame({"text": ["a", "b"]}),
        2: pd.DataFrame({"text": [" A ", "d"]}),
    }
    for fold in (1, 2):
        path = run_dir / f"heldout_fold{fold}/run_manifest.json"
        payload = json.loads(path.read_text())
        _, payload["training_text_overlap"] = exclude_eval_text_overlaps(
            frames[3 - fold], frames[fold]
        )
        payload["training_rows"] = 1
        path.write_text(json.dumps(payload))
    archive = package_results(run_dir, results_root=tmp_path)
    assert archive.is_file()
    path = run_dir / "heldout_fold1/run_manifest.json"
    payload = json.loads(path.read_text())
    payload["training_text_overlap"]["shared_normalized_texts"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training_text_overlap does not match OOF"):
        package_results(run_dir, results_root=tmp_path, overwrite=True)


def _write_overlapping_keep_run(run_dir: Path, *, version: int, record_policy: bool) -> None:
    """Build a completed run that retains normalized text overlaps across folds."""

    _write_completed_run(run_dir, schema_version=version)
    for relative in ("oof_predictions.tsv", "heldout_fold2/predictions.tsv"):
        path = run_dir / relative
        path.write_text(path.read_text().replace("2\t2\tc\t", "2\t2\t A \t"))
    frames = {
        1: pd.DataFrame({"text": ["a", "b"]}),
        2: pd.DataFrame({"text": [" A ", "d"]}),
    }
    for fold in (None, 1, 2):
        path = (
            run_dir / "training_parameters.json"
            if fold is None
            else run_dir / f"heldout_fold{fold}/run_manifest.json"
        )
        payload = json.loads(path.read_text())
        payload["training_text_overlap_contract"] = training_text_overlap_contract("keep")
        if record_policy:
            payload["training_text_overlap_policy"] = "keep"
        if fold is not None:
            _, payload["training_text_overlap"] = apply_training_text_overlap_policy(
                frames[3 - fold], frames[fold], policy="keep"
            )
        path.write_text(json.dumps(payload))


@pytest.mark.parametrize("record_policy", (False, True))
@pytest.mark.parametrize("version", (8, 9))
def test_package_keeps_cross_fold_texts_with_explicit_contract(tmp_path, version, record_policy):
    """Accept retained overlap counts while preserving the older contract-only format."""

    run_dir = tmp_path / "keep_run"
    _write_overlapping_keep_run(run_dir, version=version, record_policy=record_policy)
    archive = package_results(run_dir, results_root=tmp_path)
    with zipfile.ZipFile(archive) as saved:
        manifest_name = next(
            name for name in saved.namelist() if name.endswith("heldout_fold1/run_manifest.json")
        )
        manifest = json.loads(saved.read(manifest_name))
    assert manifest["training_rows"] == 2
    assert manifest["training_text_overlap"]["excluded_train_rows"] == 0
    assert manifest["training_text_overlap"]["overlapping_eval_rows"] == 1
    assert manifest["training_text_overlap"]["shared_normalized_texts"] == 1


@pytest.mark.parametrize("version", (8, 9))
def test_package_keep_verifies_overlap_counts_from_actual_oof_texts(tmp_path, version):
    """Retaining rows does not disable the independent OOF overlap audit."""

    run_dir = tmp_path / "keep_run"
    _write_overlapping_keep_run(run_dir, version=version, record_policy=True)
    path = run_dir / "heldout_fold1/run_manifest.json"
    payload = json.loads(path.read_text())
    payload["training_text_overlap"]["overlapping_eval_rows"] = 0
    payload["training_text_overlap"]["shared_normalized_texts"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training_text_overlap does not match OOF"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("version", (8, 9))
def test_package_keep_rejects_excluded_training_rows(tmp_path, version):
    """A keep contract cannot describe a training set with removed overlap rows."""

    run_dir = tmp_path / "keep_run"
    _write_overlapping_keep_run(run_dir, version=version, record_policy=True)
    path = run_dir / "heldout_fold1/run_manifest.json"
    payload = json.loads(path.read_text())
    payload["training_rows"] = 1
    payload["training_text_overlap"]["train_rows_after"] = 1
    payload["training_text_overlap"]["excluded_train_rows"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inconsistent training_text_overlap counts"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("version", (8, 9))
def test_package_keep_rejects_mismatched_optional_policy(tmp_path, version):
    """Do not label an explicitly retained-overlap run as an exclusion run."""

    run_dir = tmp_path / "keep_run"
    _write_overlapping_keep_run(run_dir, version=version, record_policy=True)
    _rewrite_parameter_contract(run_dir, "training_text_overlap_policy", "exclude")
    with pytest.raises(ValueError, match="training_text_overlap_policy disagrees"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("version", (8, 9))
def test_package_rejects_fold_only_overlap_policy_mismatch(tmp_path, version):
    """Validate an optional fold policy even when an older root omitted that field."""

    run_dir = tmp_path / "keep_run"
    _write_overlapping_keep_run(run_dir, version=version, record_policy=False)
    path = run_dir / "heldout_fold1/run_manifest.json"
    payload = json.loads(path.read_text())
    payload["training_text_overlap_policy"] = "exclude"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training_text_overlap_policy disagrees"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("version", (8, 9))
def test_package_rejects_fold_only_boolean_overlap_contract_version(tmp_path, version):
    """Strictly check the manifest contract before Python equates True with one."""

    run_dir = tmp_path / "keep_run"
    _write_overlapping_keep_run(run_dir, version=version, record_policy=False)
    path = run_dir / "heldout_fold1/run_manifest.json"
    payload = json.loads(path.read_text())
    payload["training_text_overlap_contract"]["version"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training_text_overlap_contract"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("field,value", (
    ("version", True),
    ("policy", "keep_unverified"),
    ("normalization", "none"),
    ("evaluation_rows", "filtered"),
    ("unexpected", "field"),
))
@pytest.mark.parametrize("version", (8, 9))
def test_package_rejects_laundered_overlap_contracts(tmp_path, version, field, value):
    """Reject noncanonical contracts even when all persisted copies agree."""

    run_dir = tmp_path / "keep_run"
    _write_overlapping_keep_run(run_dir, version=version, record_policy=False)
    contract = training_text_overlap_contract("keep")
    contract[field] = value
    for fold in (None, 1, 2):
        path = (
            run_dir / "training_parameters.json"
            if fold is None
            else run_dir / f"heldout_fold{fold}/run_manifest.json"
        )
        payload = json.loads(path.read_text())
        payload["training_text_overlap_contract"] = contract
        if fold is not None:
            payload["training_text_overlap"]["contract"] = contract
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training_text_overlap_contract"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("field", ("precision_contract", "training_text_overlap"))
@pytest.mark.parametrize("version", (8, 9))
def test_package_schema8_rejects_boolean_contract_version(tmp_path, field, version):
    """Do not accept JSON true as integer version one through dict equality."""

    run_dir = tmp_path / "current_run"
    _write_completed_run(run_dir, schema_version=version)
    path = run_dir / "heldout_fold1/run_manifest.json"
    payload = json.loads(path.read_text())
    contract = payload[field]["contract"] if field == "training_text_overlap" else payload[field]
    contract["version"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=field):
        package_results(run_dir, results_root=tmp_path)


def _tamper_relation(run_dir, target, value, *, remove=False):
    """Change exactly one persisted relation identity for cross-artifact audits."""

    if target == "root":
        path = run_dir / "training_parameters.json"
    elif target.startswith("fold"):
        path = run_dir / f"heldout_{target}" / "run_manifest.json"
    else:
        path = run_dir / "heldout_fold2" / "final_model" / "decoder_va_architecture.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    container = payload["reconstruction"] if target == "reconstruction" else payload
    if remove:
        container.pop("gaze_relation")
    else:
        container["gaze_relation"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("version,method", [(6, None), (7, None), (8, None), (8, "none"), (9, "none"), (9, "trt-relative-rotation")])
def test_package_records_distinct_relation_condition(tmp_path, version, method):
    """Preserve legacy names and explicitly identify the enabled relative-phase arm."""

    run_dir = tmp_path / "relation_run"
    relation = None if method is None else gaze_relation_contract(method)
    _write_completed_run(
        run_dir, gaze_fusion="prefix-concat", schema_version=version, gaze_relation=relation
    )
    archive_path = package_results(run_dir, results_root=tmp_path)
    assert ("relation_trt-relative-rotation" in archive_path.name) == (method == "trt-relative-rotation")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(f"{run_dir.name}/results_manifest.json"))
    assert manifest["condition"]["gaze_relation"] == (gaze_relation_contract() if relation is None else relation)


@pytest.mark.parametrize("target", ["root", "fold1", "fold2", "architecture", "reconstruction"])
@pytest.mark.parametrize("method", ["none", "trt-relative-rotation"])
def test_package_schema9_requires_explicit_relation_metadata(tmp_path, target, method):
    """Reject missing disabled metadata as well as missing enabled metadata."""

    run_dir = tmp_path / "relation_run"
    _write_completed_run(
        run_dir, gaze_fusion="prefix-concat", schema_version=9,
        gaze_relation=gaze_relation_contract(method),
    )
    _tamper_relation(run_dir, target, None, remove=True)
    with pytest.raises(ValueError, match="gaze_relation"):
        package_results(run_dir, results_root=tmp_path)
    assert not list(run_dir.glob("*.zip"))


@pytest.mark.parametrize("target", ["root", "fold1", "fold2", "architecture", "reconstruction"])
def test_package_rejects_relation_disagreement(tmp_path, target):
    """Prevent relation-enabled results being published as the ordinary gaze arm."""

    run_dir = tmp_path / "relation_run"
    _write_completed_run(
        run_dir, gaze_fusion="prefix-concat", schema_version=9,
        gaze_relation=gaze_relation_contract("trt-relative-rotation"),
    )
    _tamper_relation(run_dir, target, gaze_relation_contract())
    with pytest.raises(ValueError, match="gaze_relation"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("version", [6, 7, 8])
@pytest.mark.parametrize("target", ["root", "fold1", "architecture", "reconstruction"])
def test_package_legacy_schema_cannot_claim_enabled_relation(tmp_path, version, target):
    """A legacy schema cannot acquire a learned phase branch through relabeling."""

    run_dir = tmp_path / "legacy_run"
    _write_completed_run(run_dir, gaze_fusion="prefix-concat", schema_version=version)
    _tamper_relation(run_dir, target, gaze_relation_contract("trt-relative-rotation"))
    with pytest.raises(ValueError, match=f"schema {version} cannot enable gaze_relation"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("field,value", [("version", True), ("scale", 0), ("scope", "all_tokens"), ("alpha_init", 1.0)])
def test_package_rejects_invalid_relation_semantics(tmp_path, field, value):
    """Consistent edits across all files still must satisfy the method contract."""

    run_dir = tmp_path / "invalid_relation_run"
    relation = gaze_relation_contract("trt-relative-rotation")
    relation[field] = value
    _write_completed_run(run_dir, gaze_fusion="prefix-concat", schema_version=9, gaze_relation=relation)
    with pytest.raises(ValueError, match="relation|scale"):
        package_results(run_dir, results_root=tmp_path)


@pytest.mark.parametrize("attention", [None, "sdpa", "flash_attention_2"])
def test_package_enabled_relation_requires_recorded_eager_attention(tmp_path, attention):
    """Keep the implemented dense attention backend in persisted method identity."""

    run_dir = tmp_path / "relation_run"
    _write_completed_run(
        run_dir, gaze_fusion="prefix-concat", schema_version=9,
        gaze_relation=gaze_relation_contract("trt-relative-rotation"),
    )
    _rewrite_parameter_contract(run_dir, "attn_implementation", attention)
    with pytest.raises(ValueError, match="eager"):
        package_results(run_dir, results_root=tmp_path)


def test_package_enabled_relation_rejects_redistributed_trt(tmp_path):
    """The relative-phase experiment must preserve its raw gaze source."""

    run_dir = tmp_path / "relation_run"
    _write_completed_run(
        run_dir, gaze_fusion="prefix-concat", schema_version=9,
        gaze_relation=gaze_relation_contract("trt-relative-rotation"),
        gaze_redistribution=redistribution_contract("asym-gaussian"),
    )
    with pytest.raises(ValueError, match="raw TRT"):
        package_results(run_dir, results_root=tmp_path)


def test_package_enabled_relation_rejects_text_only_configuration(tmp_path):
    """A relation branch cannot be claimed when no TRT anchors were supplied."""

    run_dir = tmp_path / "relation_run"
    _write_completed_run(
        run_dir, schema_version=9, gaze_relation=gaze_relation_contract("trt-relative-rotation")
    )
    with pytest.raises(ValueError, match="prefix-concat"):
        package_results(run_dir, results_root=tmp_path)


def _write_relation_parameter_reports(run_dir, relation):
    """Create consistent coefficient snapshots with matching architecture metadata."""

    metadata = [{"name": "layers.3.self_attn", "layer_index": 3, "num_heads": 2, "head_dim": 4}]
    report = {
        "schema_version": 1,
        "enabled": True,
        "contract": relation,
        "input_units": "ET2 TRT prediction units; not assumed milliseconds",
        "phase_coefficient_units": "radians per ET2 TRT prediction unit",
        "installed_layers": metadata,
        "layers": [{
            "name": "backbone.layers.3.self_attn.gaze_relative_rotation",
            "num_heads": 2,
            "alpha": [0.25, -0.5],
            "alpha_l2_norm": math.hypot(0.25, -0.5),
            "alpha_max_abs": 0.5,
            "scale": relation["scale"],
            "phase_coefficient_per_trt_unit": [0.25 / relation["scale"], -0.5 / relation["scale"]],
            "trainable_parameters": 2,
        }],
        "total_trainable_parameters": 2,
    }
    for fold in (1, 2):
        fold_dir = run_dir / f"heldout_fold{fold}"
        (fold_dir / "gaze_relation_parameters.json").write_text(json.dumps(report), encoding="utf-8")
        path = fold_dir / "final_model" / "decoder_va_architecture.json"
        architecture = json.loads(path.read_text(encoding="utf-8"))
        architecture["gaze_relation_layers"] = metadata
        path.write_text(json.dumps(architecture), encoding="utf-8")


@pytest.mark.parametrize("shuffled", [False, True])
def test_package_includes_optional_relation_report_and_preserves_condition(tmp_path, shuffled):
    """Package coefficients verbatim while retaining the shuffled control identity."""

    options = {"input_mode": "shuffled", "shuffle_seed": 42} if shuffled else {}
    relation = gaze_relation_contract("trt-relative-rotation", scale=2.0, **options)
    run_dir = tmp_path / "reported_relation"
    _write_completed_run(run_dir, gaze_fusion="prefix-concat", schema_version=9, gaze_relation=relation)
    _write_relation_parameter_reports(run_dir, relation)
    archive_path = package_results(run_dir, results_root=tmp_path)
    assert ("_shuffled42" in archive_path.name) == shuffled
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(f"{run_dir.name}/results_manifest.json"))
        assert manifest["condition"]["gaze_relation"] == relation
        for fold in (1, 2):
            relative = f"heldout_fold{fold}/gaze_relation_parameters.json"
            assert archive.read(f"{run_dir.name}/{relative}") == (run_dir / relative).read_bytes()


@pytest.mark.parametrize("mismatch", [
    "shuffle_seed", "enabled", "alpha_nan", "alpha_inf", "alpha_length",
    "num_heads", "scale", "phase_coefficient", "norm", "max", "trainable_count", "architecture_layers",
])
def test_package_rejects_inconsistent_relation_parameter_report(tmp_path, mismatch):
    """Prevent finite-looking summaries or changed control seeds from mislabeling a run."""

    relation = gaze_relation_contract("trt-relative-rotation", scale=2.0, input_mode="shuffled", shuffle_seed=42)
    run_dir = tmp_path / "invalid_report"
    _write_completed_run(run_dir, gaze_fusion="prefix-concat", schema_version=9, gaze_relation=relation)
    _write_relation_parameter_reports(run_dir, relation)
    path = run_dir / "heldout_fold1/gaze_relation_parameters.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    layer = report["layers"][0]
    if mismatch == "shuffle_seed":
        report["contract"]["shuffle_seed"] = 43
    elif mismatch == "enabled":
        report["enabled"] = False
    elif mismatch in ("alpha_nan", "alpha_inf"):
        layer["alpha"][0] = float("nan" if mismatch == "alpha_nan" else "inf")
    elif mismatch == "alpha_length":
        layer["alpha"].pop()
    elif mismatch == "num_heads":
        layer["num_heads"] = 3
    elif mismatch == "scale":
        layer["scale"] = 4.0
    elif mismatch == "phase_coefficient":
        layer["phase_coefficient_per_trt_unit"][0] = 9.0
    elif mismatch == "norm":
        layer["alpha_l2_norm"] = 0.0
    elif mismatch == "max":
        layer["alpha_max_abs"] = 0.0
    elif mismatch == "trainable_count":
        report["total_trainable_parameters"] = 4
    else:
        report["installed_layers"][0]["head_dim"] = 8
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="gaze_relation_parameters"):
        package_results(run_dir, results_root=tmp_path)
    assert not list(run_dir.glob("*.zip"))


def test_optional_relation_report_symlink_uses_existing_safe_snapshot_rejection(tmp_path):
    """A new optional evidence file must obey the same symlink checks as required files."""

    relation = gaze_relation_contract("trt-relative-rotation")
    run_dir = tmp_path / "symlink_report"
    _write_completed_run(run_dir, gaze_fusion="prefix-concat", schema_version=9, gaze_relation=relation)
    outside = tmp_path / "outside_report.json"
    outside.write_text("{}", encoding="utf-8")
    (run_dir / "heldout_fold1/gaze_relation_parameters.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        package_results(run_dir, results_root=tmp_path)

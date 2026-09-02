from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from va_model_code.decoder_va.downloads import sha256_file
from va_model_code.decoder_va.external_benchmarks import (
    EXTERNAL_EVALUATION_COMPLETION_MARKER,
    IDEST_FILE,
    IDEST_NAME,
    MSP_NAME,
    OMG_NAME,
    SEMEVAL_NAME,
    SEMEVAL_TEST_FILES,
    ExternalBenchmarkData,
    TextBatchCollator,
    TokenizedTextDataset,
    audit_finetuning_text_overlap,
    build_prediction_report,
    calculate_external_metrics,
    discover_completed_run,
    fixed_unweighted_ensemble,
    load_idest_english,
    load_msp_podcast_test,
    load_msp_transcript_mapping,
    load_omg_emotion_test,
    load_semeval_2026_subtask1_test,
    model_predictions_to_native,
    normalize_msp_file_id,
    normalize_overlap_text,
    reject_benchmark_training_sources,
    semeval_subtask1_official_metrics,
    write_external_evaluation,
    _atomic_publish_directory_no_replace,
    _validate_zip_members,
)
from va_model_code.decoder_va.evaluation import calculate_va_metrics
from va_model_code.decoder_va.model import (
    ARCHITECTURE_MANIFEST_VERSION,
    ARCHITECTURE_MANIFEST_FILENAME,
    DEFAULT_CLASSIFIER_DROPOUT,
    DEFAULT_GAZE_PROJECTION_DIM,
    DEFAULT_GAZE_PROJECTION_DROPOUT,
    SAFE_WEIGHTS_FILENAME,
)
from va_model_code.decoder_va.preprocessing import FOLD_FILENAMES


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "</s>"
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(self, text, max_length, truncation, padding):
        del truncation, padding
        values = [ord(character) % 31 + 3 for character in str(text)][:max_length]
        return {"input_ids": values, "attention_mask": [1] * len(values)}

    def pad(self, features, padding, pad_to_multiple_of, return_tensors):
        del padding, return_tensors
        maximum = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of:
            maximum = ((maximum + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        ids = []
        masks = []
        for feature in features:
            amount = maximum - len(feature["input_ids"])
            ids.append(feature["input_ids"] + [self.pad_token_id] * amount)
            masks.append(feature["attention_mask"] + [0] * amount)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def _write_omg_fixture(
    directory: Path,
    *,
    missing_second: bool = False,
    duplicate_transcript: bool = False,
    invalid_valence: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    transcripts = pd.DataFrame(
        {
            "link": ["a", "b", "extra"],
            "video": ["v1", "v2", "v3"],
            "utterance": ["utterance_1.mp4", "utterance_2.mp4", "utterance_3.mp4"],
            "transcript": ["second official row", "", "no gold row"],
        }
    )
    if missing_second:
        transcripts = transcripts.iloc[[0, 2]].copy()
    if duplicate_transcript:
        transcripts = pd.concat((transcripts, transcripts.iloc[[0]]), ignore_index=True)
    labels = pd.DataFrame(
        {
            "link": ["b", "a"],
            "start": [0.0, 1.0],
            "end": [1.0, 2.0],
            "video": ["v2", "v1"],
            "utterance": ["utterance_2.mp4", "utterance_1.mp4"],
            "arousal": [0.25, 0.75],
            "valence": [2.0 if invalid_valence else -0.5, 0.5],
            "EmotionMaxVote": [4, 3],
        }
    )
    transcripts.to_csv(directory / "omg_TestTranscripts.tsv", index=False)
    labels.to_csv(directory / "omg_TestVideos_WithLabels.csv", index=False)


def _write_idest_fixture(
    directory: Path,
    *,
    duplicate_code: bool = False,
    blank_english_text: bool = False,
    invalid_valence: bool = False,
) -> None:
    """Write the smallest structurally valid IDEST-style semicolon table."""

    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "code": ["story-3", "story-1", "story-2"],
            "country": ["Finland", "Germany", "Spain"],
            "language": ["Finnish", "German", "Spanish"],
            "text_original": [
                "alkuperainen kolme",
                "urspruenglich eins",
                "original dos",
            ],
            "text_english": [
                "English translation three",
                "English translation one",
                "English translation two",
            ],
            "number_raters_English": ["20", "21", "22"],
            "valence_mean_English": ["1", "5", "9"],
            "arousal_mean_English": ["9", "5", "1"],
            "characters_English": ["25", "23", "23"],
            "words_English": ["3", "3", "3"],
            "StoryType": ["1", "2", "8"],
        }
    )
    if duplicate_code:
        frame.loc[2, "code"] = frame.loc[0, "code"]
    if blank_english_text:
        frame.loc[1, "text_english"] = "   "
    if invalid_valence:
        frame.loc[1, "valence_mean_English"] = "9.01"
    frame.to_csv(
        directory / IDEST_FILE["filename"],
        sep=";",
        index=False,
        encoding="cp1252",
    )


def _semeval_fixture_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid input-order and deliberately shuffled-gold task tables."""

    inputs = pd.DataFrame(
        {
            "user_id": ["u2", "u1", "u1"],
            "text_id": ["t3", "t1", "t2"],
            "text": ["third in input order", "first text", "second text"],
            "timestamp": [
                "2025-01-03T12:00:00",
                "2025-01-01T12:00:00",
                "2025-01-02T12:00:00",
            ],
            "collection_phase": ["2", "1", "1"],
            "is_words": ["True", "False", "True"],
            "is_seen_user": ["False", "True", "True"],
        }
    )
    labels = inputs.iloc[[2, 0, 1]].copy().reset_index(drop=True)
    native_by_key = {
        ("u2", "t3"): (-2.0, 2.0),
        ("u1", "t1"): (0.0, 1.0),
        ("u1", "t2"): (2.0, 0.0),
    }
    labels["valence"] = [
        native_by_key[(user, text)][0]
        for user, text in labels[["user_id", "text_id"]].itertuples(index=False)
    ]
    labels["arousal"] = [
        native_by_key[(user, text)][1]
        for user, text in labels[["user_id", "text_id"]].itertuples(index=False)
    ]
    return inputs, labels


def _write_semeval_fixture(
    directory: Path,
    *,
    duplicate_input_key: bool = False,
    unmatched_gold_key: bool = False,
    mismatched_shared_metadata: bool = False,
    invalid_valence: bool = False,
    invalid_arousal: bool = False,
    invalid_boolean: bool = False,
) -> None:
    """Write minimal SemEval Subtask 1 inputs and released-gold fixtures."""

    directory.mkdir(parents=True, exist_ok=True)
    inputs, labels = _semeval_fixture_tables()
    if duplicate_input_key:
        inputs = pd.concat((inputs, inputs.iloc[[0]]), ignore_index=True)
    if unmatched_gold_key:
        labels.loc[0, "text_id"] = "gold-only"
    if mismatched_shared_metadata:
        labels.loc[0, "text"] = "metadata disagreement"
    if invalid_valence:
        labels.loc[0, "valence"] = 2.01
    if invalid_arousal:
        labels.loc[0, "arousal"] = -0.01
    if invalid_boolean:
        inputs.loc[0, "is_words"] = "yes"
        matching = (
            labels["user_id"].eq(inputs.loc[0, "user_id"])
            & labels["text_id"].eq(inputs.loc[0, "text_id"])
        )
        labels.loc[matching, "is_words"] = "yes"
    inputs.to_csv(
        directory / SEMEVAL_TEST_FILES["inputs"]["filename"], index=False
    )
    labels.to_csv(
        directory / SEMEVAL_TEST_FILES["labels"]["filename"], index=False
    )


def _small_benchmark() -> ExternalBenchmarkData:
    frame = pd.DataFrame(
        {
            "index": ["x", "y", "z"],
            "benchmark_id": ["x", "y", "z"],
            "text": ["same text", "new text", ""],
            "text_sha256": ["1", "2", "3"],
            "is_empty_text": [False, False, True],
            "dataset_of_origin": ["synthetic"] * 3,
            "split": ["test"] * 3,
            "video": ["v1", "v1", "v2"],
            "utterance": ["u1.mp4", "u2.mp4", "u1.mp4"],
            "valence": [0.0, 0.5, 1.0],
            "arousal": [0.0, 0.5, 1.0],
            "native_valence": [-1.0, 0.0, 1.0],
            "native_arousal": [0.0, 0.5, 1.0],
        }
    )
    return ExternalBenchmarkData(
        name=OMG_NAME,
        version="fixture",
        split="test",
        frame=frame,
        native_scale={"valence": (-1.0, 1.0), "arousal": (0.0, 1.0)},
        model_to_native_scale=(2.0, 1.0),
        model_to_native_offset=(-1.0, 0.0),
        source_manifest={},
        join_report={},
    )


def _run_contract_fields(root: Path) -> dict:
    return {
        "architecture_manifest_version": ARCHITECTURE_MANIFEST_VERSION,
        "model": "qwen3.5-0.8b",
        "loss": "mse",
        "output_dim": 2,
        "dtype": "bfloat16",
        "model_id": "Qwen/Qwen3.5-0.8B-Base",
        "model_revision": "a" * 40,
        "finetuning_mode": "full",
        "gaze_fusion": "prefix-concat",
        "gaze_features": ["TRT"],
        "gaze_feature_indices": [3],
        "features_used": [0, 0, 0, 1, 0],
        "gaze_concat_order": "eye_start, compact_selected_gaze, eye_end, text",
        "pooling_position": "last_valid_text_token_after_gaze_prefix",
        "output_activation": "hard_sigmoid",
        "et_model_id": "skboy/et_prediction_2",
        "et_revision": "b" * 40,
        "et_filename": "et.safetensors",
        "et_cache_size": 100,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "attn_implementation": None,
        "max_length": 200,
        "eval_batch_size": 8,
        "excluded_dataset_names": ["IEMOCAP sentences"],
        "fold_sha256": {
            "full_dataset_fold1.csv": "c" * 64,
            "full_dataset_fold2.csv": "d" * 64,
        },
        "dataset_counts_after_filter": {"Emobank": 4},
        "held_out_folds": [1, 2],
        "seed": 42,
        "run_name": root.name,
        "effective_output_dir": str(root.resolve()),
    }


def _saved_metrics(frame: pd.DataFrame, *, prefix: str = "") -> dict:
    values = calculate_va_metrics(
        frame[["valence", "arousal"]].to_numpy(dtype=np.float64),
        frame[["pred_valence", "pred_arousal"]].to_numpy(dtype=np.float64),
    )
    return {
        f"{prefix}{name}": (
            float(value) if np.isfinite(value) else None
        )
        for name, value in values.items()
    }


def _write_completed_run(root: Path, *, mismatch: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    parameters = _run_contract_fields(root)
    (root / "training_parameters.json").write_text(
        json.dumps(parameters), encoding="utf-8"
    )
    fold_frames = []
    for held_out in (1, 2):
        fold = root / f"heldout_fold{held_out}"
        model = fold / "final_model"
        model.mkdir(parents=True)
        first_index = 1 if held_out == 1 else 3
        prediction_frame = pd.DataFrame(
            {
                "index": [first_index, first_index + 1],
                "held_out_fold": [held_out, held_out],
                "text": [f"text-{first_index}", f"text-{first_index + 1}"],
                "dataset_of_origin": ["Emobank", "Emobank"],
                "valence": [0.1 * first_index, 0.1 * (first_index + 1)],
                "arousal": [0.2, 0.8],
                "pred_valence": [0.15 * first_index, 0.1 * (first_index + 1)],
                "pred_arousal": [0.25, 0.75],
            }
        )
        prediction_frame.to_csv(fold / "predictions.tsv", sep="\t", index=False)
        (fold / "metrics.json").write_text(
            json.dumps(_saved_metrics(prediction_frame, prefix="test_")),
            encoding="utf-8",
        )
        manifest = {
            **parameters,
            "held_out_fold": held_out,
            "training_fold": 2 if held_out == 1 else 1,
            "fold_seed": 42 + held_out - 1,
            "training_rows": 2,
            "evaluation_rows": 2,
            "total_parameters": 100,
            "trainable_parameters": 100,
            "trainable_fraction": 1.0,
        }
        if mismatch and held_out == 2:
            manifest["max_length"] = 201
        (fold / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        architecture = {
            "schema_version": ARCHITECTURE_MANIFEST_VERSION,
            "decoder_model_id": parameters["model_id"],
            "decoder_commit": parameters["model_revision"],
            "finetuning_mode": parameters["finetuning_mode"],
            "gaze_fusion": parameters["gaze_fusion"],
            "gaze_features": parameters["gaze_features"],
            "gaze_feature_indices": parameters["gaze_feature_indices"],
            "features_used": parameters["features_used"],
            "et_model": {
                "repo_id": parameters["et_model_id"],
                "revision": parameters["et_revision"],
                "filename": parameters["et_filename"],
                "feature_names": parameters["gaze_features"],
                "feature_indices": parameters["gaze_feature_indices"],
                "features_used": parameters["features_used"],
            },
            "gaze_concat_order": parameters["gaze_concat_order"],
            "pooling_position": parameters["pooling_position"],
            "output_activation": "hard_sigmoid",
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
                "gaze_projection_dim": DEFAULT_GAZE_PROJECTION_DIM,
                "gaze_projection_dropout": list(DEFAULT_GAZE_PROJECTION_DROPOUT),
                "classifier_dropout": DEFAULT_CLASSIFIER_DROPOUT,
                "attn_implementation": parameters["attn_implementation"],
                "backbone_dtype_at_construction": parameters["dtype"],
            },
            "state_dict": {
                "filename": SAFE_WEIGHTS_FILENAME,
                "format": "safetensors",
                "scope": "complete DecoderVARegressor state_dict; ET2 remains external and frozen",
                "strict_loading": True,
            },
            "total_parameters": 100,
            "trainable_parameters": 100,
            "trainable_fraction": 1.0,
        }
        (model / ARCHITECTURE_MANIFEST_FILENAME).write_text(
            json.dumps(architecture), encoding="utf-8"
        )
        (model / SAFE_WEIGHTS_FILENAME).write_bytes(f"weights-{held_out}".encode())
        (model / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "TinyTokenizer"}), encoding="utf-8"
        )
        (model / "tokenizer.json").write_text(
            json.dumps({"version": "1.0", "model": {"type": "WordLevel"}}),
            encoding="utf-8",
        )
        fold_frames.append(prediction_frame)

    oof = pd.concat(fold_frames, ignore_index=True).sort_values(
        ["index", "held_out_fold"], kind="stable"
    )
    oof.to_csv(root / "oof_predictions.tsv", sep="\t", index=False)
    (root / "oof_metrics.json").write_text(
        json.dumps(_saved_metrics(oof)), encoding="utf-8"
    )
    dataset_row = {"dataset_of_origin": "Emobank", **_saved_metrics(oof)}
    pd.DataFrame([dataset_row]).to_csv(
        root / "metrics_by_dataset.tsv", sep="\t", index=False
    )


def test_fixed_scale_maps_and_unweighted_ensemble_exactly():
    first = np.array([[0.0, 0.25], [0.5, 1.0]])
    second = np.array([[1.0, 0.75], [0.25, 0.0]])

    ensemble = fixed_unweighted_ensemble((first, second))
    omg_native = model_predictions_to_native(
        ensemble, scale=(2.0, 1.0), offset=(-1.0, 0.0)
    )
    msp_native = model_predictions_to_native(
        ensemble, scale=(6.0, 6.0), offset=(1.0, 1.0)
    )

    np.testing.assert_allclose(ensemble, [[0.5, 0.5], [0.375, 0.5]])
    np.testing.assert_allclose(omg_native, [[0.0, 0.5], [-0.25, 0.5]])
    np.testing.assert_allclose(msp_native, [[4.0, 4.0], [3.25, 4.0]])


def test_ensemble_rejects_single_member_and_shape_mismatch():
    with pytest.raises(ValueError, match="at least two"):
        fixed_unweighted_ensemble((np.zeros((2, 2)),))
    with pytest.raises(ValueError, match="same ordered rows"):
        fixed_unweighted_ensemble((np.zeros((2, 2)), np.zeros((3, 2))))


def test_omg_loader_uses_comma_tsv_gold_order_blank_and_fixed_scale(tmp_path):
    _write_omg_fixture(tmp_path)

    data = load_omg_emotion_test(tmp_path, strict_official_contract=False)

    assert data.frame["benchmark_id"].tolist() == [
        "v2::utterance_2.mp4",
        "v1::utterance_1.mp4",
    ]
    assert data.frame["text"].tolist() == ["", "second official row"]
    assert data.frame["is_empty_text"].tolist() == [True, False]
    assert data.frame["valence"].tolist() == [0.25, 0.75]
    assert data.frame["arousal"].tolist() == [0.25, 0.75]
    assert data.join_report["transcript_only_rows"] == 1
    assert data.join_report["missing_gold_transcripts"] == 0
    assert data.version == "unverified-local"
    assert data.source_manifest["canonical_contract_verified"] is False


@pytest.mark.parametrize(
    "fixture_kwargs,error",
    (
        ({"missing_second": True}, "missing transcript"),
        ({"duplicate_transcript": True}, "not unique"),
        ({"invalid_valence": True}, "must stay"),
    ),
)
def test_omg_loader_rejects_missing_duplicate_and_out_of_range(
    tmp_path, fixture_kwargs, error
):
    _write_omg_fixture(tmp_path, **fixture_kwargs)

    with pytest.raises(ValueError, match=error):
        load_omg_emotion_test(tmp_path, strict_official_contract=False)


def test_omg_strict_contract_rejects_unpinned_fixture(tmp_path):
    _write_omg_fixture(tmp_path)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_omg_emotion_test(tmp_path, strict_official_contract=True)


def test_idest_loader_uses_english_text_source_order_and_fixed_scale(tmp_path):
    _write_idest_fixture(tmp_path)

    data = load_idest_english(tmp_path, strict_official_contract=False)

    assert data.name == IDEST_NAME
    assert data.version == "unverified-local"
    assert data.frame["benchmark_id"].tolist() == [
        "story-3",
        "story-1",
        "story-2",
    ]
    assert data.frame["text"].tolist() == [
        "English translation three",
        "English translation one",
        "English translation two",
    ]
    assert not data.frame["text"].str.contains("urspruenglich").any()
    np.testing.assert_allclose(
        data.frame[["valence", "arousal"]].to_numpy(),
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
    )
    np.testing.assert_allclose(
        data.predictions_to_native(np.array([[0.0, 0.0], [1.0, 1.0]])),
        [[1.0, 1.0], [9.0, 9.0]],
    )
    assert data.join_report["english_story_rows"] == 3
    assert data.source_manifest["english_input_column"] == "text_english"
    assert data.source_manifest["canonical_contract_verified"] is False


@pytest.mark.parametrize(
    "fixture_kwargs,error",
    (
        ({"duplicate_code": True}, "story codes are not unique"),
        ({"blank_english_text": True}, "blank text_english"),
        ({"invalid_valence": True}, r"valence must stay in \[1, 9\]"),
    ),
)
def test_idest_loader_rejects_invalid_rows(tmp_path, fixture_kwargs, error):
    _write_idest_fixture(tmp_path, **fixture_kwargs)

    with pytest.raises(ValueError, match=error):
        load_idest_english(tmp_path, strict_official_contract=False)


def test_idest_strict_contract_rejects_wrong_row_count_after_hash_check(
    tmp_path, monkeypatch
):
    _write_idest_fixture(tmp_path)
    path = tmp_path / IDEST_FILE["filename"]
    monkeypatch.setitem(IDEST_FILE, "sha256", sha256_file(path))

    with pytest.raises(ValueError, match="must contain 250 stories; got 3"):
        load_idest_english(tmp_path, strict_official_contract=True)


def test_semeval_loader_preserves_input_order_and_maps_pinned_native_scales(
    tmp_path,
):
    _write_semeval_fixture(tmp_path)

    data = load_semeval_2026_subtask1_test(
        tmp_path, strict_official_contract=False
    )

    assert data.name == SEMEVAL_NAME
    assert data.version == "unverified-local"
    assert data.frame["benchmark_id"].tolist() == [
        "u2::t3",
        "u1::t1",
        "u1::t2",
    ]
    assert data.frame["text"].tolist() == [
        "third in input order",
        "first text",
        "second text",
    ]
    np.testing.assert_allclose(
        data.frame[["native_valence", "native_arousal"]].to_numpy(),
        [[-2.0, 2.0], [0.0, 1.0], [2.0, 0.0]],
    )
    np.testing.assert_allclose(
        data.frame[["valence", "arousal"]].to_numpy(),
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
    )
    np.testing.assert_allclose(
        data.predictions_to_native(np.array([[0.0, 0.0], [1.0, 1.0]])),
        [[-2.0, 0.0], [2.0, 2.0]],
    )
    assert data.frame["is_seen_user"].tolist() == [False, True, True]
    assert data.frame["is_words"].tolist() == [True, False, True]
    assert data.join_report["join_validation"].startswith("one_to_one")
    assert data.source_manifest["canonical_contract_verified"] is False


@pytest.mark.parametrize(
    "fixture_kwargs,error",
    (
        ({"duplicate_input_key": True}, "inputs keys are not unique"),
        ({"unmatched_gold_key": True}, "keys do not match one-to-one"),
        ({"mismatched_shared_metadata": True}, "text values disagree"),
        ({"invalid_valence": True}, r"valence must stay in \[-2, 2\]"),
        ({"invalid_arousal": True}, r"arousal must stay in \[0, 2\]"),
        ({"invalid_boolean": True}, "must contain only True/False"),
    ),
)
def test_semeval_loader_rejects_contract_violations(
    tmp_path, fixture_kwargs, error
):
    _write_semeval_fixture(tmp_path, **fixture_kwargs)

    with pytest.raises(ValueError, match=error):
        load_semeval_2026_subtask1_test(
            tmp_path, strict_official_contract=False
        )


def test_semeval_strict_contract_rejects_wrong_row_count_after_hash_check(
    tmp_path, monkeypatch
):
    _write_semeval_fixture(tmp_path)
    for role, contract in SEMEVAL_TEST_FILES.items():
        path = tmp_path / contract["filename"]
        monkeypatch.setitem(contract, "sha256", sha256_file(path))

    with pytest.raises(ValueError, match="must contain 1737 rows; got 3"):
        load_semeval_2026_subtask1_test(
            tmp_path, strict_official_contract=True
        )


def test_semeval_official_scorer_matches_reference_calculation():
    users = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    labels_one_dimension = np.array([0, 1, 2, 1, 2, 4, 2, 4, 5], dtype=float)
    predictions_one_dimension = np.array(
        [0.2, 1.7, 1.3, 1.4, 1.6, 3.8, 2.5, 3.6, 5.3],
        dtype=float,
    )
    labels = np.column_stack((labels_one_dimension, labels_one_dimension))
    predictions = np.column_stack(
        (predictions_one_dimension, predictions_one_dimension)
    )

    metrics = semeval_subtask1_official_metrics(labels, predictions, users)

    assert metrics["r_composite_mean_va"] == pytest.approx(0.9820382176998993)
    assert metrics[
        "mae_composite_mean_va_official_implementation"
    ] == pytest.approx(0.26340028566258467)
    for dimension in ("valence", "arousal"):
        result = metrics["dimensions"][dimension]
        assert result["r_within"] == pytest.approx(0.875417724795919)
        assert result["r_between"] == pytest.approx(0.9975304945757417)
        assert result["r_composite"] == pytest.approx(0.9820382176998993)
        assert result["mae_within"] == pytest.approx(0.4222222222222222)
        assert result["mae_between"] == pytest.approx(0.0888888888888888)
        assert result[
            "mae_composite_official_implementation"
        ] == pytest.approx(0.26340028566258467)


def test_msp_loader_maps_official_columns_and_preserves_test_order(tmp_path):
    labels = pd.DataFrame(
        {
            "FileName": ["train.wav", "MSP_2.wav", "MSP_1.wav"],
            "Split_Set": ["Train", "Test2", "Test2"],
            "EmoAct": [4.0, 7.0, 1.0],
            "EmoVal": [4.0, 1.0, 7.0],
            "EmoDom": [4.0, 3.0, 5.0],
        }
    )
    transcripts = pd.DataFrame(
        {
            "FileName": ["MSP_1.wav", "MSP_2.wav", "train.wav"],
            "Transcript": ["positive", "negative", "training"],
        }
    )
    label_path = tmp_path / "labels_consensus.csv"
    transcript_path = tmp_path / "transcripts.tsv"
    labels.to_csv(label_path, index=False)
    transcripts.to_csv(transcript_path, sep="\t", index=False)

    data = load_msp_podcast_test(
        label_path,
        transcript_path,
        split="test2",
        strict_official_count=False,
    )

    assert data.frame["file_name"].tolist() == ["MSP_2.wav", "MSP_1.wav"]
    assert data.frame["text"].tolist() == ["negative", "positive"]
    np.testing.assert_allclose(data.frame[["valence", "arousal"]], [[0, 1], [1, 0]])
    assert data.join_report["unused_transcript_rows"] == 1
    assert data.name == MSP_NAME
    assert data.version == "unverified-local"
    assert data.source_manifest["canonical_contract_verified"] is False


def test_msp_strict_contract_rejects_noncanonical_label_columns(tmp_path):
    label_path = tmp_path / "labels.csv"
    transcript_path = tmp_path / "transcripts.csv"
    label_path.write_text(
        "FileName,Split_Set,EmoAct,EmoVal,EmoDom\na.wav,Test1,4,4,5\n"
    )
    transcript_path.write_text("FileName,Transcript\na.wav,text\n")

    with pytest.raises(ValueError, match="fixes the official label schema"):
        load_msp_podcast_test(
            label_path,
            transcript_path,
            split="test1",
            valence_column="EmoDom",
        )


@pytest.mark.parametrize("split", ("train", "development", "test3", "validation"))
def test_msp_loader_rejects_every_non_test1_test2_split(tmp_path, split):
    label_path = tmp_path / "labels.csv"
    transcript_path = tmp_path / "transcripts.csv"
    label_path.write_text("FileName,Split_Set,EmoAct,EmoVal\na.wav,Test1,4,4\n")
    transcript_path.write_text("FileName,Transcript\na.wav,text\n")

    with pytest.raises(ValueError, match="permits only test1 or test2"):
        load_msp_podcast_test(
            label_path,
            transcript_path,
            split=split,
            strict_official_count=False,
        )


def test_msp_loader_rejects_wrong_v2_official_count(tmp_path):
    label_path = tmp_path / "labels.csv"
    transcript_path = tmp_path / "transcripts.csv"
    label_path.write_text("FileName,Split_Set,EmoAct,EmoVal\na.wav,Test1,4,4\n")
    transcript_path.write_text("FileName,Transcript\na.wav,text\n")

    with pytest.raises(ValueError, match="must contain 46294"):
        load_msp_podcast_test(label_path, transcript_path, split="test1")


def test_msp_loader_rejects_missing_transcript_without_dropping(tmp_path):
    label_path = tmp_path / "labels.csv"
    transcript_path = tmp_path / "transcripts.csv"
    label_path.write_text(
        "FileName,Split_Set,EmoAct,EmoVal\na.wav,Test1,4,4\nb.wav,Test1,5,3\n"
    )
    transcript_path.write_text("FileName,Transcript\na.wav,text\n")

    with pytest.raises(ValueError, match="without transcripts"):
        load_msp_podcast_test(
            label_path,
            transcript_path,
            split="test1",
            strict_official_count=False,
        )


def test_msp_loader_rejects_out_of_range_native_labels(tmp_path):
    label_path = tmp_path / "labels.csv"
    transcript_path = tmp_path / "transcripts.csv"
    label_path.write_text("FileName,Split_Set,EmoAct,EmoVal\na.wav,Test1,8,4\n")
    transcript_path.write_text("FileName,Transcript\na.wav,text\n")

    with pytest.raises(ValueError, match=r"must stay in \[1, 7\]"):
        load_msp_podcast_test(
            label_path,
            transcript_path,
            split="test1",
            strict_official_count=False,
        )


def test_msp_loader_rejects_matched_blank_transcript(tmp_path):
    label_path = tmp_path / "labels.csv"
    transcript_path = tmp_path / "transcripts.csv"
    label_path.write_text("FileName,Split_Set,EmoAct,EmoVal\na.wav,Test1,4,4\n")
    transcript_path.write_text("FileName,Transcript\na.wav,   \n")

    with pytest.raises(ValueError, match="matched blank transcripts"):
        load_msp_podcast_test(
            label_path,
            transcript_path,
            split="test1",
            strict_official_count=False,
        )


def test_msp_transcript_table_rejects_duplicate_normalized_ids(tmp_path):
    transcript_path = tmp_path / "transcripts.csv"
    transcript_path.write_text(
        "FileName,Transcript\nMSP_1.wav,first\nmsp_1.txt,duplicate\n"
    )

    with pytest.raises(ValueError, match="duplicate normalized ID"):
        load_msp_transcript_mapping(transcript_path)


def test_msp_explicit_transcript_columns_never_fall_back_to_aliases(tmp_path):
    transcript_path = tmp_path / "transcripts.csv"
    transcript_path.write_text("FileName,Transcript\na.wav,text\n")

    with pytest.raises(ValueError, match="requested ID column"):
        load_msp_transcript_mapping(transcript_path, id_column="TypoID")
    with pytest.raises(ValueError, match="requested transcript column"):
        load_msp_transcript_mapping(transcript_path, text_column="TypoText")


def test_msp_transcript_directory_and_zip_are_keyed_by_stem(tmp_path):
    directory = tmp_path / "txt"
    directory.mkdir()
    (directory / "A.txt").write_text("first", encoding="utf-8")
    (directory / "B.txt").write_text("", encoding="utf-8")
    mapping, details = load_msp_transcript_mapping(directory)
    assert mapping == {"a": "first", "b": ""}
    assert details["format"] == "directory"

    archive_path = tmp_path / "transcripts.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/C.txt", "third")
        archive.writestr("nested/D.txt", "fourth")
    zipped, zip_details = load_msp_transcript_mapping(archive_path)
    assert zipped == {"c": "third", "d": "fourth"}
    assert zip_details["format"] == "zip-text-files"


def test_msp_transcript_zip_rejects_unsafe_member(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../A.txt", "text")

    with pytest.raises(ValueError, match="Unsafe"):
        load_msp_transcript_mapping(archive_path)


def test_msp_zip_allows_release_scale_table_but_keeps_text_member_cap(tmp_path):
    table = zipfile.ZipInfo("Transcripts.csv")
    table.file_size = 64 * 1024 * 1024
    table.flag_bits = 0
    text_member = zipfile.ZipInfo("one.txt")
    text_member.file_size = 64 * 1024 * 1024
    text_member.flag_bits = 0

    table_archive = SimpleNamespace(infolist=lambda: [table])
    assert _validate_zip_members(table_archive, tmp_path / "table.zip") == [table]
    text_archive = SimpleNamespace(infolist=lambda: [text_member])
    with pytest.raises(ValueError, match="Implausibly large MSP transcript member"):
        _validate_zip_members(text_archive, tmp_path / "text.zip")


def test_msp_transcript_zip_prefers_valid_table_over_readme_text(tmp_path):
    archive_path = tmp_path / "table.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "release notes")
        archive.writestr(
            "Transcripts.csv",
            "FileName,Transcript\nMSP_1.wav,actual transcript\n",
        )

    mapping, details = load_msp_transcript_mapping(archive_path)

    assert mapping == {"msp_1": "actual transcript"}
    assert details["format"] == "zip-table"


def test_msp_zip_explicit_table_override_never_falls_back_to_readme(tmp_path):
    archive_path = tmp_path / "table.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "release notes")
        archive.writestr(
            "Transcripts.csv",
            "FileName,Transcript\nMSP_1.wav,actual transcript\n",
        )

    with pytest.raises(ValueError, match="requested ID column"):
        load_msp_transcript_mapping(archive_path, id_column="TypoID")


def test_msp_id_normalization_rejects_blank_and_strips_extensions():
    assert normalize_msp_file_id("Folder/MSP_1.WAV") == "msp_1"
    assert normalize_msp_file_id("MSP_1.txt") == "msp_1"
    assert normalize_msp_file_id(r"Folder\MSP_1.WAV.txt") == "msp_1"
    with pytest.raises(ValueError, match="cannot be blank"):
        normalize_msp_file_id("  ")


def test_tokenized_text_dataset_never_emits_labels_and_handles_blank():
    tokenizer = TinyTokenizer()
    dataset = TokenizedTextDataset(["", "ab"], tokenizer, max_length=4)
    collator = TextBatchCollator(tokenizer, pad_to_multiple_of=4)

    blank = dataset[0]
    batch = collator([blank, dataset[1]])

    assert "labels" not in blank
    assert "labels" not in batch
    assert blank["input_ids"].tolist() == [tokenizer.eos_token_id]
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].tolist() == [[1, 0, 0, 0], [1, 1, 0, 0]]


def test_normalized_overlap_uses_nfkc_casefold_whitespace_and_ignores_empty(tmp_path):
    benchmark = _small_benchmark()
    assert normalize_overlap_text("  ＳＡＭＥ\tText ") == "same text"
    fold1 = pd.DataFrame(
        {
            "index": [1, 2],
            "text": ["Same   Text", "excluded"],
            "dataset_of_origin": ["keep", "drop"],
            "valence": [0.5, 0.5],
            "arousal": [0.5, 0.5],
        }
    )
    fold2 = pd.DataFrame(
        {
            "index": [3],
            "text": ["other"],
            "dataset_of_origin": ["keep"],
            "valence": [0.5],
            "arousal": [0.5],
        }
    )
    for filename, frame in zip(FOLD_FILENAMES, (fold1, fold2)):
        frame.to_csv(tmp_path / filename, sep="\t", index=False)
    hashes = {filename: sha256_file(tmp_path / filename) for filename in FOLD_FILENAMES}
    members = (
        SimpleNamespace(
            name="heldout_fold1",
            training_fold=2,
            run_manifest={
                "fold_sha256": hashes,
                "excluded_dataset_names": ["drop"],
                "training_rows": 1,
            },
        ),
        SimpleNamespace(
            name="heldout_fold2",
            training_fold=1,
            run_manifest={
                "fold_sha256": hashes,
                "excluded_dataset_names": ["drop"],
                "training_rows": 1,
            },
        ),
    )

    audited, report, _ = audit_finetuning_text_overlap(
        benchmark, members, tmp_path
    )

    assert audited["overlap_with_any_finetuning_text"].tolist() == [True, False, False]
    assert report["overlapping_external_rows_union"] == 1
    assert report["novel_external_rows_union"] == 2


def test_overlap_audit_rejects_changed_training_fold_hash(tmp_path):
    benchmark = _small_benchmark()
    for filename in FOLD_FILENAMES:
        pd.DataFrame(
            {
                "index": [1],
                "text": ["x"],
                "dataset_of_origin": ["keep"],
                "valence": [0.5],
                "arousal": [0.5],
            }
        ).to_csv(tmp_path / filename, sep="\t", index=False)
    member = SimpleNamespace(
        name="heldout_fold1",
        training_fold=2,
        run_manifest={
            "fold_sha256": {filename: "wrong" for filename in FOLD_FILENAMES},
            "excluded_dataset_names": [],
            "training_rows": 1,
        },
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit_finetuning_text_overlap(benchmark, (member,), tmp_path)


def test_external_metrics_report_official_nonempty_and_novel_subsets():
    benchmark = _small_benchmark()
    audited = benchmark.frame.copy()
    audited["overlap_with_any_finetuning_text"] = [True, False, False]
    predictions = benchmark.labels_model_scale().copy()

    metrics = calculate_external_metrics(
        benchmark, predictions, audited_frame=audited
    )

    assert metrics["primary"] == "subsets.official_all.native_scale"
    assert metrics["subsets"]["official_all"]["n_examples"] == 3
    assert metrics["subsets"]["nonempty_text"]["n_examples"] == 2
    assert metrics["subsets"]["finetuning_novel_text"]["n_examples"] == 2
    assert metrics["subsets"]["official_all"]["native_scale"]["mse_mean"] == 0.0


def test_external_metrics_zero_subset_has_stable_schema():
    benchmark = _small_benchmark()
    audited = benchmark.frame.copy()
    audited["overlap_with_any_finetuning_text"] = False

    metrics = calculate_external_metrics(
        benchmark,
        benchmark.labels_model_scale(),
        audited_frame=audited,
    )

    empty_subset = metrics["subsets"]["finetuning_overlap_text"]
    assert empty_subset == {
        "n_examples": 0,
        "native_scale": None,
        "model_0_1_scale": None,
    }


@pytest.mark.parametrize("operation", ("metrics", "report"))
def test_external_outputs_reject_shuffled_audited_benchmark_ids(operation):
    benchmark = _small_benchmark()
    audited = benchmark.frame.iloc[[1, 0, 2]].reset_index(drop=True)
    predictions = benchmark.labels_model_scale()

    with pytest.raises(ValueError, match="benchmark_id values/order"):
        if operation == "metrics":
            calculate_external_metrics(
                benchmark,
                predictions,
                audited_frame=audited,
            )
        else:
            build_prediction_report(
                benchmark,
                {"heldout_fold1": predictions, "heldout_fold2": predictions},
                predictions,
                audited_frame=audited,
            )


def test_external_metrics_reject_tampered_semeval_group_metadata(tmp_path):
    _write_semeval_fixture(tmp_path)
    benchmark = load_semeval_2026_subtask1_test(
        tmp_path,
        strict_official_contract=False,
    )
    audited = benchmark.frame.copy()
    audited.loc[0, "user_id"] = "tampered-user"

    with pytest.raises(ValueError, match="changed canonical user_id"):
        calculate_external_metrics(
            benchmark,
            benchmark.labels_model_scale(),
            audited_frame=audited,
        )


def test_prediction_report_excludes_raw_text_and_gold_by_default():
    benchmark = _small_benchmark()
    first = np.full((3, 2), 0.25)
    second = np.full((3, 2), 0.75)
    ensemble = fixed_unweighted_ensemble((first, second))

    report = build_prediction_report(
        benchmark,
        {"heldout_fold1": first, "heldout_fold2": second},
        ensemble,
    )

    assert "text" not in report
    assert "gold_valence_native" not in report
    assert report["ensemble_pred_valence_model_0_1"].tolist() == [0.5] * 3


def test_discover_completed_run_requires_matched_two_fold_contract(tmp_path):
    _write_completed_run(tmp_path)

    members = discover_completed_run(tmp_path)

    assert [member.held_out_fold for member in members] == [1, 2]
    assert [member.training_fold for member in members] == [2, 1]
    assert all(len(member.file_sha256["model_weights"]) == 64 for member in members)
    assert all(
        len(member.file_sha256["tokenizer/tokenizer_config.json"]) == 64
        and len(member.file_sha256["tokenizer/tokenizer.json"]) == 64
        for member in members
    )


def test_discover_sanitized_hf_bundle_without_raw_internal_predictions(tmp_path):
    run_root = tmp_path / "original-run"
    _write_completed_run(run_root)
    for filename in ("oof_metrics.json", "oof_predictions.tsv", "metrics_by_dataset.tsv"):
        (run_root / filename).unlink()
    for held_out_fold in (1, 2):
        fold_root = run_root / f"heldout_fold{held_out_fold}"
        (fold_root / "metrics.json").unlink()
        (fold_root / "predictions.tsv").unlink()

    members = discover_completed_run(run_root, require_internal_evidence=False)

    assert [member.held_out_fold for member in members] == [1, 2]
    with pytest.raises(FileNotFoundError, match="completed-run artifact"):
        discover_completed_run(run_root)


def test_discover_allows_relocated_hf_download_but_checks_internal_run_identity(tmp_path):
    original = tmp_path / "original-run"
    relocated = tmp_path / "hf-snapshot-commit"
    _write_completed_run(original)
    shutil.copytree(original, relocated)

    assert len(discover_completed_run(relocated)) == 2

    parameters_path = relocated / "training_parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    parameters["effective_output_dir"] = "/different/internal-name"
    parameters_path.write_text(json.dumps(parameters), encoding="utf-8")
    with pytest.raises(ValueError, match="run_name disagrees"):
        discover_completed_run(relocated)


def test_discover_completed_run_rejects_condition_mismatch(tmp_path):
    _write_completed_run(tmp_path, mismatch=True)

    with pytest.raises(ValueError, match="disagrees with training_parameters.json"):
        discover_completed_run(tmp_path)


def test_discover_completed_run_rejects_architecture_reconstruction_mismatch(tmp_path):
    _write_completed_run(tmp_path)
    path = (
        tmp_path
        / "heldout_fold2"
        / "final_model"
        / ARCHITECTURE_MANIFEST_FILENAME
    )
    architecture = json.loads(path.read_text(encoding="utf-8"))
    architecture["reconstruction"]["decoder_revision"] = "e" * 40
    path.write_text(json.dumps(architecture), encoding="utf-8")

    with pytest.raises(ValueError, match="reconstruction disagrees"):
        discover_completed_run(tmp_path)


def test_discover_completed_run_rejects_swapped_fold_manifests(tmp_path):
    _write_completed_run(tmp_path)
    first = tmp_path / "heldout_fold1" / "run_manifest.json"
    second = tmp_path / "heldout_fold2" / "run_manifest.json"
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    first.write_bytes(second_payload)
    second.write_bytes(first_payload)

    with pytest.raises(ValueError, match="held_out_fold mismatch"):
        discover_completed_run(tmp_path)


def test_discover_completed_run_rejects_oof_not_exact_fold_union(tmp_path):
    _write_completed_run(tmp_path)
    path = tmp_path / "oof_predictions.tsv"
    oof = pd.read_csv(path, sep="\t")
    oof.loc[0, "pred_valence"] = 0.99
    oof.to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="exact union"):
        discover_completed_run(tmp_path)


def test_discover_completed_run_proves_internal_gold_scale_is_zero_one(tmp_path):
    _write_completed_run(tmp_path)
    path = tmp_path / "heldout_fold1" / "predictions.tsv"
    predictions = pd.read_csv(path, sep="\t")
    predictions.loc[0, "valence"] = 1.1
    predictions.to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="out-of-range valence"):
        discover_completed_run(tmp_path)


def test_discover_completed_run_rejects_mismatched_tokenizer_contents(tmp_path):
    _write_completed_run(tmp_path)
    tokenizer = tmp_path / "heldout_fold2" / "final_model" / "tokenizer.json"
    tokenizer.write_text(json.dumps({"version": "different"}), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer artifact inventories or contents differ"):
        discover_completed_run(tmp_path)


def test_discover_completed_run_requires_saved_tokenizer_config(tmp_path):
    _write_completed_run(tmp_path)
    config = tmp_path / "heldout_fold1" / "final_model" / "tokenizer_config.json"
    config.unlink()

    with pytest.raises(FileNotFoundError, match="tokenizer_config.json"):
        discover_completed_run(tmp_path)


def test_discover_completed_run_rejects_results_only_archive_without_weights(tmp_path):
    _write_completed_run(tmp_path)
    (tmp_path / "heldout_fold2" / "final_model" / SAFE_WEIGHTS_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="Results-only archives omit model weights"):
        discover_completed_run(tmp_path)


def test_reject_benchmark_training_source_names():
    clean = SimpleNamespace(run_manifest={"dataset_counts_after_filter": {"Emobank": 3}})
    reject_benchmark_training_sources((clean,), OMG_NAME)
    contaminated = SimpleNamespace(
        run_manifest={"dataset_counts_after_filter": {"OMG Emotion": 3}}
    )

    with pytest.raises(ValueError, match="appears in fine-tuning sources"):
        reject_benchmark_training_sources((contaminated,), OMG_NAME)
    spelled_out = SimpleNamespace(
        run_manifest={
            "dataset_counts_after_filter": {"One Minute Gradual Emotion": 3}
        }
    )
    with pytest.raises(ValueError, match="appears in fine-tuning sources"):
        reject_benchmark_training_sources((spelled_out,), OMG_NAME)


@pytest.mark.parametrize(
    "benchmark_name,training_source",
    (
        (IDEST_NAME, "IDEST"),
        (
            IDEST_NAME,
            "International Database of Emotional Short Texts",
        ),
        (SEMEVAL_NAME, "SemEval"),
        (SEMEVAL_NAME, "SemEval 2026"),
        (SEMEVAL_NAME, "SemEval 2026 Task 2"),
        (SEMEVAL_NAME, "EmotionValArouTimeVariation"),
        (SEMEVAL_NAME, "Ecological Essays"),
    ),
)
def test_reject_new_benchmark_training_source_aliases(
    benchmark_name, training_source
):
    member = SimpleNamespace(
        run_manifest={"dataset_counts_after_filter": {training_source: 3}}
    )

    with pytest.raises(ValueError, match="appears in fine-tuning sources"):
        reject_benchmark_training_sources((member,), benchmark_name)


@pytest.mark.parametrize("benchmark_name", (IDEST_NAME, SEMEVAL_NAME))
def test_new_benchmark_source_rejection_allows_unrelated_training_data(
    benchmark_name,
):
    member = SimpleNamespace(
        run_manifest={"dataset_counts_after_filter": {"Emobank": 3}}
    )

    reject_benchmark_training_sources((member,), benchmark_name)


def test_write_external_evaluation_is_non_overwriting_and_manifest_says_no_training(
    tmp_path,
):
    run_root = tmp_path / "run"
    _write_completed_run(run_root)
    members = discover_completed_run(run_root)
    benchmark = _small_benchmark()
    audited = benchmark.frame.copy()
    audited["overlap_with_any_finetuning_text"] = False
    first = np.full((3, 2), 0.25)
    second = np.full((3, 2), 0.75)
    ensemble = fixed_unweighted_ensemble((first, second))
    output = tmp_path / "external"

    written = write_external_evaluation(
        output,
        benchmark,
        members,
        {"heldout_fold1": first, "heldout_fold2": second},
        ensemble,
        audited_frame=audited,
        overlap_report={"status": "completed"},
        evaluation_manifest={"device": "cpu"},
    )

    manifest = json.loads(
        (written / "external_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(written / "predictions.tsv", sep="\t")
    assert manifest["training_performed"] is False
    assert manifest["gradient_updates"] == 0
    assert manifest["calibration_performed"] is False
    assert manifest["external_model_selection"] is False
    assert "saved by the original completed training run" in manifest["checkpoint_origin"]
    assert "text" not in predictions
    completion = json.loads(
        (written / EXTERNAL_EVALUATION_COMPLETION_MARKER).read_text(encoding="utf-8")
    )
    assert completion == {"schema_version": 1, "status": "completed"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_external_evaluation(
            output,
            benchmark,
            members,
            {"heldout_fold1": first, "heldout_fold2": second},
            ensemble,
            audited_frame=audited,
            overlap_report={},
            evaluation_manifest={},
        )


def test_write_external_evaluation_cleans_staging_on_failure(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    _write_completed_run(run_root)
    members = discover_completed_run(run_root)
    benchmark = _small_benchmark()
    audited = benchmark.frame.copy()
    audited["overlap_with_any_finetuning_text"] = False
    first = np.full((3, 2), 0.25)
    second = np.full((3, 2), 0.75)
    ensemble = fixed_unweighted_ensemble((first, second))
    output = tmp_path / "external"
    original_to_csv = pd.DataFrame.to_csv

    def fail_prediction_write(self, path_or_buf=None, *args, **kwargs):
        if Path(path_or_buf).name == "predictions.tsv":
            raise RuntimeError("simulated write failure")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_prediction_write)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        write_external_evaluation(
            output,
            benchmark,
            members,
            {"heldout_fold1": first, "heldout_fold2": second},
            ensemble,
            audited_frame=audited,
            overlap_report={"status": "completed"},
            evaluation_manifest={"device": "cpu"},
        )

    assert not output.exists()
    assert list(tmp_path.glob(".external.tmp-*")) == []


def test_write_external_evaluation_rejects_member_ensemble_and_manifest_tampering(
    tmp_path,
):
    run_root = tmp_path / "run"
    _write_completed_run(run_root)
    members = discover_completed_run(run_root)
    benchmark = _small_benchmark()
    audited = benchmark.frame.copy()
    audited["overlap_with_any_finetuning_text"] = False
    first = np.full((3, 2), 0.25)
    second = np.full((3, 2), 0.75)
    ensemble = fixed_unweighted_ensemble((first, second))

    with pytest.raises(ValueError, match="keys must exactly match"):
        write_external_evaluation(
            tmp_path / "missing-member",
            benchmark,
            members,
            {"heldout_fold1": first},
            ensemble,
            audited_frame=audited,
            overlap_report={},
            evaluation_manifest={},
        )
    with pytest.raises(ValueError, match="exact fixed unweighted"):
        write_external_evaluation(
            tmp_path / "wrong-ensemble",
            benchmark,
            members,
            {"heldout_fold1": first, "heldout_fold2": second},
            ensemble + 0.01,
            audited_frame=audited,
            overlap_report={},
            evaluation_manifest={},
        )
    with pytest.raises(ValueError, match="cannot override authoritative"):
        write_external_evaluation(
            tmp_path / "reserved-manifest",
            benchmark,
            members,
            {"heldout_fold1": first, "heldout_fold2": second},
            ensemble,
            audited_frame=audited,
            overlap_report={},
            evaluation_manifest={"training_performed": True},
        )
    assert not (tmp_path / "reserved-manifest").exists()
    assert list(tmp_path.glob(".reserved-manifest.tmp-*")) == []


def test_atomic_publish_never_replaces_an_existing_empty_directory(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "payload").write_text("complete", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(FileExistsError):
        _atomic_publish_directory_no_replace(staged, target)

    assert staged.is_dir()
    assert target.is_dir()
    assert list(target.iterdir()) == []

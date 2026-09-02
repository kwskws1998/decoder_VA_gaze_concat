"""Focused regression tests for the inference-only external-evaluation CLI."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch


VA_MODEL_CODE = Path(__file__).resolve().parents[1]
if str(VA_MODEL_CODE) not in sys.path:
    sys.path.insert(0, str(VA_MODEL_CODE))

evaluate_external = importlib.import_module("evaluate_external")
from decoder_va.external_benchmarks import ExternalBenchmarkData


class OrderedTokenizer:
    """Encode integer text values and right-pad batches for order assertions."""

    pad_token_id = 0
    eos_token_id = 9
    eos_token = "</s>"
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(self, text, max_length, truncation, padding):
        del max_length, truncation, padding
        value = int(str(text)) + 1
        return {"input_ids": [value], "attention_mask": [1]}

    def pad(self, features, padding, pad_to_multiple_of, return_tensors):
        del padding, return_tensors
        maximum = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of:
            maximum = (
                (maximum + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of
        input_ids = []
        attention_mask = []
        for feature in features:
            missing = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * missing)
            attention_mask.append(feature["attention_mask"] + [0] * missing)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class LengthAuditTokenizer:
    """Return deterministic untruncated lengths and record label-free calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, text, max_length, truncation, padding):
        self.calls.append(
            {
                "text": text,
                "max_length": max_length,
                "truncation": truncation,
                "padding": padding,
            }
        )
        count = int(str(text))
        return {"input_ids": list(range(count)), "attention_mask": [1] * count}


class RecordingModel(torch.nn.Module):
    """Record inference state and emit deterministic bounded VA predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[dict[str, object]] = []

    def forward(self, **inputs):
        self.calls.append(
            {
                "keys": tuple(sorted(inputs)),
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "input_ids": inputs["input_ids"][:, 0].detach().cpu().tolist(),
            }
        )
        values = inputs["input_ids"][:, 0].to(dtype=torch.float32) / 10.0
        logits = torch.stack((values, values / 2.0), dim=1)
        return SimpleNamespace(logits=logits)


def _validation_namespace(**overrides) -> argparse.Namespace:
    """Build the complete argument subset consumed by `_validate_args`."""

    values = {
        "precision": "checkpoint",
        "batch_size": None,
        "et_cache_size": None,
        "preflight_rows": 32,
        "preflight_check": True,
        "require_overlap_audit": True,
        "training_data_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _preflight_fixture(tmp_path: Path, *, persisted_ids: list[str]):
    """Create one held-out frame, member, and persisted prediction table."""

    fold_dir = tmp_path / "heldout_fold1"
    model_dir = fold_dir / "final_model"
    model_dir.mkdir(parents=True)
    persisted = pd.DataFrame(
        {
            "index": persisted_ids,
            "pred_valence": [0.10, 0.20, 0.30],
            "pred_arousal": [0.40, 0.50, 0.60],
        }
    )
    persisted.to_csv(fold_dir / "predictions.tsv", sep="\t", index=False)
    heldout = pd.DataFrame(
        {
            "index": ["a", "b", "c"],
            "text": ["1", "2", "3"],
            "dataset_of_origin": ["corpus"] * 3,
        }
    )
    member = SimpleNamespace(
        name="heldout_fold1",
        held_out_fold=1,
        training_fold=2,
        model_dir=model_dir,
        run_manifest={
            "evaluation_rows": 3,
            "excluded_dataset_names": [],
            "max_length": 200,
            "dtype": "bfloat16",
        },
    )
    return member, {1: heldout}


def test_parser_rejects_fp32_precision_override() -> None:
    """Keep precision sensitivity options out of the primary benchmark parser."""

    parser = evaluate_external._build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            [
                "omg-emotion",
                "--run-dir",
                "run",
                "--raw-dir",
                "raw",
                "--precision",
                "fp32",
            ]
        )
    assert error.value.code == 2


def test_parser_rejects_noncanonical_msp_label_column_override() -> None:
    """Keep the canonical MSP gold-label semantics out of user configuration."""

    parser = evaluate_external._build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            [
                "msp-podcast",
                "--run-dir",
                "run",
                "--split",
                "test2",
                "--labels-file",
                "labels.csv",
                "--transcripts",
                "transcripts.zip",
                "--valence-column",
                "EmoDom",
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "benchmark",
    ["idest-english", "semeval-2026-task2-subtask1"],
)
def test_parser_accepts_new_text_benchmarks_without_training_options(
    benchmark: str,
) -> None:
    """Expose only raw-data download and frozen-inference controls."""

    parser = evaluate_external._build_parser()
    args = parser.parse_args(
        [benchmark, "--run-dir", "run", "--raw-dir", "raw", "--download"]
    )
    assert args.benchmark == benchmark
    assert args.raw_dir == "raw"
    assert args.download is True
    for forbidden in (
        "epochs",
        "learning_rate",
        "finetuning_mode",
        "gradient_accumulation_steps",
        "resume_from_checkpoint",
    ):
        assert not hasattr(args, forbidden)


@pytest.mark.parametrize(
    "benchmark",
    ["idest-english", "semeval-2026-task2-subtask1"],
)
def test_parser_requires_raw_dir_for_new_text_benchmarks(benchmark: str) -> None:
    """Do not guess mutable benchmark-data locations."""

    parser = evaluate_external._build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args([benchmark, "--run-dir", "run"])
    assert error.value.code == 2


def test_new_text_benchmark_dispatch_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent either new command from falling through to the MSP loader."""

    calls: list[tuple[str, object]] = []
    idest_result = object()
    semeval_result = object()
    monkeypatch.setattr(
        evaluate_external,
        "download_pinned_idest_english",
        lambda raw_dir: calls.append(("download-idest", raw_dir)),
    )
    monkeypatch.setattr(
        evaluate_external,
        "load_idest_english",
        lambda raw_dir, strict_official_contract: (
            calls.append(("load-idest", strict_official_contract)) or idest_result
        ),
    )
    monkeypatch.setattr(
        evaluate_external,
        "download_pinned_semeval_subtask1_test",
        lambda raw_dir: calls.append(("download-semeval", raw_dir)),
    )
    monkeypatch.setattr(
        evaluate_external,
        "load_semeval_2026_subtask1_test",
        lambda raw_dir, strict_official_contract: (
            calls.append(("load-semeval", strict_official_contract)) or semeval_result
        ),
    )
    monkeypatch.setattr(
        evaluate_external,
        "load_msp_podcast_test",
        lambda *args, **kwargs: pytest.fail("new commands must never call the MSP loader"),
    )
    parser = evaluate_external._build_parser()

    idest_args = parser.parse_args(
        [
            "idest-english",
            "--run-dir",
            "run",
            "--raw-dir",
            "idest-raw",
            "--download",
        ]
    )
    assert evaluate_external._load_external_benchmark(idest_args) is idest_result
    assert calls == [
        ("download-idest", "idest-raw"),
        ("load-idest", True),
    ]

    calls.clear()
    semeval_args = parser.parse_args(
        [
            "semeval-2026-task2-subtask1",
            "--run-dir",
            "run",
            "--raw-dir",
            "semeval-raw",
            "--download",
        ]
    )
    assert evaluate_external._load_external_benchmark(semeval_args) is semeval_result
    assert calls == [
        ("download-semeval", "semeval-raw"),
        ("load-semeval", True),
    ]


def test_validate_args_rejects_programmatic_precision_override() -> None:
    """Reject callers that bypass argparse's immutable precision choices."""

    with pytest.raises(ValueError, match="--precision checkpoint"):
        evaluate_external._validate_args(
            _validation_namespace(precision="fp32")
        )


def test_bf16_dry_run_returns_before_device_or_dtype_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Allow provenance validation of a BF16 run on a CPU-only control host."""

    training_data_dir = tmp_path / "training_data"
    training_data_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    benchmark = SimpleNamespace(
        name="omg-emotion",
        split="test",
        frame=pd.DataFrame({"text": ["hello"], "is_empty_text": [False]}),
        context_policy="one current utterance transcript only",
    )
    member = SimpleNamespace(
        name="heldout_fold1",
        run_manifest={
            "data_dir": str(training_data_dir),
            "dtype": "bfloat16",
            "eval_batch_size": 8,
            "max_length": 200,
        },
    )
    monkeypatch.setattr(
        evaluate_external,
        "discover_completed_run",
        lambda path, **kwargs: [member],
    )
    monkeypatch.setattr(
        evaluate_external,
        "load_omg_emotion_test",
        lambda raw_dir, strict_official_contract: benchmark,
    )
    monkeypatch.setattr(
        evaluate_external,
        "reject_benchmark_training_sources",
        lambda members, benchmark_name: None,
    )
    monkeypatch.setattr(
        evaluate_external,
        "audit_finetuning_text_overlap",
        lambda benchmark_data, members, data_dir: (
            benchmark.frame.copy(),
            {"status": "passed"},
            {1: pd.DataFrame()},
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not inspect device capabilities or load a model")

    monkeypatch.setattr(evaluate_external, "_resolve_device", forbidden)
    monkeypatch.setattr(evaluate_external, "_resolve_dtype", forbidden)
    monkeypatch.setattr(evaluate_external, "load_saved_decoder_va_model", forbidden)
    args = evaluate_external._build_parser().parse_args(
        [
            "omg-emotion",
            "--run-dir",
            str(run_dir),
            "--training-data-dir",
            str(training_data_dir),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--dry-run",
        ]
    )

    assert evaluate_external.run(args) is None
    output = capsys.readouterr().out
    assert "recorded precision=bfloat16" in output
    assert "Dry run complete" in output


def test_predict_texts_preserves_order_and_is_strictly_label_free() -> None:
    """Freeze the model and run ordered forwards only inside inference mode."""

    model = RecordingModel()
    tokenizer = OrderedTokenizer()
    texts = ["3", "0", "2", "1"]

    predictions = evaluate_external._predict_texts(
        model,
        tokenizer,
        texts,
        max_length=12,
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        autocast_enabled=False,
    )

    np.testing.assert_allclose(
        predictions,
        np.array(
            [
                [0.4, 0.2],
                [0.1, 0.05],
                [0.3, 0.15],
                [0.2, 0.1],
            ]
        ),
        rtol=0.0,
        atol=1e-7,
    )
    assert [item for call in model.calls for item in call["input_ids"]] == [4, 1, 3, 2]
    assert all(call["keys"] == ("attention_mask", "input_ids") for call in model.calls)
    assert all(call["grad_enabled"] is False for call in model.calls)
    assert all(call["inference_mode"] is True for call in model.calls)
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())


def test_tokenizer_truncation_audit_is_label_free_and_keeps_saved_limit() -> None:
    """Audit full token lengths without truncating or selecting a new max_length."""

    tokenizer = LengthAuditTokenizer()
    summary, counts, truncated = evaluate_external._tokenizer_truncation_audit(
        tokenizer,
        ["0", "3", "5", "9"],
        max_length=5,
    )

    np.testing.assert_array_equal(counts, np.array([0, 3, 5, 9]))
    np.testing.assert_array_equal(truncated, np.array([False, False, False, True]))
    assert summary["checkpoint_max_length"] == 5
    assert summary["truncated_rows"] == 1
    assert summary["truncated_fraction"] == pytest.approx(0.25)
    assert summary["token_count_max"] == 9
    assert all(call["max_length"] == 5 for call in tokenizer.calls)
    assert all(call["truncation"] is False for call in tokenizer.calls)
    assert all(call["padding"] is False for call in tokenizer.calls)


def test_preflight_rejects_persisted_id_order_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a persisted table whose identifiers do not match held-out order."""

    member, fold_frames = _preflight_fixture(
        tmp_path,
        persisted_ids=["b", "a", "c"],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("ID order must be checked before re-prediction")

    monkeypatch.setattr(evaluate_external, "_predict_texts", forbidden)
    with pytest.raises(ValueError, match="prediction order"):
        evaluate_external._preflight_member(
            member,
            model=object(),
            tokenizer=object(),
            fold_frames=fold_frames,
            rows=2,
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            precision_name="float32",
            autocast_enabled=False,
        )


def test_preflight_uses_recorded_dtype_tolerance_and_rejects_excess_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the looser recorded BF16 tolerance, while still failing above it."""

    member, fold_frames = _preflight_fixture(
        tmp_path,
        persisted_ids=["a", "b", "c"],
    )
    reference = np.array(
        [[0.10, 0.40], [0.20, 0.50], [0.30, 0.60]],
        dtype=np.float64,
    )
    monkeypatch.setattr(
        evaluate_external,
        "_predict_texts",
        lambda model, tokenizer, texts, **kwargs: reference[: len(texts)] + 0.004,
    )

    report = evaluate_external._preflight_member(
        member,
        model=object(),
        tokenizer=object(),
        fold_frames=fold_frames,
        rows=2,
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        precision_name="float32",
        autocast_enabled=False,
    )
    assert report["rows"] == 2
    assert report["recorded_prediction_dtype"] == "bfloat16"
    assert report["reload_inference_dtype"] == "float32"
    assert report["absolute_tolerance"] == pytest.approx(5e-3)
    assert report["max_absolute_prediction_error"] == pytest.approx(4e-3)

    monkeypatch.setattr(
        evaluate_external,
        "_predict_texts",
        lambda model, tokenizer, texts, **kwargs: reference[: len(texts)] + 0.006,
    )
    with pytest.raises(RuntimeError, match="exceeds 0.005"):
        evaluate_external._preflight_member(
            member,
            model=object(),
            tokenizer=object(),
            fold_frames=fold_frames,
            rows=2,
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            precision_name="float32",
            autocast_enabled=False,
        )


def test_full_cpu_run_predicts_both_members_and_atomically_writes_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise CLI orchestration from bundle discovery through final result publish."""

    run_dir = tmp_path / "portable-bundle"
    run_dir.mkdir()
    output_dir = tmp_path / "external-output"
    frame = pd.DataFrame(
        {
            "index": ["v::u0", "v::u1", "w::u0", "w::u1"],
            "benchmark_id": ["v::u0", "v::u1", "w::u0", "w::u1"],
            "text": ["0", "1", "2", "3"],
            "text_sha256": ["a", "b", "c", "d"],
            "is_empty_text": [False] * 4,
            "dataset_of_origin": ["OMG-Emotion"] * 4,
            "split": ["test"] * 4,
            "video": ["v", "v", "w", "w"],
            "utterance": ["u0", "u1", "u0", "u1"],
            "valence": [0.1, 0.2, 0.3, 0.4],
            "arousal": [0.05, 0.10, 0.15, 0.20],
            "native_valence": [-0.8, -0.6, -0.4, -0.2],
            "native_arousal": [0.05, 0.10, 0.15, 0.20],
        }
    )
    benchmark = ExternalBenchmarkData(
        name="omg-emotion",
        version="fixture",
        split="test",
        frame=frame,
        native_scale={"valence": (-1.0, 1.0), "arousal": (0.0, 1.0)},
        model_to_native_scale=(2.0, 1.0),
        model_to_native_offset=(-1.0, 0.0),
        source_manifest={"canonical_contract_verified": True},
        join_report={"matched_gold_rows": 4},
        official_group_column="video",
    )
    common_manifest = {
        "data_dir": None,
        "dtype": "float32",
        "eval_batch_size": 2,
        "max_length": 12,
        "dataset_counts_after_filter": {"Emobank": 4},
        "finetuning_mode": "full",
        "gaze_fusion": "none",
        "gaze_features": [],
        "et_revision": "fixture",
    }
    members = tuple(
        SimpleNamespace(
            name=f"heldout_fold{fold}",
            held_out_fold=fold,
            training_fold=2 if fold == 1 else 1,
            model_dir=run_dir / f"heldout_fold{fold}" / "final_model",
            run_manifest={**common_manifest, "held_out_fold": fold},
            file_sha256={"model_weights": str(fold) * 64},
        )
        for fold in (1, 2)
    )

    def discover(path, *, require_internal_evidence):
        assert Path(path) == run_dir.resolve()
        assert require_internal_evidence is False
        return members

    loaded_models = []

    def load_model(path, *, dtype, et_cache_size):
        assert dtype == torch.float32
        assert et_cache_size is None
        model = RecordingModel()
        loaded_models.append((Path(path), model))
        return model, OrderedTokenizer()

    monkeypatch.setattr(evaluate_external, "discover_completed_run", discover)
    monkeypatch.setattr(
        evaluate_external,
        "load_omg_emotion_test",
        lambda raw_dir, strict_official_contract: benchmark,
    )
    monkeypatch.setattr(evaluate_external, "load_saved_decoder_va_model", load_model)
    args = evaluate_external._build_parser().parse_args(
        [
            "omg-emotion",
            "--run-dir",
            str(run_dir),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--no-require-overlap-audit",
            "--no-preflight-check",
        ]
    )

    written = evaluate_external.run(args)

    assert written == output_dir.resolve()
    assert len(loaded_models) == 2
    assert (output_dir / "COMPLETED").is_file()
    predictions = pd.read_csv(output_dir / "predictions.tsv", sep="\t")
    assert "text" not in predictions
    manifest = pd.read_json(
        output_dir / "external_evaluation_manifest.json",
        typ="series",
    )
    assert manifest["training_performed"] is False
    assert manifest["preflight_required"] is False
    assert "raw-text-free model bundle" in manifest["internal_evidence_validation"]


def test_full_semeval_run_audits_truncation_and_reports_official_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Integrate explicit SemEval dispatch, fold-1 audit, scoring, and manifest output."""

    run_dir = tmp_path / "portable-bundle"
    run_dir.mkdir()
    output_dir = tmp_path / "semeval-output"
    frame = pd.DataFrame(
        {
            "index": ["a::0", "a::1", "b::0", "b::1"],
            "benchmark_id": ["a::0", "a::1", "b::0", "b::1"],
            "text": ["0", "1", "2", "3"],
            "text_sha256": ["a", "b", "c", "d"],
            "is_empty_text": [False] * 4,
            "dataset_of_origin": ["SemEval-2026 Task 2 Subtask 1"] * 4,
            "split": ["official-test-zero-shot"] * 4,
            "user_id": ["a", "a", "b", "b"],
            "text_id": ["0", "1", "0", "1"],
            "timestamp": ["2026-01-01"] * 4,
            "collection_phase": [1, 1, 2, 2],
            "is_words": [False, True, False, True],
            "is_seen_user": [True, True, False, False],
            "valence": [0.0, 0.25, 0.5, 1.0],
            "arousal": [0.0, 0.5, 0.0, 1.0],
            "native_valence": [-2.0, -1.0, 0.0, 2.0],
            "native_arousal": [0.0, 1.0, 0.0, 2.0],
        }
    )
    context_policy = (
        "one current test text only; user ID, timestamp, history, is_seen_user, "
        "and gold labels are excluded from model inputs"
    )
    benchmark = ExternalBenchmarkData(
        name="semeval-2026-task2-subtask1",
        version="fixture",
        split="official-test-zero-shot",
        frame=frame,
        native_scale={"valence": (-2.0, 2.0), "arousal": (0.0, 2.0)},
        model_to_native_scale=(4.0, 2.0),
        model_to_native_offset=(-2.0, 0.0),
        source_manifest={"canonical_contract_verified": True},
        join_report={"matched_rows": 4},
        official_group_column="user_id",
        context_policy=context_policy,
        prediction_metadata_columns=(
            "user_id",
            "text_id",
            "timestamp",
            "collection_phase",
            "is_words",
            "is_seen_user",
        ),
    )
    common_manifest = {
        "data_dir": None,
        "dtype": "float32",
        "eval_batch_size": 2,
        "max_length": 12,
        "dataset_counts_after_filter": {"Emobank": 4},
        "finetuning_mode": "full",
        "gaze_fusion": "none",
        "gaze_features": [],
        "et_revision": "fixture",
    }
    members = tuple(
        SimpleNamespace(
            name=f"heldout_fold{fold}",
            held_out_fold=fold,
            training_fold=2 if fold == 1 else 1,
            model_dir=run_dir / f"heldout_fold{fold}" / "final_model",
            run_manifest={**common_manifest, "held_out_fold": fold},
            file_sha256={"model_weights": str(fold) * 64},
        )
        for fold in (1, 2)
    )
    loaded_tokenizers: list[OrderedTokenizer] = []

    def load_model(path, *, dtype, et_cache_size):
        del path, dtype, et_cache_size
        tokenizer = OrderedTokenizer()
        loaded_tokenizers.append(tokenizer)
        return RecordingModel(), tokenizer

    monkeypatch.setattr(
        evaluate_external,
        "discover_completed_run",
        lambda path, require_internal_evidence: members,
    )
    monkeypatch.setattr(
        evaluate_external,
        "load_semeval_2026_subtask1_test",
        lambda raw_dir, strict_official_contract: benchmark,
    )
    monkeypatch.setattr(evaluate_external, "load_saved_decoder_va_model", load_model)
    args = evaluate_external._build_parser().parse_args(
        [
            "semeval-2026-task2-subtask1",
            "--run-dir",
            str(run_dir),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--no-require-overlap-audit",
            "--no-preflight-check",
        ]
    )

    written = evaluate_external.run(args)

    assert written == output_dir.resolve()
    assert len(loaded_tokenizers) == 2
    manifest = pd.read_json(
        output_dir / "external_evaluation_manifest.json",
        typ="series",
    )
    audit = manifest["tokenization_truncation_audit"]
    assert audit["status"] == "completed"
    assert audit["tokenizer_member"] == "heldout_fold1"
    assert audit["checkpoint_max_length"] == 12
    assert audit["truncated_rows"] == 0
    assert manifest["input_context_policy"] == context_policy
    predictions = pd.read_csv(output_dir / "predictions.tsv", sep="\t")
    assert predictions["token_count_before_truncation"].tolist() == [1, 1, 1, 1]
    assert not predictions["was_truncated_at_checkpoint_max_length"].any()
    assert "Official ensemble r_composite" in capsys.readouterr().out

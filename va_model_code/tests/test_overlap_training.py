from __future__ import annotations

import csv
from importlib import import_module
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from transformers import TrainerState

from va_model_code.decoder_va.filters import TRAIN_EVAL_TEXT_OVERLAP_CONTRACT


VA_MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(VA_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(VA_MODEL_ROOT))

train_model_module = import_module("train_model")


def _write_fold(data_root: Path, fold: int, texts: list[str]) -> pd.DataFrame:
    """Write a real fold file retaining source text and distinct example IDs."""

    frame = pd.DataFrame(
        {
            "index": [fold * 100 + offset for offset in range(len(texts))],
            "text": texts,
            "dataset_of_origin": ["Emobank"] * len(texts),
            "valence": [0.2 + 0.1 * offset for offset in range(len(texts))],
            "arousal": [0.8 - 0.1 * offset for offset in range(len(texts))],
        }
    )
    frame.to_csv(
        data_root / f"full_dataset_fold{fold}.csv",
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
    return frame


@pytest.fixture
def offline_overlap_run(tmp_path, monkeypatch):
    """Keep real fold loading, dataset construction, arguments, and OOF reports."""

    data_root = tmp_path / "data"
    data_root.mkdir()
    run_root = tmp_path / "overlap_run"
    observed = {"trainers": [], "builds": [], "predictions": []}

    class OfflineModel:
        def trainable_parameter_summary(self):
            return {"total_parameters": 8, "trainable_parameters": 8, "trainable_fraction": 1.0}

        def save_architecture_manifest(self, directory):
            Path(directory, "decoder_va_architecture.json").write_text("{}", encoding="utf-8")

    class RecordingTrainer:
        def __init__(self, **kwargs):
            self.state = TrainerState()
            self.train_dataset = kwargs["train_dataset"]
            self.eval_dataset = kwargs["eval_dataset"]
            self.args = kwargs["args"]
            self.loss_name = kwargs["loss_name"]
            observed["trainers"].append(self)

        def train(self, resume_from_checkpoint):
            assert self.args.load_best_model_at_end is True
            assert self.args.metric_for_best_model == "mse_mean"
            assert self.args.greater_is_better is False

        def save_model(self, directory):
            Path(directory).mkdir(parents=True, exist_ok=True)

        def predict(self, dataset):
            assert dataset is self.eval_dataset
            observed["predictions"].append(dataset)
            labels = np.column_stack((dataset.valence, dataset.arousal))
            predictions = (
                np.column_stack((labels, np.full_like(labels, np.log(0.01))))
                if self.loss_name == "heteroscedastic" else labels
            )
            return SimpleNamespace(label_ids=labels, predictions=predictions, metrics={})

    def build_model(tokenizer, **kwargs):
        observed["builds"].append(kwargs)
        return OfflineModel()

    monkeypatch.setattr(train_model_module, "_resolve_output_dir", lambda args: run_root)
    monkeypatch.setattr(
        train_model_module,
        "load_auto_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(pad_token_id=0),
    )
    monkeypatch.setattr(train_model_module, "build_qwen_va_model", build_model)
    monkeypatch.setattr(train_model_module, "VARegressionTrainer", RecordingTrainer)

    def arguments(*extra):
        return train_model_module._build_parser().parse_args(
            [
                "qwen3.5-0.8b", "mse", "--data-dir", str(data_root),
                "--finetuning-mode", "full", "--precision", "fp32", "--use-cpu",
                "--gaze-fusion", "none", "--run-name", "overlap_run", *extra,
            ]
        )

    return SimpleNamespace(
        data_root=data_root,
        run_root=run_root,
        observed=observed,
        arguments=arguments,
    )


def test_run_excludes_overlap_from_training_and_keeps_evaluation_reuse(offline_overlap_run):
    run = offline_overlap_run
    originals = {
        1: _write_fold(run.data_root, 1, ["Cafe\u0301  HELLO", "one unique", "another unique"]),
        2: _write_fold(run.data_root, 2, ["Ｃａｆé hello", "café  hello", "two unique", "two another"]),
    }

    assert train_model_module.run(run.arguments()) == run.run_root

    expected_training_indices = {1: [202, 203], 2: [101, 102]}
    expected_excluded = {1: 2, 2: 1}
    for fold, trainer in enumerate(run.observed["trainers"], start=1):
        assert trainer.train_dataset.index == expected_training_indices[fold]
        assert trainer.eval_dataset.index == originals[fold]["index"].tolist()
        assert trainer.eval_dataset.texts == originals[fold]["text"].tolist()
        manifest = json.loads((run.run_root / f"heldout_fold{fold}" / "run_manifest.json").read_text())
        assert manifest["training_rows"] == 2
        assert manifest["evaluation_rows"] == len(originals[fold])
        assert manifest["training_text_overlap"] == {
            "contract": TRAIN_EVAL_TEXT_OVERLAP_CONTRACT,
            "train_rows_before": len(originals[3 - fold]),
            "train_rows_after": 2,
            "excluded_train_rows": expected_excluded[fold],
            "eval_rows": len(originals[fold]),
            "overlapping_eval_rows": expected_excluded[3 - fold],
            "shared_normalized_texts": 1,
        }
        assert manifest["training_text_overlap_contract"] == manifest["training_text_overlap"]["contract"]
        predictions = pd.read_csv(run.run_root / f"heldout_fold{fold}" / "predictions.tsv", sep="\t")
        assert predictions["index"].tolist() == originals[fold]["index"].tolist()
        assert predictions["text"].tolist() == originals[fold]["text"].tolist()

    assert len(run.observed["predictions"]) == 2
    root_manifest = json.loads((run.run_root / "training_parameters.json").read_text())
    assert root_manifest["training_text_overlap_contract"] == TRAIN_EVAL_TEXT_OVERLAP_CONTRACT
    assert root_manifest["dataset_counts_after_filter"] == {"Emobank": 7}
    oof = pd.read_csv(run.run_root / "oof_predictions.tsv", sep="\t")
    assert len(oof) == 7
    assert sorted(oof["index"]) == [100, 101, 102, 200, 201, 202, 203]
    for fold in (1, 2):
        saved_input = train_model_module.load_filtered_folds(run.data_root).folds[f"full_dataset_fold{fold}.csv"]
        pd.testing.assert_frame_equal(saved_input, originals[fold])


def test_run_rejects_all_overlap_before_model_or_trainer(offline_overlap_run):
    run = offline_overlap_run
    _write_fold(run.data_root, 1, ["SAME", "HELLO"])
    _write_fold(run.data_root, 2, [" same ", "Ｈｅｌｌｏ"])

    with pytest.raises(ValueError, match="Held-out fold 1: no training rows remain"):
        train_model_module.run(run.arguments())

    assert not run.observed["builds"]
    assert not run.observed["trainers"]
    assert not (run.run_root / "heldout_fold1").exists()


def test_run_keep_overlap_delivers_all_rows_to_trainer_and_records_policy(offline_overlap_run):
    """Verify the explicit historical protocol reaches training, evaluation, and saved reports."""

    run = offline_overlap_run
    originals = {
        1: _write_fold(run.data_root, 1, ["Cafe\u0301  HELLO", "one unique", "another unique"]),
        2: _write_fold(run.data_root, 2, ["Ｃａｆé hello", "café  hello", "two unique", "two another"]),
    }
    keep_contract = {
        "version": 1,
        "normalization": "NFKC+casefold+whitespace-collapse",
        "policy": "keep_all_training_rows",
        "evaluation_rows": "unchanged",
    }

    train_model_module.run(run.arguments("--training-text-overlap", "keep"))

    for fold, trainer in enumerate(run.observed["trainers"], start=1):
        original_training = originals[3 - fold]
        assert trainer.train_dataset.index == original_training["index"].tolist()
        assert trainer.train_dataset.texts == original_training["text"].tolist()
        np.testing.assert_allclose(trainer.train_dataset.valence, original_training["valence"])
        np.testing.assert_allclose(trainer.train_dataset.arousal, original_training["arousal"])
        assert trainer.eval_dataset.index == originals[fold]["index"].tolist()
        manifest = json.loads((run.run_root / f"heldout_fold{fold}" / "run_manifest.json").read_text())
        assert manifest["training_rows"] == len(original_training)
        assert manifest["evaluation_rows"] == len(originals[fold])
        assert manifest["training_text_overlap_contract"] == keep_contract
        assert manifest["training_text_overlap"] == {
            "contract": keep_contract,
            "train_rows_before": len(original_training),
            "train_rows_after": len(original_training),
            "excluded_train_rows": 0,
            "eval_rows": len(originals[fold]),
            "overlapping_eval_rows": fold,
            "shared_normalized_texts": 1,
        }
        predictions = pd.read_csv(run.run_root / f"heldout_fold{fold}" / "predictions.tsv", sep="\t")
        assert predictions["index"].tolist() == originals[fold]["index"].tolist()
        assert predictions["text"].tolist() == originals[fold]["text"].tolist()

    root_manifest = json.loads((run.run_root / "training_parameters.json").read_text())
    assert root_manifest["training_text_overlap_contract"] == keep_contract
    assert root_manifest["training_text_overlap_policy"] == "keep"
    assert root_manifest["dataset_counts_after_filter"] == {"Emobank": 7}
    assert len(pd.read_csv(run.run_root / "oof_predictions.tsv", sep="\t")) == 7


def test_run_keep_all_overlap_is_usable_training_data(offline_overlap_run):
    """The opt-in policy must not apply the exclusion mode's empty-training preflight."""

    run = offline_overlap_run
    _write_fold(run.data_root, 1, ["SAME", "HELLO"])
    _write_fold(run.data_root, 2, [" same ", "Ｈｅｌｌｏ"])

    train_model_module.run(run.arguments("--training-text-overlap", "keep"))

    assert len(run.observed["trainers"]) == 2
    assert all(len(trainer.train_dataset) == 2 for trainer in run.observed["trainers"])
    assert len(run.observed["predictions"]) == 2


@pytest.mark.parametrize("original_policy,new_policy", [("exclude", "keep"), ("keep", "exclude")])
@pytest.mark.parametrize("shared_text", [True, False])
def test_run_resume_rejects_overlap_policy_switch_even_when_row_counts_match(
    offline_overlap_run, original_policy, new_policy, shared_text
):
    """A checkpoint's data policy cannot change, including on disjoint data with equal counts."""

    run = offline_overlap_run
    _write_fold(run.data_root, 1, ["Shared" if shared_text else "first", "only one"])
    _write_fold(run.data_root, 2, ["shared" if shared_text else "second", "only two"])
    train_model_module.run(
        run.arguments("--held-out-folds", "1", "--training-text-overlap", original_policy)
    )
    checkpoint = run.run_root / "heldout_fold1" / "checkpoints" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    run.observed["builds"].clear()
    run.observed["trainers"].clear()

    with pytest.raises(ValueError, match="training_text_overlap"):
        train_model_module.run(
            run.arguments(
                "--held-out-folds", "1", "--resume-from-checkpoint", str(checkpoint),
                "--training-text-overlap", new_policy,
            )
        )

    assert not run.observed["builds"]
    assert not run.observed["trainers"]


def test_resume_refuses_changed_overlap_audit_before_model_or_trainer(offline_overlap_run):
    run = offline_overlap_run
    _write_fold(run.data_root, 1, ["Shared", "only one"])
    _write_fold(run.data_root, 2, ["shared", "only two"])
    train_model_module.run(run.arguments("--held-out-folds", "1"))
    fold_root = run.run_root / "heldout_fold1"
    manifest_path = fold_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_text_overlap"]["excluded_train_rows"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = fold_root / "checkpoints" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    run.observed["builds"].clear()
    run.observed["trainers"].clear()

    with pytest.raises(ValueError, match="training_text_overlap"):
        train_model_module.run(
            run.arguments("--held-out-folds", "1", "--resume-from-checkpoint", str(checkpoint))
        )

    assert not run.observed["builds"]
    assert not run.observed["trainers"]


@pytest.mark.parametrize("dry_run", (False, True))
def test_overlap_preflight_fails_before_tokenizer_download(offline_overlap_run, monkeypatch, dry_run):
    """Validate actual post-exclusion training rows even in a download-free dry run."""

    run = offline_overlap_run
    _write_fold(run.data_root, 1, ["Shared"])
    _write_fold(run.data_root, 2, [" shared "])

    def unexpected_download(*args, **kwargs):
        """Fail if a model-independent validation accidentally requires downloads."""

        raise AssertionError("Tokenizer should not load for an unusable split")

    monkeypatch.setattr(train_model_module, "load_auto_tokenizer", unexpected_download)
    args = run.arguments(*(["--dry-run"] if dry_run else []))
    with pytest.raises(ValueError, match="no training rows remain"):
        train_model_module.run(args)

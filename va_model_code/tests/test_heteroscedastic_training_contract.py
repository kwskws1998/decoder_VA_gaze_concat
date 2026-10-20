"""Protect the two requested training conditions and incompatible resume boundaries."""

import copy
import json

import pandas as pd
import pytest

from va_model_code.decoder_va.paths import condition_slug
from va_model_code.decoder_va.uncertainty import prediction_contract
from va_model_code.tests.test_overlap_training import offline_overlap_run, _write_fold
from va_model_code.tests.test_train_model_contract import (
    train_model_module, _expected_resume_manifest, _check_resume_pair,
)


@pytest.mark.parametrize("fusion", ["none", "prefix-concat"])
def test_main_uses_original_readout_and_retained_data_with_gaussian_outputs(offline_overlap_run, fusion):
    """Check CLI, model construction, best-MSE selection, fold metadata and final OOF columns."""

    run = offline_overlap_run
    for fold in (1, 2):
        frame = _write_fold(run.data_root, fold, ["same", "other", "excluded"])
        frame.loc[2, "dataset_of_origin"] = "IEMOCAP sentences"
        frame.to_csv(run.data_root / f"full_dataset_fold{fold}.csv", sep="\t", index=False)
    args = run.arguments(
        "--gaze-fusion", fusion, "--gaze-features", "TRT", "--no-iemocap",
        "--training-text-overlap", "keep", "--readout", "last-token",
        "--gaze-relation", "none", "--gaze-redistribution", "none",
        "--train-batch-size", "8", "--gradient-accumulation-steps", "2",
    )
    args.loss = "heteroscedastic"
    assert train_model_module.run(args) == run.run_root
    for build in run.observed["builds"]:
        assert build["prediction_distribution"] == prediction_contract("heteroscedastic")
        assert build["readout"]["mode"] == "last-token"
        assert build["gaze_relation"]["method"] == "none"
        assert build["gaze_redistribution"]["method"] == "none"
    for trainer in run.observed["trainers"]:
        assert trainer.loss_name == "heteroscedastic"
        assert trainer.args.metric_for_best_model == "mse_mean" and trainer.args.greater_is_better is False
        assert trainer.args.load_best_model_at_end
        assert trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps == 16
        assert len(trainer.train_dataset) == len(trainer.eval_dataset) == 2
    for relative in ("training_parameters.json", "heldout_fold1/run_manifest.json", "heldout_fold2/run_manifest.json"):
        metadata = json.loads((run.run_root / relative).read_text())
        assert metadata["loss"] == "heteroscedastic"
        assert metadata["prediction_distribution"] == prediction_contract("heteroscedastic")
        assert metadata["training_text_overlap_policy"] == "keep"
        assert metadata["no_iemocap"] and not metadata["sentence_only"]
    frame = pd.read_csv(run.run_root / "oof_predictions.tsv", sep="\t")
    assert "pred_std_valence" in frame and len(frame) == 4
    assert "gaussian_nll_mean" in json.loads((run.run_root / "oof_metrics.json").read_text())


@pytest.mark.parametrize("fusion", ["none", "prefix-concat"])
def test_cli_and_default_names_keep_loss_conditions_distinct(fusion):
    args = train_model_module._build_parser().parse_args(["qwen", "heteroscedastic", "--gaze-fusion", fusion])
    train_model_module._validate_args(args)
    assert args.loss == "heteroscedastic" and args.readout == "last-token"
    shared = dict(model="qwen", finetuning_mode="full", gaze_fusion=fusion, gaze_features=["TRT"] if fusion != "none" else [], seed=42)
    point = condition_slug(**shared)
    gaussian = condition_slug(**shared, prediction_distribution=prediction_contract("heteroscedastic"))
    assert point != gaussian and "heteroscedastic" in gaussian


def test_resume_requires_matching_loss_and_distribution(tmp_path):
    expected = _expected_resume_manifest()
    expected.update(loss="heteroscedastic", prediction_distribution=prediction_contract("heteroscedastic"))
    _check_resume_pair(tmp_path / "valid", copy.deepcopy(expected), expected)
    point = copy.deepcopy(expected)
    point.update(loss="mse", prediction_distribution=prediction_contract())
    with pytest.raises(ValueError, match="loss|prediction_distribution"):
        _check_resume_pair(tmp_path / "point", point, expected)
    missing = copy.deepcopy(expected)
    missing.pop("prediction_distribution")
    with pytest.raises(ValueError, match="prediction_distribution"):
        _check_resume_pair(tmp_path / "missing", missing, expected)


def test_schema10_unchanged_point_training_can_resume(tmp_path):
    expected = _expected_resume_manifest()
    recorded = copy.deepcopy(expected)
    recorded["architecture_manifest_version"] = 10
    recorded.pop("prediction_distribution")
    _check_resume_pair(tmp_path, recorded, expected)

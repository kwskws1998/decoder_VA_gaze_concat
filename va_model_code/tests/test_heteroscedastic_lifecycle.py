"""Check Gaussian VA training, checkpoint recovery and completed-result publication end to end."""

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import TrainerCallback, TrainingArguments, default_data_collator

from va_model_code.decoder_va import model as model_module
from va_model_code.decoder_va.evaluation import (
    metrics_from_prediction_frame, trainer_compute_metrics, write_oof_reports,
)
from va_model_code.decoder_va.model import build_qwen_va_model, load_saved_decoder_va_model
from va_model_code.decoder_va.trainer import VARegressionTrainer
from va_model_code.decoder_va.uncertainty import prediction_contract, validate_saved_variance_head
from va_model_code.package_results import package_results
from va_model_code.tests.test_gaze_attention import tiny_backbone
from va_model_code.tests.test_gaze_relation_trainer_integration import ordered_rows
from va_model_code.tests.test_readout_model import SingletonGazeProvider
from va_model_code.tests.test_result_layout import _write_completed_run


@pytest.fixture
def native_builder(monkeypatch):
    """Use the real hybrid Qwen layers without network downloads or a full-size decoder."""

    monkeypatch.setattr(model_module, "load_qwen_backbone", lambda *args, **kwargs: tiny_backbone())
    monkeypatch.setattr(model_module, "ET2GazeProvider", SingletonGazeProvider)

    def build(fusion, seed=811, loss="heteroscedastic"):
        torch.manual_seed(seed)
        return build_qwen_va_model(
            SimpleNamespace(), dtype="float32", finetuning_mode="full", gaze_fusion=fusion,
            prediction_distribution=prediction_contract(loss), gaze_feature_indices=(3,),
            gaze_projection_dim=8, gaze_projection_dropout=(0., 0.), classifier_dropout=0.,
        )

    return build


class VarianceResumeObserver(TrainerCallback):
    """Capture restored variance parameters and Adam state before the first resumed step."""

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.step = state.global_step
        self.values = {name: parameter.detach().clone() for name, parameter in model.variance_head.named_parameters()}
        self.moments = {name: optimizer.state[parameter]["exp_avg"].clone() for name, parameter in model.variance_head.named_parameters()}
        self.steps = [int(optimizer.state[parameter]["step"]) for parameter in model.variance_head.parameters()]


def trainer_for(model, directory, steps, callbacks=None):
    """Exercise accumulated gradient updates, checkpoint replay, evaluation and saving."""

    return VARegressionTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(directory), max_steps=steps, per_device_train_batch_size=2,
            per_device_eval_batch_size=2, gradient_accumulation_steps=2,
            learning_rate=5e-4, lr_scheduler_type="constant", weight_decay=0.01,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="steps", save_steps=1, eval_strategy="steps", eval_steps=1,
            logging_strategy="steps", logging_steps=1, report_to="none", use_cpu=True,
            disable_tqdm=True, dataloader_pin_memory=False, remove_unused_columns=False,
            label_names=["labels"], seed=42,
        ),
        train_dataset=ordered_rows(), eval_dataset=ordered_rows(),
        data_collator=default_data_collator, loss_name="heteroscedastic",
        compute_metrics=trainer_compute_metrics, callbacks=callbacks,
    )


@pytest.mark.parametrize("fusion", ["none", "prefix-concat"])
def test_native_training_save_reload_resume_and_uncertainty_metrics(tmp_path, native_builder, fusion):
    model = native_builder(fusion)
    initial_bias = model.variance_head.bias.detach().clone()
    trainer = trainer_for(model, tmp_path, 2)
    trained = trainer.train()
    assert trainer.state.global_step == 2 and np.isfinite(trained.training_loss)
    assert model.backbone.is_gradient_checkpointing
    assert not torch.equal(initial_bias, model.variance_head.bias)
    for parameter in model.variance_head.parameters():
        assert sum(parameter is item for group in trainer.optimizer.param_groups for item in group["params"]) == 1
        assert int(trainer.optimizer.state[parameter]["step"]) == 2
        assert torch.count_nonzero(trainer.optimizer.state[parameter]["exp_avg"]) > 0
    result = trainer.predict(ordered_rows())
    assert result.predictions.shape == (4, 4)
    np.testing.assert_array_equal(result.label_ids, torch.stack([row["labels"] for row in ordered_rows()]).numpy())
    assert result.metrics["test_loss"] == pytest.approx(result.metrics["test_gaussian_nll_mean"], abs=1e-5)
    assert "eval_mse_mean" in trainer.state.log_history[1]
    checkpoint = tmp_path / "checkpoint-2"
    model.save_architecture_manifest(checkpoint)
    validate_saved_variance_head(model, checkpoint / "model.safetensors")
    saved_moments = {name: trainer.optimizer.state[p]["exp_avg"].clone() for name, p in model.variance_head.named_parameters()}
    saved_head = {name: p.detach().clone() for name, p in model.variance_head.named_parameters()}

    restored, _ = load_saved_decoder_va_model(checkpoint, tokenizer=SimpleNamespace())
    with torch.no_grad():
        batch = default_data_collator(ordered_rows())
        predictions = restored(**batch).logits.numpy()
    np.testing.assert_allclose(predictions, result.predictions, atol=1e-6, rtol=1e-6)
    fresh = native_builder(fusion, seed=912)
    observer = VarianceResumeObserver()
    resumed = trainer_for(fresh, tmp_path, 3, [observer])
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert observer.step == 2 and observer.steps == [2, 2]
    for name, p in fresh.variance_head.named_parameters():
        torch.testing.assert_close(observer.values[name], saved_head[name], atol=0, rtol=0)
        torch.testing.assert_close(observer.moments[name], saved_moments[name], atol=0, rtol=0)
        assert not torch.equal(p, saved_head[name])
    assert resumed.state.global_step == 3


@pytest.mark.parametrize("corruption", ["missing", "nonfinite", "shape", "dtype", "metadata"])
def test_saved_model_rejects_invalid_variance_state(tmp_path, native_builder, corruption):
    model = native_builder("prefix-concat")
    model.save_architecture_manifest(tmp_path)
    weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if corruption == "missing":
        weights.pop("variance_head.weight")
    elif corruption == "nonfinite":
        weights["variance_head.bias"][0] = float("nan")
    elif corruption == "shape":
        weights["variance_head.bias"] = torch.ones(3)
    elif corruption == "dtype":
        weights["variance_head.bias"] = weights["variance_head.bias"].bfloat16()
    else:
        path = tmp_path / "decoder_va_architecture.json"
        metadata = json.loads(path.read_text())
        metadata["reconstruction"]["prediction_distribution"] = prediction_contract()
        path.write_text(json.dumps(metadata))
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(ValueError):
        load_saved_decoder_va_model(tmp_path, tokenizer=SimpleNamespace())


@pytest.mark.parametrize("fusion", ["none", "prefix-concat"])
def test_schema10_point_model_remains_loadable(tmp_path, native_builder, fusion):
    model = native_builder(fusion, loss="mse").eval()
    model.save_architecture_manifest(tmp_path)
    path = tmp_path / "decoder_va_architecture.json"
    metadata = json.loads(path.read_text())
    metadata["schema_version"] = 10
    metadata.pop("prediction_distribution")
    metadata["reconstruction"].pop("prediction_distribution")
    path.write_text(json.dumps(metadata))
    save_file({name: value.detach().clone() for name, value in model.state_dict().items()}, tmp_path / "model.safetensors")
    restored, _ = load_saved_decoder_va_model(tmp_path, tokenizer=SimpleNamespace())
    assert restored.variance_head is None
    with torch.no_grad():
        batch = default_data_collator(ordered_rows())
        torch.testing.assert_close(restored(**batch).logits, model(**batch).logits, atol=0, rtol=0)


def completed_gaussian_run(run_dir, fusion, *, initialize=True):
    """Build complete two-fold reports using the production uncertainty export and metrics."""

    if initialize:
        _write_completed_run(run_dir, schema_version=11, gaze_fusion=fusion)
    for path in [run_dir / "training_parameters.json", *run_dir.glob("heldout_fold*/run_manifest.json")]:
        payload = json.loads(path.read_text())
        payload.update(loss="heteroscedastic", prediction_distribution=prediction_contract("heteroscedastic"))
        path.write_text(json.dumps(payload))
    frames = []
    for fold_dir in sorted(run_dir.glob("heldout_fold*")):
        path = fold_dir / "final_model/decoder_va_architecture.json"
        payload = json.loads(path.read_text())
        for metadata in (payload, payload["reconstruction"]):
            metadata["prediction_distribution"] = prediction_contract("heteroscedastic")
        path.write_text(json.dumps(payload))
        frame = pd.read_csv(fold_dir / "predictions.tsv", sep="\t")
        for name, values in (("valence", [0.01, 0.04]), ("arousal", [0.02, 0.03])):
            frame[f"log_variance_{name}"] = np.log(values)
        for name in ("valence", "arousal"):
            frame[f"pred_variance_{name}"] = np.exp(frame[f"log_variance_{name}"])
        for name in ("valence", "arousal"):
            frame[f"pred_std_{name}"] = np.sqrt(frame[f"pred_variance_{name}"])
        frame.to_csv(fold_dir / "predictions.tsv", sep="\t", index=False)
        metrics = {f"test_{key}": value for key, value in metrics_from_prediction_frame(frame).items()}
        metrics = {key: value if np.isfinite(value) else None for key, value in metrics.items()}
        (fold_dir / "metrics.json").write_text(json.dumps(metrics))
        frames.append(frame)
    write_oof_reports(frames, run_dir)


@pytest.mark.parametrize("fusion", ["none", "prefix-concat"])
def test_completed_results_zip_contains_validated_uncertainty_without_weights(tmp_path, fusion):
    run_dir = tmp_path / "complete"
    completed_gaussian_run(run_dir, fusion)
    archive_path = package_results(run_dir, results_root=tmp_path)
    assert "heteroscedastic" in archive_path.name
    with zipfile.ZipFile(archive_path) as archive:
        assert not any(name.endswith((".safetensors", ".pt")) for name in archive.namelist())
        metrics = json.loads(archive.read("complete/oof_metrics.json"))
        assert metrics["mse_mean"] == 0 and metrics["gaussian_nll_mean"] < 0
        assert metrics["coverage_95_valence"] == 1
        manifest = json.loads(archive.read("complete/results_manifest.json"))
        assert manifest["condition"]["loss"] == "heteroscedastic"


@pytest.mark.parametrize("corruption", ["nll", "variance", "missing", "distribution"])
def test_results_zip_rejects_uncertainty_corruption(tmp_path, corruption):
    run_dir = tmp_path / "corrupt"
    completed_gaussian_run(run_dir, "prefix-concat")
    if corruption == "nll":
        path = run_dir / "heldout_fold1/metrics.json"
        payload = json.loads(path.read_text())
        payload["test_gaussian_nll_mean"] += 1
        path.write_text(json.dumps(payload))
    elif corruption == "distribution":
        path = run_dir / "heldout_fold1/final_model/decoder_va_architecture.json"
        payload = json.loads(path.read_text())
        payload["prediction_distribution"] = prediction_contract()
        path.write_text(json.dumps(payload))
    else:
        path = run_dir / "heldout_fold1/predictions.tsv"
        frame = pd.read_csv(path, sep="\t")
        if corruption == "variance":
            frame.loc[0, "pred_variance_valence"] *= 2
        else:
            frame = frame.drop(columns="pred_std_valence")
        frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError):
        package_results(run_dir, results_root=tmp_path)


def test_external_discovery_checks_all_gaussian_internal_evidence(tmp_path):
    """External means-only evaluation must still verify the full stored probabilistic run."""

    from va_model_code.decoder_va.external_benchmarks import discover_completed_run
    from va_model_code.tests.test_external_benchmarks import _write_completed_run as external_fixture

    external_fixture(tmp_path)
    completed_gaussian_run(tmp_path, "prefix-concat", initialize=False)
    members = discover_completed_run(tmp_path)
    assert [member.held_out_fold for member in members] == [1, 2]
    path = tmp_path / "oof_predictions.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame.loc[0, "pred_std_valence"] *= 2
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError):
        discover_completed_run(tmp_path)


def test_external_inference_returns_only_means_from_declared_gaussian_model():
    """Negative log variances are valid outputs, not out-of-range VA predictions."""

    from va_model_code.tests.test_evaluate_external_cli import evaluate_external, OrderedTokenizer, RecordingModel

    class GaussianModel(RecordingModel):
        prediction_distribution = prediction_contract("heteroscedastic")

        def forward(self, **inputs):
            output = super().forward(**inputs)
            output.logits = torch.cat((output.logits, torch.full_like(output.logits, -4.)), dim=-1)
            return output

    model = GaussianModel()
    result = evaluate_external._predict_texts(
        model, OrderedTokenizer(), ["3", "0", "2", "1"], max_length=12, batch_size=2,
        device=torch.device("cpu"), dtype=torch.float32, autocast_enabled=False,
    )
    np.testing.assert_allclose(result, [[0.4, 0.2], [0.1, 0.05], [0.3, 0.15], [0.2, 0.1]], atol=1e-7)
    model.prediction_distribution = prediction_contract()
    with pytest.raises(RuntimeError, match="examples, 2"):
        evaluate_external._predict_texts(
            model, OrderedTokenizer(), ["0"], max_length=12, batch_size=2,
            device=torch.device("cpu"), dtype=torch.float32, autocast_enabled=False,
        )

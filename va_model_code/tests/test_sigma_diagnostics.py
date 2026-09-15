"""Verify frozen controls and diagnostics do not perturb the training trajectory."""

import json

import pytest
import torch
from transformers import TrainingArguments, default_data_collator

from va_model_code.decoder_va.model import DecoderVARegressor
from va_model_code.decoder_va.redistribution import GazeRedistributor, redistribution_contract, validate_redistribution_contract
from va_model_code.decoder_va.sigma_diagnostics import attach_sigma_diagnostics
from va_model_code.decoder_va.trainer import VARegressionTrainer
from va_model_code.tests.test_redistribution_transformer import FixedET2, tiny_native_backbone


def make_model(method, dtype=torch.float32):
    """Construct paired native causal models with dropout and deterministic gaze."""

    torch.manual_seed(137)
    return DecoderVARegressor(
        tiny_native_backbone(dtype=dtype),
        gaze_provider=FixedET2(feature_indices=(3,), repo_id="offline", revision="offline", filename="offline"),
        gaze_redistribution=redistribution_contract(method),
        gaze_projection_dim=8,
        gaze_projection_dropout=(0.1, 0.3), classifier_dropout=0.1,
    )


def make_trainer(directory, method, dtype=torch.float32):
    """Build an accumulated, clipped two-step run using actual Trainer callbacks."""

    rows = [
        {"input_ids": torch.tensor([1, 2, 5, 4]), "attention_mask": torch.ones(4, dtype=torch.long), "labels": torch.tensor([0.2, 0.8])},
        {"input_ids": torch.tensor([1, 8, 3, 6]), "attention_mask": torch.ones(4, dtype=torch.long), "labels": torch.tensor([0.9, 0.1])},
    ]
    args = TrainingArguments(
        output_dir=str(directory), max_steps=2, per_device_train_batch_size=1,
        gradient_accumulation_steps=2, learning_rate=0.01, max_grad_norm=0.001,
        logging_steps=1, save_strategy="no", report_to="none", use_cpu=True,
        remove_unused_columns=False, label_names=["labels"], seed=42,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16,
    )
    return VARegressionTrainer(
        model=make_model(method, dtype), args=args, train_dataset=rows,
        data_collator=default_data_collator, loss_name="mse",
        redistribution_learning_rate=0.001 if method == "asym-gaussian" else None,
    )


def test_fixed_gaussian_retains_smoothing_without_trainable_widths():
    config = redistribution_contract("fixed-gaussian")
    assert config["trainable"] is False
    assert validate_redistribution_contract(config, gaze_fusion="prefix-concat", feature_indices=(3,)) == config
    kernel = GazeRedistributor(config, (3,))
    assert all(not p.requires_grad for p in kernel.parameters())
    raw = torch.tensor([[[1.], [0.], [0.]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    expected = GazeRedistributor(redistribution_contract("asym-gaussian"), (3,))(raw, mask)
    torch.testing.assert_close(kernel(raw, mask), expected, rtol=0, atol=0)
    assert kernel(raw, mask)[0, 1, 0] > 0
    with pytest.raises(ValueError, match="equal"):
        redistribution_contract("fixed-gaussian", init_sigma_right=2.)


@pytest.mark.parametrize("method", ["none", "fixed-gaussian", "asym-gaussian"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_diagnostics_preserve_exact_training_state_and_measure_updates(tmp_path, method, dtype):
    baseline = make_trainer(tmp_path / "baseline", method, dtype)
    baseline.train()
    observed = make_trainer(tmp_path / "observed", method, dtype)
    attach_sigma_diagnostics(observed, tmp_path / "observed", every_steps=1, batch_size=2)
    observed.train()
    for name, parameter in baseline.model.state_dict().items():
        torch.testing.assert_close(observed.model.state_dict()[name], parameter, rtol=0, atol=0)
    updates = [json.loads(line) for line in (tmp_path / 'observed/sigma_updates.jsonl').read_text().splitlines()]
    probes = [json.loads(line) for line in (tmp_path / 'observed/sigma_probes.jsonl').read_text().splitlines()]
    assert len(updates) == 2
    assert all(record['microbatches'] == 2 for record in updates)
    assert probes[0]['event'] == 'train_begin'
    assert probes[-1]['event'] == 'train_end_selected_model'
    assert set(probes[0]['cases']) == {'configured', 'raw', 'symmetric_1', 'left_05_right_2', 'left_2_right_05'}
    if method == "asym-gaussian":
        update = updates[0]
        for name in update['gradient_before_clipping']:
            assert abs(update['gradient_after_clipping'][name]) <= abs(update['gradient_before_clipping'][name])
            assert update['delta_log_sigma'][name] != 0
        assert len(probes[0]['cases']['configured']['local_gradient_sample_output_side']) == 2
    elif method == "fixed-gaussian":
        assert all(not p.requires_grad for p in observed.model.gaze_redistributor.parameters())
        assert all(value == 0 for record in updates for value in record['delta_log_sigma'].values())
    else:
        assert observed.model.gaze_redistributor is None


def test_probe_restores_state_on_failure(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, "asym-gaussian")
    observer = attach_sigma_diagnostics(trainer, tmp_path, every_steps=1, batch_size=2)
    original = trainer.model.gaze_redistributor
    flags = [p.requires_grad for p in trainer.model.parameters()]
    rng = torch.get_rng_state().clone()

    def fail(*args, **kwargs):
        """Raise after diagnostics temporarily freezes parameters."""
        raise RuntimeError("probe failure")

    monkeypatch.setattr(trainer.model.gaze_provider, "compute", fail)
    with pytest.raises(RuntimeError, match="probe failure"):
        observer.probe(trainer.state, "failure_test")
    assert trainer.model.gaze_redistributor is original
    assert [p.requires_grad for p in trainer.model.parameters()] == flags
    assert trainer.model.training
    assert torch.equal(torch.get_rng_state(), rng)


def test_fixed_gaussian_stays_frozen_after_strict_reload(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from va_model_code.decoder_va import model as model_module

    monkeypatch.setattr(model_module, "_load_qwen_text_backbone", tiny_native_backbone)
    monkeypatch.setattr(model_module, "ET2GazeProvider", FixedET2)
    tokenizer = SimpleNamespace(padding_side="right")
    model = model_module.build_qwen_va_model(
        tokenizer, finetuning_mode="full", dtype=torch.float32,
        gaze_redistribution=redistribution_contract("fixed-gaussian"),
    )
    model.save_architecture_manifest(tmp_path)
    save_file(model.state_dict(), tmp_path / model_module.SAFE_WEIGHTS_FILENAME)
    restored, _ = model_module.load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)
    assert restored.gaze_redistribution["trainable"] is False
    assert all(not p.requires_grad for p in restored.gaze_redistributor.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

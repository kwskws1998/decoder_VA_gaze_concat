"""Exercise singleton TRT readout learning through the native Trainer lifecycle."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from transformers import default_data_collator

from va_model_code.decoder_va import model as model_module
from va_model_code.decoder_va.gaze_relation import gaze_relation_contract
from va_model_code.decoder_va.model import (
    build_qwen_va_model,
    load_saved_decoder_va_model,
    validate_saved_gaze_relation,
    validate_saved_readout,
)
from va_model_code.decoder_va.readout import readout_contract
from va_model_code.tests.test_gaze_attention import tiny_backbone
from va_model_code.tests.test_gaze_relation_trainer_integration import (
    ResumeObserver,
    assert_prediction_order,
    make_trainer,
    ordered_rows,
    relation_optimizer_group,
    relation_parameters,
)
from va_model_code.tests.test_readout_model import SingletonGazeProvider


def build_readout(seed):
    """Build a reloadable native hybrid decoder with one positive TRT per example."""

    torch.manual_seed(seed)
    return build_qwen_va_model(
        SimpleNamespace(),
        dtype="float32",
        finetuning_mode="full",
        gaze_relation=gaze_relation_contract("trt-readout-rotation"),
        readout=readout_contract("learned-token"),
        gaze_projection_dim=8,
        gaze_projection_dropout=(0.0, 0.0),
        classifier_dropout=0.0,
    )


class ReadoutResumeObserver(ResumeObserver):
    """Observe both the relation optimizer and the separately optimized readout."""

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Capture restored values and Adam history before another update occurs."""

        super().on_train_begin(args, state, control, model=model, optimizer=optimizer, **kwargs)
        parameter = model.readout_token
        self.observation.update({
            "readout": parameter.detach().cpu().clone(),
            "readout_step": int(optimizer.state[parameter]["step"].item()),
            "readout_first_moment": optimizer.state[parameter]["exp_avg"].detach().cpu().clone(),
        })


def test_singleton_readout_trainer_updates_roundtrips_and_resumes(tmp_path, monkeypatch):
    """Real optimizer steps must update final alpha/readout and restore their complete state."""

    monkeypatch.setattr(model_module, "load_qwen_backbone", lambda *args, **kwargs: tiny_backbone())
    monkeypatch.setattr(model_module, "ET2GazeProvider", SingletonGazeProvider)
    rows = ordered_rows()
    model = build_readout(seed=811)
    initial_readout = model.readout_token.detach().clone()
    alpha = model.backbone.layers[-1].self_attn.gaze_relative_rotation.alpha
    assert torch.count_nonzero(alpha) == 0
    trainer = make_trainer(model, rows, tmp_path, steps=2, alpha_lr=1e-2)
    result = trainer.train()
    assert trainer.state.global_step == 2
    assert np.isfinite(result.training_loss)
    assert model.backbone.is_gradient_checkpointing
    assert not torch.equal(model.readout_token, initial_readout)
    assert torch.count_nonzero(alpha) > 0
    for parameter in (model.readout_token, alpha):
        state = trainer.optimizer.state[parameter]
        assert int(state["step"].item()) == 2
        assert torch.isfinite(state["exp_avg"]).all()
        assert torch.count_nonzero(state["exp_avg"]) > 0

    alpha_group = relation_optimizer_group(model, trainer.optimizer)
    assert alpha_group["lr"] == pytest.approx(1e-2)
    assert alpha_group["weight_decay"] == 0.0
    readout_groups = [group for group in trainer.optimizer.param_groups
                      if any(parameter is model.readout_token for parameter in group["params"])]
    assert len(readout_groups) == 1
    assert readout_groups[0] is not alpha_group
    assert readout_groups[0]["lr"] == pytest.approx(5e-4)
    assert_prediction_order(trainer, rows)
    reference_predictions = trainer.predict(rows).predictions
    checkpoint = tmp_path / "checkpoint-2"
    model.save_architecture_manifest(checkpoint)
    weights = checkpoint / "model.safetensors"
    validate_saved_readout(model, weights)
    validate_saved_gaze_relation(model, weights)
    saved = load_file(weights)
    torch.testing.assert_close(saved["readout_token"], model.readout_token.detach().cpu(), atol=0, rtol=0)
    saved_readout_moment = trainer.optimizer.state[model.readout_token]["exp_avg"].detach().cpu().clone()
    saved_alpha_moments = {
        name: trainer.optimizer.state[parameter]["exp_avg"].detach().cpu().clone()
        for name, parameter in relation_parameters(model).items()
    }

    reloaded, _ = load_saved_decoder_va_model(checkpoint, tokenizer=SimpleNamespace())
    restored_predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), 2):
            batch = default_data_collator(rows[start:start + 2])
            batch.pop("labels")
            restored_predictions.append(reloaded(**batch).logits.cpu())
    np.testing.assert_array_equal(torch.cat(restored_predictions).numpy(), reference_predictions)

    fresh = build_readout(seed=912)
    assert not torch.equal(fresh.readout_token, saved["readout_token"])
    observer = ReadoutResumeObserver()
    resumed = make_trainer(fresh, rows, tmp_path, steps=3, alpha_lr=7e-3, callbacks=[observer])
    resumed_result = resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 3
    assert np.isfinite(resumed_result.training_loss)
    observation = observer.observation
    assert observation["global_step"] == 2
    assert observation["readout_step"] == 2
    assert observation["lr"] == pytest.approx(1e-2)
    torch.testing.assert_close(observation["readout"], saved["readout_token"], atol=0, rtol=0)
    torch.testing.assert_close(observation["readout_first_moment"], saved_readout_moment, atol=0, rtol=0)
    for name in relation_parameters(fresh):
        torch.testing.assert_close(observation["alpha"][name], saved[name], atol=0, rtol=0)
        torch.testing.assert_close(observation["first_moments"][name], saved_alpha_moments[name], atol=0, rtol=0)
    assert not torch.equal(fresh.readout_token.detach().cpu(), saved["readout_token"])
    final_alpha_name = next(iter(relation_parameters(fresh)))
    assert not torch.equal(relation_parameters(fresh)[final_alpha_name].detach().cpu(), saved[final_alpha_name])
    validate_saved_readout(fresh, tmp_path / "checkpoint-3" / "model.safetensors")
    validate_saved_gaze_relation(fresh, tmp_path / "checkpoint-3" / "model.safetensors")
    assert_prediction_order(resumed, rows)

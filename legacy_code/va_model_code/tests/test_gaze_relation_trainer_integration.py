"""Exercise native Qwen TRT training, prediction, checkpoint saving, and resume."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from transformers import TrainerCallback, TrainingArguments, default_data_collator

from va_model_code.decoder_va.gaze_relation import gaze_relation_contract
from va_model_code.decoder_va.model import DecoderVARegressor, validate_saved_gaze_relation
from va_model_code.decoder_va.trainer import GAZE_RELATION_PARAMETER_SUFFIX, VARegressionTrainer
from va_model_code.tests.test_gaze_attention import tiny_backbone
from va_model_code.tests.test_gaze_relation_model import SparseGazeProvider


def tiny_regressor(mode, seed, input_mode="aligned"):
    """Reconstruct the complete native hybrid model with fresh, zero-initialized alpha."""

    torch.manual_seed(seed)
    backbone = tiny_backbone()
    if mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.0,
                target_modules="all-linear", task_type=TaskType.FEATURE_EXTRACTION,
            ),
        )
    return DecoderVARegressor(
        backbone,
        gaze_provider=SparseGazeProvider(),
        gaze_relation=gaze_relation_contract(
            "trt-relative-rotation", scale=3.0, input_mode=input_mode,
            shuffle_seed=42 if input_mode == "shuffled" else None,
        ),
        gaze_projection_dim=8,
        gaze_projection_dropout=(0.0, 0.0),
        classifier_dropout=0.0,
    )


def ordered_rows():
    """Provide distinct labels and unequal right-padded sequences for prediction order."""

    token_rows = [[1, 2, 4, 8, 3], [2, 3, 5], [7, 3, 2, 1], [3, 4, 1, 5, 6]]
    labels = [[0.1, 0.8], [0.3, 0.6], [0.7, 0.2], [0.9, 0.4]]
    return [
        {
            "input_ids": torch.tensor(tokens + [0] * (5 - len(tokens))),
            "attention_mask": torch.tensor([1] * len(tokens) + [0] * (5 - len(tokens))),
            "labels": torch.tensor(label, dtype=torch.float32),
        }
        for tokens, label in zip(token_rows, labels)
    ]


def relation_parameters(model):
    """Find the learned attention phases without assuming full or PEFT state paths."""

    return {
        name: parameter for name, parameter in model.named_parameters()
        if name.endswith(GAZE_RELATION_PARAMETER_SUFFIX)
    }


def relation_optimizer_group(model, optimizer):
    """Require exactly one dedicated group containing every phase parameter."""

    alpha_ids = {id(parameter) for parameter in relation_parameters(model).values()}
    matching = [
        group for group in optimizer.param_groups
        if alpha_ids.intersection(id(parameter) for parameter in group["params"])
    ]
    assert len(matching) == 1
    assert {id(parameter) for parameter in matching[0]["params"]} == alpha_ids
    return matching[0]


def make_trainer(model, rows, output_dir, steps, alpha_lr, callbacks=None):
    """Use the actual Trainer lifecycle with checkpointing and complete optimizer state."""

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-4,
        weight_decay=0.2,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=1,
        eval_strategy="no",
        logging_strategy="no",
        report_to="none",
        use_cpu=True,
        disable_tqdm=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=43,
        data_seed=43,
    )
    return VARegressionTrainer(
        model=model,
        args=arguments,
        train_dataset=rows,
        eval_dataset=rows,
        data_collator=default_data_collator,
        loss_name="mse",
        gaze_relation_learning_rate=alpha_lr,
        callbacks=callbacks,
    )


def assert_prediction_order(trainer, rows):
    """Match Trainer labels and predictions to direct inference in original row order."""

    result = trainer.predict(rows)
    expected_labels = torch.stack([row["labels"] for row in rows]).numpy()
    np.testing.assert_array_equal(result.label_ids, expected_labels)
    manual_predictions = []
    trainer.model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 2):
            batch = default_data_collator(rows[start:start + 2])
            batch.pop("labels")
            manual_predictions.append(trainer.model(**batch).logits.cpu())
    np.testing.assert_allclose(
        result.predictions, torch.cat(manual_predictions).numpy(), rtol=1e-6, atol=1e-7,
    )
    assert result.predictions.shape == (len(rows), 2)
    assert np.isfinite(result.predictions).all()
    assert ((result.predictions >= 0) & (result.predictions <= 1)).all()


class ResumeObserver(TrainerCallback):
    """Capture model and optimizer state immediately after Trainer restores a checkpoint."""

    def __init__(self):
        self.observation = None

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Observe restored phase values, moments, learning rate, and weight decay."""

        group = relation_optimizer_group(model, optimizer)
        parameters = relation_parameters(model)
        self.observation = {
            "global_step": state.global_step,
            "alpha": {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()},
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "steps": [int(optimizer.state[parameter]["step"].item()) for parameter in parameters.values()],
            "first_moments": {
                name: optimizer.state[parameter]["exp_avg"].detach().cpu().clone()
                for name, parameter in parameters.items()
            },
        }


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("input_mode", ["aligned", "shuffled"])
def test_native_trainer_two_steps_save_predict_and_resume_third_step(tmp_path, mode, input_mode):
    """Check the production Trainer path through native attention and a resumed optimizer."""

    rows = ordered_rows()
    original = tiny_regressor(mode, seed=811, input_mode=input_mode)
    assert all(torch.count_nonzero(parameter) == 0 for parameter in relation_parameters(original).values())
    trainer = make_trainer(original, rows, tmp_path, steps=2, alpha_lr=1e-2)
    trained = trainer.train()
    assert trainer.state.global_step == 2
    assert np.isfinite(trained.training_loss)
    native = original.backbone.get_base_model() if mode == "lora" else original.backbone
    assert native.is_gradient_checkpointing
    assert_prediction_order(trainer, rows)

    checkpoint = tmp_path / "checkpoint-2"
    weights = checkpoint / "model.safetensors"
    assert weights.is_file()
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "scheduler.pt").is_file()
    validate_saved_gaze_relation(original, weights)
    saved = load_file(weights)
    saved_alpha = {name: saved[name] for name in relation_parameters(original)}
    original_moments = {}
    for name, parameter in relation_parameters(original).items():
        assert saved_alpha[name].dtype == torch.float32
        assert torch.isfinite(saved_alpha[name]).all()
        assert torch.count_nonzero(saved_alpha[name]) > 0
        torch.testing.assert_close(saved_alpha[name], parameter.detach().cpu(), rtol=0, atol=0)
        original_moments[name] = trainer.optimizer.state[parameter]["exp_avg"].detach().cpu().clone()
    group = relation_optimizer_group(original, trainer.optimizer)
    assert group["lr"] == pytest.approx(1e-2)
    assert group["weight_decay"] == 0.0

    fresh = tiny_regressor(mode, seed=912, input_mode=input_mode)
    assert all(torch.count_nonzero(parameter) == 0 for parameter in relation_parameters(fresh).values())
    validate_saved_gaze_relation(fresh, weights)
    observer = ResumeObserver()
    resumed = make_trainer(fresh, rows, tmp_path, steps=3, alpha_lr=7e-3, callbacks=[observer])
    resumed.create_optimizer()
    fresh_group = relation_optimizer_group(fresh, resumed.optimizer)
    assert fresh_group["lr"] == pytest.approx(7e-3)
    fresh_group["weight_decay"] = 0.25
    resumed_result = resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.state.global_step == 3
    assert np.isfinite(resumed_result.training_loss)
    assert observer.observation["global_step"] == 2
    assert observer.observation["lr"] == pytest.approx(1e-2)
    assert observer.observation["weight_decay"] == 0.0
    assert observer.observation["steps"] == [2] * len(saved_alpha)
    for name, value in saved_alpha.items():
        torch.testing.assert_close(observer.observation["alpha"][name], value, rtol=0, atol=0)
        torch.testing.assert_close(observer.observation["first_moments"][name], original_moments[name], rtol=0, atol=0)
    resumed_group = relation_optimizer_group(fresh, resumed.optimizer)
    assert resumed_group["lr"] == pytest.approx(1e-2)
    assert resumed_group["weight_decay"] == 0.0
    third_checkpoint = tmp_path / "checkpoint-3" / "model.safetensors"
    validate_saved_gaze_relation(fresh, third_checkpoint)
    third_saved = load_file(third_checkpoint)
    for name in saved_alpha:
        assert third_saved[name].dtype == torch.float32
        assert torch.isfinite(third_saved[name]).all()
        assert not torch.equal(third_saved[name], saved_alpha[name])
    assert_prediction_order(resumed, rows)

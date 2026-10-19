"""Optimizer isolation, relation diagnostics, and prediction identity regression tests."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from va_model_code.decoder_va.dataset import VABatchCollator
from va_model_code.decoder_va.evaluation import prediction_frame
from va_model_code.decoder_va.trainer import VARegressionTrainer


class GazeOptimizerModel(nn.Module):
    """Expose multiple relation layers, ordinary parameters, and optional learned widths."""

    def __init__(self, *, redistribution: bool = False):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)
        self.unrelated_alpha = nn.Parameter(torch.tensor(0.5))
        self.frozen = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer, values in zip(self.layers, ([0.2, -0.3], [-0.4, 0.1])):
            layer.gaze_relative_rotation = nn.Module()
            layer.gaze_relative_rotation.alpha = nn.Parameter(torch.tensor(values))
        if redistribution:
            self.gaze_redistributor = nn.Module()
            self.gaze_redistributor.kernel = nn.Module()
            self.gaze_redistributor.kernel.log_sigma_left = nn.Parameter(torch.tensor(0.0))
            self.gaze_redistributor.kernel.log_sigma_right = nn.Parameter(torch.tensor(0.0))


def optimizer_trainer(model, tmp_path, **kwargs):
    """Construct an offline CPU trainer with distinguishable base and special rates."""

    return VARegressionTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            learning_rate=6e-6,
            weight_decay=0.01,
            report_to="none",
            use_cpu=True,
            disable_tqdm=True,
        ),
        loss_name="mse",
        **kwargs,
    )


@pytest.mark.parametrize("redistribution", [False, True])
def test_relation_optimizer_isolates_every_layer_and_preserves_other_groups(tmp_path, redistribution):
    """All relation alphas share zero decay and their own scheduled rate without omissions."""

    model = GazeOptimizerModel(redistribution=redistribution)
    trainer = optimizer_trainer(
        model,
        tmp_path,
        gaze_relation_learning_rate=1e-3,
        redistribution_learning_rate=2e-3 if redistribution else None,
    )
    optimizer = trainer.create_optimizer()
    assert trainer.create_optimizer() is optimizer
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len(expected)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in expected}
    groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
    relation_parameters = [layer.gaze_relative_rotation.alpha for layer in model.layers]
    relation_group = groups[id(relation_parameters[0])]
    assert {id(parameter) for parameter in relation_group["params"]} == {
        id(parameter) for parameter in relation_parameters
    }
    assert relation_group["lr"] == pytest.approx(1e-3)
    assert relation_group["weight_decay"] == 0
    for parameter in (model.backbone.weight, model.unrelated_alpha):
        assert groups[id(parameter)]["lr"] == pytest.approx(6e-6)
        assert groups[id(parameter)]["weight_decay"] == pytest.approx(0.01)
    for parameter in (model.backbone.bias, model.norm.weight, model.norm.bias):
        assert groups[id(parameter)]["weight_decay"] == 0
    if redistribution:
        for parameter in model.gaze_redistributor.parameters():
            assert groups[id(parameter)]["lr"] == pytest.approx(2e-3)
            assert groups[id(parameter)]["weight_decay"] == 0

    before = [parameter.detach().clone() for parameter in relation_parameters]
    trainer.create_scheduler(num_training_steps=10, optimizer=optimizer)
    for parameter in relation_parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    trainer.lr_scheduler.step()
    for parameter, original in zip(relation_parameters, before):
        torch.testing.assert_close(parameter, original - 1e-3, rtol=1e-6, atol=1e-8)
    assert relation_group["lr"] / groups[id(model.backbone.weight)]["lr"] == pytest.approx(1e-3 / 6e-6)


@pytest.mark.parametrize("frozen", [False, True])
def test_relation_lr_rejects_missing_trainable_relation_parameters(tmp_path, frozen):
    """A misspelled or frozen relation cannot silently train at the wrong learning rate."""

    model = GazeOptimizerModel() if frozen else nn.Linear(2, 2)
    if frozen:
        for layer in model.layers:
            layer.gaze_relative_rotation.alpha.requires_grad_(False)
    trainer = optimizer_trainer(model, tmp_path, gaze_relation_learning_rate=1e-3)
    with pytest.raises(ValueError, match="requires trainable gaze_relative_rotation.alpha"):
        trainer.create_optimizer()


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True])
def test_relation_lr_requires_finite_positive_values(tmp_path, rate):
    """Invalid special rates fail before an optimizer or training run is created."""

    with pytest.raises(ValueError, match="finite and positive"):
        optimizer_trainer(GazeOptimizerModel(), tmp_path, gaze_relation_learning_rate=rate)


def test_relation_logs_aggregate_all_heads_and_use_current_lr_without_mutating_input(tmp_path):
    """Reported statistics cover every relation layer and track scheduler changes."""

    model = GazeOptimizerModel()
    trainer = optimizer_trainer(model, tmp_path, gaze_relation_learning_rate=1e-3)
    optimizer = trainer.create_optimizer()
    trainer.create_scheduler(num_training_steps=10, optimizer=optimizer)
    optimizer.step()
    trainer.lr_scheduler.step()
    source_logs = {"loss": 0.25}
    trainer.log(source_logs)
    assert source_logs == {"loss": 0.25}
    logged = trainer.state.log_history[-1]
    assert logged["gaze_relation_alpha_norm"] == pytest.approx(np.sqrt(0.3))
    assert logged["gaze_relation_alpha_absmax"] == pytest.approx(0.4)
    assert logged["gaze_relation_alpha_mean"] == pytest.approx(-0.1)
    assert logged["gaze_relation_learning_rate"] == pytest.approx(9e-4)


class RowIdentityModel(nn.Module):
    """Return row-encoded predictions so evaluation reordering is directly observable."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, attention_mask):
        marker = input_ids[:, 0].float() / 10.0
        return {"logits": torch.stack((marker, 1 - marker), dim=-1) + self.anchor * 0}


class PaddingTokenizer:
    """Supply the minimum padding contract for variable-length identity inputs."""

    pad_token_id = 0
    padding_side = "right"


def test_length_grouped_training_preserves_predict_and_evaluate_row_order(tmp_path):
    """Real Trainer prediction keeps metadata identities despite unequal sequence lengths."""

    lengths = [2, 9, 3, 7, 1, 6]
    rows = [
        {
            "input_ids": torch.full((length,), row + 1, dtype=torch.long),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "labels": torch.tensor([(row + 1) / 10, 1 - (row + 1) / 10]),
        }
        for row, length in enumerate(lengths)
    ]
    kwargs = {
        "output_dir": str(tmp_path),
        "per_device_eval_batch_size": 2,
        "report_to": "none",
        "use_cpu": True,
        "disable_tqdm": True,
        "remove_unused_columns": False,
        "label_names": ["labels"],
    }
    if "train_sampling_strategy" in inspect.signature(TrainingArguments).parameters:
        kwargs["train_sampling_strategy"] = "group_by_length"
    else:
        kwargs["group_by_length"] = True
    trainer = VARegressionTrainer(
        model=RowIdentityModel(),
        args=TrainingArguments(**kwargs),
        train_dataset=rows,
        eval_dataset=rows,
        data_collator=VABatchCollator(PaddingTokenizer()),
        loss_name="mse",
    )
    assert list(trainer._get_eval_sampler(rows)) == list(range(len(rows)))
    eval_markers = torch.cat([batch["input_ids"][:, 0] for batch in trainer.get_eval_dataloader()])
    assert eval_markers.tolist() == [1, 2, 3, 4, 5, 6]
    result = trainer.predict(rows)
    labels = torch.stack([row["labels"] for row in rows]).numpy()
    np.testing.assert_array_equal(result.label_ids, labels)
    np.testing.assert_allclose(result.predictions, labels, rtol=1e-6, atol=1e-8)
    metadata = pd.DataFrame(
        {
            "index": [10, 11, 12, 13, 14, 15],
            "text": [f"row {row}" for row in range(6)],
            "dataset_of_origin": ["A", "B", "A", "B", "A", "B"],
            "valence": labels[:, 0],
            "arousal": labels[:, 1],
        }
    )
    saved = prediction_frame(metadata, result.label_ids, result.predictions, fold=1)
    assert saved["index"].tolist() == metadata["index"].tolist()
    np.testing.assert_array_equal(saved[["valence", "arousal"]], labels)


def test_prediction_frame_rejects_reordered_labels_even_when_overall_error_is_zero():
    """Paired labels and predictions can have perfect MSE while metadata is wrong."""

    metadata = pd.DataFrame(
        {
            "index": [10, 20],
            "text": ["first", "second"],
            "dataset_of_origin": ["A", "B"],
            "valence": [0.2, 0.8],
            "arousal": [0.3, 0.7],
        }
    )
    labels = metadata[["valence", "arousal"]].to_numpy()[::-1].copy()
    with pytest.raises(ValueError, match="do not match evaluation metadata in row order"):
        prediction_frame(metadata, labels, labels, fold=1)


def test_prediction_frame_accepts_float32_rounding_but_rejects_partial_gold_columns():
    """The alignment guard permits dataset tensor conversion and rejects incomplete gold data."""

    metadata = pd.DataFrame(
        {
            "index": [10],
            "text": ["first"],
            "dataset_of_origin": ["A"],
            "valence": [1 / 3],
            "arousal": [2 / 3],
        }
    )
    labels = metadata[["valence", "arousal"]].to_numpy(dtype=np.float32)
    result = prediction_frame(metadata, labels, labels, fold=1)
    assert result["index"].tolist() == [10]
    with pytest.raises(ValueError, match="both valence and arousal"):
        prediction_frame(metadata.drop(columns="arousal"), labels, labels, fold=1)

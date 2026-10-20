"""Snapshot and persistence checks for actual, small TRT rotation modules."""

from __future__ import annotations

import copy
import json
import math

import pytest
import torch
from torch import nn

from va_model_code.decoder_va.gaze_relation import TRTRelativeRotation, gaze_relation_contract
from va_model_code.decoder_va.gaze_relation_reporting import (
    gaze_relation_parameter_snapshot,
    write_gaze_relation_report,
)


def _model(*, enabled=True, shuffled=False):
    """Build two registered attention modules with distinct coefficients and modes."""

    model = nn.Module()
    options = {"input_mode": "shuffled", "shuffle_seed": 42} if shuffled else {}
    model.gaze_relation = gaze_relation_contract(
        "trt-relative-rotation" if enabled else "none", scale=2.0 if enabled else 1.0,
        **options,
    )
    model.gaze_relation_layers = []
    if enabled:
        model.backbone = nn.Module()
        model.backbone.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for index, values in enumerate(([0.25, -0.5], [0.0, 1.0])):
            attention = model.backbone.layers[index]
            attention.gaze_relative_rotation = TRTRelativeRotation(2, scale=2.0)
            with torch.no_grad():
                attention.gaze_relative_rotation.alpha.copy_(torch.tensor(values))
            model.gaze_relation_layers.append({
                "name": f"layers.{index}", "layer_index": index,
                "num_heads": 2, "head_dim": 4,
            })
        model.backbone.layers[1].gaze_relative_rotation.alpha.requires_grad_(False)
    return model


@pytest.mark.parametrize("shuffled", [False, True])
def test_snapshot_contains_exact_current_parameters_and_preserves_contract(shuffled):
    model = _model(shuffled=shuffled)
    snapshot = gaze_relation_parameter_snapshot(model)
    assert snapshot["enabled"] is True
    assert snapshot["contract"] == model.gaze_relation
    assert snapshot["installed_layers"] == model.gaze_relation_layers
    assert snapshot["total_trainable_parameters"] == 2
    assert snapshot["layers"][0] == {
        "name": "backbone.layers.0.gaze_relative_rotation",
        "num_heads": 2,
        "alpha": [0.25, -0.5],
        "alpha_l2_norm": math.hypot(0.25, -0.5),
        "alpha_max_abs": 0.5,
        "scale": 2.0,
        "phase_coefficient_per_trt_unit": [0.125, -0.25],
        "trainable_parameters": 2,
    }
    assert snapshot["layers"][1]["alpha"] == [0.0, 1.0]
    assert snapshot["layers"][1]["trainable_parameters"] == 0
    assert "not assumed milliseconds" in snapshot["input_units"]
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot


def test_snapshot_does_not_modify_parameters_gradients_modes_or_metadata():
    model = _model()
    model.train()
    model.backbone.layers[0].eval()
    alpha = model.backbone.layers[0].gaze_relative_rotation.alpha
    alpha.grad = torch.tensor([0.5, 0.75])
    original_gradient = alpha.grad
    saved_gradient = alpha.grad.clone()
    saved_state = copy.deepcopy(model.state_dict())
    saved_modes = {name: module.training for name, module in model.named_modules()}
    saved_metadata = copy.deepcopy(model.gaze_relation_layers)
    snapshot = gaze_relation_parameter_snapshot(model)
    snapshot["installed_layers"][0]["name"] = "mutated"
    snapshot["contract"]["scale"] = 5.0
    snapshot["layers"][0]["alpha"][0] = 99.0
    assert model.gaze_relation_layers == saved_metadata
    assert model.gaze_relation["scale"] == 2.0
    assert {name: module.training for name, module in model.named_modules()} == saved_modes
    assert alpha.grad is original_gradient
    torch.testing.assert_close(alpha.grad, saved_gradient, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved_state[name], rtol=0, atol=0)


def test_writer_emits_snapshot_and_disabled_writer_creates_nothing(tmp_path):
    model = _model()
    output = write_gaze_relation_report(model, tmp_path / "enabled")
    assert output == tmp_path / "enabled" / "gaze_relation_parameters.json"
    assert json.loads(output.read_text()) == gaze_relation_parameter_snapshot(model)
    disabled = _model(enabled=False)
    report = gaze_relation_parameter_snapshot(disabled)
    assert report["enabled"] is False
    assert report["layers"] == []
    assert report["total_trainable_parameters"] == 0
    assert write_gaze_relation_report(disabled, tmp_path / "disabled") is None
    assert not (tmp_path / "disabled").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_alpha_is_rejected_before_creating_report(tmp_path, value):
    model = _model()
    with torch.no_grad():
        model.backbone.layers[0].gaze_relative_rotation.alpha[0] = value
    with pytest.raises(ValueError, match="alpha.*finite"):
        write_gaze_relation_report(model, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("mismatch", ["scale", "count", "head_count", "layer_name"])
def test_metadata_mismatch_fails_instead_of_exporting_misleading_values(mismatch):
    model = _model()
    if mismatch == "scale":
        model.backbone.layers[0].gaze_relative_rotation.scale = 3.0
    elif mismatch == "count":
        model.gaze_relation_layers.pop()
    elif mismatch == "head_count":
        model.gaze_relation_layers[0]["num_heads"] = 3
    else:
        model.gaze_relation_layers[0]["name"] = "unrelated_layer"
    with pytest.raises(ValueError):
        gaze_relation_parameter_snapshot(model)

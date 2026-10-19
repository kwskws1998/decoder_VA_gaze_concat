"""Export learned TRT rotation coefficients without changing model state."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from .gaze_relation import TRTRelativeRotation, validate_gaze_relation_contract


def gaze_relation_parameter_snapshot(model: nn.Module) -> dict:
    """Return a finite, JSON-compatible snapshot of the current model coefficients.

    The caller decides which checkpoint is active. This function neither loads a
    checkpoint nor changes training mode, parameters, gradients, or optimizer
    state. The phase coefficient has radians per ET2 TRT prediction unit, with
    no assumption that the predictor's outputs are measured in milliseconds.
    """

    contract = validate_gaze_relation_contract(model.gaze_relation)
    enabled = contract["method"] != "none"
    report = {
        "schema_version": 1,
        "enabled": enabled,
        "contract": contract,
        "input_units": "ET2 TRT prediction units; not assumed milliseconds",
        "phase_coefficient_units": "radians per ET2 TRT prediction unit",
        "installed_layers": [],
        "layers": [],
        "total_trainable_parameters": 0,
    }
    if not enabled:
        return report
    metadata = getattr(model, "gaze_relation_layers", None)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Enabled gaze relation requires nonempty gaze_relation_layers metadata.")
    installed_layers = json.loads(json.dumps(metadata, allow_nan=False))
    rotations = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, TRTRelativeRotation)
    ]
    if len(rotations) != len(installed_layers):
        raise ValueError("Installed TRT rotation count does not match gaze_relation_layers metadata.")
    matched_names = set()
    for name, module in rotations:
        if module.scale != contract["scale"]:
            raise ValueError(f"TRT rotation scale at {name!r} disagrees with the gaze relation contract.")
        if not name.endswith(".gaze_relative_rotation"):
            raise ValueError(f"TRT rotation at {name!r} is not attached to a full-attention module.")
        attention_name = name.removesuffix(".gaze_relative_rotation")
        matches = [
            layer for layer in installed_layers
            if isinstance(layer, dict) and isinstance(layer.get("name"), str)
            and (attention_name == layer["name"] or attention_name.endswith("." + layer["name"]))
        ]
        if len(matches) != 1 or matches[0]["name"] in matched_names:
            raise ValueError(f"TRT rotation at {name!r} does not uniquely match installed layer metadata.")
        layer_metadata = matches[0]
        matched_names.add(layer_metadata["name"])
        if type(layer_metadata.get("num_heads")) is not int or layer_metadata["num_heads"] != module.num_heads:
            raise ValueError(f"TRT rotation head count at {name!r} disagrees with installed layer metadata.")
        alpha_tensor = module.alpha.detach().to(device="cpu", dtype=torch.float32)
        if alpha_tensor.shape != (module.num_heads,) or not bool(torch.isfinite(alpha_tensor).all()):
            raise ValueError(f"TRT rotation alpha at {name!r} must be a finite vector matching num_heads.")
        alpha = alpha_tensor.tolist()
        coefficient = [value / float(module.scale) for value in alpha]
        if not all(math.isfinite(value) for value in coefficient):
            raise ValueError(f"TRT phase coefficients at {name!r} must be finite.")
        trainable = int(module.alpha.numel()) if module.alpha.requires_grad else 0
        report["layers"].append({
            "name": name,
            "num_heads": int(module.num_heads),
            "alpha": alpha,
            "alpha_l2_norm": math.hypot(*alpha),
            "alpha_max_abs": max(abs(value) for value in alpha),
            "scale": float(module.scale),
            "phase_coefficient_per_trt_unit": coefficient,
            "trainable_parameters": trainable,
        })
        report["total_trainable_parameters"] += trainable
    report["installed_layers"] = installed_layers
    return report


def write_gaze_relation_report(model: nn.Module, output_dir: str | Path) -> Path | None:
    """Write learned coefficients for enabled runs and leave disabled runs untouched."""

    report = gaze_relation_parameter_snapshot(model)
    if not report["enabled"]:
        return None
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    destination = Path(output_dir) / "gaze_relation_parameters.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    return destination


def _finite_report_number(value: object, label: str) -> float:
    """Read a finite JSON number without interpreting booleans as coefficients."""

    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number.")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number.")
    return normalized


def validate_gaze_relation_report(
    report: Mapping, *, expected_contract: Mapping, expected_layers: list | None = None,
) -> None:
    """Check saved coefficients and summaries against the persisted run identity."""

    if not isinstance(report, Mapping) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("gaze_relation_parameters report requires schema_version 1.")
    contract = validate_gaze_relation_contract(report.get("contract"))
    expected = validate_gaze_relation_contract(expected_contract)
    if contract != expected:
        raise ValueError("gaze_relation_parameters contract disagrees with the saved gaze_relation.")
    enabled = expected["method"] != "none"
    if type(report.get("enabled")) is not bool or report["enabled"] != enabled:
        raise ValueError("gaze_relation_parameters enabled flag disagrees with the saved gaze_relation.")
    if report.get("input_units") != "ET2 TRT prediction units; not assumed milliseconds":
        raise ValueError("gaze_relation_parameters must identify ET2 TRT prediction units.")
    if report.get("phase_coefficient_units") != "radians per ET2 TRT prediction unit":
        raise ValueError("gaze_relation_parameters has invalid phase coefficient units.")
    layers = report.get("layers")
    metadata = report.get("installed_layers")
    if not isinstance(layers, list) or not isinstance(metadata, list) or len(layers) != len(metadata):
        raise ValueError("gaze_relation_parameters layers must match installed_layers metadata.")
    if bool(layers) != enabled:
        raise ValueError("gaze_relation_parameters must contain layers exactly when enabled.")
    if expected_layers is not None and metadata != expected_layers:
        raise ValueError("gaze_relation_parameters installed_layers disagree with the saved architecture.")
    matched_names = set()
    total_trainable = 0
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise ValueError("gaze_relation_parameters layer must be an object.")
        name = layer.get("name")
        if not isinstance(name, str) or not name.endswith(".gaze_relative_rotation"):
            raise ValueError("gaze_relation_parameters layer name must identify a rotation module.")
        attention_name = name.removesuffix(".gaze_relative_rotation")
        matches = [
            item for item in metadata
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            and (attention_name == item["name"] or attention_name.endswith("." + item["name"]))
        ]
        if len(matches) != 1 or matches[0]["name"] in matched_names:
            raise ValueError("gaze_relation_parameters must uniquely match installed layer metadata.")
        matched_names.add(matches[0]["name"])
        heads = layer.get("num_heads")
        if type(heads) is not int or heads <= 0 or type(matches[0].get("num_heads")) is not int or heads != matches[0]["num_heads"]:
            raise ValueError("gaze_relation_parameters num_heads disagrees with installed layer metadata.")
        alpha = layer.get("alpha")
        coefficient = layer.get("phase_coefficient_per_trt_unit")
        if not isinstance(alpha, list) or not isinstance(coefficient, list) or len(alpha) != heads or len(coefficient) != heads:
            raise ValueError("gaze_relation_parameters alpha and phase coefficients must match num_heads.")
        alpha = [_finite_report_number(value, "gaze_relation_parameters alpha") for value in alpha]
        coefficient = [_finite_report_number(value, "gaze_relation_parameters phase coefficient") for value in coefficient]
        scale = _finite_report_number(layer.get("scale"), "gaze_relation_parameters scale")
        if scale != expected["scale"]:
            raise ValueError("gaze_relation_parameters scale disagrees with the saved gaze_relation.")
        for value, alpha_value in zip(coefficient, alpha):
            if not math.isclose(value, alpha_value / scale, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError("gaze_relation_parameters phase coefficient disagrees with alpha / scale.")
        for field, expected_value in (
            ("alpha_l2_norm", math.hypot(*alpha)),
            ("alpha_max_abs", max(abs(value) for value in alpha)),
        ):
            value = _finite_report_number(layer.get(field), f"gaze_relation_parameters {field}")
            if not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(f"gaze_relation_parameters {field} disagrees with alpha.")
        trainable = layer.get("trainable_parameters")
        if type(trainable) is not int or trainable not in (0, heads):
            raise ValueError("gaze_relation_parameters trainable_parameters must be zero or num_heads.")
        total_trainable += trainable
    if type(report.get("total_trainable_parameters")) is not int or report["total_trainable_parameters"] != total_trainable:
        raise ValueError("gaze_relation_parameters total_trainable_parameters disagrees with layer counts.")

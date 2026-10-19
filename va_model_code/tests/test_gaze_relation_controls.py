"""Verify production TRT controls from CLI and packing through checkpoint reload."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from va_model_code.decoder_va import model as model_module
from va_model_code.decoder_va.gaze_relation import gaze_relation_contract, validate_gaze_relation_contract
from va_model_code.decoder_va.model import DecoderVARegressor, build_qwen_va_model, load_saved_decoder_va_model
from va_model_code.decoder_va.paths import condition_slug
from va_model_code.tests.test_gaze_attention import tiny_backbone
from va_model_code.tests.test_gaze_relation_model import SparseGazeProvider
from va_model_code.tests.test_train_model_contract import (
    _check_resume_pair,
    _expected_resume_manifest,
    train_model_module,
)


def _contract(seed=42):
    """Create an explicitly versioned, deterministic shuffled relation condition."""

    return gaze_relation_contract("trt-relative-rotation", input_mode="shuffled", shuffle_seed=seed)


def _inputs():
    """Use several distinct anchors and unequal sequence lengths."""

    return {
        "input_ids": torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15, 17], [2, 4, 6, 8, 10, 12, 14, 0, 0]]),
        "attention_mask": torch.tensor([[1] * 9, [1] * 7 + [0, 0]]),
    }


def test_shuffled_contract_roundtrips_and_keeps_v1_aligned_compatible():
    """Preserve the existing aligned contract and save every shuffle semantic."""

    aligned = gaze_relation_contract("trt-relative-rotation")
    shuffled = _contract()
    assert aligned["version"] == 1 and "input_mode" not in aligned
    assert shuffled["version"] == 2
    assert shuffled["input_mode"] == "shuffled"
    assert shuffled["shuffle_seed"] == 42
    assert shuffled["shuffle_algorithm"] == "blake2b_rank_v1"
    for contract in (aligned, shuffled, gaze_relation_contract()):
        assert validate_gaze_relation_contract(json.loads(json.dumps(contract))) == contract


@pytest.mark.parametrize("field,value", [
    ("shuffle_seed", True), ("shuffle_seed", -1), ("shuffle_seed", 2**63),
    ("shuffle_seed", 42.0), ("shuffle_seed", None),
    ("shuffle_algorithm", "random_each_call"), ("input_mode", "aligned"),
    ("version", 1), ("method", "none"),
])
def test_rejects_ambiguous_or_unsupported_shuffled_contract(field, value):
    """Fail closed when a saved condition cannot reproduce the same permutation."""

    with pytest.raises(ValueError):
        validate_gaze_relation_contract({**_contract(), field: value})


@pytest.mark.parametrize("seed", [None, 9])
def test_cli_resolves_shuffle_seed_and_output_names(seed):
    """The default control seed comes from the experiment seed, not a fold or batch."""

    options = ["qwen3.5-0.8b", "mse", "--seed", "43", "--gaze-relation", "trt-relative-rotation",
               "--gaze-relation-input", "shuffled"]
    if seed is not None:
        options += ["--gaze-relation-shuffle-seed", str(seed)]
    args = train_model_module._build_parser().parse_args(options)
    train_model_module._validate_args(args)
    contract = train_model_module._gaze_relation_config(args)
    assert contract["shuffle_seed"] == (43 if seed is None else seed)
    assert args.attn_implementation == "eager"
    common = dict(model="qwen", finetuning_mode="full", gaze_fusion="prefix-concat", gaze_features=("TRT",), seed=43)
    names = [condition_slug(**common, gaze_relation=item) for item in (
        gaze_relation_contract(), gaze_relation_contract("trt-relative-rotation"), _contract(9), _contract(43),
    )]
    assert len(set(names)) == 4
    assert "shuffled9" in names[2] and "shuffled43" in names[3]


@pytest.mark.parametrize("options", [
    ["--gaze-relation-input", "shuffled"],
    ["--gaze-relation", "trt-relative-rotation", "--gaze-relation-shuffle-seed", "42"],
    ["--gaze-relation", "trt-relative-rotation", "--gaze-relation-input", "shuffled",
     "--gaze-relation-shuffle-seed", "-1"],
])
def test_cli_rejects_unused_or_invalid_control_options(options):
    """Do not silently accept a control flag that cannot affect the selected arm."""

    args = train_model_module._build_parser().parse_args(["qwen3.5-0.8b", "mse", *options])
    with pytest.raises(ValueError):
        train_model_module._validate_args(args)


def test_control_changes_only_relation_input_and_preserves_native_initialization():
    """The text, real-TRT prefix, masks, native positions, and initial output remain paired."""

    models = []
    for contract in (gaze_relation_contract("trt-relative-rotation"), _contract()):
        torch.manual_seed(19)
        models.append(DecoderVARegressor(
            tiny_backbone(), gaze_provider=SparseGazeProvider(), gaze_relation=contract,
            gaze_projection_dim=8, gaze_projection_dropout=(0, 0), classifier_dropout=0,
        ).eval())
    inputs = _inputs()
    packed = [model._gaze_inputs(
        **inputs, text_embeddings=model.backbone.get_input_embeddings()(inputs["input_ids"]),
    ) for model in models]
    for aligned, shuffled in zip(packed[0][:4], packed[1][:4]):
        torch.testing.assert_close(aligned, shuffled, atol=0, rtol=0)
    aligned_relation, shuffled_relation = packed[0][4], packed[1][4]
    mask = aligned_relation["gaze_relation_mask"]
    assert torch.equal(mask, shuffled_relation["gaze_relation_mask"])
    assert not torch.equal(aligned_relation["gaze_relation_values"], shuffled_relation["gaze_relation_values"])
    for row in range(mask.shape[0]):
        assert torch.equal(
            aligned_relation["gaze_relation_values"][row, mask[row]].sort().values,
            shuffled_relation["gaze_relation_values"][row, mask[row]].sort().values,
        )
    torch.testing.assert_close(models[0](**inputs).logits, models[1](**inputs).logits, atol=0, rtol=0)
    for model in models:
        model.backbone.layers[1].self_attn.gaze_relative_rotation.alpha.data.fill_(0.3)
    assert not torch.equal(models[0](**inputs).logits, models[1](**inputs).logits)


def test_shuffled_native_model_reload_preserves_predictions(tmp_path, monkeypatch):
    """Reconstruction restores the saved control and learned phase without caller options."""

    monkeypatch.setattr(model_module, "load_qwen_backbone", lambda *args, **kwargs: tiny_backbone())
    monkeypatch.setattr(model_module, "ET2GazeProvider", SparseGazeProvider)
    source = build_qwen_va_model(
        SimpleNamespace(), dtype="float32", finetuning_mode="full", gaze_relation=_contract(),
        gaze_projection_dropout=(0, 0), classifier_dropout=0,
    ).eval()
    source.backbone.layers[1].self_attn.gaze_relative_rotation.alpha.data.fill_(0.2)
    source.save_architecture_manifest(tmp_path)
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    loaded, _ = load_saved_decoder_va_model(tmp_path, tokenizer=SimpleNamespace())
    assert loaded.gaze_relation == _contract()
    torch.testing.assert_close(source(**_inputs()).logits, loaded(**_inputs()).logits, atol=0, rtol=0)


@pytest.mark.parametrize("changed_contract", [gaze_relation_contract("trt-relative-rotation"), _contract(43)])
def test_resume_rejects_changed_relation_input(tmp_path, changed_contract):
    """An aligned/shuffled switch or a new permutation requires a fresh run."""

    expected = _expected_resume_manifest()
    expected.update(gaze_relation=_contract(), gaze_relation_learning_rate=1e-3,
                    gaze_relation_weight_decay=0.0, attn_implementation="eager")
    recorded = copy.deepcopy(expected)
    recorded["gaze_relation"] = changed_contract
    with pytest.raises(ValueError, match="gaze_relation"):
        _check_resume_pair(tmp_path, recorded, expected)

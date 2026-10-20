"""Validate TRT packing, full regression integration, and safe checkpoint contracts."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from va_model_code.decoder_va import model as model_module
from va_model_code.decoder_va.gaze_relation import gaze_relation_contract
from va_model_code.decoder_va.model import DecoderVARegressor, build_qwen_va_model, load_saved_decoder_va_model
from va_model_code.decoder_va.packing import pack_prefix_gaze
from va_model_code.tests.test_gaze_attention import tiny_backbone


class SparseGazeProvider:
    """Supply deterministic, sparse raw TRT including a genuinely valid zero."""

    feature_indices = (3,)
    repo_id = "test/et2"
    revision = "test-et2-revision"
    filename = "test.safetensors"

    def __init__(self, **kwargs):
        self.calls = 0
        for name in ("repo_id", "revision", "filename"):
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def compute(self, input_ids, attention_mask):
        """Mark even token positions as mapped without treating missing anchors as zero TRT."""

        self.calls += 1
        mask = attention_mask.bool() & (torch.arange(input_ids.shape[1], device=input_ids.device) % 2 == 0)
        values = (input_ids.float() - 1).clamp_min(0).unsqueeze(-1)
        return values, mask


def batch():
    """Build a padded batch with distinct gaze-prefix lengths."""

    return {"input_ids": torch.tensor([[1, 2, 4, 8, 3], [2, 3, 5, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])}


def test_sparse_gaze_packing_marks_original_text_anchors_only():
    """Offsets use each compact prefix length and preserve zero-valued valid anchors."""

    inputs = batch()
    provider = SparseGazeProvider()
    raw, mask = provider.compute(**inputs)
    packed = pack_prefix_gaze(
        torch.randn(2, 5, 8), inputs["attention_mask"], torch.randn(2, 5, 8),
        mask, torch.zeros(8), torch.zeros(8), text_gaze_values=raw[..., 0],
    )
    assert packed.text_gaze_mask.nonzero().tolist() == [[0, 5], [0, 7], [0, 9], [1, 4], [1, 6]]
    torch.testing.assert_close(packed.text_gaze_values[packed.text_gaze_mask], torch.tensor([0., 3., 2., 1., 4.]))
    assert torch.count_nonzero(packed.text_gaze_values[~packed.text_gaze_mask]) == 0
    assert packed.text_gaze_indices.tolist() == [[5, 7, 9], [4, 6, 0]]
    assert packed.text_gaze_anchor_mask.tolist() == [[True, True, True], [True, True, False]]


@pytest.mark.parametrize("bf16", [False, True])
def test_zero_relation_regressor_matches_raw_and_alpha_updates(bf16, monkeypatch):
    """Exercise real packing and the VA head with checkpoint replay and autocast."""

    torch.manual_seed(12)
    backbone = tiny_backbone()
    options = {"gaze_projection_dim": 8, "gaze_projection_dropout": (0, 0), "classifier_dropout": 0}
    torch.manual_seed(13)
    raw_model = DecoderVARegressor(copy.deepcopy(backbone), gaze_provider=SparseGazeProvider(), **options)
    torch.manual_seed(13)
    model = DecoderVARegressor(backbone, gaze_provider=SparseGazeProvider(),
                               gaze_relation=gaze_relation_contract("trt-relative-rotation"), **options)
    model.gradient_checkpointing_enable({"use_reentrant": False})
    rotation = model.backbone.layers[1].self_attn.gaze_relative_rotation
    original_forward = rotation.forward
    seen_layouts = []

    def capture_layout(*args, **kwargs):
        """Ensure real regression and checkpoint replay both use the compact implementation."""

        seen_layouts.append((kwargs["anchor_indices"], kwargs["anchor_mask"]))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(rotation, "forward", capture_layout)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        expected = raw_model(**batch()).logits
        actual = model(**batch()).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.gaze_provider.calls == 1
    loss = (actual - torch.tensor([[0.2, 0.8], [0.6, 0.1]])).square().mean()
    loss.backward()
    assert len(seen_layouts) >= 2
    for indices, mask in seen_layouts:
        assert indices.tolist() == [[5, 7, 9], [4, 6, 0]]
        assert mask.tolist() == [[True, True, True], [True, True, False]]
    alpha = model.backbone.layers[1].self_attn.gaze_relative_rotation.alpha
    assert alpha.dtype == torch.float32 and alpha.grad is not None
    assert torch.isfinite(alpha.grad).all() and torch.count_nonzero(alpha.grad) > 0
    optimizer = torch.optim.AdamW([alpha], lr=1e-3, weight_decay=0)
    optimizer.step()
    assert torch.count_nonzero(alpha) > 0
    assert model.trainable_parameter_summary()["gaze_relation_trainable_parameters"] == 4


def save_tiny_model(tmp_path, monkeypatch, enabled=True):
    """Construct a reloadable native model without pretrained downloads."""

    monkeypatch.setattr(model_module, "load_qwen_backbone", lambda *args, **kwargs: tiny_backbone())
    monkeypatch.setattr(model_module, "ET2GazeProvider", SparseGazeProvider)
    model = build_qwen_va_model(
        SimpleNamespace(), dtype="float32", finetuning_mode="full",
        gaze_relation=gaze_relation_contract("trt-relative-rotation" if enabled else "none"),
        gaze_projection_dropout=(0, 0), classifier_dropout=0,
    )
    if enabled:
        model.backbone.layers[1].self_attn.gaze_relative_rotation.alpha.data.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    model.save_architecture_manifest(tmp_path)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    return model.eval()


def test_relation_roundtrip_preserves_predictions_and_contract(tmp_path, monkeypatch):
    """Nonzero learned alpha is reconstructed with the native eager backend before installation."""

    source = save_tiny_model(tmp_path, monkeypatch)
    loaded, _ = load_saved_decoder_va_model(tmp_path, tokenizer=SimpleNamespace())
    torch.testing.assert_close(source(**batch()).logits, loaded(**batch()).logits, rtol=0, atol=0)
    assert loaded._reconstruction_config["attn_implementation"] == "eager"
    assert loaded.gaze_relation == source.gaze_relation


@pytest.mark.parametrize("corruption", ["missing_metadata", "mismatched_scale", "missing_alpha", "nan_alpha", "wrong_shape"])
def test_relation_checkpoint_corruption_is_rejected(tmp_path, monkeypatch, corruption):
    """Reject changes that would silently disable or reinterpret a learned relation."""

    model = save_tiny_model(tmp_path, monkeypatch)
    path = tmp_path / "decoder_va_architecture.json"
    manifest = json.loads(path.read_text())
    state = model.state_dict()
    alpha_key = next(key for key in state if key.endswith("gaze_relative_rotation.alpha"))
    if corruption == "missing_metadata":
        del manifest["gaze_relation"]
    elif corruption == "mismatched_scale":
        manifest["gaze_relation"]["scale"] = 2.0
    elif corruption == "missing_alpha":
        del state[alpha_key]
    elif corruption == "nan_alpha":
        state[alpha_key].fill_(float("nan"))
    else:
        state[alpha_key] = torch.zeros(3)
    path.write_text(json.dumps(manifest))
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="gaze_relation"):
        load_saved_decoder_va_model(tmp_path, tokenizer=SimpleNamespace())


def test_historical_schema8_without_relation_still_loads(tmp_path, monkeypatch):
    """Preserve evaluation of existing precision-safe raw gaze checkpoints."""

    source = save_tiny_model(tmp_path, monkeypatch, enabled=False)
    path = tmp_path / "decoder_va_architecture.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 8
    del manifest["gaze_relation"]
    del manifest["reconstruction"]["gaze_relation"]
    path.write_text(json.dumps(manifest))
    loaded, _ = load_saved_decoder_va_model(tmp_path, tokenizer=SimpleNamespace())
    torch.testing.assert_close(source(**batch()).logits, loaded(**batch()).logits, rtol=0, atol=0)

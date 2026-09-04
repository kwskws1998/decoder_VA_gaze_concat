"""Integration and checkpoint invariants for opt-in gaze redistribution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from va_model_code.decoder_va import model as model_module
from va_model_code.decoder_va.model import (
    ARCHITECTURE_MANIFEST_FILENAME,
    ARCHITECTURE_MANIFEST_VERSION,
    SAFE_WEIGHTS_FILENAME,
    DecoderVARegressor,
    build_qwen_va_model,
    load_saved_decoder_va_model,
)
from va_model_code.decoder_va.redistribution import (
    redistribution_contract,
    validate_redistribution_state_file,
)
from va_model_code.tests.test_model_loss_evaluation import (
    FakeCausalBackbone,
    FakeGazeProvider,
)


class CachedGazeProvider:
    """Return the exact same frozen raw tensor on consecutive cache hits."""

    def __init__(self, raw, mask, feature_indices=(3,)):
        self.raw = raw
        self.mask = mask
        self.feature_indices = tuple(feature_indices)
        self.calls = 0

    def compute(self, input_ids, attention_mask):
        """Expose cache objects directly so mutations remain observable to tests."""

        assert input_ids.shape == self.raw.shape[:2]
        self.calls += 1
        return self.raw, self.mask


def _direct_model(provider, *, config=None, enabled=True):
    """Create a small real fusion/head path without downloading a backbone."""

    return DecoderVARegressor(
        FakeCausalBackbone(),
        gaze_provider=provider,
        gaze_fusion="prefix-concat",
        gaze_redistribution=(
            config or redistribution_contract("asym-gaussian", init_sigma_left=0.7, init_sigma_right=1.6)
            if enabled
            else {"method": "none"}
        ),
        gaze_projection_dim=8,
        gaze_projection_dropout=(0.0, 0.0),
        classifier_dropout=0.0,
    )


def _batch(length=5):
    """Return a deterministic one-row text batch whose first token lacks gaze."""

    return {
        "input_ids": torch.arange(1, length + 1).unsqueeze(0),
        "attention_mask": torch.ones(1, length, dtype=torch.long),
    }


@pytest.fixture
def fake_checkpoint_factory(monkeypatch):
    """Build saveable models using the real construction and lazy ET2 metadata."""

    monkeypatch.setattr(model_module, "load_qwen_backbone", lambda *args, **kwargs: FakeCausalBackbone())

    def build(*, enabled=True, mode="full", gaze_fusion="prefix-concat"):
        """Construct a mode-specific checkpoint with no remote inference."""

        tokenizer = SimpleNamespace(padding_side="right")
        model = build_qwen_va_model(
            tokenizer,
            model_id="Qwen/fake-redistribution",
            model_revision="fixed-decoder-commit",
            et_revision="fixed-et-commit",
            dtype=torch.float32,
            finetuning_mode=mode,
            gaze_fusion=gaze_fusion,
            gaze_redistribution=redistribution_contract("asym-gaussian" if enabled else "none"),
            gaze_feature_indices=(0, 3),
            gaze_projection_dim=8,
            gaze_projection_dropout=(0.0, 0.0),
            classifier_dropout=0.0,
        )
        return model, tokenizer

    return build


def _save(model, directory):
    """Write the exact complete model state and architecture used by the loader."""

    directory.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), directory / SAFE_WEIGHTS_FILENAME, metadata={"format": "pt"})
    model.save_architecture_manifest(directory)
    path = directory / ARCHITECTURE_MANIFEST_FILENAME
    return path, json.loads(path.read_text(encoding="utf-8"))


def _deny_backbone_load(monkeypatch):
    """Make validation-order failures observable without any weight download."""

    def fail(*args, **kwargs):
        """Fail whenever invalid metadata reaches the model-loading boundary."""

        pytest.fail("Invalid redistribution configuration reached backbone loading.")

    monkeypatch.setattr(model_module, "load_qwen_backbone", fail)


def test_redistribution_precedes_projection_and_uses_sparse_gaze_mask():
    raw = torch.tensor([[[9.0, float("nan")], [2.0, 1.0], [8.0, float("inf")], [3.0, 0.0], [4.0, -2.0]]])
    gaze_mask = torch.tensor([[0, 1, 0, 1, 1]], dtype=torch.bool)
    provider = CachedGazeProvider(raw, gaze_mask, (0, 3))
    model = _direct_model(provider).eval()
    seen = []
    hook = model.gaze_projector.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    output = model(**_batch())
    hook.remove()
    expected = model.gaze_redistributor.kernel(raw[:, :, 1], gaze_mask)
    torch.testing.assert_close(seen[0][:, :, 1], expected)
    torch.testing.assert_close(seen[0][gaze_mask][:, 0], raw[gaze_mask][:, 0])
    assert seen[0][0, 3, 1] != 0
    assert seen[0][0, 0, 1] == 0
    assert seen[0][0, 2, 1] == 0
    assert torch.isfinite(output.logits).all()
    assert raw[0, 2, 1].isinf()
    assert raw[0, 0, 1].isnan()


@pytest.mark.parametrize("freeze_backbone", [False, True])
def test_train_steps_update_sigmas_without_mutating_cached_gaze(freeze_backbone):
    torch.manual_seed(137)
    with torch.inference_mode():
        raw = torch.tensor([[[0.0], [0.2], [3.0], [0.0], [1.0]]])
    mask = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool)
    original = raw.clone()
    provider = CachedGazeProvider(raw, mask)
    model = _direct_model(provider)
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    initial = {name: value.detach().clone() for name, value in model.gaze_redistributor.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(**_batch()).logits
        loss = (logits - torch.tensor([[0.1, 0.9]])).square().mean()
        loss.backward()
        for parameter in model.gaze_redistributor.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        optimizer.step()
        assert provider.raw is raw
        assert torch.equal(raw, original)
        assert raw.requires_grad is False
        assert torch.is_inference(raw)
    assert provider.calls == 2
    assert all(not torch.equal(parameter.detach(), initial[name]) for name, parameter in model.gaze_redistributor.named_parameters())
    assert model.trainable_parameter_summary()["gaze_redistribution_trainable_parameters"] == 2
    if freeze_backbone:
        assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_none_preserves_parameter_keys_rng_and_logits():
    torch.manual_seed(111)
    implicit = DecoderVARegressor(FakeCausalBackbone(), gaze_provider=FakeGazeProvider(), gaze_fusion="prefix-concat")
    implicit_rng = torch.get_rng_state().clone()
    torch.manual_seed(111)
    explicit = DecoderVARegressor(FakeCausalBackbone(), gaze_provider=FakeGazeProvider(), gaze_fusion="prefix-concat", gaze_redistribution={"method": "none"})
    assert torch.equal(torch.get_rng_state(), implicit_rng)
    assert explicit.gaze_redistributor is None
    assert not any("redistributor" in name for name in explicit.state_dict())
    assert explicit.state_dict().keys() == implicit.state_dict().keys()
    for name, parameter in explicit.state_dict().items():
        assert torch.equal(parameter, implicit.state_dict()[name])
    explicit.eval()
    implicit.eval()
    torch.testing.assert_close(explicit(**_batch()).logits, implicit(**_batch()).logits, atol=0, rtol=0)


def test_enabled_constants_preserve_rng_and_all_other_initial_weights():
    torch.manual_seed(191)
    disabled = _direct_model(FakeGazeProvider(), enabled=False)
    disabled_rng = torch.get_rng_state().clone()
    torch.manual_seed(191)
    enabled = _direct_model(FakeGazeProvider())
    assert torch.equal(torch.get_rng_state(), disabled_rng)
    extra = set(enabled.state_dict()).difference(disabled.state_dict())
    assert extra == {"gaze_redistributor.kernel.log_sigma_left", "gaze_redistributor.kernel.log_sigma_right"}
    for name, value in disabled.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[name])
    assert enabled.trainable_parameter_summary()["total_parameters"] == disabled.trainable_parameter_summary()["total_parameters"] + 2


@pytest.mark.parametrize("fusion,indices,error", [("none", None, "requires prefix-concat"), ("prefix-concat", (0, 1), "requires the TRT")])
def test_invalid_enabled_build_rejected_before_backbone(monkeypatch, fusion, indices, error):
    _deny_backbone_load(monkeypatch)
    with pytest.raises(ValueError, match=error):
        build_qwen_va_model(SimpleNamespace(), gaze_fusion=fusion, gaze_feature_indices=indices, gaze_redistribution=redistribution_contract("asym-gaussian"))


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_v7_checkpoint_roundtrip_restores_learned_sigmas_and_predictions(tmp_path, fake_checkpoint_factory, mode):
    model, tokenizer = fake_checkpoint_factory(mode=mode)
    with torch.no_grad():
        model.gaze_redistributor.kernel.log_sigma_left.fill_(-0.45)
        model.gaze_redistributor.kernel.log_sigma_right.fill_(0.85)
    _, manifest = _save(model, tmp_path)
    assert manifest["schema_version"] == ARCHITECTURE_MANIFEST_VERSION == 7
    assert manifest["gaze_redistribution"] == manifest["reconstruction"]["gaze_redistribution"]
    assert manifest["gaze_redistribution"]["init_sigma_left"] == 1.0
    assert manifest["gaze_redistribution_trainable_parameters"] == 2
    loaded, _ = load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)
    for name, expected in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected)
    model.gaze_provider = FakeGazeProvider((0, 3))
    loaded.gaze_provider = FakeGazeProvider((0, 3))
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(**_batch()).logits, model(**_batch()).logits, atol=0, rtol=0)


@pytest.mark.parametrize("owner", ["architecture", "reconstruction"])
@pytest.mark.parametrize("change", ["remove", "width", "method", "position_space"])
def test_v7_contract_tampering_rejected_before_backbone(tmp_path, fake_checkpoint_factory, monkeypatch, owner, change):
    model, tokenizer = fake_checkpoint_factory()
    path, manifest = _save(model, tmp_path)
    target = manifest if owner == "architecture" else manifest["reconstruction"]
    if change == "remove":
        del target["gaze_redistribution"]
    elif change == "width":
        target["gaze_redistribution"]["init_sigma_left"] = 2.0
    elif change == "method":
        target["gaze_redistribution"] = {"method": "none"}
    else:
        target["gaze_redistribution"]["position_space"] = "compacted_words"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _deny_backbone_load(monkeypatch)
    with pytest.raises(ValueError, match="gaze_redistribution|redistribution"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)


@pytest.mark.parametrize("version,mode", [(5, "lora"), (6, "lora"), (6, "full")])
@pytest.mark.parametrize("fusion", ["none", "prefix-concat"])
def test_legacy_without_metadata_restores_none(tmp_path, fake_checkpoint_factory, version, mode, fusion):
    model, tokenizer = fake_checkpoint_factory(enabled=False, mode=mode, gaze_fusion=fusion)
    path, manifest = _save(model, tmp_path)
    manifest["schema_version"] = version
    del manifest["gaze_redistribution"]
    del manifest["reconstruction"]["gaze_redistribution"]
    manifest.pop("gaze_redistribution_trainable_parameters")
    if version == 5:
        del manifest["finetuning_mode"]
        del manifest["reconstruction"]["finetuning_mode"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    restored, _ = load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)
    assert restored.gaze_redistribution == {"method": "none"}
    assert restored.gaze_redistributor is None
    for name, expected in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected)


@pytest.mark.parametrize("version", [5, 6])
@pytest.mark.parametrize("owner", ["architecture", "reconstruction"])
def test_legacy_cannot_enable_redistribution(tmp_path, fake_checkpoint_factory, monkeypatch, version, owner):
    model, tokenizer = fake_checkpoint_factory(enabled=False, mode="lora")
    path, manifest = _save(model, tmp_path)
    manifest["schema_version"] = version
    target = manifest if owner == "architecture" else manifest["reconstruction"]
    target["gaze_redistribution"] = redistribution_contract("asym-gaussian")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _deny_backbone_load(monkeypatch)
    with pytest.raises(ValueError, match="Legacy manifests cannot enable"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)


@pytest.mark.parametrize("side", ["left", "right"])
def test_missing_sigma_weights_fail_strict_checkpoint_loading(tmp_path, fake_checkpoint_factory, monkeypatch, side):
    model, tokenizer = fake_checkpoint_factory()
    _save(model, tmp_path)
    state = dict(model.state_dict())
    del state[f"gaze_redistributor.kernel.log_sigma_{side}"]
    save_file(state, tmp_path / SAFE_WEIGHTS_FILENAME)
    _deny_backbone_load(monkeypatch)
    with pytest.raises(ValueError, match="state keys mismatch"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "replacement,error",
    [
        (torch.tensor([0.0]), "must be scalar"),
        (torch.empty(0), "must be scalar"),
        (torch.tensor(0.0, dtype=torch.bfloat16), "must be FP32"),
        (torch.tensor(0.0, dtype=torch.float64), "must be FP32"),
        (torch.tensor(0, dtype=torch.int64), "must be FP32"),
        (torch.tensor(float("nan")), "must be finite"),
        (torch.tensor(float("inf")), "must be finite"),
        (torch.tensor(-float("inf")), "must be finite"),
    ],
)
def test_corrupt_sigma_fails_before_backbone_loading(tmp_path, fake_checkpoint_factory, monkeypatch, side, replacement, error):
    model, tokenizer = fake_checkpoint_factory()
    _save(model, tmp_path)
    state = dict(model.state_dict())
    state[f"gaze_redistributor.kernel.log_sigma_{side}"] = replacement
    save_file(state, tmp_path / SAFE_WEIGHTS_FILENAME)
    _deny_backbone_load(monkeypatch)
    with pytest.raises(ValueError, match=error):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)


def test_extra_redistributor_state_fails_before_backbone_loading(tmp_path, fake_checkpoint_factory, monkeypatch):
    model, tokenizer = fake_checkpoint_factory()
    _save(model, tmp_path)
    state = dict(model.state_dict())
    state["gaze_redistributor.extra"] = torch.tensor(0.0)
    save_file(state, tmp_path / SAFE_WEIGHTS_FILENAME)
    _deny_backbone_load(monkeypatch)
    with pytest.raises(ValueError, match="state keys mismatch"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)


def test_state_inspector_rejects_malformed_safetensors(tmp_path):
    weights_path = tmp_path / "model.safetensors"
    weights_path.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="Cannot safely inspect gaze_redistribution"):
        validate_redistribution_state_file(weights_path)


def test_state_inspector_validates_all_shapes_before_materializing_tensors(monkeypatch):
    import safetensors

    class LazyWeights:
        """Expose an invalid right width and disallow materializing any tensor."""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return ["backbone.large_tensor", "gaze_redistributor.kernel.log_sigma_left", "gaze_redistributor.kernel.log_sigma_right"]

        def get_slice(self, key):
            return SimpleNamespace(get_shape=lambda: [100000] if key.endswith("right") else [])

        def get_tensor(self, key):
            pytest.fail(f"Materialized tensor before validating all shapes: {key}")

    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: LazyWeights())
    with pytest.raises(ValueError, match="must be scalar"):
        validate_redistribution_state_file("not-opened.safetensors")


def test_enabled_metadata_cannot_be_stripped_to_silently_disable_weights(tmp_path, fake_checkpoint_factory):
    model, tokenizer = fake_checkpoint_factory()
    path, manifest = _save(model, tmp_path)
    manifest["gaze_redistribution"] = {"method": "none"}
    manifest["reconstruction"]["gaze_redistribution"] = {"method": "none"}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unexpected key|unexpected key"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)


@pytest.mark.parametrize("features", [(3,), (0, 3)])
def test_masked_nonfinite_features_have_finite_forward_and_backward(features):
    torch.manual_seed(157)
    raw = torch.tensor([[[float("nan")], [1.0], [float("inf")], [2.0], [float("nan")]]]).expand(-1, -1, len(features)).clone()
    mask = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.bool)
    provider = CachedGazeProvider(raw, mask, features)
    model = _direct_model(provider)
    batch = _batch()
    batch["attention_mask"][0, -1] = 0
    output = model(**batch).logits
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all(), name


def test_masked_padding_does_not_change_model_predictions():
    torch.manual_seed(312)
    short_raw = torch.tensor([[[0.0], [1.0], [0.0], [-2.0]]])
    short_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.bool)
    provider = CachedGazeProvider(short_raw, short_mask)
    model = _direct_model(provider).eval()
    expected = model(**_batch(4)).logits
    provider.raw = torch.cat([short_raw, torch.full((1, 2, 1), float("nan"))], dim=1)
    provider.mask = torch.cat([short_mask, torch.zeros(1, 2, dtype=torch.bool)], dim=1)
    batch = _batch(6)
    batch["attention_mask"][:, 4:] = 0
    actual = model(**batch).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_enabled_gaze_cannot_select_a_text_padding_position():
    model = _direct_model(CachedGazeProvider(torch.ones(1, 5, 1), torch.tensor([[0, 1, 1, 1, 1]])))
    batch = _batch()
    batch["attention_mask"][0, -1] = 0
    with pytest.raises(ValueError, match="valid text positions"):
        model(**batch)


def test_cpu_bf16_autocast_preserves_fp32_sigma_gradients():
    torch.manual_seed(82)
    model = _direct_model(CachedGazeProvider(torch.tensor([[[0.0], [1.0], [3.0], [0.0], [-2.0]]]), torch.tensor([[0, 1, 1, 1, 1]])))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(**_batch()).logits
        loss = (output.float() - torch.tensor([[0.1, 0.9]])).square().mean()
    loss.backward()
    for parameter in model.gaze_redistributor.parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs() > 0

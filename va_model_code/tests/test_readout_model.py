"""Verify learned readout visibility, causal isolation, gradients, and safe reconstruction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from va_model_code.decoder_va import model as model_module
from va_model_code.decoder_va.gaze_relation import gaze_relation_contract, validate_gaze_relation_contract
from va_model_code.decoder_va.model import DecoderVARegressor, build_qwen_va_model, load_saved_decoder_va_model
from va_model_code.decoder_va.packing import append_readout_token, pack_prefix_gaze
from va_model_code.decoder_va.readout import readout_contract, saved_readout_contract, validate_readout_contract
from va_model_code.tests.test_gaze_attention import tiny_backbone


class SingletonGazeProvider:
    """Map only the first subword, including singleton words ending in punctuation."""

    feature_indices = (3,)
    repo_id = "test/et2"
    revision = "test-et2-revision"
    filename = "test.safetensors"

    def __init__(self, **kwargs):
        for name in ("repo_id", "revision", "filename"):
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def compute(self, input_ids, attention_mask):
        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        mask[:, 0] = True
        return input_ids.float().unsqueeze(-1) / 10.0, mask


def inputs():
    """Include punctuation/subword endings, distinct lengths, and right padding."""

    return {
        "input_ids": torch.tensor([[3, 4, 6, 7], [2, 8, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
    }


def make_model(*, enabled=True, readout="learned-token", fusion="prefix-concat"):
    """Build exactly matched tiny Qwen models under the same initialization seed."""

    torch.manual_seed(930)
    return DecoderVARegressor(
        tiny_backbone(), gaze_provider=SingletonGazeProvider() if fusion != "none" else None,
        gaze_fusion=fusion,
        gaze_relation=gaze_relation_contract("trt-readout-rotation" if enabled else "none"),
        readout=readout_contract(readout), gaze_projection_dim=8,
        gaze_projection_dropout=(0, 0), classifier_dropout=0,
    )


def test_readout_packing_uses_per_row_lengths_and_zero_reference():
    """Readout cannot occupy padding, prefix, or another row's absolute position."""

    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    gaze_mask = torch.tensor([[1, 0, 1], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    values = torch.tensor([[0.4, 99, 0.7], [0.2, 99, 99], [99, 99, 99]])
    source = pack_prefix_gaze(torch.randn(3, 3, 8), mask, torch.randn(3, 3, 8), gaze_mask,
                              torch.zeros(8), torch.zeros(8), text_gaze_values=values)
    token = torch.randn(8, requires_grad=True)
    packed = append_readout_token(source, token, relation_reference=True)
    assert packed.pooling_positions.tolist() == [7, 5, 3]
    assert packed.text_gaze_indices.tolist() == [[4, 6, 7], [3, 5, 0], [3, 0, 0]]
    assert packed.text_gaze_anchor_mask.tolist() == [[True, True, True], [True, True, False], [True, False, False]]
    assert packed.text_gaze_mask.nonzero().tolist() == [[0, 4], [0, 6], [0, 7], [1, 3], [1, 5], [2, 3]]
    torch.testing.assert_close(packed.inputs_embeds[torch.arange(3), packed.pooling_positions], token.expand(3, -1))
    assert torch.count_nonzero(packed.inputs_embeds[~packed.attention_mask.bool()]) == 0
    assert torch.count_nonzero(packed.text_gaze_values[torch.arange(3), packed.pooling_positions]) == 0
    packed.inputs_embeds.sum().backward()
    torch.testing.assert_close(token.grad, torch.full_like(token, 3.0))


@pytest.mark.parametrize("bf16", [False, True])
def test_zero_alpha_exactly_matches_readout_raw_and_singleton_final_alpha_learns(bf16):
    """A single positive TRT has a direct final-attention path even with punctuation."""

    model, raw = make_model(), make_model(enabled=False)
    for name, value in raw.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        prediction = model(**inputs()).logits
        baseline = raw(**inputs()).logits
    torch.testing.assert_close(prediction, baseline, rtol=0, atol=0)
    loss = (prediction - torch.tensor([[0.2, 0.8], [0.6, 0.1]])).square().mean()
    loss.backward()
    alpha = model.backbone.layers[-1].self_attn.gaze_relative_rotation.alpha
    assert alpha.dtype == torch.float32
    assert torch.isfinite(alpha.grad).all() and torch.count_nonzero(alpha.grad) > 0
    assert model.readout_token.dtype == torch.float32
    assert torch.isfinite(model.readout_token.grad).all() and torch.count_nonzero(model.readout_token.grad) > 0
    before = prediction.detach().clone()
    with torch.no_grad():
        alpha.fill_(0.5)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        changed = model(**inputs()).logits
    assert not torch.equal(changed, before)


def test_readout_batch_padding_does_not_change_predictions():
    """Readout pooling and position IDs are independent of padding width and batch mates."""

    model = make_model().eval()
    with torch.no_grad():
        model.backbone.layers[-1].self_attn.gaze_relative_rotation.alpha.fill_(0.3)
        batched = model(**inputs()).logits
        for row, length in enumerate((4, 2)):
            single = {name: value[row:row+1, :length] for name, value in inputs().items()}
            torch.testing.assert_close(model(**single).logits, batched[row:row+1], atol=1e-6, rtol=1e-6)
        dirty = inputs()
        dirty["input_ids"][1, 2:] = 31
        torch.testing.assert_close(model(**dirty).logits, batched, atol=0, rtol=0)


def test_readout_cannot_influence_preceding_text_hidden_states():
    """The appended trainable token cannot leak backwards through causal attention."""

    model = make_model().eval()
    batch = inputs()
    embeddings = model.get_input_embeddings()(batch["input_ids"])
    packed, mask, positions, pooling, relation = model._gaze_inputs(**batch, text_embeddings=embeddings)
    with torch.no_grad():
        before = model.backbone(inputs_embeds=packed, attention_mask=mask, position_ids=positions, **relation).last_hidden_state
        perturbed = packed.clone()
        perturbed[torch.arange(2), pooling] += torch.linspace(-2, 2, model.hidden_size)
        after = model.backbone(inputs_embeds=perturbed, attention_mask=mask, position_ids=positions, **relation).last_hidden_state
    for row, position in enumerate(pooling):
        torch.testing.assert_close(before[row, :position], after[row, :position], atol=0, rtol=0)
        assert not torch.equal(before[row, position], after[row, position])


def test_readout_checkpoint_recomputation_preserves_gradients():
    """Recomputed final attention receives the same readout anchor metadata."""

    ordinary, replay = make_model(), make_model()
    replay.gradient_checkpointing_enable({"use_reentrant": False})
    for model in (ordinary, replay):
        model.backbone.layers[-1].self_attn.gaze_relative_rotation.alpha.data.fill_(0.2)
        model(**inputs()).logits.square().mean().backward()
    for (name, reference), (other_name, actual) in zip(ordinary.named_parameters(), replay.named_parameters()):
        assert name == other_name
        assert (reference.grad is None) == (actual.grad is None)
        if reference.grad is not None:
            torch.testing.assert_close(reference.grad, actual.grad, atol=2e-6, rtol=2e-5)


def save_readout(tmp_path, monkeypatch, *, enabled=True, readout="learned-token"):
    """Create native safetensors and architecture metadata without network access."""

    monkeypatch.setattr(model_module, "load_qwen_backbone", lambda *args, **kwargs: tiny_backbone())
    monkeypatch.setattr(model_module, "ET2GazeProvider", SingletonGazeProvider)
    source = build_qwen_va_model(
        SimpleNamespace(), dtype="float32", finetuning_mode="full",
        gaze_relation=gaze_relation_contract("trt-readout-rotation" if enabled else "none"),
        readout=readout_contract(readout), gaze_projection_dropout=(0, 0), classifier_dropout=0,
    ).eval()
    source.save_architecture_manifest(tmp_path)
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    return source


@pytest.mark.parametrize("enabled", [False, True])
def test_readout_safe_roundtrip(tmp_path, monkeypatch, enabled):
    """Raw and rotation models reload the identical learned token and predictions."""

    source = save_readout(tmp_path, monkeypatch, enabled=enabled)
    restored, _ = load_saved_decoder_va_model(tmp_path, tokenizer=SimpleNamespace())
    torch.testing.assert_close(source(**inputs()).logits, restored(**inputs()).logits, rtol=0, atol=0)
    assert restored.readout == readout_contract("learned-token")


@pytest.mark.parametrize("corruption", ["missing_contract", "mismatched_contract", "missing_weight", "shape", "nan", "dtype"])
def test_readout_corruption_rejected(tmp_path, monkeypatch, corruption):
    """Saved readout cannot silently disappear, change semantics, or be malformed."""

    source = save_readout(tmp_path, monkeypatch)
    path = tmp_path / "decoder_va_architecture.json"
    metadata = json.loads(path.read_text())
    state = source.state_dict()
    if corruption == "missing_contract":
        del metadata["readout"]
    elif corruption == "mismatched_contract":
        metadata["readout"] = readout_contract()
    elif corruption == "missing_weight":
        del state["readout_token"]
    elif corruption == "shape":
        state["readout_token"] = torch.zeros(2)
    elif corruption == "nan":
        state["readout_token"].fill_(float("nan"))
    else:
        state["readout_token"] = state["readout_token"].bfloat16()
    path.write_text(json.dumps(metadata))
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="readout"):
        load_saved_decoder_va_model(tmp_path, tokenizer=SimpleNamespace())


@pytest.mark.parametrize("version", [8, 9])
def test_legacy_readout_absence_preserves_old_predictions(tmp_path, monkeypatch, version):
    """Missing historical readout metadata never introduces a trainable extra token."""

    source = save_readout(tmp_path, monkeypatch, enabled=False, readout="last-token")
    path = tmp_path / "decoder_va_architecture.json"
    metadata = json.loads(path.read_text())
    metadata["schema_version"] = version
    del metadata["readout"]
    del metadata["reconstruction"]["readout"]
    path.write_text(json.dumps(metadata))
    restored, _ = load_saved_decoder_va_model(tmp_path, tokenizer=SimpleNamespace())
    assert restored.readout_token is None
    torch.testing.assert_close(source(**inputs()).logits, restored(**inputs()).logits, rtol=0, atol=0)


def test_text_only_learned_readout_is_supported():
    """The matched learned-token architecture can also serve as a text-only control."""

    model = make_model(enabled=False, fusion="none")
    predictions = model(**inputs()).logits
    predictions.sum().backward()
    assert model.readout_token.grad is not None
    assert model.config.pooling_position == "learned_readout_token_after_text"


@pytest.mark.parametrize("method,mode", [("trt-readout-rotation", "last-token"), ("trt-relative-rotation", "learned-token")])
def test_incompatible_relation_readout_modes_fail(method, mode):
    """A legacy relative-rotation checkpoint cannot acquire readout semantics implicitly."""

    with pytest.raises(ValueError, match="readout"):
        DecoderVARegressor(tiny_backbone(), gaze_provider=SingletonGazeProvider(),
                           readout=readout_contract(mode), gaze_relation=gaze_relation_contract(method))


@pytest.mark.parametrize("shuffled", [False, True])
def test_readout_relation_contract_is_distinct_and_strict(shuffled):
    """The zero-reference scope is versioned separately from mapped-text-only rotation."""

    contract = gaze_relation_contract("trt-readout-rotation", input_mode="shuffled" if shuffled else "aligned",
                                     shuffle_seed=42 if shuffled else None)
    assert contract["version"] == 3
    assert contract["scope"] == "mapped_text_pairs_and_readout"
    assert validate_gaze_relation_contract(contract) == contract
    changed = dict(contract, readout_reference_trt=1.0)
    with pytest.raises(ValueError):
        validate_gaze_relation_contract(changed)
    changed = dict(contract, version=2)
    with pytest.raises(ValueError):
        validate_gaze_relation_contract(changed)


def test_readout_contract_rejects_old_schema_enablement_and_missing_current():
    """Every new manifest states its readout mode while legacy missing means last-token."""

    assert saved_readout_contract({}, schema_version=9, context="test") == readout_contract()
    with pytest.raises(ValueError):
        saved_readout_contract({}, schema_version=10, context="test")
    with pytest.raises(ValueError):
        saved_readout_contract({"readout": readout_contract("learned-token")}, schema_version=9, context="test")
    with pytest.raises(ValueError):
        validate_readout_contract({"version": True, "mode": "last-token"})


@pytest.mark.parametrize("has_anchor", [False, True])
def test_empty_or_zero_trt_preserves_exact_raw_prediction(has_anchor):
    """A reference alone or an equal zero-TRT pair naturally has zero rotation effect."""

    raw, model = make_model(enabled=False), make_model()

    def zero_provider(input_ids, attention_mask):
        """Return either no gaze anchor or one genuine zero-valued gaze anchor."""

        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        mask[:, 0] = has_anchor
        return torch.zeros_like(input_ids, dtype=torch.float32).unsqueeze(-1), mask

    for instance in (raw, model):
        instance.gaze_provider.compute = zero_provider
    model.backbone.layers[-1].self_attn.gaze_relative_rotation.alpha.data.fill_(0.7)
    reference, actual = raw(**inputs()).logits, model(**inputs()).logits
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.sum().backward()
    alpha = model.backbone.layers[-1].self_attn.gaze_relative_rotation.alpha
    assert alpha.grad is not None and torch.count_nonzero(alpha.grad) == 0

"""Offline raw ET alignment, cache, gradient checkpointing and decoder integration."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from va_model_code.decoder_va.model import DecoderVARegressor
from va_model_code.tests.test_alignment_spans import punctuation_fixture
from va_model_code.tests.test_gaze import FakeET2GazeProvider


def tiny_native_qwen_backbone(dtype):
    """Exercise Qwen3.5 linear and full attention without model files or downloads."""

    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "default", "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2],
        },
    )
    config._attn_implementation = "eager"
    return Qwen3_5TextModel(config).to(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_corrected_et_compute_cache_and_checkpointed_decoder_projector_updates(dtype):
    """Train the raw TRT projector through real provider compute with offline ET assets."""

    torch.manual_seed(17)
    tokenizer = punctuation_fixture()
    provider = FakeET2GazeProvider(tokenizer)
    input_ids = torch.tensor([tokenizer.ids + [31, 31]] * 2)
    attention_mask = torch.tensor([[1] * 8 + [0, 0]] * 2)
    expected_raw = torch.tensor([10.0, 13.0, 16.0, 0.0, 19.0, 22.0, 25.0, 0.0, 0.0, 0.0])
    expected_raw = expected_raw[None, :, None].expand(2, -1, -1)
    expected_mask = torch.tensor([[True, True, True, False, True, True, True, False, False, False]] * 2)

    raw, gaze_mask = provider.compute(input_ids, attention_mask)

    torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
    assert torch.equal(gaze_mask, expected_mask)
    assert provider._et_tokenizer.word_rows_seen == [[["I", "love", "this!!!", "I", "hate", "that!"]]]
    assert provider.fake_model.batch_sizes == [1]
    assert provider.load_count == 1
    assert not provider.fake_model.scale.requires_grad
    invalid_padding_mask = attention_mask.clone()
    invalid_padding_mask[:, 2] = 0
    with pytest.raises(ValueError, match="contiguous right padding"):
        provider.compute(input_ids, invalid_padding_mask)
    cached_raw, cached_mask = provider.compute(input_ids, attention_mask)
    torch.testing.assert_close(cached_raw, expected_raw, rtol=0, atol=0)
    assert torch.equal(cached_mask, gaze_mask)
    assert provider.fake_model.batch_sizes == [1]

    model = DecoderVARegressor(
        tiny_native_qwen_backbone(dtype=dtype),
        gaze_provider=provider,
        gaze_fusion="prefix-concat",
        classifier_dropout=0,
        gaze_projection_dropout=(0, 0),
        gaze_projection_dim=8,
    )
    model.gradient_checkpointing_enable({"use_reentrant": False}, every_n_layers=1)
    model.train()
    assert model.backbone.is_gradient_checkpointing
    assert next(model.backbone.parameters()).dtype == torch.float32
    parameters = tuple(model.gaze_projector.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    labels = torch.tensor([[0.1, 0.9], [0.7, 0.2]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    before = [parameter.detach().clone() for parameter in parameters]

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            predictions = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = (predictions - labels).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in parameters:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum().item() > 0
        optimizer.step()

    assert all(not torch.equal(initial, final) for initial, final in zip(before, parameters))
    final_raw, final_mask = provider.compute(input_ids, attention_mask)
    torch.testing.assert_close(final_raw, expected_raw, rtol=0, atol=0)
    assert torch.equal(final_mask, gaze_mask)
    assert provider.fake_model.batch_sizes == [1]
    assert provider.load_count == 1
    assert provider.fake_model.scale.grad is None
    assert all(value[0].requires_grad is False for value in provider._cache.values())

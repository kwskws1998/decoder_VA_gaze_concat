"""Native Qwen attention, masking, LoRA, and checkpoint replay for TRT rotation."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward

from va_model_code.decoder_va.gaze_attention import (
    GAZE_RELATIVE_ATTENTION_BACKEND,
    GAZE_RELATION_ARGUMENTS,
    gaze_relative_attention_forward,
    install_gaze_relative_attention,
)


def tiny_backbone(*, head_dim=8, partial_rotary_factor=1.0):
    """Construct a small hybrid model with valid full or partial native rotary dimensions."""

    rotary_half_dim = int(head_dim * partial_rotary_factor) // 2
    mrope_section = [rotary_half_dim // 3 + int(axis < rotary_half_dim % 3) for axis in range(3)]
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        attention_dropout=0.0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": partial_rotary_factor,
            "mrope_section": mrope_section,
        },
    )
    config._attn_implementation = "eager"
    return Qwen3_5TextModel(config)


def native_inputs(offset=0, *, compact=True):
    """Provide different prefix lengths, sparse text anchors, and right padding."""

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 0, 0]]) + offset,
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
        "gaze_relation_values": torch.tensor([[0., 0., 0., 1., 4., 2.], [0., 0., 3., 7., 0., 0.]]) + offset * torch.arange(6),
        "gaze_relation_mask": torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0]], dtype=torch.bool),
        "use_cache": False,
    }
    if compact:
        inputs.update(
            gaze_relation_indices=torch.tensor([[3, 4, 5], [2, 3, 0]]),
            gaze_relation_anchor_mask=torch.tensor([[True, True, True], [True, True, False]]),
        )
    return inputs


@pytest.mark.parametrize("autocast_bfloat16", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("head_dim,partial_rotary_factor", [(8, 1.0), (8, 0.25), (8, 0.5), (256, 0.25), (256, 0.5)])
def test_native_zero_rotation_matches_eager_including_gqa_padding_and_gate(
    autocast_bfloat16, head_dim, partial_rotary_factor, compact,
):
    """Full/partial native RoPE preserves zero-alpha identity and permits finite learned updates."""

    torch.manual_seed(71)
    reference = tiny_backbone(head_dim=head_dim, partial_rotary_factor=partial_rotary_factor).eval()
    assert reference.rotary_emb.inv_freq.numel() * 2 == int(head_dim * partial_rotary_factor)
    model = copy.deepcopy(reference)
    info = install_gaze_relative_attention(model, scale=0.7)
    assert info == [{"name": "layers.1.self_attn", "layer_index": 1, "num_heads": 4, "head_dim": head_dim}]
    assert model.config._attn_implementation == GAZE_RELATIVE_ATTENTION_BACKEND
    inputs = native_inputs(compact=compact)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_bfloat16):
        original = reference(**{key: value for key, value in inputs.items() if not key.startswith("gaze_")})
        actual = model(**inputs)
    torch.testing.assert_close(actual.last_hidden_state, original.last_hidden_state, rtol=0, atol=0)
    (actual.last_hidden_state * torch.randn_like(actual.last_hidden_state)).sum().backward()
    alpha = model.layers[1].self_attn.gaze_relative_rotation.alpha
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
    assert torch.count_nonzero(alpha.grad) > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        alpha.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_bfloat16):
        changed = model(**inputs).last_hidden_state
    assert torch.isfinite(changed).all()
    assert not torch.equal(changed, original.last_hidden_state)
    (changed * torch.randn_like(changed)).sum().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(alpha.grad) > 0


def test_backend_preserves_prefix_rows_causality_and_padded_key_mask():
    """Nonzero rotation changes mapped text relations without exposing future or padded keys."""

    torch.manual_seed(13)
    model = tiny_backbone().eval()
    install_gaze_relative_attention(model)
    module = model.layers[1].self_attn
    module.gaze_relative_rotation.alpha.data.fill_(0.4)
    query = torch.randn(2, 4, 6, 8)
    key = torch.randn(2, 2, 6, 8)
    value = torch.randn(2, 2, 6, 8)
    allowed = torch.ones(2, 1, 6, 6, dtype=torch.bool).tril()
    allowed[1, :, :, 4:] = False
    mask = torch.zeros(2, 1, 6, 6).masked_fill(~allowed, torch.finfo(torch.float32).min)
    inputs = native_inputs()
    baseline_output, baseline_weights = eager_attention_forward(module, query, key, value, mask, scaling=module.scaling)
    output, weights = gaze_relative_attention_forward(
        module, query, key, value, mask, scaling=module.scaling,
        gaze_relation_values=inputs["gaze_relation_values"],
        gaze_relation_mask=inputs["gaze_relation_mask"],
    )
    torch.testing.assert_close(output[:, :2], baseline_output[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(weights[:, :, :2], baseline_weights[:, :, :2], rtol=0, atol=0)
    assert torch.count_nonzero(weights.masked_select(~allowed.expand_as(weights))) == 0
    assert not torch.equal(weights[0, :, 5], baseline_weights[0, :, 5])
    torch.testing.assert_close(
        weights[0, :, 5, 0] / weights[0, :, 5, 1],
        baseline_weights[0, :, 5, 0] / baseline_weights[0, :, 5, 1],
    )


@pytest.mark.parametrize("with_lora", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.25, 0.5])
def test_checkpoint_replay_keeps_each_forward_relation_and_gradients(with_lora, partial_rotary_factor, compact):
    """Two outstanding forwards retain their own TRT metadata through backward replay."""

    torch.manual_seed(37)
    reference = tiny_backbone(partial_rotary_factor=partial_rotary_factor)
    if with_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        reference = get_peft_model(
            reference,
            LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules="all-linear", task_type=TaskType.FEATURE_EXTRACTION),
        )
    install_gaze_relative_attention(reference, scale=0.9)
    native = reference.get_base_model() if with_lora else reference
    native.layers[1].self_attn.gaze_relative_rotation.alpha.data.fill_(0.2)
    checkpointed = copy.deepcopy(reference)
    checkpointed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference.train()
    checkpointed.train()
    inputs_a, inputs_b = native_inputs(compact=compact), native_inputs(offset=2, compact=compact)
    target = torch.randn(2, 6, 32)
    for model in (reference, checkpointed):
        out_a = model(**inputs_a).last_hidden_state
        out_b = model(**inputs_b).last_hidden_state
        loss = (out_a * target).sum() + (out_b * target.flip(1)).sum()
        loss.backward()
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        expected = reference_parameters[name].grad
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-6)
    alpha_names = [name for name in reference_parameters if name.endswith("gaze_relative_rotation.alpha")]
    assert len(alpha_names) == 1
    alpha = reference_parameters[alpha_names[0]]
    assert alpha.requires_grad and alpha.dtype == torch.float32
    assert alpha.grad is not None and torch.count_nonzero(alpha.grad) > 0


def test_linear_attention_receives_no_gaze_arguments():
    """Explicit tensor metadata is consumed only by the full-attention branch."""

    model = tiny_backbone()
    install_gaze_relative_attention(model)
    seen = []

    def capture(module, args, kwargs):
        """Inspect the arguments after the installed linear-attention filter."""

        seen.append(set(kwargs))

    handle = model.layers[0].linear_attn.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model(**native_inputs())
    finally:
        handle.remove()
    assert len(seen) == 1
    assert GAZE_RELATION_ARGUMENTS.isdisjoint(seen[0])


@pytest.mark.parametrize("bf16", [False, True])
def test_compact_native_path_matches_dense_nonzero_outputs_and_all_gradients(bf16):
    """Sparse computation preserves full hybrid-model behavior after alpha has learned."""

    torch.manual_seed(51)
    dense = tiny_backbone(partial_rotary_factor=0.5)
    install_gaze_relative_attention(dense, scale=0.7)
    dense.layers[1].self_attn.gaze_relative_rotation.alpha.data.copy_(torch.tensor([0.2, -0.3, 0.1, 0.4]))
    compact = copy.deepcopy(dense)
    target = torch.randn(2, 6, 32)
    outputs = []
    for model, use_compact in ((dense, False), (compact, True)):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            output = model(**native_inputs(compact=use_compact)).last_hidden_state
            loss = (output * target).sum()
        loss.backward()
        outputs.append(output)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    dense_parameters = dict(dense.named_parameters())
    for name, parameter in compact.named_parameters():
        expected = dense_parameters[name].grad
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-6)


def test_missing_metadata_and_cached_generation_fail_clearly():
    """Reject unsupported call paths before attention can silently change semantics."""

    model = tiny_backbone()
    install_gaze_relative_attention(model)
    inputs = native_inputs()
    with pytest.raises(ValueError, match="explicit gaze_relation_values"):
        model(**{key: value for key, value in inputs.items() if key != "gaze_relation_values"})
    with pytest.raises(ValueError, match="uncached complete sequences"):
        model(**{**inputs, "use_cache": True})


def test_installation_rejects_other_models_backends_and_duplicate_installation():
    """Limit installation to the audited native Qwen implementation and eager masks."""

    with pytest.raises(TypeError, match="Qwen3.5 text backbone"):
        install_gaze_relative_attention(nn.Linear(2, 2))
    model = tiny_backbone()
    model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager Qwen3.5"):
        install_gaze_relative_attention(model)
    model.config._attn_implementation = "eager"
    install_gaze_relative_attention(model)
    with pytest.raises(ValueError, match="unsupported|already installed"):
        install_gaze_relative_attention(model)

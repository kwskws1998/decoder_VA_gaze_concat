"""Qwen3.5 eager attention with an explicit, checkpoint-safe TRT relation input."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .gaze_relation import TRTRelativeRotation


GAZE_RELATIVE_ATTENTION_BACKEND = "decoder_va_gaze_relative_eager"
GAZE_RELATION_IMPLEMENTATION = "compact_anchors_v1"
GAZE_RELATION_ARGUMENTS = frozenset(
    ("gaze_relation_values", "gaze_relation_mask", "gaze_relation_indices", "gaze_relation_anchor_mask")
)


def _strip_linear_gaze_arguments(module, args, kwargs):
    """Keep custom relation metadata away from optional fused linear-attention kernels."""

    del module
    return args, {key: value for key, value in kwargs.items() if key not in GAZE_RELATION_ARGUMENTS}


def _reject_cached_attention(module, args, kwargs):
    """Reject cache writes before native Qwen attention can update a KV cache."""

    del module
    cache = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
    if cache is not None or kwargs.get("use_cache", False):
        raise ValueError("TRT-relative attention supports uncached complete sequences only; use_cache must be False.")


def gaze_relative_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    *,
    gaze_relation_values: torch.Tensor | None = None,
    gaze_relation_mask: torch.Tensor | None = None,
    gaze_relation_indices: torch.Tensor | None = None,
    gaze_relation_anchor_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Change mapped text-pair logits and retain native mask, normalization, and values."""

    del kwargs
    if gaze_relation_values is None or gaze_relation_mask is None:
        raise ValueError("TRT-relative attention requires explicit gaze_relation_values and gaze_relation_mask.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("TRT-relative attention expects rank-4 query, key, and value tensors.")
    if query.shape[-2] != key.shape[-2] or key.shape[-2] != value.shape[-2]:
        raise ValueError("TRT-relative attention requires equal complete query/key/value sequence lengths.")
    if attention_mask is None or attention_mask.ndim != 4 or not attention_mask.is_floating_point():
        raise ValueError("TRT-relative eager attention requires an explicit 4D additive causal/padding mask.")
    if attention_mask.shape[-2:] != (query.shape[-2], key.shape[-2]):
        raise ValueError("The attention mask must match the complete query/key sequence lengths.")
    rotation = getattr(module, "gaze_relative_rotation", None)
    if not isinstance(rotation, TRTRelativeRotation):
        raise TypeError("The Qwen attention module has no installed TRT-relative rotation.")

    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    if key_states.shape[1] != query.shape[1] or value_states.shape[1] != query.shape[1]:
        raise ValueError("Grouped-query attention head counts are inconsistent.")
    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    logits = logits + rotation(
        query,
        key,
        gaze_relation_values,
        gaze_relation_mask,
        attention_scale=scaling,
        anchor_indices=gaze_relation_indices,
        anchor_mask=gaze_relation_anchor_mask,
    )
    logits = logits + attention_mask
    weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
    return output, weights


def install_gaze_relative_attention(backbone: nn.Module, scale: float = 1.0) -> list[dict[str, Any]]:
    """Install an eager TRT backend on a native or LoRA-wrapped Qwen3.5 text model."""

    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        Qwen3_5GatedDeltaNet,
        Qwen3_5TextModel,
    )

    base_model_getter = getattr(backbone, "get_base_model", None)
    text_model = base_model_getter() if callable(base_model_getter) else backbone
    if not isinstance(text_model, Qwen3_5TextModel):
        raise TypeError("TRT-relative attention requires a Qwen3.5 text backbone, optionally wrapped with LoRA.")
    if text_model.config._attn_implementation != "eager":
        raise ValueError("Install TRT-relative attention on an eager Qwen3.5 backbone; other backends are unsupported.")
    attention_modules = [
        (name, module) for name, module in text_model.named_modules() if isinstance(module, Qwen3_5Attention)
    ]
    if not attention_modules:
        raise ValueError("TRT-relative attention requires at least one Qwen3.5 full-attention layer.")
    if any(hasattr(module, "gaze_relative_rotation") for _, module in attention_modules):
        raise ValueError("TRT-relative attention is already installed on this backbone.")
    rotations = [
        TRTRelativeRotation(int(module.config.num_attention_heads), scale=scale).to(device=module.q_proj.weight.device)
        for _, module in attention_modules
    ]
    ALL_ATTENTION_FUNCTIONS.register(GAZE_RELATIVE_ATTENTION_BACKEND, gaze_relative_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(GAZE_RELATIVE_ATTENTION_BACKEND, eager_mask)
    layer_info = []
    for (name, module), rotation in zip(attention_modules, rotations):
        module.add_module("gaze_relative_rotation", rotation)
        module.register_forward_pre_hook(_reject_cached_attention, with_kwargs=True)
        layer_info.append(
            {
                "name": name,
                "layer_index": int(module.layer_idx),
                "num_heads": int(module.config.num_attention_heads),
                "head_dim": int(module.head_dim),
            }
        )
    for module in text_model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.register_forward_pre_hook(_strip_linear_gaze_arguments, with_kwargs=True)
    text_model.register_forward_pre_hook(_reject_cached_attention, with_kwargs=True)
    text_model.set_attn_implementation(GAZE_RELATIVE_ATTENTION_BACKEND)
    text_model.config.use_cache = False
    return layer_info

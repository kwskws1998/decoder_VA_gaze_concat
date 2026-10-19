"""Sample-wise causal gaze-prefix packing for text and aligned gaze embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PackedCausalInputs:
    """Hold a right-padded causal batch and optional compact text-anchor layout.

    Relation indices have shape (batch, maximum mapped anchors), refer to packed
    text positions, and use zero for inactive slots. The corresponding boolean
    anchor mask is right-padded and distinguishes those slots from actual indices.
    """

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    pooling_positions: torch.Tensor
    text_gaze_values: torch.Tensor | None = None
    text_gaze_mask: torch.Tensor | None = None
    text_gaze_indices: torch.Tensor | None = None
    text_gaze_anchor_mask: torch.Tensor | None = None


def _validate_inputs(
    text_embeddings: torch.Tensor,
    text_attention_mask: torch.Tensor,
    gaze_embeddings: torch.Tensor,
    gaze_mask: torch.Tensor,
    eye_start: torch.Tensor,
    eye_end: torch.Tensor,
    text_gaze_values: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Validate values and read compact batch dimensions with one host transfer."""

    if text_embeddings.ndim != 3 or gaze_embeddings.ndim != 3:
        raise ValueError("text_embeddings and gaze_embeddings must be rank-3.")
    if text_embeddings.shape[0] == 0:
        raise ValueError("The batch must contain at least one sample.")
    if text_embeddings.shape[1] == 0:
        raise ValueError("The text sequence must contain at least one token.")
    if text_embeddings.shape[0] != gaze_embeddings.shape[0]:
        raise ValueError("Text and gaze embeddings must have the same batch size.")
    if text_embeddings.shape[1] != gaze_embeddings.shape[1]:
        raise ValueError("Token-aligned text and gaze sequences must have equal length.")
    if text_embeddings.shape[2] != gaze_embeddings.shape[2]:
        raise ValueError("Text and gaze embeddings must have the same hidden size.")
    if tuple(text_attention_mask.shape) != tuple(text_embeddings.shape[:2]):
        raise ValueError("text_attention_mask must match the text sequence shape.")
    if tuple(gaze_mask.shape) != tuple(gaze_embeddings.shape[:2]):
        raise ValueError("gaze_mask must match the gaze sequence shape.")
    if text_attention_mask.device != text_embeddings.device:
        raise ValueError("text_attention_mask must be on the text embedding device.")
    if gaze_embeddings.device != text_embeddings.device:
        raise ValueError("Text and gaze embeddings must be on the same device.")
    if gaze_mask.device != text_embeddings.device:
        raise ValueError("gaze_mask must be on the text embedding device.")

    hidden_size = text_embeddings.shape[2]
    if eye_start.numel() != hidden_size or eye_end.numel() != hidden_size:
        raise ValueError("Eye boundary embeddings must match the hidden size.")
    if text_gaze_values is not None:
        if text_gaze_values.shape != gaze_mask.shape:
            raise ValueError("text_gaze_values must match the original text/gaze mask shape.")
        if text_gaze_values.device != text_embeddings.device:
            raise ValueError("text_gaze_values must be on the text embedding device.")

    text_mask_bool = text_attention_mask.to(dtype=torch.bool)
    gaze_mask_bool = gaze_mask.to(dtype=torch.bool)
    valid_lengths = text_mask_bool.sum(dim=1)
    gaze_counts = gaze_mask_bool.sum(dim=1)
    positions = torch.arange(
        text_embeddings.shape[1],
        device=text_embeddings.device,
    ).unsqueeze(0)
    expected_text_mask = positions < valid_lengths.unsqueeze(1)
    finite_gaze_values = (
        (~gaze_mask_bool | torch.isfinite(text_gaze_values)).all()
        if text_gaze_values is not None
        else torch.ones((), dtype=torch.bool, device=text_embeddings.device)
    )
    metadata = torch.stack(
        (
            ((text_attention_mask == 0) | (text_attention_mask == 1)).all(),
            ((gaze_mask == 0) | (gaze_mask == 1)).all(),
            (valid_lengths > 0).all(),
            (text_mask_bool == expected_text_mask).all(),
            (~(gaze_mask_bool & ~text_mask_bool)).all(),
            finite_gaze_values,
            (valid_lengths + gaze_counts + 2).max(),
            gaze_counts.max(),
        )
    ).tolist()
    errors = (
        "text_attention_mask must contain only zero and one values.",
        "gaze_mask must contain only zero and one values.",
        "Every text sequence must contain at least one valid token.",
        "text_attention_mask must use contiguous right padding.",
        "gaze_mask cannot select a padded text position.",
        "Mapped text gaze values must be finite.",
    )
    for valid, error in zip(metadata[:6], errors):
        if not valid:
            raise ValueError(error)
    return gaze_mask_bool, valid_lengths, gaze_counts, metadata[6], metadata[7]


def pack_prefix_gaze(
    text_embeddings: torch.Tensor,
    text_attention_mask: torch.Tensor,
    gaze_embeddings: torch.Tensor,
    gaze_mask: torch.Tensor,
    eye_start: torch.Tensor,
    eye_end: torch.Tensor,
    text_gaze_values: torch.Tensor | None = None,
) -> PackedCausalInputs:
    """Pack compact gaze prefixes and prepare optional mapped-text anchor indices.

    Integer gathers preserve sample-wise packing without dynamic Boolean gathers
    or per-sample device synchronization. Compact anchor metadata is shared by all
    relation attention layers rather than rediscovered inside each layer.
    """

    gaze_mask_bool, valid_lengths, gaze_counts, padded_length, max_anchors = _validate_inputs(
        text_embeddings,
        text_attention_mask,
        gaze_embeddings,
        gaze_mask,
        eye_start,
        eye_end,
        text_gaze_values,
    )
    batch_size, text_length = text_embeddings.shape[:2]
    hidden_size = text_embeddings.shape[2]
    original_positions = torch.arange(text_length, device=text_embeddings.device).unsqueeze(0)
    anchor_mask = (
        torch.arange(max_anchors, device=text_embeddings.device).unsqueeze(0)
        < gaze_counts.unsqueeze(1)
    )
    compact_indices = torch.where(gaze_mask_bool, original_positions, text_length).sort(dim=1).values
    compact_indices = torch.where(anchor_mask, compact_indices[:, :max_anchors], 0)
    compact_gaze = gaze_embeddings.gather(
        1, compact_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
    ).to(dtype=text_embeddings.dtype)
    eye_start_rows = eye_start.to(
        device=text_embeddings.device,
        dtype=text_embeddings.dtype,
    ).reshape(1, 1, hidden_size).expand(batch_size, -1, -1)
    eye_end_rows = eye_end.to(
        device=text_embeddings.device,
        dtype=text_embeddings.dtype,
    ).reshape(1, 1, hidden_size).expand(batch_size, -1, -1)
    embedding_source = torch.cat(
        (eye_start_rows, compact_gaze, eye_end_rows, text_embeddings), dim=1
    )
    packed_positions = torch.arange(padded_length, device=text_embeddings.device).unsqueeze(0)
    text_offsets = gaze_counts.unsqueeze(1) + 2
    packed_lengths = valid_lengths + gaze_counts + 2
    packed_mask = packed_positions < packed_lengths.unsqueeze(1)
    source_indices = torch.where(
        packed_positions < text_offsets - 1,
        packed_positions,
        max_anchors + packed_positions - gaze_counts.unsqueeze(1),
    )
    source_indices = torch.where(packed_mask, source_indices, 0)
    inputs_embeds = embedding_source.gather(
        1, source_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
    )
    inputs_embeds = torch.where(packed_mask.unsqueeze(-1), inputs_embeds, 0)
    packed_gaze_values = None
    packed_gaze_mask = None
    text_gaze_indices = None
    text_gaze_anchor_mask = None
    if text_gaze_values is not None:
        text_positions = packed_positions - text_offsets
        text_gather_indices = text_positions.clamp(0, text_length - 1)
        text_slots = (text_positions >= 0) & (text_positions < valid_lengths.unsqueeze(1))
        packed_gaze_mask = text_slots & gaze_mask_bool.gather(1, text_gather_indices)
        packed_gaze_values = torch.where(
            packed_gaze_mask,
            text_gaze_values.float().gather(1, text_gather_indices),
            0,
        )
        text_gaze_indices = torch.where(anchor_mask, compact_indices + text_offsets, 0)
        text_gaze_anchor_mask = anchor_mask
    return PackedCausalInputs(
        inputs_embeds=inputs_embeds,
        attention_mask=packed_mask.to(dtype=text_attention_mask.dtype),
        position_ids=torch.where(packed_mask, packed_positions, 0),
        pooling_positions=packed_lengths - 1,
        text_gaze_values=packed_gaze_values,
        text_gaze_mask=packed_gaze_mask,
        text_gaze_indices=text_gaze_indices,
        text_gaze_anchor_mask=text_gaze_anchor_mask,
    )

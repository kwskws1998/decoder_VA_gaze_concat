"""Sample-wise causal gaze-prefix packing for text and aligned gaze embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PackedCausalInputs:
    """Hold a right-padded causal batch and last-valid-text pooling positions."""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    pooling_positions: torch.Tensor


def _binary_mask(name: str, value: torch.Tensor) -> torch.Tensor:
    """Validate a binary mask and return its boolean view."""

    if value.dtype != torch.bool:
        is_binary = torch.logical_or(value == 0, value == 1)
        if not bool(is_binary.all().item()):
            raise ValueError(f"{name} must contain only zero and one values.")
    return value.to(dtype=torch.bool)


def _validate_inputs(
    text_embeddings: torch.Tensor,
    text_attention_mask: torch.Tensor,
    gaze_embeddings: torch.Tensor,
    gaze_mask: torch.Tensor,
    eye_start: torch.Tensor,
    eye_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate aligned gaze tensors and return boolean masks and text lengths."""

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

    text_mask_bool = _binary_mask("text_attention_mask", text_attention_mask)
    gaze_mask_bool = _binary_mask("gaze_mask", gaze_mask)
    valid_lengths = text_mask_bool.sum(dim=1)
    if bool((valid_lengths <= 0).any().item()):
        raise ValueError("Every text sequence must contain at least one valid token.")

    positions = torch.arange(
        text_embeddings.shape[1],
        device=text_embeddings.device,
    ).unsqueeze(0)
    expected_text_mask = positions < valid_lengths.unsqueeze(1)
    if not torch.equal(text_mask_bool, expected_text_mask):
        raise ValueError("text_attention_mask must use contiguous right padding.")
    if bool((gaze_mask_bool & ~text_mask_bool).any().item()):
        raise ValueError("gaze_mask cannot select a padded text position.")
    return text_mask_bool, gaze_mask_bool, valid_lengths


def pack_prefix_gaze(
    text_embeddings: torch.Tensor,
    text_attention_mask: torch.Tensor,
    gaze_embeddings: torch.Tensor,
    gaze_mask: torch.Tensor,
    eye_start: torch.Tensor,
    eye_end: torch.Tensor,
) -> PackedCausalInputs:
    """Prefix compact mapped gaze before valid text and then right-pad the batch."""

    _, gaze_mask_bool, valid_lengths = _validate_inputs(
        text_embeddings,
        text_attention_mask,
        gaze_embeddings,
        gaze_mask,
        eye_start,
        eye_end,
    )
    hidden_size = text_embeddings.shape[2]
    eye_start_row = eye_start.to(
        device=text_embeddings.device,
        dtype=text_embeddings.dtype,
    ).reshape(1, hidden_size)
    eye_end_row = eye_end.to(
        device=text_embeddings.device,
        dtype=text_embeddings.dtype,
    ).reshape(1, hidden_size)

    embedding_rows = []
    attention_rows = []
    position_rows = []
    pooling_positions = []
    for row_index, valid_length in enumerate(valid_lengths.tolist()):
        text_row = text_embeddings[row_index, :valid_length]
        gaze_row = gaze_embeddings[row_index, gaze_mask_bool[row_index]].to(
            dtype=text_embeddings.dtype
        )
        fused_row = torch.cat(
            (eye_start_row, gaze_row, eye_end_row, text_row),
            dim=0,
        )
        embedding_rows.append(fused_row)
        attention_rows.append(
            torch.ones(
                fused_row.shape[0],
                dtype=text_attention_mask.dtype,
                device=text_embeddings.device,
            )
        )
        position_rows.append(
            torch.arange(
                fused_row.shape[0],
                dtype=torch.long,
                device=text_embeddings.device,
            )
        )
        pooling_positions.append(fused_row.shape[0] - 1)

    padded_length = max(row.shape[0] for row in embedding_rows)
    for row_index, row in enumerate(embedding_rows):
        padding_length = padded_length - row.shape[0]
        if padding_length == 0:
            continue
        embedding_rows[row_index] = torch.cat(
            (
                row,
                row.new_zeros(padding_length, hidden_size),
            ),
            dim=0,
        )
        attention_rows[row_index] = torch.cat(
            (
                attention_rows[row_index],
                attention_rows[row_index].new_zeros(padding_length),
            ),
            dim=0,
        )
        position_rows[row_index] = torch.cat(
            (
                position_rows[row_index],
                position_rows[row_index].new_zeros(padding_length),
            ),
            dim=0,
        )

    return PackedCausalInputs(
        inputs_embeds=torch.stack(embedding_rows, dim=0),
        attention_mask=torch.stack(attention_rows, dim=0),
        position_ids=torch.stack(position_rows, dim=0),
        pooling_positions=torch.tensor(
            pooling_positions,
            dtype=torch.long,
            device=text_embeddings.device,
        ),
    )

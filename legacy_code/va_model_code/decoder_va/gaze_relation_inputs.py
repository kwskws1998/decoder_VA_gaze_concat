"""Deterministic within-sentence controls for the TRT attention relation.

Only mapped text anchors exchange values. The content key contains the seed,
active text token IDs, and original anchor positions, encoded as length-prefixed
little-endian integers. Each source ordinal receives a keyed BLAKE2b rank;
sorting those ranks produces a bijection, with the ordinal breaking collisions.
Neither TRT values nor labels contribute to the permutation. Padding, batch
composition, device, and global random state cannot change the assignment.
"""

from __future__ import annotations

import hashlib
import struct

import torch


SHUFFLE_ALGORITHM = "blake2b_rank_v1"
_HASH_DOMAIN = b"decoder_va:trt_relation:blake2b_rank_v1\x00"
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _binary_mask(value: torch.Tensor, name: str) -> torch.Tensor:
    """Require a real binary tensor and return its boolean representation."""

    if value.is_complex() or not bool(((value == 0) | (value == 1)).all().item()):
        raise ValueError(f"{name} must contain only binary zero or one values.")
    return value.bool()


def _rank_permutation(token_ids: list[int], anchors: list[int], seed: int) -> list[int]:
    """Hash a canonical sentence key and rank each source ordinal without an RNG."""

    content = bytearray(_HASH_DOMAIN)
    content.extend(struct.pack("<QQ", seed, len(token_ids)))
    for token_id in token_ids:
        content.extend(struct.pack("<q", token_id))
    content.extend(struct.pack("<Q", len(anchors)))
    for position in anchors:
        content.extend(struct.pack("<Q", position))
    sample_key = hashlib.blake2b(content, digest_size=32).digest()
    ranks = [
        hashlib.blake2b(
            struct.pack("<Q", ordinal), key=sample_key,
            digest_size=32, person=b"TRT-shuffle-v1",
        ).digest()
        for ordinal in range(len(anchors))
    ]
    return sorted(range(len(anchors)), key=lambda ordinal: (ranks[ordinal], ordinal))


def shuffle_relation_trt(
    trt_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gaze_mask: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Permute eligible TRT by content and seed, preserving dtype/device/gradients.

    The input tensors must have identical nonempty rank-two shapes and share a
    device. Text uses contiguous right padding, and gaze anchors must be a
    subset of active text positions. Active token IDs must be nonnegative
    integers; eligible TRT must be finite. Masked TRT may contain NaNs because
    it is replaced by zero before gathering. Zero or one eligible anchor is
    unchanged. Ties, equal TRT values, and identity permutations are allowed.
    """

    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1].")
    tensors = {
        "trt_values": trt_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "gaze_mask": gaze_mask,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a rank-two tensor.")
    if not trt_values.is_floating_point():
        raise ValueError("trt_values must have a floating-point dtype.")
    if input_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("input_ids must have an integer dtype other than bool.")
    if 0 in trt_values.shape:
        raise ValueError("Input batch and sequence dimensions must be nonempty.")
    for name, value in tensors.items():
        if value.shape != trt_values.shape:
            raise ValueError(f"{name} must have the same shape as trt_values.")
        if value.device != trt_values.device:
            raise ValueError(f"{name} must share the trt_values device.")
    text_mask = _binary_mask(attention_mask, "attention_mask")
    eligible = _binary_mask(gaze_mask, "gaze_mask")
    lengths = text_mask.sum(dim=1)
    if bool((lengths == 0).any().item()):
        raise ValueError("Every sentence must contain at least one active text token.")
    positions = torch.arange(trt_values.shape[1], device=trt_values.device)
    if not torch.equal(text_mask, positions[None, :] < lengths[:, None]):
        raise ValueError("attention_mask must use contiguous right padding.")
    if bool((eligible & ~text_mask).any().item()):
        raise ValueError("gaze_mask must be a subset of attention_mask.")
    if bool((input_ids[text_mask] < 0).any().item()):
        raise ValueError("Active input_ids must be nonnegative.")
    if not bool(torch.isfinite(trt_values[eligible]).all().item()):
        raise ValueError("Eligible TRT values must be finite.")

    token_rows = input_ids.detach().cpu().tolist()
    eligible_rows = eligible.detach().cpu().tolist()
    text_lengths = lengths.detach().cpu().tolist()
    gather_rows = []
    for tokens, mask, valid_length in zip(token_rows, eligible_rows, text_lengths):
        anchors = [position for position, valid in enumerate(mask) if valid]
        indices = list(range(trt_values.shape[1]))
        if len(anchors) > 1:
            permutation = _rank_permutation(tokens[:valid_length], anchors, seed)
            for target, source_ordinal in zip(anchors, permutation):
                indices[target] = anchors[source_ordinal]
        gather_rows.append(indices)
    gather_indices = torch.tensor(gather_rows, dtype=torch.long, device=trt_values.device)
    sanitized = torch.where(eligible, trt_values, torch.zeros_like(trt_values))
    return sanitized.gather(1, gather_indices)

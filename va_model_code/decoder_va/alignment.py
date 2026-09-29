"""Occurrence-preserving character alignment and explicit ET feature aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


GAZE_ALIGNMENT_CONTRACT = {
    "version": 2,
    "word_spans": "decoded_text_exact",
    "token_spans": "verified_tokenizer_offsets",
    "anchor": "first_overlapping_non_whitespace_token",
    "collision_reduction": {
        "nFix": "sum", "FFD": "mean", "GPT": "mean", "TRT": "sum", "fixProp": "mean",
    },
    "unmatched_policy": "error",
}


@dataclass(frozen=True)
class WordTokenAlignment:
    """Store overlapping target tokens and the unique first-token gaze mask."""

    word_to_token_indices: tuple[tuple[int, ...], ...]
    first_subword_mask: tuple[bool, ...]


def _as_int_list(values: Iterable[object]) -> list[int]:
    """Convert a tensor or iterable of scalar values to host integers."""

    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.detach().cpu().tolist()]
    return [int(value) for value in values]


def _word_spans(text: str, words: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Consume segments at their exact occurrences, skipping only whitespace."""

    cursor = 0
    spans = []
    for index, word in enumerate(words):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if not isinstance(word, str) or not word or not text.startswith(word, cursor):
            raise ValueError(
                f"ET segment {index} does not match decoded text at character {cursor}; "
                "segments must preserve exact source occurrences."
            )
        end = cursor + len(word)
        spans.append((cursor, end))
        cursor = end
    if text[cursor:].strip():
        raise ValueError("ET segments do not cover the complete decoded target text.")
    return tuple(spans)


def align_words_to_tokens(
    words: Sequence[str],
    token_ids: Iterable[object],
    attention_mask: Iterable[object],
    tokenizer,
    *,
    text: str | None = None,
) -> WordTokenAlignment:
    """Map source spans onto verified target offsets without searching by token text.

    Shared target tokens are allowed when a tokenizer merges several ET segments.
    Byte-level tokens may have overlapping character offsets. Each segment retains
    all visible overlapping tokens; its feature is anchored only to the first one.
    Unverifiable tokenization round trips fail rather than silently routing features
    to different token IDs or different occurrences of repeated text.
    """

    ids = _as_int_list(token_ids)
    raw_mask = attention_mask.detach().cpu().tolist() if isinstance(attention_mask, torch.Tensor) else list(attention_mask)
    if len(ids) != len(raw_mask):
        raise ValueError("token_ids and attention_mask must have the same length.")
    if any(value not in (0, 1) for value in raw_mask):
        raise ValueError("attention_mask must contain only zero and one values.")
    special_ids = {int(value) for value in (getattr(tokenizer, "all_special_ids", []) or [])}
    lexical_positions = [i for i, (token_id, valid) in enumerate(zip(ids, raw_mask)) if valid and token_id not in special_ids]
    lexical_ids = [ids[i] for i in lexical_positions]
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise ValueError("Character-span gaze alignment requires tokenizer.decode and offset mappings.")
    decoded = decoder(lexical_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if text is None:
        text = decoded
    elif text != decoded:
        raise ValueError("Supplied source text must exactly match the decoded target token IDs.")
    if not isinstance(text, str):
        raise ValueError("Decoded target text must be a string.")
    spans = _word_spans(text, words)
    if not lexical_ids:
        if spans:
            raise ValueError("Nonempty ET segments have no active lexical target tokens.")
        return WordTokenAlignment((), tuple(False for _ in ids))
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError, AttributeError) as exc:
        raise ValueError("Character-span gaze alignment requires a tokenizer with offset mappings.") from exc
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise ValueError("Tokenizer did not return input_ids and offset_mapping for gaze alignment.")
    encoded_ids = _as_int_list(encoded["input_ids"])
    if encoded_ids != lexical_ids:
        raise ValueError(
            "Decoded target text does not round-trip to the original lexical token IDs; "
            "cannot safely align gaze. Check byte-level truncation or embedded special tokens."
        )
    offsets = encoded["offset_mapping"]
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.detach().cpu().tolist()
    if len(offsets) != len(lexical_ids):
        raise ValueError("Tokenizer offsets must match the original lexical token count.")
    visible_offsets = []
    previous_start = -1
    for position, offset in zip(lexical_positions, offsets):
        if len(offset) != 2:
            raise ValueError("Each tokenizer offset must be a character start/end pair.")
        start, end = int(offset[0]), int(offset[1])
        if start != offset[0] or end != offset[1] or not 0 <= start <= end <= len(text) or start < previous_start:
            raise ValueError("Tokenizer returned invalid or nonmonotonic character offsets.")
        previous_start = start
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            visible_offsets.append((position, start, end))
    mappings = []
    first_subword_mask = [False] * len(ids)
    for word_index, (start, end) in enumerate(spans):
        overlapping = [(position, a, b) for position, a, b in visible_offsets if a < end and b > start]
        indices = tuple(position for position, _, _ in overlapping)
        if not indices:
            raise ValueError(f"ET segment {word_index} has no overlapping visible target token.")
        if any(
            not text[character].isspace() and not any(a <= character < b for _, a, b in overlapping)
            for character in range(start, end)
        ):
            raise ValueError(f"Tokenizer offsets do not cover every visible character of ET segment {word_index}.")
        mappings.append(indices)
        first_subword_mask[indices[0]] = True
    return WordTokenAlignment(tuple(mappings), tuple(first_subword_mask))


def remap_word_features_to_tokens(
    word_features,
    words: Sequence[str],
    token_ids: Iterable[object],
    attention_mask: Iterable[object],
    tokenizer,
    feature_dim: int,
    *,
    word_feature_mask: Iterable[object] | None = None,
    sum_feature_indices: Sequence[int] | None = None,
    text: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate valid features once at each segment's first overlapping target token.

    Channels listed in sum_feature_indices conserve their signed feature totals.
    Other channels use the arithmetic mean of finite, valid contributors. The
    generic default sums all channels; ET2 supplies its explicit per-channel policy.
    Invalid predictions never contribute to either the numerator or the count.
    """

    ids = _as_int_list(token_ids)
    dimension = int(feature_dim)
    if dimension <= 0:
        raise ValueError("feature_dim must be positive.")
    features = torch.as_tensor(word_features, dtype=torch.float32)
    if features.ndim != 2 or tuple(features.shape) != (len(words), dimension):
        raise ValueError(f"word_features must have shape [num_words, {dimension}].")
    if word_feature_mask is None:
        valid = torch.ones(len(words), dtype=torch.bool, device=features.device)
    else:
        supplied_mask = torch.as_tensor(word_feature_mask, device=features.device)
        if supplied_mask.shape != (len(words),) or not bool(((supplied_mask == 0) | (supplied_mask == 1)).all()):
            raise ValueError("word_feature_mask must be a binary [num_words] vector.")
        valid = supplied_mask.bool()
    valid = valid & torch.isfinite(features).all(dim=1)
    sum_indices = tuple(range(dimension)) if sum_feature_indices is None else tuple(sum_feature_indices)
    if len(set(sum_indices)) != len(sum_indices) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < dimension for i in sum_indices):
        raise ValueError("sum_feature_indices must be distinct valid feature-channel indices.")
    alignment = align_words_to_tokens(words, ids, attention_mask, tokenizer, text=text)
    output = features.new_zeros((len(ids), dimension))
    counts = features.new_zeros(len(ids))
    if words:
        destinations = torch.tensor([indices[0] for indices in alignment.word_to_token_indices], device=features.device)
        selected = destinations[valid]
        output = output.index_add(0, selected, features[valid])
        counts.index_add_(0, selected, counts.new_ones(selected.numel()))
    mean_channels = torch.tensor([i not in sum_indices for i in range(dimension)], device=features.device)
    divisors = torch.where(mean_channels.unsqueeze(0), counts.clamp_min(1).unsqueeze(1), 1.0)
    output = output / divisors
    mapped_mask = counts > 0
    if not bool(torch.isfinite(output).all()):
        raise ValueError("Aggregated gaze features must remain finite in float32.")
    return output, mapped_mask

"""Verify vectorized prefix packing against the original per-sample definition."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from va_model_code.decoder_va.packing import PackedCausalInputs, pack_prefix_gaze


def reference_pack(text, attention_mask, gaze, gaze_mask, eye_start, eye_end, values):
    """Implement the original sample-wise layout independently of gather packing."""

    rows = []
    masks = []
    positions = []
    relation_rows = []
    relation_masks = []
    pooling = []
    lengths = attention_mask.bool().sum(dim=1).tolist()
    eye_start_row = eye_start.to(text.dtype).reshape(1, -1)
    eye_end_row = eye_end.to(text.dtype).reshape(1, -1)
    for index, length in enumerate(lengths):
        selected_gaze = gaze[index, gaze_mask[index].bool()].to(text.dtype)
        row = torch.cat(
            (
                eye_start_row,
                selected_gaze,
                eye_end_row,
                text[index, :length],
            )
        )
        rows.append(row)
        masks.append(torch.ones(row.shape[0], dtype=attention_mask.dtype))
        positions.append(torch.arange(row.shape[0]))
        pooling.append(row.shape[0] - 1)
        if values is not None:
            prefix_length = selected_gaze.shape[0] + 2
            selected = gaze_mask[index, :length].bool()
            relation_rows.append(torch.cat((
                values.new_zeros(prefix_length, dtype=torch.float32),
                torch.where(selected, values[index, :length].float(), 0),
            )))
            relation_masks.append(torch.cat((
                torch.zeros(prefix_length, dtype=torch.bool), selected,
            )))
    return PackedCausalInputs(
        inputs_embeds=pad_sequence(rows, batch_first=True),
        attention_mask=pad_sequence(masks, batch_first=True),
        position_ids=pad_sequence(positions, batch_first=True),
        pooling_positions=torch.tensor(pooling),
        text_gaze_values=pad_sequence(relation_rows, batch_first=True) if values is not None else None,
        text_gaze_mask=pad_sequence(relation_masks, batch_first=True) if values is not None else None,
    )


def packing_inputs(dtype, gaze_dtype, pattern, with_values, noncontiguous=False):
    """Build mixed lengths, sparse anchors, and nonfinite values only off support."""

    torch.manual_seed(302)
    lengths = torch.tensor([7, 4, 1, 5])
    attention_mask = torch.arange(7).unsqueeze(0) < lengths.unsqueeze(1)
    if pattern == "none":
        gaze_mask = torch.zeros_like(attention_mask)
    elif pattern == "all":
        gaze_mask = attention_mask.clone()
    else:
        gaze_mask = torch.tensor([
            [0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
        ], dtype=torch.bool)
    text = torch.randn(4, 7, 6, dtype=dtype)
    gaze = torch.randn(4, 7, 6, dtype=gaze_dtype)
    text[~attention_mask] = float("nan")
    gaze[~gaze_mask] = float("nan")
    values = torch.randn(4, 7)
    values[~gaze_mask] = float("nan")
    values[0, 1] = 0
    if noncontiguous:
        text = text.transpose(1, 2).contiguous().transpose(1, 2)
        gaze = gaze.transpose(1, 2).contiguous().transpose(1, 2)
    return (
        text.requires_grad_(), attention_mask.long(), gaze.requires_grad_(), gaze_mask,
        torch.randn(6, requires_grad=True), torch.randn(1, 6, requires_grad=True),
        values.requires_grad_() if with_values else None,
    )


@pytest.mark.parametrize("dtype,gaze_dtype", [
    (torch.float32, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
])
@pytest.mark.parametrize("pattern", ["sparse", "none", "all"])
@pytest.mark.parametrize("with_values", [False, True])
def test_packing_outputs_and_gradients_match_original(dtype, gaze_dtype, pattern, with_values):
    """Keep forward values and gradients, allowing BF16 boundary reduction rounding."""

    arguments = packing_inputs(dtype, gaze_dtype, pattern, with_values, noncontiguous=True)
    expected_arguments = tuple(
        item.detach().clone().requires_grad_(item.requires_grad) if item is not None else None
        for item in arguments
    )
    actual = pack_prefix_gaze(*arguments)
    expected = reference_pack(*expected_arguments)
    names = ("inputs_embeds", "attention_mask", "position_ids", "pooling_positions",
             "text_gaze_values", "text_gaze_mask")
    for name in names:
        actual_value, expected_value = getattr(actual, name), getattr(expected, name)
        if expected_value is None:
            assert actual_value is None
        else:
            assert torch.isfinite(actual_value).all()
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    embedding_weights = torch.randn_like(actual.inputs_embeds, dtype=torch.float32)
    actual_loss = (actual.inputs_embeds.float() * embedding_weights).sum()
    expected_loss = (expected.inputs_embeds.float() * embedding_weights).sum()
    if with_values:
        relation_weights = torch.randn_like(actual.text_gaze_values)
        actual_loss = actual_loss + (actual.text_gaze_values * relation_weights).sum()
        expected_loss = expected_loss + (expected.text_gaze_values * relation_weights).sum()
    actual_loss.backward()
    expected_loss.backward()
    for index, (actual_input, expected_input) in enumerate(zip(arguments, expected_arguments)):
        if actual_input is not None and actual_input.requires_grad:
            assert actual_input.grad is not None and torch.isfinite(actual_input.grad).all()
            if dtype == torch.bfloat16 and index in (4, 5):
                torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0.02, atol=0.01)
            else:
                torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("pattern", ["sparse", "none", "all"])
def test_compact_metadata_exactly_covers_packed_mapped_text(pattern):
    """Use bounded unique indices, right padding, and original text offsets."""

    arguments = packing_inputs(torch.float32, torch.float32, pattern, True)
    packed = pack_prefix_gaze(*arguments)
    counts = arguments[3].sum(dim=1)
    assert packed.text_gaze_indices.shape == (4, int(counts.max()))
    assert packed.text_gaze_anchor_mask.dtype == torch.bool
    assert packed.text_gaze_indices.dtype == torch.long
    assert torch.count_nonzero(packed.text_gaze_indices[~packed.text_gaze_anchor_mask]) == 0
    for row in range(4):
        expected_positions = arguments[3][row].nonzero().flatten() + counts[row] + 2
        selected = packed.text_gaze_indices[row, packed.text_gaze_anchor_mask[row]]
        torch.testing.assert_close(selected, expected_positions)
        assert selected.unique().numel() == selected.numel()
        assert (selected < packed.inputs_embeds.shape[1]).all()
        assert packed.text_gaze_anchor_mask[row].tolist() == [True] * int(counts[row]) + [False] * (
            int(counts.max()) - int(counts[row])
        )
        torch.testing.assert_close(selected, packed.text_gaze_mask[row].nonzero().flatten())


def test_raw_packing_skips_relation_metadata_and_constructor_is_compatible():
    """Keep metadata optional for existing raw-path callers and constructors."""

    packed = pack_prefix_gaze(*packing_inputs(torch.float32, torch.float32, "sparse", False))
    assert packed.text_gaze_indices is None and packed.text_gaze_anchor_mask is None
    legacy = PackedCausalInputs(
        packed.inputs_embeds, packed.attention_mask, packed.position_ids, packed.pooling_positions
    )
    assert legacy.text_gaze_indices is None and legacy.text_gaze_anchor_mask is None


def test_packing_uses_single_host_metadata_read_without_scalar_reads(monkeypatch):
    """Prevent reintroducing per-sample or separate validation device syncs."""

    arguments = packing_inputs(torch.float32, torch.float32, "sparse", True)
    reads = []
    original_tolist = torch.Tensor.tolist

    def record_tolist(tensor):
        """Track the single batched metadata transfer."""

        reads.append(tuple(tensor.shape))
        return original_tolist(tensor)

    def reject_scalar_read(tensor, *args, **kwargs):
        """Fail on implicit or explicit host scalar synchronization."""

        raise AssertionError("Packing must batch device-to-host metadata reads.")

    monkeypatch.setattr(torch.Tensor, "tolist", record_tolist)
    monkeypatch.setattr(torch.Tensor, "item", reject_scalar_read)
    monkeypatch.setattr(torch.Tensor, "__bool__", reject_scalar_read)
    pack_prefix_gaze(*arguments)
    assert reads == [(8,)]


@pytest.mark.parametrize("invalid,expected_error", [
    ("text_binary", "text_attention_mask must contain only"),
    ("gaze_binary", "gaze_mask must contain only"),
    ("empty", "at least one valid token"),
    ("padding", "contiguous right padding"),
    ("gaze_padding", "padded text position"),
    ("nonfinite", "Mapped text gaze values must be finite"),
])
def test_batched_validation_preserves_invalid_input_rejection(invalid, expected_error):
    """Retain input checks while consolidating their device results."""

    arguments = list(packing_inputs(torch.float32, torch.float32, "sparse", True))
    arguments[3] = arguments[3].long()
    if invalid == "text_binary":
        arguments[1][0, 0] = 2
    elif invalid == "gaze_binary":
        arguments[3][0, 0] = 2
    elif invalid == "empty":
        arguments[1][0] = 0
    elif invalid == "padding":
        arguments[1][0, 0] = 0
    elif invalid == "gaze_padding":
        arguments[3][1, 6] = 1
    elif invalid == "nonfinite":
        with torch.no_grad():
            arguments[6][0, 1] = float("inf")
    with pytest.raises(ValueError, match=expected_error):
        pack_prefix_gaze(*arguments)

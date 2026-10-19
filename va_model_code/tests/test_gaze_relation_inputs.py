"""Verify deterministic, mask-safe, differentiable TRT permutation controls."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from va_model_code.decoder_va.gaze_relation_inputs import SHUFFLE_ALGORITHM, shuffle_relation_trt


def _batch(dtype=torch.float32):
    """Return sparse anchor rows containing zero, duplicates, signed values, and padding NaN."""

    return (
        torch.tensor([[0., 9., 2., 2., -1., float("nan")], [4., 8., 3., float("nan"), 0., 0.]], dtype=dtype),
        torch.tensor([[5, 9, 2, 6, 8, 0], [4, 3, 9, 0, 0, 0]]),
        torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]]),
        torch.tensor([[1, 0, 1, 1, 1, 0], [1, 0, 1, 0, 0, 0]]),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_shuffle_preserves_eligible_multisets_dtype_zero_and_inputs(dtype):
    """Mapped zero and duplicate values survive while every unmapped output becomes zero."""

    inputs = _batch(dtype)
    originals = [value.clone() for value in inputs]
    actual = shuffle_relation_trt(*inputs, seed=42)
    assert SHUFFLE_ALGORITHM == "blake2b_rank_v1"
    assert actual.shape == inputs[0].shape
    assert actual.dtype == dtype and actual.device == inputs[0].device
    assert torch.isfinite(actual).all()
    for row, mask in enumerate(inputs[3].bool()):
        torch.testing.assert_close(actual[row, mask].sort().values, inputs[0][row, mask].sort().values, rtol=0, atol=0)
    assert torch.count_nonzero(actual[~inputs[3].bool()]) == 0
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0, equal_nan=True)


def test_shuffle_is_repeatable_and_invariant_to_batch_order_rebatching_and_padding():
    """A sentence gets the same assignment independently of other rows or pad tokens."""

    inputs = _batch()
    expected = shuffle_relation_trt(*inputs, seed=42)
    torch.testing.assert_close(shuffle_relation_trt(*inputs, seed=42), expected, rtol=0, atol=0)
    reversed_output = shuffle_relation_trt(*(value.flip(0) for value in inputs), seed=42)
    torch.testing.assert_close(reversed_output.flip(0), expected, rtol=0, atol=0)
    for row, length in enumerate(inputs[2].sum(1).tolist()):
        individual = tuple(value[row:row + 1, :length] for value in inputs)
        actual = shuffle_relation_trt(*individual, seed=42)
        torch.testing.assert_close(actual[0], expected[row, :length], rtol=0, atol=0)
    padded = tuple(torch.cat((value, torch.full((2, 3), fill, dtype=value.dtype)), dim=1)
                   for value, fill in zip(inputs, (float("nan"), -17, 0, 0)))
    actual = shuffle_relation_trt(*padded, seed=42)
    torch.testing.assert_close(actual[:, :6], expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, 6:]) == 0


def test_versioned_hash_algorithm_matches_a_fixed_reference_assignment():
    """Pin canonical byte encoding and rank order so code drift cannot alter saved runs."""

    values = torch.arange(1., 9.).reshape(1, 8)
    ids = torch.tensor([[3, 9, 15, 21, 27, 33, 39, 45]])
    mask = torch.ones_like(ids)
    expected = torch.tensor([[5., 3., 4., 2., 8., 6., 7., 1.]])
    actual = shuffle_relation_trt(values, ids, mask, mask, seed=42)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.inference_mode():
        inference = shuffle_relation_trt(values, ids, mask, mask, seed=42)
    torch.testing.assert_close(inference, expected, rtol=0, atol=0)


def test_shuffle_permutation_depends_on_seed_content_and_anchor_positions_not_values():
    """Hash inputs control rank, while changing TRT magnitudes leaves source assignments fixed."""

    values = torch.arange(1., 17.).reshape(1, 16)
    ids = torch.arange(100, 116).reshape(1, 16)
    mask = torch.ones_like(ids)
    outputs = [shuffle_relation_trt(values, ids, mask, mask, seed=seed) for seed in (0, 1, 2**63 - 1)]
    assert all(not torch.equal(outputs[first], outputs[second]) for first, second in ((0, 1), (0, 2), (1, 2)))
    shifted_ids = ids.clone()
    shifted_ids[0, 5] += 1
    assert not torch.equal(outputs[0], shuffle_relation_trt(values, shifted_ids, mask, mask, seed=0))
    actual_scaled = shuffle_relation_trt(values * -3. + 7., ids, mask, mask, seed=0)
    torch.testing.assert_close(actual_scaled, outputs[0] * -3. + 7., rtol=0, atol=0)
    spaced_ids = torch.arange(100, 132).reshape(1, 32)
    spaced_mask = torch.ones_like(spaced_ids)
    first_anchors = torch.zeros_like(spaced_ids)
    second_anchors = torch.zeros_like(spaced_ids)
    first_anchors[:, :16] = 1
    second_anchors[:, 16:] = 1
    repeated_values = values.repeat(1, 2)
    first = shuffle_relation_trt(repeated_values, spaced_ids, spaced_mask, first_anchors, seed=0)[:, :16]
    second = shuffle_relation_trt(repeated_values, spaced_ids, spaced_mask, second_anchors, seed=0)[:, 16:]
    assert not torch.equal(first, second)


def test_shuffle_leaves_all_global_random_states_untouched():
    """Python, NumPy, and torch RNGs neither select nor advance the assignment."""

    inputs = _batch()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()
    shuffle_relation_trt(*inputs, seed=42)
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]
    assert torch.equal(torch.random.get_rng_state(), before_torch)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_zero_single_or_identical_anchors_remain_valid(count):
    """No anchors, one valid zero, and equal repeated values need no forced derangement."""

    values = torch.zeros((1, 4), requires_grad=True)
    ids = torch.arange(4).reshape(1, 4)
    attention = torch.ones_like(ids)
    gaze = torch.zeros_like(ids)
    gaze[:, :count] = 1
    actual = shuffle_relation_trt(values, ids, attention, gaze, seed=4)
    assert torch.equal(actual, values)
    actual.sum().backward()
    torch.testing.assert_close(values.grad, gaze.float(), rtol=0, atol=0)


def test_gather_routes_gradients_only_to_the_selected_source_anchors():
    """Each source anchor receives its target's gradient and invalid NaNs receive zero."""

    values = torch.tensor([[0., float("nan"), 2., 3., 4., float("nan")]], requires_grad=True)
    ids = torch.tensor([[2, 4, 6, 8, 10, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 0]])
    gaze = torch.tensor([[1, 0, 1, 1, 1, 0]])
    actual = shuffle_relation_trt(values, ids, attention, gaze, seed=9)
    weights = torch.tensor([[1., 2., 4., 8., 16., 32.]])
    (actual * weights).sum().backward()
    expected_gradient = torch.zeros_like(values)
    anchors = gaze[0].nonzero().flatten().tolist()
    for target in anchors:
        source = next(position for position in anchors if values[0, position].item() == actual[0, target].item())
        expected_gradient[0, source] = weights[0, target]
    torch.testing.assert_close(values.grad, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [True, False, -1, 2**63, 3.0, np.int64(3), "3", None])
def test_invalid_seed_is_rejected(seed):
    """The seed contract accepts exactly Python integers in the declared closed range."""

    with pytest.raises(ValueError, match="seed"):
        shuffle_relation_trt(*_batch(), seed=seed)


@pytest.mark.parametrize("field,change,error", [
    (0, lambda value: value.long(), "floating-point"),
    (1, lambda value: value.float(), "integer dtype"),
    (1, lambda value: value.bool(), "integer dtype"),
    (0, lambda value: value.unsqueeze(-1), "rank-two"),
    (2, lambda value: value[:, :-1], "same shape"),
    (2, lambda value: value * 2, "binary"),
    (3, lambda value: value.float().fill_(float("nan")), "binary"),
    (3, lambda value: value.to(torch.complex64), "binary"),
    (2, lambda value: value.roll(1, dims=1), "right padding"),
    (2, lambda value: value * 0, "active text"),
    (3, lambda value: torch.ones_like(value), "subset"),
    (1, lambda value: value * -1, "nonnegative"),
    (0, lambda value: value.fill_(float("inf")), "finite"),
])
def test_invalid_inputs_are_rejected(field, change, error):
    """Malformed shapes, dtypes, masks, token IDs, and mapped TRT fail explicitly."""

    inputs = list(_batch())
    inputs[field] = change(inputs[field])
    with pytest.raises(ValueError, match=error):
        shuffle_relation_trt(*inputs, seed=42)


@pytest.mark.parametrize("shape", [(0, 5), (1, 0)])
def test_empty_tensor_dimensions_are_rejected(shape):
    """Empty batches or sequences cannot represent a valid sentence control."""

    values = torch.zeros(shape)
    ids = torch.zeros(shape, dtype=torch.long)
    with pytest.raises(ValueError, match="nonempty"):
        shuffle_relation_trt(values, ids, ids, ids, seed=42)


@pytest.mark.parametrize("device", [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")),
])
def test_shuffle_agrees_between_cpu_and_accelerator_and_rejects_mixed_devices(device):
    """CPU and accelerator assignments agree while mixed-device inputs fail early."""

    inputs = _batch()
    expected = shuffle_relation_trt(*inputs, seed=42)
    actual = shuffle_relation_trt(*(value.to(device) for value in inputs), seed=42)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    mixed = list(inputs)
    mixed[1] = mixed[1].to(device)
    with pytest.raises(ValueError, match="device"):
        shuffle_relation_trt(*mixed, seed=42)

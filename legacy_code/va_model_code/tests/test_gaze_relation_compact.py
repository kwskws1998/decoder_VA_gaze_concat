"""Compact-anchor equivalence, autograd, numerical boundaries, and validation tests."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from va_model_code.decoder_va.gaze_relation import TRTRelativeRotation


def _inputs(dtype=torch.float32, *, length=13):
    """Build heterogeneous anchor counts, including eligible position zero and zero TRT."""

    generator = torch.Generator().manual_seed(927)
    query = torch.randn(3, 4, length, 8, generator=generator).to(dtype)
    key = torch.randn(3, 2, length, 8, generator=generator).to(dtype)
    gaze = torch.arange(3 * length).reshape(3, length).float() / 4 - 2
    mask = torch.zeros(3, length, dtype=torch.bool)
    mask[0, [0, 3, 7]] = True
    mask[1, [2]] = True
    mask[2, [1, 3, 5, 9, 12]] = True
    gaze[0, 0] = 0
    return query, key, gaze, mask


def _layout(mask):
    """Build a padded position layout independently from the production packing path."""

    rows = [row.nonzero().flatten() for row in mask]
    width = max((row.numel() for row in rows), default=0)
    indices = torch.zeros(mask.shape[0], width, dtype=torch.long, device=mask.device)
    active = torch.zeros_like(indices, dtype=torch.bool)
    for index, row in enumerate(rows):
        indices[index, :row.numel()] = row
        active[index, :row.numel()] = True
    return indices, active


def _call(module, query, key, gaze, mask, *, indices=None, active=None):
    """Run the compact path with either independently built or explicit metadata."""

    if indices is None and active is None:
        indices, active = _layout(mask)
    return module(query, key, gaze, mask, anchor_indices=indices, anchor_mask=active)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("initial_alpha", [0.0, 0.13, 1e-9])
def test_compact_matches_dense_outputs_and_all_input_gradients(dtype, initial_alpha):
    """Gather/scatter preserves the dense operator and gradient routing, including GQA."""

    query, key, gaze, mask = _inputs(dtype)
    compact_inputs = [value.clone().requires_grad_() for value in (query, key, gaze)]
    dense_inputs = [value.clone().requires_grad_() for value in (query, key, gaze)]
    compact = TRTRelativeRotation(4, scale=1.7)
    with torch.no_grad():
        compact.alpha.copy_(torch.tensor([1., -2., 3., -4.]) * initial_alpha)
    dense = copy.deepcopy(compact)
    actual = _call(compact, *compact_inputs, mask)
    expected = dense(*dense_inputs, mask)
    tolerance = 2e-3 if dtype in (torch.bfloat16, torch.float16) else 3e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    weights = torch.linspace(-1., 1., actual.numel()).reshape(actual.shape)
    (actual.float() * weights).sum().backward()
    (expected.float() * weights).sum().backward()
    for compact_input, dense_input in zip(compact_inputs, dense_inputs):
        torch.testing.assert_close(compact_input.grad, dense_input.grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(compact.alpha.grad, dense.alpha.grad, atol=3e-5, rtol=3e-5)
    if initial_alpha == 0:
        assert torch.equal(actual, torch.zeros_like(actual))
        assert torch.count_nonzero(compact.alpha.grad) == compact.num_heads
        assert torch.count_nonzero(compact_inputs[0].grad) == 0
        assert torch.count_nonzero(compact_inputs[1].grad) == 0


def test_inactive_duplicate_position_zero_does_not_clobber_real_anchor():
    """Inactive padded entries contribute zero while an actual position-zero anchor survives."""

    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.fill_(0.4)
    output = _call(module, query, key, gaze, mask)
    dense = module(query, key, gaze, mask)
    assert output[0, :, 0, 3].abs().sum() > 0
    torch.testing.assert_close(output, dense, atol=3e-6, rtol=3e-6)
    eligible = mask[:, None, :, None] & mask[:, None, None, :]
    assert torch.count_nonzero(output.masked_select(~eligible)) == 0
    assert torch.count_nonzero(output.diagonal(dim1=-2, dim2=-1)) == 0


def test_nonfinite_padding_does_not_contaminate_values_or_gradients():
    """Gathered padding that aliases a masked NaN position stays excluded before arithmetic."""

    query, key, gaze, mask = _inputs()
    query = query.masked_fill(~mask[:, None, :, None], float("nan")).requires_grad_()
    key = key.masked_fill(~mask[:, None, :, None], float("inf")).requires_grad_()
    gaze = gaze.masked_fill(~mask, float("nan")).requires_grad_()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.fill_(0.4)
    output = _call(module, query, key, gaze, mask)
    output.square().sum().backward()
    for value in (output, query.grad, key.grad, gaze.grad, module.alpha.grad):
        assert torch.isfinite(value).all()
    assert torch.count_nonzero(query.grad.masked_select(~mask[:, None, :, None])) == 0
    assert torch.count_nonzero(key.grad.masked_select(~mask[:, None, :, None])) == 0
    assert torch.count_nonzero(gaze.grad[~mask]) == 0


@pytest.mark.parametrize("batch,length,one_anchor", [(3, 13, False), (3, 13, True), (2, 0, False), (0, 13, False), (0, 0, False)])
def test_degenerate_layout_keeps_all_gradients_connected_with_exact_zero(batch, length, one_anchor):
    """Zero or one anchor returns zero without matrix products and retains every input graph."""

    query = torch.full((batch, 4, length, 8), float("nan"))
    key = torch.full((batch, 2, length, 8), float("inf"))
    gaze = torch.full((batch, length), float("nan"))
    mask = torch.zeros(batch, length, dtype=torch.bool)
    if one_anchor:
        query[:, :, 4] = torch.finfo(torch.float32).max
        key[:, :, 4] = torch.finfo(torch.float32).max
        gaze[:, 4] = torch.finfo(torch.float32).max
        mask[:, 4] = True
    for value in (query, key, gaze):
        value.requires_grad_()
    module = TRTRelativeRotation(4)
    output = _call(module, query, key, gaze, mask)
    assert output.shape == (batch, 4, length, length)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    for value in (query, key, gaze, module.alpha):
        assert value.grad is not None
        assert torch.count_nonzero(value.grad) == 0


@pytest.mark.parametrize("width", [0, 1])
@pytest.mark.parametrize("target", ["query", "key", "gaze", "alpha"])
def test_degenerate_layout_still_rejects_nonfinite_eligible_values(width, target):
    """Skipping pair arithmetic cannot skip alpha or eligible-input validation."""

    query = torch.ones(1, 2, 4, 2)
    key = torch.ones(1, 1, 4, 2)
    gaze = torch.zeros(1, 4)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    module = TRTRelativeRotation(2)
    if width:
        mask[:, 1] = True
    with torch.no_grad():
        if target == "alpha":
            module.alpha[0] = float("nan")
        elif not width:
            return
        elif target == "query":
            query[0, 0, 1, 0] = float("nan")
        elif target == "key":
            key[0, 0, 1, 0] = float("inf")
        else:
            gaze[0, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _call(module, query, key, gaze, mask)


@pytest.mark.parametrize("alpha", [1e-4, 1e-8, 1e-10])
def test_compact_preserves_tiny_phase_first_and_second_order_terms(alpha):
    """The stable trigonometric expression remains accurate after compact scattering."""

    query = torch.ones(1, 1, 7, 2)
    query[..., 1] = 0
    key = query.clone()
    gaze = torch.zeros(1, 7)
    gaze[:, 5] = 1
    mask = torch.tensor([[False, True, False, False, False, True, False]])
    module = TRTRelativeRotation(1)
    with torch.no_grad():
        module.alpha.fill_(alpha)
    phase = float(module.alpha[0].detach())
    output = module(query, key, gaze, mask, attention_scale=1., anchor_indices=_layout(mask)[0], anchor_mask=_layout(mask)[1])
    assert output[0, 0, 1, 5].item() == pytest.approx(-2 * math.sin(phase / 2) ** 2, rel=2e-7, abs=0)
    key[..., 0] = 0
    key[..., 1] = 1
    output = module(query, key, gaze, mask, attention_scale=1., anchor_indices=_layout(mask)[0], anchor_mask=_layout(mask)[1])
    assert output[0, 0, 1, 5].item() == pytest.approx(-math.sin(phase), rel=2e-7, abs=0)


def test_compact_shift_invariance_equal_trt_and_outer_autocast():
    """A representable common offset and outer mixed precision leave compact math unchanged."""

    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.copy_(torch.tensor([.13, -.27, .39, -.41]))
    output = _call(module, query, key, gaze, mask)
    shifted = _call(module, query, key, gaze + 1_000_000, mask)
    torch.testing.assert_close(output, shifted, atol=0, rtol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast = _call(module, query, key, gaze, mask)
    torch.testing.assert_close(output, autocast, atol=0, rtol=0)
    equal = _call(module, query, key, torch.full_like(gaze, 9_999_999), mask)
    assert torch.count_nonzero(equal) == 0
    assert set(module.state_dict()) == {"alpha"}


@pytest.mark.parametrize("kind", ["missing", "duplicate", "negative", "too_large", "nonzero_padding", "extra_anchor"])
def test_rejects_invalid_layout_without_unsafe_gather(kind):
    """Malformed indices are clamped before gather, then rejected as a layout violation."""

    query, key, gaze, mask = _inputs()
    indices, active = _layout(mask)
    if kind == "missing":
        active[0, 1] = False
        indices[0, 1] = 0
    elif kind == "duplicate":
        indices[0, 1] = indices[0, 0]
    elif kind == "negative":
        indices[0, 1] = -1
    elif kind == "too_large":
        indices[0, 1] = 10**6
    elif kind == "nonzero_padding":
        indices[0, -1] = 1
    else:
        indices[0, -1] = 2
        active[0, -1] = True
    with pytest.raises(ValueError, match="anchor"):
        _call(TRTRelativeRotation(4), query, key, gaze, mask, indices=indices, active=active)


@pytest.mark.parametrize("kind", ["only_indices", "only_mask", "float_indices", "int_mask", "shape", "batch", "complex_text_mask", "nonbinary_text_mask"])
def test_rejects_invalid_layout_shapes_and_types(kind):
    """Layout and mask types are checked before gathering or constructing indices."""

    query, key, gaze, mask = _inputs()
    indices, active = _layout(mask)
    if kind == "only_indices":
        active = None
    elif kind == "only_mask":
        indices = None
    elif kind == "float_indices":
        indices = indices.float()
    elif kind == "int_mask":
        active = active.long()
    elif kind == "shape":
        active = active[:, :-1]
    elif kind == "batch":
        indices = indices[:1]
        active = active[:1]
    elif kind == "complex_text_mask":
        mask = mask.to(torch.complex64)
    else:
        mask = mask.float()
        mask[0, 0] = .5
    with pytest.raises(ValueError):
        _call(TRTRelativeRotation(4), query, key, gaze, mask, indices=indices, active=active)


def test_pair_overflow_is_rejected_in_compact_path():
    """Finite anchors whose relative phase exceeds FP32 still raise an explicit error."""

    query, key, gaze, mask = _inputs()
    gaze[0, 0] = -torch.finfo(torch.float32).max
    gaze[0, 3] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="finite"):
        _call(TRTRelativeRotation(4), query, key, gaze, mask)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("binary_dtype", [torch.bool, torch.int64])
def test_successful_forward_has_only_one_tensor_scalar_read(compact, binary_dtype):
    """All value validation shares one scalar read instead of repeated synchronization."""

    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        if compact:
            _call(module, query, key, gaze, mask.to(binary_dtype))
        else:
            module(query, key, gaze, mask.to(binary_dtype))
    reads = sum(event.count for event in profiler.key_averages() if event.key == "aten::_local_scalar_dense")
    assert reads == 1


def test_empty_batch_with_explicit_nonempty_anchor_width():
    """A valid empty batch can retain a padded anchor width without ambiguous reshapes."""

    query = torch.zeros(0, 4, 13, 8, requires_grad=True)
    key = torch.zeros(0, 2, 13, 8, requires_grad=True)
    gaze = torch.zeros(0, 13, requires_grad=True)
    module = TRTRelativeRotation(4)
    output = module(
        query, key, gaze, torch.zeros(0, 13, dtype=torch.bool),
        anchor_indices=torch.zeros(0, 3, dtype=torch.long),
        anchor_mask=torch.zeros(0, 3, dtype=torch.bool),
    )
    assert output.shape == (0, 4, 13, 13)
    output.sum().backward()
    for value in (query, key, gaze, module.alpha):
        assert value.grad is not None
        assert torch.count_nonzero(value.grad) == 0


@pytest.mark.parametrize("head_dim", [8, 256])
def test_all_positions_eligible_and_reordered_layout_match_dense(head_dim):
    """The layout is a coordinate mapping and may include every token in any order."""

    generator = torch.Generator().manual_seed(123)
    query = torch.randn(2, 4, 7, head_dim, generator=generator)
    key = torch.randn(2, 1, 7, head_dim, generator=generator)
    gaze = torch.randn(2, 7, generator=generator)
    mask = torch.ones(2, 7, dtype=torch.bool)
    indices = torch.tensor([[6, 3, 1, 2, 5, 0, 4], [2, 0, 1, 3, 4, 6, 5]])
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.copy_(torch.tensor([.1, -.2, .3, -.4]))
    compact = module(query, key, gaze, mask, anchor_indices=indices, anchor_mask=mask)
    dense = module(query, key, gaze, mask)
    torch.testing.assert_close(compact, dense, atol=2e-6, rtol=2e-5)

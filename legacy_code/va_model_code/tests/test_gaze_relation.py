"""Operator, masking, dtype, gradient, and contract tests for TRT relative rotation."""

from __future__ import annotations

import math

import pytest
import torch

from va_model_code.decoder_va.gaze_relation import (
    TRTRelativeRotation,
    gaze_relation_contract,
    validate_gaze_relation_contract,
)


def _inputs(dtype=torch.float32):
    """Create reproducible signed TRT and independently sampled query/key vectors."""

    generator = torch.Generator().manual_seed(341)
    query = torch.randn(2, 4, 5, 8, generator=generator).to(dtype)
    key = torch.randn(2, 2, 5, 8, generator=generator).to(dtype)
    gaze = torch.tensor([[0.0, -0.5, 0.75, 2.0, 6.0], [-1.0, 3.0, 0.5, 1.0, 0.0]])
    mask = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 1, 1]])
    return query, key, gaze, mask


def _reference(query, key, gaze, mask, alpha, scale=1.0, attention_scale=None):
    """Evaluate the relative-phase bilinear equation independently in scalar loops."""

    batch_size, heads, length, head_dim = query.shape
    output = torch.zeros(batch_size, heads, length, length, dtype=torch.float64)
    coefficient = head_dim ** -0.5 if attention_scale is None else attention_scale
    for batch in range(batch_size):
        for head in range(heads):
            key_head = head // (heads // key.shape[1])
            for i in range(length):
                for j in range(length):
                    if not mask[batch, i] or not mask[batch, j] or i == j:
                        continue
                    phase = float(alpha[head]) * (float(gaze[batch, j]) - float(gaze[batch, i])) / scale
                    score = 0.0
                    for d in range(head_dim // 2):
                        q0, q1 = float(query[batch, head, i, d]), float(query[batch, head, i, d + head_dim // 2])
                        k0, k1 = float(key[batch, key_head, j, d]), float(key[batch, key_head, j, d + head_dim // 2])
                        score += (math.cos(phase) - 1) * (q0 * k0 + q1 * k1)
                        score += math.sin(phase) * (q1 * k0 - q0 * k1)
                    output[batch, head, i, j] = score * coefficient
    return output


def test_zero_initialization_is_exact_identity_with_nonzero_alpha_gradient():
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    output = module(query, key, gaze, mask)
    assert torch.equal(output, torch.zeros_like(output))
    weights = torch.arange(output.numel()).reshape(output.shape).float()
    (output * weights).sum().backward()
    assert module.alpha.grad is not None
    assert torch.isfinite(module.alpha.grad).all()
    assert (module.alpha.grad.abs() > 1e-4).all()


def test_matches_independent_relative_phase_equation_and_attention_scale():
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4, scale=1.7)
    with torch.no_grad():
        module.alpha.copy_(torch.tensor([0.3, -0.2, 0.05, 0.4]))
    actual = module(query, key, gaze, mask, attention_scale=0.2)
    expected = _reference(query, key, gaze, mask, module.alpha.detach(), 1.7, 0.2)
    torch.testing.assert_close(actual.double(), expected, atol=3e-7, rtol=2e-5)


def test_shift_invariance_of_eligible_pair_correction():
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    original = module(query, key, gaze, mask)
    shifted = module(query, key, gaze + torch.tensor([[4.0], [-7.0]]), mask)
    torch.testing.assert_close(original, shifted, atol=1e-6, rtol=2e-5)


def test_large_representable_common_shift_is_exactly_invariant():
    """A shared TRT offset cannot change relative phases through separate angle rounding."""

    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.copy_(torch.tensor([0.13, -0.27, 0.39, -0.41]))
    original = module(query, key, gaze, mask)
    shifted = module(query, key, gaze + torch.tensor([[1_000_000.], [-1_000_000.]]), mask)
    torch.testing.assert_close(original, shifted, atol=0, rtol=0)


@pytest.mark.parametrize("alpha", [1e-4, 1e-8, 1e-10])
def test_tiny_relative_phase_preserves_first_and_second_order_terms(alpha):
    """Small rotations retain both sine and cosine corrections without dot-product cancellation."""

    query = torch.tensor([[[[1., 0.], [1., 0.]]]])
    gaze = torch.tensor([[0., 1.]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    module = TRTRelativeRotation(1)
    with torch.no_grad():
        module.alpha.fill_(alpha)
    phase = float(module.alpha[0].detach())
    parallel_delta = module(query, query.clone(), gaze, mask, attention_scale=1.0)
    perpendicular_key = torch.tensor([[[[0., 1.], [0., 1.]]]])
    perpendicular_delta = module(query, perpendicular_key, gaze, mask, attention_scale=1.0)
    assert parallel_delta[0, 0, 0, 1] != 0
    assert perpendicular_delta[0, 0, 0, 1] != 0
    assert float(parallel_delta[0, 0, 0, 1].detach()) == pytest.approx(
        -2.0 * math.sin(phase / 2.0) ** 2, abs=0, rel=2e-7,
    )
    assert float(perpendicular_delta[0, 0, 0, 1].detach()) == pytest.approx(
        -math.sin(phase), abs=0, rel=2e-7,
    )


def test_zero_alpha_has_zero_query_key_gradient_and_nonzero_alpha_gradient():
    """The zero-initialized branch leaves backbone gradients unchanged while alpha can learn."""

    query, key, gaze, mask = _inputs()
    query.requires_grad_()
    key.requires_grad_()
    module = TRTRelativeRotation(4)
    output = module(query, key, gaze, mask)
    weights = torch.arange(output.numel()).reshape(output.shape).float() / output.numel()
    (output * weights).sum().backward()
    assert torch.equal(query.grad, torch.zeros_like(query))
    assert torch.equal(key.grad, torch.zeros_like(key))
    assert torch.isfinite(module.alpha.grad).all()
    assert (module.alpha.grad.abs() > 1e-6).all()


def test_noneligible_pairs_and_diagonal_are_exactly_unchanged():
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.fill_(0.7)
    delta = module(query, key, gaze, mask)
    base_logits = query @ key.repeat_interleave(2, dim=1).transpose(-2, -1) / math.sqrt(8)
    pair_mask = (mask[:, None, :, None] & mask[:, None, None, :]).bool().expand_as(delta)
    assert torch.equal(delta[~pair_mask], torch.zeros_like(delta[~pair_mask]))
    assert torch.equal((base_logits + delta)[~pair_mask], base_logits[~pair_mask])
    assert torch.equal(delta.diagonal(dim1=-2, dim2=-1), torch.zeros(2, 4, 5))
    assert delta[pair_mask].abs().sum() > 0


def test_grouped_keys_match_expanded_keys():
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.copy_(torch.tensor([0.1, 0.2, -0.3, 0.4]))
    grouped = module(query, key, gaze, mask)
    expanded = module(query, key.repeat_interleave(2, dim=1), gaze, mask)
    torch.testing.assert_close(grouped, expanded, atol=0, rtol=0)


def test_masked_nonfinite_padding_cannot_contaminate_values_or_gradients():
    query, key, gaze, mask = _inputs()
    query = query.masked_fill(~mask[:, None, :, None].bool(), float("nan")).requires_grad_()
    key = key.masked_fill(~mask[:, None, :, None].bool(), float("inf")).requires_grad_()
    gaze = gaze.masked_fill(~mask.bool(), -float("inf")).requires_grad_()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.fill_(0.3)
    output = module(query, key, gaze, mask)
    output.square().sum().backward()
    for tensor in (output, query.grad, key.grad, gaze.grad, module.alpha.grad):
        assert torch.isfinite(tensor).all()
    assert torch.equal(gaze.grad[~mask.bool()], torch.zeros_like(gaze.grad[~mask.bool()]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_and_fp32_parameters_survive_model_conversion(dtype):
    query, key, gaze, mask = _inputs(dtype)
    module = TRTRelativeRotation(4).to(dtype=dtype)
    assert module.alpha.dtype == torch.float32
    zero_delta = module(query, key, gaze, mask)
    assert zero_delta.dtype == dtype
    assert torch.equal(zero_delta, torch.zeros_like(zero_delta))
    with torch.no_grad():
        module.alpha.fill_(0.2)
    delta = module(query, key, gaze, mask)
    reference = _reference(query, key, gaze, mask, module.alpha.detach()).to(dtype)
    torch.testing.assert_close(delta, reference, atol=2e-3 if dtype == torch.bfloat16 else 3e-4, rtol=2e-3)
    delta.float().square().sum().backward()
    assert module.alpha.grad.dtype == torch.float32
    assert torch.isfinite(module.alpha.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_conversion_preserves_nonzero_fp32_parameter_and_gradient_values_exactly(dtype):
    module = TRTRelativeRotation(4)
    original_alpha = torch.tensor([0.123456789, -0.876543219, 1.000000119, -0.00012345678])
    original_gradient = torch.tensor([-0.234567891, 0.765432119, -1.000000238, 0.00023456789])
    with torch.no_grad():
        module.alpha.copy_(original_alpha)
    module.alpha.grad = original_gradient.clone()
    parameter = module.alpha
    module.to(dtype=dtype)
    assert module.alpha is parameter
    assert module.alpha.dtype == torch.float32
    assert module.alpha.grad.dtype == torch.float32
    assert torch.equal(module.alpha, original_alpha)
    assert torch.equal(module.alpha.grad, original_gradient)
    module.float()
    assert torch.equal(module.alpha, original_alpha)
    assert torch.equal(module.alpha.grad, original_gradient)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_conversion_preserves_fp32_values_when_moving_devices(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable.")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS device is unavailable.")
    module = TRTRelativeRotation(2)
    original_alpha = torch.tensor([0.123456789, -0.876543219])
    original_gradient = torch.tensor([-0.234567891, 0.765432119])
    with torch.no_grad():
        module.alpha.copy_(original_alpha)
    module.alpha.grad = original_gradient.clone()
    module.to(device=device, dtype=torch.float16)
    assert module.alpha.device.type == device
    assert module.alpha.grad.device.type == device
    assert module.alpha.dtype == torch.float32
    assert module.alpha.grad.dtype == torch.float32
    assert torch.equal(module.alpha.detach().cpu(), original_alpha)
    assert torch.equal(module.alpha.grad.cpu(), original_gradient)
    module.cpu()
    assert torch.equal(module.alpha, original_alpha)
    assert torch.equal(module.alpha.grad, original_gradient)


def test_outer_autocast_does_not_reduce_internal_precision():
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.fill_(0.07)
    expected = module(query, key, gaze, mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module(query, key, gaze, mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("batch_size,length", [(2, 5), (2, 0), (0, 5), (0, 0)])
def test_all_masked_or_empty_inputs_are_finite_zero(batch_size, length):
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.fill_(0.4)
    output = module(
        torch.full((batch_size, 4, length, 8), float("nan")),
        torch.full((batch_size, 2, length, 8), float("inf")),
        torch.full((batch_size, length), float("nan")),
        torch.zeros(batch_size, length),
    )
    assert output.shape == (batch_size, 4, length, length)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert torch.equal(module.alpha.grad, torch.zeros(4))


def test_one_anchor_and_equal_trt_have_no_relative_effect():
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        module.alpha.fill_(0.8)
    single_mask = torch.zeros_like(mask)
    single_mask[:, 2] = 1
    assert torch.equal(module(query, key, gaze, single_mask), torch.zeros(2, 4, 5, 5))
    for equal_trt in (1.0, 9_999_999.0):
        equal_delta = module(query, key, torch.full_like(gaze, equal_trt), mask)
        assert torch.equal(equal_delta, torch.zeros_like(equal_delta))


def test_alpha_gradient_matches_central_finite_difference_at_zero():
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    weights = torch.arange(200).reshape(2, 4, 5, 5).float() / 200
    (module(query, key, gaze, mask) * weights).sum().backward()
    analytic = module.alpha.grad.detach().clone()
    numerical = torch.zeros(4)
    for head in range(4):
        with torch.no_grad():
            module.alpha[head] = 0.001
            plus = (module(query, key, gaze, mask) * weights).sum()
            module.alpha[head] = -0.001
            minus = (module(query, key, gaze, mask) * weights).sum()
            module.alpha[head] = 0
        numerical[head] = (plus - minus) / 0.002
    torch.testing.assert_close(analytic, numerical, atol=6e-4, rtol=2e-3)


@pytest.mark.parametrize("target,index", [("query", (0, 0, 0, 2)), ("key", (0, 0, 1, 3)), ("gaze", (0, 1))])
def test_nonzero_rotation_input_gradients_match_finite_difference(target, index):
    """Validate Q, grouped K, and TRT derivatives against independent perturbed forwards."""

    query, key, gaze, mask = _inputs()
    values = {"query": query, "key": key, "gaze": gaze}
    tensor = values[target].requires_grad_()
    module = TRTRelativeRotation(4, scale=1.7)
    with torch.no_grad():
        module.alpha.copy_(torch.tensor([0.3, -0.2, 0.05, 0.4]))
    weights = torch.arange(200).reshape(2, 4, 5, 5).float() / 200
    (module(query, key, gaze, mask) * weights).sum().backward()
    analytic = float(tensor.grad[index])
    original = float(tensor[index].detach())
    epsilon = 0.002
    with torch.no_grad():
        tensor[index] = original + epsilon
        plus = float((module(query, key, gaze, mask) * weights).sum())
        tensor[index] = original - epsilon
        minus = float((module(query, key, gaze, mask) * weights).sum())
        tensor[index] = original
    assert abs(analytic) > 1e-5
    assert analytic == pytest.approx((plus - minus) / (2 * epsilon), abs=3e-4, rel=3e-3)


@pytest.mark.parametrize("num_heads", [0, -1, 2.5, True, "4"])
def test_rejects_invalid_head_counts(num_heads):
    with pytest.raises(ValueError, match="num_heads"):
        TRTRelativeRotation(num_heads)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "1", 1e-100, 1e100])
def test_rejects_invalid_scales(scale):
    with pytest.raises(ValueError, match="scale"):
        TRTRelativeRotation(4, scale=scale)
    with pytest.raises(ValueError, match="scale"):
        gaze_relation_contract("trt-relative-rotation", scale=scale)


@pytest.mark.parametrize("mask_value", [0.5, -1, float("nan"), float("inf")])
def test_rejects_nonbinary_masks(mask_value):
    query, key, gaze, mask = _inputs()
    mask = mask.float()
    mask[0, 0] = mask_value
    with pytest.raises(ValueError, match="binary"):
        TRTRelativeRotation(4)(query, key, gaze, mask)


@pytest.mark.parametrize("target", ["query", "key", "gaze", "alpha"])
def test_rejects_nonfinite_eligible_values(target):
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    with torch.no_grad():
        if target == "query":
            query[0, 0, 0, 0] = float("nan")
        elif target == "key":
            key[0, 0, 0, 0] = float("inf")
        elif target == "gaze":
            gaze[0, 0] = float("nan")
        else:
            module.alpha[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module(query, key, gaze, mask)


def test_rejects_incompatible_tensor_shapes_and_types():
    query, key, gaze, mask = _inputs()
    module = TRTRelativeRotation(4)
    invalid_inputs = [
        (query[:, 0], key, gaze, mask),
        (query[:, :2], key, gaze, mask),
        (query, key[:, :, :-1], gaze, mask),
        (query[..., :7], key[..., :7], gaze, mask),
        (query, key.repeat(1, 3, 1, 1), gaze, mask),
        (query, key.double(), gaze, mask),
        (query.long(), key.long(), gaze, mask),
        (query, key, gaze[:, :-1], mask),
        (query, key, gaze.long(), mask),
        (query, key, gaze, mask[:, :-1]),
        (query, key, gaze, mask.to(torch.complex64)),
        (query, key, gaze, None),
    ]
    for arguments in invalid_inputs:
        with pytest.raises(ValueError):
            module(*arguments)


def test_contract_roundtrip_and_explicit_metadata():
    assert gaze_relation_contract() == {"version": 1, "method": "none"}
    for method in ("none", "trt-relative-rotation"):
        contract = gaze_relation_contract(method)
        assert validate_gaze_relation_contract(contract) == contract
    contract = gaze_relation_contract("trt-relative-rotation", scale=2.0)
    assert contract["scale"] == 2.0
    assert contract["geometry"] == "native_rope_then_split_half"
    assert contract["scope"] == "mapped_text_pairs"
    assert contract["alpha_init"] == 0.0


def test_rejects_invalid_contracts():
    enabled = gaze_relation_contract("trt-relative-rotation")
    invalid_contracts = [
        None,
        {},
        {"version": 1, "method": "unknown"},
        {"version": 1, "method": "none", "scale": 1.0},
        {"method": "none"},
        {**enabled, "version": True},
        {**enabled, "version": 2},
        {**enabled, "geometry": "interleaved"},
        {**enabled, "scope": "all_tokens"},
        {**enabled, "alpha_init": 0},
        {**enabled, "compute_dtype": "float16"},
        {**enabled, "scale": -1},
        {**enabled, "unexpected": True},
        {key: value for key, value in enabled.items() if key != "scale"},
    ]
    for contract in invalid_contracts:
        with pytest.raises(ValueError):
            validate_gaze_relation_contract(contract)
    with pytest.raises(ValueError):
        gaze_relation_contract("unknown")
    with pytest.raises(ValueError, match="requires"):
        gaze_relation_contract("none", scale=2.0)

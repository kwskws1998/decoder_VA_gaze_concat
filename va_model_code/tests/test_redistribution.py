"""Numerical and mask contracts for modular asymmetric Gaussian TRT redistribution."""

from __future__ import annotations

import math

import pytest
import torch

from va_model_code.decoder_va.redistribution import (
    AsymGaussianRedistributor,
    GazeRedistributor,
    redistribution_contract,
    validate_redistribution_contract,
)


def _reference(values, mask, left=1.0, right=1.0, floor=1e-6):
    """Evaluate the supplied kernel's source-target equation independently in loops."""

    result = torch.zeros_like(values, dtype=torch.float64)
    for batch in range(values.shape[0]):
        positions = [index for index in range(values.shape[1]) if mask[batch, index]]
        for source in positions:
            weights = [
                math.exp(-0.5 * ((target - source) / ((left if target < source else right) + floor)) ** 2)
                for target in positions
            ]
            total = sum(weights)
            for target, weight in zip(positions, weights):
                result[batch, target] += weight / total * float(values[batch, source])
    return result.to(values.dtype)


def test_matches_source_equation_with_signed_values_and_gaps():
    values = torch.tensor([[2.0, -1.0, 300.0, 0.0, 4.0], [0.0, 1.0, 2.0, 3.0, 9.0]])
    mask = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 1, 1, 0]])
    actual = AsymGaussianRedistributor(0.6, 2.3)(values, mask)
    torch.testing.assert_close(actual, _reference(values, mask, 0.6, 2.3), atol=5e-7, rtol=5e-7)


def test_asymmetric_orientation():
    values = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    mask = torch.ones_like(values)
    actual = AsymGaussianRedistributor(0.25, 2.0)(values, mask)
    assert actual[0, 3] > actual[0, 1]
    assert actual[0, 4] > actual[0, 0]


def test_mass_conservation_for_signed_sources():
    values = torch.tensor([[2.0, -1.0, 0.0, 1000.0, 5.0], [-4.0, -2.0, 3.0, -5.0, -1.0]])
    mask = torch.tensor([[1, 1, 1, 0, 1], [1, 1, 1, 1, 1]])
    output = AsymGaussianRedistributor(0.5, 3.0)(values, mask)
    torch.testing.assert_close(output.sum(1), (values * mask).sum(1))
    assert output[1].min() < 0


def test_masked_nonfinite_values_never_contaminate_valid_values():
    values = torch.tensor([[1.0, float("nan"), 0.0, float("inf"), -2.0, -float("inf")]])
    mask = torch.tensor([[1, 0, 1, 0, 1, 0]])
    output = AsymGaussianRedistributor()(values, mask)
    assert torch.isfinite(output).all()
    assert torch.equal(output[~mask.bool()], torch.zeros(3))
    torch.testing.assert_close(output.sum(), torch.tensor(-1.0))


def test_active_zero_remains_a_destination():
    values = torch.tensor([[1.0, 0.0, 999.0]])
    mask = torch.tensor([[1, 1, 0]])
    output = AsymGaussianRedistributor()(values, mask)
    assert output[0, 1] > 0
    assert output[0, 2] == 0
    torch.testing.assert_close(output.sum(), torch.tensor(1.0))


def test_padding_side_and_batch_composition_invariance():
    module = AsymGaussianRedistributor(1.4, 0.8)
    short = torch.tensor([[1.0, 0.0, 2.0]])
    short_mask = torch.tensor([[1, 0, 1]])
    expected = module(short, short_mask)
    right_padded = torch.tensor([[1.0, 0.0, 2.0, float("nan"), 100.0], [4.0, 3.0, 2.0, 1.0, 0.0]])
    right_mask = torch.tensor([[1, 0, 1, 0, 0], [1, 1, 1, 1, 1]])
    left_padded = torch.tensor([[float("inf"), 9.0, 1.0, 0.0, 2.0]])
    left_mask = torch.tensor([[0, 0, 1, 0, 1]])
    torch.testing.assert_close(module(right_padded, right_mask)[:1, :3], expected)
    torch.testing.assert_close(module(left_padded, left_mask)[:, 2:], expected)


def test_gaps_preserve_aligned_qwen_coordinates():
    module = AsymGaussianRedistributor()
    compact = module(torch.tensor([[1.0, 0.0]]), torch.ones(1, 2))
    gapped = module(torch.tensor([[1.0, 500.0, 0.0]]), torch.tensor([[1, 0, 1]]))
    assert gapped[0, 2] < compact[0, 1]
    torch.testing.assert_close(gapped, _reference(torch.tensor([[1.0, 500.0, 0.0]]), torch.tensor([[1, 0, 1]])))


@pytest.mark.parametrize("shape", [(2, 5), (2, 0), (0, 5), (0, 0)])
def test_empty_and_all_masked_rows(shape):
    output = AsymGaussianRedistributor()(torch.full(shape, float("nan")), torch.zeros(shape))
    assert output.shape == shape
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros(shape))


def test_required_explicit_mask():
    module = AsymGaussianRedistributor()
    with pytest.raises(TypeError):
        module(torch.ones(1, 2))
    with pytest.raises(ValueError, match="explicit binary"):
        module(torch.ones(1, 2), None)


@pytest.mark.parametrize("mask", [torch.ones(2, 1), torch.tensor([[1.0, 0.5]]), torch.tensor([[1.0, float("nan")]]), torch.tensor([[1, -1]]), torch.ones(1, 2, dtype=torch.complex64)])
def test_rejects_invalid_mask(mask):
    with pytest.raises(ValueError, match="mask"):
        AsymGaussianRedistributor()(torch.ones(1, 2), mask)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.int64, torch.float32, torch.float64])
def test_accepts_binary_mask_dtypes(dtype):
    output = AsymGaussianRedistributor()(torch.ones(1, 2), torch.tensor([[1, 0]], dtype=dtype))
    torch.testing.assert_close(output, torch.tensor([[1.0, 0.0]]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_valid_values(value):
    with pytest.raises(ValueError, match="finite"):
        AsymGaussianRedistributor()(torch.tensor([[value]]), torch.ones(1, 1))


def test_rejects_valid_float64_values_outside_float32_range():
    with pytest.raises(ValueError, match="representable in float32"):
        AsymGaussianRedistributor()(torch.tensor([[1e100]], dtype=torch.float64), torch.ones(1, 1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_and_autocast_contract(dtype):
    module = AsymGaussianRedistributor(0.7, 2.3)
    values = torch.tensor([[1.0, -2.0, 4.0, 0.0]], dtype=dtype)
    mask = torch.tensor([[1, 1, 1, 0]])
    expected = module(values.float(), mask).to(dtype)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = module(values, mask)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_gradients_reach_both_widths_and_only_valid_inputs():
    module = AsymGaussianRedistributor(0.7, 1.4)
    values = torch.tensor([[1.0, float("nan"), 3.0, 0.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 1, 1]])
    output = module(values, mask)
    loss = (output * torch.tensor([[2.0, 7.0, -1.0, 0.5, 4.0]])).sum()
    loss.backward()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs() > 1e-5
    assert torch.isfinite(values.grad).all()
    assert values.grad[0, 1] == 0


def test_single_valid_position_is_identity_with_zero_width_gradients():
    module = AsymGaussianRedistributor(0.7, 1.4)
    values = torch.tensor([[float("nan"), 2.5, float("inf")]])
    mask = torch.tensor([[0, 1, 0]])

    output = module(values, mask)
    output.sum().backward()

    torch.testing.assert_close(output, torch.tensor([[0.0, 2.5, 0.0]]))
    assert module.log_sigma_left.grad is not None
    assert module.log_sigma_right.grad is not None
    assert module.log_sigma_left.grad.item() == 0.0
    assert module.log_sigma_right.grad.item() == 0.0


def test_width_gradients_match_finite_difference():
    module = AsymGaussianRedistributor(0.7, 1.4)
    values = torch.tensor([[1.0, 0.0, 3.0, -2.0]])
    mask = torch.ones_like(values)
    target = torch.tensor([[1.0, -3.0, 2.0, 0.5]])
    (module(values, mask) * target).sum().backward()
    epsilon = 1e-3
    for parameter in module.parameters():
        original = parameter.detach().clone()
        with torch.no_grad():
            parameter.copy_(original + epsilon)
            plus = (module(values, mask) * target).sum()
            parameter.copy_(original - epsilon)
            minus = (module(values, mask) * target).sum()
            parameter.copy_(original)
        finite_difference = (plus - minus) / (2 * epsilon)
        torch.testing.assert_close(parameter.grad, finite_difference, rtol=3e-3, atol=1e-3)


@pytest.mark.parametrize("log_width", [-1000.0, -100.0, 100.0, 1000.0])
def test_extreme_learned_widths_have_finite_outputs_and_gradients(log_width):
    module = AsymGaussianRedistributor(min_sigma=1e-30)
    with torch.no_grad():
        module.log_sigma_left.fill_(log_width)
        module.log_sigma_right.fill_(log_width)
    values = torch.tensor([[1.0, 3.0, 0.0]], requires_grad=True)
    output = module(values, torch.ones_like(values))
    output.square().sum().backward()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.sum(), values.sum())
    for parameter in module.parameters():
        assert torch.isfinite(parameter.grad).all()


def test_zero_mass_rows_do_not_break_backward():
    module = AsymGaussianRedistributor()
    values = torch.tensor([[float("nan"), float("inf")], [1.0, 0.0]])
    output = module(values, torch.tensor([[0, 0], [1, 1]]))
    output.square().sum().backward()
    assert torch.equal(output[0], torch.zeros(2))
    assert all(torch.isfinite(parameter.grad) for parameter in module.parameters())


def test_sigma_initialization_and_floor_follow_source():
    module = AsymGaussianRedistributor(0.4, 1.7, 0.25)
    torch.testing.assert_close(module.log_sigma_left, torch.tensor(math.log(0.4)))
    torch.testing.assert_close(module.sigma_left, torch.tensor(0.65))
    torch.testing.assert_close(module.sigma_right, torch.tensor(1.95))


def test_disabled_wrapper_is_exact_identity_and_parameter_free():
    module = GazeRedistributor({"method": "none"}, (3,))
    values = torch.tensor([[[float("nan")]]])
    assert module(values, None) is values
    assert list(module.parameters()) == []
    assert module.state_dict() == {}


@pytest.mark.parametrize("indices", [(3,), (1, 3), (3, 1), (0, 1, 2, 3, 4)])
def test_wrapper_replaces_trt_only_without_mutating_cache(indices):
    module = GazeRedistributor(redistribution_contract("asym-gaussian"), indices)
    values = torch.arange(3 * len(indices), dtype=torch.float32).reshape(1, 3, len(indices))
    original = values.clone()
    mask = torch.tensor([[1, 1, 0]])
    actual = module(values, mask)
    trt = indices.index(3)
    torch.testing.assert_close(actual[:, :, trt], module.kernel(values[:, :, trt], mask))
    for channel in range(len(indices)):
        if channel != trt:
            assert torch.equal(actual[:, :, channel], values[:, :, channel])
    assert torch.equal(values, original)
    actual.square().sum().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_wrapper_checkpoint_roundtrip_preserves_learned_widths():
    config = redistribution_contract("asym-gaussian")
    module = GazeRedistributor(config, (3,))
    with torch.no_grad():
        module.kernel.log_sigma_left.fill_(-0.5)
        module.kernel.log_sigma_right.fill_(1.0)
    restored = GazeRedistributor(config, (3,))
    restored.load_state_dict(module.state_dict(), strict=True)
    values = torch.tensor([[[2.0], [1.0], [-1.0]]])
    mask = torch.ones(1, 3)
    torch.testing.assert_close(restored(values, mask), module(values, mask), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(1, 3), (1, 3, 2), (1, 3, 1, 1)])
def test_enabled_wrapper_validates_selected_feature_shape(shape):
    module = GazeRedistributor(redistribution_contract("asym-gaussian"), (3,))
    with pytest.raises(ValueError, match="selected_ET2_features"):
        module(torch.ones(shape), torch.ones(1, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_autocast_is_disabled_for_kernel_math():
    module = AsymGaussianRedistributor(0.7, 2.3).cuda()
    values = torch.tensor([[1.0, -2.0, 4.0, 0.0]], device="cuda", dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1, 0]], device="cuda")
    expected = module(values, mask)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        actual = module(values, mask)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    assert all(torch.isfinite(parameter.grad) for parameter in module.parameters())


def test_contract_roundtrip_and_exact_metadata():
    contract = redistribution_contract("asym-gaussian", init_sigma_left=0.2, init_sigma_right=5.0)
    canonical = validate_redistribution_contract(contract, gaze_fusion="prefix-concat", feature_indices=(1, 3))
    assert canonical == contract
    assert canonical["position_space"] == "aligned_qwen_tokens"
    assert canonical["mask_policy"] == "valid_gaze_sources_and_targets"
    assert canonical["normalization"] == "per_source_unit_mass"
    assert canonical["trainable"] is True
    assert canonical["compute_dtype"] == "float32"
    assert redistribution_contract() == {"method": "none"}
    assert validate_redistribution_contract({"method": "none"}, gaze_fusion="none", feature_indices=(1,)) == {"method": "none"}


@pytest.mark.parametrize("name", ["init_sigma_left", "init_sigma_right", "min_sigma"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True, "1.0"])
def test_invalid_numeric_parameters(name, value):
    with pytest.raises(ValueError, match="finite positive"):
        redistribution_contract("asym-gaussian", **{name: value})
    with pytest.raises(ValueError, match="finite positive"):
        AsymGaussianRedistributor(**{name: value})


@pytest.mark.parametrize("name", ["init_sigma_left", "init_sigma_right", "min_sigma"])
def test_disabled_rejects_ignored_nondefault_parameters(name):
    with pytest.raises(ValueError, match="require asym-gaussian"):
        redistribution_contract("none", **{name: 2.0})


def test_enabled_requires_gaze_concat_and_trt():
    with pytest.raises(ValueError, match="requires prefix-concat"):
        redistribution_contract("asym-gaussian", gaze_fusion="none")
    with pytest.raises(ValueError, match="requires the TRT"):
        redistribution_contract("asym-gaussian", feature_indices=(0, 1))


@pytest.mark.parametrize("operation", ["extra", "missing", "bad_metadata", "wrong_boolean"])
def test_saved_contract_rejects_silent_metadata_changes(operation):
    contract = redistribution_contract("asym-gaussian")
    if operation == "extra":
        contract["unknown"] = 1
    elif operation == "missing":
        del contract["normalization"]
    elif operation == "bad_metadata":
        contract["position_space"] = "compacted_words"
    else:
        contract["trainable"] = 1
    with pytest.raises(ValueError):
        validate_redistribution_contract(contract, gaze_fusion="prefix-concat", feature_indices=(3,))


@pytest.mark.parametrize("contract", [None, {}, {"method": "typo"}, {"method": "none", "init_sigma_left": 1.0}])
def test_invalid_saved_contracts(contract):
    with pytest.raises(ValueError):
        validate_redistribution_contract(contract, gaze_fusion="prefix-concat", feature_indices=(3,))

"""Offline kernel audit without model downloads or production edits."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import sys

import torch

root = Path('/Users/wansookim/Documents/decoder_based_va_prediction')
sys.path.insert(0, str(root))
from va_model_code.decoder_va.redistribution import AsymGaussianRedistributor, GazeRedistributor, redistribution_contract


def main():
    """Compare reference math, reflection, boundary behavior, and sparse gaps."""
    spec = importlib.util.spec_from_file_location('reference_kernel', root / 'diagnostics/sigma_reference_20260916/reference_source/models/asym_gaussian_redistributor.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(1729)
    errors = {'output': 0.0, 'gradient': 0.0, 'mirror': 0.0}
    cases = 0
    for length in (1, 2, 7, 31, 200):
        for left, right in ((0.25, 1), (1, 0.25), (0.5, 2), (2, 0.5), (1, 4), (4, 1), (0.001, 1000), (3.7, 0.93)):
            values = torch.randn(3, length)
            mask = torch.rand(3, length) > 0.35
            kernel = AsymGaussianRedistributor(left, right)
            with contextlib.redirect_stdout(io.StringIO()):
                source = reference.AsymGaussianRedistributor(left, right)
            output, expected = kernel(values, mask), source(values, mask)
            errors['output'] = max(errors['output'], (output - expected).abs().max().item())
            loss_weights = torch.randn_like(output)
            (output * loss_weights).sum().backward()
            (expected * loss_weights).sum().backward()
            for actual_parameter, expected_parameter in zip(kernel.parameters(), source.parameters()):
                errors['gradient'] = max(errors['gradient'], (actual_parameter.grad - expected_parameter.grad).abs().max().item())
            mirrored = AsymGaussianRedistributor(right, left)(values.flip(1), mask.flip(1)).flip(1)
            errors['mirror'] = max(errors['mirror'], (output - mirrored).abs().max().item())
            cases += 1
    boundary = []
    for length in (5, 10, 30):
        for left, right in ((0.25, 1), (0.5, 2), (1, 4), (4, 1)):
            values = torch.ones(1, length)
            output = AsymGaussianRedistributor(left, right)(values, torch.ones_like(values)).detach()[0]
            boundary.append({'length': length, 'left': left, 'right': right, 'output': output.tolist(), 'sum': output.sum().item()})
    sparse = []
    for gap in (1, 2, 3):
        for left, right in ((0.25, 1), (0.1, 0.1), (1, 1)):
            values = torch.zeros(1, 2 * gap + 1)
            values[0, gap] = 1
            mask = torch.zeros_like(values)
            mask[0, ::gap] = 1
            kernel = AsymGaussianRedistributor(left, right)
            output = kernel(values, mask)
            (output * torch.arange(output.shape[1])).sum().backward()
            sparse.append({'gap': gap, 'left': left, 'right': right, 'left_center_right': output.detach()[0, ::gap].tolist(), 'left_log_gradient': kernel.log_sigma_left.grad.item(), 'right_log_gradient': kernel.log_sigma_right.grad.item()})
    modes = []
    for method in ('none', 'fixed-gaussian', 'asym-gaussian'):
        module = GazeRedistributor(redistribution_contract(method), (3,))
        modes.append({'method': method, 'num_parameters': sum(p.numel() for p in module.parameters()), 'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)})
    assert errors['output'] < 1e-5
    assert errors['gradient'] < 1e-5
    assert errors['mirror'] < 1e-5
    print(json.dumps({'torch_version': torch.__version__, 'device': 'cpu', 'random_cases': cases, 'max_abs_errors': errors, 'boundary_constant_trt': boundary, 'sparse_impulse': sparse, 'modes': modes}, indent=2))


if __name__ == '__main__':
    main()

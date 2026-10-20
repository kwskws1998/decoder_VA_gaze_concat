"""Compare reviewed Gaussian implementations and schedules on CPU, without LLM weights.

Reference files are checked against the extraction manifest. Only the standalone
kernel and the reviewed optimizer method are executed from the reference tree.
Synthetic recovery is a kernel check, not a reproduction of either downstream task.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REFERENCE = HERE / "reference_source"


def load_file(name, path):
    """Load a reviewed local module while retaining its exact implementation."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_kernels():
    """Verify reference hashes and bypass only the VA package's eager model imports."""
    manifest = json.loads((HERE / "source_manifest.json").read_text())
    for name, metadata in manifest["files"].items():
        data = (REFERENCE / name).read_bytes()
        if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Reference file changed: {name}")
    package = ModuleType("sigma_audit_va")
    package.__path__ = [str(PROJECT / "va_model_code/decoder_va")]
    sys.modules[package.__name__] = package
    current = load_file("sigma_audit_va.redistribution", Path(package.__path__[0]) / "redistribution.py")
    reference = load_file("sigma_audit_reference", REFERENCE / "models/asym_gaussian_redistributor.py")
    return reference.AsymGaussianRedistributor, current.AsymGaussianRedistributor


def new_kernel(kernel_class, left=1.0, right=1.0):
    """Suppress the reference constructor's informational printing."""
    with redirect_stdout(io.StringIO()):
        return kernel_class(left, right)


def compare_numerics(reference, current):
    """Test identical finite signed inputs, binary masks, and independent readout gradients."""
    rng = torch.Generator().manual_seed(20260916)
    cases = []
    for length in (1, 2, 8, 31, 200):
        values = torch.randn(3, length, generator=rng)
        readout = torch.randn(3, length, generator=rng)
        masks = {
            "dense": torch.ones_like(values, dtype=torch.bool),
            "right_padded": torch.arange(length)[None, :] < torch.tensor([length, max(1, length // 2), 0])[:, None],
            "sparse_interior": torch.rand(3, length, generator=rng) > 0.4,
            "all_masked": torch.zeros_like(values, dtype=torch.bool),
        }
        for mask_name, mask in masks.items():
            for left, right in ((1, 1), (0.5, 2), (2, 0.5), (0.01, 40), (5, 100)):
                modules = [new_kernel(cls, left, right) for cls in (reference, current)]
                outputs = [module(values, mask) for module in modules]
                gradients = [torch.stack(torch.autograd.grad((output * readout).mean(), tuple(module.parameters()))) for module, output in zip(modules, outputs)]
                torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(gradients[0], gradients[1], atol=2e-6, rtol=2e-5)
                for output in outputs:
                    torch.testing.assert_close(output.sum(dim=1), (values * mask).sum(dim=1), atol=2e-5, rtol=2e-5)
                    if torch.count_nonzero(output[~mask]):
                        raise AssertionError("Masked destinations must be zero.")
                cases.append({
                    "length": length, "mask": mask_name, "sigma": [left, right],
                    "output_max_abs_error": (outputs[0] - outputs[1]).abs().max().item(),
                    "log_sigma_gradient_max_abs_error": (gradients[0] - gradients[1]).abs().max().item(),
                })
    return {
        "case_count": len(cases),
        "output_max_abs_error": max(c["output_max_abs_error"] for c in cases),
        "log_sigma_gradient_max_abs_error": max(c["log_sigma_gradient_max_abs_error"] for c in cases),
        "cases": cases,
    }


def recover_synthetic_target(reference, current):
    """Fit a known right-heavy target with identical AdamW settings for both kernels."""
    rng = torch.Generator().manual_seed(42)
    values = torch.rand(8, 24, generator=rng) * 2
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[1, 18:] = False
    mask[2, 1::3] = False
    target_widths = (0.5, 4.0)
    target = new_kernel(reference, *target_widths)(values, mask).detach()
    runs = []
    for lr in (0.001, 0.05):
        for name, cls in (("reference", reference), ("current", current)):
            model = new_kernel(cls)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
            snapshots = []
            for step in range(3001):
                prediction = model(values, mask)
                loss = (prediction - target).square().mean()
                if step in (0, 100, 500, 1000, 3000):
                    snapshots.append({"step": step, "mse": loss.item(), "left": model.sigma_left.item(), "right": model.sigma_right.item()})
                if step == 3000:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            runs.append({"implementation": name, "constant_lr": lr, "snapshots": snapshots})
    for lr in (0.001, 0.05):
        pair = [r for r in runs if r["constant_lr"] == lr]
        for a, b in zip(pair[0]["snapshots"], pair[1]["snapshots"]):
            torch.testing.assert_close(torch.tensor([a["left"], a["right"]]), torch.tensor([b["left"], b["right"]]), atol=2e-5, rtol=2e-5)
    for run in runs:
        if run["constant_lr"] == 0.05:
            torch.testing.assert_close(torch.tensor([run["snapshots"][-1]["left"], run["snapshots"][-1]["right"]]), torch.tensor(target_widths), atol=2e-3, rtol=0)
    return {"target_widths": target_widths, "real_data_or_downstream_model": False, "runs": runs}


def compare_schedules(current):
    """Execute the exact reference optimizer method with hypothetical single-GPU counts."""
    path = REFERENCE / "trainers/reward_trainer_general.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardTrainerConstructorGeneral")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_optmizer_scheduler")
    scope = {"torch": torch, "math": math, "LambdaLR": torch.optim.lr_scheduler.LambdaLR}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    steps = 4490
    warmup = math.ceil(steps * 0.1)
    current_lrs = [0.001 * (step / warmup if step < warmup else (steps - step) / (steps - warmup)) for step in range(steps)]
    current_integral = sum(current_lrs)
    schedules = []
    for accumulation in (1, 8):
        samples = steps * accumulation // 2
        state = SimpleNamespace(model=new_kernel(current), sigma_learnable=True, sigma_lr=0.05, learning_rate=5e-5, weight_decay=0.1, batch_size=1, train_epochs=2, min_lr_ratio=0.7, sigma_lr_scheduler_type="cosine_with_min_lr")
        with redirect_stdout(io.StringIO()):
            scope["load_optmizer_scheduler"](state, samples)
        rates = []
        for step in range(steps):
            rates.append(state.optimizer.param_groups[-1]["lr"])
            state.optimizer.step()
            state.scheduler.step()
        schedules.append({
            "assumed_accumulation": accumulation,
            "hypothetical_sample_count": samples,
            "actual_optimizer_steps": steps,
            "reference_scheduler_horizon": samples * 2,
            "reference_warmup_steps": int(samples * 2 * 0.01),
            "reference_last_applied_lr": rates[-1],
            "reference_lr_after_final_update": state.optimizer.param_groups[-1]["lr"],
            "reference_sum_applied_lrs": sum(rates),
            "reference_over_current_sum_lr": sum(rates) / current_integral,
        })
    return {
        "scope": "Hypothetical equal 4490 optimizer steps, single GPU, no skipped steps; not historical training logs. LR sums are not parameter changes.",
        "current_peak_lr": max(current_lrs),
        "current_warmup_steps": warmup,
        "current_sum_applied_lrs": current_integral,
        "reference": schedules,
    }


def main():
    """Write reproducible numerical evidence and concise completion output."""
    torch.set_num_threads(1)
    reference, current = load_kernels()
    paths = [REFERENCE / "models/asym_gaussian_redistributor.py", PROJECT / "va_model_code/decoder_va/redistribution.py"]
    result = {
        "torch_version": torch.__version__, "device": "cpu", "dtype": "float32",
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "numerical_equivalence": compare_numerics(reference, current),
        "synthetic_recovery": recover_synthetic_target(reference, current),
        "schedule_comparison": compare_schedules(current),
    }
    output = HERE / "comparison.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "numerical_equivalence": {k: v for k, v in result["numerical_equivalence"].items() if k != "cases"}, "synthetic_final": [{"implementation": r["implementation"], "lr": r["constant_lr"], **r["snapshots"][-1]} for r in result["synthetic_recovery"]["runs"]], "schedules": result["schedule_comparison"]}, indent=2))


if __name__ == "__main__":
    main()

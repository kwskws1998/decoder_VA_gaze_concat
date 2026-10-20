"""Audit sampled sigma optimization and test the current kernel's limiting behavior."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType
import zipfile

import numpy as np
import torch

from compare_results import load_archive


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def load_recorded_runs():
    """Verify the same archives used for comparison before reading update records."""
    comparison = json.loads((HERE / "comparison.json").read_text())
    runs = {}
    for name, digest in comparison["inputs"].items():
        path = Path(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with zipfile.ZipFile(path) as archive:
            nested = [item for item in archive.namelist() if item.endswith(".zip")]
            if nested:
                for item in nested:
                    with zipfile.ZipFile(io.BytesIO(archive.read(item))) as inner:
                        label, data = load_archive(inner)
                        runs[label] = data
            else:
                label, data = load_archive(archive)
                runs[label] = data
    return runs


def adam_update(row, side):
    """Reconstruct an AdamW update assuming beta1=.9, beta2=.999, eps=1e-8, wd=0."""
    key = f"log_sigma_{side}"
    before = row["optimizer_before_update"][key]
    step = (before["step"] or 0) + 1
    gradient = row["gradient_after_clipping"][key]
    first = 0.9 * (before["exp_avg"] or 0) + 0.1 * gradient
    second = 0.999 * (before["exp_avg_sq"] or 0) + 0.001 * gradient**2
    corrected_first = first / (1 - 0.9**step)
    corrected_second_sqrt = math.sqrt(second / (1 - 0.999**step))
    predicted = -before["lr"] * corrected_first / (corrected_second_sqrt + 1e-8)
    observed = row["delta_log_sigma"][key]
    parameter = np.float32(row["log_sigma_before"][key])
    return {
        "step": row["step"], "epoch": row["epoch"],
        "sigma_before": math.exp(float(parameter)) + 1e-6,
        "gradient_before_clipping": row["gradient_before_clipping"][key],
        "gradient_after_clipping": gradient,
        "exp_avg_before": before["exp_avg"], "corrected_first_moment": corrected_first,
        "corrected_second_moment_sqrt": corrected_second_sqrt,
        "lr": before["lr"], "predicted_delta": predicted,
        "observed_delta": observed, "absolute_error": abs(predicted - observed),
        "fp32_parameter_spacing": abs(float(np.spacing(parameter))),
    }


def analyze_updates(runs):
    """Summarize actual sampled training gradients separately from fixed-example probes."""
    result = {}
    for name in ("learned_lr0.001", "learned_lr0.05"):
        run = runs[name]
        result[name] = {}
        for fold in (1, 2):
            prefix = f"heldout_fold{fold}"
            rows = [json.loads(x) for x in run["files"][f"{prefix}/sigma_updates.jsonl"].splitlines()]
            probes = [json.loads(x) for x in run["files"][f"{prefix}/sigma_probes.jsonl"].splitlines()]
            phases = {}
            for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 10)):
                records = [row for row in rows if lo < row["epoch"] <= hi]
                phases[f"epoch_{lo}_to_{hi}"] = {}
                for side in ("left", "right"):
                    key = f"log_sigma_{side}"
                    gradient = np.array([row["gradient_before_clipping"][key] for row in records])
                    clipped = np.array([row["gradient_after_clipping"][key] for row in records])
                    delta = np.array([row["delta_log_sigma"][key] for row in records])
                    phases[f"epoch_{lo}_to_{hi}"][side] = {
                        "sampled_steps": len(records),
                        "median_absolute_gradient_before_clipping": float(np.median(abs(gradient))),
                        "median_absolute_delta": float(np.median(abs(delta))),
                        "max_absolute_delta": float(max(abs(delta))),
                        "zero_gradient_count": int(sum(gradient == 0)),
                        "zero_delta_count": int(sum(delta == 0)),
                        "clipped_count": int(sum(abs(clipped) < abs(gradient) * .9999)),
                        "delta_same_sign_as_current_gradient_count": int(sum(gradient * delta > 0)),
                    }
            audits = [adam_update(row, side) for row in rows for side in ("left", "right")]
            assert max(x["absolute_error"] for x in audits) < 3e-7
            result[name][str(fold)] = {
                "phases": phases,
                "adam_equation_max_error": max(x["absolute_error"] for x in audits),
                "right_selected_update_records": [adam_update(row, "right") for row in rows if row["step"] in (350, 1400, 1450, 1500, 1800, 3000, 4000, 4450)],
                "probe_gradient_timeline": [{"event": row["event"], "epoch": row["epoch"], "sigma": row["cases"]["configured"]["sigma"], "mean_gradient_left_right": row["cases"]["configured"]["mean_gradient_left_right"], "mse_by_case": {key: value["mse"] for key, value in row["cases"].items()}} for row in probes],
            }
    return result


def load_kernel():
    """Import the current reviewed module without eager imports of full model dependencies."""
    package = ModuleType("sigma_mechanism_va")
    package.__path__ = [str(PROJECT / "va_model_code/decoder_va")]
    sys.modules[package.__name__] = package
    path = Path(package.__path__[0]) / "redistribution.py"
    spec = importlib.util.spec_from_file_location("sigma_mechanism_va.redistribution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AsymGaussianRedistributor, hashlib.sha256(path.read_bytes()).hexdigest()


def kernel_checks():
    """Verify direction, discrete narrow-width behavior, and finite-sequence boundary effects."""
    cls, digest = load_kernel()
    comparison = json.loads((HERE / "comparison.json").read_text())
    widths = {"initial": [1, 1], **{f"fold{fold}": data["selected_sigma"] for fold, data in comparison["runs"]["learned_lr0.05"]["folds"].items()}}
    weights = {}
    for name, (left, right) in widths.items():
        model = cls(left - 1e-6, right - 1e-6)
        impulse = torch.zeros(1, 65)
        impulse[0, 32] = 1
        mask = torch.ones_like(impulse, dtype=torch.bool)
        output = model(impulse, mask)
        torch.testing.assert_close(output.sum(), torch.tensor(1.0))
        mirror = cls(right - 1e-6, left - 1e-6)(impulse, mask)
        torch.testing.assert_close(output.flip(1), mirror, atol=2e-7, rtol=2e-6)
        if name == "fold2":
            assert output[0, :32].sum() > output[0, 33:].sum()
        masses = torch.stack((output[0, :32].sum(), output[0, 32], output[0, 33:].sum()))
        derivatives = torch.stack([torch.stack(torch.autograd.grad(value, tuple(model.parameters()), retain_graph=True)) for value in masses])
        constant = model(torch.ones(1, 17), torch.ones(1, 17, dtype=torch.bool))
        torch.testing.assert_close(constant.sum(), torch.tensor(17.0))
        weights[name] = {
            "sigma": [model.sigma_left.item(), model.sigma_right.item()],
            "impulse_mass_left_self_right": masses.detach().tolist(),
            "mass_log_sigma_jacobian": derivatives.tolist(),
            "constant_input_output": constant.detach().flatten().tolist(),
            "mirrored_widths_mirror_output_verified": True,
        }
    analytic = []
    for sigma in (1, .5, .3, widths["fold1"][0], widths["fold1"][1], widths["fold2"][1]):
        for distance in (1, 2):
            value = math.exp(-distance**2 / (2 * sigma**2))
            derivative = value * distance**2 / sigma**2 * (1 - 1e-6 / sigma)
            analytic.append({"sigma": sigma, "distance": distance, "unnormalized_weight": value, "d_weight_d_log_sigma": derivative})
    return {"source_sha256": digest, "current_kernel_cpu_fp32": weights, "analytic_neighbor_weights": analytic}


def main():
    """Save evidence without loading downstream weights or modifying training code."""
    torch.set_num_threads(1)
    result = {
        "limitations": ["Every 50th optimizer update plus first update, not all updates.", "Probes reuse two training examples per fold; no held-out inference interventions.", "Adam betas/epsilon assumed .9/.999/1e-8; reconstructed updates checked against records.", "Kernel tests use synthetic impulses and constant inputs, not downstream-model weights."],
        "updates": analyze_updates(load_recorded_runs()),
        "kernel_checks": kernel_checks(),
    }
    destination = HERE / "mechanism_diagnostics.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(destination)
    for name, folds in result["updates"].items():
        for fold, data in folds.items():
            print(name, fold, "late_right", data["phases"]["epoch_4_to_10"]["right"], "Adam equation error", data["adam_equation_max_error"])
    print("kernel_checks", json.dumps(result["kernel_checks"]["current_kernel_cpu_fp32"]))


if __name__ == "__main__":
    main()

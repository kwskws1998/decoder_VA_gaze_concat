"""Verify and compare raw, fixed, and two learned-sigma runs from result-only ZIPs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
ORDER = ("raw", "fixed", "learned_lr0.001", "learned_lr0.05")
LABELS = {"raw": "Raw TRT", "fixed": "Fixed sigma 1/1", "learned_lr0.001": "Learned LR 0.001", "learned_lr0.05": "Learned LR 0.05"}


def metrics(rows):
    """Independently recompute population CCC, Pearson, MSE, RMSE, and MAE."""
    output = {"n_examples": len(rows)}
    for side in ("valence", "arousal"):
        target = np.array([float(row[side]) for row in rows], dtype=np.float64)
        predicted = np.array([float(row[f"pred_{side}"]) for row in rows], dtype=np.float64)
        assert np.isfinite(target).all() and np.isfinite(predicted).all()
        covariance = np.mean((target - target.mean()) * (predicted - predicted.mean()))
        mse = np.mean((predicted - target) ** 2)
        output.update({
            f"mse_{side}": float(mse),
            f"rmse_{side}": float(np.sqrt(mse)),
            f"mae_{side}": float(np.abs(predicted - target).mean()),
            f"pearson_corr_{side}": float(covariance / np.sqrt(target.var() * predicted.var())),
            f"ccc_{side}": float(2 * covariance / (target.var() + predicted.var() + (target.mean() - predicted.mean()) ** 2)),
        })
    for name in ("mse", "mae", "pearson_corr", "ccc"):
        output[f"{name}_mean"] = (output[f"{name}_valence"] + output[f"{name}_arousal"]) / 2
    return output


def rows_from(data):
    """Read quoted TSV rows and compare examples in stable fold/index order."""
    return sorted(csv.DictReader(io.StringIO(data.decode()), delimiter="\t"), key=lambda row: (int(row["held_out_fold"]), int(row["index"])))


def load_archive(archive):
    """Verify exact archive membership, CRC, and every manifest size/hash."""
    names = archive.namelist()
    assert len(names) == len(set(names))
    assert archive.testzip() is None
    roots = {name.split("/")[0] for name in names}
    assert len(roots) == 1
    root = roots.pop()
    manifest = json.loads(archive.read(f"{root}/results_manifest.json"))
    assert set(names) == {f"{root}/{name}" for name in manifest["files"]} | {f"{root}/results_manifest.json"}
    files = {}
    for name, meta in manifest["files"].items():
        assert Path(name).suffix not in {".safetensors", ".pt", ".pth", ".bin"}
        value = archive.read(f"{root}/{name}")
        assert len(value) == meta["size_bytes"]
        assert hashlib.sha256(value).hexdigest() == meta["sha256"]
        files[name] = value
    config = json.loads(files["training_parameters.json"])
    assert config["run_name"] == root
    method = config["gaze_redistribution"]["method"]
    name = {"none": "raw", "fixed-gaussian": "fixed"}.get(method)
    if method == "asym-gaussian":
        name = f"learned_lr{config['redistribution_learning_rate']:g}"
    assert name in ORDER
    return name, {"root": root, "config": config, "files": files, "rows": rows_from(files["oof_predictions.tsv"])}


def kernel_mass(sigma):
    """Compute illustrative source mass on the complete integer grid, without boundaries."""
    if sigma is None:
        return None
    widths = [float(v) for v in sigma]
    limit = math.ceil(max(widths) * 12)
    left, right = [sum(math.exp(-0.5 * (d / width) ** 2) for d in range(1, limit + 1)) for width in widths]
    total = 1 + left + right
    return {"left_percent": 100 * left / total, "self_percent": 100 / total, "right_percent": 100 * right / total}


def compare(old, new):
    """Match run inputs and verify selected-model probes against best-epoch evidence."""
    runs = {}
    with zipfile.ZipFile(old) as outer:
        assert outer.testzip() is None
        for item in outer.namelist():
            if item.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(outer.read(item))) as archive:
                    name, run = load_archive(archive)
                    assert name not in runs
                    runs[name] = run
    with zipfile.ZipFile(new) as archive:
        name, run = load_archive(archive)
        assert name == "learned_lr0.05" and name not in runs
        runs[name] = run
    assert set(runs) == set(ORDER)
    identity_keys = ("index", "held_out_fold", "text", "dataset_of_origin", "valence", "arousal")
    identity = lambda run: [[row[key] for key in identity_keys] for row in run["rows"]]
    reference = runs["raw"]
    result = {"inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (old, new)}, "runs": {}}
    previous, current = [runs[name]["config"] for name in ("learned_lr0.001", "learned_lr0.05")]
    changes = {key: [previous.get(key), current.get(key)] for key in sorted(set(previous) | set(current)) if previous.get(key) != current.get(key)}
    assert set(changes) == {"run_name", "effective_output_dir", "redistribution_learning_rate"}
    result["lr_comparison_config_changes"] = changes
    for name in ORDER:
        run = runs[name]
        assert identity(run) == identity(reference)
        config_changes = {key for key in set(run["config"]) | set(reference["config"]) if run["config"].get(key) != reference["config"].get(key)}
        assert config_changes <= {"run_name", "effective_output_dir", "redistribution_learning_rate", "redistribution_weight_decay", "gaze_redistribution"}
        stored = json.loads(run["files"]["oof_metrics.json"])
        calculated = metrics(run["rows"])
        error = max(abs(stored[key] - calculated[key]) for key in stored)
        assert error < 1e-12
        record = {"run_name": run["root"], "verified_files": len(run["files"]), "metrics": stored, "metric_recomputation_max_abs_error": error, "folds": {}, "datasets": {}}
        for dataset in sorted({row["dataset_of_origin"] for row in run["rows"]}):
            record["datasets"][dataset] = metrics([row for row in run["rows"] if row["dataset_of_origin"] == dataset])
        for fold in (1, 2):
            prefix = f"heldout_fold{fold}"
            files = run["files"]
            state = json.loads(files[f"{prefix}/checkpoints/trainer_state.json"])
            assert state["global_step"] == 4490 and state["epoch"] == 10
            probes = [json.loads(line) for line in files[f"{prefix}/sigma_probes.jsonl"].splitlines()]
            updates = [json.loads(line) for line in files[f"{prefix}/sigma_updates.jsonl"].splitlines()]
            assert len(probes) == 12 and len(updates) == 90 and not any("error" in p for p in probes)
            selected = probes[-1]
            assert selected["event"] == "train_end_selected_model"
            checkpoint = Path(state["best_model_checkpoint"]).name
            best_step = int(checkpoint.split("-")[-1])
            best_probe = next(p for p in probes if p["event"] == "epoch_end" and p["step"] == best_step)
            for key in ("prediction", "sigma"):
                assert best_probe["cases"]["configured"][key] == selected["cases"]["configured"][key]
            best_state = json.loads(files[f"{prefix}/checkpoints/{checkpoint}/trainer_state.json"])
            assert best_state["global_step"] == best_step
            fold_rows = [row for row in run["rows"] if int(row["held_out_fold"]) == fold]
            assert rows_from(files[f"{prefix}/predictions.tsv"]) == fold_rows
            fold_metrics = metrics(fold_rows)
            stored_fold = json.loads(files[f"{prefix}/metrics.json"])
            for metric, value in fold_metrics.items():
                assert abs(stored_fold[f"test_{metric}"] - value) < 1e-12
            assert abs(fold_metrics["mse_mean"] - state["best_metric"]) < 1e-12
            sigma = selected["cases"]["configured"]["sigma"]
            fold_record = {
                "best_epoch": best_probe["epoch"], "best_step": best_step, "metrics": fold_metrics,
                "selected_sigma": sigma, "right_over_left": None if sigma is None else sigma[1] / sigma[0],
                "last_epoch_sigma": probes[-2]["cases"]["configured"]["sigma"],
                "ideal_integer_grid_mass": kernel_mass(sigma),
                "timeline": [{"epoch": p["epoch"], "step": p["step"], "sigma": p["cases"]["configured"]["sigma"]} for p in probes[:-1]],
                "eval_history": [h for h in state["log_history"] if "eval_mse_mean" in h],
                "selected_probe_cases": selected["cases"],
            }
            record["folds"][str(fold)] = fold_record
        result["runs"][name] = record
    for fold in (1, 2):
        prefix = f"heldout_fold{fold}"
        earlier, later = [json.loads(runs[name]["files"][f"{prefix}/run_manifest.json"]) for name in ("learned_lr0.001", "learned_lr0.05")]
        changed = {key for key in set(earlier) | set(later) if earlier.get(key) != later.get(key)}
        assert changed <= {"run_name", "effective_output_dir", "redistribution_learning_rate"}
        initial = [json.loads(runs[name]["files"][f"{prefix}/sigma_probes.jsonl"].splitlines()[0]) for name in ORDER]
        for case in ("raw", "symmetric_1", "left_05_right_2", "left_2_right_05"):
            assert all(p["cases"][case]["prediction"] == initial[0]["cases"][case]["prediction"] for p in initial)
    result["input_rows_labels_and_fold_ids_match"] = True
    result["lr_run_manifests_match_except_lr_and_paths"] = True
    result["initial_common_probe_predictions_match"] = True
    new_metrics = result["runs"]["learned_lr0.05"]["metrics"]
    result["new_minus_baseline"] = {}
    for name in ORDER[:-1]:
        baseline = result["runs"][name]["metrics"]
        result["new_minus_baseline"][name] = {
            "mse_relative_reduction_percent": 100 * (baseline["mse_mean"] - new_metrics["mse_mean"]) / baseline["mse_mean"],
            "metric_deltas": {key: new_metrics[key] - baseline[key] for key in baseline if key != "n_examples"},
            "fold_mse_relative_reduction_percent": {
                fold: 100 * (result["runs"][name]["folds"][fold]["metrics"]["mse_mean"] - result["runs"]["learned_lr0.05"]["folds"][fold]["metrics"]["mse_mean"]) / result["runs"][name]["folds"][fold]["metrics"]["mse_mean"] for fold in ("1", "2")
            },
        }
    return result


def plot(result):
    """Plot both learning rates with common axes and best-checkpoint markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for fold, axis in zip(("1", "2"), axes):
        for name, style, alpha in (("learned_lr0.001", "--", .6), ("learned_lr0.05", "-", 1)):
            data = result["runs"][name]["folds"][fold]
            for index, side, color in ((0, "Left", "#1762a0"), (1, "Right", "#d26425")):
                axis.plot([p["epoch"] for p in data["timeline"]], [p["sigma"][index] for p in data["timeline"]], style, color=color, alpha=alpha, marker="o", markersize=3, label=f"{side}, LR {name.split('lr')[1]}")
                axis.scatter(data["best_epoch"], data["selected_sigma"][index], marker="*", s=160, color=color, alpha=alpha, edgecolors="black", zorder=5)
        axis.set_title(f"Held-out fold {fold}")
        axis.set_xlabel("Training epoch")
        axis.set_xticks(range(0, 11, 2))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Sigma (aligned Qwen token positions)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Sentence-only, seed 42: sigma LR comparison; stars = selected checkpoints")
    fig.tight_layout()
    fig.savefig(HERE / "sigma_lr_comparison.png", dpi=180)
    plt.close(fig)


def main():
    """Write machine-readable verified results and their trajectory plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_archive", type=Path)
    parser.add_argument("new_archive", type=Path)
    args = parser.parse_args()
    result = compare(args.old_archive, args.new_archive)
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    plot(result)
    for name, run in result["runs"].items():
        print(name, json.dumps({"metrics": run["metrics"], "folds": {f: {k: d[k] for k in ("best_epoch", "selected_sigma", "right_over_left", "last_epoch_sigma", "ideal_integer_grid_mass", "metrics")} for f, d in run["folds"].items()}, "datasets": run["datasets"], "metric_error": run["metric_recomputation_max_abs_error"]}))
    print("new_minus_baseline", json.dumps(result["new_minus_baseline"]))


if __name__ == "__main__":
    main()

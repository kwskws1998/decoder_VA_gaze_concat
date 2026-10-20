"""Audit the three completed sigma experiments from the supplied nested result ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "va_model_code"))
from package_results import _validate_result_contract, _validate_training_parameters
from decoder_va.evaluation import calculate_va_metrics


def numeric_summary(values):
    """Keep counts and distribution summaries of explicitly finite observations."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic values.")
    if not len(values):
        return {"n": 0}
    return {
        "n": len(values), "min": float(values.min()),
        "median": float(np.median(values)), "max": float(values.max()),
        "mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values**2))),
    }


def read_runs(path):
    """Verify CRC, exact manifest membership, hashes, and the full result contract."""
    runs = {}
    with zipfile.ZipFile(path) as outer:
        if outer.testzip() is not None:
            raise ValueError("Outer ZIP CRC failure.")
        for item in outer.infolist():
            if not item.filename.endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(outer.read(item))) as inner:
                if inner.testzip() is not None:
                    raise ValueError(f"Inner ZIP CRC failure: {item.filename}")
                names = inner.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("Duplicate archive members.")
                roots = {name.split("/")[0] for name in names}
                if len(roots) != 1:
                    raise ValueError("Unexpected archive roots.")
                root = roots.pop()
                manifest = json.loads(inner.read(f"{root}/results_manifest.json"))
                expected = {f"{root}/{name}" for name in manifest["files"]}
                if set(names) != expected | {f"{root}/results_manifest.json"}:
                    raise ValueError("Archive membership disagrees with manifest.")
                snapshots = {}
                for name, metadata in manifest["files"].items():
                    if Path(name).suffix in {".safetensors", ".pt", ".pth", ".bin"}:
                        raise ValueError("Unexpected weights in results package.")
                    data = inner.read(f"{root}/{name}")
                    if len(data) != metadata["size_bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
                        raise ValueError(f"Manifest mismatch: {name}")
                    snapshots[name] = data
                parameters = json.loads(snapshots["training_parameters.json"])
                _validate_training_parameters(parameters)
                _validate_result_contract(Path(root), parameters, snapshots)
                method = parameters["gaze_redistribution"]["method"]
                condition = {"none": "raw", "fixed-gaussian": "fixed", "asym-gaussian": "learned"}[method]
                if condition in runs:
                    raise ValueError(f"Duplicate condition: {condition}")
                console_log = outer.read(f"logs/{root}.log").decode("utf-8")
                completion_count = console_log.count(f"Completed. OOF reports: {parameters['effective_output_dir']}")
                traceback_count = console_log.count("Traceback (most recent call last):")
                if completion_count != 1 or traceback_count:
                    raise ValueError("Unexpected completion or error evidence in console log.")
                frame = pd.read_csv(io.BytesIO(snapshots["oof_predictions.tsv"]), sep="\t").sort_values(["held_out_fold", "index"]).reset_index(drop=True)
                runs[condition] = {
                    "name": root, "parameters": parameters, "snapshots": snapshots,
                    "manifest_files": len(snapshots), "predictions": frame,
                    "metrics": json.loads(snapshots["oof_metrics.json"]),
                    "console_log_completion_count": completion_count,
                    "console_log_traceback_count": traceback_count,
                }
    if set(runs) != {"raw", "fixed", "learned"}:
        raise ValueError("Expected exactly raw, fixed and learned experiments.")
    return runs


def analyze_updates(records):
    """Summarize sampled optimizer steps without treating them as a full trajectory."""
    phases = {
        "all_sampled": records,
        "epoch_0_to_1": [r for r in records if r["epoch"] <= 1],
        "epoch_1_to_3": [r for r in records if 1 < r["epoch"] <= 3],
        "epoch_3_to_10": [r for r in records if r["epoch"] > 3],
    }
    output = {}
    for phase, subset in phases.items():
        by_side = {}
        for side in ("left", "right"):
            key = f"log_sigma_{side}"
            before = np.array([r["gradient_before_clipping"][key] for r in subset])
            after = np.array([r["gradient_after_clipping"][key] for r in subset])
            delta = np.array([r["delta_log_sigma"][key] for r in subset])
            lr = np.array([r["optimizer_before_update"][key]["lr"] for r in subset])
            ratio = np.divide(np.abs(after), np.abs(before), out=np.ones_like(after), where=before != 0)
            active = lr > 0
            by_side[side] = {
                "gradient_before_abs": numeric_summary(np.abs(before)),
                "gradient_after_abs": numeric_summary(np.abs(after)),
                "clip_multiplier": numeric_summary(ratio),
                "fraction_clipped": float(np.mean(ratio < .9999)),
                "fraction_clip_below_0.1": float(np.mean(ratio < .1)),
                "gradient_positive_fraction": float(np.mean(before > 0)),
                "gradient_temporal_coherence_sampled": float(abs(before.sum()) / np.abs(before).sum()),
                "gradient_nonzero_count": int(np.count_nonzero(before)),
                "nonzero_delta_count": int(np.count_nonzero(delta)),
                "nonzero_lr_count": int(active.sum()),
                "delta_abs": numeric_summary(np.abs(delta)),
                "delta_abs_divided_by_lr": numeric_summary(np.abs(delta[active]) / lr[active]),
                "delta_temporal_coherence_sampled": float(abs(delta.sum()) / np.abs(delta).sum()),
                "delta_positive_fraction": float(np.mean(delta > 0)),
            }
        output[phase] = by_side
    return output


def compact_probe(record):
    """Retain intervention statistics and gradients, labeling their tiny fixed sample."""
    return {key: record[key] for key in (
        "event", "step", "epoch", "best_checkpoint", "scope",
        "example_indices_in_training_dataset", "valid_gaze_counts",
        "raw_trt", "qwen_token_gaps", "cases",
    )}


def audit_adam_equation(records):
    """Check the observed update against AdamW with stated default-hyperparameter assumptions."""
    beta1, beta2, epsilon = .9, .999, 1e-8
    errors = []
    for record in records:
        for side in ("left", "right"):
            key = f"log_sigma_{side}"
            previous = record["optimizer_before_update"][key]
            gradient = record["gradient_after_clipping"][key]
            step = int(previous["step"] or 0) + 1
            first = beta1 * (previous["exp_avg"] or 0) + (1 - beta1) * gradient
            second = beta2 * (previous["exp_avg_sq"] or 0) + (1 - beta2) * gradient**2
            predicted = -previous["lr"] * first / (1 - beta1**step) / (np.sqrt(second / (1 - beta2**step)) + epsilon)
            errors.append(abs(predicted - record["delta_log_sigma"][key]))
    return {
        "assumed_beta1": beta1, "assumed_beta2": beta2, "assumed_epsilon": epsilon,
        "zero_weight_decay_recorded": True,
        "predicted_vs_observed_delta_absolute_error": numeric_summary(errors),
        "limitation": "Optimizer hyperparameters were not directly saved in the results archive; this is an equation consistency check.",
    }


def analyze_runs(runs):
    """Compare OOF outcomes, checkpoint selection, interventions, and optimizer evidence."""
    reference = runs["raw"]
    changed = {
        key: {condition: run["parameters"].get(key) for condition, run in runs.items()}
        for key in reference["parameters"]
        if any(run["parameters"].get(key) != reference["parameters"][key] for run in runs.values())
    }
    allowed = {"run_name", "effective_output_dir", "gaze_redistribution", "redistribution_learning_rate", "redistribution_weight_decay"}
    if set(changed) - allowed:
        raise ValueError(f"Unexpected configuration differences: {set(changed) - allowed}")
    summary = {"configuration_differences": changed, "runs": {}}
    identity = ["index", "held_out_fold", "text", "dataset_of_origin", "valence", "arousal"]
    for condition, run in runs.items():
        frame = run["predictions"]
        if not frame[identity].equals(reference["predictions"][identity]):
            raise ValueError("OOF inputs/labels are not matched.")
        predictions = frame[["pred_valence", "pred_arousal"]].to_numpy()
        labels = frame[["valence", "arousal"]].to_numpy()
        recomputed = calculate_va_metrics(labels, predictions)
        error = max(abs(recomputed[key] - value) for key, value in run["metrics"].items())
        if error > 1e-12:
            raise ValueError("Recomputed OOF metrics disagree with saved metrics.")
        record = {
            "run_name": run["name"], "verified_manifest_files": run["manifest_files"],
            "console_log_completion_count": run["console_log_completion_count"],
            "console_log_traceback_count": run["console_log_traceback_count"],
            "metric_recomputation_max_abs_error": error,
            "metrics": run["metrics"], "folds": {},
            "metrics_by_dataset": {},
        }
        for dataset, subset in frame.groupby("dataset_of_origin"):
            record["metrics_by_dataset"][dataset] = calculate_va_metrics(
                subset[["valence", "arousal"]].to_numpy(),
                subset[["pred_valence", "pred_arousal"]].to_numpy(),
            )
        for fold in (1, 2):
            prefix = f"heldout_fold{fold}"
            snapshots = run["snapshots"]
            state = json.loads(snapshots[f"{prefix}/checkpoints/trainer_state.json"])
            probes = [json.loads(line) for line in snapshots[f"{prefix}/sigma_probes.jsonl"].splitlines()]
            updates = [json.loads(line) for line in snapshots[f"{prefix}/sigma_updates.jsonl"].splitlines()]
            if len(probes) != 12 or any("error" in p for p in probes):
                raise ValueError("Expected twelve successful probes per fold.")
            if len(updates) != 90 or [r["step"] for r in updates] != [1, *range(50, 4491, 50)]:
                raise ValueError("Unexpected optimizer diagnostic cadence.")
            selected = probes[-1]
            if selected["event"] != "train_end_selected_model":
                raise ValueError("Missing selected-model probe.")
            best_step = int(Path(state["best_model_checkpoint"]).name.split("-")[-1])
            best_probe = next(p for p in probes if p["step"] == best_step and p["event"] == "epoch_end")
            same_prediction = best_probe["cases"]["configured"]["prediction"] == selected["cases"]["configured"]["prediction"]
            same_sigma = best_probe["cases"]["configured"]["sigma"] == selected["cases"]["configured"]["sigma"]
            if not same_prediction or not same_sigma:
                raise ValueError("Selected-model probe differs from its saved epoch probe.")
            timeline = [{
                "event": p["event"], "epoch": p["epoch"], "step": p["step"],
                "sigma": p["cases"]["configured"]["sigma"],
                "gradient_coherence": p["cases"]["configured"].get("gradient_coherence_left_right"),
            } for p in probes]
            fold_result = {
                "final_training_epoch": state["epoch"], "optimizer_steps": state["global_step"],
                "best_epoch": best_probe["epoch"], "best_step": best_step,
                "best_metric": state["best_metric"], "selected_model_matches_epoch_probe": True,
                "probe_count": len(probes), "sampled_update_count": len(updates),
                "selected_sigma": selected["cases"]["configured"]["sigma"],
                "last_epoch_sigma": probes[-2]["cases"]["configured"]["sigma"],
                "sigma_timeline": timeline, "selected_probe": compact_probe(selected),
                "initial_probe": compact_probe(probes[0]),
                "metrics": json.loads(snapshots[f"{prefix}/metrics.json"]),
                "eval_history": [r for r in state["log_history"] if "eval_mse_mean" in r],
            }
            if condition == "learned":
                fold_result["optimizer_diagnostics"] = analyze_updates(updates)
                fold_result["adam_equation_audit"] = audit_adam_equation(updates)
                fold_result["first_ten_sampled_updates"] = updates[:10]
            elif condition == "fixed":
                if any(any(d != 0 for d in r["delta_log_sigma"].values()) for r in updates):
                    raise ValueError("Frozen sigma moved.")
                fold_result["frozen_sigma_zero_displacement_verified"] = True
            record["folds"][str(fold)] = fold_result
        summary["runs"][condition] = record
    for fold in ("1", "2"):
        for case in ("raw", "symmetric_1", "left_05_right_2", "left_2_right_05"):
            predictions = [r["folds"][fold]["initial_probe"]["cases"][case]["prediction"] for r in summary["runs"].values()]
            if any(value != predictions[0] for value in predictions[1:]):
                raise ValueError("Initial common-intervention predictions differ across conditions.")
    summary["initial_common_intervention_predictions_exactly_matched"] = True
    return summary


def make_plot(summary, destination):
    """Plot learned widths at every epoch and the selected-checkpoint markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharex=True, sharey=True)
    for fold, axis in zip(("1", "2"), axes):
        record = summary["runs"]["learned"]["folds"][fold]
        timeline = record["sigma_timeline"][:-1]
        epochs = [p["epoch"] or 0 for p in timeline]
        for side, color, label in ((0, "#2463a6", "Left sigma"), (1, "#c96a27", "Right sigma")):
            axis.plot(epochs, [p["sigma"][side] for p in timeline], "o-", color=color, label=label, markersize=4)
            axis.scatter([record["best_epoch"]], [record["selected_sigma"][side]], marker="*", s=150, color=color, edgecolor="black", zorder=4)
        axis.axhline(1, color="gray", linestyle="--", linewidth=1)
        axis.axvline(record["best_epoch"], color="gray", linestyle=":", linewidth=1)
        axis.set_title(f"Held-out fold {fold} (selected epoch {record['best_epoch']:g})")
        axis.set_xlabel("Training epoch")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Sigma in aligned Qwen token positions")
    axes[0].legend(frameon=False)
    fig.suptitle("Learned redistribution: sentence-only, seed 42; stars mark selected models")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    """Produce an auditable JSON summary and figure without extracting training text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    runs = read_runs(args.archive)
    summary = analyze_runs(runs)
    summary["input_archive"] = str(args.archive.resolve())
    summary["input_sha256"] = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "analysis.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    make_plot(summary, args.output_dir / "sigma_trajectory.png")
    print(output)
    for name, record in summary["runs"].items():
        print(name, "MSE", record["metrics"]["mse_mean"], "Pearson", record["metrics"]["pearson_corr_mean"], "verified_files", record["verified_manifest_files"], "metric_error", record["metric_recomputation_max_abs_error"])
        for fold, details in record["folds"].items():
            print(" fold", fold, "best_epoch", details["best_epoch"], "selected_sigma", details["selected_sigma"])


if __name__ == "__main__":
    main()

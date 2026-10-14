"""Verify completed alignv2 results and compare them with historical result ZIPs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
HELPER_PATH = HERE.parent / "sigma_lr005_20260916" / "compare_results.py"
SPEC = importlib.util.spec_from_file_location("historical_result_tools", HELPER_PATH)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
IDENTITY_KEYS = ("index", "held_out_fold", "text", "dataset_of_origin", "valence", "arousal")


def identity(run):
    """Return the sorted sample identity, including targets and evaluation fold."""
    return [tuple(row[key] for key in IDENTITY_KEYS) for row in run["rows"]]


def metric_error(stored, calculated, prefix=""):
    """Require agreement of independently recomputed prediction metrics."""
    error = max(abs(float(stored[prefix + key]) - value) for key, value in calculated.items())
    if error > 1e-12:
        raise ValueError(f"Metric recomputation disagreement: {error}")
    return error


def read_new_bundle(path):
    """Read data only, validating CRC, file sizes, hashes, and selected run names."""
    runs = {}
    with zipfile.ZipFile(path) as outer:
        assert outer.testzip() is None
        assert len(outer.namelist()) == len(set(outer.namelist()))
        selected = json.loads(outer.read("selected_runs.json"))
        for label in ("raw", "right", "left"):
            with zipfile.ZipFile(io.BytesIO(outer.read(f"{label}_results_only.zip"))) as inner:
                _, run = HELPERS.load_archive(inner)
            assert run["root"] == selected[label]
            runs[label] = run
    return runs


def summarize_new(run):
    """Verify OOF, fold, dataset, selected-checkpoint, and fixed-sigma evidence."""
    files, config, rows = run["files"], run["config"], run["rows"]
    assert len(rows) == 14352
    assert len({(row["index"], row["held_out_fold"]) for row in rows}) == len(rows)
    assert config["gaze_alignment"]["version"] == 2
    stored = json.loads(files["oof_metrics.json"])
    summary = {
        "run_name": run["root"],
        "verified_manifest_files": len(files),
        "config": config,
        "metrics": stored,
        "oof_recomputation_max_abs_error": metric_error(stored, HELPERS.metrics(rows)),
        "folds": {},
        "datasets": {},
    }
    dataset_table = list(csv.DictReader(io.StringIO(files["metrics_by_dataset.tsv"].decode()), delimiter="\t"))
    for entry in dataset_table:
        dataset = entry["dataset_of_origin"]
        subset = [row for row in rows if row["dataset_of_origin"] == dataset]
        calculated = HELPERS.metrics(subset)
        metric_error(entry, calculated)
        assert len(subset) == config["dataset_counts_after_filter"][dataset]
        summary["datasets"][dataset] = calculated
    assert set(summary["datasets"]) == set(config["dataset_counts_after_filter"])
    for fold in (1, 2):
        prefix = f"heldout_fold{fold}/"
        manifest = json.loads(files[prefix + "run_manifest.json"])
        architecture = json.loads(files[prefix + "final_model/decoder_va_architecture.json"])
        assert manifest["gaze_alignment"] == architecture["gaze_alignment"] == config["gaze_alignment"]
        assert manifest["fold_seed"] == 42 + fold - 1
        assert manifest["training_fold"] == 3 - fold
        subset = [row for row in rows if int(row["held_out_fold"]) == fold]
        assert len(subset) == {1: 7175, 2: 7177}[fold]
        assert HELPERS.rows_from(files[prefix + "predictions.tsv"]) == subset
        calculated = HELPERS.metrics(subset)
        metric_error(json.loads(files[prefix + "metrics.json"]), calculated, "test_")
        state = json.loads(files[prefix + "checkpoints/trainer_state.json"])
        assert state["global_step"] == 4490 and state["epoch"] == 10
        best_name = Path(state["best_model_checkpoint"]).name
        best_state = json.loads(files[prefix + f"checkpoints/{best_name}/trainer_state.json"])
        best_step = int(best_name.split("-")[-1])
        assert best_state["global_step"] == best_step
        assert abs(state["best_metric"] - calculated["mse_mean"]) < 1e-12
        probes = [json.loads(line) for line in files[prefix + "sigma_probes.jsonl"].splitlines()]
        updates = [json.loads(line) for line in files[prefix + "sigma_updates.jsonl"].splitlines()]
        assert len(probes) == 12 and len(updates) == 90
        assert not any("error" in probe for probe in probes)
        selected = probes[-1]
        assert selected["event"] == "train_end_selected_model"
        best_probe = next(probe for probe in probes if probe["event"] == "epoch_end" and probe["step"] == best_step)
        for key in ("prediction", "sigma"):
            assert selected["cases"]["configured"][key] == best_probe["cases"]["configured"][key]
        sigma = selected["cases"]["configured"]["sigma"]
        assert all(probe["cases"]["configured"]["sigma"] == sigma for probe in probes)
        assert all(all(value == 0 for value in update["delta_log_sigma"].values()) for update in updates)
        summary["folds"][str(fold)] = {
            "metrics": calculated,
            "completed_epoch": state["epoch"],
            "completed_step": state["global_step"],
            "selected_epoch": best_state["epoch"],
            "selected_step": best_step,
            "selected_sigma": sigma,
            "all_recorded_sigma_updates_zero": True,
            "initial_probe_cases": {name: case["prediction"] for name, case in probes[0]["cases"].items() if name != "configured"},
        }
    return summary


def compare(new_path, old_path, lr_path):
    """Build exact within-version comparisons and explicitly separate historical runs."""
    runs = read_new_bundle(new_path)
    result = {
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (new_path, old_path, lr_path)},
        "new_runs": {},
        "historical_runs": {},
        "new_training_config_differences_from_raw": {},
        "comparisons": {},
        "limitations": [
            "One paired seed per held-out fold; no across-seed uncertainty estimate.",
            "Held-out fold MSE also selects the best epoch; this is not an independent final test set.",
            "Historical runs differ in alignment and GPU hardware; cross-version deltas are descriptive.",
        ],
    }
    reference = runs["raw"]
    for label, run in runs.items():
        assert identity(run) == identity(reference)
        result["new_runs"][label] = summarize_new(run)
        differences = {key: [reference["config"].get(key), run["config"].get(key)] for key in set(reference["config"]) | set(run["config"]) if reference["config"].get(key) != run["config"].get(key)}
        assert set(differences) <= {"run_name", "effective_output_dir", "gaze_redistribution", "redistribution_sigma_left", "redistribution_sigma_right"}
        result["new_training_config_differences_from_raw"][label] = differences
    for fold in ("1", "2"):
        initial = result["new_runs"]["raw"]["folds"][fold]["initial_probe_cases"]
        assert all(run["folds"][fold]["initial_probe_cases"] == initial for run in result["new_runs"].values())
    result["new_rows_labels_folds_match"] = True
    result["initial_common_probe_predictions_match"] = True
    historical = HELPERS.compare(old_path, lr_path)
    for label, run in historical["runs"].items():
        result["historical_runs"][label] = {key: run[key] for key in ("run_name", "metrics", "metric_recomputation_max_abs_error", "datasets")}
    with zipfile.ZipFile(old_path) as outer:
        for name in outer.namelist():
            if name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                    old_label, old_run = HELPERS.load_archive(inner)
                assert identity(old_run) == identity(reference)
                assert old_run["config"]["fold_sha256"] == reference["config"]["fold_sha256"]
                if old_label == "raw":
                    result["new_raw_config_differences_from_historical_raw"] = {
                        key: [old_run["config"].get(key), reference["config"].get(key)]
                        for key in set(old_run["config"]) | set(reference["config"])
                        if old_run["config"].get(key) != reference["config"].get(key)
                    }
    result["historical_rows_labels_folds_match_new"] = True
    for label, baseline in (("new_raw", result["new_runs"]["raw"]), ("new_right", result["new_runs"]["right"]), ("old_learned_lr0.05", result["historical_runs"]["learned_lr0.05"])):
        left, base = result["new_runs"]["left"]["metrics"], baseline["metrics"]
        result["comparisons"][f"new_left_vs_{label}"] = {
            "mse_reduction_percent": 100 * (base["mse_mean"] - left["mse_mean"]) / base["mse_mean"],
            "metric_deltas": {key: left[key] - base[key] for key in left if key != "n_examples"},
        }
    return result


def main():
    """Run the read-only archive audit and save the computed evidence as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--old-lr005", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "comparison.json")
    args = parser.parse_args()
    result = compare(args.new, args.old, args.old_lr005)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "new_oof_metrics": {label: run["metrics"] for label, run in result["new_runs"].items()}, "comparisons": result["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()

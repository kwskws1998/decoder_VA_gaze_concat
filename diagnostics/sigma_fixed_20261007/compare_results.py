"""Audit fixed 1/4 and 4/1 result archives against the prior alignv2 runs."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "previous_audit", HERE.parent / "alignv2_results_20260930/compare_results.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
DOWNLOADS = Path("/Users/wansookim/Downloads")


def read_bundle(path):
    """Validate nested archive manifests and bind each archive to its selected run."""
    runs = {}
    with zipfile.ZipFile(path) as outer:
        assert outer.testzip() is None
        assert len(outer.namelist()) == len(set(outer.namelist()))
        selected = json.loads(outer.read("selected_runs.json"))
        for label, name in selected.items():
            with zipfile.ZipFile(io.BytesIO(outer.read(f"{label}_results_only.zip"))) as inner:
                _, run = AUDIT.HELPERS.load_archive(inner)
            assert run["root"] == name
            runs[label] = run
    return runs


def differences(reference, other):
    """Return all configuration differences without suppressing runtime changes."""
    return {
        key: [reference.get(key), other.get(key)]
        for key in sorted(set(reference) | set(other))
        if reference.get(key) != other.get(key)
    }


def main():
    """Recompute metrics and verify completion, fixed sigma, data, and initialization."""
    paths = [
        DOWNLOADS / "sigma_sentence_only_alignv2_results_only.zip",
        DOWNLOADS / "sigma_sentence_only_l1r4_l4r1_results_only_20261007_084809.zip",
    ]
    runs = {}
    for path in paths:
        runs.update(read_bundle(path))
    result = {
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "runs": {},
        "config_differences_from_prior_raw": {},
    }
    for label, run in runs.items():
        assert AUDIT.identity(run) == AUDIT.identity(runs["raw"])
        assert run["config"]["fold_sha256"] == runs["raw"]["config"]["fold_sha256"]
        result["runs"][label] = AUDIT.summarize_new(run)
        result["config_differences_from_prior_raw"][label] = differences(runs["raw"]["config"], run["config"])
    result["new_pair_config_differences"] = differences(runs["right_l1_r4"]["config"], runs["left_l4_r1"]["config"])
    assert set(result["new_pair_config_differences"]) <= {
        "run_name", "effective_output_dir", "gaze_redistribution",
        "redistribution_sigma_left", "redistribution_sigma_right",
    }
    for label, expected in (("right_l1_r4", [1.0, 4.0]), ("left_l4_r1", [4.0, 1.0])):
        assert runs[label]["config"]["gaze_redistribution"]["trainable"] is False
        for fold in ("1", "2"):
            actual = result["runs"][label]["folds"][fold]["selected_sigma"]
            assert all(abs(a - b) <= 2e-6 for a, b in zip(actual, expected))
    result["all_sample_identities_labels_and_folds_match"] = True
    result["new_pair_initial_common_probe_predictions_match"] = all(
        result["runs"]["right_l1_r4"]["folds"][fold]["initial_probe_cases"]
        == result["runs"]["left_l4_r1"]["folds"][fold]["initial_probe_cases"]
        for fold in ("1", "2")
    )
    result["prior_vs_new_initial_common_probe_predictions_match"] = all(
        result["runs"]["raw"]["folds"][fold]["initial_probe_cases"]
        == result["runs"]["right_l1_r4"]["folds"][fold]["initial_probe_cases"]
        for fold in ("1", "2")
    )
    result["mse_reduction_percent"] = {}
    for candidate, baseline in (("right_l1_r4", "left_l4_r1"), ("right_l1_r4", "raw"), ("right_l1_r4", "left"), ("left_l4_r1", "raw")):
        c, b = [result["runs"][label]["metrics"]["mse_mean"] for label in (candidate, baseline)]
        result["mse_reduction_percent"][f"{candidate}_vs_{baseline}"] = 100 * (b-c)/b
    output = HERE / "comparison.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for label, summary in result["runs"].items():
        print(label, json.dumps({
            "metrics": summary["metrics"], "folds": summary["folds"],
            "datasets": summary["datasets"],
            "recomputation_error": summary["oof_recomputation_max_abs_error"],
        }))
    for key in result:
        if key != "runs":
            print(key, json.dumps(result[key]))
    print("Saved:", output)


if __name__ == "__main__":
    main()

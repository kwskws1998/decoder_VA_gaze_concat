"""Verify narrow fixed-sigma runs and compare all available alignv2 archives."""

import hashlib
import json

from compare_results import AUDIT, DOWNLOADS, HERE, differences, read_bundle


def main():
    """Audit archives, recalculate OOF metrics, and record matched comparisons."""
    paths = [
        DOWNLOADS / "sigma_sentence_only_alignv2_results_only.zip",
        DOWNLOADS / "sigma_sentence_only_l1r4_l4r1_results_only_20261007_084809.zip",
        DOWNLOADS / "sigma_sentence_only_l0p25r1_l1r0p25_results_only_20261007_135016.zip",
    ]
    runs = {}
    for path in paths:
        incoming = read_bundle(path)
        assert not (runs.keys() & incoming.keys())
        runs.update(incoming)
    result = {
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "runs": {},
        "config_differences_from_right_l1_r4": {},
        "initial_common_probe_match_with_right_l1_r4": {},
        "comparisons": {},
    }
    allowed = {
        "run_name", "effective_output_dir", "gaze_redistribution",
        "redistribution_sigma_left", "redistribution_sigma_right",
    }
    for label, run in runs.items():
        assert AUDIT.identity(run) == AUDIT.identity(runs["raw"])
        assert run["config"]["fold_sha256"] == runs["raw"]["config"]["fold_sha256"]
        result["runs"][label] = AUDIT.summarize_new(run)
        result["config_differences_from_right_l1_r4"][label] = differences(
            runs["right_l1_r4"]["config"], run["config"]
        )
    for label, expected in (("right_l0p25_r1", [0.25, 1.0]), ("left_l1_r0p25", [1.0, 0.25])):
        assert set(result["config_differences_from_right_l1_r4"][label]) <= allowed
        assert runs[label]["config"]["gaze_redistribution"]["trainable"] is False
        for fold in ("1", "2"):
            actual = result["runs"][label]["folds"][fold]["selected_sigma"]
            assert len(actual) == len(expected)
            assert all(abs(a - b) <= 2e-6 for a, b in zip(actual, expected))
    for label in runs:
        result["initial_common_probe_match_with_right_l1_r4"][label] = all(
            result["runs"][label]["folds"][fold]["initial_probe_cases"]
            == result["runs"]["right_l1_r4"]["folds"][fold]["initial_probe_cases"]
            for fold in ("1", "2")
        )
    for candidate, baseline in (
        ("left_l1_r0p25", "right_l0p25_r1"),
        ("left_l1_r0p25", "right_l1_r4"),
        ("left_l1_r0p25", "raw"),
        ("left_l1_r0p25", "left"),
        ("right_l0p25_r1", "raw"),
    ):
        c, b = [result["runs"][label] for label in (candidate, baseline)]
        record = {"mse_reduction_percent": 100 * (1 - c["metrics"]["mse_mean"] / b["metrics"]["mse_mean"])}
        for category in ("folds", "datasets"):
            record[category] = {}
            for key in c[category]:
                cm, bm = c[category][key], b[category][key]
                if category == "folds":
                    cm, bm = cm["metrics"], bm["metrics"]
                record[category][key] = 100 * (1 - cm["mse_mean"] / bm["mse_mean"])
        result["comparisons"][f"{candidate}_vs_{baseline}"] = record
    result["all_sample_identities_labels_and_folds_match"] = True
    output = HERE / "comparison_narrow.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for label, summary in result["runs"].items():
        print(label, json.dumps({
            "metrics": summary["metrics"],
            "folds": {f: {k: v[k] for k in ("metrics", "selected_epoch", "selected_sigma", "all_recorded_sigma_updates_zero")} for f, v in summary["folds"].items()},
            "datasets": {d: v["mse_mean"] for d, v in summary["datasets"].items()},
            "recomputation_error": summary["oof_recomputation_max_abs_error"],
        }))
    for key in ("comparisons", "initial_common_probe_match_with_right_l1_r4", "config_differences_from_right_l1_r4"):
        print(key, json.dumps(result[key]))
    print("Saved:", output)


if __name__ == "__main__":
    main()

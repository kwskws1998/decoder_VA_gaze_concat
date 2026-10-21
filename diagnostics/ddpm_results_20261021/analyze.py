"""Validate completed DDPM artifacts and compare verified historical OOF metrics.

Run from any directory with the project dependencies installed. Input archives
are read only; reports are written next to this script. Historical Raw metadata
is audited, never silently repaired or used for subgroup attribution.
"""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from diffusion_code.package_results import package_results, verify_metrics
from va_model_code.decoder_va.evaluation import metrics_from_prediction_frame
from va_model_code.decoder_va.filters import load_filtered_folds, SENTENCE_ONLY_DATASET_NAMES

OUTPUT = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
ARCHIVES = {
    "ddpm": "ddpm_completed_results_only_20261020_215711_576273.zip",
    "historical_raw": "qwen_full_gaze_TRT_results_only_20260805_130426.zip",
    "readout_rotation": "qwen3.5-0.8b_full_gaze_TRT_readout_learned-token_relation_trt-readout-rotation_scale1.0_no_iemocap_seed42_results_only.zip",
}


def read_archive(filename):
    """Load one uniquely identified OOF run and check ZIP CRC without extracting."""
    path = DOWNLOADS / filename
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        manifests = [name for name in names if name.endswith("/training_parameters.json")]
        assert len(manifests) == 1
        root = manifests[0].removesuffix("training_parameters.json")
        contents = {name[len(root):]: archive.read(name) for name in names
                    if name.startswith(root) and not name.endswith("/")}
    frame = pd.read_csv(io.BytesIO(contents["oof_predictions.tsv"]), sep="\t", keep_default_na=False)
    manifest = json.loads(contents["training_parameters.json"])
    verify_metrics(frame, json.loads(contents["oof_metrics.json"]))
    assert not frame[["index", "held_out_fold"]].duplicated().any()
    return contents, frame, manifest, hashlib.sha256(path.read_bytes()).hexdigest()


def verify_ddpm(contents, manifest):
    """Verify SHA256 provenance and exercise the production completion validator."""
    hashes = json.loads(contents["PACKAGE_MANIFEST.json"])["sha256"]
    assert set(hashes) == set(contents) - {"PACKAGE_MANIFEST.json"}
    for name, expected in hashes.items():
        assert hashlib.sha256(contents[name]).hexdigest() == expected, name
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest["fold_sha256"].items():
        assert hashlib.sha256((ROOT / "va_model_code/data/paper7_seed42" / name).read_bytes()).hexdigest() == expected
    with tempfile.TemporaryDirectory(prefix="ddpm-result-audit-") as temp:
        run = Path(temp).resolve() / "verified_run"
        for name, raw in contents.items():
            relative = PurePosixPath(name)
            assert not relative.is_absolute() and ".." not in relative.parts
            target = run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        verified_archive = package_results(run)
        assert verified_archive.is_file()
    return {"crc": "pass", "package_sha256_files": len(hashes),
            "matching_current_source_files": len(manifest["source_sha256"]),
            "matching_local_fold_files": len(manifest["fold_sha256"]),
            "production_completion_validator": "pass"}


def label_multiset(frame):
    """Represent label pairs at training FP32 precision independently of row order."""
    values = frame[["valence", "arousal"]].to_numpy(dtype=np.float32)
    return Counter(map(tuple, values.view(np.uint32).tolist()))


def audit_metadata(frame, folds):
    """Compare original text, source, labels and fold coverage to immutable data files."""
    result = {}
    for fold in (1, 2):
        actual = frame.loc[frame.held_out_fold.eq(fold)].set_index("index").sort_index()
        expected = folds[f"full_dataset_fold{fold}.csv"].set_index("index").sort_index()
        assert actual.index.equals(expected.index)
        metadata_ok = actual[["text", "dataset_of_origin"]].equals(expected[["text", "dataset_of_origin"]])
        matches = np.isclose(actual[["valence", "arousal"]], expected[["valence", "arousal"]],
                             rtol=1e-7, atol=1e-8).all(axis=1)
        result[str(fold)] = {"rows": len(actual), "text_source_match": metadata_ok,
                             "label_metadata_mismatch_rows": int((~matches).sum()),
                             "label_pair_multiset_match": label_multiset(actual) == label_multiset(expected)}
        assert result[str(fold)]["label_pair_multiset_match"]
    return result


def residual_summary(frame):
    """Describe saved sample means; their across-input spread is not uncertainty."""
    result = {}
    for target in ("valence", "arousal"):
        y = frame[target].to_numpy()
        p = frame[f"pred_{target}"].to_numpy()
        error = p - y
        result[target] = {"target_mean": float(y.mean()), "prediction_mean": float(p.mean()),
                          "bias": float(error.mean()), "target_std": float(y.std()),
                          "prediction_std": float(p.std()), "mse": float(np.mean(error ** 2)),
                          "bias_squared_fraction_of_mse": float(error.mean() ** 2 / np.mean(error ** 2))}
    return result


def plot_curves(curves):
    """Export learning curves with best checkpoints marked and no smoothed values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    columns = [("eval_mse_mean", "Generated-VA MSE (lower is better)"),
               ("eval_pearson_corr_valence", "Valence Pearson (higher is better)"),
               ("eval_pearson_corr_arousal", "Arousal Pearson (higher is better)")]
    for fold, color in [(1, "#216e9e"), (2, "#bf652e")]:
        selected = curves.loc[curves.held_out_fold.eq(fold)]
        best = selected.loc[selected.eval_mse_mean.idxmin()]
        for axis, (column, title) in zip(axes, columns):
            axis.plot(selected.epoch, selected[column], marker="o", color=color, label=f"Fold {fold}")
            axis.scatter([best.epoch], [best[column]], marker="*", s=150, color=color, zorder=4)
            axis.set(title=title, xlabel="Epoch", xticks=range(1, 11))
            axis.grid(alpha=0.2)
    axes[0].legend()
    fig.savefig(OUTPUT / "learning_curves.png", dpi=170)
    plt.close(fig)


def main():
    """Produce reproducible global, fold, subgroup and provenance analysis."""
    folds = load_filtered_folds(ROOT / "va_model_code/data/paper7_seed42", no_iemocap=True).folds
    report = {"archives": {}, "caveat": "Historical comparisons are not matched ablations; Raw metadata is not used for subgroups."}
    all_metrics, subgroup_metrics, frames, curves = [], [], {}, []
    for label, filename in ARCHIVES.items():
        contents, frame, manifest, sha256 = read_archive(filename)
        audit = audit_metadata(frame, folds)
        metrics = metrics_from_prediction_frame(frame)
        frames[label] = frame
        report["archives"][label] = {"file": str(DOWNLOADS / filename), "sha256": sha256,
                                     "metadata_audit": audit, "metrics": metrics,
                                     "residuals": residual_summary(frame), "manifest": manifest}
        all_metrics.append({"condition": label, **metrics})
        aligned = all(item["label_metadata_mismatch_rows"] == 0 and item["text_source_match"] for item in audit.values())
        if aligned:
            for dataset, subset in frame.groupby("dataset_of_origin"):
                subgroup_metrics.append({"condition": label, "group_type": "dataset", "group": dataset,
                                         **metrics_from_prediction_frame(subset)})
            is_sentence = frame.dataset_of_origin.isin(SENTENCE_ONLY_DATASET_NAMES)
            for name, selected in [("sentence_sources", is_sentence), ("word_sources", ~is_sentence)]:
                subgroup_metrics.append({"condition": label, "group_type": "source_type", "group": name,
                                         **metrics_from_prediction_frame(frame.loc[selected])})
        if label == "ddpm":
            assert aligned
            report["validation"] = verify_ddpm(contents, manifest)
            report["folds"] = {}
            for fold in (1, 2):
                prefix = f"heldout_fold{fold}/"
                state = json.loads(contents[prefix + "trainer_state.json"])
                contract = json.loads(contents[prefix + "run_manifest.json"])
                metrics_fold = metrics_from_prediction_frame(frame.loc[frame.held_out_fold.eq(fold)])
                history = [entry for entry in state["log_history"] if "eval_mse_mean" in entry]
                best = min(history, key=lambda entry: entry["eval_mse_mean"])
                assert len(history) == 10
                assert np.isclose(best["eval_mse_mean"], state["best_metric"], rtol=1e-7, atol=1e-10)
                assert np.isclose(metrics_fold["mse_mean"], state["best_metric"], rtol=1e-7, atol=1e-10)
                assert state["best_model_checkpoint"].endswith(f"checkpoint-{best['step']}")
                report["folds"][str(fold)] = {"trained_epoch": state["epoch"], "global_step": state["global_step"],
                                              "max_steps": state["max_steps"], "best_epoch": best["epoch"],
                                              "best_step": best["step"], "metrics": metrics_fold,
                                              "training_rows": contract["training_rows"],
                                              "fold_seed": contract["fold_seed"]}
                curves.extend({"held_out_fold": fold, **entry} for entry in history)
    pd.testing.assert_frame_equal(frames["ddpm"].drop(columns=["pred_valence", "pred_arousal"]),
                                  frames["readout_rotation"].drop(columns=["pred_valence", "pred_arousal"]),
                                  check_dtype=False, rtol=1e-7, atol=1e-8)
    ddpm_metrics = report["archives"]["ddpm"]["metrics"]
    report["changes_percent"] = {}
    for label in ("historical_raw", "readout_rotation"):
        previous = report["archives"][label]["metrics"]
        report["changes_percent"][label] = {key: 100 * (ddpm_metrics[key] / previous[key] - 1)
                                             for key in ("mse_mean", "mse_valence", "mse_arousal", "mae_mean")}
    pd.DataFrame(all_metrics).to_csv(OUTPUT / "overall_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(subgroup_metrics).to_csv(OUTPUT / "subgroup_comparison.tsv", sep="\t", index=False)
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(OUTPUT / "epoch_metrics.tsv", sep="\t", index=False)
    (OUTPUT / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    plot_curves(curve_frame)
    print(json.dumps({key: value for key, value in report.items() if key != "archives"}, indent=2))
    print(pd.DataFrame(all_metrics).to_string(index=False))
    print(pd.DataFrame(subgroup_metrics)[["condition", "group", "n_examples", "mse_mean", "pearson_corr_valence", "pearson_corr_arousal"]].to_string(index=False))
    for label, value in report["archives"].items():
        print(label, "metadata", value["metadata_audit"], "residuals", value["residuals"])


if __name__ == "__main__":
    main()

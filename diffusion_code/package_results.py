"""Package verified completed DDPM results without weights or optimizer files."""

import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from va_model_code.decoder_va.evaluation import metrics_from_prediction_frame
from va_model_code.decoder_va.paths import resolve_run_directory
from diffusion_code.config import DiffusionConfig
from diffusion_code.mean_guided import FAMILY as MEAN_FAMILY, MEAN_CONTRACT


def verify_metrics(frame, recorded, prefix=""):
    """Recompute every VA metric, allowing null only when Pearson/CCC is undefined."""

    computed = metrics_from_prediction_frame(frame)
    for name, value in computed.items():
        key = prefix + name
        if key not in recorded:
            raise ValueError(f"Missing saved metric: {key}")
        saved = recorded[key]
        if not np.isfinite(value):
            if saved is not None and not pd.isna(saved):
                raise ValueError(f"Undefined metric has an incompatible value: {key}")
        elif saved is None or not np.isclose(saved, value, rtol=1e-6, atol=1e-9):
            raise ValueError(f"Saved metric disagrees with predictions: {key}")


def package_results(run_dir):
    """Certify the requested folds, OOF union, metrics, and immutable bytes before zipping."""

    run_dir = Path(run_dir)
    if run_dir.is_symlink():
        raise ValueError("Run directory cannot be a symlink.")
    content = {}

    def read(relative):
        """Read one regular artifact once, rejecting links and missing completion artifacts."""

        path = run_dir / relative
        if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
            raise ValueError(f"Required regular result artifact missing: {relative}")
        raw = path.read_bytes()
        content[str(relative)] = raw
        return raw

    def read_json(relative):
        """Decode a previously verified artifact as JSON."""

        return json.loads(read(relative))

    manifest = read_json("training_parameters.json")
    family = manifest.get("model_family")
    if family not in ("conditional_ddpm_va", MEAN_FAMILY) or manifest.get("schema_version") != 1:
        raise ValueError("Use this packager only for supported version-1 DDPM results.")
    mean_guided = family == MEAN_FAMILY
    if mean_guided and (manifest.get("mean_guidance") != MEAN_CONTRACT
                        or manifest.get("diffusion_role") != "base_schedule_only_target_is_defined_by_mean_guidance"):
        raise ValueError("Invalid mean-guided run semantics.")
    settings = DiffusionConfig.from_dict(manifest["diffusion"])
    if mean_guided and not settings.clip_clean:
        raise ValueError("Mean-guided results require clipping in original VA bounds.")
    completion = read_json("COMPLETED.json")
    folds = manifest["arguments"]["held_out_folds"]
    if not folds or len(set(folds)) != len(folds) or set(folds) - {1, 2}:
        raise ValueError("Invalid requested held-out folds.")
    if completion["held_out_folds"] != folds:
        raise ValueError("Not all requested folds completed.")
    fold_frames, mean_frames, comparison_rows = [], [], []
    for fold in folds:
        directory = f"heldout_fold{fold}"
        completed = read_json(f"{directory}/COMPLETED.json")
        contract = read_json(f"{directory}/run_manifest.json")
        if any(contract.get(key) != value for key, value in manifest.items()):
            raise ValueError("Fold training contract disagrees with the run.")
        if contract["held_out_fold"] != fold or contract["training_fold"] != 3 - fold:
            raise ValueError("Fold identity mismatch.")
        state = read_json(f"{directory}/trainer_state.json")
        if state["global_step"] != completed["global_step"] or state["global_step"] < 1:
            raise ValueError("Fold training state is incomplete.")
        if state["global_step"] < state["max_steps"]:
            raise ValueError("Fold did not finish its scheduled training steps.")
        if state["best_model_checkpoint"] != completed["best_model_checkpoint"]:
            raise ValueError("Best checkpoint identity mismatch.")
        if state["best_metric"] != completed["best_metric"] or state["best_metric"] is None:
            raise ValueError("Missing or inconsistent best-checkpoint metric.")
        architecture_name = "mean_guided_architecture.json" if mean_guided else "diffusion_architecture.json"
        architecture = read_json(f"{directory}/final_model/{architecture_name}")
        if architecture["base_schedule" if mean_guided else "diffusion"] != settings.to_dict():
            raise ValueError("Final model DDPM settings disagree with training.")
        if architecture["model_family"] != family or architecture["schema_version"] != 1:
            raise ValueError("Final model family disagrees with training.")
        encoder = read_json(f"{directory}/final_model/encoder_architecture.json")
        if mean_guided:
            expected_reference = manifest["mean_reference"]["folds"][str(fold)]
            if architecture["mean_guidance"] != MEAN_CONTRACT or architecture["reference"] != expected_reference:
                raise ValueError("Final mean predictor reference disagrees with training.")
            if architecture["encoder_sha256"] != hashlib.sha256(content[f"{directory}/final_model/encoder_architecture.json"]).hexdigest():
                raise ValueError("Mean predictor blueprint changed after saving.")
            if encoder["role"] != "frozen_raw_mean_predictor_with_regression_head":
                raise ValueError("Invalid frozen mean component.")
        if read_json(f"{directory}/final_model/diffusion_training_contract.json") != contract:
            raise ValueError("Saved final-model training contract disagrees with the fold.")
        read_json(f"{directory}/gpu_memory.json")
        read_json(f"{directory}/parameter_counts.json")
        raw = read(f"{directory}/predictions.tsv")
        if hashlib.sha256(raw).hexdigest() != completed["predictions_sha256"]:
            raise ValueError("Fold predictions changed after completion.")
        frame = pd.read_csv(io.BytesIO(raw), sep="\t", keep_default_na=False)
        if len(frame) != contract["evaluation_rows"] or not frame["held_out_fold"].eq(fold).all():
            raise ValueError("Prediction rows disagree with the held-out fold.")
        if frame[["index", "held_out_fold"]].duplicated().any():
            raise ValueError("Duplicate evaluation row IDs.")
        verify_metrics(frame, read_json(f"{directory}/metrics.json"), prefix="test_")
        fold_frames.append(frame)
        if mean_guided:
            raw_mean = read(f"{directory}/frozen_mean/predictions.tsv")
            if hashlib.sha256(raw_mean).hexdigest() != completed["frozen_mean_predictions_sha256"]:
                raise ValueError("Frozen mean predictions changed after completion.")
            mean = pd.read_csv(io.BytesIO(raw_mean), sep="\t", keep_default_na=False)
            identity = ["index", "held_out_fold", "text", "dataset_of_origin", "valence", "arousal"]
            pd.testing.assert_frame_equal(mean[identity], frame[identity], check_dtype=False, rtol=1e-7, atol=1e-9)
            verify_metrics(mean, read_json(f"{directory}/frozen_mean/metrics.json"))
            read_json(f"{directory}/frozen_mean/reproduction.json")
            mean_frames.append(mean)
            for name, values in [("frozen_raw_mean", mean), ("mean_guided_ddpm", frame)]:
                comparison_rows.append({"held_out_fold": str(fold), "condition": name, **metrics_from_prediction_frame(values)})
    raw = read("oof_predictions.tsv")
    if hashlib.sha256(raw).hexdigest() != completion["oof_sha256"]:
        raise ValueError("OOF predictions changed after completion.")
    oof = pd.read_csv(io.BytesIO(raw), sep="\t", keep_default_na=False)
    expected = pd.concat(fold_frames).sort_values(["index", "held_out_fold"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(oof, expected, check_dtype=False, rtol=1e-7, atol=1e-9)
    verify_metrics(oof, read_json("oof_metrics.json"))
    by_dataset = pd.read_csv(io.BytesIO(read("metrics_by_dataset.tsv")), sep="\t")
    if set(by_dataset.dataset_of_origin) != set(oof.dataset_of_origin):
        raise ValueError("Dataset metric names disagree with OOF data.")
    if by_dataset.dataset_of_origin.duplicated().any():
        raise ValueError("Duplicate dataset metric rows.")
    for name, frame in oof.groupby("dataset_of_origin"):
        verify_metrics(frame, by_dataset.loc[by_dataset.dataset_of_origin.eq(name)].iloc[0].to_dict())
    if mean_guided:
        raw_mean = read("frozen_mean_oof/oof_predictions.tsv")
        if hashlib.sha256(raw_mean).hexdigest() != completion["frozen_mean_oof_sha256"]:
            raise ValueError("Frozen mean OOF predictions changed after completion.")
        means = pd.read_csv(io.BytesIO(raw_mean), sep="\t", keep_default_na=False)
        expected = pd.concat(mean_frames).sort_values(["index", "held_out_fold"], kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(means, expected, check_dtype=False, rtol=1e-7, atol=1e-9)
        verify_metrics(means, read_json("frozen_mean_oof/oof_metrics.json"))
        read_json("frozen_mean_oof/training_parameters.json")
        read("frozen_mean_oof/metrics_by_dataset.tsv")
        for name, frame in [("frozen_raw_mean", means), ("mean_guided_ddpm", oof)]:
            comparison_rows.append({"held_out_fold": "oof", "condition": name, **metrics_from_prediction_frame(frame)})
        raw_comparison = read("mean_comparison.tsv")
        if hashlib.sha256(raw_comparison).hexdigest() != completion["mean_comparison_sha256"]:
            raise ValueError("Mean comparison changed after completion.")
        saved = pd.read_csv(io.BytesIO(raw_comparison), sep="\t", dtype={"held_out_fold": str})
        pd.testing.assert_frame_equal(saved, pd.DataFrame(comparison_rows), check_dtype=False, rtol=1e-6, atol=1e-9)
    package_manifest = {
        "model_family": family, "held_out_folds": folds,
        "weights_included": False,
        "sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in content.items()},
    }
    content["PACKAGE_MANIFEST.json"] = json.dumps(package_manifest, indent=2).encode()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "mean_guided_ddpm" if mean_guided else "ddpm"
    archive = run_dir / f"{prefix}_completed_results_only_{stamp}.zip"
    created = False
    try:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
            created = True
            for name, raw in content.items():
                output.writestr(f"{run_dir.name}/{name}", raw)
        with zipfile.ZipFile(archive) as result:
            if result.testzip() is not None:
                raise RuntimeError("ZIP integrity check failed.")
    except BaseException:
        if created:
            archive.unlink(missing_ok=True)
        raise
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    print(package_results(resolve_run_directory(args.run_name)))

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
    if manifest.get("model_family") != "conditional_ddpm_va" or manifest.get("schema_version") != 1:
        raise ValueError("Use this packager only for version-1 DDPM results.")
    settings = DiffusionConfig.from_dict(manifest["diffusion"])
    completion = read_json("COMPLETED.json")
    folds = manifest["arguments"]["held_out_folds"]
    if not folds or len(set(folds)) != len(folds) or set(folds) - {1, 2}:
        raise ValueError("Invalid requested held-out folds.")
    if completion["held_out_folds"] != folds:
        raise ValueError("Not all requested folds completed.")
    fold_frames = []
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
        architecture = read_json(f"{directory}/final_model/diffusion_architecture.json")
        if architecture["diffusion"] != settings.to_dict():
            raise ValueError("Final model DDPM settings disagree with training.")
        read_json(f"{directory}/final_model/encoder_architecture.json")
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
    package_manifest = {
        "model_family": "conditional_ddpm_va", "held_out_folds": folds,
        "weights_included": False,
        "sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in content.items()},
    }
    content["PACKAGE_MANIFEST.json"] = json.dumps(package_manifest, indent=2).encode()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive = run_dir / f"ddpm_completed_results_only_{stamp}.zip"
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

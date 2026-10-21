"""Validate and archive completed DDPM diagnostics, keeping subsets separate from full OOF."""

import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from diffusion_code.package_results import verify_metrics


KIND = "ddpm_post_training_diagnostics"


def package_diagnostics(directory):
    """Check completion, exact bytes, sampling membership and probe metrics before creating a ZIP."""
    directory = Path(directory)
    if directory.is_symlink() or any(p.is_symlink() for p in directory.parents):
        raise ValueError("Diagnostic directory cannot be a symbolic link.")
    marker = directory / "COMPLETED.json"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("Diagnostics need a regular completion marker.")
    completed = json.loads(marker.read_text())
    if completed.get("kind") != KIND or completed.get("schema_version") != 1:
        raise ValueError("Unsupported or incomplete diagnostics.")
    content = {}
    for name, digest in completed["sha256"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix in (".safetensors", ".pt", ".pth", ".bin", ".zip"):
            raise ValueError("Diagnostic manifest contains a non-result path.")
        path = directory / relative
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"Missing regular diagnostic artifact: {name}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Diagnostic artifact changed after completion: {name}")
        content[name] = raw

    def read_json(name):
        """Require artifacts to be covered by the completion hashes."""
        return json.loads(content[name])

    def read_table(name):
        """Load one hash-verified prediction/metric table."""
        return pd.read_csv(io.BytesIO(content[name]), sep="\t", keep_default_na=False)

    parameters = read_json("diagnostic_parameters.json")
    folds, mode = completed["held_out_folds"], completed["mode"]
    if (not folds or len(set(folds)) != len(folds) or set(folds) - {1, 2}
            or parameters["held_out_folds"] != folds or parameters["arguments"]["mode"] != mode
            or mode not in ("sampling", "probe", "all") or parameters["kind"] != KIND):
        raise ValueError("Diagnostic completion disagrees with the requested folds/mode.")
    probe_frames, original_frames, sampling_rows = [], [], []
    for fold in folds:
        root = f"heldout_fold{fold}"
        if read_json(f"{root}/COMPLETED.json") != {"fold": fold, "mode": mode}:
            raise ValueError("A requested diagnostic fold is incomplete.")
        if mode in ("sampling", "all"):
            for partition in ("train", "heldout"):
                selected = read_table(f"{root}/{partition}_selected_rows.tsv")
                for memory in parameters["memory_modes"]:
                    for samples, seed in parameters["sampling_cases"]:
                        prefix = f"{root}/sampling/{partition}/{memory}_n{samples}_seed{seed}"
                        frame = read_table(prefix + "/predictions.tsv")
                        if frame["index"].duplicated().any() or frame["index"].tolist() != selected["index"].tolist():
                            raise ValueError("Sampling cases must use identical unique selected rows.")
                        for name in ("text", "dataset_of_origin"):
                            if frame[name].tolist() != selected[name].tolist():
                                raise ValueError("Sampling metadata differs from selected rows.")
                        np.testing.assert_allclose(frame[["valence", "arousal"]], selected[["valence", "arousal"]], rtol=1e-7, atol=1e-8)
                        if not frame.held_out_fold.eq(fold if partition == "heldout" else 3 - fold).all():
                            raise ValueError("Sampling partition fold IDs disagree.")
                        row = read_json(prefix + "/sampling_metrics.json")
                        identity = {"model_heldout_fold": fold, "partition": partition, "memory_trt": memory,
                                    "samples": samples, "sampling_seed": seed,
                                    "requested_seed_sweep": [samples, seed] in parameters["requested_sampling_cases"]}
                        if any(row.get(key) != value for key, value in identity.items()):
                            raise ValueError("Sampling metrics belong to a different diagnostic condition.")
                        verify_metrics(frame, row)
                        verify_metrics(frame, read_json(prefix + "/metrics.json"))
                        dispersion = read_table(prefix + "/sample_dispersion.tsv")
                        pd.testing.assert_frame_equal(dispersion[["index", "held_out_fold"]], frame[["index", "held_out_fold"]])
                        variance = dispersion[["generated_variance_valence", "generated_variance_arousal"]].to_numpy()
                        if not np.isfinite(variance).all() or (variance < 0).any():
                            raise ValueError("Sample dispersion must be finite and nonnegative.")
                        if (not np.isclose(variance.mean(), row["mean_generated_variance"], rtol=1e-5, atol=1e-9)
                                or not np.isclose(variance.mean()/samples, row["estimated_mc_mean_variance"], rtol=1e-5, atol=1e-9)):
                            raise ValueError("Dispersion summary does not match saved samples.")
                        sampling_rows.append(row)
        if mode in ("probe", "all"):
            root += "/probe"
            state = read_json(root + "/probe_state.json")
            if len(state["history"]) != state["epochs"] or not 1 <= state["best_epoch"] <= state["epochs"]:
                raise ValueError("Probe training history is incomplete.")
            for partition in ("train", "heldout"):
                frame = read_table(f"{root}/{partition}/predictions.tsv")
                metrics = read_json(f"{root}/{partition}/metrics.json")
                verify_metrics(frame, metrics)
                if partition == "heldout":
                    if not np.isclose(metrics["mse_mean"], state["best_metric"], rtol=1e-6, atol=1e-9):
                        raise ValueError("Probe final metrics do not match its selected head.")
                    probe_frames.append(frame)
            original = read_table(root + "/source_ddpm/predictions.tsv")
            verify_metrics(original, read_json(root + "/source_ddpm/metrics.json"))
            identity = ["index", "held_out_fold", "text", "dataset_of_origin", "valence", "arousal"]
            pd.testing.assert_frame_equal(original[identity], frame[identity], check_dtype=False, rtol=1e-7, atol=1e-9)
            original_frames.append(original)
    if probe_frames:
        oof = read_table("probe_oof/oof_predictions.tsv")
        expected = pd.concat(probe_frames).sort_values(["index", "held_out_fold"], kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(oof, expected, check_dtype=False, rtol=1e-7, atol=1e-9)
        verify_metrics(oof, read_json("probe_oof/oof_metrics.json"))
        comparison = read_table("probe_comparison.tsv")
        expected_keys = []
        for fold, original, probe in [*zip(folds, original_frames, probe_frames),
                                      ("oof", pd.concat(original_frames), pd.concat(probe_frames))]:
            for condition, frame in [("saved_ddpm", original), ("frozen_encoder_direct_head", probe)]:
                groups = [("all", frame), *[(f"dataset:{name}", group) for name, group in frame.groupby("dataset_of_origin")]]
                for scope, group in groups:
                    expected_keys.append((str(fold), scope, condition))
                    row = comparison.loc[comparison.held_out_fold.astype(str).eq(str(fold))
                                         & comparison.scope.eq(scope) & comparison.condition.eq(condition)]
                    if len(row) != 1:
                        raise ValueError("Missing or duplicate probe comparison condition.")
                    verify_metrics(group, row.replace("", np.nan).iloc[0].to_dict())
        if len(comparison) != len(expected_keys):
            raise ValueError("Unexpected probe comparison conditions.")
    if sampling_rows:
        saved = pd.read_csv(io.BytesIO(content["sampling_comparison.tsv"]), sep="\t")
        expected = pd.DataFrame(sampling_rows)
        keys = ["model_heldout_fold", "partition", "memory_trt", "samples", "sampling_seed"]
        pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True),
                                      expected[saved.columns].sort_values(keys).reset_index(drop=True),
                                      check_dtype=False, rtol=1e-6, atol=1e-9)
    content["COMPLETED.json"] = (directory / "COMPLETED.json").read_bytes()
    content["PACKAGE_MANIFEST.json"] = json.dumps({
        "kind": KIND, "held_out_folds": folds, "mode": mode, "weights_included": False,
        "sampling_scope": parameters["sampling_scope"], "probe_scope": parameters["probe_scope"],
        "sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in content.items()},
    }, indent=2).encode()
    archive = directory / f"ddpm_diagnostics_results_only_{datetime.now():%Y%m%d_%H%M%S_%f}.zip"
    created = False
    try:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
            created = True
            for name, raw in content.items():
                zipped.writestr(f"{directory.name}/{name}", raw)
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise RuntimeError("Diagnostic ZIP verification failed.")
    except BaseException:
        if created:
            archive.unlink(missing_ok=True)
        raise
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(package_diagnostics(parser.parse_args().output_dir))

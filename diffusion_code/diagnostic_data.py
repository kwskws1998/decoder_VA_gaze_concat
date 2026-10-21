"""Restore immutable DDPM fold identities for post-training diagnostics."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from diffusion_code.config import DiffusionConfig
from diffusion_code.package_results import verify_metrics
from va_model_code.decoder_va.downloads import sha256_file
from va_model_code.decoder_va.filters import (
    apply_training_text_overlap_policy, dataset_counts, load_filtered_folds,
)


def read_json(path):
    """Decode one saved JSON artifact without changing it."""
    return json.loads(Path(path).read_text())


def verified_data(run_dir, data_dir):
    """Require original fold bytes and restore the recorded filtering policy."""
    manifest = read_json(Path(run_dir) / "training_parameters.json")
    if manifest.get("model_family") != "conditional_ddpm_va" or manifest.get("schema_version") != 1:
        raise ValueError("Diagnostics require a version-1 DDPM run.")
    DiffusionConfig.from_dict(manifest["diffusion"])
    for name in ("full_dataset_fold1.csv", "full_dataset_fold2.csv"):
        if sha256_file(Path(data_dir) / name) != manifest["fold_sha256"][name]:
            raise ValueError(f"Original data hash mismatch: {name}")
    args = manifest["arguments"]
    filtered = load_filtered_folds(
        data_dir, exclude_dataset=args["exclude_dataset"],
        no_iemocap=args["no_iemocap"], sentence_only=args["sentence_only"],
    )
    if dataset_counts(filtered.folds) != manifest["dataset_counts"]:
        raise ValueError("Restored dataset counts differ from the source run.")
    return manifest, filtered


def verified_fold(run_dir, manifest, filtered, fold):
    """Bind a final model to the correct training/evaluation fold and best checkpoint."""
    if fold not in manifest["arguments"]["held_out_folds"]:
        raise ValueError("Requested fold does not belong to the source run.")
    directory = Path(run_dir) / f"heldout_fold{fold}"
    contract = read_json(directory / "run_manifest.json")
    if any(contract.get(key) != value for key, value in manifest.items()):
        raise ValueError("Fold manifest differs from its run.")
    if contract["held_out_fold"] != fold or contract["training_fold"] != 3 - fold:
        raise ValueError("Incorrect checkpoint fold identity.")
    final = directory / "final_model"
    if read_json(final / "diffusion_training_contract.json") != contract:
        raise ValueError("Final model training contract does not match its fold.")
    if read_json(final / "diffusion_architecture.json")["diffusion"] != manifest["diffusion"]:
        raise ValueError("Final model sampler settings disagree with its run.")
    completed = read_json(directory / "COMPLETED.json")
    state = read_json(directory / "trainer_state.json")
    if state["global_step"] < state["max_steps"] or state["global_step"] <= 0:
        raise ValueError("Source fold did not complete training.")
    for key in ("global_step", "best_model_checkpoint", "best_metric"):
        if completed[key] != state[key] or completed[key] is None:
            raise ValueError(f"Source completion mismatch: {key}")
    if sha256_file(directory / "predictions.tsv") != completed["predictions_sha256"]:
        raise ValueError("Source predictions changed after completion.")
    train = filtered.folds[f"full_dataset_fold{3 - fold}.csv"]
    evaluation = filtered.folds[f"full_dataset_fold{fold}.csv"]
    train, audit = apply_training_text_overlap_policy(
        train, evaluation, policy=manifest["arguments"]["training_text_overlap_policy"],
    )
    if audit != contract["training_text_overlap"]:
        raise ValueError("Restored overlap policy differs from source training.")
    if len(train) != contract["training_rows"] or len(evaluation) != contract["evaluation_rows"]:
        raise ValueError("Restored fold row counts differ from source training.")
    predictions = pd.read_csv(directory / "predictions.tsv", sep="\t", keep_default_na=False)
    expected = evaluation.reset_index(drop=True)
    if not predictions.held_out_fold.eq(fold).all():
        raise ValueError("Source prediction fold IDs do not match.")
    for name in ("index", "text", "dataset_of_origin"):
        if predictions[name].tolist() != expected[name].tolist():
            raise ValueError("Source prediction row identities differ from original data.")
    pd.testing.assert_frame_equal(predictions[["valence", "arousal"]], expected[["valence", "arousal"]],
                                  check_dtype=False, rtol=1e-7, atol=1e-8)
    verify_metrics(predictions, read_json(directory / "metrics.json"), prefix="test_")
    if not (final / "model.safetensors").is_file():
        raise FileNotFoundError(f"Weights required for diagnostics: {final / 'model.safetensors'}")
    return final, train.reset_index(drop=True), evaluation.reset_index(drop=True), contract


def fixed_subset(frame, *, per_dataset, seed, fold, partition):
    """Choose rows by label-independent hashes, stable under input row reordering."""
    if per_dataset < 1:
        raise ValueError("per_dataset must be positive.")
    if frame["index"].duplicated().any():
        raise ValueError("Subset selection requires unique source row IDs.")
    parts = []
    for source, group in frame.groupby("dataset_of_origin", sort=True):
        ranks = [hashlib.sha256(json.dumps(
            [int(seed), int(fold), partition, source, int(row.index), row.text],
            ensure_ascii=False, separators=(",", ":"),
        ).encode()).hexdigest() for row in group.itertuples()]
        selected = group.assign(_selection_hash=ranks).sort_values(
            ["_selection_hash", "index"], kind="stable",
        ).head(per_dataset).drop(columns="_selection_hash")
        parts.append(selected)
    if not parts:
        raise ValueError("Cannot select a subset of an empty frame.")
    return pd.concat(parts).sort_values("index", kind="stable").reset_index(drop=True)


def require_new_output(path, source_run):
    """Keep diagnostic output separate from source checkpoints and existing reports."""
    output, source = Path(path).resolve(), Path(source_run).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Diagnostic output must be separate from the source run.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Diagnostic output is not empty: {output}")
    return output

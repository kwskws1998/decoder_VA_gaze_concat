"""Validate a completed current Raw TRT model before freezing it as a mean predictor."""

from pathlib import Path

import numpy as np
import pandas as pd

from diffusion_code.diagnostic_data import read_json
from diffusion_code.package_results import verify_metrics
from va_model_code.decoder_va.downloads import sha256_file
from va_model_code.decoder_va.filters import apply_training_text_overlap_policy, load_filtered_folds, dataset_counts
from va_model_code.decoder_va.model import load_saved_decoder_va_model


class RawMeanReference:
    """Bind each frozen model to its original fold, data, preprocessing and final weights."""

    def __init__(self, directory, data_dir, folds=None):
        self.directory = Path(directory).resolve()
        self.manifest = read_json(self.directory / "training_parameters.json")
        m = self.manifest
        if m.get("model_family") is not None or m.get("loss") != "mse":
            raise ValueError("The mean source must be a completed direct-MSE Raw TRT run.")
        if (m.get("gaze_fusion") != "prefix-concat" or m.get("gaze_feature_indices") != [3]
                or m.get("readout", {}).get("mode") != "last-token"
                or m.get("gaze_relation", {}).get("method") != "none"
                or m.get("gaze_redistribution", {}).get("method") != "none"):
            raise ValueError("The mean source must use Raw TRT, last-text-token readout, and no rotation/redistribution.")
        self.folds = list(folds or m["held_out_folds"])
        if not self.folds or len(set(self.folds)) != len(self.folds) or not set(self.folds) <= set(m["held_out_folds"]):
            raise ValueError("Requested folds are not unique source folds.")
        for name, expected in m["fold_sha256"].items():
            if name not in ("full_dataset_fold1.csv", "full_dataset_fold2.csv") or sha256_file(Path(data_dir) / name) != expected:
                raise ValueError("Raw source data hashes do not match the supplied folds.")
        if set(m["fold_sha256"]) != {"full_dataset_fold1.csv", "full_dataset_fold2.csv"}:
            raise ValueError("Raw source must record both input fold hashes.")
        filtered = load_filtered_folds(data_dir, exclude_dataset=m["exclude_dataset"],
                                       no_iemocap=m["no_iemocap"], sentence_only=m["sentence_only"])
        if dataset_counts(filtered.folds) != m["dataset_counts_after_filter"]:
            raise ValueError("Raw source dataset counts differ from the restored data.")
        self.records, all_frames = {}, []
        for fold in m["held_out_folds"]:
            folder = self.directory / f"heldout_fold{fold}"
            contract = read_json(folder / "run_manifest.json")
            if any(contract.get(key) != value for key, value in m.items()):
                raise ValueError("Raw fold and root manifests disagree.")
            if contract["held_out_fold"] != fold or contract["training_fold"] != 3 - fold:
                raise ValueError("Raw checkpoint belongs to the wrong fold.")
            state_path = folder / "trainer_state.json"
            if not state_path.is_file():
                raise ValueError("Raw source lacks final trainer_state.json. Train the current matched Raw control first.")
            state = read_json(state_path)
            if (state["global_step"] < 1 or state["global_step"] < state["max_steps"]
                    or state["best_metric"] is None or not state["best_model_checkpoint"]):
                raise ValueError("Raw source did not finish its scheduled training and checkpoint selection.")
            if not np.isfinite(state["best_metric"]):
                raise ValueError("Raw best metric is nonfinite.")
            train = filtered.folds[f"full_dataset_fold{3 - fold}.csv"]
            evaluation = filtered.folds[f"full_dataset_fold{fold}.csv"]
            train, audit = apply_training_text_overlap_policy(train, evaluation, policy=m["training_text_overlap_policy"])
            if (audit != contract["training_text_overlap"] or len(train) != contract["training_rows"]
                    or len(evaluation) != contract["evaluation_rows"]):
                raise ValueError("Raw source training/evaluation membership changed.")
            predictions = pd.read_csv(folder / "predictions.tsv", sep="\t", keep_default_na=False)
            expected = evaluation.reset_index(drop=True)
            for name in ("index", "text", "dataset_of_origin"):
                if predictions[name].tolist() != expected[name].tolist():
                    raise ValueError("Raw prediction row identities do not match the source evaluation fold.")
            if not predictions.held_out_fold.eq(fold).all():
                raise ValueError("Raw prediction fold IDs do not match.")
            np.testing.assert_allclose(predictions[["valence", "arousal"]],
                                       expected[["valence", "arousal"]].to_numpy(dtype=np.float32), rtol=0, atol=1e-7)
            metrics = read_json(folder / "metrics.json")
            verify_metrics(predictions, metrics, prefix="test_")
            if not np.isclose(metrics["test_mse_mean"], state["best_metric"], rtol=1e-5, atol=1e-8):
                raise ValueError("Raw final predictions do not match the selected checkpoint metric.")
            all_frames.append(predictions)
            final = folder / "final_model"
            files = ("model.safetensors", "decoder_va_architecture.json", "tokenizer.json", "tokenizer_config.json")
            if fold in self.folds:
                if any(not (final / name).is_file() or (final / name).is_symlink() for name in files):
                    raise ValueError("Raw final model needs regular weights, architecture and tokenizer files.")
                architecture = read_json(final / "decoder_va_architecture.json")
                for key in ("gaze_fusion", "readout", "gaze_relation", "gaze_redistribution", "gaze_alignment", "prediction_distribution"):
                    if architecture[key] != m[key]:
                        raise ValueError(f"Raw architecture and run disagree: {key}")
                reconstruction = architecture["reconstruction"]
                for key, saved_key in (("model_id", "decoder_model_id"), ("model_revision", "decoder_revision"),
                                       ("et_model_id", "et_repo_id"), ("et_revision", "et_revision"),
                                       ("et_filename", "et_filename"), ("finetuning_mode", "finetuning_mode"),
                                       ("attn_implementation", "attn_implementation")):
                    if reconstruction[saved_key] != m[key]:
                        raise ValueError(f"Raw reconstruction and training disagree: {key}")
                self.records[fold] = {
                    "final_model": str(final), "weights_sha256": sha256_file(final / "model.safetensors"),
                    "architecture_sha256": sha256_file(final / "decoder_va_architecture.json"),
                    "tokenizer_sha256": {name: sha256_file(final / name) for name in files[2:]},
                    "predictions_sha256": sha256_file(folder / "predictions.tsv"),
                    "held_out_fold": fold, "training_fold": 3 - fold,
                    "best_model_checkpoint": state["best_model_checkpoint"], "best_metric": state["best_metric"],
                    "training_rows": len(train), "evaluation_rows": len(evaluation),
                }
        oof = pd.read_csv(self.directory / "oof_predictions.tsv", sep="\t", keep_default_na=False)
        expected = pd.concat(all_frames).sort_values(["index", "held_out_fold"], kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(oof, expected, check_dtype=False, rtol=1e-7, atol=1e-9)
        verify_metrics(oof, read_json(self.directory / "oof_metrics.json"))
        tokenizers = [record["tokenizer_sha256"] for record in self.records.values()]
        if any(value != tokenizers[0] for value in tokenizers):
            raise ValueError("Raw source folds have different saved tokenizers.")

    def provenance(self):
        """Record explicit source folds and immutable file hashes in every new checkpoint."""
        return {"source_run": str(self.directory), "source_manifest": self.manifest,
                "folds": {str(key): value for key, value in self.records.items()}}

    def load(self, fold, *, dtype, et_cache_size, tokenizer=None):
        """Strictly load only the requested opposite-fold pretrained predictor."""
        record = self.records[fold]
        directory = Path(record["final_model"])
        if sha256_file(directory / "model.safetensors") != record["weights_sha256"]:
            raise ValueError("Raw source weights changed after validation.")
        if sha256_file(directory / "decoder_va_architecture.json") != record["architecture_sha256"]:
            raise ValueError("Raw source architecture changed after validation.")
        if any(sha256_file(directory / name) != digest for name, digest in record["tokenizer_sha256"].items()):
            raise ValueError("Raw source tokenizer changed after validation.")
        return load_saved_decoder_va_model(directory, dtype=dtype, et_cache_size=et_cache_size, tokenizer=tokenizer)

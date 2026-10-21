"""Diagnose a completed DDPM using frozen-feature probes and fixed sampling subsets."""

import argparse
from dataclasses import replace
from datetime import datetime
import gc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from safetensors.torch import save_file
import torch
from torch.utils.data import DataLoader

from diffusion_code.ddpm import GaussianDDPM
from diffusion_code.diagnostic_data import verified_data, verified_fold, fixed_subset, require_new_output
from diffusion_code.model import load_diffusion_model
from diffusion_code.probe import train_probe, predict_head
from diffusion_code.train import baseline, source_hashes, write_json
from va_model_code.decoder_va.dataset import TokenizedVADataset, VABatchCollator
from va_model_code.decoder_va.downloads import sha256_file
from va_model_code.decoder_va.evaluation import prediction_frame, metrics_from_prediction_frame, write_oof_reports
from va_model_code.decoder_va.paths import resolve_run_directory


def build_parser():
    """Expose independent diagnostics without changing training or saved model semantics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--data-dir", default=str(ROOT / "va_model_code/data/paper7_seed42"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--mode", choices=("sampling", "probe", "all"), default="all")
    p.add_argument("--held-out-folds", type=int, nargs="+", choices=(1, 2))
    p.add_argument("--precision", choices=baseline.PRECISION_MODES)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rows-per-dataset", type=int, default=48)
    p.add_argument("--subset-seed", type=int, default=2026)
    p.add_argument("--samples", type=int, nargs="+", default=[16, 64])
    p.add_argument("--sampling-seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    p.add_argument("--memory-trt", choices=("aligned", "zero"), nargs="+", default=["aligned"])
    p.add_argument("--probe-epochs", type=int, default=30)
    p.add_argument("--probe-learning-rate", type=float, default=1e-3)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--probe-weight-decay", type=float, default=0.01)
    p.add_argument("--probe-seed", type=int, default=42)
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def data_loader(frame, tokenizer, max_length, batch_size):
    """Use the original truncation and padding policy with sequential row order."""
    return DataLoader(TokenizedVADataset(frame, tokenizer, max_length=max_length),
                      batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=VABatchCollator(tokenizer, pad_to_multiple_of=8))


def write_predictions(directory, frame, predictions, fold, *, dispersion=None):
    """Persist aligned predictions and recomputed metrics, keeping partition roles separate."""
    directory.mkdir(parents=True, exist_ok=True)
    result = prediction_frame(frame, frame[["valence", "arousal"]].to_numpy(dtype=np.float32), predictions, fold=fold)
    metrics = metrics_from_prediction_frame(result)
    result.to_csv(directory / "predictions.tsv", sep="\t", index=False)
    write_json(directory / "metrics.json", metrics)
    rows = [{"dataset_of_origin": name, **metrics_from_prediction_frame(group)}
            for name, group in result.groupby("dataset_of_origin")]
    pd.DataFrame(rows).to_csv(directory / "metrics_by_dataset.tsv", sep="\t", index=False)
    if dispersion is not None:
        table = result[["index", "held_out_fold"]].copy()
        table[["generated_variance_valence", "generated_variance_arousal"]] = dispersion
        table.to_csv(directory / "sample_dispersion.tsv", sep="\t", index=False)
    return result, metrics


@torch.no_grad()
def sampling_predictions(model, loader, *, cases, memory_modes, device, dtype, amp):
    """Encode once per batch and reuse label-free conditions across every sampler case."""
    if "zero" in memory_modes and model.settings.conditioning != "token-trt":
        raise ValueError("Direct-memory TRT intervention requires token-trt conditioning.")
    samplers = {case: GaussianDDPM(replace(model.settings, samples=case[0], sampling_seed=case[1])).to(device)
                for case in cases}
    means = {(mode, *case): [] for mode in memory_modes for case in cases}
    variances = {key: [] for key in means}
    model.eval()
    for batch in loader:
        inputs = {key: batch[key].to(device) for key in ("input_ids", "attention_mask")}
        with torch.autocast(device.type, dtype=dtype, enabled=amp):
            features = model.encoder.encode(**inputs, return_aligned_tokens=model.settings.conditioning == "token-trt")
        for mode in memory_modes:
            modified = features if mode == "aligned" else replace(features, raw_gaze=torch.zeros_like(features.raw_gaze))
            with torch.autocast(device.type, enabled=False):
                condition = model.denoiser.prepare(modified)
                for case, sampler in samplers.items():
                    draws = sampler.sample(lambda noisy, time: model.denoiser(noisy, time, condition), batch_size=len(inputs["input_ids"]))
                    means[(mode, *case)].append(draws.mean(1).cpu())
                    variances[(mode, *case)].append(draws.var(1, unbiased=True).cpu())
    return {key: (torch.cat(means[key]).numpy(), torch.cat(variances[key]).numpy()) for key in means}


@torch.no_grad()
def frozen_features(encoder, loader, *, device, dtype, amp):
    """Extract pooled features without labels, dropout or parameter gradients."""
    encoder.eval()
    encoder.requires_grad_(False)
    values = []
    for batch in loader:
        inputs = {key: batch[key].to(device) for key in ("input_ids", "attention_mask")}
        with torch.autocast(device.type, dtype=dtype, enabled=amp):
            values.append(encoder.encode(**inputs).pooled.detach().float().cpu())
    return torch.cat(values)


def write_probe_comparison(source, output, folds, probe_frames):
    """Compare the frozen-feature head with saved DDPM predictions on identical held-out rows."""
    originals = [pd.read_csv(source / f"heldout_fold{fold}/predictions.tsv", sep="\t", keep_default_na=False)
                 for fold in folds]
    for fold, original in zip(folds, originals):
        directory = output / f"heldout_fold{fold}/probe/source_ddpm"
        directory.mkdir()
        original.to_csv(directory / "predictions.tsv", sep="\t", index=False)
        write_json(directory / "metrics.json", metrics_from_prediction_frame(original))
    rows = []
    for scope, original, probe in [*zip(folds, originals, probe_frames),
                                    ("oof", pd.concat(originals), pd.concat(probe_frames))]:
        identity = ["index", "held_out_fold", "text", "dataset_of_origin", "valence", "arousal"]
        pd.testing.assert_frame_equal(original[identity].reset_index(drop=True), probe[identity].reset_index(drop=True),
                                      check_dtype=False, rtol=1e-7, atol=1e-9)
        for condition, frame in [("saved_ddpm", original), ("frozen_encoder_direct_head", probe)]:
            rows.append({"held_out_fold": scope, "scope": "all", "condition": condition,
                         **metrics_from_prediction_frame(frame)})
            for name, group in frame.groupby("dataset_of_origin", sort=True):
                rows.append({"held_out_fold": scope, "scope": f"dataset:{name}", "condition": condition,
                             **metrics_from_prediction_frame(group)})
    pd.DataFrame(rows).to_csv(output / "probe_comparison.tsv", sep="\t", index=False)


def run(args):
    """Validate ownership first, run requested diagnostics, and certify outputs last."""
    if min(args.batch_size, args.rows_per_dataset, args.probe_epochs, args.probe_batch_size) < 1:
        raise ValueError("Batch sizes, subset size and epochs must be positive.")
    if not args.samples or min(args.samples) < 2 or len(set(args.samples)) != len(args.samples):
        raise ValueError("Use unique sample counts >= 2 to estimate dispersion.")
    if not args.sampling_seeds or any(not 0 <= s < 2**63 for s in args.sampling_seeds):
        raise ValueError("Sampling seeds must be in [0, 2**63).")
    if len(set(args.sampling_seeds)) != len(args.sampling_seeds) or len(set(args.memory_trt)) != len(args.memory_trt):
        raise ValueError("Duplicate diagnostic settings.")
    if not np.isfinite(args.probe_learning_rate) or args.probe_learning_rate <= 0:
        raise ValueError("Probe learning rate must be positive and finite.")
    if not np.isfinite(args.probe_weight_decay) or args.probe_weight_decay < 0:
        raise ValueError("Probe weight decay must be nonnegative and finite.")
    if not 0 <= args.probe_seed < 2**32 - 1 or not 0 <= args.subset_seed < 2**63:
        raise ValueError("Invalid diagnostic seed.")
    source = resolve_run_directory(args.run_name)
    manifest, filtered = verified_data(source, args.data_dir)
    folds = args.held_out_folds or manifest["arguments"]["held_out_folds"]
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("Requested folds must be unique and nonempty.")
    contexts = {fold: verified_fold(source, manifest, filtered, fold) for fold in folds}
    precision = args.precision or manifest["arguments"]["precision"]
    dtype, bf16, fp16 = baseline._runtime_precision(args.use_cpu, precision)
    if not args.use_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --use-cpu explicitly for CPU diagnostics.")
    device = torch.device("cpu" if args.use_cpu else "cuda")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = require_new_output(args.output_dir or source.parent / f"{source.name}_diagnostics_{stamp}", source)
    requested_cases = [(n, seed) for n in args.samples for seed in args.sampling_seeds]
    original_case = (manifest["diffusion"]["samples"], manifest["diffusion"]["sampling_seed"])
    if original_case[0] < 2:
        raise ValueError("Source sampler needs >=2 samples for dispersion diagnostics.")
    cases = sorted(set([*requested_cases, original_case]))
    memory_modes = list(dict.fromkeys(["aligned", *args.memory_trt]))
    if "zero" in memory_modes and manifest["diffusion"]["conditioning"] != "token-trt":
        raise ValueError("Direct-memory TRT intervention requires token-trt conditioning.")
    selections = {}
    for fold, (_, train, evaluation, _) in contexts.items():
        selections[fold] = {partition: fixed_subset(frame, per_dataset=args.rows_per_dataset,
                           seed=args.subset_seed, fold=fold, partition=partition)
                           for partition, frame in [("train", train), ("heldout", evaluation)]}
        print(f"Diagnostic fold {fold}: train={len(train):,}; heldout={len(evaluation):,}; "
              f"sampling subset train={len(selections[fold]['train'])}, heldout={len(selections[fold]['heldout'])}.", flush=True)
    if args.dry_run:
        print(f"Dry run: no models loaded. Planned diagnostics: {output}")
        return None
    output.mkdir(parents=True, exist_ok=True)
    parameters = {"schema_version": 1, "kind": "ddpm_post_training_diagnostics", "arguments": vars(args),
                  "source_run": str(source), "source_manifest": manifest, "precision": precision,
                  "held_out_folds": folds, "sampling_cases": cases, "requested_sampling_cases": requested_cases,
                  "memory_modes": memory_modes, "original_case": original_case,
                  "source_sha256": source_hashes(), "source_weights_sha256": {},
                  "sampling_selection": "label_independent_hash_per_source_and_fold",
                  "sampling_scope": "source_stratified_subsets_not_full_OOF",
                  "probe_scope": "full_original_folds_with_frozen_DDPM_encoder",
                  "memory_zero_scope": "only_direct_memory_TRT_changes_prefix_and_pooled_context_unchanged",
                  "noise_policy": "legacy_common_bank_per_count_not_nested_between_counts"}
    for fold, (final, _, _, _) in contexts.items():
        parameters["source_weights_sha256"][str(fold)] = sha256_file(final / "model.safetensors")
    write_json(output / "diagnostic_parameters.json", parameters)
    sampling_rows, probe_frames = [], []
    for fold, (final, train, evaluation, contract) in contexts.items():
        destination = output / f"heldout_fold{fold}"
        destination.mkdir()
        for partition, selected in selections[fold].items():
            selected.to_csv(destination / f"{partition}_selected_rows.tsv", sep="\t", index=False)
        print(f"Loading fold {fold} weights for {args.mode} diagnostics.", flush=True)
        model, tokenizer = load_diffusion_model(final, dtype=dtype)
        if sha256_file(final / "model.safetensors") != parameters["source_weights_sha256"][str(fold)]:
            raise ValueError("Source weights changed while the diagnostic model was loading.")
        model.to(device).eval().requires_grad_(False)
        max_length = contract["arguments"]["max_length"]
        if args.mode in ("sampling", "all"):
            for partition, selected in selections[fold].items():
                print(f"Fold {fold}: sampling {partition}, {len(selected)} rows, {len(cases)} settings.", flush=True)
                loader = data_loader(selected, tokenizer, max_length, args.batch_size)
                results = sampling_predictions(model, loader, cases=cases, memory_modes=memory_modes,
                                               device=device, dtype=dtype, amp=bf16 or fp16)
                saved_predictions = None
                if partition == "heldout":
                    saved_predictions = pd.read_csv(source / f"heldout_fold{fold}/predictions.tsv", sep="\t").set_index("index")
                for (mode, count, seed), (predictions, variance) in results.items():
                    directory = destination / "sampling" / partition / f"{mode}_n{count}_seed{seed}"
                    _, metrics = write_predictions(directory, selected, predictions,
                        fold if partition == "heldout" else 3 - fold, dispersion=variance)
                    row = {"model_heldout_fold": fold, "partition": partition, "memory_trt": mode,
                           "samples": count, "sampling_seed": seed,
                           "requested_seed_sweep": (count, seed) in requested_cases,
                           "mean_generated_variance": float(variance.mean()),
                           "estimated_mc_mean_variance": float(variance.mean() / count), **metrics}
                    if partition == "heldout" and mode == "aligned" and (count, seed) == original_case:
                        original = saved_predictions.loc[selected["index"], ["pred_valence", "pred_arousal"]].to_numpy()
                        row["max_abs_difference_from_source_predictions"] = float(np.max(np.abs(original - predictions)))
                    sampling_rows.append(row)
                    write_json(directory / "sampling_metrics.json", row)
                    print({key: row[key] for key in ["model_heldout_fold", "partition", "memory_trt", "samples", "sampling_seed", "mse_mean"]}, flush=True)
        if args.mode in ("probe", "all"):
            print(f"Fold {fold}: extracting frozen pooled features for the full train/heldout folds.", flush=True)
            features = {partition: frozen_features(model.encoder, data_loader(frame, tokenizer, max_length, args.batch_size),
                         device=device, dtype=dtype, amp=bf16 or fp16)
                        for partition, frame in [("train", train), ("heldout", evaluation)]}
            dropout = model.encoder.classifier_dropout
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.mode in ("probe", "all"):
            labels = {partition: torch.tensor(frame[["valence", "arousal"]].to_numpy(), dtype=torch.float32)
                      for partition, frame in [("train", train), ("heldout", evaluation)]}
            head, state = train_probe(features["train"], labels["train"], features["heldout"], labels["heldout"],
                seed=args.probe_seed + fold - 1, epochs=args.probe_epochs,
                learning_rate=args.probe_learning_rate, batch_size=args.probe_batch_size,
                weight_decay=args.probe_weight_decay, dropout=dropout, device=device,
                callback=lambda row: print({"probe_fold": fold, **row}, flush=True))
            probe_directory = destination / "probe"
            probe_directory.mkdir()
            save_file({name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()},
                      str(probe_directory / "probe_head.safetensors"))
            write_json(probe_directory / "probe_state.json", state)
            write_json(probe_directory / "probe_architecture.json", {
                "width": features["train"].shape[1], "dropout": dropout,
                "layers": "LayerNorm-Dropout-Linear-Hardsigmoid", "compute_dtype": "float32",
                "encoder_frozen_eval": True, "source_weights_sha256": parameters["source_weights_sha256"][str(fold)],
                "seed": args.probe_seed + fold - 1,
            })
            for partition, frame in [("train", train), ("heldout", evaluation)]:
                predictions = predict_head(head, features[partition], batch_size=args.probe_batch_size, device=device)
                result, metrics = write_predictions(probe_directory / partition, frame, predictions,
                                                    fold if partition == "heldout" else 3 - fold)
                if partition == "heldout":
                    if not np.isclose(metrics["mse_mean"], state["best_metric"], rtol=1e-6, atol=1e-9):
                        raise ValueError("Restored probe head does not reproduce the selected metric.")
                    probe_frames.append(result)
            del features, labels, head
        write_json(destination / "COMPLETED.json", {"fold": fold, "mode": args.mode})
    if sampling_rows:
        table = pd.DataFrame(sampling_rows)
        table.to_csv(output / "sampling_comparison.tsv", sep="\t", index=False)
        columns = ["mse_mean", "mae_mean", "pearson_corr_valence", "pearson_corr_arousal", "ccc_mean", "estimated_mc_mean_variance"]
        sweep = table.loc[table.requested_seed_sweep]
        aggregated = sweep.groupby(["model_heldout_fold", "partition", "memory_trt", "samples"])[columns].agg(["mean", "std"])
        aggregated.columns = ["_".join(name) for name in aggregated.columns]
        aggregated.to_csv(output / "sampling_seed_summary.tsv", sep="\t")
    if probe_frames:
        write_oof_reports(probe_frames, output / "probe_oof", run_parameters={
            "kind": "frozen_DDPM_encoder_direct_regression_probe", "held_out_folds": folds,
            "source_run": str(source), "selection": "same_heldout_reuse_as_original_protocol",
        })
        write_probe_comparison(source, output, folds, probe_frames)
    files = {str(path.relative_to(output)): sha256_file(path) for path in output.rglob("*")
             if path.is_file() and path.suffix != ".safetensors"}
    write_json(output / "COMPLETED.json", {"schema_version": 1, "kind": "ddpm_post_training_diagnostics",
               "held_out_folds": folds, "mode": args.mode, "sha256": files})
    print(f"Completed DDPM diagnostics: {output}", flush=True)
    return output


if __name__ == "__main__":
    run(build_parser().parse_args())

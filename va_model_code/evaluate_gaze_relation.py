"""Evaluate paired TRT-rotation controls on saved, completed held-out checkpoints."""

from __future__ import annotations

import argparse
import gc
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from decoder_va.external_benchmarks import TextBatchCollator, TokenizedTextDataset, discover_completed_run
from decoder_va.gaze_relation_diagnostics import (
    ARMS, evaluate_paired_batches, paired_prediction_frame,
    relation_modules, validate_heldout_data, write_diagnostic_results,
)
from decoder_va.model import load_saved_decoder_va_model


def build_parser():
    """Expose frozen-checkpoint diagnostics without training or overwrite options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed two-fold run with final_model weights.")
    parser.add_argument("--data-dir", help="Original fold files; SHA256 must match the saved run.")
    parser.add_argument("--held-out-folds", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--eval-batch-size", type=int, help="Default: the saved evaluation batch size.")
    parser.add_argument("--precision", choices=("checkpoint", "fp32", "bf16", "fp16"), default="checkpoint")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, help="New directory; existing paths are never overwritten.")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate saved artifacts/data without loading models.")
    return parser


def resolve_data_dir(explicit, recorded, run_dir):
    """Resolve original data location without silently selecting an ambiguous directory."""

    if explicit:
        result = Path(explicit).expanduser().resolve()
        if not result.is_dir():
            raise FileNotFoundError(result)
        return result
    if not recorded:
        raise ValueError("Saved run has no data_dir; provide --data-dir.")
    path = Path(str(recorded)).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path(__file__).resolve().parent / path,
                                                    Path(run_dir).parent.parent / "va_model_code" / path]
    existing = {candidate.resolve() for candidate in candidates if candidate.is_dir()}
    if len(existing) != 1:
        raise ValueError("Cannot resolve original fold directory uniquely; provide --data-dir.")
    return existing.pop()


def resolve_precision(requested, recorded, device):
    """Default to checkpoint computation; make any override explicit in provenance."""

    name = recorded if requested == "checkpoint" else {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[requested]
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported checkpoint precision: {name!r}.")
    if device.type == "cpu" and name != "float32":
        raise ValueError("CPU diagnostics require --precision fp32; this is an explicit precision override.")
    if device.type == "cuda" and name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support BF16.")
    return mapping[name], name


def run(args):
    """Validate source identity, evaluate paired controls, and publish a fresh report."""

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if len(set(args.held_out_folds)) != len(args.held_out_folds):
        raise ValueError("--held-out-folds must not contain duplicates.")
    if not 0 <= args.shuffle_seed <= 2**63 - 1:
        raise ValueError("--shuffle-seed must be in [0, 2**63 - 1].")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be positive.")
    members = discover_completed_run(args.run_dir)
    members = tuple(member for member in members if member.held_out_fold in args.held_out_folds)
    for member in members:
        relation = member.run_manifest.get("gaze_relation", {})
        if relation.get("method") not in {"trt-relative-rotation", "trt-readout-rotation"}:
            raise ValueError("Diagnostics require a trained TRT rotation checkpoint.")
        if relation.get("input_mode", "aligned") != "aligned":
            raise ValueError("These diagnostic controls require aligned TRT training.")
    data_dir = resolve_data_dir(args.data_dir, members[0].run_manifest.get("data_dir"), args.run_dir)
    frames, hashes = validate_heldout_data(members, data_dir)
    report = {
        "version": 1,
        "purpose": "Same trained checkpoint inference ablations; not a Raw-versus-rotation training comparison.",
        "arms": list(ARMS), "shuffle_seed": args.shuffle_seed,
        "shuffle_scope": "Only mapped relation TRT values within each example; original raw TRT prefix unchanged. Zero/singleton anchors cannot be shuffled.",
        "run_dir": str(Path(args.run_dir).resolve()), "source_fold_sha256": hashes,
        "data_dir": str(data_dir), "held_out_folds": list(args.held_out_folds),
        "output_contains_raw_text": False,
        "members": {},
    }
    for member in members:
        report["members"][member.name] = {"saved_file_sha256": dict(member.file_sha256),
                                          "relation": dict(member.run_manifest["gaze_relation"]),
                                          "readout": member.run_manifest.get("readout", {"mode": "last-token"})}
    if args.dry_run:
        print(json.dumps({"status": "preflight_passed", **report}, ensure_ascii=False, indent=2))
        return report
    device = torch.device("cpu" if args.use_cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; specify --use-cpu --precision fp32 for CPU evaluation.")
    outputs = []
    for member in members:
        manifest = member.run_manifest
        batch_size = args.eval_batch_size or int(manifest["eval_batch_size"])
        recorded = str(manifest.get("compute_dtype", manifest.get("dtype")))
        dtype, precision_name = resolve_precision(args.precision, recorded, device)
        print(f"Loading {member.name}: paired learned / zero alpha / shuffled TRT", flush=True)
        model, tokenizer = load_saved_decoder_va_model(member.model_dir, dtype=dtype)
        model.to(device)
        if model.gaze_relation != dict(manifest["gaze_relation"]):
            raise ValueError("Reloaded relation contract disagrees with saved run.")
        if "readout" in manifest and getattr(model, "readout", None) != manifest["readout"]:
            raise ValueError("Reloaded readout contract disagrees with saved run.")
        frame = frames[member.held_out_fold]
        loader = DataLoader(TokenizedTextDataset(frame["text"].tolist(), tokenizer, max_length=int(manifest["max_length"])),
                            batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=TextBatchCollator(tokenizer, pad_to_multiple_of=8))
        started = time.perf_counter()
        predictions, descriptions, phases = evaluate_paired_batches(model, loader, device=device, dtype=dtype,
                                                                    shuffle_seed=args.shuffle_seed)
        persisted = pd.read_csv(member.model_dir.parent / "predictions.tsv", sep="\t")
        reference = persisted[["pred_valence", "pred_arousal"]].to_numpy(float)
        error = float(np.max(np.abs(predictions["learned"] - reference)))
        tolerances = {"bfloat16": .005, "float16": .002, "float32": 1e-5}
        tolerance = max(tolerances[recorded], tolerances[precision_name])
        if error > tolerance:
            raise RuntimeError(f"{member.name}: learned-arm predictions differ from saved results (max {error:.8g} > {tolerance:.8g}); refusing to report ablations.")
        member_report = report["members"][member.name]
        member_report.update({"rows": len(frame), "batch_size": batch_size,
                              "precision": precision_name, "recorded_precision": recorded,
                              "precision_override": precision_name != recorded,
                              "reload_max_absolute_prediction_error": error, "reload_absolute_tolerance": tolerance,
                              "phase_diagnostics": phases,
                              "learned_alpha": {name: {"alpha": module.alpha.detach().cpu().tolist(), "scale": module.scale}
                                                for name, module in relation_modules(model)},
                              "elapsed_seconds": time.perf_counter() - started})
        outputs.append(paired_prediction_frame(frame, predictions, descriptions, held_out_fold=member.held_out_fold))
        print(f"Completed {member.name}: {len(frame):,} paired rows; reload max error {error:.6g}", flush=True)
        del model, loader, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["device"] = str(device)
    report["packages"] = {}
    for package in ("torch", "transformers", "flash-linear-attention", "causal-conv1d"):
        try:
            report["packages"][package] = version(package)
        except PackageNotFoundError:
            report["packages"][package] = None
    report["status"] = "completed"
    destination = write_diagnostic_results(output, pd.concat(outputs, ignore_index=True), report)
    print(f"Completed paired TRT diagnostics: {destination}", flush=True)
    return report


def main():
    """Parse and execute the inference-only command."""

    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

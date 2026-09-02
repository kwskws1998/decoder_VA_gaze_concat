"""Inference-only zero-shot evaluation on OMG-Emotion and MSP-Podcast 2.0."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from decoder_va.external_benchmarks import (
    MSP_NAME,
    OMG_NAME,
    TextBatchCollator,
    TokenizedTextDataset,
    audit_finetuning_text_overlap,
    discover_completed_run,
    download_pinned_omg_test,
    fixed_unweighted_ensemble,
    load_msp_podcast_test,
    load_omg_emotion_test,
    reject_benchmark_training_sources,
    write_external_evaluation,
)
from decoder_va.model import load_saved_decoder_va_model


SCRIPT_DIR = Path(__file__).resolve().parent
PRECISION_CHOICES = ("checkpoint",)
DEVICE_CHOICES = ("auto", "cuda", "cpu")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options shared by both immutable external benchmark protocols."""

    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Two-fold evaluation bundle containing training_parameters.json, both "
            "heldout_fold*/run_manifest.json files, and both complete final_model "
            "directories. Default preflight additionally requires the original "
            "internal prediction/metric reports."
        ),
    )
    parser.add_argument(
        "--training-data-dir",
        help=(
            "Original full_dataset_fold1.csv/full_dataset_fold2.csv directory. "
            "When omitted, the evaluator resolves the recorded run data_dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "New result directory. Default: <run-dir>/external_benchmarks/"
            "<benchmark>/<split>. Existing paths are never overwritten."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Inference batch size. Default: the saved training eval_batch_size.",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Inference device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISION_CHOICES,
        default="checkpoint",
        help=(
            "Use the checkpoint-recorded training dtype and matching CUDA autocast. "
            "Precision overrides are deliberately unavailable in the primary "
            "external-generalization evaluator."
        ),
    )
    parser.add_argument(
        "--et-cache-size",
        type=int,
        help="Optional ET2 LRU override; the saved architecture value is used by default.",
    )
    parser.add_argument(
        "--require-overlap-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verify original fold hashes and report exact normalized fine-tuning text "
            "overlap before external inference (default: true)."
        ),
    )
    parser.add_argument(
        "--preflight-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Re-predict saved internal held-out rows before external inference and "
            "compare against persisted predictions (default: true). Disable only for "
            "a raw-text-free Hugging Face model bundle that deliberately omits "
            "raw-text internal prediction files."
        ),
    )
    parser.add_argument(
        "--preflight-rows",
        type=int,
        default=32,
        help="Number of internal rows checked per fold member (default: 32).",
    )
    parser.add_argument(
        "--include-gold-labels",
        action="store_true",
        help=(
            "Include native gold VA in local predictions.tsv. Raw transcript text is "
            "never persisted. Do not redistribute licensed MSP item data."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, run, scale, split, hashes, and overlap without loading models.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Define dataset-specific zero-shot commands without any training options."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate both frozen fold checkpoints and their fixed unweighted ensemble "
            "on untouched external VA benchmarks. This program has no training path."
        )
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    omg = subparsers.add_parser(
        OMG_NAME,
        help="Official pinned OMG-Emotion test transcript-only transfer.",
    )
    _add_common_arguments(omg)
    omg.add_argument(
        "--raw-dir",
        required=True,
        help="Local directory for the two pinned official OMG test CSV files.",
    )
    omg.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download the pinned official transcript and gold-label files into "
            "--raw-dir after checking the corpus terms."
        ),
    )

    msp = subparsers.add_parser(
        MSP_NAME,
        help="Licensed MSP-Podcast 2.0 Test1 or Test2 text-only transfer.",
    )
    _add_common_arguments(msp)
    msp.add_argument(
        "--split",
        required=True,
        choices=("test1", "test2"),
        help="Evaluate Test1 or Test2 separately. Train/Development/Test3 are rejected.",
    )
    msp.add_argument(
        "--labels-file",
        required=True,
        help="Authorized local MSP-Podcast 2.0 labels_consensus.csv.",
    )
    msp.add_argument(
        "--transcripts",
        required=True,
        help="Authorized transcript CSV/TSV, directory of TXT files, or ZIP archive.",
    )
    msp.add_argument("--transcript-id-column")
    msp.add_argument("--transcript-text-column")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject non-inference or irreproducible values before model loading."""

    if args.precision != "checkpoint":
        raise ValueError(
            "External benchmark inference must use --precision checkpoint. Precision "
            "sensitivity runs are not allowed to share this primary evaluation path."
        )
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.et_cache_size is not None and args.et_cache_size < 0:
        raise ValueError("--et-cache-size cannot be negative.")
    if args.preflight_rows <= 0:
        raise ValueError("--preflight-rows must be positive.")


def _package_version(distribution: str) -> str | None:
    """Read an installed distribution version for the evaluation manifest."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _runtime_compatibility(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Report exact training/evaluation runtime differences without hiding preflight."""

    recorded = {
        "python": run_manifest.get("python_version"),
        "torch": run_manifest.get("torch_version"),
        "transformers": run_manifest.get("transformers_version"),
        "peft": run_manifest.get("peft_version"),
        "cuda_runtime": run_manifest.get("cuda_runtime"),
    }
    current = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "cuda_runtime": torch.version.cuda,
    }
    mismatches = {
        name: {"recorded": recorded[name], "current": current[name]}
        for name in recorded
        if recorded[name] is not None and recorded[name] != current[name]
    }
    return {
        "recorded": recorded,
        "current": current,
        "exact_version_match": not mismatches,
        "mismatches": mismatches,
        "policy": (
            "differences are reported; the numerical reload preflight is the canonical "
            "compatibility gate when enabled"
        ),
    }


def _resolve_training_data_dir(
    explicit: str | None,
    recorded: object,
    *,
    required: bool,
) -> Path | None:
    """Resolve the exact original fold directory without silently choosing ambiguity."""

    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Training data directory not found: {path}")
        return path
    if recorded is None:
        if required:
            raise ValueError(
                "The run manifest does not record data_dir; provide --training-data-dir."
            )
        return None
    recorded_path = Path(str(recorded)).expanduser()
    candidates = (
        (recorded_path.resolve(),)
        if recorded_path.is_absolute()
        else (
            (Path.cwd() / recorded_path).resolve(),
            (SCRIPT_DIR / recorded_path).resolve(),
        )
    )
    existing = []
    for candidate in candidates:
        if candidate.is_dir() and candidate not in existing:
            existing.append(candidate)
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(
            "Recorded relative data_dir resolves to multiple directories; provide "
            "--training-data-dir explicitly: " + ", ".join(map(str, existing))
        )
    if required:
        raise FileNotFoundError(
            f"Cannot resolve recorded training data_dir {recorded!r}; provide "
            "--training-data-dir from the training server."
        )
    return None


def _resolve_device(requested: str) -> torch.device:
    """Choose one explicit inference device and validate CUDA availability."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is not available.")
    return device


def _resolve_dtype(
    requested: str,
    recorded: str,
    device: torch.device,
) -> tuple[torch.dtype, str, bool]:
    """Resolve checkpoint-faithful construction and forward-autocast precision."""

    precision = recorded if requested == "checkpoint" else {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }[requested]
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if precision not in mapping:
        raise ValueError(f"Unsupported inference precision: {precision!r}.")
    dtype = mapping[precision]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError(
            f"The recorded {precision} checkpoint requires CUDA for checkpoint-faithful "
            "external inference. CPU can be used for --dry-run validation only."
        )
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 external inference requires a BF16-capable CUDA GPU.")
    autocast_enabled = device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}
    return dtype, precision, autocast_enabled


def _autocast_context(
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
):
    """Mirror Trainer CUDA autocast while leaving the saved FP32 path untouched."""

    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _predict_texts(
    model,
    tokenizer,
    texts: Sequence[object],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    autocast_enabled: bool,
) -> np.ndarray:
    """Run ordered label-free inference and require finite bounded two-output rows."""

    dataset = TokenizedTextDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=TextBatchCollator(tokenizer, pad_to_multiple_of=8),
    )
    batches = []
    model.eval()
    model.requires_grad_(False)
    with torch.inference_mode():
        for batch in loader:
            model_inputs = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            with _autocast_context(device, dtype, autocast_enabled):
                output = model(**model_inputs)
            logits = output.logits
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(
                    f"Saved VA model must return [examples, 2], got {tuple(logits.shape)}."
                )
            batches.append(logits.detach().to(device="cpu", dtype=torch.float64).numpy())
    if not batches:
        raise ValueError("Cannot evaluate an empty text sequence.")
    predictions = np.concatenate(batches, axis=0)
    if len(predictions) != len(texts):
        raise RuntimeError("Inference row count does not match the ordered input rows.")
    if not np.isfinite(predictions).all():
        raise RuntimeError("Saved model produced non-finite external predictions.")
    if bool(((predictions < -1e-6) | (predictions > 1.0 + 1e-6)).any()):
        raise RuntimeError("Saved hard-sigmoid model produced predictions outside [0,1].")
    return predictions


def _filtered_fold_for_member(
    member,
    fold_frames: Mapping[int, pd.DataFrame],
    *,
    held_out: bool,
) -> pd.DataFrame:
    """Reconstruct one member's filtered training or held-out frame from provenance."""

    fold = member.held_out_fold if held_out else member.training_fold
    frame = fold_frames[fold]
    excluded = set(map(str, member.run_manifest.get("excluded_dataset_names", ())))
    return frame.loc[~frame["dataset_of_origin"].astype(str).isin(excluded)].copy()


def _preflight_member(
    member,
    model,
    tokenizer,
    fold_frames: Mapping[int, pd.DataFrame],
    *,
    rows: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    precision_name: str,
    autocast_enabled: bool,
) -> dict[str, Any]:
    """Reproduce persisted internal held-out predictions before touching benchmark scores."""

    heldout = _filtered_fold_for_member(member, fold_frames, held_out=True).reset_index(
        drop=True
    )
    expected_rows = int(member.run_manifest.get("evaluation_rows", -1))
    if len(heldout) != expected_rows:
        raise ValueError(
            f"Held-out row provenance mismatch for {member.name}: manifest "
            f"{expected_rows}, reconstructed {len(heldout)}."
        )
    persisted_path = member.model_dir.parent / "predictions.tsv"
    persisted = pd.read_csv(persisted_path, sep="\t", keep_default_na=False)
    if len(persisted) != len(heldout):
        raise ValueError(
            f"Persisted prediction row count mismatch for {member.name}: "
            f"{len(persisted)} != {len(heldout)}."
        )
    count = min(
        ((int(rows) + int(batch_size) - 1) // int(batch_size)) * int(batch_size),
        len(heldout),
    )
    expected_ids = heldout["index"].astype(str).iloc[:count].tolist()
    persisted_ids = persisted["index"].astype(str).iloc[:count].tolist()
    if expected_ids != persisted_ids:
        raise ValueError(
            f"Persisted prediction order does not match reconstructed {member.name} rows."
        )
    actual = _predict_texts(
        model,
        tokenizer,
        heldout["text"].iloc[:count].tolist(),
        max_length=int(member.run_manifest["max_length"]),
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        autocast_enabled=autocast_enabled,
    )
    reference = persisted.loc[
        : count - 1, ["pred_valence", "pred_arousal"]
    ].to_numpy(dtype=np.float64)
    max_abs_error = float(np.max(np.abs(actual - reference)))
    tolerances = {
        "bfloat16": 5e-3,
        "float16": 2e-3,
        "float32": 1e-5,
    }
    recorded_precision = str(member.run_manifest["dtype"])
    tolerance = max(tolerances[precision_name], tolerances[recorded_precision])
    if max_abs_error > tolerance:
        raise RuntimeError(
            f"{member.name} reload preflight differs from saved predictions: max abs "
            f"error {max_abs_error:.8g} exceeds {tolerance:.8g}. Refusing external scoring."
        )
    return {
        "status": "passed",
        "rows": int(count),
        "row_selection": (
            "first contiguous persisted held-out rows to preserve original eval "
            "batch grouping"
        ),
        "max_absolute_prediction_error": max_abs_error,
        "absolute_tolerance": tolerance,
        "recorded_prediction_dtype": recorded_precision,
        "reload_inference_dtype": precision_name,
        "reference": str(persisted_path.resolve()),
    }


def _cuda_memory_snapshot(device: torch.device) -> dict[str, Any]:
    """Record external-inference peak CUDA memory without invoking training memory code."""

    if device.type != "cuda":
        return {"cuda_enabled": False}
    index = device.index if device.index is not None else torch.cuda.current_device()
    gib = float(1024**3)
    allocated = int(torch.cuda.max_memory_allocated(index))
    reserved = int(torch.cuda.max_memory_reserved(index))
    return {
        "cuda_enabled": True,
        "device_index": int(index),
        "device_name": torch.cuda.get_device_name(index),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "peak_allocated_gib": round(allocated / gib, 4),
        "peak_reserved_gib": round(reserved / gib, 4),
    }


def _default_output_dir(run_dir: Path, benchmark: str, split: str) -> Path:
    """Anchor external results under the immutable source training run by default."""

    return run_dir / "external_benchmarks" / benchmark / split


def run(args: argparse.Namespace) -> Path | None:
    """Validate provenance, infer sequentially, ensemble once, and write final reports."""

    _validate_args(args)
    run_dir = Path(args.run_dir).expanduser().resolve()
    members = discover_completed_run(
        run_dir,
        require_internal_evidence=bool(args.preflight_check),
    )
    if args.benchmark == OMG_NAME:
        if args.download:
            download_pinned_omg_test(args.raw_dir)
        benchmark = load_omg_emotion_test(args.raw_dir, strict_official_contract=True)
    else:
        benchmark = load_msp_podcast_test(
            args.labels_file,
            args.transcripts,
            split=args.split,
            transcript_id_column=args.transcript_id_column,
            transcript_text_column=args.transcript_text_column,
            strict_official_count=True,
        )
    reject_benchmark_training_sources(members, benchmark.name)

    needs_training_folds = bool(args.require_overlap_audit or args.preflight_check)
    training_data_dir = None
    if needs_training_folds or args.training_data_dir is not None:
        training_data_dir = _resolve_training_data_dir(
            args.training_data_dir,
            members[0].run_manifest.get("data_dir"),
            required=needs_training_folds,
        )
    if training_data_dir is not None:
        audited_frame, overlap_report, fold_frames = audit_finetuning_text_overlap(
            benchmark,
            members,
            training_data_dir,
        )
    else:
        audited_frame = benchmark.frame.copy()
        overlap_report = {
            "status": "not_run",
            "reason": "--no-require-overlap-audit and --no-preflight-check",
            "base_pretraining_contamination": "not auditable from this repository",
            "frozen_et2_training_contamination": (
                "not auditable from this repository; consult the pinned ET2 model card"
            ),
        }
        fold_frames = None

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(run_dir, benchmark.name, benchmark.split)
    )
    if output_dir.exists():
        raise FileExistsError(
            f"External evaluation output already exists; refusing to overwrite: {output_dir}"
        )
    batch_size = args.batch_size or int(members[0].run_manifest["eval_batch_size"])
    recorded_precision = str(members[0].run_manifest["dtype"])
    runtime_compatibility = _runtime_compatibility(members[0].run_manifest)
    print(
        f"Validated {benchmark.name} {benchmark.split}: {len(benchmark.frame):,} rows; "
        f"empty transcripts: {int(benchmark.frame['is_empty_text'].sum()):,}."
    )
    print(
        "External protocol: zero gradient updates, no calibration, no checkpoint "
        "selection, fixed two-member arithmetic mean."
    )
    if runtime_compatibility["mismatches"]:
        mismatch_names = ", ".join(runtime_compatibility["mismatches"])
        warning_suffix = (
            "The reload preflight must pass before canonical scoring."
            if args.preflight_check
            else "Reload preflight is disabled, so this weaker portability run must "
            "retain the version-mismatch warning."
        )
        print(
            "WARNING: evaluation runtime differs from training for: "
            f"{mismatch_names}. {warning_suffix}"
        )
    if args.dry_run:
        print(
            f"Planned inference device={args.device}; recorded precision="
            f"{recorded_precision}; batch_size={batch_size}; "
            f"max_length={members[0].run_manifest['max_length']}."
        )
        print(f"Dry run complete. Planned output: {output_dir}")
        return None

    device = _resolve_device(args.device)
    dtype, precision_name, autocast_enabled = _resolve_dtype(
        args.precision,
        recorded_precision,
        device,
    )
    print(
        f"Inference device={device}; precision={precision_name}; batch_size={batch_size}; "
        f"max_length={members[0].run_manifest['max_length']}."
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    member_predictions: dict[str, np.ndarray] = {}
    member_runtime: dict[str, Any] = {}
    preflight_report: dict[str, Any] = {}
    for member in members:
        print(f"Loading frozen member: {member.model_dir}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model, tokenizer = load_saved_decoder_va_model(
            member.model_dir,
            dtype=dtype,
            et_cache_size=args.et_cache_size,
        )
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        trainable_after_freeze = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if trainable_after_freeze != 0:
            raise RuntimeError("External evaluator failed to freeze every saved-model parameter.")
        if args.preflight_check:
            if fold_frames is None:
                raise RuntimeError("Internal preflight requires verified training fold frames.")
            preflight_report[member.name] = _preflight_member(
                member,
                model,
                tokenizer,
                fold_frames,
                rows=args.preflight_rows,
                batch_size=int(member.run_manifest["eval_batch_size"]),
                device=device,
                dtype=dtype,
                precision_name=precision_name,
                autocast_enabled=autocast_enabled,
            )
            print(
                f"Preflight {member.name}: max_abs_error="
                f"{preflight_report[member.name]['max_absolute_prediction_error']:.8g}."
            )
        else:
            preflight_report[member.name] = {
                "status": "not_run",
                "reason": "--no-preflight-check",
            }
        started = time.perf_counter()
        predictions = _predict_texts(
            model,
            tokenizer,
            benchmark.frame["text"].tolist(),
            max_length=int(member.run_manifest["max_length"]),
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            autocast_enabled=autocast_enabled,
        )
        elapsed = time.perf_counter() - started
        member_predictions[member.name] = predictions
        member_runtime[member.name] = {
            "runtime_seconds": float(elapsed),
            "samples_per_second": float(len(predictions) / elapsed),
            "trainable_parameters_during_evaluation": 0,
            "gpu_memory": _cuda_memory_snapshot(device),
        }
        print(
            f"Predicted {member.name}: {len(predictions):,} rows in {elapsed / 60:.2f} min."
        )
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble = fixed_unweighted_ensemble(list(member_predictions.values()))
    evaluation_manifest = {
        "run_dir": str(run_dir),
        "training_data_dir": str(training_data_dir) if training_data_dir else None,
        "device": str(device),
        "precision_request": args.precision,
        "inference_dtype": precision_name,
        "cuda_autocast": bool(autocast_enabled),
        "tf32_enabled": False,
        "batch_size": int(batch_size),
        "max_length_source": "saved run manifest",
        "et_cache_size_override": args.et_cache_size,
        "preflight": preflight_report,
        "runtime_compatibility": runtime_compatibility,
        "runtime": member_runtime,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "overlap_audit_required": bool(args.require_overlap_audit),
        "preflight_required": bool(args.preflight_check),
        "internal_evidence_validation": (
            "full persisted fold and OOF predictions/metrics"
            if args.preflight_check
            else "raw-text-free model bundle; persisted internal predictions not required"
        ),
    }
    written = write_external_evaluation(
        output_dir,
        benchmark,
        members,
        member_predictions,
        ensemble,
        audited_frame=audited_frame,
        overlap_report=overlap_report,
        evaluation_manifest=evaluation_manifest,
        include_gold_labels=args.include_gold_labels,
    )
    with open(written / "metrics.json", "r", encoding="utf-8") as input_file:
        metrics = json.load(input_file)
    primary = metrics["ensemble"]["subsets"]["official_all"]["native_scale"]
    print(
        f"Completed {benchmark.name} {benchmark.split}. Ensemble CCC: "
        f"V={primary['ccc_valence']:.6f}, A={primary['ccc_arousal']:.6f}, "
        f"mean={primary['ccc_mean']:.6f}."
    )
    print(f"Results: {written}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for immutable external evaluation."""

    args = _build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train and evaluate Qwen decoder VA models with optional ET2 gaze concat."""

from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
from typing import Sequence

import numpy as np
import torch
from transformers import TrainingArguments, set_seed

from decoder_va.dataset import TokenizedVADataset, VABatchCollator, load_auto_tokenizer
from decoder_va.downloads import sha256_file
from decoder_va.evaluation import (
    prediction_frame,
    trainer_compute_metrics,
    write_oof_reports,
)
from decoder_va.filters import dataset_counts, load_filtered_folds
from decoder_va.model import (
    ARCHITECTURE_MANIFEST_VERSION,
    DEFAULT_DECODER_MODEL_ID,
    DEFAULT_DECODER_REVISION,
    DEFAULT_ET2_REVISION,
    GAZE_PREFIX_ORDER,
    GAZE_PREFIX_POOLING,
    build_qwen_va_model,
)
from decoder_va.trainer import VARegressionTrainer


MODEL_ALIASES = ("qwen3.5-0.8b", "qwen")
LOSS_CHOICES = (
    "mse",
    "ccc",
    "mse+ccc",
    "heteroscedastic+ccc",
    "hetero+ccc",
)


def _build_parser() -> argparse.ArgumentParser:
    """Define the decoder training command-line contract."""

    parser = argparse.ArgumentParser(
        description=(
            "Two-fold out-of-fold VA regression with Qwen3.5-0.8B-Base and "
            "optional frozen ET2 TRT prefix concat."
        )
    )
    parser.add_argument("model", nargs="?", default="qwen3.5-0.8b", choices=MODEL_ALIASES)
    parser.add_argument(
        "loss",
        nargs="?",
        default="heteroscedastic+ccc",
        choices=LOSS_CHOICES,
    )
    parser.add_argument("--model-id", default=DEFAULT_DECODER_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_DECODER_REVISION)
    parser.add_argument(
        "--gaze-fusion",
        choices=("prefix-concat", "none"),
        default="prefix-concat",
    )
    parser.add_argument("--et-model-id", default="skboy/et_prediction_2")
    parser.add_argument("--et-revision", default=DEFAULT_ET2_REVISION)
    parser.add_argument(
        "--et-filename",
        default="et_predictor2_seed123.safetensors",
    )
    parser.add_argument("--et-cache-size", type=int, default=70000)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument("--no-iemocap", action="store_true")
    parser.add_argument(
        "--no-ieomcap",
        action="store_true",
        help="Typo-compatible alias for --no-iemocap.",
    )
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--held-out-folds",
        nargs="+",
        type=int,
        choices=(1, 2),
        default=(1, 2),
    )
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--hetero-mse-weight", type=float, default=0.1)
    parser.add_argument("--hetero-ccc-weight", type=float, default=0.1)
    parser.add_argument("--hetero-logvar-min", type=float, default=-5.0)
    parser.add_argument("--hetero-logvar-max", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attn-implementation")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--report-to", action="append", default=[])
    parser.add_argument("--use-cpu", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Fail before downloading models when a run configuration is invalid."""

    positive_integers = (
        "max_length",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "lora_rank",
        "lora_alpha",
    )
    for name in positive_integers:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or a positive integer.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in the interval [0, 1).")
    if args.et_cache_size < 0:
        raise ValueError("--et-cache-size cannot be negative.")
    if args.hetero_logvar_min >= args.hetero_logvar_max:
        raise ValueError("--hetero-logvar-min must be smaller than --hetero-logvar-max.")
    if args.hetero_mse_weight < 0 or args.hetero_ccc_weight < 0:
        raise ValueError("Heteroscedastic anchor weights cannot be negative.")
    if len(set(args.held_out_folds)) != len(args.held_out_folds):
        raise ValueError("--held-out-folds cannot contain duplicates.")
    if len(args.held_out_folds) != 1 and args.resume_from_checkpoint:
        raise ValueError(
            "--resume-from-checkpoint is ambiguous for multiple fold directories; "
            "resume one --held-out-folds value at a time."
        )


def _loss_name(raw_name: str) -> str:
    """Normalize the short heteroscedastic loss alias."""

    return (
        "heteroscedastic+ccc"
        if raw_name == "hetero+ccc"
        else raw_name
    )


def _output_dimension(loss_name: str) -> int:
    """Select the point-only or mean/log-variance output schema."""

    return 4 if loss_name == "heteroscedastic+ccc" else 2


def _runtime_precision(use_cpu: bool) -> tuple[torch.dtype, bool, bool]:
    """Choose a conservative dtype for CPU or the active CUDA GPU."""

    if use_cpu or not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def _default_output_dir() -> Path:
    """Create a collision-resistant run directory name."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = platform.node() or "unknown-host"
    return Path("Preds") / f"{timestamp}_{hostname}_qwen35"


def _json_ready(value):
    """Recursively convert argparse, NumPy, Torch, and path values to JSON types."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _validate_resume_contract(
    args: argparse.Namespace,
    fold_output: Path,
    expected_manifest: dict[str, object],
) -> None:
    """Reject checkpoints whose recorded architecture or training contract differs."""

    if not args.resume_from_checkpoint:
        return
    checkpoint_path = Path(args.resume_from_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {checkpoint_path}")
    checkpoint_root = (fold_output / "checkpoints").resolve()
    try:
        relative_checkpoint = checkpoint_path.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError(
            "Resume checkpoint must belong to the selected held-out fold at "
            f"{checkpoint_root}."
        ) from exc
    if (
        relative_checkpoint == Path(".")
        or len(relative_checkpoint.parts) != 1
        or not relative_checkpoint.name.startswith("checkpoint-")
    ):
        raise ValueError(
            "Resume checkpoint must be one direct Trainer checkpoint-* directory under "
            f"{checkpoint_root}."
        )

    manifest_path = fold_output / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Cannot safely resume without the original fold run manifest: "
            f"{manifest_path}"
        )
    try:
        with open(manifest_path, "r", encoding="utf-8") as input_file:
            recorded_manifest = json.load(input_file)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read resume run manifest: {manifest_path}") from exc
    if not isinstance(recorded_manifest, dict):
        raise ValueError(f"Resume run manifest must contain a JSON object: {manifest_path}")

    contract_fields = (
        "architecture_manifest_version",
        "model",
        "loss",
        "output_dim",
        "dtype",
        "model_id",
        "model_revision",
        "gaze_fusion",
        "gaze_concat_order",
        "pooling_position",
        "et_model_id",
        "et_revision",
        "et_filename",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "attn_implementation",
        "gradient_checkpointing",
        "max_length",
        "train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "hetero_mse_weight",
        "hetero_ccc_weight",
        "hetero_logvar_min",
        "hetero_logvar_max",
        "seed",
        "held_out_fold",
        "training_fold",
        "fold_sha256",
        "excluded_dataset_names",
        "dataset_counts_after_filter",
    )
    mismatches = []
    for field in contract_fields:
        expected_value = _json_ready(expected_manifest[field])
        if field not in recorded_manifest:
            mismatches.append(f"{field}: missing, expected {expected_value!r}")
            continue
        recorded_value = recorded_manifest[field]
        if recorded_value != expected_value:
            mismatches.append(
                f"{field}: recorded {recorded_value!r}, expected {expected_value!r}"
            )
    if mismatches:
        raise ValueError(
            "Resume checkpoint contract mismatch; refusing to reinterpret old or "
            "incompatible weights:\n- "
            + "\n- ".join(mismatches)
        )


def _print_dataset_counts(title: str, counts: dict[str, int]) -> None:
    """Print one stable dataset-of-origin count table."""

    print(title)
    print("dataset_of_origin\tnum_samples")
    for name, count in counts.items():
        print(f"{name}\t{count}")


def _package_version(distribution: str) -> str | None:
    """Read one installed package version without importing optional modules."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _fold_specifications(filtered_folds, held_out_folds: Sequence[int]):
    """Yield train/evaluation frames for each requested held-out fold."""

    frames = {1: filtered_folds.fold1, 2: filtered_folds.fold2}
    for held_out_fold in held_out_folds:
        training_fold = 2 if held_out_fold == 1 else 1
        yield (
            held_out_fold,
            training_fold,
            frames[training_fold],
            frames[held_out_fold],
        )


def _training_arguments(
    args: argparse.Namespace,
    fold_output_dir: Path,
    *,
    bf16: bool,
    fp16: bool,
) -> TrainingArguments:
    """Build version-stable Trainer arguments for one held-out fold."""

    training_kwargs = {
        "output_dir": str(fold_output_dir / "checkpoints"),
        "do_train": True,
        "do_eval": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "data_seed": args.seed,
        "bf16": bf16,
        "fp16": fp16,
        "use_cpu": args.use_cpu,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "mse_mean",
        "greater_is_better": False,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to or "none",
        "run_name": f"{args.model}-heldout-fold-{fold_output_dir.name}",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": (
            torch.cuda.is_available() and not args.use_cpu
        ),
    }
    transformers_version = _package_version("transformers")
    try:
        transformers_major = int(str(transformers_version).split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise RuntimeError(
            f"Cannot parse installed transformers version: {transformers_version!r}"
        ) from exc
    if transformers_major >= 5:
        training_kwargs["warmup_steps"] = args.warmup_ratio
    else:
        training_kwargs["warmup_ratio"] = args.warmup_ratio

    return TrainingArguments(**training_kwargs)


def run(args: argparse.Namespace) -> Path | None:
    """Execute filtering, two-fold training, prediction, and dynamic reporting."""

    _validate_args(args)
    loss_name = _loss_name(args.loss)
    filtered = load_filtered_folds(
        args.data_dir,
        exclude_dataset=args.exclude_dataset,
        no_iemocap=args.no_iemocap,
        no_ieomcap=args.no_ieomcap,
    )
    filtered_counts = dataset_counts(filtered.folds)
    if args.list_datasets:
        _print_dataset_counts("Datasets after requested exclusions:", filtered_counts)
        return None
    if any(frame.empty for frame in filtered.folds.values()):
        raise ValueError("Dataset exclusions left at least one fold empty.")
    if args.dry_run:
        _, bf16, fp16 = _runtime_precision(args.use_cpu)
        dry_run_root = (
            Path(args.output_dir)
            if args.output_dir
            else Path("Preds") / "dry_run"
        )
        for held_out_fold in args.held_out_folds:
            _training_arguments(
                args,
                dry_run_root / f"heldout_fold{held_out_fold}",
                bf16=bf16,
                fp16=fp16,
            )
        _print_dataset_counts("Validated datasets:", filtered_counts)
        print(f"Excluded: {', '.join(filtered.excluded_names) or '<none>'}")
        print(
            "Dry run complete; Trainer arguments are compatible and no tokenizer "
            "or model was downloaded."
        )
        return None

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not args.resume_from_checkpoint
    ):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tokenizer = load_auto_tokenizer(
        args.model_id,
        revision=args.model_revision,
        use_fast=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "right"
    collator = VABatchCollator(tokenizer, pad_to_multiple_of=8)
    dtype, bf16, fp16 = _runtime_precision(args.use_cpu)

    run_manifest = {
        **vars(args),
        "architecture_manifest_version": ARCHITECTURE_MANIFEST_VERSION,
        "loss": loss_name,
        "output_dim": _output_dimension(loss_name),
        "dtype": dtype,
        "effective_output_dir": str(output_dir.resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and not args.use_cpu
            else None
        ),
        "fold_sha256": {
            filename: sha256_file(Path(args.data_dir) / filename)
            for filename in ("full_dataset_fold1.csv", "full_dataset_fold2.csv")
        },
        "excluded_dataset_names": filtered.excluded_names,
        "dataset_counts_after_filter": filtered_counts,
        "fold_protocol": "two-fold out-of-fold; train the opposite fold and predict held-out",
        "gaze_feature": "TRT" if args.gaze_fusion != "none" else None,
        "gaze_feature_index": 3 if args.gaze_fusion != "none" else None,
        "gaze_concat_order": (
            GAZE_PREFIX_ORDER
            if args.gaze_fusion == "prefix-concat"
            else None
        ),
        "pooling_position": (
            GAZE_PREFIX_POOLING
            if args.gaze_fusion == "prefix-concat"
            else "last_valid_text_token"
        ),
    }
    fold_predictions = []

    for held_out_fold, training_fold, train_frame, eval_frame in _fold_specifications(
        filtered,
        args.held_out_folds,
    ):
        fold_output = output_dir / f"heldout_fold{held_out_fold}"
        expected_fold_manifest = {
            **run_manifest,
            "held_out_fold": held_out_fold,
            "training_fold": training_fold,
            "training_rows": len(train_frame),
            "evaluation_rows": len(eval_frame),
        }
        _validate_resume_contract(args, fold_output, expected_fold_manifest)
        training_arguments = _training_arguments(
            args,
            fold_output,
            bf16=bf16,
            fp16=fp16,
        )
        fold_output.mkdir(parents=True, exist_ok=True)
        print(
            f"Training fold {training_fold} ({len(train_frame):,} rows); "
            f"evaluating held-out fold {held_out_fold} ({len(eval_frame):,} rows)."
        )
        train_dataset = TokenizedVADataset(
            train_frame,
            tokenizer,
            max_length=args.max_length,
        )
        eval_dataset = TokenizedVADataset(
            eval_frame,
            tokenizer,
            max_length=args.max_length,
        )
        model = build_qwen_va_model(
            tokenizer,
            model_id=args.model_id,
            model_revision=args.model_revision,
            gaze_fusion=args.gaze_fusion,
            et_repo_id=args.et_model_id,
            et_revision=args.et_revision,
            et_filename=args.et_filename,
            et_cache_size=args.et_cache_size,
            output_dim=_output_dimension(loss_name),
            dtype=dtype,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            attn_implementation=args.attn_implementation,
        )
        fold_manifest = {
            **expected_fold_manifest,
            **model.trainable_parameter_summary(),
        }
        with open(fold_output / "run_manifest.json", "w", encoding="utf-8") as output_file:
            json.dump(_json_ready(fold_manifest), output_file, indent=2, sort_keys=True)
            output_file.write("\n")

        trainer = VARegressionTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            processing_class=tokenizer,
            compute_metrics=partial(
                trainer_compute_metrics,
                logvar_min=args.hetero_logvar_min,
                logvar_max=args.hetero_logvar_max,
            ),
            loss_name=loss_name,
            hetero_mse_weight=args.hetero_mse_weight,
            hetero_ccc_weight=args.hetero_ccc_weight,
            hetero_logvar_min=args.hetero_logvar_min,
            hetero_logvar_max=args.hetero_logvar_max,
        )
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(str(fold_output / "final_model"))
        model.save_architecture_manifest(fold_output / "final_model")
        prediction_output = trainer.predict(eval_dataset)
        fold_frame = prediction_frame(
            eval_frame,
            prediction_output.label_ids,
            prediction_output.predictions,
            fold=held_out_fold,
        )
        fold_frame.to_csv(
            fold_output / "predictions.tsv",
            sep="\t",
            index=False,
        )
        with open(fold_output / "metrics.json", "w", encoding="utf-8") as output_file:
            json.dump(
                _json_ready(prediction_output.metrics),
                output_file,
                indent=2,
                sort_keys=True,
            )
            output_file.write("\n")
        fold_predictions.append(fold_frame)

        del trainer, model, train_dataset, eval_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_oof_reports(
        fold_predictions,
        output_dir,
        run_parameters=_json_ready(run_manifest),
        logvar_min=args.hetero_logvar_min,
        logvar_max=args.hetero_logvar_max,
    )
    print(f"Completed. OOF reports: {output_dir.resolve()}")
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = _build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

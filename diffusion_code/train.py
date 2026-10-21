"""Train a separate conditional DDPM experiment using the established two-fold protocol."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "va_model_code"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch
import pandas as pd
from transformers import set_seed
import train_model as baseline
from va_model_code.decoder_va.dataset import TokenizedVADataset, VABatchCollator, load_auto_tokenizer
from va_model_code.decoder_va.downloads import sha256_file
from va_model_code.decoder_va.evaluation import prediction_frame, trainer_compute_metrics, write_oof_reports
from va_model_code.decoder_va.filters import (
    apply_training_text_overlap_policy, dataset_counts, load_filtered_folds,
    training_text_overlap_contract,
)
from va_model_code.decoder_va.model import (
    DEFAULT_DECODER_MODEL_ID, DEFAULT_DECODER_REVISION, DEFAULT_ET2_REVISION,
    build_qwen_va_model, validate_saved_trainable_precision,
)
from va_model_code.decoder_va.paths import resolve_run_directory
from va_model_code.decoder_va.training_kernels import training_kernel_report
from diffusion_code.config import DiffusionConfig, positive_finite
from diffusion_code.model import DiffusionVA
from diffusion_code.trainer import DiffusionTrainer


def build_parser():
    """Expose explicit DDPM choices alongside familiar baseline data and optimizer options."""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(ROOT / "va_model_code/data/paper7_seed42"))
    p.add_argument("--run-name")
    p.add_argument("--model-id", default=DEFAULT_DECODER_MODEL_ID)
    p.add_argument("--model-revision", default=DEFAULT_DECODER_REVISION)
    p.add_argument("--finetuning-mode", choices=("full", "lora"), default="full")
    p.add_argument("--precision", choices=baseline.PRECISION_MODES, default="bf16")
    p.add_argument("--gaze-fusion", choices=("none", "prefix-concat"), default="prefix-concat")
    p.add_argument("--conditioning", choices=("pooled", "token-trt"), default="token-trt")
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--diffusion-width", type=int, default=128)
    p.add_argument("--diffusion-blocks", type=int, default=3)
    p.add_argument("--diffusion-samples", type=int, default=16)
    p.add_argument("--sampling-seed", type=int, default=2026)
    p.add_argument("--head-learning-rate", type=float, default=1e-4)
    p.add_argument("--et-model-id", default="skboy/et_prediction_2")
    p.add_argument("--et-revision", default=DEFAULT_ET2_REVISION)
    p.add_argument("--et-filename", default="et_predictor2_seed123.safetensors")
    p.add_argument("--et-cache-size", type=int, default=70000)
    p.add_argument("--exclude-dataset", action="append", default=[])
    p.add_argument("--no-iemocap", action="store_true")
    p.add_argument("--sentence-only", action="store_true")
    p.add_argument("--training-text-overlap", dest="training_text_overlap_policy",
                   choices=("exclude", "keep"), default="keep")
    p.add_argument("--held-out-folds", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    for flag, default in (
        ("max-length", 200), ("train-batch-size", 8), ("eval-batch-size", 8),
        ("gradient-accumulation-steps", 2), ("logging-steps", 200),
        ("save-total-limit", 1), ("seed", 42), ("max-steps", -1),
        ("lora-rank", 16), ("lora-alpha", 32),
    ):
        p.add_argument("--" + flag, type=int, default=default)
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attn-implementation")
    p.add_argument("--resume-from-checkpoint")
    p.add_argument("--report-to", action="append", default=[])
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-datasets", action="store_true")
    p.set_defaults(
        model="qwen3.5-0.8b", loss="mse", gaze_features=["TRT"],
        no_ieomcap=False, gaze_redistribution="none", gaze_relation="none", readout="last-token",
        redistribution_learning_rate=None, redistribution_weight_decay=None,
        gaze_relation_learning_rate=None, gaze_relation_weight_decay=None,
    )
    return p


def write_json(path, payload):
    """Write strict, human-readable JSON with shared NumPy/Torch normalization."""

    Path(path).write_text(json.dumps(baseline._json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_hashes():
    """Record exact experiment and shared-encoder code, including uncommitted local changes."""

    paths = list((ROOT / "diffusion_code").glob("*.py"))
    paths += list((ROOT / "va_model_code/decoder_va").glob("*.py"))
    paths += [ROOT / "va_model_code/train_model.py", ROOT / "requirements.txt"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def validate_resume(path, fold_output, contract):
    """Require checkpoint ownership, optimizer state and identical experimental semantics."""

    path = Path(path).resolve()
    if path.parent != (fold_output / "checkpoints").resolve() or not path.name.startswith("checkpoint-"):
        raise ValueError("Resume must use a checkpoint belonging to this run and held-out fold.")
    for name in (
        "model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth",
        "trainer_state.json", "diffusion_architecture.json", "diffusion_training_contract.json",
    ):
        if not (path / name).is_file() or (path / name).is_symlink():
            raise ValueError(f"Resume requires complete regular checkpoint file: {name}")
    previous = json.loads((path / "diffusion_training_contract.json").read_text())
    if previous != contract:
        raise ValueError("Resume contract changed; start a new run instead of reinterpreting weights.")


def run(args, *, mean_reference=None):
    """Run matched fold training and label-free DDPM evaluation; certify completion last."""

    baseline._validate_args(args)
    if mean_reference is not None and (args.gradient_checkpointing or args.resume_from_checkpoint):
        raise ValueError("Frozen-mean runs use no encoder checkpointing and currently require a fresh run.")
    settings = DiffusionConfig(
        timesteps=args.diffusion_steps, hidden_dim=args.diffusion_width,
        blocks=args.diffusion_blocks, conditioning=args.conditioning,
        samples=args.diffusion_samples, sampling_seed=args.sampling_seed,
    )
    positive_finite(args.head_learning_rate, "head-learning-rate")
    if args.conditioning == "token-trt" and args.gaze_fusion != "prefix-concat":
        raise ValueError("token-trt requires --gaze-fusion prefix-concat.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
        not args.use_cpu and torch.cuda.device_count() > 1
    ):
        raise ValueError("This DDPM experiment currently supports one training process/GPU.")
    if args.logging_steps <= 0 or args.save_total_limit < 1:
        raise ValueError("logging-steps and save-total-limit must be positive.")
    for name in ("epochs", "learning_rate"):
        positive_finite(getattr(args, name), name)
    if not torch.isfinite(torch.tensor(args.weight_decay)) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and nonnegative.")
    filtered = load_filtered_folds(
        args.data_dir, exclude_dataset=args.exclude_dataset,
        no_iemocap=args.no_iemocap, sentence_only=args.sentence_only,
    )
    if args.list_datasets:
        baseline._print_dataset_counts("Selected datasets:", dataset_counts(filtered.folds))
        return None
    folds = []
    for held, training, train, evaluation in baseline._fold_specifications(filtered, args.held_out_folds):
        train, audit = apply_training_text_overlap_policy(
            train, evaluation, policy=args.training_text_overlap_policy,
        )
        if train.empty or evaluation.empty:
            raise ValueError("Every requested fold needs nonempty training and evaluation sets.")
        for frame in (train, evaluation):
            labels = torch.tensor(frame[["valence", "arousal"]].to_numpy())
            if not bool((torch.isfinite(labels) & (labels >= 0) & (labels <= 1)).all()):
                raise ValueError("All VA labels must be finite and normalized to [0, 1].")
        folds.append((held, training, train, evaluation, audit))
    dtype, bf16, fp16 = baseline._runtime_precision(args.use_cpu, args.precision)
    if args.run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.run_name = f"ddpm_{args.gaze_fusion}_{args.conditioning}_{args.precision}_seed{args.seed}_{stamp}"
    output = resolve_run_directory(args.run_name)
    if args.dry_run:
        for held, _, train, evaluation, _ in folds:
            baseline._training_arguments(args, output / f"heldout_fold{held}", bf16=bf16, fp16=fp16)
            print(f"Held-out fold {held}: train={len(train):,}; evaluation={len(evaluation):,}.")
        print(json.dumps(settings.to_dict(), indent=2))
        print(f"Dry run: no models downloaded. Planned results: {output}")
        return None
    if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError(f"Run directory already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_source = (mean_reference.records[args.held_out_folds[0]]["final_model"]
                        if mean_reference is not None else args.model_id)
    tokenizer = load_auto_tokenizer(tokenizer_source, revision=args.model_revision, use_fast=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    collator = VABatchCollator(tokenizer, pad_to_multiple_of=8)
    public_args = {k: v for k, v in vars(args).items() if k not in ("resume_from_checkpoint", "loss")}
    manifest = baseline._json_ready({
        "schema_version": 1, "model_family": "conditional_ddpm_va", "arguments": public_args,
        "diffusion": settings.to_dict(), "loss": "epsilon_mse",
        "evaluation_loss": "generated_va_mse", "best_checkpoint_metric": "eval_mse_mean",
        "source_sha256": source_hashes(),
        "fold_sha256": {name: sha256_file(Path(args.data_dir) / name) for name in (
            "full_dataset_fold1.csv", "full_dataset_fold2.csv",
        )},
        "training_text_overlap_contract": training_text_overlap_contract(args.training_text_overlap_policy),
        "dataset_counts": dataset_counts(filtered.folds), "excluded_names": filtered.excluded_names,
        "compute_dtype": dtype, "head_dtype": "float32",
        "trainable_parameter_dtype": "float32",
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": baseline._package_version("transformers"),
        "training_kernels": training_kernel_report(),
    })
    if mean_reference is not None:
        from diffusion_code.mean_guided import FAMILY, MEAN_CONTRACT
        manifest.update({"model_family": FAMILY, "mean_guidance": MEAN_CONTRACT,
                         "diffusion_role": "base_schedule_only_target_is_defined_by_mean_guidance",
                         "mean_reference": mean_reference.provenance()})
    requested_folds = list(args.held_out_folds)
    if args.resume_from_checkpoint:
        original = json.loads((output / "training_parameters.json").read_text())
        requested_folds = original["arguments"]["held_out_folds"]
        if not set(args.held_out_folds).issubset(requested_folds):
            raise ValueError("Resume fold was not requested in the original run.")
        manifest["arguments"]["held_out_folds"] = requested_folds
        if manifest != original:
            raise ValueError("Run configuration/source changed; resume requires the original contract.")
    print("Training loss: epsilon MSE. Evaluation loss / best checkpoint: generated-VA MSE.")
    print("Qwen runs once per batch; all DDPM steps reuse its features and projected keys/values.")
    print(f"Diffusion: {settings.to_dict()}")
    if mean_reference is not None:
        print("Mean guidance: frozen Raw TRT encoder/head; residual epsilon MSE; only denoiser trains.")
    dispatch = manifest["training_kernels"]["dispatch"]["torch_chunk_gated_delta_rule"]
    print(f"Qwen linear-attention implementation: {dispatch['selected'] or 'unresolved'}")
    if not args.use_cpu and not dispatch["is_fla"]:
        print("Warning: FLA is inactive; run setup_environment.py --current-env --kernels-only.")
    for held, training, train, evaluation, audit in folds:
        fold_output = output / f"heldout_fold{held}"
        contract = {**manifest, "held_out_fold": held, "training_fold": training,
                    "fold_seed": args.seed + held - 1, "training_rows": len(train),
                    "evaluation_rows": len(evaluation), "training_text_overlap": audit}
        if args.resume_from_checkpoint:
            validate_resume(args.resume_from_checkpoint, fold_output, contract)
        (output / "COMPLETED.json").unlink(missing_ok=True)
        fold_output.mkdir(parents=True, exist_ok=True)
        (fold_output / "COMPLETED.json").unlink(missing_ok=True)
        write_json(output / "training_parameters.json", manifest)
        write_json(fold_output / "run_manifest.json", contract)
        set_seed(contract["fold_seed"])
        baseline._reset_cuda_peak_memory(args.use_cpu)
        if mean_reference is None:
            encoder = build_qwen_va_model(
                tokenizer, model_id=args.model_id, model_revision=args.model_revision,
                gaze_fusion=args.gaze_fusion, gaze_feature_indices=(3,),
                et_repo_id=args.et_model_id, et_revision=args.et_revision,
                et_filename=args.et_filename, et_cache_size=args.et_cache_size,
                dtype=dtype, finetuning_mode=args.finetuning_mode,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                attn_implementation=args.attn_implementation,
            )
            model = DiffusionVA(encoder, settings)
        else:
            from diffusion_code.mean_guided import MeanGuidedDiffusionVA
            encoder, _ = mean_reference.load(held, dtype=dtype, et_cache_size=args.et_cache_size, tokenizer=tokenizer)
            model = MeanGuidedDiffusionVA(encoder, settings, reference=mean_reference.records[held])
        counts = {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "denoiser": sum(p.numel() for p in model.denoiser.parameters()),
        }
        print(f"Parameters: {counts}")
        write_json(fold_output / "parameter_counts.json", counts)
        if args.resume_from_checkpoint:
            validate_saved_trainable_precision(model, Path(args.resume_from_checkpoint) / "model.safetensors")
        train_data = TokenizedVADataset(train, tokenizer, max_length=args.max_length)
        eval_data = TokenizedVADataset(evaluation, tokenizer, max_length=args.max_length)
        print(f"Training fold {training} ({len(train):,} rows); evaluating held-out fold {held} ({len(evaluation):,} rows).")
        print(f"Training-text-overlap: {args.training_text_overlap_policy}; audit: {audit}")
        trainer = DiffusionTrainer(
            model=model,
            args=baseline._training_arguments(
                args, fold_output, bf16=bf16, fp16=fp16, fold_seed=contract["fold_seed"],
            ),
            train_dataset=train_data, eval_dataset=eval_data, data_collator=collator,
            processing_class=tokenizer, compute_metrics=trainer_compute_metrics,
            head_learning_rate=args.head_learning_rate, run_contract=contract,
        )
        if mean_reference is not None:
            from diffusion_code.mean_reports import evaluate_frozen_mean
            evaluate_frozen_mean(trainer, eval_data, evaluation, held, fold_output, mean_reference)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(str(fold_output / "final_model"))
        trainer.state.save_to_json(str(fold_output / "trainer_state.json"))
        predictions = trainer.predict(eval_data)
        frame = prediction_frame(evaluation, predictions.label_ids, predictions.predictions, fold=held)
        frame.to_csv(fold_output / "predictions.tsv", sep="\t", index=False)
        write_json(fold_output / "metrics.json", predictions.metrics)
        write_json(fold_output / "gpu_memory.json", baseline._cuda_memory_snapshot(args.use_cpu))
        write_json(fold_output / "COMPLETED.json", {
            "global_step": trainer.state.global_step,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "predictions_sha256": sha256_file(fold_output / "predictions.tsv"),
            **({"frozen_mean_predictions_sha256": sha256_file(fold_output / "frozen_mean/predictions.tsv")}
               if mean_reference is not None else {}),
        })
        del trainer, model, encoder, train_data, eval_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    missing = [held for held in requested_folds if not (output / f"heldout_fold{held}/COMPLETED.json").is_file()]
    if missing:
        print(f"Requested fold finished; this run still needs held-out folds {missing}. No complete-run ZIP is available.")
        return output
    frames = [
        pd.read_csv(output / f"heldout_fold{held}/predictions.tsv", sep="\t", keep_default_na=False)
        for held in requested_folds
    ]
    write_oof_reports(frames, output, run_parameters=manifest)
    if mean_reference is not None:
        from diffusion_code.mean_reports import write_mean_comparison
        write_mean_comparison(output, requested_folds)
    write_json(output / "COMPLETED.json", {
        "held_out_folds": requested_folds,
        "oof_sha256": sha256_file(output / "oof_predictions.tsv"),
        **({"frozen_mean_oof_sha256": sha256_file(output / "frozen_mean_oof/oof_predictions.tsv"),
            "mean_comparison_sha256": sha256_file(output / "mean_comparison.tsv")}
           if mean_reference is not None else {}),
    })
    print(f"Completed. DDPM OOF reports: {output}")
    return output


if __name__ == "__main__":
    run(build_parser().parse_args())

"""Explicit single-GPU fits; only validation may select a checkpoint."""

import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from trt_pipeline.core.dataset import TokenizedVADataset, VABatchCollator
from trt_pipeline.core.evaluation import calculate_va_metrics
from trt_pipeline.core.training_kernels import require_fla_dispatch, training_kernel_report

from .config import ARMS
from .model import build_model, precision_proof, restore_weights, save_model
from .protocol import digest, load_lock, read_partition, utc_now, write_json


def seed_everything(seed):
    """Reset initialization, dropout and augmentation seeds for each declared fit."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_check():
    """Fail before allocation on unsupported versions, distributed launch or slow kernels."""

    from environment_runtime import require_active_conda

    conda = require_active_conda()
    for package, expected in (("torch", "2.12.1"), ("transformers", "5.14.1")):
        if version(package).split("+")[0] != expected:
            raise RuntimeError(f"Requires {package}=={expected}; install root requirements.txt.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use one process and one CUDA GPU for this protocol.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required.")
    report = training_kernel_report()
    report["conda"] = conda
    require_fla_dispatch(report)
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "safetensors", "tokenizers"):
        report["packages"][package] = version(package)
    report["python"] = sys.version
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    report["device"] = torch.cuda.get_device_name(0)
    report["cuda"] = torch.version.cuda
    report["tf32_matmul"] = False
    report["optimizer"] = "torch.optim.AdamW_foreach_false_fused_false_no_master_copy"
    return report


def epoch_order(rows, seed, epoch):
    """Keep clean-example batch order identical across arms regardless of noise RNG use."""

    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(rows, generator=generator).tolist()


def data_loader(frame, tokenizer, settings, *, order=None):
    """Use sequential evaluation or an explicit shared training permutation."""

    dataset = TokenizedVADataset(frame, tokenizer, max_length=settings.max_length)
    return DataLoader(dataset, batch_size=settings.batch_size if order is not None else settings.eval_batch_size,
                      sampler=order, shuffle=False, collate_fn=VABatchCollator(tokenizer),
                      num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)


def move_batch(batch, device):
    """Send tensors to the selected device without changing label precision."""

    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def loss_for_arm(outputs, gold, arm):
    """Average MSE over examples, axes and both companion views in FP32."""

    if arm not in ARMS:
        raise ValueError("Unknown training arm.")
    predictions = outputs["view_predictions"] if arm == "companion" else outputs["logits"]
    labels = torch.cat((gold, gold), dim=0) if arm == "companion" else gold
    if predictions.shape != labels.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("Training outputs do not match the declared objective.")
    return F.mse_loss(predictions.float(), labels.float(), reduction="mean")


def optimizer_for(model, settings):
    """Use identical AdamW groups, plus a declared zero-decay group for Gaussian widths."""

    decay, no_decay, widths = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("redistributor."):
            widths.append(parameter)
        elif parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.casefold():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "lr": settings.learning_rate, "weight_decay": settings.weight_decay},
        {"params": no_decay, "lr": settings.learning_rate, "weight_decay": 0.0},
    ]
    if widths:
        if len(widths) != 2:
            raise ValueError("Redistribution requires exactly two global learned widths.")
        groups.append({"params": widths, "lr": settings.redistribution_learning_rate, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)


def learning_rate_factor(step, *, warmup, total):
    """Match linear warmup and decay with an update-count budget shared by all fits."""

    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


class ValidationSelection:
    """Select the first minimum validation mean MSE without a test input interface."""

    def __init__(self):
        self.best = math.inf
        self.epoch = None
        self.step = None

    def observe(self, *, partition, metrics, epoch, step):
        """Fail closed on non-validation scores or invalid checkpoint metrics."""

        if partition != "val":
            raise ValueError("Only validation scores can select checkpoints.")
        score = float(metrics["mse_mean"])
        if not math.isfinite(score):
            raise ValueError("Validation mean MSE is not finite.")
        if score < self.best:
            self.best, self.epoch, self.step = score, epoch, step
            return True
        return False


def finite_metrics(metrics):
    """Represent mathematically undefined correlations as JSON null without hiding MSE."""

    return {key: (float(value) if np.isfinite(value) else None) for key, value in metrics.items()}


@torch.inference_mode()
def predict(model, loader, device):
    """Produce exactly one clean, sequential forward pass over this loader."""

    model.eval()
    predictions = []
    for batch in loader:
        inputs = move_batch(batch, device)
        inputs.pop("labels", None)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**inputs)["logits"]
        predictions.append(logits.float().cpu().numpy())
    result = np.concatenate(predictions)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError("Invalid prediction array.")
    return result


def optimizer_proof(model, optimizer):
    """Check every live AdamW moment against native parameter storage after an update."""

    state = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        slot = optimizer.state.get(parameter)
        if not slot or not {"step", "exp_avg", "exp_avg_sq"}.issubset(slot):
            raise ValueError(f"No live AdamW state for {name}")
        for moment in ("exp_avg", "exp_avg_sq"):
            if slot[moment].dtype != parameter.dtype or slot[moment].shape != parameter.shape:
                raise ValueError(f"AdamW moment storage differs from parameter: {name}")
            if not torch.isfinite(slot[moment]).all():
                raise ValueError(f"Nonfinite AdamW moment: {name}")
        state[name] = {"parameter_dtype": str(parameter.dtype), "moment_dtype": str(slot["exp_avg"].dtype),
                       "step": int(slot["step"].item())}
    return {"status": "passed", "scope": "all live trainable parameter moments; no CPU reload claimed",
            "parameters": state, "precision": precision_proof(model)}


def train(run, arm):
    """Fit train only, select on validation each epoch, and save a strict reload proof."""

    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    run = Path(run).resolve()
    lock, settings = load_lock(run)
    if (run / "selection_lock.json").exists() or (run / "evaluation_started.json").exists():
        raise RuntimeError("Selection is sealed; no further training is permitted in this run.")
    output = run / arm
    if output.exists():
        raise FileExistsError(f"Arm already started; no implicit overwrite or restart: {output}")
    runtime = runtime_check()
    output.mkdir()
    write_json(output / "started.json", {"utc": utc_now(), "arm": arm,
               "protocol_sha256": lock["protocol_sha256"]}, exclusive=True)
    started = time.perf_counter()
    try:
        result = _fit(run, output, arm, lock, settings, runtime, started)
    except BaseException as error:
        write_json(output / "failure.json", {"utc": utc_now(), "error": type(error).__name__,
                   "message": str(error), "elapsed_seconds": time.perf_counter() - started})
        raise
    return result


def _fit(run, output, arm, lock, settings, runtime, started, *, device=None):
    """Keep test rows and labels outside the complete fitting and selection path."""

    train_frame = read_partition(run / "splits/train.tsv")
    val_frame = read_partition(run / "splits/val.tsv")
    seed_everything(settings.seed)
    device = torch.device("cuda:0") if device is None else device
    model, tokenizer = build_model(settings, arm)
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    initial_precision = precision_proof(model)
    optimizer = optimizer_for(model, settings)
    total = lock["total_updates_per_fit"]
    warmup = math.ceil(total * settings.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, warmup=warmup, total=total))
    val_loader = data_loader(val_frame, tokenizer, settings)
    selection = ValidationSelection()
    history, actual_step = [], 0
    best_probe = None
    best_manifest = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    write_json(output / "environment.json", runtime)
    write_json(output / "initial_precision.json", initial_precision)
    for epoch in range(1, settings.epochs + 1):
        loader = data_loader(train_frame, tokenizer, settings,
                             order=epoch_order(len(train_frame), settings.seed, epoch))
        model.train()
        running_loss, seen = 0.0, 0
        epoch_started = time.perf_counter()
        for batch in loader:
            inputs = move_batch(batch, device)
            gold = inputs.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(**inputs)
                loss = loss_for_arm(outputs, gold, arm)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; stopping without restart.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm,
                                                  error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            actual_step += 1
            seen += len(gold)
            running_loss += float(loss.detach()) * len(gold)
            if actual_step == 1:
                write_json(output / "optimizer_audit.json", optimizer_proof(model, optimizer))
            if actual_step % 50 == 0 or actual_step == total:
                progress = {"utc": utc_now(), "arm": arm, "epoch": epoch,
                            "actual_optimizer_updates": actual_step, "total_optimizer_updates": total,
                            "elapsed_seconds": time.perf_counter() - started,
                            "mean_training_loss": running_loss / seen,
                            "last_grad_norm": float(norm), "learning_rates": scheduler.get_last_lr()}
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        if seen != len(train_frame) or actual_step != epoch * lock["updates_per_epoch"]:
            raise AssertionError("Actual examples or optimizer updates disagree with the locked budget.")
        train_seconds = time.perf_counter() - epoch_started
        val_started = time.perf_counter()
        predictions = predict(model, val_loader, device)
        metrics = calculate_va_metrics(val_frame[["valence", "arousal"]].to_numpy(float), predictions)
        improved = selection.observe(partition="val", metrics=metrics, epoch=epoch, step=actual_step)
        event = {"stage": "validation_checkpoint_selection", "arm": arm, "partition": "val",
                 "epoch": epoch, "actual_optimizer_updates": actual_step,
                 "metrics": finite_metrics(metrics), "selected": improved,
                 "train_seconds": train_seconds, "validation_seconds": time.perf_counter() - val_started,
                 "train_loss": running_loss / seen, "utc": utc_now()}
        if improved:
            best_manifest = save_model(model, output / "model", settings, lock["protocol_sha256"])
            tokenizer.save_pretrained(output / "model")
            best_probe = predictions[:settings.eval_batch_size].copy()
            event["selected_weights_sha256"] = best_manifest["weights_sha256"]
        history.append(event)
        write_json(output / "validation_history.json", history)
        print(json.dumps(event, allow_nan=False), flush=True)
    if actual_step != total or best_manifest is None:
        raise AssertionError("Incomplete declared training budget or missing selected checkpoint.")
    write_json(output / "final_optimizer_audit.json", optimizer_proof(model, optimizer))
    del optimizer, scheduler, outputs, loss, gold, inputs, batch, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    restore_weights(model, output / "model", expected_sha256=best_manifest["weights_sha256"])
    probe_loader = data_loader(val_frame.iloc[:settings.eval_batch_size], tokenizer, settings)
    reloaded_probe = predict(model, probe_loader, device)
    np.testing.assert_allclose(reloaded_probe, best_probe, rtol=1e-4, atol=2e-5)
    write_json(output / "strict_reload.json", {
        "status": "passed", "weights_sha256": best_manifest["weights_sha256"],
        "probe_partition": "val", "rows": len(best_probe),
        "max_abs_prediction_difference": float(np.max(np.abs(reloaded_probe - best_probe))),
        "scope": "full checkpoint keys/shapes/dtypes and selected validation probe",
    })
    complete = {
        "status": "completed", "arm": arm, "utc": utc_now(),
        "protocol_sha256": lock["protocol_sha256"], "actual_optimizer_updates": actual_step,
        "epochs": settings.epochs, "selected_epoch": selection.epoch, "selected_step": selection.step,
        "selection_partition": "val", "selected_validation_mse_mean": selection.best,
        "selected_weights_sha256": best_manifest["weights_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
        "training_device": str(device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "test_evaluations_during_training": 0,
        "validation_history_sha256": digest(output / "validation_history.json"),
        "strict_reload_sha256": digest(output / "strict_reload.json"),
    }
    write_json(output / "train_complete.json", complete, exclusive=True)
    print(json.dumps(complete), flush=True)
    del model, val_loader, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return complete

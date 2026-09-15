"""Opt-in, single-process sigma diagnostics with reversible evaluation probes."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from transformers import TrainerCallback

from .redistribution import GazeRedistributor, redistribution_contract


def _scalar(value):
    """Serialize scalar tensors while retaining nonfinite diagnostic evidence."""

    if value is None:
        return None
    result = float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)
    return result if math.isfinite(result) else str(result)


def _summary(values):
    """Summarize finite observations and report missing/nonfinite counts."""

    flat = values.detach().float().cpu().flatten()
    finite = flat[torch.isfinite(flat)]
    result = {"count": flat.numel(), "nonfinite": flat.numel() - finite.numel()}
    if finite.numel():
        quantiles = torch.quantile(finite, torch.tensor([0., .05, .5, .95, 1.]))
        result.update(zip(("min", "p05", "median", "p95", "max"), quantiles.tolist()))
        result["negative_fraction"] = float((finite < 0).float().mean())
    return result


def _relative_change(current, reference):
    """Report absolute RMS and relative L2 differences, including zero references."""

    current, reference = current.float(), reference.float()
    difference = current - reference
    denominator = reference.norm().item()
    return {
        "rms": _scalar(difference.square().mean().sqrt()) if difference.numel() else None,
        "relative_l2": _scalar(difference.norm() / denominator) if denominator > 0 else None,
    }


class SigmaDiagnostics(TrainerCallback):
    """Record update mechanics and fixed-training-batch sensitivity separately."""

    def __init__(self, trainer, output_dir, *, every_steps, batch_size):
        self.trainer = trainer
        self.output_dir = Path(output_dir)
        self.every_steps = int(every_steps)
        self.batch_size = min(int(batch_size), len(trainer.train_dataset))
        self.pending = None
        if trainer.args.world_size != 1 or trainer.args.fp16:
            raise ValueError("Sigma diagnostics require single-process FP32/BF16 training.")
        if self.every_steps <= 0 or self.batch_size <= 0:
            raise ValueError("Sigma diagnostics require positive step interval and batch size.")
        if trainer.model.gaze_fusion != "prefix-concat" or tuple(trainer.model.gaze_provider.feature_indices) != (3,):
            raise ValueError("Sigma diagnostics require TRT-only prefix-concat.")

    def _write(self, filename, record):
        """Append one immediately readable record without opening persistent handles."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")

    def _kernel(self):
        """Find the original model kernel after distributed setup is ruled out."""

        return getattr(self.trainer.model.gaze_redistributor, "kernel", None)

    def _values(self, gradients=False):
        """Read trainable/frozen kernel values without changing optimizer state."""

        kernel = self._kernel()
        if kernel is None:
            return {}
        return {
            name: _scalar(parameter.grad if gradients else parameter)
            for name, parameter in kernel.named_parameters()
        }

    def on_step_begin(self, args, state, control, **kwargs):
        """Select the first and each Nth optimizer update, including accumulation."""

        selected = state.global_step == 0 or (state.global_step + 1) % self.every_steps == 0
        self.pending = {
            "event": "optimizer_update", "step": state.global_step + 1,
            "log_sigma_before": self._values(), "microbatches": 0,
        } if selected else None

    def after_backward(self):
        """Capture accumulated gradients after backward and before global clipping."""

        if self.pending is not None:
            self.pending["gradient_before_clipping"] = self._values(gradients=True)
            self.pending["microbatches"] += 1

    def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        """Capture Trainer-clipped gradients and pre-update Adam moments."""

        if self.pending is None:
            return
        self.pending["gradient_after_clipping"] = self._values(gradients=True)
        kernel = self._kernel()
        moments = {}
        if kernel is not None and optimizer is not None:
            for name, parameter in kernel.named_parameters():
                saved = optimizer.state.get(parameter, {})
                group = next((g for g in optimizer.param_groups if any(p is parameter for p in g['params'])), None)
                moments[name] = {
                    "lr": group["lr"] if group is not None else None,
                    **{key: _scalar(saved.get(key)) for key in ("step", "exp_avg", "exp_avg_sq")},
                }
        self.pending["optimizer_before_update"] = moments

    def on_step_end(self, args, state, control, **kwargs):
        """Measure actual parameter displacement after the completed optimizer step."""

        if self.pending is None:
            return
        after = self._values()
        self.pending["log_sigma_after"] = after
        self.pending["delta_log_sigma"] = {
            key: value - self.pending["log_sigma_before"][key]
            if isinstance(value, float) and isinstance(self.pending["log_sigma_before"][key], float) else None
            for key, value in after.items()
        }
        self.pending["epoch"] = state.epoch
        self._write("sigma_updates.jsonl", self.pending)
        self.pending = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Probe the initial or resumed model without consuming training RNG."""

        self.probe(state, "train_begin")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Probe before epoch evaluation and checkpoint selection."""

        self.probe(state, "epoch_end")

    def on_train_end(self, args, state, control, **kwargs):
        """Probe the selected model after Trainer restores its best checkpoint."""

        self.probe(state, "train_end_selected_model")

    def probe(self, state, event):
        """Temporarily change only redistribution, restoring flags, modules and RNG."""

        model = self.trainer.model
        device = next(model.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        original = model.gaze_redistributor
        parameter_flags = [(p, p.requires_grad) for p in model.parameters()]
        module_flags = [(m, m.training) for m in model.modules()]
        original_kernel = getattr(original, "kernel", None)
        original_logs = tuple(_scalar(p) for p in original_kernel.parameters()) if original_kernel is not None else None
        floor = original_kernel.min_sigma if original_kernel is not None else 1e-6
        if original_logs is not None and not all(isinstance(v, float) for v in original_logs):
            self._write("sigma_probes.jsonl", {"event": event, "step": state.global_step, "error": "nonfinite sigma"})
            return
        handles = []
        try:
            with torch.random.fork_rng(devices=devices):
                batch = self.trainer.data_collator([self.trainer.train_dataset[i] for i in range(self.batch_size)])
                batch = self.trainer._prepare_inputs(batch)
                labels = batch.pop("labels").float()
                model.eval()
                for parameter, _ in parameter_flags:
                    parameter.requires_grad_(False)
                raw, mask = model.gaze_provider.compute(batch["input_ids"], batch["attention_mask"])
                mask = mask.bool()
                gaps = [row.nonzero().flatten().diff() for row in mask]
                record = {
                    "event": event, "step": state.global_step, "epoch": state.epoch,
                    "best_checkpoint": state.best_model_checkpoint,
                    "scope": "first training examples; dropout disabled; local sensitivity, not validation metrics",
                    "example_indices_in_training_dataset": list(range(self.batch_size)),
                    "raw_trt": _summary(raw[:, :, 0][mask]),
                    "valid_gaze_counts": mask.sum(1).tolist(),
                    "qwen_token_gaps": _summary(torch.cat(gaps)),
                    "cases": {},
                }
                captured = {}

                def capture_input(module, inputs):
                    """Retain post-redistribution TRT without an autograd reference."""
                    captured["trt"] = inputs[0].detach().float().cpu()

                def capture_output(module, inputs, output):
                    """Retain projector output before conversion to backbone dtype."""
                    captured["projection"] = output.detach().float().cpu()

                handles = [model.gaze_projector.register_forward_pre_hook(capture_input), model.gaze_projector.register_forward_hook(capture_output)]
                configured = original_logs
                cases = {"configured": configured, "raw": None, "symmetric_1": (0., 0.), "left_05_right_2": (math.log(.5), math.log(2.)), "left_2_right_05": (math.log(2.), math.log(.5))}
                reference = None
                cpu_mask = mask.cpu()
                for name, log_widths in cases.items():
                    replacement = None
                    if log_widths is not None:
                        replacement = GazeRedistributor(redistribution_contract("asym-gaussian", min_sigma=floor), (3,)).to(device)
                        with torch.no_grad():
                            for parameter, value in zip(replacement.kernel.parameters(), log_widths):
                                parameter.fill_(value)
                    model.gaze_redistributor = replacement
                    with torch.enable_grad(), self.trainer.accelerator.autocast():
                        prediction = model(**batch).logits.float()
                        squared_error = (prediction - labels).square()
                        sample_gradients = []
                        if replacement is not None:
                            for row in range(prediction.shape[0]):
                                by_output = []
                                for dimension in range(2):
                                    gradients = torch.autograd.grad(squared_error[row, dimension], tuple(replacement.kernel.parameters()), retain_graph=True, allow_unused=True) if squared_error.requires_grad else (None, None)
                                    by_output.append([0. if g is None else _scalar(g) for g in gradients])
                                sample_gradients.append(by_output)
                        payload = {
                            "mse": _scalar(squared_error.mean()),
                            "mse_valence": _scalar(squared_error[:, 0].mean()),
                            "mse_arousal": _scalar(squared_error[:, 1].mean()),
                            "prediction": prediction.detach().cpu().tolist(),
                            "sigma": None if replacement is None else [_scalar(replacement.kernel.sigma_left), _scalar(replacement.kernel.sigma_right)],
                            "local_gradient_sample_output_side": sample_gradients,
                        }
                        if sample_gradients and all(isinstance(v, float) for row in sample_gradients for dim in row for v in dim):
                            gradients = torch.tensor(sample_gradients)
                            means, magnitudes = gradients.mean((0, 1)), gradients.abs().mean((0, 1))
                            payload["mean_gradient_left_right"] = means.tolist()
                            payload["gradient_coherence_left_right"] = [float(abs(m) / a) if a > 0 else None for m, a in zip(means, magnitudes)]
                    observed = {
                        "trt": captured["trt"][cpu_mask],
                        "projection": captured["projection"][cpu_mask],
                        "prediction": prediction.detach().cpu(),
                    }
                    if reference is None:
                        reference = observed
                    payload["change_from_configured"] = {key: _relative_change(value, reference[key]) for key, value in observed.items()}
                    record["cases"][name] = payload
                    del prediction, squared_error
                self._write("sigma_probes.jsonl", record)
        finally:
            for handle in handles:
                handle.remove()
            model.gaze_redistributor = original
            for parameter, flag in parameter_flags:
                parameter.requires_grad_(flag)
            for module, flag in module_flags:
                module.training = flag
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)


def attach_sigma_diagnostics(trainer, output_dir, *, every_steps=50, batch_size=2):
    """Attach diagnostics to a Trainer without changing its optimization settings."""

    observer = SigmaDiagnostics(trainer, output_dir, every_steps=every_steps, batch_size=batch_size)
    trainer.sigma_diagnostics = observer
    trainer.add_callback(observer)
    return observer

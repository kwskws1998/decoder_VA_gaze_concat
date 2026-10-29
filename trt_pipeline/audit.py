"""Passive historical-baseline optimizer and native-save audits."""
import collections,datetime,time
from pathlib import Path
import torch
from transformers import TrainerCallback
from trt_pipeline.historical import recipe
from trt_pipeline.io_utils import write_json

def audit_optimizer(optimizer, serialized=None):
    """Require finite moments in each parameter's native dtype, including CPU reload."""
    if serialized is None:
        groups = optimizer.param_groups
        state = optimizer.state
        pairs = [(parameter, state.get(parameter, {})) for group in groups for parameter in group["params"]]
    else:
        loaded = torch.load(serialized, map_location="cpu", weights_only=True)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        identifiers = [p for group in loaded["param_groups"] for p in group["params"]]
        if len(parameters) != len(identifiers):
            raise ValueError("Serialized optimizer parameter count changed.")
        pairs = [(p, loaded["state"].get(i, {})) for p, i in zip(parameters, identifiers)]
    counts = collections.Counter()
    steps = set()
    for parameter, values in pairs:
        for name, value in values.items():
            if not torch.is_tensor(value):
                continue
            if not torch.isfinite(value).all().item():
                raise FloatingPointError(f"Nonfinite optimizer state: {name}")
            counts[str(value.dtype)] += 1
            if name in ("exp_avg", "exp_avg_sq") and value.dtype != parameter.dtype:
                raise ValueError(f"Optimizer moment {name} differs from native parameter dtype.")
            if name == "step":
                steps.add(float(value.item()))
    if not steps or min(steps) <= 0:
        raise ValueError("Optimizer has not actually updated parameters.")
    return {"passed": True, "tensor_dtype_counts": dict(counts), "steps": sorted(steps)}


class IntegrityCallback(TrainerCallback):
    """Observe training without sampling RNG or changing its optimization policy."""

    def __init__(self, fold):
        self.fold = fold
        self.started = None
        self.events = []

    def location(self, args):
        """Locate this run's audit directory beside its checkpoint directory."""
        return Path(args.output_dir).parent / "integrity"

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Record actual native parameter dtypes before the first update."""
        self.started = time.monotonic()
        parameters = collections.Counter(str(p.dtype) for p in model.parameters())
        if parameters["torch.bfloat16"] == 0:
            raise ValueError("The historical backbone is not native BF16.")
        write_json(self.location(args) / "training_begin.json", {
            "parameter_dtype_counts": dict(parameters), "max_steps": state.max_steps,
            "precision_contract": recipe.PRECISION,
        })

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        """Abort on numerical implementation failure, never on validation performance."""
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"Nonfinite gradient at update {state.global_step + 1}: {name}")

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        """Audit the first two real updates and periodically record measured progress."""
        if state.global_step == 2:
            write_json(self.location(args) / "first_two_actual_updates.json", audit_optimizer(optimizer))
        if state.global_step % 50 == 0 or state.global_step == 2:
            torch.cuda.synchronize()
            elapsed = time.monotonic() - self.started
            write_json(self.location(args) / "progress.json", {
                "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "actual_updates": state.global_step, "total_updates": state.max_steps,
                "epoch": state.epoch, "elapsed_seconds": elapsed,
                "updates_per_second": state.global_step / elapsed,
                "eta_training_seconds": (state.max_steps - state.global_step) * elapsed / state.global_step,
                "includes": "Periodic evaluation/save/audits completed so far; future overhead not guaranteed.",
            })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Persist every main epoch's six metrics with actual step and epoch."""
        event = {"fold": self.fold, "partition": "outer_development", "step": state.global_step,
                 "epoch": state.epoch, **(metrics or {})}
        self.events.append(event)
        write_json(self.location(args) / "validation_history.json", self.events)

    def on_save(self, args, state, control, optimizer=None, model=None, **kwargs):
        """Verify live and CPU-reloaded optimizer state plus saved native tensor dtypes."""
        from safetensors import safe_open
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        audit = {"step": state.global_step, "epoch": state.epoch,
                 "live": audit_optimizer(optimizer),
                 "cpu_reload": audit_optimizer(optimizer, checkpoint / "optimizer.pt")}
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as saved:
            for name, parameter in model.named_parameters():
                if saved.get_slice(name).get_dtype() != {torch.bfloat16: "BF16", torch.float32: "F32"}[parameter.dtype]:
                    raise ValueError(f"Saved dtype changed: {name}")
        audit["saved_native_dtype_passed"] = True
        write_json(self.location(args) / f"save_{state.global_step}.json", audit)

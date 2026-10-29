"""Only the matched clean/noisy MSE trainer and its native optimizer audits."""
from pathlib import Path
import hashlib,json,math
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import SequentialSampler
from transformers import Trainer
from trt_pipeline.two_view import TwoViewVAEncoder,load_encoder_base_weights,save_encoder_base
from trt_pipeline.progress import atomic_write_json
PRIVATE_PARAMETERS=("regression_head.2.weight","regression_head.2.bias")

def unwrap_base(model):
    """Accept only the established single-device two-view wrapper."""

    current = model.module if hasattr(model, "module") else model
    if not isinstance(current, TwoViewVAEncoder):
        raise TypeError("Follow-up training requires the verified TwoViewVAEncoder.")
    return current.base_model

def parameter_partition(base_model):
    """Pin the shared versus output-specific parameter identities before training."""

    named = [(name, parameter) for name, parameter in base_model.named_parameters() if parameter.requires_grad]
    private = {name: tuple(parameter.shape) for name, parameter in named if name in PRIVATE_PARAMETERS}
    if private != {PRIVATE_PARAMETERS[0]: (2, 1024), PRIVATE_PARAMETERS[1]: (2,)}:
        raise ValueError("The private output Linear shape/names differ from the predeclared 2050 values.")
    flags = [name not in PRIVATE_PARAMETERS for name, _ in named]
    if not any(flags) or sum(parameter.numel() for name, parameter in named if name in PRIVATE_PARAMETERS) != 2050:
        raise ValueError("The shared/private parameter partition is invalid.")
    return named, flags

def optimizer_audit(optimizer, parameter_names):
    """Inspect actual moment storage, finite values, shapes, steps, and group settings."""

    groups, moments, initialized = [], {}, 0
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = parameter_names[id(parameter)]
            names.append(name)
            state = optimizer.state.get(parameter, {})
            if not state:
                continue
            initialized += 1
            tensors = {}
            for key, value in state.items():
                if not torch.is_tensor(value):
                    if isinstance(value, (float, int)) and not math.isfinite(value):
                        raise FloatingPointError("Nonfinite scalar optimizer state.")
                    continue
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Nonfinite optimizer tensor: {name}/{key}")
                if key in ("exp_avg", "exp_avg_sq"):
                    if value.shape != parameter.shape or value.dtype != parameter.dtype:
                        raise ValueError(f"Optimizer moment shape/native dtype mismatch: {name}/{key}")
                tensors[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                "value": float(value) if value.numel() == 1 and key == "step" else None}
            moments[name] = tensors
        groups.append({"name": group.get("group_name"), "lr": float(group["lr"]),
                       "weight_decay": float(group["weight_decay"]), "parameter_names": names})
    return {"status": "passed", "all_state_tensors_finite": True, "initialized_parameters": initialized,
            "moments_match_native_parameter_dtype_and_shape": True, "groups": groups, "by_parameter": moments}

def serialized_optimizer_audit(path, optimizer, parameter_names):
    """Reload a saved optimizer on CPU and verify every stored tensor without GPU allocation."""

    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if len(saved["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Serialized optimizer group count changed.")
    tensor_count, state_count = 0, 0
    dtypes = {}
    for source_group, live_group in zip(saved["param_groups"], optimizer.param_groups):
        if len(source_group["params"]) != len(live_group["params"]):
            raise ValueError("Serialized optimizer parameter group changed.")
        for state_id, parameter in zip(source_group["params"], live_group["params"]):
            state = saved["state"].get(state_id, {})
            if state:
                state_count += 1
            for key, value in state.items():
                if not torch.is_tensor(value):
                    if isinstance(value, (float, int)) and not math.isfinite(value):
                        raise FloatingPointError("Serialized optimizer contains a nonfinite scalar.")
                    continue
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Serialized optimizer contains a nonfinite tensor.")
                if key in ("exp_avg", "exp_avg_sq") and (value.shape != parameter.shape or value.dtype != parameter.dtype):
                    raise ValueError("Serialized optimizer moment shape/native dtype changed.")
                tensor_count += 1
                dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + 1
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"status": "passed", "cpu_reload": True, "all_state_tensors_finite": True,
            "state_parameters": state_count, "state_tensors": tensor_count, "tensor_dtypes": dtypes,
            "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class CompanionTrainer(Trainer):
    """Preserve the matched-MSE path while excluding all other loss/gradient methods."""

    def __init__(self, *args, condition, companion_config, **kwargs):
        """Bind the fixed physical-batch recipe and audit parameter identities."""
        if condition != 'matched_mse':
            raise ValueError('Only matched_mse companion training is retained')
        self.condition = condition
        self.companion_config = companion_config
        self.last_loss_components = {}
        self.initial_batch_fingerprints = []
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1 or self.is_deepspeed_enabled or self.is_fsdp_enabled or self.args.gradient_accumulation_steps != 1:
            raise ValueError('Companion requires one GPU and accumulation one')
        self.model_accepts_loss_kwargs = False
        self.named_base_parameters, self.shared_flags = parameter_partition(unwrap_base(self.model))

    def _get_eval_sampler(self, eval_dataset):
        """Keep validation predictions in canonical row order."""
        return None if eval_dataset is None else SequentialSampler(eval_dataset)

    def create_optimizer(self):
        """Use the original model AdamW groups with unchanged settings."""
        result = super().create_optimizer()
        for group in self.optimizer.param_groups:
            group.setdefault('group_name', 'model')
        return result

    def create_scheduler(self, num_training_steps, optimizer=None):
        """Keep the original warmup and linear decay at every optimizer step."""
        if self.lr_scheduler is None:
            optimizer = self.optimizer if optimizer is None else optimizer
            warmup = self.args.get_warmup_steps(num_training_steps)

            def model_factor(step):
                """Compute the unchanged per-step learning-rate multiplier."""
                if step < warmup:
                    return float(step) / max(1, warmup)
                return max(0., float(num_training_steps-step) / max(1, num_training_steps-warmup))

            self.lr_scheduler = LambdaLR(optimizer, [model_factor for group in optimizer.param_groups])
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Average squared error over both training views and both VA dimensions."""
        labels = inputs['labels'].float()
        outputs = model(**{key: value for key, value in inputs.items() if key != 'labels'})
        if model.training:
            loss = F.mse_loss(outputs['view_predictions'].float(), torch.cat((labels, labels), dim=0))
            self.last_loss_components = {'two_view_mse': float(loss.detach()), 'method_loss': float(loss.detach())}
            outputs = {'logits': outputs['logits']}
        else:
            loss = F.mse_loss(outputs['logits'].float(), labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Companion loss is nonfinite')
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Use the same matched-MSE forward/backward and initial batch fingerprints."""
        model.train()
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        if len(self.initial_batch_fingerprints) < 2:
            digest = hashlib.sha256()
            for key in sorted(inputs):
                value = inputs[key]
                if torch.is_tensor(value):
                    raw = value.detach().cpu().contiguous()
                    digest.update(key.encode())
                    digest.update(str((str(raw.dtype), tuple(raw.shape))).encode())
                    digest.update(raw.reshape(-1).view(torch.uint8).numpy().tobytes())
            rng = {'cpu': hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()}
            if self.args.device.type == 'cuda':
                rng['cuda'] = hashlib.sha256(torch.cuda.get_rng_state(self.args.device).cpu().numpy().tobytes()).hexdigest()
            self.initial_batch_fingerprints.append({'step': int(self.state.global_step+1), 'input_labels_sha256': digest.hexdigest(), 'rng_before_forward': rng})
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        self.accelerator.backward(loss)
        return loss.detach()

    def log(self, logs, *args, **kwargs):
        """Record diagnostics without changing data order or random state."""
        if 'loss' in logs:
            logs = {**logs, **self.last_loss_components}
        return super().log(logs, *args, **kwargs)

    def parameter_names(self):
        """Map the unchanged model optimizer tensors to their original names."""
        return {id(parameter): name for name, parameter in self.named_base_parameters}

    def _save(self, output_dir=None, state_dict=None):
        """Save the native base, tokenizer and matching companion configuration."""
        output = Path(self.args.output_dir if output_dir is None else output_dir)
        save_encoder_base(unwrap_base(self.model), output)
        if callable(getattr(self.processing_class, 'save_pretrained', None)):
            self.processing_class.save_pretrained(str(output))
        atomic_write_json(output/'method_state.json', {'condition': self.condition, 'config': self.companion_config.to_dict()})
        atomic_write_json(output/'training_arguments.json', self.args.to_dict())

    def _load_best_model(self):
        """Strictly reload the checkpoint selected only by clean Bvalidation MSE."""
        if self.state.best_model_checkpoint is None:
            raise RuntimeError('No selected companion checkpoint')
        path = Path(self.state.best_model_checkpoint)
        state = json.loads((path/'method_state.json').read_text())
        if state != {'condition': self.condition, 'config': self.companion_config.to_dict()}:
            raise ValueError('Selected checkpoint has a different companion contract')
        load_encoder_base_weights(unwrap_base(self.model), path)

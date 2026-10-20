"""DDPM-specific training, label-free evaluation, and self-describing checkpoints."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from va_model_code.decoder_va.trainer import VARegressionTrainer
from .config import positive_finite


class DiffusionTrainer(VARegressionTrainer):
    """Log training epsilon MSE and select checkpoints by generated-VA MSE."""

    def __init__(self, *args, head_learning_rate=1e-4, run_contract=None, **kwargs):
        self.head_learning_rate = positive_finite(head_learning_rate, "head_learning_rate")
        self.run_contract = run_contract
        super().__init__(*args, loss_name="mse", **kwargs)

    def create_optimizer(self):
        """Give the new denoiser its explicit LR while retaining encoder optimizer policy."""

        if self.optimizer is None:
            super().create_optimizer()
            groups = []
            head_ids = {id(p) for p in self.model.denoiser.parameters()}
            for group in self.optimizer.param_groups:
                for is_head in (False, True):
                    params = [p for p in group["params"] if (id(p) in head_ids) == is_head]
                    if params:
                        part = {**group, "params": params}
                        part["lr"] = self.head_learning_rate if is_head else self.args.learning_rate
                        part["initial_lr"] = part["lr"]
                        groups.append(part)
            self.optimizer.param_groups.clear()
            for group in groups:
                self.optimizer.add_param_group(group)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Backpropagate only epsilon prediction MSE during training."""

        del num_items_in_batch, kwargs
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("DDPM training must return epsilon MSE.")
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Generate without labels, then score samples against the untouched gold values."""

        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        with torch.no_grad(), self.compute_loss_context_manager():
            logits = model(**model_inputs).logits
        loss = None if labels is None else F.mse_loss(logits.float(), labels.float()).detach()
        if prediction_loss_only:
            return loss, None, None
        return loss, logits.detach(), None if labels is None else labels.detach()

    def _save(self, output_dir=None, state_dict=None):
        """Write the full reconstruction and run contracts beside every saved weight file."""

        super()._save(output_dir, state_dict=state_dict)
        directory = Path(output_dir or self.args.output_dir)
        self.model.save_architecture_manifest(directory)
        if self.run_contract is not None:
            (directory / "diffusion_training_contract.json").write_text(
                json.dumps(self.run_contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )

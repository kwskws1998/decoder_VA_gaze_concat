"""Transformers Trainer integration for decoder-based VA regression."""

from __future__ import annotations

from typing import Any

from transformers import Trainer

from .losses import va_regression_loss


REDISTRIBUTION_PARAMETER_SUFFIXES = (
    "gaze_redistributor.kernel.log_sigma_left",
    "gaze_redistributor.kernel.log_sigma_right",
)


class VARegressionTrainer(Trainer):
    """Trainer that applies the selected two-output VA objective."""

    def __init__(
        self,
        *args: Any,
        loss_name: str,
        redistribution_learning_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.loss_name = loss_name
        self.redistribution_learning_rate = redistribution_learning_rate
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def create_optimizer(self, model=None):
        """Create a dedicated zero-decay optimizer group for learned gaze widths."""

        if self.redistribution_learning_rate is None:
            if model is None:
                return super().create_optimizer()
            return super().create_optimizer(model=model)
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model if model is None else model
        named_trainable = [
            (name, parameter)
            for name, parameter in opt_model.named_parameters()
            if parameter.requires_grad
        ]
        sigma_named = [
            (name, parameter)
            for name, parameter in named_trainable
            if name.endswith(REDISTRIBUTION_PARAMETER_SUFFIXES)
        ]
        observed_suffixes = {
            suffix
            for name, _ in sigma_named
            for suffix in REDISTRIBUTION_PARAMETER_SUFFIXES
            if name.endswith(suffix)
        }
        if (
            observed_suffixes != set(REDISTRIBUTION_PARAMETER_SUFFIXES)
            or len(sigma_named) != 2
        ):
            observed = ", ".join(name for name, _ in sigma_named) or "<none>"
            raise ValueError(
                "A redistribution-specific learning rate requires exactly the two "
                "trainable log-sigma parameters; observed: " + observed
            )

        sigma_parameter_ids = {id(parameter) for _, parameter in sigma_named}
        decay_parameter_names = set(self.get_decay_parameter_names(opt_model))
        grouped_parameters = []
        base_decay = [
            parameter
            for name, parameter in named_trainable
            if id(parameter) not in sigma_parameter_ids and name in decay_parameter_names
        ]
        base_no_decay = [
            parameter
            for name, parameter in named_trainable
            if id(parameter) not in sigma_parameter_ids and name not in decay_parameter_names
        ]
        if base_decay:
            grouped_parameters.append(
                {"params": base_decay, "weight_decay": self.args.weight_decay}
            )
        if base_no_decay:
            grouped_parameters.append(
                {"params": base_no_decay, "weight_decay": 0.0}
            )
        grouped_parameters.append(
            {
                "params": [parameter for _, parameter in sigma_named],
                "lr": float(self.redistribution_learning_rate),
                "weight_decay": 0.0,
            }
        )

        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            optimizer_kwargs = dict(optimizer_kwargs)
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
            optimizer_kwargs = dict(optimizer_kwargs)
        incompatible = {"params", "model", "optimizer_dict"}.intersection(
            optimizer_kwargs
        )
        if incompatible:
            raise RuntimeError(
                "The selected Trainer optimizer cannot preserve the dedicated "
                "redistribution parameter group: " + ", ".join(sorted(incompatible))
            )
        self.optimizer = optimizer_cls(grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def log(self, logs, *args: Any, **kwargs: Any) -> None:
        """Add learned widths and their scheduled LR to ordinary Trainer logs."""

        enriched = dict(logs)
        redistributor = getattr(self.model, "gaze_redistributor", None)
        kernel = getattr(redistributor, "kernel", None)
        if kernel is not None:
            enriched["redistribution_sigma_left"] = float(
                kernel.sigma_left.detach().cpu()
            )
            enriched["redistribution_sigma_right"] = float(
                kernel.sigma_right.detach().cpu()
            )
            if self.optimizer is not None:
                sigma_ids = {id(parameter) for parameter in kernel.parameters()}
                matching_groups = [
                    group
                    for group in self.optimizer.param_groups
                    if sigma_ids.intersection(id(parameter) for parameter in group["params"])
                ]
                if len(matching_groups) == 1:
                    enriched["redistribution_learning_rate"] = float(
                        matching_groups[0]["lr"]
                    )
        return super().log(enriched, *args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
        **kwargs,
    ):
        """Compute loss without mutating the batch supplied by Trainer."""

        del num_items_in_batch, kwargs
        if "labels" not in inputs:
            raise KeyError("Training batches must contain a 'labels' tensor.")
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs["logits"]
        breakdown = va_regression_loss(logits, labels, self.loss_name)
        return (breakdown.total, outputs) if return_outputs else breakdown.total

"""Transformers Trainer integration for decoder-based VA regression."""

from __future__ import annotations

from typing import Any

from transformers import Trainer

from .losses import va_regression_loss


class VARegressionTrainer(Trainer):
    """Trainer that applies the selected point or heteroscedastic VA objective."""

    def __init__(
        self,
        *args: Any,
        loss_name: str = "heteroscedastic+ccc",
        hetero_mse_weight: float = 0.1,
        hetero_ccc_weight: float = 0.1,
        hetero_logvar_min: float = -5.0,
        hetero_logvar_max: float = 3.0,
        **kwargs: Any,
    ) -> None:
        self.loss_name = loss_name
        self.hetero_mse_weight = hetero_mse_weight
        self.hetero_ccc_weight = hetero_ccc_weight
        self.hetero_logvar_min = hetero_logvar_min
        self.hetero_logvar_max = hetero_logvar_max
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

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
        breakdown = va_regression_loss(
            logits,
            labels,
            self.loss_name,
            ccc_weight=self.hetero_ccc_weight,
            mse_weight=self.hetero_mse_weight,
            min_log_variance=self.hetero_logvar_min,
            max_log_variance=self.hetero_logvar_max,
        )
        return (breakdown.total, outputs) if return_outputs else breakdown.total

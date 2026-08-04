"""Transformers Trainer integration for decoder-based VA regression."""

from __future__ import annotations

from typing import Any

from transformers import Trainer

from .losses import va_regression_loss


class VARegressionTrainer(Trainer):
    """Trainer that applies the selected two-output VA objective."""

    def __init__(
        self,
        *args: Any,
        loss_name: str,
        **kwargs: Any,
    ) -> None:
        self.loss_name = loss_name
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
        breakdown = va_regression_loss(logits, labels, self.loss_name)
        return (breakdown.total, outputs) if return_outputs else breakdown.total

"""Transformers Trainer integration for decoder-based VA regression."""

from __future__ import annotations

from typing import Any

from torch.utils.data import SequentialSampler
from transformers import Trainer
from transformers.trainer_utils import has_length

from .losses import va_regression_loss


class VARegressionTrainer(Trainer):
    """Trainer for the original VA mean objectives with sequential evaluation."""

    def __init__(self, *args: Any, loss_name: str, **kwargs: Any) -> None:
        """Use the ordinary optimizer and the original VA regression objectives."""

        self.loss_name = loss_name
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def _get_eval_sampler(self, eval_dataset):
        """Keep prediction rows aligned with metadata independently of training grouping."""

        if eval_dataset is None or not has_length(eval_dataset):
            return None
        return SequentialSampler(eval_dataset)


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

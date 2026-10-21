"""Train only a baseline-shaped FP32 head on frozen DDPM encoder features."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from va_model_code.decoder_va.evaluation import calculate_va_metrics


class FrozenFeatureHead(nn.Module):
    """Mirror the direct VA head: LayerNorm, dropout, linear, bounded VA means."""

    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 2)).float()

    def forward(self, features):
        """Compute direct VA means with no autocast or encoder update."""
        return F.hardsigmoid(self.layers(features.float()))


@torch.no_grad()
def predict_head(head, features, *, batch_size, device):
    """Read frozen features in sequential order without labels."""
    head.eval()
    return torch.cat([head(x.to(device)).cpu() for x in features.split(batch_size)]).numpy()


def train_probe(train_features, train_labels, eval_features, eval_labels, *,
                seed, epochs, learning_rate, batch_size, weight_decay, dropout, device, callback=None):
    """Fit on training labels only; retain the original held-out checkpoint-selection protocol."""
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Probe epochs, batch size and learning rate must be positive.")
    if not np.isfinite(weight_decay) or weight_decay < 0 or not 0 <= dropout < 1:
        raise ValueError("Invalid probe regularization.")
    for features, labels in [(train_features, train_labels), (eval_features, eval_labels)]:
        if features.ndim != 2 or labels.shape != (len(features), 2) or not len(features):
            raise ValueError("Expected nonempty [N,D] features and [N,2] labels.")
        if features.requires_grad or labels.requires_grad:
            raise ValueError("Probe inputs must be detached from the encoder.")
        if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
            raise ValueError("Probe inputs must be finite.")
        if not ((labels >= 0) & (labels <= 1)).all():
            raise ValueError("Probe targets must be normalized VA.")
    if train_features.shape[1] != eval_features.shape[1]:
        raise ValueError("Feature widths disagree.")
    torch.manual_seed(seed)
    head = FrozenFeatureHead(train_features.shape[1], dropout).to(device)
    groups = [{"params": [p for p in head.parameters() if (p.ndim >= 2) == decay],
               "weight_decay": weight_decay if decay else 0.0} for decay in (True, False)]
    optimizer = torch.optim.AdamW(groups, lr=learning_rate)
    loader = DataLoader(TensorDataset(train_features.cpu(), train_labels.cpu()),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history, best, best_weights, best_epoch = [], float("inf"), None, None
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = head(features.to(device))
            loss = F.mse_loss(prediction, labels.to(device).float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite probe training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach().item() * len(features)
        predictions = predict_head(head, eval_features, batch_size=batch_size, device=device)
        metrics = calculate_va_metrics(eval_labels.cpu().numpy(), predictions)
        row = {"epoch": epoch, "train_update_mse": loss_sum / len(train_features),
               **{f"eval_{name}": value for name, value in metrics.items()}}
        history.append(row)
        if metrics["mse_mean"] < best:
            best, best_epoch = metrics["mse_mean"], epoch
            best_weights = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        if callback:
            callback(row)
    head.load_state_dict(best_weights)
    head.eval()
    return head, {"best_epoch": best_epoch, "best_metric": best, "epochs": epochs,
                  "history": history, "optimizer": "AdamW", "learning_rate_schedule": "constant",
                  "selection": "minimum_heldout_mse_existing_protocol"}

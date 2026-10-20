import math

import torch
import torch.nn as nn

class AsymGaussianRedistributor(nn.Module):
    def __init__(
        self,
        init_sigma_left: float = 1.0,
        init_sigma_right: float = 1.0,
        min_sigma: float = 1e-6,
    ):
        super().__init__()
        print("=" * 25)
        print(f"Initializing AsymGaussianRedistributor")
        print(f"init_sigma_left={init_sigma_left}, init_sigma_right={init_sigma_right}")
        print("=" * 25)

        self.log_sigma_left = nn.Parameter(
            torch.tensor(math.log(init_sigma_left), dtype=torch.float32)
        )
        self.log_sigma_right = nn.Parameter(
            torch.tensor(math.log(init_sigma_right), dtype=torch.float32)
        )

        self.min_sigma = min_sigma

    @property
    def sigma_left(self):
        return torch.exp(self.log_sigma_left) + self.min_sigma

    @property
    def sigma_right(self):
        return torch.exp(self.log_sigma_right) + self.min_sigma

    def forward(self, trt_values: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        trt_values: [B, T]
        attention_mask: [B, T] (1=valid, 0=pad), optional
        return: [B, T]
        """
        if trt_values.dim() != 2:
            raise ValueError(f"trt_values must be [B, T], got shape {tuple(trt_values.shape)}")

        x = trt_values.float()
        _, T = x.shape
        device = x.device

        if attention_mask is None:
            attention_mask = torch.ones_like(x, dtype=torch.float32, device=device)
        else:
            attention_mask = attention_mask.float()

        pos = torch.arange(T, device=device, dtype=torch.float32)

        target_pos = pos.view(1, T, 1)
        source_pos = pos.view(1, 1, T)
        diff = target_pos - source_pos  # [1, T, T]
        abs_diff = diff.abs()

        sigma = torch.where(diff < 0, self.sigma_left, self.sigma_right)  # [1, T, T]

        weights = torch.exp(-0.5 * (abs_diff / sigma) ** 2)  # [B, T, T]

        src_mask = attention_mask.unsqueeze(1)
        tgt_mask = attention_mask.unsqueeze(2)
        weights = weights * src_mask * tgt_mask

        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights = weights / denom

        redistributed = torch.bmm(weights, x.unsqueeze(-1)).squeeze(-1)
        redistributed = redistributed * attention_mask

        return redistributed.to(dtype=trt_values.dtype)
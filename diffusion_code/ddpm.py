"""Gaussian DDPM epsilon training and ancestral sampling in two-dimensional VA."""

import math
import torch
from torch import nn

from .config import DiffusionConfig


class GaussianDDPM(nn.Module):
    """Use FP32 arithmetic with a cosine schedule built accurately in FP64."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.settings = config
        t = torch.linspace(0, 1, config.timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), abar[:-1]))
        values = {
            "alpha_bar": abar,
            "posterior_variance": beta * (1 - previous) / (1 - abar),
            "posterior_clean_coefficient": beta * previous.sqrt() / (1 - abar),
            "posterior_noisy_coefficient": (1 - previous) * alpha.sqrt() / (1 - abar),
        }
        for name, value in values.items():
            self.register_buffer(name, value.float(), persistent=False)
        generator = torch.Generator(device="cpu").manual_seed(config.sampling_seed)
        bank = torch.randn(config.timesteps + 1, config.samples, 2, generator=generator)
        self.register_buffer("sampling_noise", bank, persistent=False)

    def noise_targets(self, labels, *, timesteps=None, noise=None):
        """Corrupt normalized gold VA only during training; never use it for inference."""

        if labels.ndim != 2 or labels.shape[1] != 2 or not len(labels):
            raise ValueError("labels must be a nonempty [batch, 2] tensor.")
        if not bool((torch.isfinite(labels) & (labels >= 0) & (labels <= 1)).all()):
            raise ValueError("DDPM labels must be finite normalized VA values in [0, 1].")
        labels = labels.float()
        if timesteps is None:
            timesteps = torch.randint(self.settings.timesteps, (len(labels),), device=labels.device)
        if timesteps.shape != (len(labels),) or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be an int64 vector of length batch.")
        if not bool(((timesteps >= 0) & (timesteps < self.settings.timesteps)).all()):
            raise ValueError("Training timestep outside the DDPM schedule.")
        noise = torch.randn_like(labels) if noise is None else noise.float()
        if noise.shape != labels.shape or not bool(torch.isfinite(noise).all()):
            raise ValueError("noise must be a finite tensor shaped like labels.")
        clean = 2 * labels - 1
        abar = self.alpha_bar[timesteps, None]
        noisy = abar.sqrt() * clean + (1 - abar).sqrt() * noise
        return noisy, timesteps, noise

    def reverse_step(self, noisy, epsilon, step: int, noise):
        """Apply the analytical DDPM posterior, omitting random noise at the final step."""

        if not 0 <= step < self.settings.timesteps:
            raise ValueError("Sampling timestep outside the DDPM schedule.")
        abar = self.alpha_bar[step]
        clean = (noisy.float() - (1 - abar).sqrt() * epsilon.float()) / abar.sqrt()
        if self.settings.clip_clean:
            clean = clean.clamp(-1, 1)
        mean = (
            self.posterior_clean_coefficient[step] * clean
            + self.posterior_noisy_coefficient[step] * noisy.float()
        )
        return mean if step == 0 else mean + self.posterior_variance[step].sqrt() * noise

    @torch.no_grad()
    def sample(self, denoise, *, batch_size: int):
        """Generate with a private common noise bank, independent of labels and batching."""

        noisy = self.sampling_noise[0].unsqueeze(0).expand(batch_size, -1, -1).clone()
        for step in reversed(range(self.settings.timesteps)):
            time = torch.full((batch_size,), step, dtype=torch.long, device=noisy.device)
            epsilon = denoise(noisy, time)
            noisy = self.reverse_step(noisy, epsilon, step, self.sampling_noise[step + 1])
        samples = ((noisy + 1) / 2).clamp(0, 1)
        if not bool(torch.isfinite(samples).all()):
            raise FloatingPointError("DDPM sampling produced nonfinite VA values.")
        return samples

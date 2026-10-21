"""Experimental frozen-mean residual DDPM; separate semantics from the original VA DDPM."""

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from diffusion_code.config import DiffusionConfig
from diffusion_code.ddpm import GaussianDDPM
from diffusion_code.model import ConditionalDenoiser
from va_model_code.decoder_va.dataset import load_auto_tokenizer
from va_model_code.decoder_va.model import reconstruct_decoder_va_model


FAMILY = "conditional_mean_guided_ddpm_va"
MANIFEST = "mean_guided_architecture.json"
MEAN_CONTRACT = {
    "version": 1,
    "source": "pretrained_raw_trt_direct_mse_last_text_token",
    "encoder_and_mean_head": "frozen_eval",
    "trainable": "denoiser_only",
    "target": "residual_r0=(2*y-1)-(2*mu(x)-1)",
    "forward": "rt=sqrt(alpha_bar)*r0+sqrt(1-alpha_bar)*epsilon",
    "mean_condition": "learned_projection_of_2*mu(x)-1_added_to_denoiser_pooled_context",
    "clip_clean_residual": "[-1-(2*mu-1),1-(2*mu-1)]",
    "inverse_transform": "(r0+(2*mu-1)+1)/2_then_clamp_0_1",
    "training_residuals": "in_sample_residuals_of_frozen_mean_predictor",
}


def validate_mean(mean):
    """Require finite normalized, detached two-axis predictions from the frozen model."""
    if mean.ndim != 2 or mean.shape[1] != 2 or not len(mean):
        raise ValueError("Frozen mean must have nonempty shape [batch,2].")
    if mean.requires_grad or not bool((torch.isfinite(mean) & (mean >= 0) & (mean <= 1)).all()):
        raise ValueError("Frozen means must be detached finite VA values in [0,1].")


class MeanCenteredDDPM(GaussianDDPM):
    """Diffuse centered residuals using the same coefficients but mean-dependent bounds."""

    def noise_targets(self, labels, mean, *, timesteps=None, noise=None):
        """Center normalized targets using predictions only, without changing the noise law."""
        validate_mean(mean)
        if labels.shape != mean.shape or labels.device != mean.device:
            raise ValueError("Labels and frozen means must share shape and device.")
        noisy, time, epsilon = super().noise_targets(labels, timesteps=timesteps, noise=noise)
        centered = noisy - self.alpha_bar[time, None].sqrt() * (2 * mean.float() - 1)
        return centered, time, epsilon

    def reverse_step(self, noisy, epsilon, step, noise, mean):
        """Clip reconstructed residuals to the valid VA interval around each frozen mean."""
        if not 0 <= step < self.settings.timesteps:
            raise ValueError("Sampling timestep outside the DDPM schedule.")
        center = 2 * mean.float() - 1
        while center.ndim < noisy.ndim:
            center = center.unsqueeze(1)
        abar = self.alpha_bar[step]
        clean = (noisy.float() - (1 - abar).sqrt() * epsilon.float()) / abar.sqrt()
        if self.settings.clip_clean:
            clean = torch.maximum(torch.minimum(clean, 1 - center), -1 - center)
        posterior = (self.posterior_clean_coefficient[step] * clean
                     + self.posterior_noisy_coefficient[step] * noisy.float())
        return posterior if step == 0 else posterior + self.posterior_variance[step].sqrt() * noise

    @torch.no_grad()
    def sample(self, denoise, *, mean):
        """Generate residuals and restore the frozen mean; use no gold labels."""
        validate_mean(mean)
        noisy = self.sampling_noise[0][None].expand(len(mean), -1, -1).clone()
        for step in reversed(range(self.settings.timesteps)):
            time = torch.full((len(mean),), step, dtype=torch.long, device=noisy.device)
            epsilon = denoise(noisy, time)
            noisy = self.reverse_step(noisy, epsilon, step, self.sampling_noise[step + 1], mean)
        samples = ((noisy + (2 * mean.float() - 1)[:, None] + 1) / 2).clamp(0, 1)
        if not torch.isfinite(samples).all():
            raise FloatingPointError("Mean-guided sampling produced nonfinite VA.")
        return samples


class MeanConditionedDenoiser(ConditionalDenoiser):
    """Add an explicit frozen-mean embedding to the DDPM pooled conditioning vector."""

    def __init__(self, encoder_dim, settings):
        super().__init__(encoder_dim, settings)
        self.mean_projection = nn.Linear(2, settings.hidden_dim).float()

    def prepare(self, features, mean):
        """Preserve token/TRT memory and expose the mean directly to every reverse step."""
        pooled, keys, values, mask = super().prepare(features)
        pooled = pooled + self.mean_projection(2 * mean.float() - 1)
        return pooled, keys, values, mask


class MeanGuidedDiffusionVA(nn.Module):
    """Train a denoiser on top of a frozen Raw TRT encoder and direct regression head."""

    supports_gradient_checkpointing = False

    def __init__(self, encoder, settings, *, reference):
        super().__init__()
        if not settings.clip_clean:
            raise ValueError("The mean-guided contract requires clipping in original VA bounds.")
        if (encoder.gaze_fusion != "prefix-concat" or tuple(encoder.gaze_provider.feature_indices) != (3,)
                or encoder.readout["mode"] != "last-token"
                or encoder.gaze_redistribution["method"] != "none" or encoder.gaze_relation["method"] != "none"
                or isinstance(encoder.regression_head, nn.Identity)):
            raise ValueError("Mean guidance requires an intact Raw TRT direct regressor.")
        self.encoder = encoder.eval().requires_grad_(False)
        self.settings = settings
        self.reference = copy.deepcopy(reference)
        self.config = copy.deepcopy(encoder.config)
        self.config.prediction_distribution = {"family": FAMILY, "schedule": settings.to_dict(), "mean_guidance": MEAN_CONTRACT}
        self.config.va_output_activation = "mean_of_mean_centered_generated_va_samples"
        self.denoiser = MeanConditionedDenoiser(encoder.hidden_size, settings)
        self.diffusion = MeanCenteredDDPM(settings)

    def train(self, mode=True):
        """Keep the pretrained predictor in eval mode even while the denoiser is training."""
        super().train(mode)
        self.encoder.eval()
        return self

    def get_input_embeddings(self):
        """Expose the frozen decoder embedding layer to the Trainer."""
        return self.encoder.get_input_embeddings()

    def condition(self, input_ids, attention_mask):
        """Run Qwen once under no_grad and build trainable denoiser memory outside no_grad."""
        if any(p.requires_grad for p in self.encoder.parameters()) or self.encoder.training:
            raise RuntimeError("Mean-guided training requires an entirely frozen eval-mode source model.")
        with torch.no_grad():
            features = self.encoder.encode(input_ids, attention_mask,
                                           return_aligned_tokens=self.settings.conditioning == "token-trt")
            with torch.autocast(input_ids.device.type, enabled=False):
                mean = F.hardsigmoid(self.encoder.regression_head(features.pooled.float()))
        if any(p.dtype != torch.float32 for p in self.denoiser.parameters()):
            raise ValueError("The mean-guided denoiser must remain FP32.")
        with torch.autocast(input_ids.device.type, enabled=False):
            return self.denoiser.prepare(features, mean), mean

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """Optimize residual epsilon MSE; evaluation ignores gold labels and restores mean VA."""
        del kwargs
        condition, mean = self.condition(input_ids, attention_mask)
        with torch.autocast(input_ids.device.type, enabled=False):
            if self.training:
                if labels is None:
                    raise ValueError("Mean-guided training requires labels.")
                noisy, time, epsilon = self.diffusion.noise_targets(labels, mean)
                loss = F.mse_loss(self.denoiser(noisy[:, None], time, condition)[:, 0], epsilon)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite mean-guided training loss.")
                return SequenceClassifierOutput(loss=loss)
            samples = self.diffusion.sample(lambda noisy, time: self.denoiser(noisy, time, condition), mean=mean)
        return SequenceClassifierOutput(logits=samples.mean(1))

    def save_architecture_manifest(self, directory):
        """Save a self-contained combined checkpoint with distinct residual-DDPM semantics."""
        directory = Path(directory)
        original = self.encoder.save_architecture_manifest(directory)
        blueprint = json.loads(original.read_text())
        blueprint["role"] = "frozen_raw_mean_predictor_with_regression_head"
        blueprint["state_dict"]["scope"] = "encoder.* of MeanGuidedDiffusionVA"
        raw = (json.dumps(blueprint, indent=2, sort_keys=True) + "\n").encode()
        (directory / "encoder_architecture.json").write_bytes(raw)
        original.unlink()
        payload = {
            "schema_version": 1, "model_family": FAMILY,
            "base_schedule": self.settings.to_dict(), "mean_guidance": MEAN_CONTRACT,
            "encoder_manifest": "encoder_architecture.json", "encoder_sha256": hashlib.sha256(raw).hexdigest(),
            "weights": "model.safetensors", "head_compute_dtype": "float32", "reference": self.reference,
        }
        (directory / MANIFEST).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_mean_guided_model(directory, *, dtype=torch.float32, et_cache_size=70000):
    """Strictly reload combined frozen-source and denoiser weights without the original run."""
    from safetensors import safe_open
    from safetensors.torch import load_model

    directory = Path(directory)
    payload = json.loads((directory / MANIFEST).read_text())
    required = {"schema_version", "model_family", "base_schedule", "mean_guidance", "encoder_manifest",
                "encoder_sha256", "weights", "head_compute_dtype", "reference"}
    if (set(payload) != required or payload["schema_version"] != 1 or payload["model_family"] != FAMILY
            or payload["mean_guidance"] != MEAN_CONTRACT or payload["head_compute_dtype"] != "float32"
            or payload["encoder_manifest"] != "encoder_architecture.json" or payload["weights"] != "model.safetensors"):
        raise ValueError("Unsupported mean-guided reconstruction contract.")
    raw = (directory / "encoder_architecture.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != payload["encoder_sha256"]:
        raise ValueError("Mean predictor blueprint hash mismatch.")
    blueprint = json.loads(raw)
    if blueprint.get("role") != "frozen_raw_mean_predictor_with_regression_head":
        raise ValueError("Incorrect mean predictor component role.")
    tokenizer = load_auto_tokenizer(directory, use_fast=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    encoder, tokenizer = reconstruct_decoder_va_model(blueprint, tokenizer=tokenizer, dtype=dtype, et_cache_size=et_cache_size)
    settings = DiffusionConfig.from_dict(payload["base_schedule"])
    model = MeanGuidedDiffusionVA(encoder, settings, reference=payload["reference"])
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as saved:
        for name, parameter in model.named_parameters():
            if name in saved.keys() and parameter.dtype == torch.float32 and saved.get_slice(name).get_dtype() != "F32":
                raise ValueError(f"Saved FP32 parameter has an incompatible dtype: {name}")
    load_model(model, directory / "model.safetensors", strict=True, device="cpu")
    model.eval()
    return model, tokenizer

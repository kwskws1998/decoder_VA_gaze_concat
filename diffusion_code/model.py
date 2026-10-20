"""Small timestep-conditioned denoiser with optional contextual text/TRT memory."""

import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from va_model_code.decoder_va.model import (
    build_qwen_va_model, validate_saved_trainable_precision,
    validate_gaze_alignment_contract, validate_precision_contract,
)
from va_model_code.decoder_va.dataset import load_auto_tokenizer
from .config import DiffusionConfig
from .ddpm import GaussianDDPM


MANIFEST = "diffusion_architecture.json"


class ConditionalDenoiser(nn.Module):
    """Condition every reverse step on pooled text and aligned word/TRT value vectors."""

    def __init__(self, encoder_dim, settings):
        super().__init__()
        self.settings = settings
        width = settings.hidden_dim
        self.pooled_projection = nn.Sequential(nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, width))
        self.state_projection = nn.Linear(2, width)
        self.time_projection = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
            for _ in range(settings.blocks)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))
        if settings.conditioning == "token-trt":
            self.text_projection = nn.Sequential(nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, width))
            self.trt_projection = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.float()

    def prepare(self, features):
        """Project keys/values once per Qwen pass, preserving gradients during training."""

        pooled = self.pooled_projection(features.pooled.float())
        memory = pooled[:, None]
        mask = torch.ones((len(pooled), 1), dtype=torch.bool, device=pooled.device)
        if self.settings.conditioning == "token-trt":
            if features.raw_gaze is None or features.raw_gaze.shape[-1] != 1:
                raise ValueError("token-trt conditioning requires exactly one raw TRT channel.")
            anchor_mask = features.gaze_mask.bool()
            count = int(anchor_mask.sum(1).max().item())
            if count:
                positions = torch.arange(anchor_mask.shape[1], device=pooled.device)[None]
                indices = torch.where(anchor_mask, positions, anchor_mask.shape[1]).sort(1).values[:, :count]
                valid = torch.arange(count, device=pooled.device)[None] < anchor_mask.sum(1)[:, None]
                indices = torch.where(valid, indices, 0)
                text = features.text_hidden.float().gather(
                    1, indices[..., None].expand(-1, -1, features.text_hidden.shape[-1])
                )
                trt = features.raw_gaze.float().gather(1, indices[..., None])
                if not bool(torch.isfinite(trt[valid]).all()):
                    raise ValueError("Valid raw TRT values must be finite.")
                text = text.masked_fill(~valid[..., None], 0)
                trt = trt.masked_fill(~valid[..., None], 0)
                anchors = self.text_projection(text) + self.trt_projection(trt)
                memory = torch.cat((memory, anchors), dim=1)
                mask = torch.cat((mask, valid), dim=1)
        return pooled, self.key(memory), self.value(memory), mask

    def forward(self, noisy, timesteps, condition):
        """Predict epsilon without accessing gold VA; accept batched Monte Carlo samples."""

        pooled, keys, values, mask = condition
        width = self.settings.hidden_dim
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(width // 2, device=noisy.device) / (width // 2 - 1)
        )
        phase = timesteps.float()[:, None] * frequencies[None]
        time = self.time_projection(torch.cat((phase.sin(), phase.cos()), dim=-1))
        state = self.state_projection(noisy.float()) + (pooled + time)[:, None]
        weights = torch.einsum("bsd,bld->bsl", self.query(state), keys) / math.sqrt(width)
        weights = weights.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        state = state + torch.einsum("bsl,bld->bsd", weights, values)
        for block in self.blocks:
            state = state + block(state)
        return self.output(state)


class DiffusionVA(nn.Module):
    """Reuse the baseline encoder while replacing its regression head with a DDPM."""

    supports_gradient_checkpointing = True

    def __init__(self, encoder, settings: DiffusionConfig):
        super().__init__()
        if settings.conditioning == "token-trt" and (
            encoder.gaze_fusion != "prefix-concat"
            or tuple(encoder.gaze_provider.feature_indices) != (3,)
        ):
            raise ValueError("token-trt requires prefix-concat with TRT only.")
        self.encoder = encoder
        self.encoder.regression_head = nn.Identity()
        self.settings = settings
        self.config = copy.deepcopy(encoder.config)
        self.config.prediction_distribution = {"family": "conditional_ddpm", **settings.to_dict()}
        self.config.va_output_activation = "sample_mean_after_clipped_inverse_transform"
        self.denoiser = ConditionalDenoiser(encoder.hidden_size, settings)
        self.diffusion = GaussianDDPM(settings)

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate checkpointing to the unchanged Qwen encoder."""

        return self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Delegate checkpoint disabling to the encoder."""

        return self.encoder.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        """Expose Qwen embeddings to Transformers utilities."""

        return self.encoder.get_input_embeddings()

    def condition(self, input_ids, attention_mask):
        """Encode text once, then prepare FP32 memory reused across denoising steps."""

        features = self.encoder.encode(
            input_ids, attention_mask,
            return_aligned_tokens=self.settings.conditioning == "token-trt",
        )
        if any(p.dtype != torch.float32 for p in self.denoiser.parameters()):
            raise ValueError("The DDPM head must remain FP32; use backbone autocast.")
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            return self.denoiser.prepare(features)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """Train with noise MSE; in evaluation sample without using labels."""

        del kwargs
        condition = self.condition(input_ids, attention_mask)
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            if self.training:
                if labels is None:
                    raise ValueError("DDPM training requires labels.")
                noisy, time, epsilon = self.diffusion.noise_targets(labels)
                prediction = self.denoiser(noisy[:, None], time, condition)[:, 0]
                loss = F.mse_loss(prediction, epsilon)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite DDPM epsilon loss.")
                return SequenceClassifierOutput(loss=loss)
            samples = self.diffusion.sample(
                lambda noisy, time: self.denoiser(noisy, time, condition),
                batch_size=len(input_ids),
            )
        return SequenceClassifierOutput(logits=samples.mean(1))

    def save_architecture_manifest(self, directory):
        """Persist the original encoder reconstruction separately from DDPM semantics."""

        directory = Path(directory)
        encoder_path = self.encoder.save_architecture_manifest(directory)
        encoder_manifest = json.loads(encoder_path.read_text())
        encoder_manifest["role"] = "encoder_blueprint_before_regression_head_replacement"
        encoder_manifest["state_dict"] = {
            "filename": "model.safetensors", "format": "safetensors",
            "scope": "encoder.* component of DiffusionVA; regression_head removed",
            "strict_loading": True,
        }
        (directory / "encoder_architecture.json").write_text(
            json.dumps(encoder_manifest, indent=2, sort_keys=True) + "\n"
        )
        encoder_path.unlink()
        payload = {
            "schema_version": 1, "model_family": "conditional_ddpm_va",
            "diffusion": self.settings.to_dict(),
            "encoder_manifest": "encoder_architecture.json",
            "weights": "model.safetensors",
            "head_compute_dtype": "float32",
            "prediction": "mean_of_samples_in_normalized_va_units",
        }
        (directory / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n")


def load_diffusion_model(directory, *, dtype=torch.float32, et_cache_size=70000):
    """Rebuild and strictly reload DDPM weights, never interpreting them as a baseline."""

    from safetensors.torch import load_model

    directory = Path(directory)
    payload = json.loads((directory / MANIFEST).read_text())
    if payload.get("schema_version") != 1 or payload.get("model_family") != "conditional_ddpm_va":
        raise ValueError("Not a supported conditional DDPM checkpoint.")
    settings = DiffusionConfig.from_dict(payload["diffusion"])
    expected = {
        "schema_version": 1, "model_family": "conditional_ddpm_va",
        "diffusion": settings.to_dict(), "encoder_manifest": "encoder_architecture.json",
        "weights": "model.safetensors", "head_compute_dtype": "float32",
        "prediction": "mean_of_samples_in_normalized_va_units",
    }
    if payload != expected:
        raise ValueError("Unsupported or modified DDPM reconstruction semantics.")
    architecture = json.loads((directory / "encoder_architecture.json").read_text())
    if architecture.get("role") != "encoder_blueprint_before_regression_head_replacement":
        raise ValueError("Missing DDPM encoder-component identity.")
    reconstruction = architecture["reconstruction"]
    validate_precision_contract(architecture["precision_contract"], context="DDPM encoder")
    validate_gaze_alignment_contract(
        architecture["gaze_alignment"], gaze_fusion=reconstruction["gaze_fusion"],
        context="DDPM encoder",
    )
    if reconstruction["gaze_alignment"] != architecture["gaze_alignment"]:
        raise ValueError("Saved encoder alignment metadata is inconsistent.")
    for field in ("gaze_fusion", "gaze_redistribution", "gaze_relation", "readout", "prediction_distribution"):
        if reconstruction[field] != architecture[field]:
            raise ValueError(f"Saved encoder metadata is inconsistent: {field}")
    tokenizer = load_auto_tokenizer(
        directory, use_fast=True, trust_remote_code=False,
    )
    tokenizer.padding_side = "right"
    encoder = build_qwen_va_model(
        tokenizer,
        model_id=reconstruction["decoder_model_id"],
        model_revision=reconstruction["decoder_revision"],
        gaze_fusion=reconstruction["gaze_fusion"],
        gaze_feature_indices=reconstruction["et_feature_indices"] or None,
        et_repo_id=reconstruction["et_repo_id"], et_revision=reconstruction["et_revision"],
        et_filename=reconstruction["et_filename"], et_cache_size=et_cache_size,
        dtype=dtype, finetuning_mode=reconstruction["finetuning_mode"],
        lora_rank=reconstruction["lora_rank"], lora_alpha=reconstruction["lora_alpha"],
        lora_dropout=reconstruction["lora_dropout"],
        gaze_projection_dim=reconstruction["gaze_projection_dim"],
        gaze_projection_dropout=tuple(reconstruction["gaze_projection_dropout"]),
        classifier_dropout=reconstruction["classifier_dropout"],
        attn_implementation=reconstruction["attn_implementation"],
        gaze_redistribution=architecture["gaze_redistribution"],
        gaze_relation=architecture["gaze_relation"], readout=architecture["readout"],
        prediction_distribution=architecture["prediction_distribution"],
    )
    model = DiffusionVA(encoder, settings)
    validate_saved_trainable_precision(model, directory / "model.safetensors")
    load_model(model, directory / "model.safetensors", strict=True, device="cpu")
    return model.eval(), tokenizer

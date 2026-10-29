"""The retained clean/noisy companion operator; noise is training-only."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from trt_pipeline.data import text_groups
from trt_pipeline.precision import audit_model_precision,validate_saved_precision
from trt_pipeline.progress import atomic_write_json
from trt_pipeline.core.evaluation import trainer_compute_metrics
NOISE_STD_FRACTION=0.01

def gaussian_text_view(embeddings, attention_mask, std_fraction=NOISE_STD_FRACTION):
    """Add FP32 Gaussian noise using a detached per-example valid-token RMS scale."""

    if embeddings.ndim != 3 or attention_mask.shape != embeddings.shape[:2]:
        raise ValueError("Embedding noise requires [batch, tokens, hidden] and matching attention_mask.")
    if not 0 < float(std_fraction) < 1:
        raise ValueError("std_fraction must lie strictly between zero and one.")
    valid = attention_mask.bool().unsqueeze(-1)
    if not valid.any(dim=1).all():
        raise ValueError("Every text example needs at least one valid token.")
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        clean = embeddings.float()
        denominator = attention_mask.bool().sum(dim=1).float() * embeddings.shape[-1]
        rms = (clean.detach().square().masked_fill(~valid, 0).sum(dim=(1, 2)) / denominator).sqrt()
        noise = torch.randn_like(clean) * (float(std_fraction) * rms[:, None, None])
        noise = noise.masked_fill(~valid, 0)
        return (clean + noise).to(embeddings.dtype)


class TwoViewVAEncoder(nn.Module):

    def __init__(self, base_model):
        """Register exactly one base model and no extra trainable parameters."""

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.main_input_name = "input_ids"

    def get_input_embeddings(self):
        """Expose the native embedding interface used by Trainer utilities."""

        return self.base_model.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        """Delegate checkpointing without altering native backbone precision."""

        return self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs, **kwargs)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None, **kwargs):
        """Run two matched training views together or exactly one clean evaluation view."""

        del labels, token_type_ids, kwargs
        if not self.training:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        clean = self.get_input_embeddings()(input_ids)
        noisy = gaussian_text_view(clean, attention_mask)
        embeddings = torch.cat((clean, noisy), dim=0)
        ids = torch.cat((input_ids, input_ids), dim=0)
        mask = torch.cat((attention_mask, attention_mask), dim=0)
        raw_gaze, gaze_mask = self.base_model.gaze_provider.compute(input_ids, attention_mask)
        gaze = (torch.cat((raw_gaze, raw_gaze), dim=0), torch.cat((gaze_mask, gaze_mask), dim=0))
        packed, fused_mask, positions, pooling = self.base_model._gaze_inputs(ids, mask, embeddings, gaze)
        outputs = self.base_model.backbone(input_ids=None, inputs_embeds=packed, attention_mask=fused_mask,
                                           position_ids=positions, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state
        pooled = hidden[torch.arange(len(hidden), device=hidden.device), pooling]
        predictions = self.base_model.predict_pooled(pooled)
        return {"logits": predictions[:len(input_ids)], "view_predictions": predictions,
                "view_features": pooled.float()}


def save_encoder_base(base_model, output_dir):
    """Save complete unprefixed weights atomically and verify their actual storage dtype."""

    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / ".model.safetensors.tmp"
    state = {name: value.detach().cpu().contiguous() for name, value in base_model.state_dict().items()}
    try:
        save_file(state, str(temporary), metadata={"format": "pt"})
        temporary.replace(output_dir / "model.safetensors")
    finally:
        temporary.unlink(missing_ok=True)
    del state
    if callable(getattr(base_model.config, "save_pretrained", None)):
        base_model.config.save_pretrained(str(output_dir))
    base_model.save_architecture_manifest(output_dir)
    audit = validate_saved_precision(base_model, output_dir / "model.safetensors")
    atomic_write_json(output_dir / "precision_audit.json", audit)
    return audit


def load_encoder_base_weights(base_model, directory):
    """Restore a verified best checkpoint strictly without constructing another encoder."""

    from safetensors.torch import load_model

    validate_saved_precision(base_model, Path(directory) / "model.safetensors")
    load_model(base_model, str(Path(directory) / "model.safetensors"), strict=True, device="cpu")
    audit_model_precision(base_model)


def _training_device(config):
    """Require the requested real single-GPU BF16 profile instead of substituting CPU training."""

    baseline_settings = config.baseline
    if (baseline_settings.backbone_weight_dtype != "bfloat16" or baseline_settings.precision != "bf16"
            or baseline_settings.finetuning_mode != "full" or baseline_settings.gaze_fusion != "prefix-concat"
            or tuple(baseline_settings.gaze_features) != ("TRT",) or baseline_settings.use_cpu
            or baseline_settings.loss != "mse"):
        raise ValueError("Encoder stages require the full BF16 Raw TRT/MSE configuration on CUDA.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required; CPU is not an efficacy fallback.")
    return torch.device("cuda")


def _check_partitions(train_frame, validation_frame, outer_frame):
    """Reject empty or overlapping A/Bval rows while preserving the requested outer text policy."""

    partitions = (train_frame, validation_frame, outer_frame)
    for frame in partitions:
        if frame.empty:
            raise ValueError("A, Bval, and outer partitions must all be nonempty.")
        if frame["index"].astype(str).duplicated().any():
            raise ValueError("Each partition must have unique row identities.")
        values = frame[["valence", "arousal"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("VA labels must be finite values in [0,1].")
    ids = [set(frame["index"].astype(str)) for frame in partitions]
    if any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("A, Bval, and outer row identities must be disjoint.")
    if set(text_groups(train_frame)) & set(text_groups(validation_frame)):
        raise ValueError("A and Bval must be split by normalized text groups.")


def _encoder_metrics(eval_prediction):
    """Expose the six original metrics plus both MSE and mean RMSE on clean Bval."""

    result = trainer_compute_metrics(eval_prediction)
    result["rmse_mean"] = (result["rmse_valence"] + result["rmse_arousal"]) / 2
    return result

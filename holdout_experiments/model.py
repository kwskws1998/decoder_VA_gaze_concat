"""Shared native BF16 TRT encoder with isolated two-view and redistribution arms."""

import json
from pathlib import Path

import torch

from trt_pipeline.core.dataset import load_auto_tokenizer
from trt_pipeline.precision import audit_model_precision, build_bf16_model
from trt_pipeline.two_view import TwoViewVAEncoder

from .config import ARMS, Settings
from .protocol import digest, write_json
from .redistribution import ARCHIVED_SOURCE_SHA256, AsymGaussianRedistributor


class ExperimentModel(TwoViewVAEncoder):
    """Change only the training views or the TRT operator around a shared base model."""

    def __init__(self, base_model, arm, settings):
        super().__init__(base_model)
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        self.arm = arm
        self.redistributor = (
            AsymGaussianRedistributor(settings.sigma_left, settings.sigma_right, settings.min_sigma)
            if arm == "redistribution" else None
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """Use noisy embeddings only in companion training; redistribute before projection."""

        if self.arm == "companion":
            return super().forward(input_ids, attention_mask, labels=labels, **kwargs)
        if self.redistributor is None:
            return {"logits": self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits}
        base = self.base_model
        embeddings = base.get_input_embeddings()(input_ids)
        raw, gaze_mask = base.gaze_provider.compute(input_ids, attention_mask)
        trt = self.redistributor(raw[:, :, 0], gaze_mask)
        gaze = (trt.unsqueeze(-1), gaze_mask)
        packed, mask, positions, pooling = base._gaze_inputs(input_ids, attention_mask, embeddings, gaze)
        outputs = base.backbone(input_ids=None, inputs_embeds=packed, attention_mask=mask,
                                position_ids=positions, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state
        pooled = hidden[torch.arange(len(hidden), device=hidden.device), pooling]
        return {"logits": base.predict_pooled(pooled)}


def build_model(settings, arm, *, tokenizer=None):
    """Construct all arms from the same pinned pretrained backbone and frozen ET2."""

    if tokenizer is None:
        tokenizer = load_auto_tokenizer(settings.model_id, revision=settings.model_revision,
                                        use_fast=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    base = build_bf16_model(
        tokenizer, model_id=settings.model_id, model_revision=settings.model_revision,
        gaze_feature_indices=(3,), et_repo_id=settings.et_model_id,
        et_revision=settings.et_revision, et_filename=settings.et_filename,
        attn_implementation=None,
    )
    return ExperimentModel(base, arm, settings), tokenizer


def precision_proof(model):
    """Audit native BF16 storage and the FP32 learned Gaussian widths."""

    proof = audit_model_precision(model.base_model)
    proof.pop("gradient_dtypes", None)
    if model.redistributor is not None:
        widths = dict(model.redistributor.named_parameters())
        if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in widths.values()):
            raise ValueError("Redistribution widths must be finite FP32 parameters.")
        proof["redistribution"] = {
            "archived_source_sha256": ARCHIVED_SOURCE_SHA256,
            "width_reporting": "FP32 exponential on CPU from exact stored log parameters",
            "sigma_left": float(model.redistributor.log_sigma_left.detach().float().cpu().exp()
                                + model.redistributor.min_sigma),
            "sigma_right": float(model.redistributor.log_sigma_right.detach().float().cpu().exp()
                                 + model.redistributor.min_sigma),
            "parameter_dtypes": {name: str(p.dtype) for name, p in widths.items()},
        }
    return proof


def save_model(model, directory, settings, protocol_sha256):
    """Atomically replace the selected complete safetensors checkpoint and its manifest."""

    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    temporary = directory / "model.safetensors.tmp"
    save_file(state, str(temporary), metadata={"format": "pt"})
    del state
    temporary.replace(directory / "model.safetensors")
    manifest = {
        "schema": 1, "arm": model.arm, "settings": settings.as_dict(),
        "protocol_sha256": protocol_sha256, "weights_sha256": digest(directory / "model.safetensors"),
        "precision": precision_proof(model), "strict_loading": True,
        "reconstruction": model.base_model._reconstruction_config,
    }
    write_json(directory / "model_manifest.json", manifest)
    return manifest


def restore_weights(model, directory, *, expected_sha256):
    """Check every saved key, shape and dtype before strict loading without silent casting."""

    from safetensors import safe_open
    from safetensors.torch import load_model

    weights = Path(directory) / "model.safetensors"
    if digest(weights) != expected_sha256:
        raise ValueError("Selected model SHA-256 changed.")
    state = model.state_dict()
    with safe_open(weights, framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(state):
            raise ValueError("Checkpoint keys differ from the reconstructed model.")
        for name, tensor in state.items():
            stored = saved.get_slice(name)
            expected_dtype = {torch.bfloat16: "BF16", torch.float32: "F32", torch.int64: "I64",
                              torch.int32: "I32", torch.bool: "BOOL"}.get(tensor.dtype)
            if stored.get_shape() != list(tensor.shape) or stored.get_dtype() != expected_dtype:
                raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
    load_model(model, str(weights), strict=True, device="cpu")
    return precision_proof(model)


def load_saved(directory, settings, protocol_sha256, arm):
    """Reconstruct the selected model and require its exact run and precision contract."""

    directory = Path(directory)
    manifest = json.loads((directory / "model_manifest.json").read_text())
    if (manifest.get("schema") != 1 or manifest.get("settings") != settings.as_dict()
            or manifest.get("protocol_sha256") != protocol_sha256 or manifest.get("arm") != arm
            or manifest.get("strict_loading") is not True):
        raise ValueError("Saved model does not belong to this locked protocol and arm.")
    tokenizer = load_auto_tokenizer(directory, local_files_only=True, use_fast=True, trust_remote_code=False)
    model, tokenizer = build_model(settings, arm, tokenizer=tokenizer)
    if model.base_model._reconstruction_config != manifest.get("reconstruction"):
        raise ValueError("Saved base architecture differs from reconstruction.")
    proof = restore_weights(model, directory, expected_sha256=manifest["weights_sha256"])
    if proof != manifest["precision"]:
        raise ValueError("Reloaded model precision differs from its saved proof.")
    return model.eval(), tokenizer, manifest


def settings_from_manifest(directory):
    """Expose checkpoint settings for standalone downstream inference code."""

    manifest = json.loads((Path(directory) / "model_manifest.json").read_text())
    return Settings(**manifest["settings"]), manifest

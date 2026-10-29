"""Preserve schema-6 predictions while exposing frozen features to residual DDPM."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from trt_pipeline.historical import recipe as profile
from trt_pipeline.historical.snapshot import model as original
from trt_pipeline.historical.snapshot.packing import pack_prefix_gaze
from trt_pipeline.core.baseline_contracts import readout_contract, prediction_contract, gaze_relation_contract
from trt_pipeline.core.dataset import load_auto_tokenizer
from trt_pipeline.core.model import EncodedVAFeatures


class HistoricalRawVA(nn.Module):
    """Wrap the original model without reinitializing or promoting any parameters."""

    supports_gradient_checkpointing = True

    def __init__(self, raw):
        super().__init__()
        self.raw = raw
        self.config = raw.config
        self.hidden_size = raw.hidden_size
        self.gaze_fusion = raw.gaze_fusion
        self.readout = readout_contract()
        self.gaze_relation = gaze_relation_contract()
        self.gaze_redistribution = {"method": "none"}
        self.prediction_distribution = prediction_contract()

    @property
    def gaze_provider(self):
        """Expose the original frozen ET2 provider without duplicating its state."""
        return self.raw.gaze_provider

    @property
    def regression_head(self):
        """Expose the intact original mean head."""
        return self.raw.regression_head

    def get_input_embeddings(self):
        """Delegate embedding access for the Hugging Face trainer."""
        return self.raw.get_input_embeddings()

    def gradient_checkpointing_enable(self, **kwargs):
        """Keep the original nonreentrant gradient checkpoint behavior."""
        return self.raw.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Delegate checkpoint disabling to the original model."""
        return self.raw.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """Use the unchanged original forward for Raw training and mean evaluation."""
        return self.raw(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def predict_pooled(self, pooled):
        """Preserve the original head autocast; conversion to FP32 happens after prediction."""
        dtype = next(self.regression_head.parameters()).dtype
        return F.hardsigmoid(self.regression_head(pooled.to(dtype=dtype)))

    def encode(self, input_ids, attention_mask, *, return_aligned_tokens=False):
        """Mirror the original prefix packing and expose its text/TRT memory in one pass."""
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Input IDs and attention mask must have equal rank-2 shapes.")
        embeddings = self.get_input_embeddings()(input_ids)
        gaze, mask = self.gaze_provider.compute(input_ids, attention_mask)
        dtype = next(self.raw.gaze_projector.parameters()).dtype
        projected = self.raw.gaze_projector(gaze.to(device=embeddings.device, dtype=dtype)).to(embeddings.dtype)
        packed = pack_prefix_gaze(embeddings, attention_mask, projected, mask, self.raw.eye_start, self.raw.eye_end)
        hidden = self.raw.backbone(
            input_ids=None, inputs_embeds=packed.inputs_embeds, attention_mask=packed.attention_mask,
            position_ids=packed.position_ids, use_cache=False, return_dict=True,
        ).last_hidden_state
        pooled = hidden[torch.arange(len(hidden), device=hidden.device), packed.pooling_positions]
        if not return_aligned_tokens:
            return EncodedVAFeatures(pooled=pooled)
        lengths = attention_mask.long().sum(1)
        positions = packed.pooling_positions[:, None] + 1 - lengths[:, None]
        positions = positions + torch.arange(input_ids.shape[1], device=hidden.device)[None]
        positions = positions.clamp_max(hidden.shape[1] - 1)
        text = hidden.gather(1, positions[..., None].expand(-1, -1, hidden.shape[-1]))
        text = text.masked_fill(~attention_mask.bool()[..., None], 0)
        return EncodedVAFeatures(pooled=pooled, text_hidden=text, raw_gaze=gaze, gaze_mask=mask)

    def trainable_parameter_summary(self):
        """Retain original counts and report the actual parameter dtype distribution."""
        counts = {}
        for parameter in self.parameters():
            key = str(parameter.dtype).replace("torch.", "")
            counts[key] = counts.get(key, 0) + parameter.numel()
        return {**self.raw.trainable_parameter_summary(), "parameter_elements_by_dtype": counts}

    def save_architecture_manifest(self, directory):
        """Bind the snapshot, all parameter dtypes, and original reconstruction settings."""
        path = self.raw.save_architecture_manifest(directory)
        manifest = json.loads(path.read_text())
        manifest.update({
            "schema_version": 1, "implementation_profile": profile.PROFILE,
            "historical_source": json.loads((profile.ROOT / "snapshot/source.json").read_text()),
            "gaze_alignment": profile.ALIGNMENT, "precision_contract": profile.PRECISION,
            "readout": self.readout, "gaze_relation": self.gaze_relation,
            "gaze_redistribution": self.gaze_redistribution, "prediction_distribution": self.prediction_distribution,
            "parameter_dtypes": {name: str(p.dtype).replace("torch.", "") for name, p in self.named_parameters()},
        })
        manifest["state_dict"]["scope"] = "complete HistoricalRawVA state_dict; raw.* schema6 model"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return path


def build_model(tokenizer, **kwargs):
    """Build from the original implementation without changing initialization RNG order."""
    return HistoricalRawVA(original.build_qwen_va_model(tokenizer, **kwargs))


def reconstruct_model(manifest, *, tokenizer, et_cache_size=70000):
    """Reconstruct only the declared historical contract, independently of modern defaults."""
    profile.verify_runtime()
    expected = {
        "implementation_profile": profile.PROFILE, "schema_version": 1,
        "gaze_alignment": profile.ALIGNMENT, "precision_contract": profile.PRECISION,
        "historical_source": json.loads((profile.ROOT / "snapshot/source.json").read_text()),
        "gaze_fusion": "prefix-concat", "readout": readout_contract(),
        "gaze_relation": gaze_relation_contract(), "gaze_redistribution": {"method": "none"},
        "prediction_distribution": prediction_contract(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Historical reconstruction contract mismatch: {key}")
    r = manifest["reconstruction"]
    if r["finetuning_mode"] != "full" or r["et_feature_indices"] != [3] or r["backbone_dtype_at_construction"] != "bfloat16":
        raise ValueError("Historical source must be full BF16 Raw TRT.")
    model = build_model(
        tokenizer, model_id=r["decoder_model_id"], model_revision=r["decoder_revision"],
        finetuning_mode="full", gaze_fusion="prefix-concat", gaze_feature_indices=(3,),
        et_repo_id=r["et_repo_id"], et_revision=r["et_revision"], et_filename=r["et_filename"],
        et_cache_size=et_cache_size, dtype=torch.bfloat16, attn_implementation=r["attn_implementation"],
        gaze_projection_dim=r["gaze_projection_dim"], gaze_projection_dropout=tuple(r["gaze_projection_dropout"]),
        classifier_dropout=r["classifier_dropout"],
    )
    actual = {name: str(p.dtype).replace("torch.", "") for name, p in model.named_parameters()}
    if actual != manifest["parameter_dtypes"]:
        raise ValueError("Historical reconstructed parameter dtypes changed; refusing silent conversion.")
    return model, tokenizer


def load_model(directory, *, et_cache_size=70000, tokenizer=None):
    """Strictly restore BF16/FP32 tensors without converting stored weights."""
    from safetensors import safe_open
    from safetensors.torch import load_model as load_weights

    directory = Path(directory)
    manifest = json.loads((directory / original.ARCHITECTURE_MANIFEST_FILENAME).read_text())
    if tokenizer is None:
        tokenizer = load_auto_tokenizer(directory, use_fast=True, trust_remote_code=False, local_files_only=True)
    tokenizer.padding_side = "right"
    model, tokenizer = reconstruct_model(manifest, tokenizer=tokenizer, et_cache_size=et_cache_size)
    path = directory / "model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as saved:
        for name, p in model.named_parameters():
            if saved.get_slice(name).get_dtype() != {torch.float32: "F32", torch.bfloat16: "BF16"}[p.dtype]:
                raise ValueError(f"Historical saved parameter dtype mismatch: {name}")
    load_weights(model, path, strict=True, device="cpu")
    return model.eval(), tokenizer

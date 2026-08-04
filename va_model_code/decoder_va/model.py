"""Qwen decoder backbone with frozen ET2 gaze-prefix fusion for VA regression."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from .gaze import (
    DEFAULT_ET2_FILENAME,
    DEFAULT_ET2_REPO_ID,
    DEFAULT_ET2_REVISION,
    ET2GazeProvider,
)
from .packing import pack_prefix_gaze


DEFAULT_DECODER_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_DECODER_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
GAZE_FUSIONS = ("none", "prefix-concat")
GAZE_PREFIX_ORDER = "eye_start, compact_trt_gaze, eye_end, text"
GAZE_PREFIX_POOLING = "last_valid_text_token_after_gaze_prefix"
ARCHITECTURE_MANIFEST_FILENAME = "decoder_va_architecture.json"
ARCHITECTURE_MANIFEST_VERSION = 3
SAFE_WEIGHTS_FILENAME = "model.safetensors"


def _hidden_size(config: Any) -> int:
    """Read a text hidden size from ordinary or nested multimodal configs."""

    value = getattr(config, "hidden_size", None)
    if value is None:
        text_config = getattr(config, "text_config", None)
        value = getattr(text_config, "hidden_size", None)
    if value is None or int(value) <= 0:
        raise ValueError("The decoder config does not expose a positive hidden_size.")
    return int(value)


def _normalize_gaze_fusion(value: str) -> str:
    """Normalize supported gaze-fusion spellings."""

    aliases = {
        "prefix": "prefix-concat",
        "concat": "prefix-concat",
        "gaze-concat": "prefix-concat",
        "off": "none",
    }
    requested = str(value).strip().lower()
    if requested in {"postfix", "postfix-concat"}:
        raise ValueError(
            "Postfix gaze fusion is no longer supported. Use 'prefix-concat'; "
            "old postfix checkpoints are architecture-incompatible."
        )
    normalized = aliases.get(requested, requested)
    if normalized not in GAZE_FUSIONS:
        raise ValueError(f"Unknown gaze fusion {value!r}; choose one of {GAZE_FUSIONS}.")
    return normalized


class DecoderVARegressor(nn.Module):
    """Pool a causal decoder at the final valid text token."""

    supports_gradient_checkpointing = True

    def __init__(
        self,
        backbone: nn.Module,
        *,
        gaze_provider: ET2GazeProvider | Any | None = None,
        gaze_fusion: str = "prefix-concat",
        output_dim: int = 4,
        gaze_projection_dim: int = 128,
        gaze_projection_dropout: tuple[float, float] = (0.1, 0.3),
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_dim not in (2, 4):
            raise ValueError("output_dim must be 2 for point regression or 4 for uncertainty.")
        self.backbone = backbone
        if not hasattr(backbone, "config"):
            raise TypeError("backbone must expose a Transformers-compatible config.")
        backbone_config = backbone.config
        try:
            _hidden_size(backbone_config)
        except ValueError:
            base_model_getter = getattr(backbone, "get_base_model", None)
            base_model = base_model_getter() if callable(base_model_getter) else None
            if base_model is None or not hasattr(base_model, "config"):
                raise
            backbone_config = base_model.config
        self.config = copy.deepcopy(backbone_config)
        self.hidden_size = _hidden_size(self.config)
        self.output_dim = int(output_dim)
        self.gaze_fusion = _normalize_gaze_fusion(gaze_fusion)
        self.gaze_provider = gaze_provider
        self.gaze_projection_dim = int(gaze_projection_dim)
        self.gaze_projection_dropout = tuple(
            float(value) for value in gaze_projection_dropout
        )
        self.classifier_dropout = float(classifier_dropout)
        self._reconstruction_config: dict[str, Any] | None = None
        if self.gaze_fusion == "prefix-concat" and self.gaze_provider is None:
            raise ValueError("prefix-concat requires an ET2 gaze provider.")
        if self.gaze_fusion == "none" and self.gaze_provider is not None:
            raise ValueError("gaze_provider must be None when gaze_fusion='none'.")
        if self.gaze_projection_dim <= 0:
            raise ValueError("gaze_projection_dim must be positive.")
        if len(self.gaze_projection_dropout) != 2:
            raise ValueError("gaze_projection_dropout must contain exactly two values.")
        if any(not 0.0 <= value < 1.0 for value in self.gaze_projection_dropout):
            raise ValueError("gaze projection dropout values must be in [0, 1).")
        if not 0.0 <= self.classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1).")

        if self.gaze_fusion == "prefix-concat":
            first_dropout, second_dropout = self.gaze_projection_dropout
            self.gaze_projector = nn.Sequential(
                nn.Linear(1, self.gaze_projection_dim),
                nn.LayerNorm(self.gaze_projection_dim),
                nn.GELU(),
                nn.Dropout(first_dropout),
                nn.Linear(self.gaze_projection_dim, self.hidden_size),
                nn.Dropout(second_dropout),
                nn.LayerNorm(self.hidden_size),
            )
            self.eye_start = nn.Parameter(torch.zeros(self.hidden_size))
            self.eye_end = nn.Parameter(torch.zeros(self.hidden_size))
        else:
            self.gaze_projector = None
            self.register_parameter("eye_start", None)
            self.register_parameter("eye_end", None)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.classifier_dropout),
            nn.Linear(self.hidden_size, self.output_dim),
        )
        self.config.num_labels = self.output_dim
        self.config.problem_type = "regression"
        self.config.gaze_fusion = self.gaze_fusion
        self.config.gaze_concat_order = (
            GAZE_PREFIX_ORDER if self.gaze_fusion == "prefix-concat" else None
        )
        self.config.pooling_position = (
            GAZE_PREFIX_POOLING
            if self.gaze_fusion == "prefix-concat"
            else "last_valid_text_token"
        )
        self.config.decoder_va_architecture_version = ARCHITECTURE_MANIFEST_VERSION
        self.config.va_output_names = (
            ["valence", "arousal"]
            if self.output_dim == 2
            else [
                "valence_mu",
                "arousal_mu",
                "valence_logvar",
                "arousal_logvar",
            ]
        )
        if hasattr(backbone_config, "use_cache"):
            backbone_config.use_cache = False

    def get_input_embeddings(self):
        """Expose decoder token embeddings to Trainer and the fusion path."""

        getter = getattr(self.backbone, "get_input_embeddings", None)
        if callable(getter):
            return getter()
        base_model_getter = getattr(self.backbone, "get_base_model", None)
        base_model = base_model_getter() if callable(base_model_getter) else None
        if base_model is None or not hasattr(base_model, "get_input_embeddings"):
            raise AttributeError("The decoder backbone does not expose token embeddings.")
        return base_model.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate checkpointing to the decoder and make embedded inputs differentiable."""

        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        method = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if not callable(method):
            raise RuntimeError("The selected decoder does not support gradient checkpointing.")
        try:
            method(gradient_checkpointing_kwargs=kwargs)
        except TypeError:
            method()
        input_grad_method = getattr(self.backbone, "enable_input_require_grads", None)
        if callable(input_grad_method):
            input_grad_method()

    def gradient_checkpointing_disable(self):
        """Disable decoder gradient checkpointing."""

        method = getattr(self.backbone, "gradient_checkpointing_disable", None)
        if callable(method):
            method()

    def _text_only_inputs(
        self,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build causal positions and explicit last-valid-token pooling indices."""

        if attention_mask.dtype != torch.bool:
            is_binary = torch.logical_or(attention_mask == 0, attention_mask == 1)
            if not bool(is_binary.all().item()):
                raise ValueError("attention_mask must contain only zero and one values.")
        mask_bool = attention_mask.to(dtype=torch.bool)
        lengths = mask_bool.to(dtype=torch.long).sum(dim=1)
        if bool((lengths <= 0).any().item()):
            raise ValueError("Every example must contain at least one active text token.")
        column_positions = torch.arange(
            attention_mask.shape[1],
            device=attention_mask.device,
        ).unsqueeze(0)
        if not torch.equal(mask_bool, column_positions < lengths.unsqueeze(1)):
            raise ValueError("attention_mask must use contiguous right padding.")
        positions = mask_bool.to(dtype=torch.long).cumsum(dim=1) - 1
        positions = positions.masked_fill(~mask_bool, 0)
        return text_embeddings, attention_mask, positions, lengths - 1

    def _gaze_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer frozen TRT and prefix its compact projection before each text."""

        raw_gaze, gaze_mask = self.gaze_provider.compute(input_ids, attention_mask)
        projector_dtype = next(self.gaze_projector.parameters()).dtype
        projected_gaze = self.gaze_projector(
            raw_gaze.to(device=text_embeddings.device, dtype=projector_dtype)
        ).to(dtype=text_embeddings.dtype)
        packed = pack_prefix_gaze(
            text_embeddings=text_embeddings,
            text_attention_mask=attention_mask,
            gaze_embeddings=projected_gaze,
            gaze_mask=gaze_mask,
            eye_start=self.eye_start,
            eye_end=self.eye_end,
        )
        return (
            packed.inputs_embeds,
            packed.attention_mask,
            packed.position_ids,
            packed.pooling_positions,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Return bounded VA means followed by optional unconstrained log variances."""

        del labels, kwargs
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must both be rank-2.")
        if tuple(input_ids.shape) != tuple(attention_mask.shape):
            raise ValueError("input_ids and attention_mask must have identical shape.")
        if input_ids.device != attention_mask.device:
            raise ValueError("input_ids and attention_mask must be on the same device.")

        text_embeddings = self.get_input_embeddings()(input_ids)
        if self.gaze_fusion == "prefix-concat":
            inputs_embeds, fused_mask, position_ids, pooling_positions = self._gaze_inputs(
                input_ids,
                attention_mask,
                text_embeddings,
            )
        else:
            inputs_embeds, fused_mask, position_ids, pooling_positions = (
                self._text_only_inputs(text_embeddings, attention_mask)
            )

        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=fused_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        row_indices = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[row_indices, pooling_positions]
        head_dtype = next(self.regression_head.parameters()).dtype
        raw_logits = self.regression_head(pooled.to(dtype=head_dtype))
        means = torch.sigmoid(raw_logits[:, :2])
        logits = (
            means
            if self.output_dim == 2
            else torch.cat((means, raw_logits[:, 2:]), dim=-1)
        )
        return SequenceClassifierOutput(logits=logits)

    def trainable_parameter_summary(self) -> dict[str, int | float]:
        """Return trainable and total parameter counts for run manifests."""

        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "trainable_fraction": float(trainable / total) if total else 0.0,
        }

    def save_architecture_manifest(self, output_dir: str | Path) -> Path:
        """Persist a complete, strict reconstruction contract beside model weights."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if self._reconstruction_config is None:
            raise RuntimeError(
                "A reloadable manifest is available only for models constructed "
                "with build_qwen_va_model()."
            )
        manifest_path = output_path / ARCHITECTURE_MANIFEST_FILENAME
        payload = {
            "schema_version": ARCHITECTURE_MANIFEST_VERSION,
            "decoder_model_id": self._reconstruction_config["decoder_model_id"],
            "decoder_model_type": getattr(self.config, "model_type", None),
            "decoder_commit": self._reconstruction_config["decoder_revision"],
            "gaze_fusion": self.gaze_fusion,
            "gaze_feature": "TRT" if self.gaze_fusion != "none" else None,
            "et_model": (
                {
                    "repo_id": self.gaze_provider.repo_id,
                    "revision": self.gaze_provider.revision,
                    "filename": self.gaze_provider.filename,
                    "feature_index": self.gaze_provider.feature_index,
                }
                if self.gaze_provider is not None
                else None
            ),
            "gaze_concat_order": (
                GAZE_PREFIX_ORDER if self.gaze_fusion == "prefix-concat" else None
            ),
            "pooling_position": (
                GAZE_PREFIX_POOLING
                if self.gaze_fusion == "prefix-concat"
                else "last_valid_text_token"
            ),
            "output_names": list(self.config.va_output_names),
            "reconstruction": copy.deepcopy(self._reconstruction_config),
            "state_dict": {
                "filename": SAFE_WEIGHTS_FILENAME,
                "format": "safetensors",
                "scope": "complete DecoderVARegressor state_dict; ET2 remains external and frozen",
                "strict_loading": True,
            },
            **self.trainable_parameter_summary(),
        }
        with open(manifest_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        return manifest_path


def load_qwen_backbone_with_lora(
    model_id: str = DEFAULT_DECODER_MODEL_ID,
    *,
    revision: str = DEFAULT_DECODER_REVISION,
    dtype: torch.dtype | str = torch.bfloat16,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    attn_implementation: str | None = None,
):
    """Load the text-only Qwen causal backbone and apply LoRA to every linear layer."""

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise ImportError(
            "Qwen LoRA training requires recent transformers and peft; "
            "install the repository-root requirements.txt."
        ) from exc

    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": False,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    try:
        causal_lm = Qwen3_5ForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (KeyError, ValueError, ImportError) as exc:
        raise RuntimeError(
            "The installed Transformers build cannot load Qwen3.5. "
            "Install the versions in the repository-root requirements.txt."
        ) from exc
    if not hasattr(causal_lm, "model"):
        raise TypeError(f"{model_id} does not expose a causal text backbone at '.model'.")
    backbone = causal_lm.model
    del causal_lm
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False

    lora_config = LoraConfig(
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        target_modules="all-linear",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(backbone, lora_config)


def build_qwen_va_model(
    tokenizer,
    *,
    model_id: str = DEFAULT_DECODER_MODEL_ID,
    model_revision: str = DEFAULT_DECODER_REVISION,
    gaze_fusion: str = "prefix-concat",
    et_repo_id: str = DEFAULT_ET2_REPO_ID,
    et_revision: str = DEFAULT_ET2_REVISION,
    et_filename: str = DEFAULT_ET2_FILENAME,
    et_cache_size: int = 70000,
    output_dim: int = 4,
    dtype: torch.dtype | str = torch.bfloat16,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gaze_projection_dim: int = 128,
    gaze_projection_dropout: tuple[float, float] = (0.1, 0.3),
    classifier_dropout: float = 0.1,
    attn_implementation: str | None = None,
) -> DecoderVARegressor:
    """Construct the selected Qwen/LoRA VA model and optional pinned ET2 provider."""

    normalized_fusion = _normalize_gaze_fusion(gaze_fusion)
    backbone = load_qwen_backbone_with_lora(
        model_id,
        revision=model_revision,
        dtype=dtype,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        attn_implementation=attn_implementation,
    )
    gaze_provider = (
        ET2GazeProvider(
            tokenizer=tokenizer,
            repo_id=et_repo_id,
            revision=et_revision,
            filename=et_filename,
            feature_index=3,
            cache_size=et_cache_size,
        )
        if normalized_fusion == "prefix-concat"
        else None
    )
    model = DecoderVARegressor(
        backbone,
        gaze_provider=gaze_provider,
        gaze_fusion=normalized_fusion,
        output_dim=output_dim,
        gaze_projection_dim=gaze_projection_dim,
        gaze_projection_dropout=gaze_projection_dropout,
        classifier_dropout=classifier_dropout,
    )
    model._reconstruction_config = {
        "decoder_model_id": str(model_id),
        "decoder_revision": str(model_revision),
        "gaze_fusion": normalized_fusion,
        "et_repo_id": str(et_repo_id),
        "et_revision": str(et_revision),
        "et_filename": str(et_filename),
        "et_feature_index": 3,
        "et_cache_size": int(et_cache_size),
        "output_dim": int(output_dim),
        "lora_rank": int(lora_rank),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_target_modules": "all-linear",
        "lora_task_type": "FEATURE_EXTRACTION",
        "gaze_projection_dim": int(gaze_projection_dim),
        "gaze_projection_dropout": [
            float(value) for value in gaze_projection_dropout
        ],
        "classifier_dropout": float(classifier_dropout),
        "attn_implementation": attn_implementation,
        "backbone_dtype_at_construction": str(dtype).replace("torch.", ""),
    }
    return model


def load_saved_decoder_va_model(
    model_dir: str | Path,
    *,
    tokenizer=None,
    dtype: torch.dtype | str = "auto",
    et_cache_size: int | None = None,
) -> tuple[DecoderVARegressor, Any]:
    """Strictly reconstruct a saved final model and its local tokenizer."""

    model_path = Path(model_dir)
    manifest_path = model_path / ARCHITECTURE_MANIFEST_FILENAME
    weights_path = model_path / SAFE_WEIGHTS_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Architecture manifest not found: {manifest_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Safe model weights not found: {weights_path}")

    with open(manifest_path, "r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if manifest.get("schema_version") != ARCHITECTURE_MANIFEST_VERSION:
        raise ValueError(
            "Unsupported decoder VA architecture manifest version: "
            f"{manifest.get('schema_version')!r}"
        )
    state_contract = manifest.get("state_dict")
    if not isinstance(state_contract, dict):
        raise ValueError("Architecture manifest is missing the state_dict contract.")
    if (
        state_contract.get("filename") != SAFE_WEIGHTS_FILENAME
        or state_contract.get("format") != "safetensors"
        or state_contract.get("strict_loading") is not True
    ):
        raise ValueError("Architecture manifest does not declare the expected safe state dict.")

    reconstruction = manifest.get("reconstruction")
    if not isinstance(reconstruction, dict):
        raise ValueError("Architecture manifest is missing reconstruction metadata.")
    required = {
        "decoder_model_id",
        "decoder_revision",
        "gaze_fusion",
        "et_repo_id",
        "et_revision",
        "et_filename",
        "et_feature_index",
        "et_cache_size",
        "output_dim",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "lora_task_type",
        "gaze_projection_dim",
        "gaze_projection_dropout",
        "classifier_dropout",
        "attn_implementation",
    }
    missing = sorted(required.difference(reconstruction))
    if missing:
        raise ValueError(
            "Architecture manifest is missing reconstruction field(s): "
            + ", ".join(missing)
        )
    if reconstruction["et_feature_index"] != 3:
        raise ValueError("Only raw TRT at ET2 feature index 3 is supported.")
    if reconstruction["lora_target_modules"] != "all-linear":
        raise ValueError("Saved model does not use the supported all-linear LoRA contract.")
    if reconstruction["lora_task_type"] != "FEATURE_EXTRACTION":
        raise ValueError("Saved model does not use the supported LoRA task type.")
    saved_fusion = reconstruction["gaze_fusion"]
    if saved_fusion not in GAZE_FUSIONS:
        raise ValueError(
            f"Saved model declares unsupported gaze fusion {saved_fusion!r}."
        )
    if manifest.get("gaze_fusion") != saved_fusion:
        raise ValueError(
            "Architecture manifest gaze_fusion disagrees with reconstruction metadata."
        )
    expected_order = GAZE_PREFIX_ORDER if saved_fusion == "prefix-concat" else None
    expected_pooling = (
        GAZE_PREFIX_POOLING
        if saved_fusion == "prefix-concat"
        else "last_valid_text_token"
    )
    if manifest.get("gaze_concat_order") != expected_order:
        raise ValueError("Architecture manifest declares an incompatible gaze concat order.")
    if manifest.get("pooling_position") != expected_pooling:
        raise ValueError("Architecture manifest declares an incompatible pooling position.")

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=False,
            local_files_only=True,
        )
    tokenizer.padding_side = "right"

    effective_cache_size = (
        int(reconstruction["et_cache_size"])
        if et_cache_size is None
        else int(et_cache_size)
    )
    model = build_qwen_va_model(
        tokenizer,
        model_id=str(reconstruction["decoder_model_id"]),
        model_revision=str(reconstruction["decoder_revision"]),
        gaze_fusion=str(reconstruction["gaze_fusion"]),
        et_repo_id=str(reconstruction["et_repo_id"]),
        et_revision=str(reconstruction["et_revision"]),
        et_filename=str(reconstruction["et_filename"]),
        et_cache_size=effective_cache_size,
        output_dim=int(reconstruction["output_dim"]),
        dtype=dtype,
        lora_rank=int(reconstruction["lora_rank"]),
        lora_alpha=int(reconstruction["lora_alpha"]),
        lora_dropout=float(reconstruction["lora_dropout"]),
        gaze_projection_dim=int(reconstruction["gaze_projection_dim"]),
        gaze_projection_dropout=tuple(
            float(value) for value in reconstruction["gaze_projection_dropout"]
        ),
        classifier_dropout=float(reconstruction["classifier_dropout"]),
        attn_implementation=reconstruction["attn_implementation"],
    )

    from safetensors.torch import load_model

    load_model(model, weights_path, strict=True, device="cpu")
    model.eval()
    return model, tokenizer

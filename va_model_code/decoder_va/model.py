"""Qwen decoder backbone with frozen ET2 gaze-prefix fusion for VA regression."""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from .alignment import GAZE_ALIGNMENT_CONTRACT
from .gaze import (
    DEFAULT_ET2_FILENAME,
    DEFAULT_ET2_REPO_ID,
    DEFAULT_ET2_REVISION,
    ET2_FEATURE_NAMES,
    ET2GazeProvider,
    et2_feature_names_from_indices,
    et2_features_used_from_indices,
    normalize_et2_feature_indices,
)
from .packing import PackedCausalInputs, append_readout_token, pack_prefix_gaze
from .uncertainty import (
    prediction_contract, validate_prediction_contract, saved_prediction_contract,
    make_variance_head, log_variance_from_raw, validate_saved_variance_head,
)
from .readout import (
    readout_contract, validate_readout_contract, saved_readout_contract,
    readout_pooling_position, readout_concat_order,
)
from .gaze_relation import gaze_relation_contract, validate_gaze_relation_contract
from .redistribution import (
    GazeRedistributor,
    validate_redistribution_contract,
    validate_redistribution_state_file,
)


DEFAULT_DECODER_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_DECODER_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_GAZE_PROJECTION_DIM = 128
DEFAULT_GAZE_PROJECTION_DROPOUT = (0.1, 0.3)
DEFAULT_CLASSIFIER_DROPOUT = 0.1
FINETUNING_MODES = ("lora", "full")
GAZE_FUSIONS = ("none", "prefix-concat")
GAZE_PREFIX_ORDER = "eye_start, compact_selected_gaze, eye_end, text"
GAZE_PREFIX_POOLING = "last_valid_text_token_after_gaze_prefix"
OUTPUT_ACTIVATION = "hard_sigmoid"
ARCHITECTURE_MANIFEST_FILENAME = "decoder_va_architecture.json"
ARCHITECTURE_MANIFEST_VERSION = 11
PRE_UNCERTAINTY_MANIFEST_VERSION = 10
PRE_READOUT_MANIFEST_VERSION = 9
PRE_RELATION_MANIFEST_VERSION = 8
PRE_PRECISION_MANIFEST_VERSION = 7
PRECISION_CONTRACT = {
    "version": 1,
    "trainable_parameter_dtype": "float32",
    "regression_head_compute_dtype": "float32",
    "output_dtype": "float32",
    "mixed_precision": "autocast",
}
PRE_REDISTRIBUTION_MANIFEST_VERSION = 6
LEGACY_LORA_MANIFEST_VERSION = 5
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


def _normalize_finetuning_mode(value: str) -> str:
    """Normalize the explicit Qwen fine-tuning strategy."""

    normalized = str(value).strip().lower()
    if normalized not in FINETUNING_MODES:
        raise ValueError(
            f"Unknown fine-tuning mode {value!r}; choose one of {FINETUNING_MODES}."
        )
    return normalized


def backbone_parameter_dtype(dtype: torch.dtype | str, finetuning_mode: str) -> torch.dtype:
    """Keep all full-tuning parameters in FP32 and allow lower precision frozen bases."""

    mode = _normalize_finetuning_mode(finetuning_mode)
    normalized = str(dtype).replace("torch.", "").lower()
    supported = {
        "auto": torch.float32,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in supported:
        raise ValueError(f"Unsupported backbone dtype {dtype!r}; use float32, float16, bfloat16 or auto.")
    return torch.float32 if mode == "full" else supported[normalized]


def _promote_trainable_parameters(module: nn.Module) -> None:
    """Preserve small optimizer updates by storing every trainable weight in FP32."""

    for parameter in module.parameters():
        if parameter.requires_grad and parameter.is_floating_point():
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()


def validate_saved_trainable_precision(model: nn.Module, weights_path: str | Path) -> None:
    """Check on-disk parameter precision before final-model or Trainer checkpoint loading."""

    from safetensors import safe_open

    path = Path(weights_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Precision-safe loading requires a regular model.safetensors file: {path}.")
    with safe_open(path, framework="pt", device="cpu") as saved_weights:
        available = set(saved_weights.keys())
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name in available:
                if saved_weights.get_slice(name).get_dtype() != "F32":
                    raise ValueError(f"Saved trainable parameter {name!r} must be FP32.")


def validate_saved_gaze_relation(model: nn.Module, weights_path: str | Path) -> None:
    """Reject absent, unexpected, malformed, or nonfinite relation checkpoint weights."""

    from safetensors import safe_open

    expected = {
        name: parameter for name, parameter in model.named_parameters()
        if name.endswith("gaze_relative_rotation.alpha")
    }
    with safe_open(weights_path, framework="pt", device="cpu") as saved:
        available = {name for name in saved.keys() if name.endswith("gaze_relative_rotation.alpha")}
        if available != set(expected):
            raise ValueError("Saved gaze_relation parameter keys disagree with the model contract.")
        for name, parameter in expected.items():
            value = saved.get_tensor(name)
            if value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError(f"Saved gaze_relation parameter {name!r} has incompatible shape or dtype.")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"Saved gaze_relation parameter {name!r} must be finite.")


def validate_saved_readout(model: nn.Module, weights_path: str | Path) -> None:
    """Reject missing, extra, malformed, or nonfinite learned readout checkpoint values."""

    from safetensors import safe_open

    expected = {
        name: parameter for name, parameter in model.named_parameters()
        if name == "readout_token" or name.endswith(".readout_token")
    }
    with safe_open(weights_path, framework="pt", device="cpu") as saved:
        available = {
            name for name in saved.keys()
            if name == "readout_token" or name.endswith(".readout_token")
        }
        if available != set(expected):
            raise ValueError("Saved readout parameter keys disagree with the model contract.")
        for name, parameter in expected.items():
            value = saved.get_tensor(name)
            if value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError(f"Saved readout parameter {name!r} has incompatible shape or dtype.")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"Saved readout parameter {name!r} must be finite.")


def validate_precision_contract(value: Any, *, context: str) -> dict[str, Any]:
    """Reject saved semantics that predate FP32 weights and FP32 regression output."""

    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value != PRECISION_CONTRACT
    ):
        raise ValueError(
            f"{context}: missing or incompatible precision_contract. "
            "Start a new run; evaluate historical weights with their original code."
        )
    return copy.deepcopy(PRECISION_CONTRACT)


def validate_gaze_alignment_contract(
    value: Any,
    *,
    gaze_fusion: str,
    context: str,
) -> dict[str, Any] | None:
    """Reject gaze weights trained with an unknown or incompatible alignment."""

    if gaze_fusion == "none":
        if value is not None:
            raise ValueError(f"{context}: text-only models must not declare gaze_alignment.")
        return None
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value != GAZE_ALIGNMENT_CONTRACT
    ):
        raise ValueError(
            f"{context}: missing or incompatible gaze_alignment contract. "
            "Old gaze checkpoints cannot be loaded or resumed with the corrected "
            "ET2-to-token alignment. Start a new training run with the current "
            "alignment; evaluate historical weights using their original code."
        )
    return copy.deepcopy(GAZE_ALIGNMENT_CONTRACT)


def validate_gaze_relation_setup(
    value: Mapping | None, *, gaze_fusion: str, feature_indices: Sequence[int],
    gaze_redistribution: Mapping, attn_implementation: str | None = None,
    readout: Mapping | None = None,
) -> dict[str, Any]:
    """Require raw TRT prefix input and eager attention for the relation experiment."""

    relation = validate_gaze_relation_contract(
        gaze_relation_contract() if value is None else value
    )
    pooling = validate_readout_contract(readout_contract() if readout is None else readout)
    if relation["method"] == "trt-readout-rotation" and pooling["mode"] != "learned-token":
        raise ValueError("trt-readout-rotation requires learned-token readout.")
    if relation["method"] == "trt-relative-rotation" and pooling["mode"] != "last-token":
        raise ValueError("Legacy trt-relative-rotation requires last-token readout; use trt-readout-rotation.")
    if relation["method"] != "none":
        if gaze_fusion != "prefix-concat" or 3 not in feature_indices:
            raise ValueError("TRT gaze_relation requires prefix-concat with the TRT feature.")
        if gaze_redistribution["method"] != "none":
            raise ValueError("gaze_relation requires raw TRT; disable gaze_redistribution.")
        if attn_implementation not in (None, "eager"):
            raise ValueError("gaze_relation currently requires attn_implementation='eager'.")
    return relation


class DecoderVARegressor(nn.Module):
    """Pool a causal decoder at the final valid text token or an explicit learned readout."""

    supports_gradient_checkpointing = True

    def __init__(
        self,
        backbone: nn.Module,
        *,
        gaze_provider: ET2GazeProvider | Any | None = None,
        gaze_fusion: str = "prefix-concat",
        gaze_redistribution: Mapping | None = None,
        gaze_relation: Mapping | None = None,
        readout: Mapping | None = None,
        prediction_distribution: Mapping | None = None,
        gaze_projection_dim: int = DEFAULT_GAZE_PROJECTION_DIM,
        gaze_projection_dropout: tuple[
            float,
            float,
        ] = DEFAULT_GAZE_PROJECTION_DROPOUT,
        classifier_dropout: float = DEFAULT_CLASSIFIER_DROPOUT,
    ) -> None:
        super().__init__()
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
        self.output_dim = 2
        self.prediction_distribution = validate_prediction_contract(
            prediction_contract() if prediction_distribution is None else prediction_distribution
        )
        self.gaze_fusion = _normalize_gaze_fusion(gaze_fusion)
        self.readout = validate_readout_contract(readout_contract() if readout is None else readout)
        self.gaze_alignment = (
            copy.deepcopy(GAZE_ALIGNMENT_CONTRACT)
            if self.gaze_fusion == "prefix-concat"
            else None
        )
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
        provider_indices = getattr(self.gaze_provider, "feature_indices", None)
        self.gaze_redistribution = validate_redistribution_contract(
            {"method": "none"} if gaze_redistribution is None else gaze_redistribution,
            gaze_fusion=self.gaze_fusion,
            feature_indices=() if provider_indices is None else provider_indices,
        )
        self.gaze_redistributor = (
            GazeRedistributor(self.gaze_redistribution, provider_indices)
            if self.gaze_redistribution["method"] != "none"
            else None
        )
        self.gaze_relation = validate_gaze_relation_setup(
            gaze_relation, gaze_fusion=self.gaze_fusion,
            feature_indices=() if provider_indices is None else provider_indices,
            gaze_redistribution=self.gaze_redistribution,
            readout=self.readout,
        )
        self.gaze_relation_layers = []
        if self.gaze_relation["method"] != "none":
            from .gaze_attention import install_gaze_relative_attention

            self.gaze_relation_layers = install_gaze_relative_attention(
                self.backbone, scale=self.gaze_relation["scale"],
            )
        if self.gaze_projection_dim <= 0:
            raise ValueError("gaze_projection_dim must be positive.")
        if len(self.gaze_projection_dropout) != 2:
            raise ValueError("gaze_projection_dropout must contain exactly two values.")
        if any(not 0.0 <= value < 1.0 for value in self.gaze_projection_dropout):
            raise ValueError("gaze projection dropout values must be in [0, 1).")
        if not 0.0 <= self.classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1).")

        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.classifier_dropout),
            nn.Linear(self.hidden_size, self.output_dim),
        ).float()
        if self.gaze_fusion == "prefix-concat":
            provider_feature_indices = getattr(
                self.gaze_provider,
                "feature_indices",
                None,
            )
            self.gaze_feature_count = (
                len(provider_feature_indices)
                if provider_feature_indices is not None
                else 1
            )
            if self.gaze_feature_count <= 0:
                raise ValueError("The gaze provider must select at least one feature.")
            first_dropout, second_dropout = self.gaze_projection_dropout
            self.gaze_projector = nn.Sequential(
                nn.Linear(self.gaze_feature_count, self.gaze_projection_dim),
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
            self.gaze_feature_count = 0
            self.gaze_projector = None
            self.register_parameter("eye_start", None)
            self.register_parameter("eye_end", None)
        if self.readout["mode"] == "learned-token":
            self.readout_token = nn.Parameter(torch.empty(self.hidden_size, dtype=torch.float32))
            nn.init.normal_(self.readout_token, mean=0.0, std=0.02)
        else:
            self.register_parameter("readout_token", None)
        self.variance_head = (
            make_variance_head(self.hidden_size)
            if self.prediction_distribution["family"] == "diagonal_gaussian" else None
        )
        self.config.prediction_distribution = copy.deepcopy(self.prediction_distribution)
        self.config.readout = copy.deepcopy(self.readout)
        self.config.num_labels = self.output_dim
        self.config.problem_type = "regression"
        self.config.gaze_fusion = self.gaze_fusion
        self.config.gaze_alignment = copy.deepcopy(self.gaze_alignment)
        self.config.gaze_redistribution = copy.deepcopy(self.gaze_redistribution)
        self.config.gaze_relation = copy.deepcopy(self.gaze_relation)
        self.config.gaze_feature_count = self.gaze_feature_count
        self.config.gaze_concat_order = readout_concat_order(self.gaze_fusion, self.readout)
        self.config.pooling_position = readout_pooling_position(self.gaze_fusion, self.readout)
        self.config.decoder_va_architecture_version = ARCHITECTURE_MANIFEST_VERSION
        self.config.va_output_names = ["valence", "arousal"]
        self.config.va_output_activation = OUTPUT_ACTIVATION
        if hasattr(backbone_config, "use_cache"):
            backbone_config.use_cache = False
        _promote_trainable_parameters(self)

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

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs=None,
        every_n_layers: int = 1,
    ):
        """Delegate version-compatible checkpointing to the decoder backbone."""

        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        method = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if not callable(method):
            raise RuntimeError("The selected decoder does not support gradient checkpointing.")
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Cannot inspect the decoder gradient-checkpointing API."
            ) from exc
        accepts_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        method_kwargs = {}
        if "gradient_checkpointing_kwargs" in parameters or accepts_keywords:
            method_kwargs["gradient_checkpointing_kwargs"] = kwargs
        if "every_n_layers" in parameters or accepts_keywords:
            method_kwargs["every_n_layers"] = every_n_layers
        elif every_n_layers != 1:
            raise ValueError(
                "The selected decoder cannot checkpoint every_n_layers != 1."
            )
        method(**method_kwargs)
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
        if self.readout_token is not None:
            packed = append_readout_token(
                PackedCausalInputs(text_embeddings, attention_mask, positions, lengths - 1),
                self.readout_token,
            )
            return packed.inputs_embeds, packed.attention_mask, packed.position_ids, packed.pooling_positions
        return text_embeddings, attention_mask, positions, lengths - 1

    def _gaze_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Infer raw ET2 channels, optionally redistribute TRT, then project/pack."""

        raw_gaze, gaze_mask = self.gaze_provider.compute(input_ids, attention_mask)
        if self.gaze_redistributor is not None:
            if gaze_mask.shape != attention_mask.shape:
                raise ValueError("gaze_mask must match the text attention_mask shape.")
            if gaze_mask.device != attention_mask.device:
                raise ValueError("gaze_mask and attention_mask must share a device.")
            if bool((gaze_mask.bool() & ~attention_mask.bool()).any().item()):
                raise ValueError("Valid gaze positions must also be valid text positions.")
            raw_gaze = self.gaze_redistributor(raw_gaze, gaze_mask)
            raw_gaze = torch.where(gaze_mask.bool().unsqueeze(-1), raw_gaze, 0.0)
        projector_dtype = next(self.gaze_projector.parameters()).dtype
        projected_gaze = self.gaze_projector(
            raw_gaze.to(device=text_embeddings.device, dtype=projector_dtype)
        ).to(dtype=text_embeddings.dtype)
        relation_values = None
        if self.gaze_relation["method"] != "none":
            relation_values = raw_gaze[..., tuple(self.gaze_provider.feature_indices).index(3)]
            if self.gaze_relation.get("input_mode", "aligned") == "shuffled":
                from .gaze_relation_inputs import shuffle_relation_trt

                relation_values = shuffle_relation_trt(
                    relation_values, input_ids, attention_mask, gaze_mask,
                    seed=self.gaze_relation["shuffle_seed"],
                )
        packed = pack_prefix_gaze(
            text_embeddings=text_embeddings,
            text_attention_mask=attention_mask,
            gaze_embeddings=projected_gaze,
            gaze_mask=gaze_mask,
            eye_start=self.eye_start,
            eye_end=self.eye_end,
            text_gaze_values=relation_values,
        )
        if self.readout_token is not None:
            packed = append_readout_token(
                packed, self.readout_token,
                relation_reference=self.gaze_relation["method"] == "trt-readout-rotation",
            )
        relation_kwargs = {}
        if self.gaze_relation["method"] != "none":
            relation_kwargs = {
                "gaze_relation_values": packed.text_gaze_values,
                "gaze_relation_mask": packed.text_gaze_mask,
                "gaze_relation_indices": packed.text_gaze_indices,
                "gaze_relation_anchor_mask": packed.text_gaze_anchor_mask,
            }
        return (
            packed.inputs_embeds,
            packed.attention_mask,
            packed.position_ids,
            packed.pooling_positions,
            relation_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Return bounded VA means, optionally followed by two unbounded log variances."""

        del labels, kwargs
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must both be rank-2.")
        if tuple(input_ids.shape) != tuple(attention_mask.shape):
            raise ValueError("input_ids and attention_mask must have identical shape.")
        if input_ids.device != attention_mask.device:
            raise ValueError("input_ids and attention_mask must be on the same device.")

        text_embeddings = self.get_input_embeddings()(input_ids)
        relation_kwargs = {}
        if self.gaze_fusion == "prefix-concat":
            inputs_embeds, fused_mask, position_ids, pooling_positions, relation_kwargs = self._gaze_inputs(
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
            **relation_kwargs,
        )
        hidden = outputs.last_hidden_state
        row_indices = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[row_indices, pooling_positions]
        if any(parameter.dtype != torch.float32 for parameter in self.regression_head.parameters()):
            raise ValueError("The regression head must remain FP32; use autocast for mixed precision.")
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            raw_logits = self.regression_head(pooled.float())
            logits = F.hardsigmoid(raw_logits)
            if self.variance_head is not None:
                if any(parameter.dtype != torch.float32 for parameter in self.variance_head.parameters()):
                    raise ValueError("The variance head must remain FP32; use autocast for mixed precision.")
                variance_features = self.regression_head[0](pooled.float())
                log_variance = log_variance_from_raw(self.variance_head(variance_features))
                logits = torch.cat((logits, log_variance), dim=-1)
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
            "variance_head_trainable_parameters": sum(
                parameter.numel() for parameter in self.variance_head.parameters()
            ) if self.variance_head is not None else 0,
            "gaze_redistribution_trainable_parameters": sum(
                parameter.numel()
                for parameter in (
                    self.gaze_redistributor.parameters()
                    if self.gaze_redistributor is not None
                    else ()
                )
                if parameter.requires_grad
            ),
            "readout_trainable_parameters": self.readout_token.numel() if self.readout_token is not None else 0,
            "gaze_relation_trainable_parameters": sum(
                parameter.numel() for name, parameter in self.named_parameters()
                if name.endswith("gaze_relative_rotation.alpha") and parameter.requires_grad
            ),
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
        active_indices = (
            tuple(self.gaze_provider.feature_indices)
            if self.gaze_provider is not None
            else ()
        )
        active_names = (
            et2_feature_names_from_indices(active_indices)
            if active_indices
            else ()
        )
        features_used = (
            et2_features_used_from_indices(active_indices)
            if active_indices
            else (0,) * len(ET2_FEATURE_NAMES)
        )
        payload = {
            "schema_version": ARCHITECTURE_MANIFEST_VERSION,
            "precision_contract": copy.deepcopy(PRECISION_CONTRACT),
            "decoder_model_id": self._reconstruction_config["decoder_model_id"],
            "decoder_model_type": getattr(self.config, "model_type", None),
            "decoder_commit": self._reconstruction_config["decoder_revision"],
            "finetuning_mode": self._reconstruction_config["finetuning_mode"],
            "gaze_fusion": self.gaze_fusion,
            "gaze_alignment": copy.deepcopy(self.gaze_alignment),
            "gaze_redistribution": copy.deepcopy(self.gaze_redistribution),
            "gaze_relation": copy.deepcopy(self.gaze_relation),
            "readout": copy.deepcopy(self.readout),
            "prediction_distribution": copy.deepcopy(self.prediction_distribution),
            "gaze_relation_layers": copy.deepcopy(self.gaze_relation_layers),
            "gaze_features": list(active_names),
            "gaze_feature_indices": list(active_indices),
            "features_used": list(features_used),
            "et_model": (
                {
                    "repo_id": self.gaze_provider.repo_id,
                    "revision": self.gaze_provider.revision,
                    "filename": self.gaze_provider.filename,
                    "feature_names": list(active_names),
                    "feature_indices": list(active_indices),
                    "features_used": list(features_used),
                }
                if self.gaze_provider is not None
                else None
            ),
            "gaze_concat_order": readout_concat_order(self.gaze_fusion, self.readout),
            "pooling_position": readout_pooling_position(self.gaze_fusion, self.readout),
            "output_names": list(self.config.va_output_names),
            "output_activation": OUTPUT_ACTIVATION,
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


def _load_qwen_text_backbone(
    model_id: str = DEFAULT_DECODER_MODEL_ID,
    *,
    revision: str = DEFAULT_DECODER_REVISION,
    dtype: torch.dtype | str = torch.bfloat16,
    attn_implementation: str | None = None,
):
    """Load only the trainable text backbone from the pinned Qwen causal LM."""

    try:
        from transformers import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise ImportError(
            "Qwen training requires a recent Transformers build; "
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
    backbone.requires_grad_(True)
    return backbone


def load_qwen_backbone(
    model_id: str = DEFAULT_DECODER_MODEL_ID,
    *,
    revision: str = DEFAULT_DECODER_REVISION,
    dtype: torch.dtype | str = torch.bfloat16,
    finetuning_mode: str = "lora",
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    attn_implementation: str | None = None,
):
    """Load a full-trainable or all-linear-LoRA Qwen text backbone."""

    normalized_mode = _normalize_finetuning_mode(finetuning_mode)
    if normalized_mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "Qwen LoRA training requires PEFT; install the repository-root "
                "requirements.txt."
            ) from exc
    backbone = _load_qwen_text_backbone(
        model_id,
        revision=revision,
        dtype=backbone_parameter_dtype(dtype, normalized_mode),
        attn_implementation=attn_implementation,
    )
    if normalized_mode == "full":
        _promote_trainable_parameters(backbone)
        return backbone

    lora_config = LoraConfig(
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        target_modules="all-linear",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    adapted_backbone = get_peft_model(backbone, lora_config)
    _promote_trainable_parameters(adapted_backbone)
    return adapted_backbone


def load_qwen_backbone_with_lora(
    model_id: str = DEFAULT_DECODER_MODEL_ID,
    *,
    revision: str = DEFAULT_DECODER_REVISION,
    dtype: torch.dtype | str = torch.bfloat16,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    attn_implementation: str | None = None,
):
    """Backward-compatible wrapper for the explicit LoRA loader mode."""

    return load_qwen_backbone(
        model_id,
        revision=revision,
        dtype=dtype,
        finetuning_mode="lora",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        attn_implementation=attn_implementation,
    )


def build_qwen_va_model(
    tokenizer,
    *,
    model_id: str = DEFAULT_DECODER_MODEL_ID,
    model_revision: str = DEFAULT_DECODER_REVISION,
    gaze_fusion: str = "prefix-concat",
    gaze_redistribution: Mapping | None = None,
    gaze_relation: Mapping | None = None,
    readout: Mapping | None = None,
    prediction_distribution: Mapping | None = None,
    et_repo_id: str = DEFAULT_ET2_REPO_ID,
    et_revision: str = DEFAULT_ET2_REVISION,
    et_filename: str = DEFAULT_ET2_FILENAME,
    et_cache_size: int = 70000,
    gaze_feature_indices: Sequence[int] | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    finetuning_mode: str = "lora",
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    gaze_projection_dim: int = DEFAULT_GAZE_PROJECTION_DIM,
    gaze_projection_dropout: tuple[
        float,
        float,
    ] = DEFAULT_GAZE_PROJECTION_DROPOUT,
    classifier_dropout: float = DEFAULT_CLASSIFIER_DROPOUT,
    attn_implementation: str | None = None,
) -> DecoderVARegressor:
    """Construct the selected Qwen VA model and optional pinned ET2 provider."""

    normalized_fusion = _normalize_gaze_fusion(gaze_fusion)
    normalized_readout = validate_readout_contract(readout_contract() if readout is None else readout)
    normalized_prediction = validate_prediction_contract(
        prediction_contract() if prediction_distribution is None else prediction_distribution
    )
    normalized_finetuning_mode = _normalize_finetuning_mode(finetuning_mode)
    active_feature_indices = (
        normalize_et2_feature_indices(gaze_feature_indices)
        if normalized_fusion == "prefix-concat"
        else ()
    )
    normalized_redistribution = validate_redistribution_contract(
        {"method": "none"} if gaze_redistribution is None else gaze_redistribution,
        gaze_fusion=normalized_fusion,
        feature_indices=active_feature_indices,
    )
    normalized_relation = validate_gaze_relation_setup(
        gaze_relation, gaze_fusion=normalized_fusion,
        feature_indices=active_feature_indices,
        gaze_redistribution=normalized_redistribution,
        attn_implementation=attn_implementation,
        readout=normalized_readout,
    )
    if normalized_relation["method"] != "none":
        attn_implementation = "eager"
    active_feature_names = (
        et2_feature_names_from_indices(active_feature_indices)
        if active_feature_indices
        else ()
    )
    active_features_used = (
        et2_features_used_from_indices(active_feature_indices)
        if active_feature_indices
        else (0,) * len(ET2_FEATURE_NAMES)
    )
    backbone = load_qwen_backbone(
        model_id,
        revision=model_revision,
        dtype=backbone_parameter_dtype(dtype, normalized_finetuning_mode),
        finetuning_mode=normalized_finetuning_mode,
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
            feature_indices=active_feature_indices,
            cache_size=et_cache_size,
        )
        if normalized_fusion == "prefix-concat"
        else None
    )
    model = DecoderVARegressor(
        backbone,
        gaze_provider=gaze_provider,
        gaze_fusion=normalized_fusion,
        gaze_redistribution=normalized_redistribution,
        gaze_relation=normalized_relation,
        readout=normalized_readout,
        gaze_projection_dim=gaze_projection_dim,
        prediction_distribution=normalized_prediction,
        gaze_projection_dropout=gaze_projection_dropout,
        classifier_dropout=classifier_dropout,
    )
    _promote_trainable_parameters(model)
    model.config.precision_contract = copy.deepcopy(PRECISION_CONTRACT)
    model.config.finetuning_mode = normalized_finetuning_mode
    model.finetuning_mode = normalized_finetuning_mode
    uses_lora = normalized_finetuning_mode == "lora"
    model._reconstruction_config = {
        "decoder_model_id": str(model_id),
        "decoder_revision": str(model_revision),
        "finetuning_mode": normalized_finetuning_mode,
        "gaze_fusion": normalized_fusion,
        "gaze_alignment": copy.deepcopy(model.gaze_alignment),
        "gaze_redistribution": copy.deepcopy(normalized_redistribution),
        "gaze_relation": copy.deepcopy(normalized_relation),
        "readout": copy.deepcopy(normalized_readout),
        "prediction_distribution": copy.deepcopy(normalized_prediction),
        "et_repo_id": str(et_repo_id),
        "et_revision": str(et_revision),
        "et_filename": str(et_filename),
        "et_feature_names": list(active_feature_names),
        "et_feature_indices": list(active_feature_indices),
        "features_used": list(active_features_used),
        "et_cache_size": int(et_cache_size),
        "output_dim": 2,
        "output_activation": OUTPUT_ACTIVATION,
        "lora_rank": int(lora_rank) if uses_lora else None,
        "lora_alpha": int(lora_alpha) if uses_lora else None,
        "lora_dropout": float(lora_dropout) if uses_lora else None,
        "lora_target_modules": "all-linear" if uses_lora else None,
        "lora_task_type": "FEATURE_EXTRACTION" if uses_lora else None,
        "gaze_projection_dim": int(gaze_projection_dim),
        "gaze_projection_dropout": [
            float(value) for value in gaze_projection_dropout
        ],
        "classifier_dropout": float(classifier_dropout),
        "attn_implementation": attn_implementation,
        "backbone_dtype_at_construction": str(
            backbone_parameter_dtype(dtype, normalized_finetuning_mode)
        ).replace("torch.", ""),
        "precision_contract": copy.deepcopy(PRECISION_CONTRACT),
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
    manifest_version = manifest.get("schema_version")
    if type(manifest_version) is not int or manifest_version not in (
        PRE_RELATION_MANIFEST_VERSION, PRE_READOUT_MANIFEST_VERSION,
        PRE_UNCERTAINTY_MANIFEST_VERSION, ARCHITECTURE_MANIFEST_VERSION,
    ):
        raise ValueError(
            "Unsupported decoder VA architecture manifest version: "
            f"{manifest_version!r}. Historical precision and gaze_alignment semantics "
            "require the original code; start a new training run with version 11."
        )
    validate_precision_contract(manifest.get("precision_contract"), context="Saved architecture")
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
    validate_precision_contract(reconstruction.get("precision_contract"), context="Saved reconstruction")
    required = {
        "decoder_model_id",
        "decoder_revision",
        "gaze_fusion",
        "et_repo_id",
        "et_revision",
        "et_filename",
        "et_feature_names",
        "et_feature_indices",
        "features_used",
        "et_cache_size",
        "output_dim",
        "output_activation",
        "finetuning_mode",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "lora_task_type",
        "gaze_projection_dim",
        "gaze_projection_dropout",
        "classifier_dropout",
        "attn_implementation",
        "backbone_dtype_at_construction",
    }
    missing = sorted(required.difference(reconstruction))
    if missing:
        raise ValueError(
            "Architecture manifest is missing reconstruction field(s): "
            + ", ".join(missing)
        )
    saved_finetuning_mode = _normalize_finetuning_mode(reconstruction["finetuning_mode"])
    if manifest.get("finetuning_mode") != saved_finetuning_mode:
        raise ValueError(
            "Architecture manifest fine-tuning mode disagrees with "
            "reconstruction metadata."
        )
    saved_backbone_dtype = reconstruction["backbone_dtype_at_construction"]
    if saved_backbone_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Saved backbone_dtype_at_construction must be an explicit floating-point dtype.")
    if saved_finetuning_mode == "full" and saved_backbone_dtype != "float32":
        raise ValueError("Full fine-tuning backbone_dtype_at_construction must be float32.")

    saved_fusion = reconstruction["gaze_fusion"]
    if saved_fusion not in GAZE_FUSIONS:
        raise ValueError(
            f"Saved model declares unsupported gaze fusion {saved_fusion!r}."
        )
    for owner, metadata in (("architecture", manifest), ("reconstruction", reconstruction)):
        validate_gaze_alignment_contract(
            metadata.get("gaze_alignment"),
            gaze_fusion=saved_fusion,
            context=f"Saved {owner} metadata",
        )
    raw_feature_indices = reconstruction["et_feature_indices"]
    if not isinstance(raw_feature_indices, list):
        raise ValueError("Saved ET2 feature indices must be a JSON list.")
    if saved_fusion == "prefix-concat":
        try:
            saved_feature_indices = normalize_et2_feature_indices(raw_feature_indices)
        except (TypeError, ValueError) as exc:
            raise ValueError("Saved ET2 feature indices are invalid.") from exc
        if list(saved_feature_indices) != raw_feature_indices:
            raise ValueError("Saved ET2 feature indices are not in canonical order.")
    else:
        if raw_feature_indices:
            raise ValueError("Text-only models must not declare active ET2 features.")
        saved_feature_indices = ()
    redistribution_configs = []
    for owner, metadata in (("architecture", manifest), ("reconstruction", reconstruction)):
        if "gaze_redistribution" not in metadata:
            raise ValueError(f"Missing gaze_redistribution in {owner} metadata.")
        configuration = validate_redistribution_contract(
            metadata.get("gaze_redistribution", {"method": "none"}),
            gaze_fusion=saved_fusion,
            feature_indices=saved_feature_indices,
        )
        redistribution_configs.append(configuration)
    if redistribution_configs[0] != redistribution_configs[1]:
        raise ValueError("Architecture gaze_redistribution disagrees with reconstruction metadata.")
    saved_redistribution = redistribution_configs[0]
    readouts = [
        saved_readout_contract(metadata, schema_version=manifest_version, context=f"Saved {owner}")
        for owner, metadata in (("architecture", manifest), ("reconstruction", reconstruction))
    ]
    if readouts[0] != readouts[1]:
        raise ValueError("Architecture readout disagrees with reconstruction metadata.")
    saved_readout = readouts[0]
    prediction_configs = [
        saved_prediction_contract(metadata, schema_version=manifest_version, context=f"Saved {owner}")
        for owner, metadata in (("architecture", manifest), ("reconstruction", reconstruction))
    ]
    if prediction_configs[0] != prediction_configs[1]:
        raise ValueError("Architecture prediction_distribution disagrees with reconstruction metadata.")
    saved_prediction = prediction_configs[0]
    relation_configs = []
    for owner, metadata in (("architecture", manifest), ("reconstruction", reconstruction)):
        if manifest_version >= PRE_READOUT_MANIFEST_VERSION and "gaze_relation" not in metadata:
            raise ValueError(f"Missing gaze_relation in {owner} metadata.")
        if "gaze_relation" in metadata and not isinstance(metadata["gaze_relation"], Mapping):
            raise ValueError(f"Saved {owner} gaze_relation must be a mapping.")
        relation = validate_gaze_relation_setup(
            metadata.get("gaze_relation"), gaze_fusion=saved_fusion,
            feature_indices=saved_feature_indices,
            gaze_redistribution=saved_redistribution,
            attn_implementation=reconstruction["attn_implementation"],
            readout=saved_readout,
        )
        if manifest_version == PRE_RELATION_MANIFEST_VERSION and relation["method"] != "none":
            raise ValueError("Schema 8 cannot enable gaze_relation.")
        relation_configs.append(relation)
    if relation_configs[0] != relation_configs[1]:
        raise ValueError("Architecture gaze_relation disagrees with reconstruction metadata.")
    saved_relation = relation_configs[0]
    if saved_relation["method"] != "none" and reconstruction["attn_implementation"] != "eager":
        raise ValueError("Saved gaze_relation requires explicit attn_implementation='eager'.")
    if reconstruction["output_dim"] != 2:
        raise ValueError("Only the two-output valence/arousal head is supported.")
    if reconstruction["output_activation"] != OUTPUT_ACTIVATION:
        raise ValueError(
            f"Only the {OUTPUT_ACTIVATION!r} VA output activation is supported."
        )
    lora_fields = {
        "lora_rank": reconstruction["lora_rank"],
        "lora_alpha": reconstruction["lora_alpha"],
        "lora_dropout": reconstruction["lora_dropout"],
        "lora_target_modules": reconstruction["lora_target_modules"],
        "lora_task_type": reconstruction["lora_task_type"],
    }
    if saved_finetuning_mode == "lora":
        try:
            lora_rank = int(lora_fields["lora_rank"])
            lora_alpha = int(lora_fields["lora_alpha"])
            lora_dropout = float(lora_fields["lora_dropout"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Saved LoRA hyperparameters are invalid.") from exc
        if lora_rank <= 0 or lora_alpha <= 0:
            raise ValueError("Saved LoRA rank and alpha must be positive.")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError("Saved LoRA dropout must be in [0, 1).")
        if lora_fields["lora_target_modules"] != "all-linear":
            raise ValueError(
                "Saved model does not use the supported all-linear LoRA contract."
            )
        if lora_fields["lora_task_type"] != "FEATURE_EXTRACTION":
            raise ValueError("Saved model does not use the supported LoRA task type.")
    else:
        non_null_lora_fields = sorted(
            field for field, value in lora_fields.items() if value is not None
        )
        if non_null_lora_fields:
            raise ValueError(
                "Full fine-tuning manifests must set all LoRA metadata to null: "
                + ", ".join(non_null_lora_fields)
            )
        lora_rank = DEFAULT_LORA_RANK
        lora_alpha = DEFAULT_LORA_ALPHA
        lora_dropout = DEFAULT_LORA_DROPOUT
    if manifest.get("gaze_fusion") != saved_fusion:
        raise ValueError(
            "Architecture manifest gaze_fusion disagrees with reconstruction metadata."
        )
    expected_order = readout_concat_order(saved_fusion, saved_readout)
    expected_pooling = readout_pooling_position(saved_fusion, saved_readout)
    if manifest.get("gaze_concat_order") != expected_order:
        raise ValueError("Architecture manifest declares an incompatible gaze concat order.")
    if manifest.get("pooling_position") != expected_pooling:
        raise ValueError("Architecture manifest declares an incompatible pooling position.")
    expected_feature_names = (
        list(et2_feature_names_from_indices(saved_feature_indices))
        if saved_feature_indices
        else []
    )
    expected_features_used = (
        list(et2_features_used_from_indices(saved_feature_indices))
        if saved_feature_indices
        else [0] * len(ET2_FEATURE_NAMES)
    )
    if reconstruction["et_feature_names"] != expected_feature_names:
        raise ValueError("Reconstruction metadata declares incompatible gaze feature names.")
    if reconstruction["features_used"] != expected_features_used:
        raise ValueError("Reconstruction metadata declares an incompatible gaze feature mask.")
    if manifest.get("gaze_feature_indices") != list(saved_feature_indices):
        raise ValueError("Architecture manifest declares incompatible gaze feature indices.")
    if manifest.get("gaze_features") != expected_feature_names:
        raise ValueError("Architecture manifest declares incompatible gaze feature names.")
    if manifest.get("features_used") != expected_features_used:
        raise ValueError("Architecture manifest declares an incompatible gaze feature mask.")
    if manifest.get("output_activation") != OUTPUT_ACTIVATION:
        raise ValueError("Architecture manifest declares an incompatible output activation.")
    et_model = manifest.get("et_model")
    if saved_fusion == "prefix-concat":
        if not isinstance(et_model, dict):
            raise ValueError("Architecture manifest is missing ET2 model metadata.")
        expected_et_model = {
            "repo_id": reconstruction["et_repo_id"],
            "revision": reconstruction["et_revision"],
            "filename": reconstruction["et_filename"],
            "feature_names": expected_feature_names,
            "feature_indices": list(saved_feature_indices),
            "features_used": expected_features_used,
        }
        if et_model != expected_et_model:
            raise ValueError("Architecture manifest declares incompatible ET2 model metadata.")
    elif et_model is not None:
        raise ValueError("Text-only models must not declare ET2 model metadata.")

    if saved_redistribution["method"] != "none":
        validate_redistribution_state_file(weights_path)

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=False,
            local_files_only=True,
        )
    tokenizer.padding_side = "right"

    if dtype == "auto":
        dtype = reconstruction["backbone_dtype_at_construction"]
    effective_cache_size = (
        int(reconstruction["et_cache_size"])
        if et_cache_size is None
        else int(et_cache_size)
    )
    model = build_qwen_va_model(
        tokenizer,
        model_id=str(reconstruction["decoder_model_id"]),
        model_revision=str(reconstruction["decoder_revision"]),
        finetuning_mode=saved_finetuning_mode,
        gaze_fusion=str(reconstruction["gaze_fusion"]),
        gaze_redistribution=saved_redistribution,
        gaze_relation=saved_relation,
        readout=saved_readout,
        et_repo_id=str(reconstruction["et_repo_id"]),
        prediction_distribution=saved_prediction,
        et_revision=str(reconstruction["et_revision"]),
        et_filename=str(reconstruction["et_filename"]),
        et_cache_size=effective_cache_size,
        gaze_feature_indices=saved_feature_indices or None,
        dtype=dtype,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        gaze_projection_dim=int(reconstruction["gaze_projection_dim"]),
        gaze_projection_dropout=tuple(
            float(value) for value in reconstruction["gaze_projection_dropout"]
        ),
        classifier_dropout=float(reconstruction["classifier_dropout"]),
        attn_implementation=reconstruction["attn_implementation"],
    )

    from safetensors.torch import load_model

    validate_saved_trainable_precision(model, weights_path)
    validate_saved_gaze_relation(model, weights_path)
    validate_saved_readout(model, weights_path)
    validate_saved_variance_head(model, weights_path)
    load_model(model, weights_path, strict=True, device="cpu")
    model.eval()
    return model, tokenizer

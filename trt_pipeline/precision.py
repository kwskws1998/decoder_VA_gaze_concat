"""Native BF16 backbone storage with the unchanged current TRT and FP32 VA head."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import torch

from trt_pipeline.core import model as baseline
from trt_pipeline.core.gaze import (
    DEFAULT_ET2_FILENAME, DEFAULT_ET2_REPO_ID, DEFAULT_ET2_REVISION,
    ET2GazeProvider, et2_feature_names_from_indices, et2_features_used_from_indices,
    normalize_et2_feature_indices,
)


BF16_PROFILE = "native_bf16_backbone_current_trt_v1"
BF16_PRECISION_CONTRACT = {
    "version": 1,
    "backbone_parameter_storage": "native_bfloat16_preserve_framework_float32_exceptions",
    "trainable_parameter_dtype": "mixed_bfloat16_float32",
    "non_backbone_parameter_dtype": "float32",
    "regression_head_compute_dtype": "float32",
    "output_dtype": "float32",
    "mixed_precision": "bfloat16_autocast",
    "additional_fp32_master_weights": False,
}
_PARAMETER_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_STORAGE_DTYPES = {
    torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16",
    torch.float64: "F64", torch.int64: "I64", torch.int32: "I32",
    torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
}


def _dtype_name(dtype):
    """Represent actual storage dtypes without serializing PyTorch objects."""

    return str(dtype).removeprefix("torch.")


class BF16DecoderVARegressor(baseline.DecoderVARegressor):
    """Restore native backbone dtypes after the baseline constructor's FP32 promotion."""

    def __init__(self, backbone, *, native_backbone_dtypes, **kwargs):
        """Keep pretrained BF16 values and framework-selected FP32 exceptions intact."""

        native = dict(native_backbone_dtypes)
        if not native or any(dtype not in _PARAMETER_DTYPES for dtype in native.values()):
            raise ValueError("Native backbone parameters must have explicit BF16/FP32 storage dtypes.")
        if "bfloat16" not in native.values():
            raise ValueError("The native backbone loader did not produce BF16 parameters.")
        super().__init__(backbone, **kwargs)
        parameters = dict(self.backbone.named_parameters())
        if set(parameters) != set(native):
            raise ValueError("Backbone parameter identities changed during model construction.")
        for name, parameter in parameters.items():
            parameter.data = parameter.data.to(dtype=_PARAMETER_DTYPES[native[name]])
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(dtype=_PARAMETER_DTYPES[native[name]])
        self._native_backbone_parameter_dtypes = native
        self.implementation_profile = BF16_PROFILE
        self.config.implementation_profile = BF16_PROFILE
        self.config.decoder_va_architecture_version = 1
        self.config.precision_contract = copy.deepcopy(BF16_PRECISION_CONTRACT)

    def save_architecture_manifest(self, output_dir):
        """Save this isolated storage policy without declaring the active FP32 schema."""

        from .progress import atomic_write_json

        if self._reconstruction_config is None:
            raise RuntimeError("This BF16 model has no reconstruction configuration.")
        audit = audit_model_precision(self)
        reconstruction = copy.deepcopy(self._reconstruction_config)
        indices = tuple(reconstruction["et_feature_indices"])
        payload = {
            "schema_version": 1, "implementation_profile": BF16_PROFILE,
            "decoder_model_id": reconstruction["decoder_model_id"],
            "decoder_commit": reconstruction["decoder_revision"],
            "decoder_model_type": getattr(self.config, "model_type", None),
            "finetuning_mode": "full", "gaze_fusion": self.gaze_fusion,
            "gaze_alignment": copy.deepcopy(self.gaze_alignment),
            "gaze_redistribution": copy.deepcopy(self.gaze_redistribution),
            "gaze_relation": copy.deepcopy(self.gaze_relation),
            "gaze_relation_layers": copy.deepcopy(self.gaze_relation_layers),
            "readout": copy.deepcopy(self.readout),
            "prediction_distribution": copy.deepcopy(self.prediction_distribution),
            "gaze_features": list(et2_feature_names_from_indices(indices)),
            "gaze_feature_indices": list(indices),
            "features_used": list(et2_features_used_from_indices(indices)),
            "gaze_concat_order": baseline.GAZE_PREFIX_ORDER,
            "pooling_position": baseline.GAZE_PREFIX_POOLING,
            "output_names": ["valence", "arousal"], "output_activation": "hard_sigmoid",
            "precision_contract": copy.deepcopy(BF16_PRECISION_CONTRACT),
            "parameter_dtypes": audit["parameter_dtypes"],
            "parameter_elements_by_dtype": audit["parameter_elements_by_dtype"],
            "native_backbone_parameter_dtypes": copy.deepcopy(self._native_backbone_parameter_dtypes),
            "reconstruction": reconstruction,
            "et_model": {"repo_id": reconstruction["et_repo_id"], "revision": reconstruction["et_revision"],
                         "filename": reconstruction["et_filename"],
                         "feature_names": list(et2_feature_names_from_indices(indices)),
                         "feature_indices": list(indices),
                         "features_used": list(et2_features_used_from_indices(indices))},
            "state_dict": {"filename": "model.safetensors", "format": "safetensors",
                           "scope": "complete BF16DecoderVARegressor with current TRT alignment",
                           "strict_loading": True},
            **self.trainable_parameter_summary(),
        }
        return atomic_write_json(Path(output_dir) / "decoder_va_architecture.json", payload)


def build_bf16_model(
    tokenizer, *, model_id=baseline.DEFAULT_DECODER_MODEL_ID,
    model_revision=baseline.DEFAULT_DECODER_REVISION,
    finetuning_mode="full", gaze_fusion="prefix-concat", gaze_feature_indices=None,
    dtype=torch.bfloat16, et_repo_id=DEFAULT_ET2_REPO_ID,
    et_revision=DEFAULT_ET2_REVISION, et_filename=DEFAULT_ET2_FILENAME,
    et_cache_size=70000, lora_rank=None, lora_alpha=None, lora_dropout=None,
    attn_implementation=None, gaze_projection_dim=baseline.DEFAULT_GAZE_PROJECTION_DIM,
    gaze_projection_dropout=baseline.DEFAULT_GAZE_PROJECTION_DROPOUT,
    classifier_dropout=baseline.DEFAULT_CLASSIFIER_DROPOUT,
    gaze_redistribution=None, gaze_relation=None, readout=None,
    prediction_distribution=None,
):
    """Load native BF16 Qwen storage; LoRA-only arguments have no effect in full mode."""

    del lora_rank, lora_alpha, lora_dropout
    if finetuning_mode != "full" or baseline._normalize_gaze_fusion(gaze_fusion) != "prefix-concat":
        raise ValueError("The isolated BF16 profile requires full tuning and the original TRT prefix.")
    if dtype not in (torch.bfloat16, "bf16", "bfloat16", "torch.bfloat16"):
        raise ValueError("The isolated BF16 profile requires BF16 compute and native backbone storage.")
    indices = normalize_et2_feature_indices(gaze_feature_indices)
    if indices != (3,):
        raise ValueError("The isolated BF16 experiment uses only the existing TRT channel.")
    redistribution = baseline.validate_redistribution_contract(
        {"method": "none"} if gaze_redistribution is None else gaze_redistribution,
        gaze_fusion="prefix-concat", feature_indices=indices)
    relation = baseline.validate_gaze_relation_setup(
        gaze_relation, gaze_fusion="prefix-concat", feature_indices=indices,
        gaze_redistribution=redistribution, attn_implementation=attn_implementation)
    chosen_readout = baseline.validate_readout_contract(baseline.readout_contract() if readout is None else readout)
    distribution = baseline.validate_prediction_contract(
        baseline.prediction_contract() if prediction_distribution is None else prediction_distribution)
    backbone = baseline._load_qwen_text_backbone(
        model_id, revision=model_revision, dtype=torch.bfloat16, attn_implementation=attn_implementation)
    native_dtypes = {name: _dtype_name(parameter.dtype) for name, parameter in backbone.named_parameters()}
    provider = ET2GazeProvider(tokenizer=tokenizer, repo_id=et_repo_id, revision=et_revision,
                               filename=et_filename, feature_indices=indices, cache_size=et_cache_size)
    model = BF16DecoderVARegressor(
        backbone, native_backbone_dtypes=native_dtypes, gaze_provider=provider,
        gaze_fusion="prefix-concat", gaze_redistribution=redistribution, gaze_relation=relation,
        readout=chosen_readout, prediction_distribution=distribution,
        gaze_projection_dim=gaze_projection_dim, gaze_projection_dropout=gaze_projection_dropout,
        classifier_dropout=classifier_dropout)
    model.finetuning_mode = "full"
    model.config.finetuning_mode = "full"
    model._reconstruction_config = {
        "implementation_profile": BF16_PROFILE, "decoder_model_id": str(model_id),
        "decoder_revision": str(model_revision), "finetuning_mode": "full",
        "gaze_fusion": "prefix-concat", "gaze_alignment": copy.deepcopy(model.gaze_alignment),
        "gaze_redistribution": copy.deepcopy(redistribution), "gaze_relation": copy.deepcopy(relation),
        "readout": copy.deepcopy(chosen_readout), "prediction_distribution": copy.deepcopy(distribution),
        "et_repo_id": str(et_repo_id), "et_revision": str(et_revision), "et_filename": str(et_filename),
        "et_feature_names": list(et2_feature_names_from_indices(indices)),
        "et_feature_indices": list(indices), "features_used": list(et2_features_used_from_indices(indices)),
        "et_cache_size": int(et_cache_size), "output_dim": 2, "output_activation": "hard_sigmoid",
        "lora_rank": None, "lora_alpha": None, "lora_dropout": None,
        "lora_target_modules": None, "lora_task_type": None,
        "gaze_projection_dim": int(gaze_projection_dim),
        "gaze_projection_dropout": list(gaze_projection_dropout), "classifier_dropout": float(classifier_dropout),
        "attn_implementation": attn_implementation, "backbone_dtype_at_construction": "bfloat16",
        "precision_contract": copy.deepcopy(BF16_PRECISION_CONTRACT),
        "native_backbone_parameter_dtypes": copy.deepcopy(native_dtypes),
    }
    audit_model_precision(model)
    return model


def audit_model_precision(model):
    """Assert actual native storage and expose JSON-safe parameter and gradient dtypes."""

    if getattr(model, "implementation_profile", None) != BF16_PROFILE:
        raise ValueError("Model does not declare the isolated BF16 implementation profile.")
    if getattr(model.config, "precision_contract", None) != BF16_PRECISION_CONTRACT:
        raise ValueError("Model precision metadata disagrees with the isolated BF16 contract.")
    parameters = dict(model.named_parameters())
    native = model._native_backbone_parameter_dtypes
    actual_backbone = {name: _dtype_name(parameter.dtype) for name, parameter in model.backbone.named_parameters()}
    if actual_backbone != native:
        raise ValueError("Backbone dtypes differ from the native pretrained loader's BF16/FP32 map.")
    if not any(dtype == "bfloat16" for dtype in actual_backbone.values()):
        raise ValueError("No native BF16 backbone parameters remain.")
    outside = {name: _dtype_name(parameter.dtype) for name, parameter in parameters.items() if not name.startswith("backbone.")}
    if any(dtype != "float32" for dtype in outside.values()):
        raise ValueError("The VA head, gaze projector, and boundary parameters must remain FP32.")
    elements, tensors, backbone_elements = Counter(), Counter(), Counter()
    for name, parameter in parameters.items():
        dtype = _dtype_name(parameter.dtype)
        elements[dtype] += parameter.numel()
        tensors[dtype] += 1
        if name.startswith("backbone."):
            backbone_elements[dtype] += parameter.numel()
    return {
        "profile": BF16_PROFILE, "precision_contract": copy.deepcopy(BF16_PRECISION_CONTRACT),
        "parameter_elements_by_dtype": dict(elements), "parameter_tensors_by_dtype": dict(tensors),
        "parameter_dtypes": {name: _dtype_name(parameter.dtype) for name, parameter in parameters.items()},
        "backbone_parameter_elements_by_dtype": dict(backbone_elements),
        "backbone_parameter_dtypes": actual_backbone, "non_backbone_parameter_dtypes": outside,
        "native_float32_exception_names": [name for name, dtype in native.items() if dtype == "float32"],
        "backbone_bfloat16_fraction": backbone_elements["bfloat16"] / sum(backbone_elements.values()),
        "gradient_dtypes": {name: _dtype_name(parameter.grad.dtype) for name, parameter in parameters.items()
                            if parameter.grad is not None},
        "buffer_dtypes": {name: _dtype_name(buffer.dtype) for name, buffer in model.named_buffers()},
        "output_dtype": "float32",
    }


def validate_saved_precision(model, path):
    """Reject checkpoint key or dtype drift before tensors can be silently converted."""

    from safetensors import safe_open

    audit = audit_model_precision(model)
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file() or path.is_symlink():
        raise ValueError("BF16 precision validation requires a regular model.safetensors file.")
    state = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(state):
            raise ValueError("Saved state keys do not match the complete BF16 model.")
        storage = {}
        for name, tensor in state.items():
            saved_slice = saved.get_slice(name)
            dtype = saved_slice.get_dtype()
            if _STORAGE_DTYPES.get(tensor.dtype) != dtype:
                raise ValueError(f"Saved tensor dtype mismatch for {name}: {dtype} vs {tensor.dtype}.")
            if saved_slice.get_shape() != list(tensor.shape):
                raise ValueError(f"Saved tensor shape mismatch for {name}.")
            storage[name] = dtype
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {**audit, "storage_parameter_dtypes": storage, "weights_sha256": digest.hexdigest(),
            "weights_path": str(path.resolve())}


def load_saved_bf16_model(directory, *, tokenizer=None, et_cache_size=None):
    """Reconstruct only this truthful profile and restore tensors without dtype conversion."""

    from safetensors.torch import load_model
    from trt_pipeline.core.dataset import load_auto_tokenizer

    directory = Path(directory)
    manifest_path = directory / "decoder_va_architecture.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("BF16 reconstruction requires its regular architecture manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("implementation_profile") != BF16_PROFILE
            or manifest.get("precision_contract") != BF16_PRECISION_CONTRACT):
        raise ValueError("Checkpoint does not declare the isolated native BF16 precision profile.")
    reconstruction = manifest.get("reconstruction")
    if not isinstance(reconstruction, dict) or (
        reconstruction.get("implementation_profile") != BF16_PROFILE
        or reconstruction.get("precision_contract") != BF16_PRECISION_CONTRACT
        or reconstruction.get("backbone_dtype_at_construction") != "bfloat16"
        or reconstruction.get("finetuning_mode") != "full"
        or reconstruction.get("gaze_fusion") != "prefix-concat"
        or reconstruction.get("et_feature_indices") != [3]
    ):
        raise ValueError("Saved BF16 reconstruction settings are incomplete or incompatible.")
    state_contract = manifest.get("state_dict", {})
    if (state_contract.get("filename") != "model.safetensors" or state_contract.get("format") != "safetensors"
            or state_contract.get("strict_loading") is not True):
        raise ValueError("BF16 checkpoint must declare a complete strict safetensors state.")
    if tokenizer is None:
        tokenizer = load_auto_tokenizer(directory, use_fast=True, trust_remote_code=False, local_files_only=True)
    tokenizer.padding_side = "right"
    try:
        model = build_bf16_model(
            tokenizer, model_id=reconstruction["decoder_model_id"], model_revision=reconstruction["decoder_revision"],
            finetuning_mode="full", gaze_fusion="prefix-concat", gaze_feature_indices=(3,), dtype=torch.bfloat16,
            et_repo_id=reconstruction["et_repo_id"], et_revision=reconstruction["et_revision"],
            et_filename=reconstruction["et_filename"],
            et_cache_size=reconstruction["et_cache_size"] if et_cache_size is None else et_cache_size,
            gaze_projection_dim=reconstruction["gaze_projection_dim"],
            gaze_projection_dropout=tuple(reconstruction["gaze_projection_dropout"]),
            classifier_dropout=reconstruction["classifier_dropout"],
            attn_implementation=reconstruction["attn_implementation"],
            gaze_redistribution=reconstruction["gaze_redistribution"], gaze_relation=reconstruction["gaze_relation"],
            readout=reconstruction["readout"], prediction_distribution=reconstruction["prediction_distribution"])
    except KeyError as error:
        raise ValueError(f"Saved BF16 reconstruction is missing {error.args[0]}.") from error
    actual = audit_model_precision(model)
    if (actual["parameter_dtypes"] != manifest.get("parameter_dtypes")
            or actual["backbone_parameter_dtypes"] != manifest.get("native_backbone_parameter_dtypes")
            or actual["backbone_parameter_dtypes"] != reconstruction.get("native_backbone_parameter_dtypes")):
        raise ValueError("Reconstructed native BF16/FP32 exception map differs from the saved model.")
    for name in ("gaze_alignment", "gaze_redistribution", "gaze_relation", "readout", "prediction_distribution"):
        if getattr(model, name) != manifest.get(name) or getattr(model, name) != reconstruction.get(name):
            raise ValueError(f"Saved BF16 {name} contract differs from current reconstruction.")
    weights = directory / "model.safetensors"
    validate_saved_precision(model, weights)
    load_model(model, weights, strict=True, device="cpu")
    audit_model_precision(model)
    return model.eval(), tokenizer

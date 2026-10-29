"""Explicit reproduction contract for the archived BF16, batch-16 Raw TRT recipe."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

PROFILE = "schema6_bf16_raw_trt_v1"
ROOT = Path(__file__).resolve().parent
ALIGNMENT = {
    "version": "schema6_snapshot",
    "segmentation": "punctuation_symbol_cjk_split",
    "alignment": "historical_token_search_first_subword",
    "collision": "last_write",
    "negative_trt": "preserve",
}
PRECISION = {
    "version": "schema6_snapshot",
    "backbone": "load_bfloat16_preserve_transformers_fp32_exceptions",
    "new_head_projector_boundaries": "float32_parameters_bfloat16_autocast",
    "mean_head_compute": "historical_autocast",
    "master_weights": "no_additional_fp32_master_copy",
}
FOLD_HASHES = {
    "full_dataset_fold1.csv": "86c882ca8b895f1008863a0dad909a1dd02c2c5e9007af2a4c3dc864e2d988b5",
    "full_dataset_fold2.csv": "47ad2496e7c20ddb214684a748086c4d54b983b1655c3e8fe53ce4cbbb7e899a",
}


def verify_runtime():
    """Fail before training if the recovered model code or pinned framework has changed."""
    for package, expected in (("transformers", "5.14.1"), ("torch", "2.12.1")):
        if version(package).split("+")[0] != expected:
            raise ValueError(f"Historical BF16 requires {package}=={expected}; use its separate environment.")
    source = json.loads((ROOT / "snapshot/source.json").read_text())
    for name, expected in source["files"].items():
        if hashlib.sha256((ROOT / "snapshot" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Historical snapshot changed: {name}")


def validate_args(args):
    """Keep reproduction settings explicit and reject silent migration to modern defaults."""
    verify_runtime()
    expected = {
        "model_id": "Qwen/Qwen3.5-0.8B-Base",
        "model_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "et_model_id": "skboy/et_prediction_2",
        "et_revision": "5785e77309d9fce8b88e908a9db100c1a0a63456",
        "et_filename": "et_predictor2_seed123.safetensors",
        "finetuning_mode": "full", "precision": "bf16", "loss": "mse",
        "gaze_fusion": "prefix-concat", "gaze_feature_indices": (3,),
        "train_batch_size": 16, "eval_batch_size": 16, "gradient_accumulation_steps": 1,
        "training_text_overlap_policy": "keep", "no_iemocap": True, "sentence_only": False,
        "exclude_dataset": [], "max_length": 200, "epochs": 10, "max_steps": -1,
        "learning_rate": 6e-6, "weight_decay": .01, "warmup_ratio": .1,
        "gradient_checkpointing": True, "group_by_length": True,
        "attn_implementation": None, "resume_from_checkpoint": None,
    }
    for key, value in expected.items():
        found = getattr(args, key)
        if key == "gaze_feature_indices":
            found = tuple(found)
        if found != value:
            raise ValueError(f"Historical recipe requires {key}={value!r}; got {found!r}.")
    for name, expected_hash in FOLD_HASHES.items():
        if hashlib.sha256((Path(args.data_dir) / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Historical paper7_seed42 data hash mismatch: {name}")


def manifest_overrides():
    """Describe the recovered recipe honestly, including the retained reporting correction."""
    source = json.loads((ROOT / "snapshot/source.json").read_text())
    return {
        "implementation_profile": PROFILE, "historical_source": source,
        "architecture_manifest_version": 1,
        "gaze_alignment": ALIGNMENT, "precision_contract": PRECISION,
        "dtype": "bfloat16", "compute_dtype": "bfloat16",
        "backbone_parameter_dtype": "bfloat16_with_transformers_fp32_exceptions",
        "trainable_parameter_dtype": "mixed_bfloat16_float32_as_loaded",
        "reporting_correction": "sequential_evaluation_preserves_prediction_row_identity",
        "reproduction_scope": "historical_model_and_training_recipe_not_exact_hardware_or_unrecorded_kernel_state",
        "profile_code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob("*.py"))
        },
    }


def build_model(tokenizer, **kwargs):
    """Translate shared runner arguments to the separate, unchanged schema-6 builder."""
    from trt_pipeline.historical.model import build_model as build
    for key in ("gaze_redistribution", "gaze_relation", "readout", "prediction_distribution"):
        kwargs.pop(key, None)
    return build(tokenizer, **kwargs)

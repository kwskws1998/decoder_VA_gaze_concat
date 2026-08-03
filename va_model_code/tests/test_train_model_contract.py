from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


VA_MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(VA_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(VA_MODEL_ROOT))

train_model_module = import_module("train_model")


def _expected_resume_manifest() -> dict[str, object]:
    """Build the architecture-sensitive subset recorded before Trainer resume."""

    return {
        "architecture_manifest_version": 3,
        "model": "qwen3.5-0.8b",
        "loss": "heteroscedastic+ccc",
        "output_dim": 4,
        "dtype": "float32",
        "model_id": "Qwen/fake",
        "model_revision": "decoder-commit",
        "gaze_fusion": "prefix-concat",
        "gaze_concat_order": "eye_start, compact_trt_gaze, eye_end, text",
        "pooling_position": "last_valid_text_token_after_gaze_prefix",
        "et_model_id": "ET/fake",
        "et_revision": "et-commit",
        "et_filename": "et.safetensors",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "attn_implementation": None,
        "gradient_checkpointing": True,
        "max_length": 200,
        "train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "hetero_mse_weight": 0.1,
        "hetero_ccc_weight": 0.1,
        "hetero_logvar_min": -5.0,
        "hetero_logvar_max": 3.0,
        "seed": 42,
        "held_out_fold": 1,
        "training_fold": 2,
        "fold_sha256": {"full_dataset_fold1.csv": "abc"},
        "excluded_dataset_names": [],
        "dataset_counts_after_filter": {"Emobank": 10},
    }


def test_cli_defaults_to_prefix_and_rejects_postfix():
    parser = train_model_module._build_parser()

    assert parser.parse_args([]).gaze_fusion == "prefix-concat"
    with pytest.raises(SystemExit):
        parser.parse_args(["--gaze-fusion", "postfix-concat"])


def test_resume_contract_accepts_matching_prefix_checkpoint(tmp_path):
    fold_output = tmp_path / "run" / "heldout_fold1"
    checkpoint = fold_output / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    expected = _expected_resume_manifest()
    (fold_output / "run_manifest.json").write_text(
        json.dumps(expected),
        encoding="utf-8",
    )
    args = SimpleNamespace(resume_from_checkpoint=str(checkpoint))

    train_model_module._validate_resume_contract(args, fold_output, expected)


def test_resume_contract_rejects_old_postfix_manifest(tmp_path):
    fold_output = tmp_path / "run" / "heldout_fold1"
    checkpoint = fold_output / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    recorded = _expected_resume_manifest()
    recorded.update(
        {
            "architecture_manifest_version": 2,
            "gaze_fusion": "postfix-concat",
            "gaze_concat_order": "text, eye_start, compact_trt_gaze, eye_end",
            "pooling_position": "eye_end",
        }
    )
    (fold_output / "run_manifest.json").write_text(
        json.dumps(recorded),
        encoding="utf-8",
    )
    args = SimpleNamespace(resume_from_checkpoint=str(checkpoint))

    with pytest.raises(ValueError, match="refusing to reinterpret"):
        train_model_module._validate_resume_contract(
            args,
            fold_output,
            _expected_resume_manifest(),
        )


def test_resume_contract_rejects_checkpoint_from_another_fold(tmp_path):
    fold_output = tmp_path / "run" / "heldout_fold1"
    checkpoint = tmp_path / "run" / "heldout_fold2" / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    args = SimpleNamespace(resume_from_checkpoint=str(checkpoint))

    with pytest.raises(ValueError, match="selected held-out fold"):
        train_model_module._validate_resume_contract(
            args,
            fold_output,
            _expected_resume_manifest(),
        )

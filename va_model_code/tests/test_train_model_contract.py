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
        "architecture_manifest_version": 6,
        "model": "qwen3.5-0.8b",
        "loss": "mse",
        "output_dim": 2,
        "dtype": "float32",
        "model_id": "Qwen/fake",
        "model_revision": "decoder-commit",
        "finetuning_mode": "lora",
        "gaze_fusion": "prefix-concat",
        "gaze_features": ["nFix", "TRT"],
        "gaze_feature_indices": [0, 3],
        "features_used": [1, 0, 0, 1, 0],
        "gaze_concat_order": "eye_start, compact_selected_gaze, eye_end, text",
        "pooling_position": "last_valid_text_token_after_gaze_prefix",
        "output_activation": "hard_sigmoid",
        "paired_ablation_seed_policy": {
            "purpose": "pair baseline and gaze initialization within each held-out fold",
            "formula": "base_seed + held_out_fold - 1",
            "paper_protocol_requirement": False,
        },
        "et_model_id": "ET/fake",
        "et_revision": "et-commit",
        "et_filename": "et.safetensors",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "attn_implementation": None,
        "group_by_length": True,
        "gradient_checkpointing": True,
        "max_length": 200,
        "train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "seed": 42,
        "fold_seed": 42,
        "held_out_fold": 1,
        "training_fold": 2,
        "fold_sha256": {"full_dataset_fold1.csv": "abc"},
        "excluded_dataset_names": [],
        "dataset_counts_after_filter": {"Emobank": 10},
    }


def test_cli_defaults_to_prefix_and_requires_explicit_training_loss():
    parser = train_model_module._build_parser()

    defaults = parser.parse_args([])

    assert defaults.gaze_fusion == "prefix-concat"
    assert defaults.gaze_features == ("TRT",)
    assert defaults.finetuning_mode == "lora"
    assert defaults.group_by_length is True
    assert defaults.loss is None
    assert defaults.run_name is None
    with pytest.raises(ValueError, match="must be explicit"):
        train_model_module._validate_args(defaults)
    train_model_module._validate_args(parser.parse_args(["--dry-run"]))
    with pytest.raises(SystemExit):
        parser.parse_args(["--gaze-fusion", "postfix-concat"])
    with pytest.raises(SystemExit):
        parser.parse_args(["qwen3.5-0.8b", "heteroscedastic+ccc"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--finetuning-mode", "adapter"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "Preds/old-layout"])


def test_cli_resolves_mode_specific_learning_rates():
    parser = train_model_module._build_parser()
    lora_args = parser.parse_args(["qwen3.5-0.8b", "mse"])
    full_args = parser.parse_args(
        ["qwen3.5-0.8b", "mse", "--finetuning-mode", "full"]
    )
    explicit_full_args = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--finetuning-mode",
            "full",
            "--learning-rate",
            "2e-6",
        ]
    )

    train_model_module._validate_args(lora_args)
    train_model_module._validate_args(full_args)
    train_model_module._validate_args(explicit_full_args)

    assert lora_args.learning_rate == pytest.approx(1e-4)
    assert full_args.learning_rate == pytest.approx(6e-6)
    assert explicit_full_args.learning_rate == pytest.approx(2e-6)
    assert full_args.lora_rank is None
    assert full_args.lora_alpha is None
    assert full_args.lora_dropout is None
    train_model_module._validate_args(full_args)


def test_cli_rejects_inapplicable_or_invalid_lora_settings():
    parser = train_model_module._build_parser()
    full_with_lora_override = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--finetuning-mode",
            "full",
            "--lora-rank",
            "8",
        ]
    )
    lora_with_invalid_dropout = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--finetuning-mode",
            "lora",
            "--lora-dropout",
            "1.0",
        ]
    )

    with pytest.raises(ValueError, match="LoRA-only"):
        train_model_module._validate_args(full_with_lora_override)
    with pytest.raises(ValueError, match="lora-dropout"):
        train_model_module._validate_args(lora_with_invalid_dropout)


def test_cli_canonicalizes_named_gaze_subsets_and_rejects_duplicates():
    parser = train_model_module._build_parser()
    args = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--gaze-features",
            "TRT",
            "nFix",
        ]
    )

    train_model_module._validate_args(args)

    assert args.gaze_features == ("nFix", "TRT")
    assert args.gaze_feature_indices == (0, 3)

    duplicate_args = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--gaze-features",
            "TRT",
            "TRT",
        ]
    )
    with pytest.raises(ValueError, match="duplicates"):
        train_model_module._validate_args(duplicate_args)


def test_cli_rejects_misleading_condition_names() -> None:
    parser = train_model_module._build_parser()
    gaze_named_baseline = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--gaze-fusion",
            "prefix-concat",
            "--run-name",
            "paper7_full_baseline_seed42",
        ]
    )
    baseline_named_gaze = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--gaze-fusion",
            "none",
            "--run-name",
            "paper7_full_gaze_TRT_seed42",
        ]
    )
    lora_named_full = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--finetuning-mode",
            "lora",
            "--run-name",
            "paper7_full_gaze_seed42",
        ]
    )
    included_named_excluded = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--run-name",
            "paper7_no_iemocap_gaze_seed42",
        ]
    )
    wrong_seed = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--seed",
            "43",
            "--run-name",
            "paper7_gaze_seed42",
        ]
    )

    with pytest.raises(ValueError, match="says baseline"):
        train_model_module._validate_args(gaze_named_baseline)
    with pytest.raises(ValueError, match="says gaze"):
        train_model_module._validate_args(baseline_named_gaze)
    with pytest.raises(ValueError, match="says full"):
        train_model_module._validate_args(lora_named_full)
    with pytest.raises(ValueError, match="exclusion flag is absent"):
        train_model_module._validate_args(included_named_excluded)
    with pytest.raises(ValueError, match="seed tag"):
        train_model_module._validate_args(wrong_seed)


@pytest.mark.parametrize("warmup_ratio", (-0.01, 1.0))
def test_training_contract_rejects_invalid_warmup_ratio(warmup_ratio):
    args = train_model_module._build_parser().parse_args(
        ["qwen3.5-0.8b", "mse", "--warmup-ratio", str(warmup_ratio)]
    )

    with pytest.raises(ValueError, match="interval"):
        train_model_module._validate_args(args)


@pytest.mark.parametrize(
    (
        "transformers_version",
        "expected_warmup_argument",
        "training_argument_names",
        "expected_sampling_argument",
        "expected_sampling_value",
    ),
    (
        (
            "4.49.0",
            "warmup_ratio",
            {"group_by_length"},
            "group_by_length",
            True,
        ),
        (
            "5.14.1",
            "warmup_steps",
            {"train_sampling_strategy"},
            "train_sampling_strategy",
            "group_by_length",
        ),
    ),
)
def test_training_arguments_are_compatible_across_transformers_versions(
    monkeypatch,
    tmp_path,
    transformers_version,
    expected_warmup_argument,
    training_argument_names,
    expected_sampling_argument,
    expected_sampling_value,
):
    captured_kwargs = {}

    def capture_training_arguments(**kwargs):
        captured_kwargs.update(kwargs)
        return captured_kwargs

    monkeypatch.setattr(
        train_model_module,
        "_package_version",
        lambda distribution: transformers_version,
    )
    monkeypatch.setattr(
        train_model_module,
        "TrainingArguments",
        capture_training_arguments,
    )
    monkeypatch.setattr(
        train_model_module,
        "_training_argument_names",
        lambda: frozenset(training_argument_names),
    )
    args = train_model_module._build_parser().parse_args(
        ["qwen3.5-0.8b", "mse"]
    )

    result = train_model_module._training_arguments(
        args,
        tmp_path / "heldout_fold1",
        bf16=False,
        fp16=False,
        fold_seed=43,
    )

    assert result is captured_kwargs
    assert captured_kwargs[expected_warmup_argument] == pytest.approx(0.1)
    assert (
        {"warmup_ratio", "warmup_steps"} - {expected_warmup_argument}
    ).isdisjoint(captured_kwargs)
    assert "overwrite_output_dir" not in captured_kwargs
    assert "save_safetensors" not in captured_kwargs
    assert captured_kwargs["seed"] == 43
    assert "data_seed" not in captured_kwargs
    assert captured_kwargs[expected_sampling_argument] == expected_sampling_value
    assert (
        {"group_by_length", "train_sampling_strategy"}
        - {expected_sampling_argument}
    ).isdisjoint(captured_kwargs)


def test_transformers_v5_can_disable_length_grouping(monkeypatch, tmp_path):
    captured_kwargs = {}

    monkeypatch.setattr(
        train_model_module,
        "_package_version",
        lambda distribution: "5.14.1",
    )
    monkeypatch.setattr(
        train_model_module,
        "_training_argument_names",
        lambda: frozenset({"train_sampling_strategy"}),
    )
    monkeypatch.setattr(
        train_model_module,
        "TrainingArguments",
        lambda **kwargs: captured_kwargs.update(kwargs) or captured_kwargs,
    )
    args = train_model_module._build_parser().parse_args(
        ["qwen3.5-0.8b", "mse", "--no-group-by-length"]
    )

    train_model_module._training_arguments(
        args,
        tmp_path / "heldout_fold1",
        bf16=False,
        fp16=False,
    )

    assert captured_kwargs["train_sampling_strategy"] == "random"


def test_training_argument_names_inspects_the_runtime_constructor(monkeypatch):
    class FakeTrainingArguments:
        def __init__(
            self,
            output_dir=None,
            train_sampling_strategy="random",
        ):
            self.output_dir = output_dir
            self.train_sampling_strategy = train_sampling_strategy

    monkeypatch.setattr(
        train_model_module,
        "TrainingArguments",
        FakeTrainingArguments,
    )

    names = train_model_module._training_argument_names()

    assert "output_dir" in names
    assert "train_sampling_strategy" in names
    assert "group_by_length" not in names


def test_cuda_memory_reporting_contract(monkeypatch):
    cuda = train_model_module.torch.cuda
    calls = []
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "current_device", lambda: 1)
    monkeypatch.setattr(cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(
        cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", device)),
    )
    monkeypatch.setattr(cuda, "memory_allocated", lambda device: 1 * 1024**3)
    monkeypatch.setattr(cuda, "memory_reserved", lambda device: 2 * 1024**3)
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda device: 3 * 1024**3)
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda device: 4 * 1024**3)
    monkeypatch.setattr(cuda, "get_device_name", lambda device: "Fake GPU")

    assert train_model_module._reset_cuda_peak_memory(use_cpu=False) is True
    snapshot = train_model_module._cuda_memory_snapshot(use_cpu=False)

    assert calls == ["empty_cache", ("reset", 1)]
    assert snapshot["cuda_enabled"] is True
    assert snapshot["device_index"] == 1
    assert snapshot["device_name"] == "Fake GPU"
    assert snapshot["peak_allocated_bytes"] == 3 * 1024**3
    assert snapshot["peak_reserved_gib"] == pytest.approx(4.0)
    assert train_model_module._cuda_memory_snapshot(use_cpu=True) == {
        "cuda_enabled": False
    }


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


def test_resume_contract_accepts_legacy_v5_lora_checkpoint(tmp_path):
    fold_output = tmp_path / "run" / "heldout_fold1"
    checkpoint = fold_output / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    expected = _expected_resume_manifest()
    recorded = dict(expected)
    recorded["architecture_manifest_version"] = 5
    recorded.pop("finetuning_mode")
    (fold_output / "run_manifest.json").write_text(
        json.dumps(recorded),
        encoding="utf-8",
    )
    args = SimpleNamespace(resume_from_checkpoint=str(checkpoint))

    train_model_module._validate_resume_contract(args, fold_output, expected)


def test_resume_contract_rejects_finetuning_mode_mismatch(tmp_path):
    fold_output = tmp_path / "run" / "heldout_fold1"
    checkpoint = fold_output / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    recorded = _expected_resume_manifest()
    recorded["finetuning_mode"] = "lora"
    (fold_output / "run_manifest.json").write_text(
        json.dumps(recorded),
        encoding="utf-8",
    )
    expected = _expected_resume_manifest()
    expected["finetuning_mode"] = "full"
    expected["learning_rate"] = 6e-6
    args = SimpleNamespace(resume_from_checkpoint=str(checkpoint))

    with pytest.raises(ValueError, match="finetuning_mode"):
        train_model_module._validate_resume_contract(args, fold_output, expected)


def test_full_resume_contract_ignores_inactive_lora_hyperparameters(tmp_path):
    fold_output = tmp_path / "run" / "heldout_fold1"
    checkpoint = fold_output / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    expected = _expected_resume_manifest()
    expected.update(
        {
            "finetuning_mode": "full",
            "learning_rate": 6e-6,
            "lora_rank": None,
            "lora_alpha": None,
            "lora_dropout": None,
        }
    )
    recorded = dict(expected)
    recorded.update(
        {
            "lora_rank": 1,
            "lora_alpha": 1,
            "lora_dropout": 0.9,
        }
    )
    (fold_output / "run_manifest.json").write_text(
        json.dumps(recorded),
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
            "gaze_concat_order": "text, eye_start, compact_selected_gaze, eye_end",
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


def test_resume_contract_rejects_old_four_output_manifest(tmp_path):
    fold_output = tmp_path / "run" / "heldout_fold1"
    checkpoint = fold_output / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    recorded = _expected_resume_manifest()
    recorded.update(
        {
            "architecture_manifest_version": 3,
            "loss": "heteroscedastic+ccc",
            "output_dim": 4,
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

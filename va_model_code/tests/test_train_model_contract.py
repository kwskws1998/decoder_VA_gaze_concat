from __future__ import annotations

from datetime import datetime
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
    assert defaults.gaze_redistribution == "none"
    assert defaults.redistribution_learning_rate is None
    assert defaults.sentence_only is False
    assert train_model_module._redistribution_config(
        SimpleNamespace(**vars(defaults), gaze_feature_indices=(3,))
    ) == {"method": "none"}
    assert defaults.finetuning_mode == "lora"
    assert defaults.precision == "auto"
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


def test_cli_resolves_and_validates_redistribution_optimizer_policy():
    parser = train_model_module._build_parser()
    defaults = parser.parse_args(
        ["qwen3.5-0.8b", "mse", "--gaze-redistribution", "asym-gaussian"]
    )
    explicit = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--gaze-redistribution",
            "asym-gaussian",
            "--redistribution-learning-rate",
            "2e-3",
        ]
    )

    train_model_module._validate_args(defaults)
    train_model_module._validate_args(explicit)
    assert defaults.redistribution_learning_rate == pytest.approx(1e-3)
    assert explicit.redistribution_learning_rate == pytest.approx(2e-3)
    assert defaults.redistribution_weight_decay == pytest.approx(0.0)
    assert explicit.redistribution_weight_decay == pytest.approx(0.0)

    disabled = parser.parse_args(
        ["qwen3.5-0.8b", "mse", "--redistribution-learning-rate", "1e-3"]
    )
    with pytest.raises(ValueError, match="requires --gaze-redistribution"):
        train_model_module._validate_args(disabled)

    for invalid in ("0", "-0.1", "nan", "inf"):
        args = parser.parse_args(
            [
                "qwen3.5-0.8b",
                "mse",
                "--gaze-redistribution",
                "asym-gaussian",
                "--redistribution-learning-rate",
                invalid,
            ]
        )
        with pytest.raises(ValueError, match="finite and positive"):
            train_model_module._validate_args(args)


def test_sentence_only_dry_run_filters_both_folds_without_model_loading(
    tmp_path, monkeypatch, capsys
):
    import pandas as pd

    data_root = tmp_path / "paper_folds"
    data_root.mkdir()
    sources = (
        "EmoTales sentences",
        "Emobank",
        "fb",
        "GlasgowNorms",
        "IEMOCAP sentences",
        "nrc-vad",
        "word ratings ENG",
    )
    for fold in (1, 2):
        pd.DataFrame(
            {
                "index": [fold * 100 + offset for offset in range(len(sources))],
                "text": [f"text-{fold}-{offset}" for offset in range(len(sources))],
                "dataset_of_origin": sources,
                "valence": [0.5] * len(sources),
                "arousal": [0.5] * len(sources),
            }
        ).to_csv(data_root / f"full_dataset_fold{fold}.csv", sep="\t", index=False)
    monkeypatch.setattr(
        train_model_module,
        "load_auto_tokenizer",
        lambda *args, **kwargs: pytest.fail("Dry-run attempted to load a tokenizer."),
    )
    args = train_model_module._build_parser().parse_args(
        [
            "--dry-run",
            "--precision",
            "fp32",
            "--use-cpu",
            "--data-dir",
            str(data_root),
            "--sentence-only",
            "--no-iemocap",
            "--run-name",
            "paper7_sentence_only_no_iemocap_gaze_seed42",
        ]
    )

    assert train_model_module.run(args) is None
    output = capsys.readouterr().out
    assert "EmoTales sentences\t2" in output
    assert "Emobank\t2" in output
    assert "fb\t2" in output
    assert "GlasgowNorms\t" not in output
    assert "Dataset scope: EmoTales sentences, Emobank, fb" in output


def test_runtime_precision_resolves_explicit_fp32_and_cuda_modes(monkeypatch):
    cuda = train_model_module.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "is_bf16_supported", lambda: True)

    assert train_model_module._runtime_precision(False, "fp32") == (
        train_model_module.torch.float32,
        False,
        False,
    )
    assert train_model_module._runtime_precision(False, "bf16") == (
        train_model_module.torch.bfloat16,
        True,
        False,
    )
    assert train_model_module._runtime_precision(False, "fp16") == (
        train_model_module.torch.float16,
        False,
        True,
    )


def test_runtime_precision_rejects_mixed_precision_without_cuda(monkeypatch):
    monkeypatch.setattr(train_model_module.torch.cuda, "is_available", lambda: False)

    assert train_model_module._runtime_precision(False, "fp32") == (
        train_model_module.torch.float32,
        False,
        False,
    )
    with pytest.raises(ValueError, match="requires CUDA"):
        train_model_module._runtime_precision(False, "bf16")


def test_cli_rejects_precision_run_name_mismatch():
    parser = train_model_module._build_parser()
    args = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--precision",
            "fp32",
            "--run-name",
            "qwen_lora_gaze_bf16_seed42",
        ]
    )

    with pytest.raises(ValueError, match="precision tag"):
        train_model_module._validate_args(args)


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


def test_cli_redistribution_config_and_names_keep_raw_condition_unchanged():
    from decoder_va.paths import condition_slug, default_run_name

    parser = train_model_module._build_parser()
    args = parser.parse_args(
        [
            "qwen3.5-0.8b", "mse", "--gaze-redistribution", "asym-gaussian",
            "--redistribution-sigma-left", "0.75",
            "--redistribution-sigma-right", "1.5",
            "--gaze-features", "TRT", "nFix",
        ]
    )
    train_model_module._validate_args(args)
    train_model_module._validate_args(args)
    contract = train_model_module._redistribution_config(args)
    assert contract["method"] == "asym-gaussian"
    assert contract["init_sigma_left"] == 0.75
    assert contract["init_sigma_right"] == 1.5
    assert contract["feature"] == "TRT"
    assert contract["mask_policy"] == "valid_gaze_sources_and_targets"
    shared = {
        "model": "qwen3.5-0.8b",
        "finetuning_mode": "full",
        "gaze_fusion": "prefix-concat",
        "gaze_features": ("TRT",),
        "seed": 43,
        "no_iemocap": True,
    }
    original = condition_slug(**shared)
    assert original == "qwen3.5-0.8b_full_gaze_TRT_no_iemocap_seed43"
    assert condition_slug(**shared, gaze_redistribution={"method": "none"}) == original
    enabled = condition_slug(**shared, gaze_redistribution=contract)
    assert enabled != original
    assert "gaze_TRT_redistribution_asym-gaussian" in enabled
    timestamp = datetime(2026, 9, 4, 12)
    assert default_run_name(
        **shared, timestamp=timestamp, gaze_redistribution=contract
    ).endswith(enabled)


@pytest.mark.parametrize(
    "options",
    (
        ["--gaze-fusion", "none"],
        ["--gaze-features", "FFD", "GPT"],
        ["--redistribution-sigma-left", "0"],
        ["--redistribution-sigma-right", "-1"],
        ["--redistribution-min-sigma", "0"],
        ["--redistribution-sigma-left", "nan"],
        ["--redistribution-sigma-right", "inf"],
        ["--redistribution-min-sigma", "nan"],
    ),
)
def test_cli_rejects_invalid_redistribution_before_loading(options):
    args = train_model_module._build_parser().parse_args(
        ["qwen3.5-0.8b", "mse", "--gaze-redistribution", "asym-gaussian", *options]
    )
    with pytest.raises((TypeError, ValueError)):
        train_model_module._validate_args(args)


@pytest.mark.parametrize(
    "option", ("--redistribution-sigma-left", "--redistribution-sigma-right", "--redistribution-min-sigma")
)
def test_cli_rejects_ignored_redistribution_options(option):
    args = train_model_module._build_parser().parse_args(
        ["qwen3.5-0.8b", "mse", option, "0.5"]
    )
    with pytest.raises(ValueError, match="require --gaze-redistribution"):
        train_model_module._validate_args(args)


def test_cli_rejects_redistribution_name_when_disabled():
    args = train_model_module._build_parser().parse_args(
        ["qwen3.5-0.8b", "mse", "--run-name", "qwen_gaze_TRT_redistribution_seed42"]
    )
    with pytest.raises(ValueError, match="says redistribution"):
        train_model_module._validate_args(args)


@pytest.mark.parametrize("method", ("none", "asym-gaussian"))
def test_training_records_and_passes_canonical_redistribution(tmp_path, monkeypatch, method):
    """Run the real two-fold orchestration with offline model/trainer stand-ins."""

    import pandas as pd

    data_root = tmp_path / "data"
    data_root.mkdir()
    for fold in (1, 2):
        pd.DataFrame(
            {
                "index": [fold * 3, fold * 3 + 1, fold * 3 + 2],
                "text": [
                    f"short text {fold}",
                    f"second text {fold}",
                    f"third text {fold}",
                ],
                "dataset_of_origin": ["EmoTales sentences", "Emobank", "fb"],
                "valence": [0.2, 0.8, 0.4],
                "arousal": [0.3, 0.7, 0.5],
            }
        ).to_csv(data_root / f"full_dataset_fold{fold}.csv", sep="\t", index=False)
    build_contracts = []
    trainer_redistribution_lrs = []
    run_root = tmp_path / "test_run"

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def trainable_parameter_summary(self):
            return {
                "total_parameters": 8,
                "trainable_parameters": 8,
                "trainable_fraction": 1.0,
                "gaze_redistribution_trainable_parameters": 2 if method != "none" else 0,
            }

        def save_architecture_manifest(self, directory):
            Path(directory, "decoder_va_architecture.json").write_text(
                json.dumps({"gaze_redistribution": self.config}), encoding="utf-8"
            )

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            trainer_redistribution_lrs.append(
                kwargs["redistribution_learning_rate"]
            )

        def train(self, resume_from_checkpoint):
            assert resume_from_checkpoint is None

        def save_model(self, directory):
            Path(directory).mkdir()

        def predict(self, frame):
            labels = frame[["valence", "arousal"]].to_numpy()
            return SimpleNamespace(label_ids=labels, predictions=labels, metrics={})

    def build_model(tokenizer, **kwargs):
        build_contracts.append(kwargs["gaze_redistribution"])
        return FakeModel(kwargs["gaze_redistribution"])

    monkeypatch.setattr(train_model_module, "_resolve_output_dir", lambda args: run_root)
    monkeypatch.setattr(train_model_module, "load_auto_tokenizer", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(train_model_module, "VABatchCollator", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_model_module, "TokenizedVADataset", lambda frame, *args, **kwargs: frame)
    monkeypatch.setattr(train_model_module, "_training_arguments", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_model_module, "build_qwen_va_model", build_model)
    monkeypatch.setattr(train_model_module, "VARegressionTrainer", FakeTrainer)
    args = train_model_module._build_parser().parse_args(
        [
            "qwen3.5-0.8b", "mse", "--data-dir", str(data_root),
            "--finetuning-mode", "full", "--precision", "fp32", "--use-cpu",
            "--gaze-redistribution", method, "--sentence-only", "--run-name", "test_run",
        ]
    )
    assert train_model_module.run(args) == run_root
    expected = train_model_module.redistribution_contract(method)
    assert build_contracts == [expected, expected]
    root_manifest = json.loads((run_root / "training_parameters.json").read_text())
    assert root_manifest["architecture_manifest_version"] == 7
    assert root_manifest["gaze_redistribution"] == expected
    assert root_manifest["sentence_only"] is True
    assert root_manifest["dataset_counts_after_filter"] == {
        "EmoTales sentences": 2,
        "Emobank": 2,
        "fb": 2,
    }
    assert trainer_redistribution_lrs == [
        1e-3 if method != "none" else None,
        1e-3 if method != "none" else None,
    ]
    assert root_manifest["redistribution_learning_rate"] == (
        pytest.approx(1e-3) if method != "none" else None
    )
    assert root_manifest["redistribution_weight_decay"] == (
        pytest.approx(0.0) if method != "none" else None
    )
    for fold in (1, 2):
        fold_manifest = json.loads((run_root / f"heldout_fold{fold}" / "run_manifest.json").read_text())
        assert fold_manifest["gaze_redistribution"] == expected
        assert fold_manifest["gaze_redistribution_trainable_parameters"] == (2 if method != "none" else 0)


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
    omitted_sentence_scope = parser.parse_args(
        [
            "qwen3.5-0.8b",
            "mse",
            "--run-name",
            "paper7_sentence_only_gaze_seed42",
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
    with pytest.raises(ValueError, match="sentence_only"):
        train_model_module._validate_args(omitted_sentence_scope)


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
            "5.16.1",
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
    assert captured_kwargs["tf32"] is None
    assert "data_seed" not in captured_kwargs
    assert captured_kwargs[expected_sampling_argument] == expected_sampling_value
    assert (
        {"group_by_length", "train_sampling_strategy"}
        - {expected_sampling_argument}
    ).isdisjoint(captured_kwargs)


def test_fp32_training_arguments_disable_tf32(monkeypatch, tmp_path):
    captured_kwargs = {}
    monkeypatch.setattr(train_model_module, "_package_version", lambda name: "5.16.1")
    monkeypatch.setattr(
        train_model_module,
        "_training_argument_names",
        lambda: frozenset({"train_sampling_strategy", "tf32"}),
    )
    monkeypatch.setattr(
        train_model_module,
        "TrainingArguments",
        lambda **kwargs: captured_kwargs.update(kwargs) or captured_kwargs,
    )
    args = train_model_module._build_parser().parse_args(
        ["qwen3.5-0.8b", "mse", "--precision", "fp32"]
    )

    train_model_module._training_arguments(
        args,
        tmp_path / "heldout_fold1",
        bf16=False,
        fp16=False,
    )

    assert captured_kwargs["bf16"] is False
    assert captured_kwargs["fp16"] is False
    assert captured_kwargs["tf32"] is False


def test_transformers_v5_can_disable_length_grouping(monkeypatch, tmp_path):
    captured_kwargs = {}

    monkeypatch.setattr(
        train_model_module,
        "_package_version",
        lambda distribution: "5.16.1",
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


def _check_resume_pair(tmp_path, recorded, expected, checkpoint_tensors=None):
    """Exercise the real resume gate on two supplied on-disk contracts."""

    fold_output = tmp_path / "run" / "heldout_fold1"
    checkpoint = fold_output / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    if checkpoint_tensors is not None:
        from safetensors.torch import save_file

        save_file(checkpoint_tensors, checkpoint / "model.safetensors")
    (fold_output / "run_manifest.json").write_text(json.dumps(recorded), encoding="utf-8")
    train_model_module._validate_resume_contract(
        SimpleNamespace(resume_from_checkpoint=str(checkpoint)), fold_output, expected
    )


def _current_resume_manifest(method="none"):
    """Build an explicit schema-7 contract for compatibility regression tests."""

    manifest = _expected_resume_manifest()
    manifest["architecture_manifest_version"] = train_model_module.ARCHITECTURE_MANIFEST_VERSION
    manifest["gaze_redistribution"] = train_model_module.redistribution_contract(
        method, gaze_fusion="prefix-concat", feature_indices=(0, 3)
    )
    if method != "none":
        manifest["redistribution_learning_rate"] = 1e-3
        manifest["redistribution_weight_decay"] = 0.0
    return manifest


@pytest.mark.parametrize("legacy_version", (5, 6))
def test_resume_legacy_checkpoints_only_upgrade_with_redistribution_disabled(tmp_path, legacy_version):
    recorded = _expected_resume_manifest()
    recorded["architecture_manifest_version"] = legacy_version
    if legacy_version == 5:
        recorded.pop("finetuning_mode")
    _check_resume_pair(tmp_path, recorded, _current_resume_manifest())


@pytest.mark.parametrize("method", ("none", "asym-gaussian"))
def test_resume_current_requires_explicit_redistribution_metadata(tmp_path, method):
    recorded = _current_resume_manifest(method)
    recorded.pop("gaze_redistribution")
    with pytest.raises(ValueError, match="missing gaze_redistribution"):
        _check_resume_pair(tmp_path, recorded, _current_resume_manifest(method))


@pytest.mark.parametrize("legacy_version", (5, 6))
def test_resume_legacy_cannot_reset_into_redistribution(tmp_path, legacy_version):
    recorded = _expected_resume_manifest()
    recorded["architecture_manifest_version"] = legacy_version
    if legacy_version == 5:
        recorded.pop("finetuning_mode")
    with pytest.raises(ValueError, match="gaze_redistribution"):
        _check_resume_pair(tmp_path, recorded, _current_resume_manifest("asym-gaussian"))


def test_resume_rejects_enabled_contract_falsely_tagged_as_legacy(tmp_path):
    recorded = _current_resume_manifest("asym-gaussian")
    recorded["architecture_manifest_version"] = 6
    with pytest.raises(ValueError, match="Legacy resume manifests"):
        _check_resume_pair(tmp_path, recorded, _current_resume_manifest("asym-gaussian"))


@pytest.mark.parametrize("field", ("method", "init_sigma_left", "init_sigma_right", "min_sigma"))
def test_resume_rejects_changed_redistribution_configuration(tmp_path, field):
    recorded = _current_resume_manifest("asym-gaussian")
    expected = _current_resume_manifest("asym-gaussian")
    if field == "method":
        expected["gaze_redistribution"] = {"method": "none"}
    else:
        expected["gaze_redistribution"][field] *= 2
    with pytest.raises(ValueError, match="gaze_redistribution"):
        _check_resume_pair(tmp_path, recorded, expected)


def test_resume_accepts_matching_enabled_contract(tmp_path):
    _check_resume_pair(
        tmp_path,
        _current_resume_manifest("asym-gaussian"),
        _current_resume_manifest("asym-gaussian"),
        checkpoint_tensors=_sigma_checkpoint_tensors(),
    )


@pytest.mark.parametrize(
    "field,new_value",
    (
        ("redistribution_learning_rate", 2e-3),
        ("redistribution_weight_decay", 0.1),
    ),
)
def test_resume_rejects_changed_redistribution_optimizer_policy(
    tmp_path, field, new_value
):
    recorded = _current_resume_manifest("asym-gaussian")
    expected = _current_resume_manifest("asym-gaussian")
    expected[field] = new_value

    with pytest.raises(ValueError, match=field):
        _check_resume_pair(
            tmp_path,
            recorded,
            expected,
            checkpoint_tensors=_sigma_checkpoint_tensors(),
        )


@pytest.mark.parametrize(
    "field", ("redistribution_learning_rate", "redistribution_weight_decay")
)
def test_resume_rejects_old_redistribution_optimizer_layout(tmp_path, field):
    recorded = _current_resume_manifest("asym-gaussian")
    recorded.pop(field)

    with pytest.raises(ValueError, match=field):
        _check_resume_pair(
            tmp_path,
            recorded,
            _current_resume_manifest("asym-gaussian"),
            checkpoint_tensors=_sigma_checkpoint_tensors(),
        )


def test_resume_rejects_changed_sentence_only_scope(tmp_path):
    recorded = _current_resume_manifest()
    expected = _current_resume_manifest()
    recorded["sentence_only"] = False
    expected["sentence_only"] = True

    with pytest.raises(ValueError, match="sentence_only"):
        _check_resume_pair(tmp_path, recorded, expected)


def _sigma_checkpoint_tensors():
    """Build safe scalar learned-width state, deliberately different from initialization."""

    torch = train_model_module.torch
    return {
        "gaze_redistributor.kernel.log_sigma_left": torch.tensor(-0.6, dtype=torch.float32),
        "gaze_redistributor.kernel.log_sigma_right": torch.tensor(0.8, dtype=torch.float32),
        "backbone.dummy": torch.zeros(1),
    }


def test_enabled_resume_rejects_missing_safe_weights_before_metadata_mutation(tmp_path):
    recorded = _current_resume_manifest("asym-gaussian")
    with pytest.raises(ValueError, match="regular single-file"):
        _check_resume_pair(tmp_path, recorded, dict(recorded))
    assert json.loads((tmp_path / "run" / "heldout_fold1" / "run_manifest.json").read_text()) == recorded


@pytest.mark.parametrize(
    "defect, error",
    (
        ("both_missing", "state keys mismatch"),
        ("left_missing", "state keys mismatch"),
        ("unexpected_key", "state keys mismatch"),
        ("vector", "must be scalar"),
        ("bf16", "must be FP32"),
        ("fp64", "must be FP32"),
        ("nan", "must be finite"),
        ("inf", "must be finite"),
    ),
)
def test_enabled_resume_rejects_corrupt_sigma_state(tmp_path, defect, error):
    torch = train_model_module.torch
    state = _sigma_checkpoint_tensors()
    left = "gaze_redistributor.kernel.log_sigma_left"
    right = "gaze_redistributor.kernel.log_sigma_right"
    if defect == "both_missing":
        state.pop(left)
        state.pop(right)
    elif defect == "left_missing":
        state.pop(left)
    elif defect == "unexpected_key":
        state["gaze_redistributor.sigma_left"] = torch.tensor(1.0)
    elif defect == "vector":
        state[left] = torch.tensor([0.0])
    elif defect == "bf16":
        state[right] = state[right].to(torch.bfloat16)
    elif defect == "fp64":
        state[left] = state[left].to(torch.float64)
    else:
        state[left] = torch.tensor(float(defect), dtype=torch.float32)
    recorded = _current_resume_manifest("asym-gaussian")
    with pytest.raises(ValueError, match=error):
        _check_resume_pair(tmp_path, recorded, dict(recorded), checkpoint_tensors=state)
    assert json.loads((tmp_path / "run" / "heldout_fold1" / "run_manifest.json").read_text()) == recorded


@pytest.mark.parametrize("defect", ("symlink", "malformed", "sharded", "pickle", "adapter"))
def test_enabled_resume_rejects_unsafe_or_unsupported_weight_layout(tmp_path, defect):
    from safetensors.torch import save_file

    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    if defect == "symlink":
        target = tmp_path / "linked.safetensors"
        save_file(_sigma_checkpoint_tensors(), target)
        weights.symlink_to(target)
    elif defect == "malformed":
        weights.write_bytes(b"not safetensors")
    else:
        save_file(_sigma_checkpoint_tensors(), weights)
        alternate = {
            "sharded": "model.safetensors.index.json",
            "pickle": "pytorch_model.bin",
            "adapter": "adapter_model.safetensors",
        }[defect]
        (checkpoint / alternate).write_bytes(b"not loaded")
    with pytest.raises(ValueError, match="gaze_redistribution"):
        train_model_module._validate_redistribution_checkpoint(checkpoint)


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

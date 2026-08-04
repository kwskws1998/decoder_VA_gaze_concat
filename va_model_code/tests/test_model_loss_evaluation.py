from __future__ import annotations

import json
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from safetensors.torch import save_file
from transformers import TrainingArguments, default_data_collator
from transformers.modeling_outputs import BaseModelOutput

import va_model_code.decoder_va.model as model_module
from va_model_code.decoder_va.evaluation import (
    calculate_va_metrics,
    prediction_frame,
    write_oof_reports,
)
from va_model_code.decoder_va.dataset import TokenizedVADataset
from va_model_code.decoder_va.losses import (
    concordance_correlation_coefficient,
    va_regression_loss,
)
from va_model_code.decoder_va.model import (
    ARCHITECTURE_MANIFEST_FILENAME,
    ARCHITECTURE_MANIFEST_VERSION,
    GAZE_PREFIX_ORDER,
    GAZE_PREFIX_POOLING,
    OUTPUT_ACTIVATION,
    SAFE_WEIGHTS_FILENAME,
    DecoderVARegressor,
    build_qwen_va_model,
    load_saved_decoder_va_model,
    load_qwen_backbone_with_lora,
)
from va_model_code.decoder_va.trainer import VARegressionTrainer


class FakeCausalBackbone(nn.Module):
    def __init__(self, hidden_size: int = 6):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            use_cache=True,
            model_type="fake-causal",
        )
        self.embedding = nn.Embedding(32, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_call = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids,
        inputs_embeds,
        attention_mask,
        position_ids,
        use_cache,
        return_dict,
    ):
        hidden = self.projection(inputs_embeds).cumsum(dim=1)
        self.last_call = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds.detach().clone(),
            "attention_mask": attention_mask.detach().clone(),
            "position_ids": position_ids.detach().clone(),
            "use_cache": use_cache,
            "return_dict": return_dict,
            "last_hidden_state": hidden.detach().clone(),
        }
        return BaseModelOutput(last_hidden_state=hidden)


class FakeGazeProvider:
    def __init__(self, feature_indices=(3,)):
        self.feature_indices = tuple(feature_indices)

    def compute(self, input_ids, attention_mask):
        base = input_ids.to(dtype=torch.float32).unsqueeze(-1) / 10.0
        features = base.repeat(1, 1, len(self.feature_indices))
        mask = attention_mask.to(dtype=torch.bool)
        mask[:, 0] = False
        return features.masked_fill(~mask.unsqueeze(-1), 0.0), mask


def test_prefix_model_pools_last_text_and_returns_two_output_va():
    backbone = FakeCausalBackbone()
    model = DecoderVARegressor(
        backbone,
        gaze_provider=FakeGazeProvider(),
        gaze_fusion="prefix-concat",
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    pooled_inputs = []
    hook = model.regression_head.register_forward_pre_hook(
        lambda _module, arguments: pooled_inputs.append(arguments[0].detach().clone())
    )

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    hook.remove()

    assert output.logits.shape == (2, 2)
    assert torch.all((output.logits >= 0.0) & (output.logits <= 1.0))
    assert model.config.gaze_concat_order == GAZE_PREFIX_ORDER
    assert model.config.pooling_position == GAZE_PREFIX_POOLING
    assert (
        model.config.decoder_va_architecture_version
        == ARCHITECTURE_MANIFEST_VERSION
    )
    assert backbone.last_call["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0],
    ]
    assert backbone.last_call["position_ids"].tolist() == [
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 0, 0],
    ]
    assert backbone.last_call["input_ids"] is None
    assert backbone.last_call["use_cache"] is False
    original_text = backbone.embedding(input_ids)
    torch.testing.assert_close(
        backbone.last_call["inputs_embeds"][0, 4:7],
        original_text[0],
    )
    torch.testing.assert_close(
        backbone.last_call["inputs_embeds"][1, 3:5],
        original_text[1, :2],
    )
    expected_pooled = backbone.last_call["last_hidden_state"][
        torch.arange(2),
        torch.tensor([6, 4]),
    ]
    torch.testing.assert_close(pooled_inputs[0], expected_pooled)
    output.logits.sum().backward()
    assert model.eye_end.grad is not None
    assert model.gaze_projector[0].weight.grad is not None
    assert backbone.embedding.weight.grad[3].abs().sum() > 0
    assert backbone.embedding.weight.grad[5].abs().sum() > 0


def test_prefix_model_readout_depends_on_text_after_eye_end():
    class ConstantGazeProvider:
        def compute(self, input_ids, attention_mask):
            features = torch.ones((*input_ids.shape, 1), dtype=torch.float32)
            mask = attention_mask.to(dtype=torch.bool)
            mask[:, 0] = False
            return features, mask

    backbone = FakeCausalBackbone()
    model = DecoderVARegressor(
        backbone,
        gaze_provider=ConstantGazeProvider(),
        gaze_fusion="prefix-concat",
    )
    model.eval()
    pooled_inputs = []
    hook = model.regression_head.register_forward_pre_hook(
        lambda _module, arguments: pooled_inputs.append(arguments[0].detach().clone())
    )

    model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )
    first_eye_end_hidden = backbone.last_call["last_hidden_state"][:, 3].clone()
    model(
        input_ids=torch.tensor([[1, 2, 4]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )
    second_eye_end_hidden = backbone.last_call["last_hidden_state"][:, 3].clone()
    hook.remove()

    torch.testing.assert_close(first_eye_end_hidden, second_eye_end_hidden)
    assert not torch.equal(pooled_inputs[0], pooled_inputs[1])


def test_prefix_model_readout_depends_on_gaze_before_text():
    class MutableGazeProvider:
        value = 1.0

        def compute(self, input_ids, attention_mask):
            features = torch.full(
                (*input_ids.shape, 1),
                self.value,
                dtype=torch.float32,
            )
            mask = attention_mask.to(dtype=torch.bool)
            mask[:, 0] = False
            return features, mask

    backbone = FakeCausalBackbone()
    with torch.no_grad():
        backbone.projection.weight.copy_(torch.eye(backbone.config.hidden_size))
    gaze_provider = MutableGazeProvider()
    model = DecoderVARegressor(
        backbone,
        gaze_provider=gaze_provider,
        gaze_fusion="prefix-concat",
    )
    model.gaze_projector = nn.Linear(1, backbone.config.hidden_size, bias=False)
    with torch.no_grad():
        model.gaze_projector.weight.fill_(1.0)
    model.eval()
    pooled_inputs = []
    hook = model.regression_head.register_forward_pre_hook(
        lambda _module, arguments: pooled_inputs.append(arguments[0].detach().clone())
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    model(**batch)
    gaze_provider.value = 2.0
    model(**batch)
    hook.remove()

    assert not torch.equal(pooled_inputs[0], pooled_inputs[1])


def test_prefix_projector_accepts_multiple_selected_gaze_channels():
    backbone = FakeCausalBackbone()
    model = DecoderVARegressor(
        backbone,
        gaze_provider=FakeGazeProvider((0, 2, 3)),
        gaze_fusion="prefix-concat",
    )

    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    assert model.gaze_feature_count == 3
    assert model.gaze_projector[0].in_features == 3
    assert output.logits.shape == (1, 2)


def test_common_regression_head_is_paired_before_gaze_only_parameters():
    torch.manual_seed(123)
    baseline = DecoderVARegressor(
        FakeCausalBackbone(),
        gaze_provider=None,
        gaze_fusion="none",
    )
    torch.manual_seed(123)
    gaze = DecoderVARegressor(
        FakeCausalBackbone(),
        gaze_provider=FakeGazeProvider(),
        gaze_fusion="prefix-concat",
    )

    for baseline_parameter, gaze_parameter in zip(
        baseline.regression_head.parameters(),
        gaze.regression_head.parameters(),
    ):
        assert torch.equal(baseline_parameter, gaze_parameter)


def test_va_outputs_use_hard_sigmoid_not_logistic_sigmoid():
    model = DecoderVARegressor(
        FakeCausalBackbone(),
        gaze_provider=None,
        gaze_fusion="none",
    )
    with torch.no_grad():
        model.regression_head[-1].weight.zero_()
        model.regression_head[-1].bias.copy_(torch.tensor([-4.0, 4.0]))

    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    assert output.logits.tolist() == [[0.0, 1.0]]
    assert model.config.va_output_activation == OUTPUT_ACTIVATION


def test_regression_head_accepts_lower_precision_backbone_output():
    class LowerPrecisionOutputBackbone(FakeCausalBackbone):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            return BaseModelOutput(
                last_hidden_state=output.last_hidden_state.to(torch.bfloat16)
            )

    model = DecoderVARegressor(
        LowerPrecisionOutputBackbone(),
        gaze_provider=None,
        gaze_fusion="none",
    )
    model.eval()

    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    assert output.logits.dtype == torch.float32


def test_text_only_model_pools_last_valid_token_without_left_padding():
    backbone = FakeCausalBackbone()
    model = DecoderVARegressor(
        backbone,
        gaze_provider=None,
        gaze_fusion="none",
    )
    model.eval()
    pooled_inputs = []
    hook = model.regression_head.register_forward_pre_hook(
        lambda _module, arguments: pooled_inputs.append(arguments[0].detach().clone())
    )
    output = model(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
    )
    hook.remove()

    assert output.logits.shape == (2, 2)
    assert backbone.last_call["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert backbone.last_call["position_ids"].tolist() == [[0, 1, 2], [0, 1, 0]]
    expected_pooled = backbone.last_call["last_hidden_state"][
        torch.arange(2),
        torch.tensor([2, 1]),
    ]
    torch.testing.assert_close(pooled_inputs[0], expected_pooled)


@pytest.mark.parametrize(
    "attention_mask, error",
    (
        (torch.tensor([[0, 1, 1]]), "contiguous right padding"),
        (torch.tensor([[1, 0, 1]]), "contiguous right padding"),
        (torch.tensor([[1, 2, 0]]), "only zero and one"),
    ),
)
def test_text_only_model_rejects_invalid_attention_masks(attention_mask, error):
    model = DecoderVARegressor(
        FakeCausalBackbone(),
        gaze_provider=None,
        gaze_fusion="none",
    )

    with pytest.raises(ValueError, match=error):
        model(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=attention_mask,
        )


@pytest.mark.parametrize(
    "gaze_fusion, expected",
    (
        ("prefix-concat", "prefix-concat"),
        ("prefix", "prefix-concat"),
        ("concat", "prefix-concat"),
        ("gaze-concat", "prefix-concat"),
        ("none", "none"),
        ("off", "none"),
    ),
)
def test_gaze_fusion_aliases_are_canonical(gaze_fusion, expected):
    assert model_module._normalize_gaze_fusion(gaze_fusion) == expected


@pytest.mark.parametrize("gaze_fusion", ("postfix", "postfix-concat"))
def test_legacy_postfix_fusion_is_rejected(gaze_fusion):
    with pytest.raises(ValueError, match="architecture-incompatible"):
        model_module._normalize_gaze_fusion(gaze_fusion)


@pytest.mark.parametrize("loss_name", ("mse", "ccc", "mse+ccc"))
def test_two_output_losses_are_finite_for_singleton_batches(loss_name):
    labels = torch.tensor([[0.25, 0.75]])
    logits = torch.tensor([[0.3, 0.7]], requires_grad=True)

    ccc = concordance_correlation_coefficient(logits, labels)
    breakdown = va_regression_loss(logits, labels, loss_name)

    assert torch.isfinite(ccc).all()
    assert torch.isfinite(breakdown.total)
    breakdown.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_four_output_predictions_are_rejected():
    labels = torch.tensor([[0.25, 0.75]])
    logits = torch.tensor([[0.3, 0.7, -1.0, 1.0]])

    with pytest.raises(ValueError, match=r"\[batch, 2\]"):
        va_regression_loss(logits, labels, "mse")


def test_metrics_preserve_point_regression_names():
    labels = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    predictions = np.array(
        [
            [0.1, 0.9],
            [0.9, 0.1],
            [0.5, 0.5],
        ]
    )

    metrics = calculate_va_metrics(labels, predictions)

    assert metrics["mse_valence"] == pytest.approx(0.0066666667)
    assert metrics["mae_arousal"] == pytest.approx(0.0666666667)
    assert metrics["pearson_corr_valence"] > 0.99
    assert "ccc_arousal" in metrics
    assert not any("nll" in name or "logvar" in name for name in metrics)


def test_oof_reports_only_include_datasets_that_remain(tmp_path):
    metadata = pd.DataFrame(
        {
            "index": [0, 1],
            "text": ["good", "bad"],
            "dataset_of_origin": ["Emobank", "Emobank"],
        }
    )
    labels = np.array([[0.8, 0.6], [0.2, 0.4]])
    predictions = np.array([[0.7, 0.5], [0.3, 0.5]])
    frame = prediction_frame(metadata, labels, predictions, fold=1)

    combined, _, by_source = write_oof_reports([frame], tmp_path)

    assert len(combined) == 2
    assert by_source["dataset_of_origin"].tolist() == ["Emobank"]
    assert (tmp_path / "oof_predictions.tsv").is_file()
    assert (tmp_path / "metrics_by_dataset.tsv").is_file()


def test_qwen_loader_uses_text_only_class_and_all_linear_lora(monkeypatch):
    calls = {}
    fake_backbone = FakeCausalBackbone()

    class FakeQwenForCausalLM:
        def __init__(self):
            self.model = fake_backbone

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model_id"] = model_id
            calls["load_kwargs"] = kwargs
            return cls()

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            calls["lora_kwargs"] = kwargs

    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = FakeLoraConfig
    fake_peft.TaskType = SimpleNamespace(FEATURE_EXTRACTION="feature-extraction")

    def fake_get_peft_model(backbone, config):
        calls["backbone"] = backbone
        calls["lora_config"] = config
        return backbone

    fake_peft.get_peft_model = fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    import transformers

    monkeypatch.setattr(
        transformers,
        "Qwen3_5ForCausalLM",
        FakeQwenForCausalLM,
        raising=False,
    )

    result = load_qwen_backbone_with_lora(
        "Qwen/fake",
        revision="fixed-commit",
        dtype=torch.bfloat16,
    )

    assert result is fake_backbone
    assert calls["model_id"] == "Qwen/fake"
    assert calls["load_kwargs"]["revision"] == "fixed-commit"
    assert calls["load_kwargs"]["trust_remote_code"] is False
    assert calls["load_kwargs"]["dtype"] is torch.bfloat16
    assert calls["lora_kwargs"]["target_modules"] == "all-linear"
    assert calls["lora_kwargs"]["task_type"] == "feature-extraction"


def test_saved_prefix_model_has_strict_reload_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_module,
        "load_qwen_backbone_with_lora",
        lambda *args, **kwargs: FakeCausalBackbone(),
    )
    tokenizer = SimpleNamespace(padding_side="left")
    source = build_qwen_va_model(
        tokenizer,
        model_id="Qwen/fake",
        model_revision="fixed-decoder-commit",
        gaze_fusion="prefix-concat",
        et_revision="fixed-et-commit",
        dtype=torch.float32,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.02,
        gaze_feature_indices=(0, 3),
    )
    with torch.no_grad():
        source.regression_head[-1].bias.copy_(torch.tensor([0.1, 0.2]))
    save_file(
        source.state_dict(),
        tmp_path / SAFE_WEIGHTS_FILENAME,
        metadata={"format": "pt"},
    )
    source.save_architecture_manifest(tmp_path)

    loaded, loaded_tokenizer = load_saved_decoder_va_model(
        tmp_path,
        tokenizer=tokenizer,
        dtype=torch.float32,
        et_cache_size=17,
    )

    manifest = json.loads(
        (tmp_path / ARCHITECTURE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == ARCHITECTURE_MANIFEST_VERSION
    assert manifest["decoder_commit"] == "fixed-decoder-commit"
    assert manifest["gaze_fusion"] == "prefix-concat"
    assert manifest["gaze_features"] == ["nFix", "TRT"]
    assert manifest["gaze_feature_indices"] == [0, 3]
    assert manifest["features_used"] == [1, 0, 0, 1, 0]
    assert manifest["gaze_concat_order"] == GAZE_PREFIX_ORDER
    assert manifest["pooling_position"] == GAZE_PREFIX_POOLING
    assert manifest["output_activation"] == OUTPUT_ACTIVATION
    assert manifest["reconstruction"]["gaze_fusion"] == "prefix-concat"
    assert manifest["reconstruction"]["et_feature_names"] == ["nFix", "TRT"]
    assert manifest["reconstruction"]["et_feature_indices"] == [0, 3]
    assert manifest["reconstruction"]["features_used"] == [1, 0, 0, 1, 0]
    assert manifest["reconstruction"]["output_dim"] == 2
    assert manifest["reconstruction"]["output_activation"] == OUTPUT_ACTIVATION
    assert manifest["reconstruction"]["lora_rank"] == 8
    assert manifest["reconstruction"]["et_revision"] == "fixed-et-commit"
    assert manifest["state_dict"]["strict_loading"] is True
    assert loaded_tokenizer is tokenizer
    assert loaded_tokenizer.padding_side == "right"
    assert loaded.gaze_provider is not None
    assert loaded.gaze_provider.feature_indices == (0, 3)
    assert loaded.gaze_projector[0].in_features == 2
    assert loaded._reconstruction_config["et_cache_size"] == 17
    assert loaded.training is False
    for name, expected in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected)

    source.gaze_provider = FakeGazeProvider((0, 3))
    loaded.gaze_provider = FakeGazeProvider((0, 3))
    source.eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }
    with torch.no_grad():
        expected_logits = source(**batch).logits
        loaded_logits = loaded(**batch).logits
    torch.testing.assert_close(loaded_logits, expected_logits)

    manifest["reconstruction"]["output_dim"] = 4
    (tmp_path / ARCHITECTURE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two-output"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)

    manifest["reconstruction"]["output_dim"] = 2
    manifest["schema_version"] = 2
    (tmp_path / ARCHITECTURE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest version"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)

    manifest["schema_version"] = ARCHITECTURE_MANIFEST_VERSION
    manifest["gaze_concat_order"] = "text, eye_start, compact_trt_gaze, eye_end"
    (tmp_path / ARCHITECTURE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gaze concat order"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)

    manifest["gaze_concat_order"] = GAZE_PREFIX_ORDER
    manifest["pooling_position"] = "eye_end"
    (tmp_path / ARCHITECTURE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pooling position"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)

    manifest["pooling_position"] = GAZE_PREFIX_POOLING
    manifest["features_used"] = [0, 0, 0, 1, 0]
    (tmp_path / ARCHITECTURE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="feature mask"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)

    manifest["features_used"] = [1, 0, 0, 1, 0]
    manifest["et_model"]["feature_names"] = ["TRT"]
    (tmp_path / ARCHITECTURE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ET2 model metadata"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)

    manifest["et_model"]["feature_names"] = ["nFix", "TRT"]
    manifest["reconstruction"]["et_feature_names"] = ["TRT"]
    (tmp_path / ARCHITECTURE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Reconstruction metadata"):
        load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=torch.float32)


def test_saved_text_only_model_has_no_active_gaze_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_module,
        "load_qwen_backbone_with_lora",
        lambda *args, **kwargs: FakeCausalBackbone(),
    )
    tokenizer = SimpleNamespace(padding_side="left")
    source = build_qwen_va_model(
        tokenizer,
        model_id="Qwen/fake",
        model_revision="fixed-decoder-commit",
        gaze_fusion="none",
        dtype=torch.float32,
    )
    save_file(
        source.state_dict(),
        tmp_path / SAFE_WEIGHTS_FILENAME,
        metadata={"format": "pt"},
    )
    source.save_architecture_manifest(tmp_path)

    loaded, _ = load_saved_decoder_va_model(
        tmp_path,
        tokenizer=tokenizer,
        dtype=torch.float32,
    )
    manifest = json.loads(
        (tmp_path / ARCHITECTURE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    assert manifest["gaze_features"] == []
    assert manifest["gaze_feature_indices"] == []
    assert manifest["features_used"] == [0, 0, 0, 0, 0]
    assert manifest["et_model"] is None
    assert manifest["reconstruction"]["et_feature_names"] == []
    assert manifest["reconstruction"]["et_feature_indices"] == []
    assert manifest["reconstruction"]["features_used"] == [0, 0, 0, 0, 0]
    assert loaded.gaze_provider is None
    assert loaded.gaze_projector is None


def test_blank_text_uses_one_active_eos_token():
    class EmptyTokenizer:
        eos_token_id = 7
        pad_token_id = 7

        def __call__(self, text, max_length, truncation, padding):
            assert text == ""
            return {"input_ids": [], "attention_mask": []}

    frame = pd.DataFrame(
        {
            "index": [0],
            "text": [""],
            "dataset_of_origin": ["Emobank"],
            "valence": [0.5],
            "arousal": [0.5],
        }
    )
    dataset = TokenizedVADataset(frame, EmptyTokenizer(), max_length=8)

    item = dataset[0]

    assert item["input_ids"].tolist() == [7]
    assert item["attention_mask"].tolist() == [1]


def test_trainer_runs_one_step_and_predicts_two_outputs(tmp_path):
    class TinyDataset(Dataset):
        rows = (
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([0.2, 0.8]),
            },
            {
                "input_ids": torch.tensor([4, 5, 6]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([0.8, 0.2]),
            },
        )

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    dataset = TinyDataset()
    model = DecoderVARegressor(
        FakeCausalBackbone(),
        gaze_provider=FakeGazeProvider(),
        gaze_fusion="prefix-concat",
    )
    arguments = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        use_cpu=True,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = VARegressionTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=default_data_collator,
        loss_name="mse",
    )

    assert trainer.model_accepts_loss_kwargs is False
    result = trainer.train()
    predictions = trainer.predict(dataset)

    assert np.isfinite(result.training_loss)
    assert predictions.predictions.shape == (2, 2)
    assert predictions.label_ids.shape == (2, 2)

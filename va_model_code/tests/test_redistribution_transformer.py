"""Offline native-transformer smoke tests for trainable gaze redistribution."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaModel, TrainingArguments, default_data_collator

import va_model_code.decoder_va.model as model_module
from va_model_code.decoder_va.model import (
    SAFE_WEIGHTS_FILENAME,
    build_qwen_va_model,
    load_saved_decoder_va_model,
)
from va_model_code.decoder_va.redistribution import redistribution_contract
from va_model_code.decoder_va.trainer import VARegressionTrainer


class FixedET2:
    """Provide deterministic raw gaze without downloading or training an ET model."""

    def __init__(self, *, feature_indices, repo_id, revision, filename, **kwargs):
        self.feature_indices = tuple(feature_indices)
        self.repo_id = repo_id
        self.revision = revision
        self.filename = filename

    def compute(self, input_ids, attention_mask):
        gaze_mask = attention_mask.bool().clone()
        gaze_mask[:, 0] = False
        gaze_mask[:, 2] = False
        raw = input_ids.float().square().unsqueeze(-1) / 5
        raw = torch.where(gaze_mask.unsqueeze(-1), raw, 0.0)
        return raw, gaze_mask


def tiny_native_backbone(*args, dtype=torch.float32, **kwargs):
    """Construct a small causal Transformers decoder entirely offline."""

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return LlamaModel(config).to(dtype=dtype)


@pytest.mark.parametrize("finetuning_mode", ["full", "lora"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_decoder_checkpointed_training_and_strict_reload(
    tmp_path, monkeypatch, finetuning_mode, dtype
):
    """Exercise real causal attention, checkpoint recomputation, optimizer and reload."""

    if finetuning_mode == "lora":
        pytest.importorskip("peft")
    monkeypatch.setattr(model_module, "_load_qwen_text_backbone", tiny_native_backbone)
    monkeypatch.setattr(model_module, "ET2GazeProvider", FixedET2)
    torch.manual_seed(12)
    tokenizer = SimpleNamespace(padding_side="right")
    model = build_qwen_va_model(
        tokenizer,
        model_id="offline/tiny-causal",
        model_revision="offline",
        dtype=dtype,
        finetuning_mode=finetuning_mode,
        gaze_redistribution=redistribution_contract(
            "asym-gaussian", init_sigma_left=1.5, init_sigma_right=2.0
        ),
        classifier_dropout=0,
        gaze_projection_dropout=(0, 0),
        gaze_projection_dim=8,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0,
    )
    model.gradient_checkpointing_enable({"use_reentrant": False})
    assert model.backbone.is_gradient_checkpointing
    kernel = model.gaze_redistributor.kernel
    assert kernel.log_sigma_left.dtype == torch.float32
    assert kernel.log_sigma_right.dtype == torch.float32
    assert next(model.backbone.parameters()).dtype == dtype
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 7, 4, 0], [1, 8, 5, 3, 9, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
    }
    labels = torch.tensor([[0.1, 0.9], [0.7, 0.2]])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    before = [p.detach().clone() for p in kernel.parameters()]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            loss = (model(**inputs).logits - labels).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in kernel.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().item() > 0
        optimizer.step()
    for initial, updated in zip(before, kernel.parameters()):
        assert not torch.equal(initial, updated)
    model.eval()
    with torch.inference_mode():
        expected = model(**inputs).logits
    save_file(model.state_dict(), tmp_path / SAFE_WEIGHTS_FILENAME)
    model.save_architecture_manifest(tmp_path)
    restored, _ = load_saved_decoder_va_model(tmp_path, tokenizer=tokenizer, dtype=dtype)
    with torch.inference_mode():
        actual = restored(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for source, saved in zip(kernel.parameters(), restored.gaze_redistributor.kernel.parameters()):
        torch.testing.assert_close(saved, source, rtol=0, atol=0)


def test_real_trainer_checkpoint_resume_keeps_sigma_state(tmp_path, monkeypatch):
    """Check native Trainer safe checkpoints contain and restore both learned widths."""

    monkeypatch.setattr(model_module, "_load_qwen_text_backbone", tiny_native_backbone)
    monkeypatch.setattr(model_module, "ET2GazeProvider", FixedET2)
    tokenizer = SimpleNamespace(padding_side="right")
    config = redistribution_contract("asym-gaussian")

    def make_model():
        """Create the same architecture for training and checkpoint resumption."""

        return build_qwen_va_model(
            tokenizer,
            finetuning_mode="full",
            dtype=torch.float32,
            gaze_redistribution=config,
            classifier_dropout=0,
            gaze_projection_dropout=(0, 0),
            gaze_projection_dim=8,
        )

    arguments = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=1,
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        save_safetensors=True,
        report_to="none",
        use_cpu=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=0.01,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    rows = [
        {"input_ids": torch.tensor([1, 2, 5, 4]), "attention_mask": torch.ones(4, dtype=torch.long), "labels": torch.tensor([0.2, 0.8])},
        {"input_ids": torch.tensor([1, 8, 3, 6]), "attention_mask": torch.ones(4, dtype=torch.long), "labels": torch.tensor([0.9, 0.1])},
    ]
    trainer = VARegressionTrainer(
        model=make_model(), args=arguments, train_dataset=rows,
        data_collator=default_data_collator, loss_name="mse",
    )
    trainer.train()
    checkpoint = tmp_path / "checkpoint-1"
    assert (checkpoint / SAFE_WEIGHTS_FILENAME).is_file()
    expected = {name: p.detach().clone() for name, p in trainer.model.gaze_redistributor.named_parameters()}
    restored = VARegressionTrainer(
        model=make_model(), args=arguments, train_dataset=rows,
        data_collator=default_data_collator, loss_name="mse",
    )
    restored._load_from_checkpoint(str(checkpoint))
    for name, parameter in restored.model.gaze_redistributor.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)

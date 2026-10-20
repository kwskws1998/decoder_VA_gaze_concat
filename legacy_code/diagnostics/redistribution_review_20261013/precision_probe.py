"""Reproduce full-BF16 update loss and output rounding without trained weights."""

import copy
import json
from pathlib import Path
import sys

import torch
from transformers import Trainer, TrainingArguments


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from va_model_code.decoder_va.model import DecoderVARegressor
from va_model_code.decoder_va.redistribution import redistribution_contract
from va_model_code.tests.test_alignment_training_integration import tiny_native_qwen_backbone
from va_model_code.tests.test_model_loss_evaluation import FakeGazeProvider


def optimizer_for(model):
    """Use the same pinned Trainer default optimizer and recorded learning rate."""
    args = TrainingArguments(
        output_dir="/private/tmp/decoder-va-review-precision",
        use_cpu=True, report_to="none", learning_rate=6e-6, weight_decay=0.01,
    )
    cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return cls(model.parameters(), weight_decay=args.weight_decay, **kwargs)


def scalar_probe(dtype):
    """Track known nonzero-gradient updates at representative parameter magnitudes."""
    model = torch.nn.Linear(4, 1, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.01, 0.02, 0.1, 1.0]]))
    optimizer = optimizer_for(model)
    initial = model.weight.detach().float().clone()
    for _ in range(100):
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    state = optimizer.state[model.weight]
    return {
        "dtype": str(dtype), "steps": 100,
        "initial": initial.flatten().tolist(),
        "final": model.weight.detach().float().flatten().tolist(),
        "moment_dtype": str(state["exp_avg"].dtype),
        "fused": optimizer.defaults.get("fused"),
    }


def native_probe():
    """Compare paired tiny native Qwen paths with BF16 versus FP32 stored weights."""
    torch.manual_seed(42)
    model = DecoderVARegressor(
        tiny_native_qwen_backbone(torch.bfloat16),
        gaze_provider=FakeGazeProvider(), gaze_fusion="prefix-concat",
        gaze_redistribution=redistribution_contract(
            "fixed-gaussian", init_sigma_left=1.0, init_sigma_right=0.25,
        ),
        gaze_projection_dropout=(0, 0), classifier_dropout=0,
    )
    variants = {"bf16_stored": model, "fp32_stored_bf16_compute": copy.deepcopy(model)}
    variants["fp32_stored_bf16_compute"].backbone.float()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 7, 0], [1, 8, 4, 6, 3, 7]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
    }
    labels = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    results = {}
    for name, model in variants.items():
        model.train()
        optimizer = optimizer_for(model)
        initial = {k: v.detach().clone() for k, v in model.backbone.named_parameters()}
        first_gradient_nonzero = {}
        losses = []
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                outputs = model(**inputs).logits
                loss = (outputs.float() - labels).square().mean()
            loss.backward()
            if step == 0:
                first_gradient_nonzero = {
                    k: p.grad.detach().ne(0) if p.grad is not None else torch.zeros_like(p, dtype=torch.bool)
                    for k, p in model.backbone.named_parameters()
                }
            optimizer.step()
            losses.append(loss.item())
        total = sum(p.numel() for p in model.backbone.parameters())
        nonzero = sum(x.sum().item() for x in first_gradient_nonzero.values())
        changed = sum((p.detach() != initial[k]).sum().item() for k, p in model.backbone.named_parameters())
        changed_with_gradient = sum(
            ((p.detach() != initial[k]) & first_gradient_nonzero[k]).sum().item()
            for k, p in model.backbone.named_parameters()
        )
        results[name] = {
            "steps": 10, "total_backbone_elements": total,
            "nonzero_first_gradient_elements": nonzero,
            "changed_elements": changed,
            "changed_fraction_among_first_nonzero_gradients": changed_with_gradient / nonzero,
            "losses": losses,
        }
    return results


def main():
    """Write inspectable numerical evidence and limit it to CPU synthetic inputs."""
    import transformers
    values = torch.tensor([-0.001, 0, 0.001], dtype=torch.bfloat16)
    result = {
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "device": "cpu", "trained_checkpoint_used": False,
        "scalar_optimizer": [scalar_probe(d) for d in (torch.bfloat16, torch.float32)],
        "hardsigmoid": {
            "input": values.float().tolist(),
            "bf16": torch.nn.functional.hardsigmoid(values).float().tolist(),
            "fp32": torch.nn.functional.hardsigmoid(values.float()).tolist(),
        },
        "tiny_native_qwen": native_probe(),
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

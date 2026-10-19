"""Inspect Qwen training kernels and optionally validate CUDA BF16 forward/backward paths."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
from typing import Any

if __package__:
    from .decoder_va.training_kernels import require_fla_dispatch, training_kernel_report
else:
    from decoder_va.training_kernels import require_fla_dispatch, training_kernel_report


def comparison_stats(actual, expected, *, label: str, rtol: float = 0.08, atol: float = 0.002) -> dict[str, Any]:
    """Check finite BF16 results with an elementwise and scale-normalized error bound."""

    import torch

    actual, expected = actual.detach().float(), expected.detach().float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"{label}: non-finite output or gradient")
    difference = actual - expected
    rms_reference = expected.square().mean().sqrt().item()
    rms_error = difference.square().mean().sqrt().item()
    normalized_rmse = rms_error / max(rms_reference, 1e-12)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=label)
    if rms_reference > 1e-8 and normalized_rmse > rtol:
        raise AssertionError(f"{label}: normalized RMSE {normalized_rmse:.6g} exceeds {rtol}")
    return {
        "max_abs_error": difference.abs().max().item(),
        "reference_rms": rms_reference,
        "rmse": rms_error,
        "normalized_rmse": normalized_rmse,
        "rtol": rtol,
        "atol": atol,
    }


def check_chunk_kernel(qwen, *, length: int) -> dict[str, Any]:
    """Compare active fused delta-rule outputs and five input gradients to native Torch."""

    import torch

    active = qwen.torch_chunk_gated_delta_rule
    reference = inspect.unwrap(active)
    if reference is active or not reference.__module__.startswith("transformers.models.qwen3_5"):
        raise RuntimeError("Cannot identify the original Torch delta-rule reference through __wrapped__.")
    torch.manual_seed(410 + length)
    shape = (2, length, 2, 64)
    template = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    template.extend((
        -torch.rand(shape[:-1], device="cuda", dtype=torch.float32),
        torch.rand(shape[:-1], device="cuda", dtype=torch.bfloat16),
    ))
    cotangent = torch.randn(shape, device="cuda", dtype=torch.float32) / (2 * length * 2 * 64) ** 0.5
    executions = []
    for function in (reference, active):
        values = [value.detach().clone().requires_grad_(True) for value in template]
        output, state = function(
            *values[:3], g=values[3], beta=values[4], initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        if state is not None:
            raise AssertionError("Uncached training unexpectedly returned recurrent state")
        (output.float() * cotangent).sum().backward()
        if any(value.grad is None for value in values):
            raise AssertionError("Delta-rule kernel did not propagate every input gradient")
        executions.append((output, [value.grad for value in values]))
    expected, actual = executions
    return {
        "shape": list(shape),
        "dtypes": {"qkv": "bfloat16", "g": "float32", "beta": "bfloat16"},
        "output": comparison_stats(actual[0], expected[0], label=f"L={length}: output"),
        "gradients": {
            name: comparison_stats(found, wanted, label=f"L={length}: {name} gradient")
            for name, found, wanted in zip(("q", "k", "v", "g", "beta"), actual[1], expected[1])
        },
    }


def check_hybrid_checkpoint() -> dict[str, Any]:
    """Exercise native hybrid Qwen, production TRT rotation, and checkpoint replay on CUDA."""

    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    if __package__:
        from .decoder_va.gaze_attention import install_gaze_relative_attention
    else:
        from decoder_va.gaze_attention import install_gaze_relative_attention

    torch.manual_seed(912)
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=64, linear_value_head_dim=64,
        layer_types=["linear_attention", "full_attention"], max_position_embeddings=128,
        attention_dropout=0.0,
        rope_parameters={
            "rope_type": "default", "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5, "mrope_section": [3, 3, 2],
        },
    )
    config._attn_implementation = "eager"
    reference = Qwen3_5TextModel(config).cuda().train()
    install_gaze_relative_attention(reference)
    with torch.no_grad():
        reference.layers[1].self_attn.gaze_relative_rotation.alpha.fill_(0.2)
    replay = copy.deepcopy(reference)
    replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    length = 65
    mask = torch.ones(2, length, device="cuda", dtype=torch.long)
    mask[1, -3:] = 0
    anchors = torch.tensor([[3, 7, 9], [4, 8, 0]], device="cuda")
    anchor_mask = torch.tensor([[True, True, True], [True, True, False]], device="cuda")
    gaze_mask = torch.zeros(2, length, device="cuda", dtype=torch.bool)
    gaze_mask[0, [3, 7, 9]] = True
    gaze_mask[1, [4, 8]] = True
    values = torch.zeros(2, length, device="cuda")
    values[0, [3, 7, 9]] = torch.tensor([1.0, 4.0, 2.0], device="cuda")
    values[1, [4, 8]] = torch.tensor([2.0, 5.0], device="cuda")
    inputs = {
        "input_ids": torch.randint(1, 64, (2, length), device="cuda"),
        "attention_mask": mask, "use_cache": False,
        "gaze_relation_values": values, "gaze_relation_mask": gaze_mask,
        "gaze_relation_indices": anchors, "gaze_relation_anchor_mask": anchor_mask,
    }
    target = torch.randn(2, length, 128, device="cuda") / (2 * length * 128) ** 0.5
    results = []
    for model in (reference, replay):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs).last_hidden_state
            loss = (output.float() * target).sum()
        loss.backward()
        results.append(output)
    output_stats = comparison_stats(results[1], results[0], label="hybrid checkpoint output", rtol=0.02, atol=0.001)
    expected_parameters = dict(reference.named_parameters())
    largest_rmse = 0.0
    checked = 0
    alpha_norm = None
    for name, parameter in replay.named_parameters():
        expected = expected_parameters[name].grad
        if expected is None:
            if parameter.grad is not None:
                raise AssertionError(f"Unexpected checkpoint gradient: {name}")
            continue
        if parameter.grad is None:
            raise AssertionError(f"Missing checkpoint gradient: {name}")
        stats = comparison_stats(parameter.grad, expected, label=name, rtol=0.03, atol=0.002)
        largest_rmse = max(largest_rmse, stats["normalized_rmse"])
        checked += 1
        if name.endswith("gaze_relative_rotation.alpha"):
            alpha_norm = parameter.grad.float().norm().item()
    if alpha_norm is None or alpha_norm <= 0:
        raise AssertionError("TRT rotation alpha did not receive a finite nonzero gradient")
    return {
        "output": output_stats, "checked_parameter_gradients": checked,
        "largest_gradient_normalized_rmse": largest_rmse, "alpha_gradient_norm": alpha_norm,
        "shape": [2, length, 128], "gradient_checkpointing": "nonreentrant",
        "precision": "FP32 parameters with CUDA BF16 autocast",
    }


def run_smoke_test(report: dict[str, Any]) -> dict[str, Any]:
    """Require active FLA and validate CUDA execution without downloads or backend replacement."""

    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    require_fla_dispatch(report)
    if not torch.cuda.is_available():
        raise RuntimeError("--smoke-test requires CUDA; no GPU test has been run")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("--smoke-test requires BF16 support on the selected CUDA device")
    device = torch.cuda.current_device()
    metadata = {
        "name": torch.cuda.get_device_name(device),
        "device": device, "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_cuda_runtime": torch.version.cuda,
    }
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        chunk_results = []
        for length in (8, 65):
            print(
                f"Checking CUDA BF16 delta-rule forward/backward at L={length}; "
                "first use may compile Triton kernels.",
                file=sys.stderr, flush=True,
            )
            chunk_results.append(check_chunk_kernel(qwen, length=length))
        print("Checking hybrid Qwen, TRT rotation, and gradient checkpoint replay.", file=sys.stderr, flush=True)
        hybrid = check_hybrid_checkpoint()
        torch.cuda.synchronize()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_matmul, old_cudnn
    return {
        "status": "passed", "gpu": metadata, "delta_rule": chunk_results,
        "hybrid_checkpoint": hybrid,
        "scope": "Synthetic correctness smoke test, not full-model throughput or training-efficacy evidence.",
    }


def main(argv: list[str] | None = None) -> int:
    """Print machine-readable diagnostics and fail closed when a requested check fails."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-fla", action="store_true", help="Fail unless Qwen selected the external FLA chunk kernel")
    parser.add_argument("--smoke-test", action="store_true", help="Run CUDA BF16 forward/backward and checkpoint checks; requires FLA")
    parser.add_argument("--output", type=Path, help="Also save the JSON report at this path")
    args = parser.parse_args(argv)
    report = {}
    status = 0
    try:
        report = training_kernel_report()
        if args.require_fla:
            require_fla_dispatch(report)
        if args.smoke_test:
            report["cuda_smoke_test"] = run_smoke_test(report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if args.smoke_test:
            report["cuda_smoke_test"] = {"status": "failed", "error": report["error"]}
        status = 1
    result = json.dumps(report, indent=2, sort_keys=True)
    print(result, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

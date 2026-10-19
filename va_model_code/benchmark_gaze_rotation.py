"""Benchmark one synthetic Qwen full-attention layer without downloading models.

This measures already-projected, already-position-rotated Q/K/V attention with
forward and backward, optionally replayed by non-reentrant checkpointing. It
does not measure ET2, packing, Q/K/V projections, native RoPE, linear-attention
layers, the MLP, optimizer updates, evaluation, or full-training ETA. Repeating
the same synthetic microbatch models gradient accumulation timing only.

The legacy correction is frozen from commit 1f95381 so its original synchronous
validation remains in the baseline even when the production code improves.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

try:
    from .decoder_va.gaze_attention import gaze_relative_attention_forward
    from .decoder_va.gaze_relation import TRTRelativeRotation, _positive_float
except ImportError:
    from decoder_va.gaze_attention import gaze_relative_attention_forward
    from decoder_va.gaze_relation import TRTRelativeRotation, _positive_float


LEGACY_REFERENCE_COMMIT = "1f9538168167110ebf2be0eaf5884d757c1e0bd9"
ARMS = ("raw_eager", "legacy_dense", "compact")


class LegacyDenseRotation(TRTRelativeRotation):
    """Preserve the pre-optimization correction, including GPU-to-host checks."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        trt_values: torch.Tensor,
        text_gaze_mask: torch.Tensor,
        *,
        attention_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Evaluate the dense implementation frozen from the recorded source commit."""

        if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
            raise ValueError("query and key must be floating-point tensors.")
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must have shape [batch, heads, length, head_dim].")
        if not query.is_floating_point() or not key.is_floating_point():
            raise ValueError("query and key must be floating-point tensors.")
        if key.device != query.device or key.dtype != query.dtype:
            raise ValueError("query and key must have the same device and dtype.")
        batch_size, heads, length, head_dim = query.shape
        if heads != self.num_heads:
            raise ValueError("query head count must match num_heads.")
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer.")
        if key.shape[0] != batch_size or key.shape[2:] != (length, head_dim):
            raise ValueError("query and key must have matching batch, length, and head_dim.")
        key_heads = key.shape[1]
        if key_heads <= 0 or heads % key_heads:
            raise ValueError("query heads must be divisible by the positive key head count.")
        if self.alpha.device != query.device:
            raise ValueError("alpha and query must be on the same device.")
        if not isinstance(trt_values, torch.Tensor) or not trt_values.is_floating_point():
            raise ValueError("trt_values must be a floating-point tensor.")
        if trt_values.shape != (batch_size, length) or trt_values.device != query.device:
            raise ValueError("trt_values must match query batch, length, and device.")
        if not isinstance(text_gaze_mask, torch.Tensor):
            raise ValueError("text_gaze_mask must be an explicit binary tensor.")
        if text_gaze_mask.shape != (batch_size, length) or text_gaze_mask.device != query.device:
            raise ValueError("text_gaze_mask must match query batch, length, and device.")
        if text_gaze_mask.is_complex() or not bool(
            ((text_gaze_mask == 0) | (text_gaze_mask == 1)).all()
        ):
            raise ValueError("text_gaze_mask must contain only binary zero or one values.")
        score_scale = (
            1.0 / math.sqrt(head_dim)
            if attention_scale is None
            else _positive_float(attention_scale, "attention_scale")
        )
        valid = text_gaze_mask.bool()
        token_mask = valid[:, None, :, None]
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = torch.where(token_mask, query.float(), 0.0)
            k = torch.where(token_mask, key.float(), 0.0)
            gaze = torch.where(valid, trt_values.float(), 0.0)
            alpha = self.alpha.float()
            if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(k).all()):
                raise ValueError("query and key must be finite FP32 values at eligible text anchors.")
            if not bool(torch.isfinite(gaze).all()) or not bool(torch.isfinite(alpha).all()):
                raise ValueError("eligible TRT values and alpha must be finite FP32 values.")
            k = k.repeat_interleave(heads // key_heads, dim=1)
            pair_mask = valid[:, None, :, None] & valid[:, None, None, :]
            pair_mask = pair_mask & ~torch.eye(length, dtype=torch.bool, device=query.device)[None, None]
            relative_gaze = torch.where(
                pair_mask, gaze[:, None, None, :] - gaze[:, None, :, None], 0.0,
            ) / self.scale
            phase = alpha[None, :, None, None] * relative_gaze
            if not bool(torch.isfinite(phase).all()):
                raise ValueError("TRT rotation phase must remain finite in FP32.")
            k_half = torch.cat((-k[..., head_dim // 2 :], k[..., : head_dim // 2]), dim=-1)
            original_scores = torch.matmul(q, k.transpose(-2, -1))
            cross_scores = torch.matmul(q, k_half.transpose(-2, -1))
            cosine_minus_one = -2.0 * (phase * 0.5).sin().square()
            delta = torch.where(
                pair_mask,
                (cosine_minus_one * original_scores + phase.sin() * cross_scores) * score_scale,
                0.0,
            )
            if not bool(torch.isfinite(delta).all()):
                raise ValueError("TRT attention correction must remain finite in FP32.")
            return delta.to(query.dtype)


def parse_args(argv=None):
    """Read explicit, offline shape and timing controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--config", type=Path, help="Local Qwen config.json; overrides head dimensions without downloading.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 64, 202, 402])
    parser.add_argument("--anchor-counts", nargs="+", default=["0", "1", "half"], help="Nonnegative counts or half; clipped to available text positions.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1, help="CPU thread count; does not alter CUDA thread settings.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="Write an additional Chrome profiler trace per case and arm.")
    parser.add_argument("--output", type=Path, default=Path("../results/gaze_rotation_benchmark.json"))
    args = parser.parse_args(argv)
    args.shape_source = "explicit_cli_defaults_not_loaded_model"
    if args.config is not None:
        document = json.loads(args.config.read_text())
        config = document.get("text_config", document)
        args.heads = int(config["num_attention_heads"])
        args.kv_heads = int(config["num_key_value_heads"])
        args.head_dim = int(config.get("head_dim") or config["hidden_size"] // args.heads)
        args.shape_source = str(args.config.resolve())
    if min(args.batch_size, args.heads, args.kv_heads, args.head_dim, args.iterations, args.threads, args.gradient_accumulation_steps) <= 0:
        parser.error("Batch, head, iteration, thread, and accumulation counts must be positive.")
    if args.heads % args.kv_heads or args.head_dim % 2:
        parser.error("Query heads must be divisible by KV heads; head_dim must be even.")
    if args.warmup < 0 or min(args.lengths) < 4:
        parser.error("Warmup must be nonnegative and packed lengths must be at least four.")
    if any(not math.isfinite(value) for value in args.alphas):
        parser.error("Alpha values must be finite.")
    if any(value != "half" and (not value.isdecimal()) for value in args.anchor_counts):
        parser.error("Anchor counts must be nonnegative integers or half.")
    return args


def make_case(args, length, anchor_count, *, device, dtype):
    """Create shared post-RoPE tensors, causal/padding masks, and compact metadata."""

    generator = torch.Generator(device="cpu").manual_seed(args.seed + length)
    q = torch.randn(args.batch_size, args.heads, length, args.head_dim, generator=generator).to(device=device, dtype=dtype).requires_grad_()
    k = torch.randn(args.batch_size, args.kv_heads, length, args.head_dim, generator=generator).to(device=device, dtype=dtype).requires_grad_()
    v = torch.randn(args.batch_size, args.kv_heads, length, args.head_dim, generator=generator).to(device=device, dtype=dtype).requires_grad_()
    target = torch.randn(args.batch_size, length, args.heads, args.head_dim, generator=generator).to(device=device)
    start = length // 2
    requested = length // 2 if anchor_count == "half" else int(anchor_count)
    counts = [min(requested, max(0, length - row % 3 - start)) for row in range(args.batch_size)]
    indices = torch.zeros(args.batch_size, max(counts), dtype=torch.long)
    anchor_mask = torch.zeros_like(indices, dtype=torch.bool)
    text_mask = torch.zeros(args.batch_size, length, dtype=torch.bool)
    active = torch.arange(length)[None, :] < (length - torch.arange(args.batch_size) % 3)[:, None]
    values = torch.zeros(args.batch_size, length)
    for row, count in enumerate(counts):
        if count:
            row_indices = torch.arange(start, start + count)
            indices[row, :count] = row_indices
            anchor_mask[row, :count] = True
            text_mask[row, row_indices] = True
            values[row, row_indices] = torch.linspace(0.0, 3.0, count)
    allowed = torch.ones(length, length, dtype=torch.bool).tril()[None, None] & active[:, None, None, :]
    mask = torch.zeros(args.batch_size, 1, length, length, dtype=dtype).masked_fill_(~allowed, torch.finfo(dtype).min)
    return {
        "q": q, "k": k, "v": v, "target": target,
        "mask": mask.to(device), "values": values.to(device),
        "text_mask": text_mask.to(device), "indices": indices.to(device),
        "anchor_mask": anchor_mask.to(device), "counts": counts,
    }


def make_module(args, arm, device, alpha):
    """Construct attention metadata and an identical alpha vector for each arm."""

    module = nn.Module()
    module.num_key_value_groups = args.heads // args.kv_heads
    if arm != "raw_eager":
        rotation_type = LegacyDenseRotation if arm == "legacy_dense" else TRTRelativeRotation
        module.add_module("gaze_relative_rotation", rotation_type(args.heads).to(device=device))
        with torch.no_grad():
            module.gaze_relative_rotation.alpha.copy_(torch.linspace(0.5 * alpha, alpha, args.heads, device=device))
    return module.train()


def legacy_attention_forward(module, query, key, value, attention_mask, scaling, *, gaze_relation_values, gaze_relation_mask):
    """Preserve the old wrapper, including expanding grouped keys before correction."""

    if gaze_relation_values is None or gaze_relation_mask is None:
        raise ValueError("TRT-relative attention requires explicit gaze_relation_values and gaze_relation_mask.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("TRT-relative attention expects rank-4 query, key, and value tensors.")
    if query.shape[-2] != key.shape[-2] or key.shape[-2] != value.shape[-2]:
        raise ValueError("TRT-relative attention requires equal complete query/key/value sequence lengths.")
    if attention_mask is None or attention_mask.ndim != 4 or not attention_mask.is_floating_point():
        raise ValueError("TRT-relative eager attention requires an explicit 4D additive causal/padding mask.")
    if attention_mask.shape[-2:] != (query.shape[-2], key.shape[-2]):
        raise ValueError("The attention mask must match the complete query/key sequence lengths.")
    rotation = getattr(module, "gaze_relative_rotation", None)
    if not isinstance(rotation, TRTRelativeRotation):
        raise TypeError("The Qwen attention module has no installed TRT-relative rotation.")

    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    if key_states.shape[1] != query.shape[1] or value_states.shape[1] != query.shape[1]:
        raise ValueError("Grouped-query attention head counts are inconsistent.")
    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    logits = logits + rotation(query, key_states, gaze_relation_values, gaze_relation_mask, attention_scale=scaling)
    logits = logits + attention_mask
    weights = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = torch.nn.functional.dropout(weights, p=0.0, training=module.training)
    return torch.matmul(weights, value_states).transpose(1, 2).contiguous()


def forward_attention(module, arm, case, q, k, v):
    """Use native eager or the production eager wrapper with the selected correction."""

    scaling = q.shape[-1] ** -0.5
    if arm == "raw_eager":
        from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward

        return eager_attention_forward(module, q, k, v, case["mask"], scaling, dropout=0.0)[0]
    kwargs = {"gaze_relation_values": case["values"], "gaze_relation_mask": case["text_mask"]}
    if arm == "legacy_dense":
        return legacy_attention_forward(module, q, k, v, case["mask"], scaling, **kwargs)
    if arm == "compact":
        kwargs.update(gaze_relation_indices=case["indices"], gaze_relation_anchor_mask=case["anchor_mask"])
    return gaze_relative_attention_forward(module, q, k, v, case["mask"], scaling, dropout=0.0, **kwargs)[0]


def backward_step(module, arm, case, args):
    """Accumulate two microbatch gradients by default; do not perform an optimizer update."""

    module.zero_grad(set_to_none=True)
    for name in ("q", "k", "v"):
        case[name].grad = None
    last_output = None
    for _ in range(args.gradient_accumulation_steps):
        def attention(q, k, v):
            """Recompute the same attention graph during optional checkpoint backward."""

            return forward_attention(module, arm, case, q, k, v)

        inputs = (case["q"], case["k"], case["v"])
        output = checkpoint(attention, *inputs, use_reentrant=False) if args.gradient_checkpointing else attention(*inputs)
        loss = (output.float() * case["target"]).sum() / (output.numel() ** 0.5 * args.gradient_accumulation_steps)
        loss.backward()
        last_output = output.detach()
    return last_output


def verify_case(modules, case, args, alpha):
    """Fail before timing if compact outputs or gradients disagree with the legacy path."""

    results = {}
    for arm, module in modules.items():
        output = backward_step(module, arm, case, args)
        results[arm] = {"output": output.float().cpu()}
        for name in ("q", "k", "v"):
            results[arm][name] = case[name].grad.float().cpu().clone()
        if arm != "raw_eager":
            gradient = module.gaze_relative_rotation.alpha.grad
            if gradient is None:
                raise AssertionError(f"{arm} lost the alpha gradient graph.")
            results[arm]["alpha"] = gradient.float().cpu().clone()
    tolerances = (2e-4, 2e-5) if args.dtype == "float32" else (3e-2, 2e-3)
    max_errors = {}
    for name in results["legacy_dense"]:
        actual, expected = results["compact"][name], results["legacy_dense"][name]
        torch.testing.assert_close(actual, expected, rtol=tolerances[0], atol=tolerances[1], msg=f"Compact/legacy mismatch: {name}")
        max_errors[name] = float((actual - expected).abs().max())
    if alpha == 0.0 or max(case["counts"]) <= 1:
        for arm in ("legacy_dense", "compact"):
            for name in ("output", "q", "k", "v"):
                torch.testing.assert_close(results[arm][name], results["raw_eager"][name], rtol=tolerances[0], atol=tolerances[1], msg=f"Native identity mismatch: {arm}/{name}")
    if alpha == 0.0 and max(case["counts"]) > 1 and not torch.any(results["compact"]["alpha"] != 0):
        raise AssertionError("Zero initialization unexpectedly eliminated all alpha gradients.")
    return {"passed": True, "rtol": tolerances[0], "atol": tolerances[1], "max_abs_errors": max_errors}


def synchronize(device):
    """Synchronize only at benchmark boundaries, never inside the optimized operator."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_arm(module, arm, case, args, device, trace_path=None):
    """Report synchronized wall time, CUDA event time, and allocated memory peak."""

    for _ in range(args.warmup):
        backward_step(module, arm, case, args)
    module.zero_grad(set_to_none=True)
    for name in ("q", "k", "v"):
        case[name].grad = None
    synchronize(device)
    baseline = None
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall_times, event_times = [], []
    for _ in range(args.iterations):
        synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        started = time.perf_counter()
        if start_event is not None:
            start_event.record()
        backward_step(module, arm, case, args)
        if end_event is not None:
            end_event.record()
        synchronize(device)
        wall_times.append((time.perf_counter() - started) * 1000)
        if start_event is not None:
            event_times.append(start_event.elapsed_time(end_event))
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    if trace_path is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            backward_step(module, arm, case, args)
            synchronize(device)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
    return {
        "wall_ms_median": statistics.median(wall_times), "wall_ms_samples": wall_times,
        "cuda_event_ms_median": statistics.median(event_times) if event_times else None,
        "cuda_event_ms_samples": event_times, "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_increment_allocated_bytes": None if peak is None else peak - baseline,
        "trace": None if trace_path is None else str(trace_path),
    }


def main(argv=None):
    """Run all arms on identical synthetic cases and persist a standalone JSON report."""

    args = parse_args(argv)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu --dtype float32 for a correctness smoke test.")
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support BF16.")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    from transformers import __version__ as transformers_version

    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "single_full_attention_forward_backward_microbenchmark_not_full_training_eta",
        "excludes": ["ET2", "packing", "projections", "native_RoPE", "linear_attention", "MLP", "optimizer_update", "evaluation"],
        "legacy_reference_commit": LEGACY_REFERENCE_COMMIT,
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("benchmark_gaze_rotation.py", "decoder_va/gaze_relation.py", "decoder_va/gaze_attention.py")
        },
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {
            "torch": torch.__version__, "transformers": transformers_version, "cuda_runtime": torch.version.cuda,
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "num_cpu_threads": torch.get_num_threads(),
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for length in args.lengths:
        for anchor_count in args.anchor_counts:
            case = make_case(args, length, anchor_count, device=device, dtype=dtype)
            for alpha in args.alphas:
                modules = {arm: make_module(args, arm, device, alpha) for arm in ARMS}
                checked = verify_case(modules, case, args, alpha)
                entry = {"length": length, "anchor_request": anchor_count, "anchors_per_row": case["counts"], "alpha_range": [0.5 * alpha, alpha], "verification": checked, "arms": {}}
                for arm, module in modules.items():
                    trace_path = args.output.parent / (args.output.stem + "_traces") / f"length{length}_anchors{anchor_count}_alpha{alpha}_{arm}.json" if args.profile else None
                    entry["arms"][arm] = measure_arm(module, arm, case, args, device, trace_path)
                old_ms = entry["arms"]["legacy_dense"]["wall_ms_median"]
                new_ms = entry["arms"]["compact"]["wall_ms_median"]
                entry["legacy_over_compact_wall_ratio"] = old_ms / new_ms
                report["results"].append(entry)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                timings = " ".join(f"{arm}={entry['arms'][arm]['wall_ms_median']:.2f}ms" for arm in ARMS)
                print(f"L={length} anchors={min(case['counts'])}..{max(case['counts'])} alpha={alpha:g} {timings}", flush=True)
                del modules
            del case
    print(f"Saved: {args.output.resolve()}")
    print("These are attention-only timings per accumulation group, not a full-training ETA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

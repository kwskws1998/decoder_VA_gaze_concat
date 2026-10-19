"""Offline benchmark contracts and equality gates for TRT attention timing."""

import json

import pytest
import torch

from va_model_code.benchmark_gaze_rotation import (
    ARMS,
    LEGACY_REFERENCE_COMMIT,
    LegacyDenseRotation,
    main,
    make_case,
    make_module,
    parse_args,
    verify_case,
)
from va_model_code.decoder_va.gaze_relation import TRTRelativeRotation


def tiny_args(*extra):
    """Use a tiny CPU shape while retaining the production accumulation contract."""

    return parse_args([
        "--device", "cpu", "--dtype", "float32", "--batch-size", "2",
        "--heads", "2", "--kv-heads", "1", "--head-dim", "8",
        "--lengths", "8", "--iterations", "1", "--warmup", "0", *extra,
    ])


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("anchors", ["0", "1", "half"])
@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_benchmark_equality_gate_covers_identity_and_learned_rotation(checkpointing, anchors, alpha):
    """Timing is permitted only after matching outputs and every input/alpha gradient."""

    args = tiny_args()
    args.gradient_checkpointing = checkpointing
    case = make_case(args, 8, anchors, device=torch.device("cpu"), dtype=torch.float32)
    modules = {arm: make_module(args, arm, torch.device("cpu"), alpha) for arm in ARMS}
    result = verify_case(modules, case, args, alpha)
    assert result["passed"]
    assert set(result["max_abs_errors"]) == {"output", "q", "k", "v", "alpha"}


def test_legacy_reference_matches_retained_dense_operator():
    """The frozen implementation remains mathematically equal after production changes."""

    args = tiny_args()
    case = make_case(args, 8, "half", device=torch.device("cpu"), dtype=torch.float32)
    old = LegacyDenseRotation(args.heads)
    new = TRTRelativeRotation(args.heads)
    with torch.no_grad():
        old.alpha.copy_(torch.tensor([0.1, -0.3]))
        new.alpha.copy_(old.alpha)
    expected = old(case["q"], case["k"], case["values"], case["text_mask"])
    actual = new(case["q"], case["k"], case["values"], case["text_mask"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_benchmark_local_config_overrides_shape_without_model_loading(tmp_path):
    """Read nested model configuration locally without invoking Hub APIs."""

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"text_config": {"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16}}))
    args = tiny_args("--config", str(config_path))
    assert (args.heads, args.kv_heads, args.head_dim) == (4, 2, 16)
    assert args.shape_source == str(config_path.resolve())


@pytest.mark.parametrize("extra", [
    ["--anchor-counts", "-1"], ["--heads", "3"], ["--head-dim", "7"],
    ["--iterations", "0"], ["--lengths", "1"], ["--alphas", "nan"],
])
def test_benchmark_rejects_invalid_shapes_and_timing_parameters(extra):
    """Avoid misleading timings from invalid or empty measurement workloads."""

    if extra == ["--heads", "3"]:
        extra = [*extra, "--kv-heads", "2"]
    with pytest.raises(SystemExit):
        tiny_args(*extra)


def test_benchmark_smoke_report_labels_limits_and_measurements(tmp_path):
    """The CPU smoke report must not masquerade as a measured CUDA training ETA."""

    output = tmp_path / "benchmark.json"
    result = main([
        "--device", "cpu", "--dtype", "float32", "--batch-size", "2",
        "--heads", "2", "--kv-heads", "1", "--head-dim", "8", "--lengths", "8",
        "--anchor-counts", "half", "--alphas", "0", "0.1", "--iterations", "1",
        "--warmup", "0", "--output", str(output),
    ])
    report = json.loads(output.read_text())
    assert result == 0
    assert report["scope"] == "single_full_attention_forward_backward_microbenchmark_not_full_training_eta"
    assert report["legacy_reference_commit"] == LEGACY_REFERENCE_COMMIT
    assert report["environment"]["gpu_name"] is None
    assert len(report["results"]) == 2
    for entry in report["results"]:
        assert entry["verification"]["passed"]
        assert set(entry["arms"]) == set(ARMS)
        for arm in entry["arms"].values():
            assert arm["wall_ms_median"] > 0
            assert arm["cuda_event_ms_median"] is None
            assert arm["peak_allocated_bytes"] is None

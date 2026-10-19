"""Offline dispatch reporting, explicit failure gates, and numerical-check contracts."""

import functools
import json
from types import SimpleNamespace

import pytest
import torch

from va_model_code import check_training_kernels as probe
from va_model_code.decoder_va import training_kernels as kernels


def wrapped_dispatch(*, external, module):
    """Reproduce Transformers' frozen fallback closure without importing optional packages."""

    def reference(*args, **kwargs):
        return args, kwargs

    def implementation(*args, **kwargs):
        return args, kwargs

    implementation.__module__ = module
    is_new_implementation = external

    @functools.wraps(reference)
    def dispatch(*args, **kwargs):
        if is_new_implementation:
            return implementation(*args, **kwargs)
        return implementation(*args, **kwargs)

    @functools.wraps(dispatch)
    def outer(*args, **kwargs):
        return dispatch(*args, **kwargs)

    return outer


@pytest.mark.parametrize("external,module,is_fla", [
    (True, "fla.ops.gated_delta_rule.chunk", True),
    (True, "causal_conv1d.causal_conv1d_interface", False),
    (False, "transformers.models.qwen3_5.modeling_qwen3_5", False),
])
def test_dispatch_reads_selected_closure_through_outer_wrappers(external, module, is_fla):
    """A decorated function's public name is insufficient to detect its selected implementation."""

    result = kernels.inspect_kernel_dispatch(wrapped_dispatch(external=external, module=module))
    assert result["resolved"]
    assert result["selected"].startswith(module + ".")
    assert result["external_package"] is external
    assert result["is_fla"] is is_fla
    assert len(result["wrapper_chain"]) == 2


def test_dispatch_cycle_or_unknown_function_is_unresolved():
    """Unexpected decorators never imply a fast kernel and cannot cause an infinite loop."""

    def function():
        return None

    function.__wrapped__ = function
    result = kernels.inspect_kernel_dispatch(function)
    assert not result["resolved"] and not result["is_fla"]
    assert result["selected"] is None and result["external_package"] is None
    assert len(result["wrapper_chain"]) == 1
    assert not kernels.inspect_kernel_dispatch(None)["resolved"]


def test_report_imports_qwen_before_any_fla_probe_and_is_json_plain(monkeypatch):
    """Read-only inspection does not preload FLA, initialize CUDA, or run a model."""

    imported = []
    model = SimpleNamespace(**{
        name: wrapped_dispatch(external=False, module="transformers.models.qwen3_5.modeling_qwen3_5")
        for name in kernels.KERNEL_FUNCTIONS
    })

    def fake_import(name):
        imported.append(name)
        assert name == "transformers.models.qwen3_5.modeling_qwen3_5"
        return model

    def fake_version(name):
        if name == "fla-core":
            raise kernels.PackageNotFoundError(name)
        return "test-version"

    monkeypatch.setattr(kernels.importlib, "import_module", fake_import)
    monkeypatch.setattr(kernels, "version", fake_version)
    report = kernels.training_kernel_report()
    assert imported == ["transformers.models.qwen3_5.modeling_qwen3_5"]
    assert report["packages"]["fla-core"] is None
    assert report["cuda_smoke_test"] == "not_run"
    assert json.loads(json.dumps(report)) == report


def test_require_fla_surfaces_import_exception(monkeypatch):
    """Broken installations report their original import error instead of silently falling back."""

    def broken_import(name):
        raise OSError("undefined symbol: example_cuda_symbol")

    monkeypatch.setattr(kernels.importlib, "import_module", broken_import)
    with pytest.raises(RuntimeError, match="OSError: undefined symbol") as found:
        kernels.require_fla_dispatch({})
    assert isinstance(found.value.__cause__, OSError)


@pytest.mark.parametrize("dispatch", [{}, {"resolved": True, "is_fla": False, "selected": "torch.fallback"}])
def test_require_fla_rejects_cached_fallback_even_after_successful_import(monkeypatch, dispatch):
    """Import success alone cannot prove that the already-imported Qwen uses FLA."""

    monkeypatch.setattr(kernels.importlib, "import_module", lambda name: SimpleNamespace(chunk_gated_delta_rule=lambda: None))
    with pytest.raises(RuntimeError, match="has not selected"):
        kernels.require_fla_dispatch({"dispatch": {"torch_chunk_gated_delta_rule": dispatch}})


def test_require_fla_accepts_verified_selected_kernel(monkeypatch):
    """A real selected FLA closure is required in addition to a successful direct import."""

    monkeypatch.setattr(kernels.importlib, "import_module", lambda name: SimpleNamespace(chunk_gated_delta_rule=lambda: None))
    kernels.require_fla_dispatch({"dispatch": {"torch_chunk_gated_delta_rule": {
        "resolved": True, "is_fla": True, "selected": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
    }}})


def test_default_cli_reports_without_running_cuda(monkeypatch, capsys, tmp_path):
    """The default inspection mode remains usable while another GPU training job is running."""

    report = {"dispatch": {}, "packages": {}, "cuda_smoke_test": "not_run"}
    monkeypatch.setattr(probe, "training_kernel_report", lambda: report.copy())

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA smoke must be explicit")

    monkeypatch.setattr(probe, "run_smoke_test", forbidden)
    path = tmp_path / "nested" / "kernels.json"
    assert probe.main(["--output", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert json.loads(path.read_text()) == report


def test_cli_explicit_required_failure_retains_report(monkeypatch, capsys):
    """Requested fast-path checks stop with a nonzero exit status and useful diagnostic output."""

    monkeypatch.setattr(probe, "training_kernel_report", lambda: {"packages": {"torch": "test"}})

    def reject(report):
        raise RuntimeError("missing FLA")

    monkeypatch.setattr(probe, "require_fla_dispatch", reject)
    assert probe.main(["--require-fla"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["packages"]["torch"] == "test"
    assert "missing FLA" in result["error"]


def test_cli_failed_smoke_cannot_report_pass(monkeypatch, capsys):
    """Unavailable CUDA or failed gradients cannot accidentally produce a successful GPU claim."""

    monkeypatch.setattr(probe, "training_kernel_report", lambda: {"cuda_smoke_test": "not_run"})

    def fail(report):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(probe, "run_smoke_test", fail)
    assert probe.main(["--smoke-test"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["cuda_smoke_test"]["status"] == "failed"


def test_comparison_rejects_nonfinite_and_large_relative_error():
    """Both absolute tolerance and a normalized bound are enforced for small gradients."""

    expected = torch.tensor([0.001, -0.002])
    assert probe.comparison_stats(expected, expected, label="same")["normalized_rmse"] == 0
    with pytest.raises(RuntimeError, match="non-finite"):
        probe.comparison_stats(torch.tensor([float("nan"), 0.0]), expected, label="nan")
    with pytest.raises(AssertionError, match="normalized RMSE"):
        probe.comparison_stats(expected * 1.5, expected, label="small gradients")


def test_smoke_rejects_no_cuda_before_constructing_model(monkeypatch):
    """CPU-only environments must not be treated as validation of a fused GPU training path."""

    monkeypatch.setattr(probe, "require_fla_dispatch", lambda report: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        probe.run_smoke_test({})

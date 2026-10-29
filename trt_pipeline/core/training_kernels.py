"""Inspect Qwen's import-time optional-kernel dispatch without downloading or running models."""

from __future__ import annotations

import importlib
import inspect
from importlib.metadata import PackageNotFoundError, version
from typing import Any


KERNEL_PACKAGES = (
    "torch", "transformers", "triton", "kernels", "flash-linear-attention", "fla-core", "causal-conv1d",
)
KERNEL_FUNCTIONS = ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")


def qualified_name(value: Any) -> str:
    """Return an implementation identifier without invoking the implementation."""

    return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"


def inspect_kernel_dispatch(function: Any) -> dict[str, Any]:
    """Find Transformers' frozen package fallback through transparent decorator wrappers."""

    chain = []
    seen = set()
    while callable(function) and id(function) not in seen:
        seen.add(id(function))
        chain.append(qualified_name(function))
        try:
            captured = inspect.getclosurevars(function).nonlocals
        except (TypeError, ValueError):
            captured = {}
        implementation = captured.get("implementation")
        external = captured.get("is_new_implementation")
        if callable(implementation) and isinstance(external, bool):
            selected = qualified_name(implementation)
            return {
                "selected": selected,
                "external_package": external,
                "is_fla": external and selected.startswith("fla."),
                "resolved": True,
                "wrapper_chain": chain,
            }
        function = getattr(function, "__wrapped__", None)
    return {
        "selected": None,
        "external_package": None,
        "is_fla": False,
        "resolved": False,
        "wrapper_chain": chain,
    }


def training_kernel_report() -> dict[str, Any]:
    """Report versions and package dispatch without explicit CUDA queries or model execution."""

    packages = {}
    for name in KERNEL_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    qwen = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    dispatch = {name: inspect_kernel_dispatch(getattr(qwen, name, None)) for name in KERNEL_FUNCTIONS}
    if not dispatch["torch_chunk_gated_delta_rule"]["resolved"]:
        active = getattr(qwen, "chunk_gated_delta_rule", None)
        fallback = getattr(qwen, "torch_chunk_gated_delta_rule", None)
        if callable(active) or callable(fallback):
            selected = qualified_name(active if callable(active) else fallback)
            dispatch["torch_chunk_gated_delta_rule"].update({
                "selected": selected, "external_package": callable(active),
                "is_fla": selected.startswith("fla."), "resolved": True,
            })
    return {
        "packages": packages,
        "dispatch": dispatch,
        "dispatch_scope": "Import-time package dispatch; explicit use_kernels=True hub overrides are not inspected.",
        "cuda_smoke_test": "not_run",
    }


def require_fla_dispatch(report: dict[str, Any]) -> None:
    """Surface the original FLA import error and reject silently selected Torch fallback."""

    try:
        module = importlib.import_module("fla.ops.gated_delta_rule")
        if not callable(getattr(module, "chunk_gated_delta_rule", None)):
            raise ImportError("fla.ops.gated_delta_rule.chunk_gated_delta_rule is missing or not callable")
    except Exception as error:
        raise RuntimeError(
            f"FLA could not be imported: {type(error).__name__}: {error}"
        ) from error
    dispatch = report.get("dispatch", {}).get("torch_chunk_gated_delta_rule", {})
    if not dispatch.get("resolved") or not dispatch.get("is_fla"):
        raise RuntimeError(
            "Qwen has not selected the FLA chunk kernel; selected="
            f"{dispatch.get('selected')!r}. Restart Python after installing FLA and check the import above."
        )

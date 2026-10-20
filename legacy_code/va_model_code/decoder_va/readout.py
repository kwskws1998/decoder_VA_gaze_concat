"""Versioned pooling contracts for legacy text pooling and a learned causal readout."""

from __future__ import annotations

from collections.abc import Mapping


READOUT_MODES = ("last-token", "learned-token")


def readout_contract(mode: str = "last-token") -> dict:
    """Declare the pooling token and reproducible learned-token initialization."""

    if mode not in READOUT_MODES:
        raise ValueError(f"Unknown readout mode: {mode!r}.")
    result = {"version": 1, "mode": mode}
    if mode == "learned-token":
        result.update(position="after_text", initialization="normal_0.02")
    return result


def validate_readout_contract(value: Mapping) -> dict:
    """Reject incomplete or ambiguous pooling semantics."""

    if not isinstance(value, Mapping):
        raise ValueError("readout must be a mapping.")
    expected = readout_contract(value.get("mode"))
    if type(value.get("version")) is not int or dict(value) != expected:
        raise ValueError("Missing or incompatible readout contract.")
    return expected


def saved_readout_contract(metadata: Mapping, *, schema_version: int, context: str) -> dict:
    """Preserve legacy pooling for schema 6–9 and require explicit schema-10 readout."""

    if type(schema_version) is not int or schema_version not in (6, 7, 8, 9, 10, 11):
        raise ValueError(f"{context}: unsupported readout schema {schema_version!r}.")
    if "readout" not in metadata:
        if schema_version < 10:
            return readout_contract()
        raise ValueError(f"{context}: missing readout contract.")
    contract = validate_readout_contract(metadata["readout"])
    if schema_version < 10 and contract != readout_contract():
        raise ValueError(f"{context}: schema {schema_version} cannot enable learned-token readout.")
    return contract


def readout_pooling_position(gaze_fusion: str, readout: Mapping) -> str:
    """Return the metadata description of the exact hidden state used for regression."""

    contract = validate_readout_contract(readout)
    if gaze_fusion not in ("none", "prefix-concat"):
        raise ValueError(f"Unsupported gaze fusion: {gaze_fusion!r}.")
    if contract["mode"] == "learned-token":
        return "learned_readout_token_after_text"
    return "last_valid_text_token_after_gaze_prefix" if gaze_fusion == "prefix-concat" else "last_valid_text_token"


def readout_concat_order(gaze_fusion: str, readout: Mapping) -> str | None:
    """Return the gaze-prefix layout including an optional final learned readout."""

    contract = validate_readout_contract(readout)
    if gaze_fusion not in ("none", "prefix-concat"):
        raise ValueError(f"Unsupported gaze fusion: {gaze_fusion!r}.")
    if gaze_fusion == "none":
        return None
    order = "eye_start, compact_selected_gaze, eye_end, text"
    return order + ", readout" if contract["mode"] == "learned-token" else order

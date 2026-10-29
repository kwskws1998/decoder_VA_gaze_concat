"""Baseline metadata compatibility and rejection of retired experiment profiles.

Keep the disabled schema-8–11 identities so existing raw TRT checkpoints remain
loadable. Enabled experimental metadata must fail before loading a backbone.
"""

from collections.abc import Mapping


LEGACY_HINT = 'Retired experiment source is preserved in artifacts/cleanup_20261029/original_workspace.zip.'
UNCERTAINTY_TABLE_COLUMNS = (
    "log_variance_valence", "log_variance_arousal",
    "pred_variance_valence", "pred_variance_arousal",
    "pred_std_valence", "pred_std_arousal",
)


def _require(value, expected, field):
    """Accept only the exact baseline identity, including integer schema types."""

    if (
        not isinstance(value, Mapping)
        or dict(value) != expected
        or ("version" in expected and type(value.get("version")) is not int)
    ):
        raise ValueError(f"Incompatible {field}: only the baseline contract is active. {LEGACY_HINT}")
    return dict(expected)


def validate_redistribution_contract(value, *, gaze_fusion, feature_indices):
    """Reject redistribution while preserving raw checkpoint metadata."""

    return _require(value, {"method": "none"}, "gaze_redistribution")


def gaze_relation_contract():
    """Describe ordinary backbone attention without any TRT rotation."""

    return {"version": 1, "method": "none"}


def validate_gaze_relation_contract(value):
    """Reject all added attention mechanisms."""

    return _require(value, gaze_relation_contract(), "gaze_relation")


def readout_contract(mode="last-token"):
    """Describe the original last-valid-text-token prediction."""

    if mode != "last-token":
        raise ValueError(f"Only last-token readout is active. {LEGACY_HINT}")
    return {"version": 1, "mode": "last-token"}


def validate_readout_contract(value):
    """Reject appended learned readout tokens."""

    return _require(value, readout_contract(), "readout")


def saved_readout_contract(metadata, *, schema_version, context):
    """Read baseline pooling from supported old and current manifests."""

    if type(schema_version) is not int or schema_version not in (6, 7, 8, 9, 10, 11):
        raise ValueError(f"{context}: unsupported readout schema {schema_version!r}.")
    if "readout" not in metadata:
        if schema_version < 10:
            return readout_contract()
        raise ValueError(f"{context}: missing readout contract.")
    return validate_readout_contract(metadata["readout"])


def readout_pooling_position(gaze_fusion, readout):
    """Return the original pooling location for baseline or gaze concat."""

    validate_readout_contract(readout)
    if gaze_fusion not in ("none", "prefix-concat"):
        raise ValueError(f"Unsupported gaze fusion: {gaze_fusion!r}.")
    return "last_valid_text_token_after_gaze_prefix" if gaze_fusion == "prefix-concat" else "last_valid_text_token"


def readout_concat_order(gaze_fusion, readout):
    """Return the original gaze-prefix sequence layout."""

    readout_pooling_position(gaze_fusion, readout)
    return "eye_start, compact_selected_gaze, eye_end, text" if gaze_fusion == "prefix-concat" else None


def prediction_contract(loss=None):
    """Describe two bounded VA means and the original regression objectives."""

    if loss not in (None, "mse", "ccc", "mse+ccc"):
        raise ValueError(f"Unsupported baseline loss {loss!r}. {LEGACY_HINT}")
    return {"version": 1, "family": "point", "output_columns": ["pred_valence", "pred_arousal"]}


def validate_prediction_contract(value):
    """Reject uncertainty heads rather than silently dropping their outputs."""

    return _require(value, prediction_contract(), "prediction_distribution")


def saved_prediction_contract(metadata, *, schema_version, context):
    """Read point-output semantics without changing historical manifest schemas."""

    if type(schema_version) is not int or schema_version not in (6, 7, 8, 9, 10, 11):
        raise ValueError(f"{context}: unsupported prediction schema {schema_version!r}.")
    value = metadata.get("prediction_distribution")
    if value is None:
        if schema_version >= 11:
            raise ValueError(f"{context}: missing prediction_distribution.")
        value = prediction_contract()
    contract = validate_prediction_contract(value)
    if "loss" in metadata and prediction_contract(metadata["loss"]) != contract:
        raise ValueError(f"{context}: loss disagrees with prediction_distribution.")
    return contract


def reject_experimental_weights(weights_path):
    """Fail closed when weights contain experimental modules despite raw metadata."""

    from safetensors import safe_open

    with safe_open(weights_path, framework="pt", device="cpu") as saved:
        keys = [
            key for key in saved.keys()
            if key.startswith(("gaze_redistributor.", "variance_head."))
            or "gaze_relative_rotation." in key
            or key == "readout_token" or key.endswith(".readout_token")
        ]
    if keys:
        raise ValueError(f"Experimental weights in baseline checkpoint: {keys}. {LEGACY_HINT}")

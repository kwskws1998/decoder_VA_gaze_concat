"""Canonical repository paths and experiment run naming."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Sequence


ACTIVE_CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ACTIVE_CODE_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _slug_component(value: object, *, field_name: str) -> str:
    """Normalize one recorded condition value into a portable filename component."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    if not normalized:
        raise ValueError(f"{field_name} cannot produce an empty filename component.")
    return normalized


def validate_run_name(run_name: str) -> str:
    """Require one portable directory name with no path components."""

    normalized = run_name.strip()
    if not normalized or not _RUN_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "--run-name must be one directory name containing only letters, "
            "numbers, dot, underscore, or hyphen."
        )
    if normalized in {".", ".."} or Path(normalized).name != normalized:
        raise ValueError("--run-name cannot contain a path or traversal component.")
    return normalized


def condition_slug(
    *,
    model: str,
    finetuning_mode: str,
    gaze_fusion: str,
    gaze_features: Sequence[str],
    seed: int,
    no_iemocap: bool = False,
) -> str:
    """Build a condition-identifying slug from the actual training contract."""

    if finetuning_mode not in {"lora", "full"}:
        raise ValueError(f"Unsupported finetuning_mode: {finetuning_mode!r}")
    if gaze_fusion not in {"none", "prefix-concat"}:
        raise ValueError(f"Unsupported gaze_fusion: {gaze_fusion!r}")
    model_slug = _slug_component(model, field_name="model")
    mode_slug = _slug_component(finetuning_mode, field_name="finetuning_mode")
    if gaze_fusion == "none":
        condition = "baseline"
    else:
        if not gaze_features:
            raise ValueError("A gaze condition must record at least one gaze feature.")
        feature_slug = "-".join(
            _slug_component(value, field_name="gaze_feature")
            for value in gaze_features
        )
        condition = f"gaze_{feature_slug}"
    exclusion = "_no_iemocap" if no_iemocap else ""
    return f"{model_slug}_{mode_slug}_{condition}{exclusion}_seed{int(seed)}"


def default_run_name(
    *,
    model: str,
    finetuning_mode: str,
    gaze_fusion: str,
    gaze_features: Sequence[str],
    seed: int,
    no_iemocap: bool = False,
    timestamp: datetime | None = None,
) -> str:
    """Create a sortable, collision-resistant, condition-aware run name."""

    run_time = timestamp or datetime.now()
    prefix = run_time.strftime("%Y%m%d_%H%M%S_%f")
    condition = condition_slug(
        model=model,
        finetuning_mode=finetuning_mode,
        gaze_fusion=gaze_fusion,
        gaze_features=gaze_features,
        seed=seed,
        no_iemocap=no_iemocap,
    )
    return f"{prefix}_{condition}"


def resolve_run_directory(
    run_name: str,
    *,
    results_root: str | Path = RESULTS_ROOT,
) -> Path:
    """Resolve one run strictly below the single repository results root."""

    safe_name = validate_run_name(run_name)
    requested_root = Path(results_root).expanduser()
    if requested_root.is_symlink():
        raise ValueError(f"Results root cannot be a symbolic link: {requested_root}")
    root = requested_root.resolve()
    requested_candidate = root / safe_name
    if requested_candidate.is_symlink():
        raise ValueError(
            f"Run directory cannot be a symbolic link: {requested_candidate}"
        )
    candidate = requested_candidate.resolve()
    if candidate.parent != root:
        raise ValueError(f"Run directory must remain directly below {root}.")
    return candidate

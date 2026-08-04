"""Reproducible preprocessing for the seven-source English VA bundle."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable
import uuid

import numpy as np
import pandas as pd

from .downloads import extract_tsv_zip, file_hash_records, sha256_file, validate_tsv_zip


OUTPUT_COLUMNS = ("index", "text", "dataset_of_origin", "valence", "arousal")
FOLD_FILENAMES = ("full_dataset_fold1.csv", "full_dataset_fold2.csv")
MERGED_FILENAME = "full_dataset_english_all.csv"
MANIFEST_FILENAME = "english_dataset_manifest.json"
MANIFEST_VERSION = 2
NORMALIZATION_CHOICES = ("observed", "source-scale")
LEGACY_PROTOCOL = "legacy-global-shuffle"
PAPER_PROTOCOL = "paper-source-wise-two-fold"
LEGACY_TEXT_POLICY = "collapse-whitespace-strip-fill-missing-empty"
PAPER_TEXT_POLICY = "preserve-source-whitespace-fill-missing-empty"
LEGACY_DEDUP_POLICY = "within-source-text-first-then-text-and-dataset-first"
PAPER_DEDUP_POLICY = "none"
LEGACY_SPLIT_STRATEGY = "seeded-global-shuffle-then-contiguous-row-halves"
PAPER_SPLIT_STRATEGY = (
    "independently-seeded-per-source-shuffle-half-split-then-concatenate"
)

SOURCE_NAME_MAP = {
    "emobank": "Emobank",
    "emotales": "EmoTales sentences",
    "facebook_va": "fb",
    "fb": "fb",
    "iemocap": "IEMOCAP sentences",
    "nrc_vad": "nrc-vad",
    "scott_et_al": "GlasgowNorms",
    "warriner_et_al": "word ratings ENG",
}
EXPECTED_SOURCE_STEMS = frozenset(
    {
        "emobank",
        "emotales",
        "facebook_va",
        "iemocap",
        "nrc_vad",
        "scott_et_al",
        "warriner_et_al",
    }
)
SOURCE_SCALE_BOUNDS = {
    "fb": {"valence": (1.0, 9.0), "arousal": (1.0, 9.0)},
}


@dataclass(frozen=True)
class BuildResult:
    fold1_path: Path
    fold2_path: Path
    merged_path: Path
    manifest_path: Path
    total_rows: int
    dataset_counts: dict[str, int]


def clean_text(series: pd.Series) -> pd.Series:
    """Collapse whitespace while retaining missing/blank text as an empty string."""

    cleaned = series.fillna("").astype(str)
    cleaned = cleaned.str.replace(r"[\r\n\t]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    return cleaned.str.strip()


def preserve_text(series: pd.Series) -> pd.Series:
    """Preserve source whitespace while representing missing text as empty."""

    return series.fillna("").astype(str)


def _normalize_observed(series: pd.Series) -> tuple[pd.Series, dict[str, object]]:
    minimum = float(series.min())
    maximum = float(series.max())
    already_unit = bool(series.between(0.0, 1.0, inclusive="both").all())
    if already_unit:
        normalized = series.astype(float)
        method = "unchanged-unit-interval"
    elif maximum == minimum:
        normalized = pd.Series(0.0, index=series.index, dtype=float)
        method = "observed-minmax-constant-to-zero"
    else:
        normalized = (series - minimum) / (maximum - minimum)
        normalized = normalized.clip(0.0, 1.0).astype(float)
        method = "observed-minmax"
    return normalized, {
        "method": method,
        "observed_min": minimum,
        "observed_max": maximum,
    }


def _normalize_dimension(
    series: pd.Series,
    dataset_name: str,
    dimension: str,
    normalization: str,
    *,
    strict_source_scale: bool = False,
) -> tuple[pd.Series, dict[str, object]]:
    if normalization == "source-scale":
        bounds = SOURCE_SCALE_BOUNDS.get(dataset_name, {}).get(dimension)
        if bounds is not None:
            lower, upper = bounds
            normalized = ((series - lower) / (upper - lower)).clip(0.0, 1.0)
            return normalized.astype(float), {
                "method": "source-scale",
                "source_min": lower,
                "source_max": upper,
                "observed_min": float(series.min()),
                "observed_max": float(series.max()),
            }
        if bool(series.between(0.0, 1.0, inclusive="both").all()):
            return series.astype(float), {
                "method": "unchanged-unit-interval",
                "source_min": 0.0,
                "source_max": 1.0,
                "observed_min": float(series.min()),
                "observed_max": float(series.max()),
            }
        if strict_source_scale:
            raise ValueError(
                "Paper protocol requires declared source-scale bounds for "
                f"{dataset_name!r} {dimension}, whose values are outside [0, 1]."
            )
    return _normalize_observed(series)


def canonical_dataset_name(path: str | os.PathLike[str]) -> str:
    stem = Path(path).stem.casefold().replace("-", "_")
    try:
        return SOURCE_NAME_MAP[stem]
    except KeyError as exc:
        available = ", ".join(sorted(EXPECTED_SOURCE_STEMS))
        raise ValueError(
            f"Unknown English VA source file {Path(path).name!r}. "
            f"Expected source stems: {available}"
        ) from exc


def discover_source_files(
    source_dir: str | os.PathLike[str],
    *,
    require_all_sources: bool = True,
) -> tuple[Path, ...]:
    directory = Path(source_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Source TSV directory not found: {directory}")
    paths = tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.casefold() == ".tsv"
                and not path.name.startswith("._")
            ),
            key=lambda path: path.name.casefold(),
        )
    )
    if not paths:
        raise FileNotFoundError(f"No TSV source files found in: {directory}")

    stems: list[str] = []
    for path in paths:
        canonical_dataset_name(path)
        stems.append(path.stem.casefold().replace("-", "_"))
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate source TSV stems are not allowed.")
    if require_all_sources:
        missing = sorted(EXPECTED_SOURCE_STEMS.difference(stems))
        extra = sorted(set(stems).difference(EXPECTED_SOURCE_STEMS))
        if missing or extra:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unexpected: " + ", ".join(extra))
            raise ValueError("The seven-source VA bundle is incomplete (" + "; ".join(details) + ").")
    return paths


def _read_source(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            sep="\t",
            keep_default_na=False,
            na_filter=False,
            dtype=str,
        )
    except Exception as exc:
        raise ValueError(f"Could not read TSV source {path}: {exc}") from exc
    required = {"text", "valence", "arousal"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    return frame


def process_source(
    path: str | os.PathLike[str],
    *,
    normalization: str = "observed",
    paper_protocol: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Prepare one source according to the selected preprocessing protocol."""

    if normalization not in NORMALIZATION_CHOICES:
        raise ValueError(
            f"normalization must be one of {NORMALIZATION_CHOICES}; got {normalization!r}."
        )
    source_path = Path(path)
    dataset_name = canonical_dataset_name(source_path)
    raw = _read_source(source_path)
    selected = pd.DataFrame(
        {
            "text": (
                preserve_text(raw["text"])
                if paper_protocol
                else clean_text(raw["text"])
            ),
            "valence": pd.to_numeric(raw["valence"], errors="coerce"),
            "arousal": pd.to_numeric(raw["arousal"], errors="coerce"),
        }
    )
    finite_mask = np.isfinite(selected["valence"].to_numpy(dtype=float)) & np.isfinite(
        selected["arousal"].to_numpy(dtype=float)
    )
    invalid_va_rows = int((~finite_mask).sum())
    selected = selected.loc[finite_mask].copy()
    if selected.empty:
        raise ValueError(f"{source_path} has no rows with finite valence and arousal.")

    selected["dataset_of_origin"] = dataset_name
    selected["valence"], valence_normalization = _normalize_dimension(
        selected["valence"],
        dataset_name,
        "valence",
        normalization,
        strict_source_scale=paper_protocol,
    )
    selected["arousal"], arousal_normalization = _normalize_dimension(
        selected["arousal"],
        dataset_name,
        "arousal",
        normalization,
        strict_source_scale=paper_protocol,
    )

    rows_before_dedup = len(selected)
    if not paper_protocol:
        selected = selected.drop_duplicates(subset=["text"], keep="first").copy()
    selected = selected[["text", "dataset_of_origin", "valence", "arousal"]]
    selected.reset_index(drop=True, inplace=True)

    summary: dict[str, object] = {
        "filename": source_path.name,
        "dataset_of_origin": dataset_name,
        "sha256": sha256_file(source_path),
        "size_bytes": source_path.stat().st_size,
        "input_rows": int(len(raw)),
        "invalid_va_rows_dropped": invalid_va_rows,
        "duplicate_text_rows_dropped": int(rows_before_dedup - len(selected)),
        "blank_text_rows_retained": int((selected["text"] == "").sum()),
        "output_rows": int(len(selected)),
        "text_policy": (
            PAPER_TEXT_POLICY if paper_protocol else LEGACY_TEXT_POLICY
        ),
        "dedup_policy": (
            PAPER_DEDUP_POLICY if paper_protocol else LEGACY_DEDUP_POLICY
        ),
        "normalization": {
            "valence": valence_normalization,
            "arousal": arousal_normalization,
        },
    }
    return selected, summary


def _prepare_frames(
    source_paths: Iterable[Path],
    *,
    seed: int,
    normalization: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for path in sorted(source_paths, key=lambda item: item.name.casefold()):
        frame, summary = process_source(path, normalization=normalization)
        frames.append(frame)
        summaries.append(summary)

    merged = pd.concat(frames, ignore_index=True)
    before_cross_file_dedup = len(merged)
    merged = merged.drop_duplicates(
        subset=["text", "dataset_of_origin"], keep="first"
    ).reset_index(drop=True)
    cross_file_duplicates = before_cross_file_dedup - len(merged)
    if cross_file_duplicates:
        counts = merged.groupby("dataset_of_origin").size()
        for summary in summaries:
            name = str(summary["dataset_of_origin"])
            summary["canonical_source_output_rows"] = int(counts.get(name, 0))

    shuffled = merged.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    shuffled.insert(0, "index", shuffled.index.astype(np.int64))
    shuffled = shuffled[list(OUTPUT_COLUMNS)]
    midpoint = len(shuffled) // 2
    fold1 = shuffled.iloc[:midpoint].copy()
    fold2 = shuffled.iloc[midpoint:].copy()
    return fold1, fold2, shuffled, summaries


def _independent_source_seed(seed: int, source_path: Path) -> int:
    """Derive a stable independent RNG seed for one source file."""

    identity = f"{int(seed)}\0{source_path.name.casefold()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def _prepare_paper_frames(
    source_paths: Iterable[Path],
    *,
    seed: int,
    normalization: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Split each source independently in half before combining the folds."""

    merged_frames: list[pd.DataFrame] = []
    fold1_frames: list[pd.DataFrame] = []
    fold2_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    next_index = 0

    for path in sorted(source_paths, key=lambda item: item.name.casefold()):
        frame, summary = process_source(
            path,
            normalization=normalization,
            paper_protocol=True,
        )
        frame = frame.copy()
        frame.insert(
            0,
            "index",
            np.arange(next_index, next_index + len(frame), dtype=np.int64),
        )
        next_index += len(frame)
        frame = frame[list(OUTPUT_COLUMNS)]
        merged_frames.append(frame)

        source_seed = _independent_source_seed(seed, path)
        permutation = np.random.default_rng(source_seed).permutation(len(frame))
        midpoint = len(frame) // 2
        source_fold1 = frame.iloc[permutation[:midpoint]].copy()
        source_fold2 = frame.iloc[permutation[midpoint:]].copy()
        fold1_frames.append(source_fold1)
        fold2_frames.append(source_fold2)
        summary["split"] = {
            "seed": source_seed,
            "fold1_rows": int(len(source_fold1)),
            "fold2_rows": int(len(source_fold2)),
            "total_rows": int(len(frame)),
        }
        summaries.append(summary)

    merged = pd.concat(merged_frames, ignore_index=True)
    fold1 = pd.concat(fold1_frames, ignore_index=True)
    fold2 = pd.concat(fold2_frames, ignore_index=True)
    return fold1, fold2, merged, summaries


def _per_source_fold_counts(
    fold1: pd.DataFrame,
    fold2: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Record the two fold sizes for every retained source."""

    fold1_counts = fold1.groupby("dataset_of_origin", sort=True).size()
    fold2_counts = fold2.groupby("dataset_of_origin", sort=True).size()
    names = sorted(
        set(fold1_counts.index).union(fold2_counts.index),
        key=lambda value: str(value).casefold(),
    )
    return {
        str(name): {
            "fold1_rows": int(fold1_counts.get(name, 0)),
            "fold2_rows": int(fold2_counts.get(name, 0)),
            "total_rows": int(fold1_counts.get(name, 0) + fold2_counts.get(name, 0)),
        }
        for name in names
    }


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
        lineterminator="\n",
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _outputs_match_manifest(output_dir: Path, manifest: dict[str, object]) -> bool:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        return False
    for filename in (*FOLD_FILENAMES, MERGED_FILENAME):
        metadata = outputs.get(filename)
        path = output_dir / filename
        if not isinstance(metadata, dict) or not path.is_file():
            return False
        if metadata.get("size_bytes") != path.stat().st_size:
            return False
        if metadata.get("sha256") != sha256_file(path):
            return False
    return True


def _cached_result(
    output_dir: Path,
    build_identity: dict[str, object],
) -> BuildResult | None:
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as input_file:
            manifest = json.load(input_file)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if manifest.get("schema_version") != MANIFEST_VERSION:
        return None
    if manifest.get("build") != build_identity:
        return None
    if not _outputs_match_manifest(output_dir, manifest):
        return None
    counts = {
        str(name): int(count)
        for name, count in dict(manifest.get("dataset_counts", {})).items()
    }
    return BuildResult(
        fold1_path=output_dir / FOLD_FILENAMES[0],
        fold2_path=output_dir / FOLD_FILENAMES[1],
        merged_path=output_dir / MERGED_FILENAME,
        manifest_path=manifest_path,
        total_rows=int(manifest.get("total_rows", sum(counts.values()))),
        dataset_counts=counts,
    )


def _build_from_paths(
    source_paths: tuple[Path, ...],
    output_dir: Path,
    *,
    seed: int,
    normalization: str,
    paper_protocol: bool,
    force: bool,
    archive_record: dict[str, object] | None,
) -> BuildResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_records = file_hash_records(source_paths)
    protocol = PAPER_PROTOCOL if paper_protocol else LEGACY_PROTOCOL
    text_policy = PAPER_TEXT_POLICY if paper_protocol else LEGACY_TEXT_POLICY
    dedup_policy = PAPER_DEDUP_POLICY if paper_protocol else LEGACY_DEDUP_POLICY
    build_identity: dict[str, object] = {
        "protocol": protocol,
        "paper_protocol": bool(paper_protocol),
        "seed": int(seed),
        "normalization": normalization,
        "text_policy": text_policy,
        "dedup_policy": dedup_policy,
        "source_files": source_records,
        "archive": archive_record,
    }
    if not force:
        cached = _cached_result(output_dir, build_identity)
        if cached is not None:
            return cached

    if paper_protocol:
        fold1, fold2, merged, source_summaries = _prepare_paper_frames(
            source_paths,
            seed=seed,
            normalization=normalization,
        )
        split_strategy = PAPER_SPLIT_STRATEGY
    else:
        fold1, fold2, merged, source_summaries = _prepare_frames(
            source_paths,
            seed=seed,
            normalization=normalization,
        )
        split_strategy = LEGACY_SPLIT_STRATEGY
    per_source_counts = _per_source_fold_counts(fold1, fold2)
    for summary in source_summaries:
        name = str(summary["dataset_of_origin"])
        summary.setdefault("split", dict(per_source_counts[name]))

    outputs = {
        FOLD_FILENAMES[0]: fold1,
        FOLD_FILENAMES[1]: fold2,
        MERGED_FILENAME: merged,
    }
    temporary_paths: dict[str, Path] = {}
    output_records: dict[str, dict[str, object]] = {}
    try:
        for filename, frame in outputs.items():
            temporary = output_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
            _write_tsv(frame, temporary)
            temporary_paths[filename] = temporary
            output_records[filename] = {
                "rows": int(len(frame)),
                "sha256": sha256_file(temporary),
                "size_bytes": temporary.stat().st_size,
            }
        for filename, temporary in temporary_paths.items():
            os.replace(temporary, output_dir / filename)
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)

    counts_series = merged.groupby("dataset_of_origin", sort=True).size()
    dataset_counts = {str(name): int(value) for name, value in counts_series.items()}
    split_manifest: dict[str, object] = {
        "strategy": split_strategy,
        "seed": int(seed),
        "fold1_rows": int(len(fold1)),
        "fold2_rows": int(len(fold2)),
        "per_source_fold_counts": per_source_counts,
    }
    if not paper_protocol:
        split_manifest["midpoint"] = int(len(merged) // 2)
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_VERSION,
        "build": build_identity,
        "protocol": protocol,
        "paper_protocol": bool(paper_protocol),
        "text_policy": text_policy,
        "dedup_policy": dedup_policy,
        "normalization": normalization,
        "columns": list(OUTPUT_COLUMNS),
        "split": split_manifest,
        "total_rows": int(len(merged)),
        "blank_text_rows_retained": int((merged["text"] == "").sum()),
        "dataset_counts": dataset_counts,
        "sources": source_summaries,
        "outputs": output_records,
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    _atomic_json(manifest_path, manifest)
    return BuildResult(
        fold1_path=output_dir / FOLD_FILENAMES[0],
        fold2_path=output_dir / FOLD_FILENAMES[1],
        merged_path=output_dir / MERGED_FILENAME,
        manifest_path=manifest_path,
        total_rows=len(merged),
        dataset_counts=dataset_counts,
    )


def build_english_dataset(
    output_dir: str | os.PathLike[str],
    *,
    source_dir: str | os.PathLike[str] | None = None,
    archive_path: str | os.PathLike[str] | None = None,
    expected_sha256: str | None = None,
    seed: int = 42,
    normalization: str | None = None,
    paper_protocol: bool = False,
    force: bool = False,
    require_all_sources: bool = True,
) -> BuildResult:
    """Build deterministic folds from either source TSVs or one validated ZIP."""

    if (source_dir is None) == (archive_path is None):
        raise ValueError("Provide exactly one of source_dir or archive_path.")
    if normalization is None:
        normalization = "source-scale" if paper_protocol else "observed"
    if normalization not in NORMALIZATION_CHOICES:
        raise ValueError(
            f"normalization must be one of {NORMALIZATION_CHOICES}; got {normalization!r}."
        )
    if paper_protocol and normalization != "source-scale":
        raise ValueError(
            "Paper protocol requires normalization='source-scale'; "
            f"got {normalization!r}."
        )
    output_path = Path(output_dir)
    if archive_path is None:
        paths = discover_source_files(
            source_dir,
            require_all_sources=require_all_sources,
        )
        return _build_from_paths(
            paths,
            output_path,
            seed=seed,
            normalization=normalization,
            paper_protocol=paper_protocol,
            force=force,
            archive_record=None,
        )

    archive = Path(archive_path)
    validate_tsv_zip(archive, expected_sha256)
    archive_record = {
        "filename": archive.name,
        "sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
    }
    with tempfile.TemporaryDirectory(prefix="decoder-va-sources-") as temporary_root:
        extraction_dir = Path(temporary_root) / "source_tsv"
        extract_tsv_zip(archive, extraction_dir, expected_sha256)
        paths = discover_source_files(
            extraction_dir,
            require_all_sources=require_all_sources,
        )
        return _build_from_paths(
            paths,
            output_path,
            seed=seed,
            normalization=normalization,
            paper_protocol=paper_protocol,
            force=force,
            archive_record=archive_record,
        )

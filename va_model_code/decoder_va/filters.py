"""In-memory dataset-of-origin filtering shared by training and evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .preprocessing import FOLD_FILENAMES


IEMOCAP_PATTERN = "IEMOCAP"
IEMOCAP_TYPOS = frozenset({"ieomcap"})


@dataclass(frozen=True)
class FilteredFolds:
    folds: dict[str, pd.DataFrame]
    requested_patterns: tuple[str, ...]
    excluded_names: tuple[str, ...]

    @property
    def fold1(self) -> pd.DataFrame:
        return self.folds[FOLD_FILENAMES[0]]

    @property
    def fold2(self) -> pd.DataFrame:
        return self.folds[FOLD_FILENAMES[1]]


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def split_exclusion_patterns(
    raw_patterns: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Expand repeated and comma-separated exclusion values in stable order."""

    if raw_patterns is None:
        values: Iterable[str] = ()
    elif isinstance(raw_patterns, str):
        values = (raw_patterns,)
    else:
        values = raw_patterns
    patterns: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for part in str(raw).split(","):
            pattern = part.strip()
            if not pattern:
                continue
            if _slug(pattern) in IEMOCAP_TYPOS:
                pattern = IEMOCAP_PATTERN
            key = pattern.casefold()
            if key not in seen:
                patterns.append(pattern)
                seen.add(key)
    return tuple(patterns)


def collect_exclude_patterns(
    exclude_dataset: str | Iterable[str] | None = None,
    *,
    no_iemocap: bool = False,
    no_ieomcap: bool = False,
) -> tuple[str, ...]:
    """Combine --exclude-dataset values with both IEMOCAP flag spellings."""

    patterns = list(split_exclusion_patterns(exclude_dataset))
    if no_iemocap or no_ieomcap:
        patterns.append(IEMOCAP_PATTERN)
    return split_exclusion_patterns(patterns)


def resolve_excluded_datasets(
    available_names: Iterable[object],
    requested_patterns: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Resolve exact/slug/substring patterns or fail with all available names."""

    available = tuple(sorted({str(name) for name in available_names}, key=str.casefold))
    patterns = split_exclusion_patterns(requested_patterns)
    matched: set[str] = set()
    unresolved: list[str] = []
    for pattern in patterns:
        lowered = pattern.casefold()
        pattern_slug = _slug(pattern)
        exact = tuple(
            name
            for name in available
            if name.casefold() == lowered or (pattern_slug and _slug(name) == pattern_slug)
        )
        if exact:
            matched.update(exact)
            continue
        loose = tuple(
            name for name in available if pattern_slug and pattern_slug in _slug(name)
        )
        if loose:
            matched.update(loose)
        else:
            unresolved.append(pattern)

    if unresolved:
        choices = "\n".join(f"  - {name}" for name in available) or "  - <none>"
        raise ValueError(
            "Could not match excluded dataset pattern(s): "
            + ", ".join(unresolved)
            + "\nAvailable dataset_of_origin values:\n"
            + choices
        )
    return tuple(sorted(matched, key=str.casefold))


def read_fold(path: str | os.PathLike[str]) -> pd.DataFrame:
    fold_path = Path(path)
    if not fold_path.is_file():
        raise FileNotFoundError(
            f"Fold TSV not found: {fold_path}. On a fresh machine, enter "
            "va_model_code and run `python prepare_english_data.py "
            "--download-default --output-dir <dir>` before training. For "
            "paper-protocol folds, run `python prepare_english_data.py "
            "--download-default --paper-protocol --output-dir <dir>`, then pass "
            "the same directory to training with `--data-dir <dir>`."
        )
    frame = pd.read_csv(
        fold_path,
        sep="\t",
        quotechar='"',
        engine="python",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
        keep_default_na=False,
        dtype={"text": str, "dataset_of_origin": str},
    )
    required = {"index", "text", "dataset_of_origin", "valence", "arousal"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{fold_path} is missing required column(s): {', '.join(missing)}")
    return frame


def load_folds(data_dir: str | os.PathLike[str]) -> dict[str, pd.DataFrame]:
    directory = Path(data_dir)
    return {filename: read_fold(directory / filename) for filename in FOLD_FILENAMES}


def dataset_counts(folds: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    if not folds:
        return {}
    merged = pd.concat(tuple(folds.values()), ignore_index=True)
    if "dataset_of_origin" not in merged.columns:
        raise ValueError("Every fold must contain dataset_of_origin.")
    counts = merged.groupby("dataset_of_origin", sort=True).size()
    return {str(name): int(value) for name, value in counts.items()}


def filter_fold_frames(
    folds: Mapping[str, pd.DataFrame],
    *,
    exclude_dataset: str | Iterable[str] | None = None,
    no_iemocap: bool = False,
    no_ieomcap: bool = False,
) -> FilteredFolds:
    """Apply one resolved exclusion set to every fold without writing new files."""

    if not folds:
        raise ValueError("At least one fold is required.")
    for filename, frame in folds.items():
        if "dataset_of_origin" not in frame.columns:
            raise ValueError(f"{filename} is missing required column: dataset_of_origin")
    requested = collect_exclude_patterns(
        exclude_dataset,
        no_iemocap=no_iemocap,
        no_ieomcap=no_ieomcap,
    )
    available = {
        str(name)
        for frame in folds.values()
        for name in frame["dataset_of_origin"].unique()
    }
    excluded = resolve_excluded_datasets(available, requested)
    excluded_set = set(excluded)
    filtered = {
        filename: frame.loc[
            ~frame["dataset_of_origin"].isin(excluded_set)
        ].copy()
        for filename, frame in folds.items()
    }
    return FilteredFolds(
        folds=filtered,
        requested_patterns=requested,
        excluded_names=excluded,
    )


def apply_dataset_filters(
    fold1: pd.DataFrame,
    fold2: pd.DataFrame,
    *,
    exclude_dataset: str | Iterable[str] | None = None,
    no_iemocap: bool = False,
    no_ieomcap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Tuple-oriented adapter for callers that already hold the two folds."""

    result = filter_fold_frames(
        {FOLD_FILENAMES[0]: fold1, FOLD_FILENAMES[1]: fold2},
        exclude_dataset=exclude_dataset,
        no_iemocap=no_iemocap,
        no_ieomcap=no_ieomcap,
    )
    return result.fold1, result.fold2, result.excluded_names


def load_filtered_folds(
    data_dir: str | os.PathLike[str],
    *,
    exclude_dataset: str | Iterable[str] | None = None,
    no_iemocap: bool = False,
    no_ieomcap: bool = False,
) -> FilteredFolds:
    return filter_fold_frames(
        load_folds(data_dir),
        exclude_dataset=exclude_dataset,
        no_iemocap=no_iemocap,
        no_ieomcap=no_ieomcap,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve and preview training-time dataset filters. This command does "
            "not rewrite fold files."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing full_dataset_fold1.csv and full_dataset_fold2.csv.",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        help="Dataset name/pattern. Repeat or provide a comma-separated list.",
    )
    parser.add_argument("--no-iemocap", action="store_true")
    parser.add_argument(
        "--no-ieomcap",
        action="store_true",
        help="Backward-compatible typo alias for --no-iemocap.",
    )
    parser.add_argument("--list-datasets", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    folds = load_folds(args.data_dir)
    before = dataset_counts(folds)
    if args.list_datasets:
        print("dataset_of_origin\tnum_samples")
        for name, count in before.items():
            print(f"{name}\t{count}")
        return 0

    result = filter_fold_frames(
        folds,
        exclude_dataset=args.exclude_dataset,
        no_iemocap=args.no_iemocap,
        no_ieomcap=args.no_ieomcap,
    )
    if not result.requested_patterns:
        raise ValueError(
            "Provide --exclude-dataset, --no-iemocap, or --no-ieomcap; "
            "use --list-datasets to inspect choices."
        )
    print("Excluded dataset_of_origin values:")
    for name in result.excluded_names:
        print(f"  - {name}")
    for filename in FOLD_FILENAMES:
        print(f"{filename}: {len(folds[filename])} -> {len(result.folds[filename])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""In-memory dataset-of-origin filtering shared by training and evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence
import unicodedata

import pandas as pd

from .preprocessing import FOLD_FILENAMES


IEMOCAP_PATTERN = "IEMOCAP"
IEMOCAP_TYPOS = frozenset({"ieomcap"})
SENTENCE_ONLY_DATASET_NAMES = (
    "EmoTales sentences",
    "Emobank",
    "fb",
)
TRAIN_EVAL_TEXT_OVERLAP_CONTRACT = {
    "version": 1,
    "normalization": "NFKC+casefold+whitespace-collapse",
    "policy": "exclude_train_rows_matching_eval_text",
    "evaluation_rows": "unchanged",
}
TRAIN_EVAL_TEXT_OVERLAP_POLICIES = ("exclude", "keep")


def training_text_overlap_contract(policy: str = "exclude") -> dict[str, object]:
    """Return a fresh contract for the selected training text overlap policy."""

    if policy not in TRAIN_EVAL_TEXT_OVERLAP_POLICIES:
        raise ValueError(f"Unsupported training text overlap policy: {policy!r}.")
    contract = dict(TRAIN_EVAL_TEXT_OVERLAP_CONTRACT)
    if policy == "keep":
        contract["policy"] = "keep_all_training_rows"
    return contract


def validate_training_text_overlap_contract(
    value: object,
    *,
    context: str = "Training text overlap",
) -> dict[str, object]:
    """Accept only complete, supported overlap contracts without type coercion."""

    if isinstance(value, dict) and type(value.get("version")) is int:
        for policy in TRAIN_EVAL_TEXT_OVERLAP_POLICIES:
            expected = training_text_overlap_contract(policy)
            if value == expected:
                return expected
    raise ValueError(f"{context}: invalid training_text_overlap_contract.")


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


def resolve_sentence_only_datasets(
    available_names: Iterable[object],
) -> tuple[str, ...]:
    """Resolve the three approved non-IEMOCAP sentence corpora exactly."""

    available = tuple(sorted({str(name) for name in available_names}, key=str.casefold))
    available_set = set(available)
    missing = [
        expected
        for expected in SENTENCE_ONLY_DATASET_NAMES
        if expected not in available_set
    ]
    if missing:
        choices = "\n".join(f"  - {name}" for name in available) or "  - <none>"
        raise ValueError(
            "--sentence-only requires exactly the approved sentence datasets "
            f"(missing: {', '.join(missing)}).\n"
            f"Available dataset_of_origin values:\n{choices}"
        )
    return SENTENCE_ONLY_DATASET_NAMES


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


def _normalized_overlap_keys(frame: pd.DataFrame, *, split_name: str) -> list[str]:
    """Normalize valid string texts without silently converting missing values."""

    if list(frame.columns).count("text") != 1:
        raise ValueError(f"{split_name} frame must contain exactly one text column.")
    keys: list[str] = []
    for row_position, value in enumerate(frame["text"]):
        if not isinstance(value, str):
            raise ValueError(
                f"{split_name} text at row position {row_position} must be a string; "
                f"got {type(value).__name__}. Missing texts must be handled before splitting."
            )
        keys.append(" ".join(unicodedata.normalize("NFKC", value).casefold().split()))
    return keys


def apply_training_text_overlap_policy(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    policy: str = "exclude",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep or exclude shared training texts while auditing both input folds.

    Matching ignores Unicode compatibility differences, case, and whitespace;
    labels and corpus names do not affect the audit. Neither input is mutated.
    Keeping overlaps retains every training row, label, index, and its order.
    Excluding an all-overlap input returns an empty frame for callers to reject.
    """

    contract = training_text_overlap_contract(policy)
    train_keys = _normalized_overlap_keys(train_frame, split_name="Training")
    eval_keys = _normalized_overlap_keys(eval_frame, split_name="Evaluation")
    eval_key_set = set(eval_keys)
    shared_keys = set(train_keys).intersection(eval_key_set)
    if policy == "exclude":
        keep_train_rows = [key not in eval_key_set for key in train_keys]
        retained = train_frame.loc[keep_train_rows].copy()
    else:
        retained = train_frame.copy()
    audit: dict[str, object] = {
        "contract": contract,
        "train_rows_before": len(train_frame),
        "train_rows_after": len(retained),
        "excluded_train_rows": len(train_frame) - len(retained),
        "eval_rows": len(eval_frame),
        "overlapping_eval_rows": sum(key in shared_keys for key in eval_keys),
        "shared_normalized_texts": len(shared_keys),
    }
    return retained, audit


def exclude_eval_text_overlaps(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Preserve the existing exclusion API and its saved audit contract."""

    return apply_training_text_overlap_policy(train_frame, eval_frame, policy="exclude")


def filter_fold_frames(
    folds: Mapping[str, pd.DataFrame],
    *,
    exclude_dataset: str | Iterable[str] | None = None,
    no_iemocap: bool = False,
    no_ieomcap: bool = False,
    sentence_only: bool = False,
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
    excluded_set = set(resolve_excluded_datasets(available, requested))
    if sentence_only:
        included = set(resolve_sentence_only_datasets(available))
        explicitly_removed = sorted(included.intersection(excluded_set), key=str.casefold)
        if explicitly_removed:
            raise ValueError(
                "--sentence-only keeps exactly EmoTales sentences, Emobank, and fb; "
                "do not also exclude: " + ", ".join(explicitly_removed)
            )
        for filename, frame in folds.items():
            fold_names = {str(name) for name in frame["dataset_of_origin"].unique()}
            missing_from_fold = sorted(included.difference(fold_names), key=str.casefold)
            if missing_from_fold:
                raise ValueError(
                    f"--sentence-only requires all three approved datasets in {filename}; "
                    "missing: " + ", ".join(missing_from_fold)
                )
        excluded_set.update(available.difference(included))
    excluded = tuple(sorted(excluded_set, key=str.casefold))
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
    sentence_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Tuple-oriented adapter for callers that already hold the two folds."""

    result = filter_fold_frames(
        {FOLD_FILENAMES[0]: fold1, FOLD_FILENAMES[1]: fold2},
        exclude_dataset=exclude_dataset,
        no_iemocap=no_iemocap,
        no_ieomcap=no_ieomcap,
        sentence_only=sentence_only,
    )
    return result.fold1, result.fold2, result.excluded_names


def load_filtered_folds(
    data_dir: str | os.PathLike[str],
    *,
    exclude_dataset: str | Iterable[str] | None = None,
    no_iemocap: bool = False,
    no_ieomcap: bool = False,
    sentence_only: bool = False,
) -> FilteredFolds:
    return filter_fold_frames(
        load_folds(data_dir),
        exclude_dataset=exclude_dataset,
        no_iemocap=no_iemocap,
        no_ieomcap=no_ieomcap,
        sentence_only=sentence_only,
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
    parser.add_argument(
        "--sentence-only",
        action="store_true",
        help="Keep only EmoTales sentences, Emobank, and fb.",
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
        sentence_only=args.sentence_only,
    )
    if not result.requested_patterns and not args.sentence_only:
        raise ValueError(
            "Provide --exclude-dataset, --no-iemocap, --no-ieomcap, or "
            "--sentence-only; "
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

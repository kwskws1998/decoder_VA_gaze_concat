"""Canonical data identities and normalized-group splits."""
import hashlib,json,unicodedata
import numpy as np
import pandas as pd

def normalize_text(value: object) -> str:
    """Match the baseline's NFKC, casefold, and whitespace normalization."""

    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def text_groups(frame: pd.DataFrame) -> np.ndarray:
    """Return stable text identities without inspecting any VA labels."""

    return np.asarray([normalize_text(value) for value in frame["text"].fillna("")], dtype=str)


def frame_sha256(frame: pd.DataFrame) -> str:
    """Hash ordered identities, texts, sources, and labels for exact cache ownership."""

    columns = ["index", "text", "dataset_of_origin", "valence", "arousal"]
    return hashlib.sha256(frame.loc[:, columns].to_csv(index=False).encode("utf-8")).hexdigest()


def split_experiences(frame: pd.DataFrame, fraction: float, seed: int):
    """Hold out training text groups for out-of-sample residual experiences."""

    if not 0.0 < fraction < 0.5:
        raise ValueError("experience_fraction must lie strictly between zero and 0.5.")
    if frame.empty:
        raise ValueError("Cannot split an empty training fold.")
    groups = text_groups(frame)
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < 4:
        raise ValueError("At least four distinct training text groups are required.")
    order = np.random.default_rng(seed).permutation(len(unique))
    cumulative = np.cumsum(counts[order])
    cut = int(np.searchsorted(cumulative, max(2, int(round(len(frame) * fraction))))) + 1
    cut = min(max(cut, 2), len(unique) - 2)
    chosen = set(unique[order[:cut]].tolist())
    experience_mask = np.fromiter((group in chosen for group in groups), dtype=bool, count=len(groups))
    base = frame.loc[~experience_mask].reset_index(drop=True).copy()
    experiences = frame.loc[experience_mask].reset_index(drop=True).copy()
    if set(text_groups(base)).intersection(text_groups(experiences)):
        raise AssertionError("Base and experience text groups overlap.")
    return base, experiences


def source_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Persist source composition for each immutable data partition."""

    return {str(key): int(value) for key, value in frame["dataset_of_origin"].value_counts().sort_index().items()}


def contract_sha256(value: dict) -> str:
    """Hash an explicit JSON experiment contract without Python pickle state."""

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

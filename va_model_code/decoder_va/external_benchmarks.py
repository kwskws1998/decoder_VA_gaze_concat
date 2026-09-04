"""Leakage-audited adapters and reports for frozen external VA evaluation."""

from __future__ import annotations

from collections import Counter
import ctypes
from dataclasses import dataclass
import csv
import errno
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import unicodedata
from urllib.request import urlopen
import zipfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import Dataset

from .downloads import sha256_file
from .evaluation import calculate_va_metrics
from .filters import read_fold
from .model import (
    ARCHITECTURE_MANIFEST_FILENAME,
    ARCHITECTURE_MANIFEST_VERSION,
    DEFAULT_CLASSIFIER_DROPOUT,
    DEFAULT_GAZE_PROJECTION_DIM,
    DEFAULT_GAZE_PROJECTION_DROPOUT,
    OUTPUT_ACTIVATION,
    PRE_REDISTRIBUTION_MANIFEST_VERSION,
    SAFE_WEIGHTS_FILENAME,
)
from .preprocessing import FOLD_FILENAMES
from .redistribution import validate_redistribution_contract


EXTERNAL_EVALUATION_SCHEMA_VERSION = 1
OMG_NAME = "omg-emotion"
MSP_NAME = "msp-podcast"
IDEST_NAME = "idest-english"
SEMEVAL_NAME = "semeval-2026-task2-subtask1"
EXTERNAL_BENCHMARKS = (OMG_NAME, MSP_NAME, IDEST_NAME, SEMEVAL_NAME)
OMG_REVISION = "5931b237e92d68d04931bb932854fac6d9cd6a41"
OMG_TEST_ROWS = 2229
MSP_OFFICIAL_TEST_ROWS = {"test1": 46294, "test2": 14822}
IDEST_TEST_ROWS = 250
IDEST_OSF_PROJECT = "9tga3"
IDEST_OSF_FILE_ID = "xh4kv"
IDEST_FILE = {
    "filename": "IDEST_Database.csv",
    "url": f"https://osf.io/download/{IDEST_OSF_FILE_ID}/",
    "sha256": "91e3e05a9495833cafe6fcb59a445b3f248c7c2a3881549e9d8741a7f94bc0b7",
}
SEMEVAL_REVISION = "50abd23fb884d3dd693c2df479124bcf6c153086"
SEMEVAL_TEST_ROWS = 1737
SEMEVAL_TEST_USERS = 91
SEMEVAL_TEST_FILES = {
    "inputs": {
        "filename": "test_subtask1.csv",
        "repository_path": "datasets/TEST_RELEASE_5JAN2026/test_subtask1.csv",
        "sha256": "61500316be2d5fcd88979e7f12885e4a42d3b9e71e1e2feb15deeda6134ff5fd",
    },
    "labels": {
        "filename": "test_labels_subtask1.csv",
        "repository_path": (
            "datasets/TEST_LABELS_RELEASE_23FEB2026/test_labels_subtask1.csv"
        ),
        "sha256": "9d4734b93112c9db07144f404e013f55abb3f847c96b3cfd2550d8340cf5be3c",
    },
}
MSP_CANONICAL_LABEL_COLUMNS = {
    "label_id_column": "FileName",
    "split_column": "Split_Set",
    "arousal_column": "EmoAct",
    "valence_column": "EmoVal",
}
MODEL_OUTPUT_NAMES = ("valence", "arousal")
EXTERNAL_EVALUATION_COMPLETION_MARKER = "COMPLETED"

PERSISTED_PREDICTION_COLUMNS = (
    "index",
    "held_out_fold",
    "text",
    "dataset_of_origin",
    "valence",
    "arousal",
    "pred_valence",
    "pred_arousal",
)
PERSISTED_METRIC_NAMES = (
    "n_examples",
    "mse_valence",
    "rmse_valence",
    "mae_valence",
    "pearson_corr_valence",
    "ccc_valence",
    "mse_arousal",
    "rmse_arousal",
    "mae_arousal",
    "pearson_corr_arousal",
    "ccc_arousal",
    "mse_mean",
    "mae_mean",
    "pearson_corr_mean",
    "ccc_mean",
)
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
TOKENIZER_PAYLOAD_FILENAMES = frozenset(
    {
        "tokenizer.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "spiece.model",
        "vocab.json",
        "vocab.txt",
    }
)
MAX_MSP_ZIP_TABLE_MEMBER_BYTES = 256 * 1024 * 1024
MAX_MSP_ZIP_TEXT_MEMBER_BYTES = 32 * 1024 * 1024
MAX_MSP_ZIP_TOTAL_BYTES = 512 * 1024 * 1024

OMG_TEST_FILES = {
    "transcripts": {
        "filename": "omg_TestTranscripts.tsv",
        "sha256": "448e9705d888cd3512ccb1151d11bbc00d4f9931c9a9a1bcd387ac81ba1f1b50",
    },
    "labels": {
        "filename": "omg_TestVideos_WithLabels.csv",
        "sha256": "1a43518cf8f0c8f67c56393171c23aab9b9326976dafc0f8aea76c03fb29ee03",
    },
}

RUN_CONTRACT_FIELDS = (
    "architecture_manifest_version",
    "model",
    "loss",
    "output_dim",
    "dtype",
    "model_id",
    "model_revision",
    "finetuning_mode",
    "gaze_fusion",
    "gaze_features",
    "gaze_feature_indices",
    "features_used",
    "gaze_concat_order",
    "pooling_position",
    "output_activation",
    "et_model_id",
    "et_revision",
    "et_filename",
    "max_length",
    "eval_batch_size",
    "excluded_dataset_names",
    "fold_sha256",
)


@dataclass(frozen=True)
class ExternalBenchmarkData:
    """Canonical text rows, fixed scale mapping, and immutable source metadata."""

    name: str
    version: str
    split: str
    frame: pd.DataFrame
    native_scale: Mapping[str, tuple[float, float]]
    model_to_native_scale: tuple[float, float]
    model_to_native_offset: tuple[float, float]
    source_manifest: Mapping[str, Any]
    join_report: Mapping[str, Any]
    official_group_column: str | None = None
    context_policy: str = "one current text only"
    prediction_metadata_columns: tuple[str, ...] = ()

    def labels_model_scale(self) -> np.ndarray:
        return self.frame.loc[:, ["valence", "arousal"]].to_numpy(dtype=np.float64)

    def labels_native_scale(self) -> np.ndarray:
        return self.frame.loc[:, ["native_valence", "native_arousal"]].to_numpy(
            dtype=np.float64
        )

    def predictions_to_native(self, predictions) -> np.ndarray:
        return model_predictions_to_native(
            predictions,
            scale=self.model_to_native_scale,
            offset=self.model_to_native_offset,
        )


@dataclass(frozen=True)
class SavedRunMember:
    """One immutable fold-specific final model from a completed two-fold run."""

    name: str
    held_out_fold: int
    training_fold: int
    model_dir: Path
    run_manifest_path: Path
    run_manifest: Mapping[str, Any]
    architecture_manifest_path: Path
    architecture_manifest: Mapping[str, Any]
    weights_path: Path
    file_sha256: Mapping[str, str]


class TokenizedTextDataset(Dataset):
    """Label-free tokenizer dataset used only for frozen model inference."""

    def __init__(self, texts: Sequence[object], tokenizer, *, max_length: int) -> None:
        if int(max_length) <= 0:
            raise ValueError("max_length must be positive.")
        self.texts = ["" if text is None else str(text) for text in texts]
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[item],
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        output: dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if key not in {"input_ids", "attention_mask", "token_type_ids"}:
                continue
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            if tensor.ndim == 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 1:
                raise ValueError(f"Tokenizer returned non-vector {key} for one example.")
            output[key] = tensor.to(dtype=torch.long)
        if "input_ids" not in output:
            raise ValueError("Tokenizer output is missing input_ids.")
        if output["input_ids"].numel() == 0:
            fallback_id = getattr(self.tokenizer, "eos_token_id", None)
            if fallback_id is None:
                fallback_id = getattr(self.tokenizer, "pad_token_id", None)
            if fallback_id is None:
                raise ValueError(
                    "Tokenizer returned no tokens for blank text and has no EOS or pad token."
                )
            output["input_ids"] = torch.tensor([fallback_id], dtype=torch.long)
            output["attention_mask"] = torch.ones(1, dtype=torch.long)
            if "token_type_ids" in output:
                output["token_type_ids"] = torch.zeros(1, dtype=torch.long)
        if "attention_mask" not in output:
            output["attention_mask"] = torch.ones_like(output["input_ids"])
        return output


class TextBatchCollator:
    """Right-pad label-free external-evaluation batches with the saved tokenizer."""

    def __init__(self, tokenizer, *, pad_to_multiple_of: int | None = 8) -> None:
        if pad_to_multiple_of is not None and int(pad_to_multiple_of) <= 0:
            raise ValueError("pad_to_multiple_of must be positive when provided.")
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token_id", None) is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Iterable[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        items = [dict(feature) for feature in features]
        if not items:
            raise ValueError("Cannot collate an empty text batch.")
        serializable = [
            {
                key: value.tolist() if torch.is_tensor(value) else value
                for key, value in item.items()
            }
            for item in items
        ]
        batch = self.tokenizer.pad(
            serializable,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return {key: torch.as_tensor(value) for key, value in dict(batch).items()}


def _json_ready(value):
    """Convert paths, NumPy values, and non-finite floats into strict JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    """Read a required JSON object with one contextual error contract."""

    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain one JSON object: {path}")
    return value


def _require_regular_artifact(path: Path, description: str) -> None:
    """Require one local regular file and reject provenance-escaping symlinks."""

    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    if path.is_symlink():
        raise ValueError(f"{description} must not be a symlink: {path}")


def _atomic_publish_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically rename a staged directory while refusing every existing target."""

    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_bytes, -100, target_bytes, 1)
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), target)
        raise OSError(error, os.strerror(error), target)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, target_bytes, 0x00000004)
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), target)
        raise OSError(error, os.strerror(error), target)
    if target.exists():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), target)
    os.rename(source, target)


def _require_columns(frame: pd.DataFrame, required: Sequence[str], description: str) -> None:
    """Reject incomplete external tables without guessing missing fields."""

    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(
            f"{description} is missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(map(str, frame.columns))}."
        )


def _numeric_column(frame: pd.DataFrame, name: str, description: str) -> np.ndarray:
    """Convert one label column and require a complete finite vector."""

    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = np.flatnonzero(~np.isfinite(values))[:10].tolist()
        raise ValueError(
            f"{description} column {name!r} contains missing/non-finite values at "
            f"row position(s): {bad}."
        )
    return values


def _require_range(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    name: str,
    description: str,
    tolerance: float = 1e-9,
) -> None:
    """Require a documented native scale rather than observed test normalization."""

    invalid = np.logical_or(values < lower - tolerance, values > upper + tolerance)
    if invalid.any():
        examples = values[invalid][:10].tolist()
        raise ValueError(
            f"{description} {name} must stay in [{lower:g}, {upper:g}]; "
            f"found {examples}."
        )


def _text_sha256(value: object) -> str:
    """Hash one exact transcript without persisting licensed text in results."""

    return sha256(str(value or "").encode("utf-8")).hexdigest()


def normalize_overlap_text(value: object) -> str:
    """Normalize Unicode, case, and whitespace for fine-tuning overlap auditing."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(normalized.split())


def model_predictions_to_native(
    predictions,
    *,
    scale: Sequence[float],
    offset: Sequence[float],
) -> np.ndarray:
    """Apply one prespecified affine map from model [V,A] scale to native [V,A]."""

    array = np.asarray(predictions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Predictions must have shape [examples, 2], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Predictions must contain only finite values.")
    if len(scale) != 2 or len(offset) != 2:
        raise ValueError("Scale and offset must each contain valence and arousal values.")
    if bool(((array < -1e-6) | (array > 1.0 + 1e-6)).any()):
        raise ValueError("Hard-sigmoid model predictions must stay in [0, 1].")
    return array * np.asarray(scale, dtype=np.float64) + np.asarray(
        offset, dtype=np.float64
    )


def fixed_unweighted_ensemble(member_predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Average every prespecified fold member without labels or learned weights."""

    if len(member_predictions) < 2:
        raise ValueError("The external ensemble requires at least two model members.")
    arrays = [np.asarray(value, dtype=np.float64) for value in member_predictions]
    expected_shape = arrays[0].shape
    if len(expected_shape) != 2 or expected_shape[1] != 2:
        raise ValueError("Every member prediction array must have shape [examples, 2].")
    for array in arrays:
        if array.shape != expected_shape:
            raise ValueError("All ensemble members must predict the same ordered rows.")
        if not np.isfinite(array).all():
            raise ValueError("Ensemble members must contain only finite predictions.")
    return np.mean(np.stack(arrays, axis=0), axis=0)


def _download_one(
    url: str,
    target: Path,
    expected_sha256: str,
    *,
    description: str,
) -> None:
    """Download one pinned public benchmark file atomically and verify its digest."""

    if target.exists():
        if not target.is_file():
            raise FileExistsError(
                f"{description} target exists and is not a file: {target}"
            )
        actual = sha256_file(target)
        if actual != expected_sha256:
            raise ValueError(
                f"Existing {description} file has SHA256 {actual}, expected "
                f"{expected_sha256}: {target}"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Incomplete download already exists: {temporary}")
    try:
        with urlopen(url, timeout=120) as response, open(temporary, "wb") as output_file:
            shutil.copyfileobj(response, output_file)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise ValueError(
                f"Downloaded {description} file has SHA256 {actual}, expected "
                f"{expected_sha256}."
            )
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def download_pinned_omg_test(raw_dir: str | Path) -> dict[str, Path]:
    """Download only official pinned OMG test transcript and gold-label tables."""

    directory = Path(raw_dir).expanduser().resolve()
    resolved: dict[str, Path] = {}
    for role, contract in OMG_TEST_FILES.items():
        target = directory / contract["filename"]
        url = (
            "https://raw.githubusercontent.com/knowledgetechnologyuhh/"
            f"OMGEmotionChallenge/{OMG_REVISION}/{contract['filename']}"
        )
        _download_one(
            url,
            target,
            contract["sha256"],
            description=f"OMG {role}",
        )
        resolved[role] = target
    return resolved


def load_omg_emotion_test(
    raw_dir: str | Path,
    *,
    strict_official_contract: bool = True,
) -> ExternalBenchmarkData:
    """Join official OMG test gold rows to comma-delimited utterance transcripts."""

    directory = Path(raw_dir).expanduser().resolve()
    transcript_path = directory / OMG_TEST_FILES["transcripts"]["filename"]
    label_path = directory / OMG_TEST_FILES["labels"]["filename"]
    if not transcript_path.is_file():
        raise FileNotFoundError(f"OMG test transcript file not found: {transcript_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"OMG test label file not found: {label_path}")
    input_hashes = {
        "transcripts": sha256_file(transcript_path),
        "labels": sha256_file(label_path),
    }
    if strict_official_contract:
        for role, actual in input_hashes.items():
            expected = OMG_TEST_FILES[role]["sha256"]
            if actual != expected:
                raise ValueError(
                    f"OMG {role} SHA256 mismatch: {actual}; expected {expected}. "
                    f"Use pinned revision {OMG_REVISION}."
                )

    transcripts = pd.read_csv(
        transcript_path,
        sep=",",
        quotechar='"',
        keep_default_na=False,
        dtype=str,
    )
    labels = pd.read_csv(
        label_path,
        sep=",",
        quotechar='"',
        keep_default_na=False,
        dtype=str,
    )
    transcript_columns = ("link", "video", "utterance", "transcript")
    label_columns = (
        "link",
        "start",
        "end",
        "video",
        "utterance",
        "arousal",
        "valence",
        "EmotionMaxVote",
    )
    _require_columns(transcripts, transcript_columns, "OMG transcript table")
    _require_columns(labels, label_columns, "OMG gold-label table")
    key_columns = ["video", "utterance"]
    for description, frame in (("transcript", transcripts), ("gold-label", labels)):
        blank_key = frame.loc[:, key_columns].apply(
            lambda column: column.astype(str).str.strip().eq("")
        )
        if bool(blank_key.any(axis=None)):
            raise ValueError(f"OMG {description} table contains a blank join key.")
        duplicated = frame.duplicated(key_columns, keep=False)
        if bool(duplicated.any()):
            examples = frame.loc[duplicated, key_columns].head(10).to_dict("records")
            raise ValueError(f"OMG {description} join keys are not unique: {examples}.")

    label_keys = pd.MultiIndex.from_frame(labels.loc[:, key_columns])
    transcript_keys = pd.MultiIndex.from_frame(transcripts.loc[:, key_columns])
    transcript_only = transcript_keys.difference(label_keys)
    labels = labels.copy()
    labels["_official_row"] = np.arange(len(labels), dtype=np.int64)
    transcript_subset = transcripts.loc[:, key_columns + ["transcript"]].copy()
    transcript_subset["_transcript_present"] = True
    joined = labels.merge(
        transcript_subset,
        on=key_columns,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    missing = joined["_transcript_present"].isna()
    if bool(missing.any()):
        examples = joined.loc[missing, key_columns].head(10).to_dict("records")
        raise ValueError(f"OMG gold rows are missing transcript matches: {examples}.")
    joined = joined.sort_values("_official_row", kind="stable").reset_index(drop=True)
    if strict_official_contract and len(joined) != OMG_TEST_ROWS:
        raise ValueError(
            f"OMG official test must contain {OMG_TEST_ROWS} gold rows; got {len(joined)}."
        )

    native_arousal = _numeric_column(joined, "arousal", "OMG test")
    native_valence = _numeric_column(joined, "valence", "OMG test")
    _require_range(
        native_arousal,
        lower=0.0,
        upper=1.0,
        name="arousal",
        description="OMG test",
    )
    _require_range(
        native_valence,
        lower=-1.0,
        upper=1.0,
        name="valence",
        description="OMG test",
    )
    text = joined["transcript"].astype(str)
    frame = pd.DataFrame(
        {
            "index": [
                f"{video}::{utterance}"
                for video, utterance in zip(joined["video"], joined["utterance"])
            ],
            "benchmark_id": [
                f"{video}::{utterance}"
                for video, utterance in zip(joined["video"], joined["utterance"])
            ],
            "text": text,
            "text_sha256": [_text_sha256(value) for value in text],
            "is_empty_text": text.str.strip().eq("").to_numpy(dtype=bool),
            "dataset_of_origin": "OMG-Emotion",
            "split": "test",
            "video": joined["video"].astype(str),
            "utterance": joined["utterance"].astype(str),
            "valence": (native_valence + 1.0) / 2.0,
            "arousal": native_arousal,
            "native_valence": native_valence,
            "native_arousal": native_arousal,
        }
    )
    return ExternalBenchmarkData(
        name=OMG_NAME,
        version=(
            f"official-github-{OMG_REVISION}"
            if strict_official_contract
            else "unverified-local"
        ),
        split="test",
        frame=frame,
        native_scale={"valence": (-1.0, 1.0), "arousal": (0.0, 1.0)},
        model_to_native_scale=(2.0, 1.0),
        model_to_native_offset=(-1.0, 0.0),
        source_manifest={
            "canonical_contract_verified": bool(strict_official_contract),
            "official_reference_revision": OMG_REVISION,
            "raw_directory": str(directory),
            "files": {
                "transcripts": {
                    "path": str(transcript_path),
                    "sha256": input_hashes["transcripts"],
                    "delimiter": "comma",
                },
                "labels": {
                    "path": str(label_path),
                    "sha256": input_hashes["labels"],
                    "delimiter": "comma",
                },
            },
            "license": "CC BY-NC-SA 3.0 DE corpus terms in the official README",
            "source_url": (
                "https://github.com/knowledgetechnologyuhh/"
                f"OMGEmotionChallenge/tree/{OMG_REVISION}"
            ),
        },
        join_report={
            "gold_rows": int(len(labels)),
            "transcript_rows": int(len(transcripts)),
            "matched_gold_rows": int(len(joined)),
            "missing_gold_transcripts": 0,
            "transcript_only_rows": int(len(transcript_only)),
            "empty_matched_transcripts": int(frame["is_empty_text"].sum()),
            "join_keys": key_columns,
            "join_validation": "one_to_one; gold-left; official gold order preserved",
        },
        official_group_column="video",
        context_policy="one current utterance transcript only",
        prediction_metadata_columns=("video", "utterance"),
    )


def download_pinned_idest_english(raw_dir: str | Path) -> Path:
    """Download the pinned OSF IDEST CSV without vendoring the public dataset."""

    directory = Path(raw_dir).expanduser().resolve()
    target = directory / IDEST_FILE["filename"]
    _download_one(
        IDEST_FILE["url"],
        target,
        IDEST_FILE["sha256"],
        description="IDEST English database",
    )
    return target


def load_idest_english(
    raw_dir: str | Path,
    *,
    strict_official_contract: bool = True,
) -> ExternalBenchmarkData:
    """Load all 250 English IDEST stories with their published English mean VA."""

    directory = Path(raw_dir).expanduser().resolve()
    path = directory / IDEST_FILE["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"IDEST database file not found: {path}")
    digest = sha256_file(path)
    if strict_official_contract and digest != IDEST_FILE["sha256"]:
        raise ValueError(
            f"IDEST SHA256 mismatch: {digest}; expected {IDEST_FILE['sha256']}. "
            f"Use pinned OSF file {IDEST_OSF_FILE_ID}."
        )

    raw = pd.read_csv(
        path,
        sep=";",
        quotechar='"',
        encoding="cp1252",
        keep_default_na=False,
        dtype=str,
    )
    required = (
        "code",
        "country",
        "language",
        "text_english",
        "number_raters_English",
        "valence_mean_English",
        "arousal_mean_English",
        "characters_English",
        "words_English",
        "StoryType",
    )
    _require_columns(raw, required, "IDEST database")
    blank_cells = raw.apply(lambda column: column.astype(str).str.strip().eq(""))
    fully_blank = blank_cells.all(axis=1).to_numpy(dtype=bool)
    if fully_blank.any():
        first_blank = int(np.flatnonzero(fully_blank)[0])
        if bool((~fully_blank[first_blank:]).any()):
            raise ValueError("IDEST fully blank rows must be trailing rows only.")
    selected = raw.loc[~fully_blank].copy().reset_index(drop=True)
    if strict_official_contract and len(selected) != IDEST_TEST_ROWS:
        raise ValueError(
            f"IDEST English must contain {IDEST_TEST_ROWS} stories; got {len(selected)}."
        )

    for column in ("code", "language", "text_english"):
        if selected[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"IDEST database contains a blank {column} value.")
    if selected["code"].duplicated(keep=False).any():
        examples = selected.loc[
            selected["code"].duplicated(keep=False), "code"
        ].head(10).tolist()
        raise ValueError(f"IDEST story codes are not unique: {examples}.")

    native_valence = _numeric_column(
        selected, "valence_mean_English", "IDEST English"
    )
    native_arousal = _numeric_column(
        selected, "arousal_mean_English", "IDEST English"
    )
    _require_range(
        native_valence,
        lower=1.0,
        upper=9.0,
        name="valence",
        description="IDEST English",
    )
    _require_range(
        native_arousal,
        lower=1.0,
        upper=9.0,
        name="arousal",
        description="IDEST English",
    )
    number_raters = _numeric_column(
        selected, "number_raters_English", "IDEST English"
    )
    characters = _numeric_column(selected, "characters_English", "IDEST English")
    words = _numeric_column(selected, "words_English", "IDEST English")
    story_type = _numeric_column(selected, "StoryType", "IDEST English")
    if bool((number_raters <= 0).any()):
        raise ValueError("IDEST English number_raters_English must be positive.")
    if bool((characters <= 0).any() or (words <= 0).any()):
        raise ValueError("IDEST English character and word counts must be positive.")
    if bool((story_type < 1).any() or (story_type > 8).any()):
        raise ValueError("IDEST English StoryType must stay in [1, 8].")

    text = selected["text_english"].astype(str)
    normalized_text = text.map(normalize_overlap_text)
    frame = pd.DataFrame(
        {
            "index": selected["code"].astype(str),
            "benchmark_id": selected["code"].astype(str),
            "text": text,
            "text_sha256": [_text_sha256(value) for value in text],
            "is_empty_text": text.str.strip().eq("").to_numpy(dtype=bool),
            "dataset_of_origin": "IDEST English",
            "split": "all-250-zero-shot-test",
            "idest_code": selected["code"].astype(str),
            "source_country": selected["country"].astype(str),
            "source_language": selected["language"].astype(str),
            "number_raters_english": number_raters,
            "characters_english": characters,
            "words_english": words,
            "story_type": story_type.astype(np.int64),
            "valence": (native_valence - 1.0) / 8.0,
            "arousal": (native_arousal - 1.0) / 8.0,
            "native_valence": native_valence,
            "native_arousal": native_arousal,
        }
    )
    return ExternalBenchmarkData(
        name=IDEST_NAME,
        version=(
            f"osf-{IDEST_OSF_FILE_ID}-v1"
            if strict_official_contract
            else "unverified-local"
        ),
        split="all-250-zero-shot-test",
        frame=frame,
        native_scale={"valence": (1.0, 9.0), "arousal": (1.0, 9.0)},
        model_to_native_scale=(8.0, 8.0),
        model_to_native_offset=(1.0, 1.0),
        source_manifest={
            "canonical_contract_verified": bool(strict_official_contract),
            "osf_project": IDEST_OSF_PROJECT,
            "osf_file_id": IDEST_OSF_FILE_ID,
            "file": {
                "path": str(path),
                "sha256": digest,
                "delimiter": "semicolon",
                "encoding": "Windows-1252",
                "encoding_note": (
                    "the OSF README calls the file Latin1, but its smart-punctuation "
                    "bytes use Windows-1252 code points"
                ),
            },
            "english_input_column": "text_english",
            "english_valence_column": "valence_mean_English",
            "english_arousal_column": "arousal_mean_English",
            "license": (
                "OSF node has no explicit dataset license metadata; obtain from the "
                "public source and cite Kaakinen et al. (2022); do not redistribute "
                "from evaluation outputs"
            ),
            "source_url": f"https://osf.io/{IDEST_OSF_PROJECT}/",
        },
        join_report={
            "raw_csv_rows": int(len(raw)),
            "fully_blank_trailing_rows_ignored": int(fully_blank.sum()),
            "english_story_rows": int(len(frame)),
            "unique_story_codes": int(frame["benchmark_id"].nunique()),
            "unique_normalized_english_texts": int(normalized_text.nunique()),
            "duplicate_normalized_english_text_rows": int(
                normalized_text.duplicated(keep=False).sum()
            ),
            "row_policy": "official nonblank code rows in source-file order",
            "label_policy": "published English-translation mean ratings only",
        },
        official_group_column=None,
        context_policy="one English translated short story only",
        prediction_metadata_columns=(
            "idest_code",
            "source_country",
            "source_language",
            "number_raters_english",
            "characters_english",
            "words_english",
            "story_type",
        ),
    )


def download_pinned_semeval_subtask1_test(raw_dir: str | Path) -> dict[str, Path]:
    """Download only pinned SemEval Subtask 1 test inputs and released gold labels."""

    directory = Path(raw_dir).expanduser().resolve()
    resolved: dict[str, Path] = {}
    root = (
        "https://raw.githubusercontent.com/semeval2026task2/"
        f"EmotionValArouTimeVariation2026/{SEMEVAL_REVISION}"
    )
    for role, contract in SEMEVAL_TEST_FILES.items():
        target = directory / contract["filename"]
        _download_one(
            f"{root}/{contract['repository_path']}",
            target,
            contract["sha256"],
            description=f"SemEval Subtask 1 {role}",
        )
        resolved[role] = target
    return resolved


def _strict_boolean_column(
    frame: pd.DataFrame,
    name: str,
    description: str,
) -> np.ndarray:
    """Parse a canonical True/False CSV field without Python truthiness guessing."""

    normalized = frame[name].astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin(("true", "false"))
    if invalid.any():
        examples = frame.loc[invalid, name].head(10).tolist()
        raise ValueError(
            f"{description} column {name!r} must contain only True/False; found "
            f"{examples}."
        )
    return normalized.eq("true").to_numpy(dtype=bool)


def load_semeval_2026_subtask1_test(
    raw_dir: str | Path,
    *,
    strict_official_contract: bool = True,
) -> ExternalBenchmarkData:
    """Join pinned SemEval-2026 Task 2 Subtask 1 test text to released gold VA."""

    directory = Path(raw_dir).expanduser().resolve()
    paths = {
        role: directory / contract["filename"]
        for role, contract in SEMEVAL_TEST_FILES.items()
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"SemEval Subtask 1 {role} file not found: {path}")
    hashes = {role: sha256_file(path) for role, path in paths.items()}
    if strict_official_contract:
        for role, actual in hashes.items():
            expected = SEMEVAL_TEST_FILES[role]["sha256"]
            if actual != expected:
                raise ValueError(
                    f"SemEval Subtask 1 {role} SHA256 mismatch: {actual}; expected "
                    f"{expected}. Use pinned revision {SEMEVAL_REVISION}."
                )

    inputs = pd.read_csv(paths["inputs"], keep_default_na=False, dtype=str)
    labels = pd.read_csv(paths["labels"], keep_default_na=False, dtype=str)
    key_columns = ["user_id", "text_id"]
    shared_columns = [
        "text",
        "timestamp",
        "collection_phase",
        "is_words",
        "is_seen_user",
    ]
    input_columns = tuple(key_columns + shared_columns)
    label_columns = tuple(
        key_columns
        + ["text", "timestamp", "collection_phase", "is_words"]
        + ["valence", "arousal", "is_seen_user"]
    )
    _require_columns(inputs, input_columns, "SemEval Subtask 1 test inputs")
    _require_columns(labels, label_columns, "SemEval Subtask 1 test labels")
    for description, table in (("inputs", inputs), ("labels", labels)):
        blank_key = table.loc[:, key_columns].apply(
            lambda column: column.astype(str).str.strip().eq("")
        )
        if bool(blank_key.any(axis=None)):
            raise ValueError(f"SemEval Subtask 1 {description} contains a blank key.")
        duplicated = table.duplicated(key_columns, keep=False)
        if duplicated.any():
            examples = table.loc[duplicated, key_columns].head(10).to_dict("records")
            raise ValueError(
                f"SemEval Subtask 1 {description} keys are not unique: {examples}."
            )

    inputs = inputs.copy()
    inputs["_official_row"] = np.arange(len(inputs), dtype=np.int64)
    joined = inputs.merge(
        labels,
        on=key_columns,
        how="outer",
        sort=False,
        suffixes=("", "_gold"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = joined["_merge"].ne("both")
    if unmatched.any():
        examples = joined.loc[unmatched, key_columns + ["_merge"]].head(10)
        raise ValueError(
            "SemEval Subtask 1 input/label keys do not match one-to-one: "
            f"{examples.to_dict('records')}."
        )
    for column in shared_columns:
        mismatch = joined[column].astype(str).ne(joined[f"{column}_gold"].astype(str))
        if mismatch.any():
            examples = joined.loc[mismatch, key_columns].head(10).to_dict("records")
            raise ValueError(
                f"SemEval Subtask 1 input/label {column} values disagree: {examples}."
            )
    joined = joined.sort_values("_official_row", kind="stable").reset_index(drop=True)
    if strict_official_contract and len(joined) != SEMEVAL_TEST_ROWS:
        raise ValueError(
            f"SemEval Subtask 1 test must contain {SEMEVAL_TEST_ROWS} rows; "
            f"got {len(joined)}."
        )

    if joined["text"].astype(str).str.strip().eq("").any():
        raise ValueError("SemEval Subtask 1 test contains blank text.")
    try:
        pd.to_datetime(joined["timestamp"], errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError("SemEval Subtask 1 contains an invalid timestamp.") from exc
    collection_phase = _numeric_column(
        joined, "collection_phase", "SemEval Subtask 1 test"
    )
    if bool(
        (collection_phase < 1).any()
        or (collection_phase > 7).any()
        or (collection_phase != np.floor(collection_phase)).any()
    ):
        raise ValueError("SemEval Subtask 1 collection_phase must be an integer in [1, 7].")
    is_words = _strict_boolean_column(joined, "is_words", "SemEval Subtask 1 test")
    is_seen_user = _strict_boolean_column(
        joined, "is_seen_user", "SemEval Subtask 1 test"
    )
    seen_by_user = pd.DataFrame(
        {"user_id": joined["user_id"].astype(str), "is_seen_user": is_seen_user}
    ).groupby("user_id", sort=False)["is_seen_user"].nunique()
    if bool((seen_by_user != 1).any()):
        raise ValueError("SemEval Subtask 1 is_seen_user must be constant within a user.")
    user_count = int(joined["user_id"].astype(str).nunique())
    if strict_official_contract and user_count != SEMEVAL_TEST_USERS:
        raise ValueError(
            f"SemEval Subtask 1 test must contain {SEMEVAL_TEST_USERS} users; "
            f"got {user_count}."
        )

    native_valence = _numeric_column(joined, "valence", "SemEval Subtask 1 test")
    native_arousal = _numeric_column(joined, "arousal", "SemEval Subtask 1 test")
    _require_range(
        native_valence,
        lower=-2.0,
        upper=2.0,
        name="valence",
        description="SemEval Subtask 1 released test",
    )
    _require_range(
        native_arousal,
        lower=0.0,
        upper=2.0,
        name="arousal",
        description="SemEval Subtask 1 released test",
    )

    user_id = joined["user_id"].astype(str)
    text_id = joined["text_id"].astype(str)
    benchmark_id = [
        f"{user}::{text}" for user, text in zip(user_id.tolist(), text_id.tolist())
    ]
    text = joined["text"].astype(str)
    normalized_text = text.map(normalize_overlap_text)
    frame = pd.DataFrame(
        {
            "index": benchmark_id,
            "benchmark_id": benchmark_id,
            "text": text,
            "text_sha256": [_text_sha256(value) for value in text],
            "is_empty_text": text.str.strip().eq("").to_numpy(dtype=bool),
            "dataset_of_origin": "SemEval-2026 Task 2 Subtask 1",
            "split": "official-test-zero-shot",
            "user_id": user_id,
            "text_id": text_id,
            "timestamp": joined["timestamp"].astype(str),
            "collection_phase": collection_phase.astype(np.int64),
            "is_words": is_words,
            "is_seen_user": is_seen_user,
            "valence": (native_valence + 2.0) / 4.0,
            "arousal": native_arousal / 2.0,
            "native_valence": native_valence,
            "native_arousal": native_arousal,
        }
    )
    return ExternalBenchmarkData(
        name=SEMEVAL_NAME,
        version=f"official-github-{SEMEVAL_REVISION}" if strict_official_contract else "unverified-local",
        split="official-test-zero-shot",
        frame=frame,
        native_scale={"valence": (-2.0, 2.0), "arousal": (0.0, 2.0)},
        model_to_native_scale=(4.0, 2.0),
        model_to_native_offset=(-2.0, 0.0),
        source_manifest={
            "canonical_contract_verified": bool(strict_official_contract),
            "official_reference_revision": SEMEVAL_REVISION,
            "files": {
                role: {
                    "path": str(paths[role]),
                    "repository_path": SEMEVAL_TEST_FILES[role]["repository_path"],
                    "sha256": hashes[role],
                    "delimiter": "comma",
                    "encoding": "UTF-8",
                }
                for role in ("inputs", "labels")
            },
            "license": "CC0-1.0 at the official task repository",
            "source_url": (
                "https://github.com/semeval2026task2/"
                f"EmotionValArouTimeVariation2026/tree/{SEMEVAL_REVISION}"
            ),
            "label_scale_note": (
                "the pinned train and released test-label CSVs use centered valence "
                "[-2,2] and arousal [0,2], although the task paper describes original "
                "valence [0,4]; the pinned files control this evaluator"
            ),
            "target_training_data_used": False,
        },
        join_report={
            "test_input_rows": int(len(inputs)),
            "gold_label_rows": int(len(labels)),
            "matched_rows": int(len(frame)),
            "users": user_count,
            "official_seen_user_rows": int(is_seen_user.sum()),
            "official_unseen_user_rows": int((~is_seen_user).sum()),
            "essay_rows": int((~is_words).sum()),
            "feeling_word_rows": int(is_words.sum()),
            "unique_normalized_texts": int(normalized_text.nunique()),
            "duplicate_normalized_text_rows": int(
                normalized_text.duplicated(keep=False).sum()
            ),
            "join_keys": key_columns,
            "join_validation": (
                "one_to_one; exact shared metadata match; official test-input order "
                "preserved"
            ),
        },
        official_group_column="user_id",
        context_policy=(
            "one current test text only; user ID, timestamp, history, is_seen_user, "
            "and gold labels are excluded from model inputs"
        ),
        prediction_metadata_columns=(
            "user_id",
            "text_id",
            "timestamp",
            "collection_phase",
            "is_words",
            "is_seen_user",
        ),
    )


def _sniff_delimiter(text: str, description: str) -> str:
    """Distinguish comma and tab tables from their header content, not suffix."""

    first_lines = "\n".join(text.splitlines()[:10])
    try:
        dialect = csv.Sniffer().sniff(first_lines, delimiters=",\t")
    except csv.Error as exc:
        raise ValueError(f"Cannot infer delimiter for {description}.") from exc
    return dialect.delimiter


def _read_transcript_table_bytes(
    payload: bytes,
    *,
    description: str,
    id_column: str | None,
    text_column: str | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Read one UTF-8 CSV/TSV transcript table with explicit or safe alias columns."""

    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{description} must use UTF-8 text encoding.") from exc
    delimiter = _sniff_delimiter(decoded, description)
    frame = pd.read_csv(
        io.StringIO(decoded),
        sep=delimiter,
        quotechar='"',
        keep_default_na=False,
        dtype=str,
    )
    if id_column is not None and id_column not in frame.columns:
        raise ValueError(
            f"{description} does not contain requested ID column {id_column!r}. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )
    if text_column is not None and text_column not in frame.columns:
        raise ValueError(
            f"{description} does not contain requested transcript column "
            f"{text_column!r}. Available columns: "
            + ", ".join(map(str, frame.columns))
        )
    id_candidates = (
        "FileName",
        "filename",
        "file_name",
        "Utterance_ID",
        "utterance_id",
        "id",
    )
    text_candidates = ("Transcript", "transcript", "Text", "text")
    resolved_id = id_column or next(
        (candidate for candidate in id_candidates if candidate in frame.columns),
        None,
    )
    resolved_text = text_column or next(
        (candidate for candidate in text_candidates if candidate in frame.columns),
        None,
    )
    if resolved_id is None or resolved_text is None:
        raise ValueError(
            f"{description} needs ID and transcript columns. Available columns: "
            + ", ".join(map(str, frame.columns))
        )
    mapping = _mapping_from_pairs(
        zip(frame[resolved_id].tolist(), frame[resolved_text].tolist()),
        description=description,
    )
    return mapping, {
        "format": "table",
        "delimiter": "tab" if delimiter == "\t" else "comma",
        "columns": list(map(str, frame.columns)),
        "id_column": resolved_id,
        "text_column": resolved_text,
        "rows": int(len(frame)),
    }


def normalize_msp_file_id(value: object) -> str:
    """Map MSP audio/transcript filenames to a case-insensitive extension-free ID."""

    name = PurePosixPath(str(value).strip().replace("\\", "/")).name
    while True:
        lowered = name.casefold()
        matched_suffix = next(
            (suffix for suffix in (".wav", ".txt") if lowered.endswith(suffix)),
            None,
        )
        if matched_suffix is None:
            break
        name = name[: -len(matched_suffix)]
    normalized = name.strip().casefold()
    if not normalized:
        raise ValueError("MSP transcript/file IDs cannot be blank.")
    return normalized


def _mapping_from_pairs(
    pairs: Iterable[tuple[object, object]],
    *,
    description: str,
) -> dict[str, str]:
    """Build one unique normalized-ID transcript mapping or fail on collisions."""

    mapping: dict[str, str] = {}
    originals: dict[str, str] = {}
    for raw_id, raw_text in pairs:
        key = normalize_msp_file_id(raw_id)
        original = str(raw_id)
        if key in mapping:
            raise ValueError(
                f"{description} contains duplicate normalized ID {key!r}: "
                f"{originals[key]!r} and {original!r}."
            )
        mapping[key] = "" if raw_text is None else str(raw_text)
        originals[key] = original
    if not mapping:
        raise ValueError(f"{description} contains no transcripts.")
    return mapping


def _directory_sha256(directory: Path, files: Sequence[Path]) -> str:
    """Hash stable relative paths and bytes for a transcript directory contract."""

    digest = sha256()
    for path in sorted(files, key=lambda value: value.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with open(path, "rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _load_msp_transcript_directory(directory: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Read recursively keyed UTF-8 TXT files from an authorized local release."""

    files = tuple(path for path in directory.rglob("*") if path.is_file() and path.suffix.casefold() == ".txt")
    if not files:
        raise ValueError(f"MSP transcript directory contains no .txt files: {directory}")
    pairs = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"MSP transcript must use UTF-8 encoding: {path}") from exc
        pairs.append((path.name, text))
    return _mapping_from_pairs(pairs, description="MSP transcript directory"), {
        "format": "directory",
        "path": str(directory),
        "sha256_tree": _directory_sha256(directory, files),
        "text_files": int(len(files)),
    }


def _validate_zip_members(archive: zipfile.ZipFile, archive_path: Path) -> list[zipfile.ZipInfo]:
    """Reject unsafe members while allowing one release-scale transcript table."""

    members = [member for member in archive.infolist() if not member.is_dir()]
    seen: set[str] = set()
    total_size = 0
    for member in members:
        pure = PurePosixPath(member.filename)
        if pure.is_absolute() or ".." in pure.parts or not pure.name:
            raise ValueError(f"Unsafe MSP transcript ZIP member: {member.filename!r}")
        normalized = pure.as_posix().casefold()
        if normalized in seen:
            raise ValueError(f"Duplicate MSP transcript ZIP member: {member.filename!r}")
        if member.flag_bits & 0x1:
            raise ValueError(f"Encrypted MSP transcript ZIP member: {member.filename!r}")
        suffix = pure.suffix.casefold()
        member_limit = (
            MAX_MSP_ZIP_TABLE_MEMBER_BYTES
            if suffix in {".csv", ".tsv"}
            else MAX_MSP_ZIP_TEXT_MEMBER_BYTES
        )
        if member.file_size > member_limit:
            raise ValueError(
                "Implausibly large MSP transcript member "
                f"({member.file_size} bytes; limit {member_limit}): {member.filename!r}"
            )
        total_size += int(member.file_size)
        if total_size > MAX_MSP_ZIP_TOTAL_BYTES:
            raise ValueError(
                "MSP transcript ZIP expands beyond the bounded "
                f"{MAX_MSP_ZIP_TOTAL_BYTES}-byte budget: {archive_path}"
            )
        seen.add(normalized)
    if not members:
        raise ValueError(f"MSP transcript ZIP is empty: {archive_path}")
    return members


def _load_msp_transcript_zip(
    archive_path: Path,
    *,
    id_column: str | None,
    text_column: str | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Read TXT members or one unambiguous CSV/TSV table without extracting a ZIP."""

    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Cannot read MSP transcript ZIP: {archive_path}") from exc
    with archive:
        members = _validate_zip_members(archive, archive_path)
        text_members = [
            member
            for member in members
            if PurePosixPath(member.filename).suffix.casefold() == ".txt"
        ]
        table_members = [
            member
            for member in members
            if PurePosixPath(member.filename).suffix.casefold() in {".csv", ".tsv"}
        ]
        table_error: ValueError | None = None
        if len(table_members) == 1:
            member = table_members[0]
            try:
                mapping, details = _read_transcript_table_bytes(
                    archive.read(member),
                    description=f"MSP transcript ZIP member {member.filename}",
                    id_column=id_column,
                    text_column=text_column,
                )
            except ValueError as exc:
                table_error = exc
                if id_column is not None or text_column is not None:
                    raise
            else:
                return mapping, {
                    **details,
                    "format": "zip-table",
                    "path": str(archive_path),
                    "sha256": sha256_file(archive_path),
                    "member": member.filename,
                }
        if text_members:
            pairs = []
            for member in text_members:
                try:
                    text = archive.read(member).decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"MSP transcript ZIP member must use UTF-8: {member.filename}"
                    ) from exc
                pairs.append((PurePosixPath(member.filename).name, text))
            mapping = _mapping_from_pairs(pairs, description="MSP transcript ZIP")
            details = {
                "format": "zip-text-files",
                "path": str(archive_path),
                "sha256": sha256_file(archive_path),
                "text_files": int(len(text_members)),
            }
            return mapping, details
        if len(table_members) != 1:
            raise ValueError(
                "MSP transcript ZIP without TXT members must contain exactly one "
                f"CSV/TSV table; found {len(table_members)}."
            )
        assert table_error is not None
        raise table_error


def load_msp_transcript_mapping(
    source: str | Path,
    *,
    id_column: str | None = None,
    text_column: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load authorized MSP transcripts from a table, directory, or ZIP contract."""

    path = Path(source).expanduser().resolve()
    if path.is_dir():
        return _load_msp_transcript_directory(path)
    if not path.is_file():
        raise FileNotFoundError(f"MSP transcript source not found: {path}")
    if path.suffix.casefold() == ".zip":
        return _load_msp_transcript_zip(
            path,
            id_column=id_column,
            text_column=text_column,
        )
    if path.suffix.casefold() not in {".csv", ".tsv"}:
        raise ValueError(
            "MSP transcript source must be a CSV/TSV table, directory of TXT files, "
            f"or ZIP archive: {path}"
        )
    mapping, details = _read_transcript_table_bytes(
        path.read_bytes(),
        description=f"MSP transcript table {path}",
        id_column=id_column,
        text_column=text_column,
    )
    return mapping, {**details, "path": str(path), "sha256": sha256_file(path)}


def load_msp_podcast_test(
    labels_file: str | Path,
    transcripts: str | Path,
    *,
    split: str,
    label_id_column: str = "FileName",
    split_column: str = "Split_Set",
    arousal_column: str = "EmoAct",
    valence_column: str = "EmoVal",
    transcript_id_column: str | None = None,
    transcript_text_column: str | None = None,
    strict_official_count: bool = True,
) -> ExternalBenchmarkData:
    """Load licensed MSP-Podcast 2.0 Test1 or Test2 as text-only VA rows."""

    normalized_split = str(split).strip().casefold()
    if normalized_split not in MSP_OFFICIAL_TEST_ROWS:
        raise ValueError(
            "MSP text-only external evaluation permits only test1 or test2. "
            "Train/Development are prohibited and Test3 hides transcripts."
        )
    requested_label_columns = {
        "label_id_column": label_id_column,
        "split_column": split_column,
        "arousal_column": arousal_column,
        "valence_column": valence_column,
    }
    if strict_official_count and requested_label_columns != MSP_CANONICAL_LABEL_COLUMNS:
        mismatches = [
            f"{name}={requested_label_columns[name]!r} "
            f"(required {expected!r})"
            for name, expected in MSP_CANONICAL_LABEL_COLUMNS.items()
            if requested_label_columns[name] != expected
        ]
        raise ValueError(
            "Canonical MSP-Podcast 2.0 evaluation fixes the official label schema: "
            + "; ".join(mismatches)
            + ". Only transcript ID/text columns may be overridden."
        )
    split_value = "Test1" if normalized_split == "test1" else "Test2"
    label_path = Path(labels_file).expanduser().resolve()
    if not label_path.is_file():
        raise FileNotFoundError(f"MSP labels_consensus.csv not found: {label_path}")
    labels = pd.read_csv(
        label_path,
        sep=",",
        quotechar='"',
        keep_default_na=False,
        dtype=str,
    )
    _require_columns(
        labels,
        (
            label_id_column,
            split_column,
            arousal_column,
            valence_column,
        ),
        "MSP labels table",
    )
    selected = labels.loc[labels[split_column] == split_value].copy()
    if selected.empty:
        available = sorted(set(labels[split_column].astype(str)))
        raise ValueError(
            f"MSP labels contain no exact {split_value!r} rows. Available split values: "
            f"{available}."
        )
    if strict_official_count and len(selected) != MSP_OFFICIAL_TEST_ROWS[normalized_split]:
        raise ValueError(
            f"MSP-Podcast 2.0 {split_value} must contain "
            f"{MSP_OFFICIAL_TEST_ROWS[normalized_split]} rows; got {len(selected)}. "
            "Confirm that the licensed release is version 2.0."
        )
    normalized_ids = selected[label_id_column].map(normalize_msp_file_id)
    duplicated = normalized_ids.duplicated(keep=False)
    if bool(duplicated.any()):
        examples = selected.loc[duplicated, label_id_column].head(10).tolist()
        raise ValueError(f"MSP selected label IDs are not unique: {examples}.")

    native_arousal = _numeric_column(selected, arousal_column, f"MSP {split_value}")
    native_valence = _numeric_column(selected, valence_column, f"MSP {split_value}")
    _require_range(
        native_arousal,
        lower=1.0,
        upper=7.0,
        name="arousal",
        description=f"MSP {split_value}",
    )
    _require_range(
        native_valence,
        lower=1.0,
        upper=7.0,
        name="valence",
        description=f"MSP {split_value}",
    )
    transcript_mapping, transcript_manifest = load_msp_transcript_mapping(
        transcripts,
        id_column=transcript_id_column,
        text_column=transcript_text_column,
    )
    missing_ids = [key for key in normalized_ids if key not in transcript_mapping]
    if missing_ids:
        raise ValueError(
            f"MSP {split_value} has {len(missing_ids)} label rows without transcripts; "
            f"examples: {missing_ids[:10]}. No rows were dropped."
        )
    text = pd.Series(
        [transcript_mapping[key] for key in normalized_ids],
        index=selected.index,
        dtype=str,
    ).reset_index(drop=True)
    empty_text = text.str.strip().eq("")
    if bool(empty_text.any()):
        empty_ids = selected.loc[
            empty_text.to_numpy(dtype=bool), label_id_column
        ].astype(str)
        raise ValueError(
            f"MSP {split_value} has {int(empty_text.sum())} matched blank transcripts; "
            f"examples: {empty_ids.head(10).tolist()}. Refusing a potentially wrong "
            "transcript schema instead of silently evaluating blank model inputs."
        )
    original_ids = selected[label_id_column].astype(str).reset_index(drop=True)
    group_ids = [
        re.sub(r"_[^_]+(?:\.wav)?$", "", identifier, flags=re.IGNORECASE)
        for identifier in original_ids
    ]
    frame = pd.DataFrame(
        {
            "index": original_ids,
            "benchmark_id": original_ids,
            "text": text,
            "text_sha256": [_text_sha256(value) for value in text],
            "is_empty_text": text.str.strip().eq("").to_numpy(dtype=bool),
            "dataset_of_origin": "MSP-Podcast-2.0",
            "split": normalized_split,
            "file_name": original_ids,
            "podcast_group": group_ids,
            "valence": (native_valence - 1.0) / 6.0,
            "arousal": (native_arousal - 1.0) / 6.0,
            "native_valence": native_valence,
            "native_arousal": native_arousal,
        }
    )
    selected_id_set = set(normalized_ids)
    return ExternalBenchmarkData(
        name=MSP_NAME,
        version="2.0" if strict_official_count else "unverified-local",
        split=normalized_split,
        frame=frame,
        native_scale={"valence": (1.0, 7.0), "arousal": (1.0, 7.0)},
        model_to_native_scale=(6.0, 6.0),
        model_to_native_offset=(1.0, 1.0),
        source_manifest={
            "canonical_contract_verified": bool(strict_official_count),
            "version_validation": (
                "canonical label schema and published Test split row count; local "
                "input SHA256 recorded because no release-wide digest is published"
                if strict_official_count
                else "disabled for local fixture/noncanonical use"
            ),
            "labels": {
                "path": str(label_path),
                "sha256": sha256_file(label_path),
                "columns": list(map(str, labels.columns)),
                "label_id_column": label_id_column,
                "split_column": split_column,
                "arousal_column": arousal_column,
                "valence_column": valence_column,
            },
            "transcripts": transcript_manifest,
            "license": "MSP-Podcast Academic License; local authorized copy required",
            "source_url": "https://www.lab-msp.com/MSP/MSP-Podcast.html",
        },
        join_report={
            "selected_label_rows": int(len(selected)),
            "available_transcript_rows": int(len(transcript_mapping)),
            "matched_label_rows": int(len(frame)),
            "missing_label_transcripts": 0,
            "unused_transcript_rows": int(len(set(transcript_mapping) - selected_id_set)),
            "empty_matched_transcripts": int(frame["is_empty_text"].sum()),
            "join_key": label_id_column,
            "normalized_join_key": "casefolded basename without .wav/.txt",
            "join_validation": "many-source-to-selected one_to_one; label order preserved",
        },
        official_group_column=None,
        context_policy="one current utterance transcript only",
        prediction_metadata_columns=("file_name",),
    )


def _read_persisted_prediction_table(
    path: Path,
    *,
    description: str,
) -> pd.DataFrame:
    """Read and validate one canonical internal held-out prediction table."""

    try:
        frame = pd.read_csv(
            path,
            sep="\t",
            keep_default_na=False,
            dtype=str,
        )
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read {description}: {path}") from exc
    if tuple(frame.columns) != PERSISTED_PREDICTION_COLUMNS:
        raise ValueError(
            f"{description} must contain the canonical internal prediction columns."
        )
    if frame.empty:
        raise ValueError(f"{description} contains no prediction rows.")
    if not frame["index"].str.fullmatch(r"[0-9]+").all():
        raise ValueError(f"{description} contains a non-numeric item index.")
    for column in ("valence", "arousal", "pred_valence", "pred_arousal"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError(
                f"{description} contains non-finite or out-of-range {column} values."
            )
    return frame


def _persisted_prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Recompute one internal prediction table with production metric semantics."""

    labels = frame.loc[:, ["valence", "arousal"]].to_numpy(dtype=np.float64)
    predictions = frame.loc[:, ["pred_valence", "pred_arousal"]].to_numpy(
        dtype=np.float64
    )
    return calculate_va_metrics(labels, predictions)


def _require_recorded_metrics(
    recorded: Mapping[str, Any],
    computed: Mapping[str, float],
    *,
    prefix: str,
    description: str,
) -> None:
    """Require saved metrics to be a faithful recomputation of saved predictions."""

    for name in PERSISTED_METRIC_NAMES:
        key = f"{prefix}{name}"
        if key not in recorded:
            raise ValueError(f"{description} is missing {key}.")
        expected = float(computed[name])
        actual = recorded[key]
        if not math.isfinite(expected):
            if actual is not None:
                raise ValueError(
                    f"{description}.{key} must be null because it is undefined."
                )
            continue
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{description}.{key} must be a finite JSON number.")
        numeric = float(actual)
        if not math.isfinite(numeric) or not math.isclose(
            numeric, expected, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"{description}.{key} does not match prediction recomputation."
            )


def _saved_redistribution_contract(
    metadata: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Read a schema-aware redistribution identity without guessing new metadata."""

    version = parameters["architecture_manifest_version"]
    if "gaze_redistribution" not in metadata and version != PRE_REDISTRIBUTION_MANIFEST_VERSION:
        raise ValueError(f"{label} is missing gaze_redistribution.")
    contract = validate_redistribution_contract(
        metadata.get("gaze_redistribution", {"method": "none"}),
        gaze_fusion=parameters["gaze_fusion"],
        feature_indices=parameters["gaze_feature_indices"],
    )
    if version == PRE_REDISTRIBUTION_MANIFEST_VERSION and contract["method"] != "none":
        raise ValueError(f"{label}: schema 6 cannot enable gaze_redistribution.")
    return contract


def _validate_architecture_provenance(
    architecture: Mapping[str, Any],
    parameters: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    held_out_fold: int,
) -> None:
    """Cross-check every reload-critical architecture field against run provenance."""

    expected_top_level = {
        "schema_version": parameters["architecture_manifest_version"],
        "decoder_model_id": parameters["model_id"],
        "decoder_commit": parameters["model_revision"],
        "finetuning_mode": parameters["finetuning_mode"],
        "gaze_fusion": parameters["gaze_fusion"],
        "gaze_features": parameters["gaze_features"],
        "gaze_feature_indices": parameters["gaze_feature_indices"],
        "features_used": parameters["features_used"],
        "gaze_concat_order": parameters["gaze_concat_order"],
        "pooling_position": parameters["pooling_position"],
        "output_activation": parameters["output_activation"],
        "output_names": list(MODEL_OUTPUT_NAMES),
    }
    for field, expected in expected_top_level.items():
        if architecture.get(field) != expected:
            raise ValueError(
                f"Held-out fold {held_out_fold} architecture disagrees with "
                f"training provenance for {field}."
            )

    reconstruction = architecture.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise ValueError(
            f"Held-out fold {held_out_fold} architecture reconstruction must be an object."
        )
    expected_redistribution = _saved_redistribution_contract(
        parameters, parameters, label="training_parameters.json"
    )
    for source, label in (
        (architecture, "architecture"),
        (reconstruction, "architecture reconstruction"),
    ):
        if _saved_redistribution_contract(
            source, parameters, label=f"Held-out fold {held_out_fold} {label}"
        ) != expected_redistribution:
            raise ValueError(
                f"Held-out fold {held_out_fold} {label} disagrees with "
                "training provenance for gaze_redistribution."
            )
    uses_lora = parameters["finetuning_mode"] == "lora"
    expected_reconstruction = {
        "decoder_model_id": parameters["model_id"],
        "decoder_revision": parameters["model_revision"],
        "finetuning_mode": parameters["finetuning_mode"],
        "gaze_fusion": parameters["gaze_fusion"],
        "et_repo_id": parameters["et_model_id"],
        "et_revision": parameters["et_revision"],
        "et_filename": parameters["et_filename"],
        "et_feature_names": parameters["gaze_features"],
        "et_feature_indices": parameters["gaze_feature_indices"],
        "features_used": parameters["features_used"],
        "et_cache_size": parameters["et_cache_size"],
        "output_dim": parameters["output_dim"],
        "output_activation": parameters["output_activation"],
        "lora_rank": parameters["lora_rank"],
        "lora_alpha": parameters["lora_alpha"],
        "lora_dropout": parameters["lora_dropout"],
        "lora_target_modules": "all-linear" if uses_lora else None,
        "lora_task_type": "FEATURE_EXTRACTION" if uses_lora else None,
        "gaze_projection_dim": DEFAULT_GAZE_PROJECTION_DIM,
        "gaze_projection_dropout": list(DEFAULT_GAZE_PROJECTION_DROPOUT),
        "classifier_dropout": DEFAULT_CLASSIFIER_DROPOUT,
        "attn_implementation": parameters["attn_implementation"],
        "backbone_dtype_at_construction": parameters["dtype"],
    }
    for field, expected in expected_reconstruction.items():
        if reconstruction.get(field) != expected:
            raise ValueError(
                f"Held-out fold {held_out_fold} architecture reconstruction "
                f"disagrees with training provenance for {field}."
            )

    expected_et_model = None
    if parameters["gaze_fusion"] == "prefix-concat":
        expected_et_model = {
            "repo_id": parameters["et_model_id"],
            "revision": parameters["et_revision"],
            "filename": parameters["et_filename"],
            "feature_names": parameters["gaze_features"],
            "feature_indices": parameters["gaze_feature_indices"],
            "features_used": parameters["features_used"],
        }
    if architecture.get("et_model") != expected_et_model:
        raise ValueError(
            f"Held-out fold {held_out_fold} architecture has incompatible ET2 provenance."
        )
    expected_state = {
        "filename": SAFE_WEIGHTS_FILENAME,
        "format": "safetensors",
        "scope": "complete DecoderVARegressor state_dict; ET2 remains external and frozen",
        "strict_loading": True,
    }
    if architecture.get("state_dict") != expected_state:
        raise ValueError(
            f"Held-out fold {held_out_fold} architecture has an invalid state_dict contract."
        )
    for field in ("total_parameters", "trainable_parameters", "trainable_fraction"):
        if (
            field not in architecture
            or field not in manifest
            or architecture[field] != manifest[field]
        ):
            raise ValueError(
                f"Held-out fold {held_out_fold} architecture disagrees with its run "
                f"manifest for {field}."
            )


def _tokenizer_artifact_hashes(model_dir: Path) -> dict[str, str]:
    """Inventory and hash the complete local tokenizer payload used for reload."""

    config_path = model_dir / TOKENIZER_CONFIG_FILENAME
    if not config_path.is_file() or config_path.is_symlink():
        raise FileNotFoundError(
            f"Saved final_model lacks a regular {TOKENIZER_CONFIG_FILENAME}: {model_dir}"
        )
    _read_json_object(config_path, "saved tokenizer configuration")
    ignored = {
        ARCHITECTURE_MANIFEST_FILENAME,
        SAFE_WEIGHTS_FILENAME,
        "training_args.bin",
    }
    artifacts: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Saved final_model contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        if relative in ignored:
            continue
        artifacts[f"tokenizer/{relative}"] = sha256_file(path)
    payload_names = {Path(name).name for name in artifacts}
    if not payload_names.intersection(TOKENIZER_PAYLOAD_FILENAMES):
        raise FileNotFoundError(
            "Saved final_model has tokenizer_config.json but no tokenizer vocabulary/model "
            f"payload: {model_dir}"
        )
    return artifacts


def _validate_root_run_parameters(root: Path, parameters: Mapping[str, Any]) -> None:
    """Validate the root identity and fields needed for safe model reconstruction."""

    required = {
        *RUN_CONTRACT_FIELDS,
        "held_out_folds",
        "dataset_counts_after_filter",
        "seed",
        "run_name",
        "effective_output_dir",
        "et_cache_size",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "attn_implementation",
    }
    missing = sorted(required.difference(parameters))
    if missing:
        raise ValueError(
            "training_parameters.json is missing reconstruction/provenance field(s): "
            + ", ".join(missing)
        )
    held_out_folds = parameters["held_out_folds"]
    if (
        not isinstance(held_out_folds, list)
        or len(held_out_folds) != 2
        or any(type(value) is not int for value in held_out_folds)
        or set(held_out_folds) != {1, 2}
    ):
        raise ValueError(
            "External evaluation requires the complete held-out folds 1 and 2."
        )
    if type(parameters["architecture_manifest_version"]) is not int or parameters[
        "architecture_manifest_version"
    ] not in {PRE_REDISTRIBUTION_MANIFEST_VERSION, ARCHITECTURE_MANIFEST_VERSION}:
        raise ValueError(
            "training_parameters.json uses an unsupported architecture manifest version."
        )
    if parameters["output_dim"] != 2 or parameters["output_activation"] != OUTPUT_ACTIVATION:
        raise ValueError("External evaluation requires a hard-sigmoid two-output VA run.")
    if parameters["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported saved dtype: {parameters['dtype']!r}.")
    if parameters["finetuning_mode"] not in {"full", "lora"}:
        raise ValueError("Saved finetuning_mode must be full or lora.")
    if parameters["gaze_fusion"] not in {"none", "prefix-concat"}:
        raise ValueError("Saved gaze_fusion must be none or prefix-concat.")
    _saved_redistribution_contract(
        parameters, parameters, label="training_parameters.json"
    )
    if isinstance(parameters["seed"], bool) or not isinstance(parameters["seed"], int):
        raise ValueError("training_parameters.json seed must be an integer.")
    for field in ("max_length", "eval_batch_size"):
        if type(parameters[field]) is not int or parameters[field] <= 0:
            raise ValueError(f"Saved {field} must be a positive integer.")
    counts = parameters["dataset_counts_after_filter"]
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("dataset_counts_after_filter must be a non-empty object.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise ValueError("dataset_counts_after_filter values must be positive integers.")
    run_name = parameters["run_name"]
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("training_parameters.json run_name must be a non-empty string.")
    effective_name = Path(str(parameters["effective_output_dir"])).name
    if effective_name != run_name:
        raise ValueError(
            "training_parameters.json run_name disagrees with effective_output_dir."
        )


def _validate_root_internal_evidence(
    root: Path,
    training_parameters: Mapping[str, Any],
    fold_prediction_frames: Sequence[pd.DataFrame],
) -> None:
    """Validate OOF and per-dataset reports against both persisted fold predictions."""

    oof_path = root / "oof_predictions.tsv"
    oof_frame = _read_persisted_prediction_table(
        oof_path,
        description="out-of-fold predictions",
    )
    fold_records = Counter(
        tuple(row)
        for frame in fold_prediction_frames
        for row in frame.loc[:, list(PERSISTED_PREDICTION_COLUMNS)].itertuples(
            index=False, name=None
        )
    )
    oof_records = Counter(
        tuple(row)
        for row in oof_frame.loc[:, list(PERSISTED_PREDICTION_COLUMNS)].itertuples(
            index=False, name=None
        )
    )
    if oof_records != fold_records:
        raise ValueError(
            "OOF predictions are not the exact union of both held-out fold predictions."
        )
    oof_metrics = _read_json_object(root / "oof_metrics.json", "OOF metrics")
    _require_recorded_metrics(
        oof_metrics,
        _persisted_prediction_metrics(oof_frame),
        prefix="",
        description="oof_metrics.json",
    )
    observed_counts = {
        str(name): int(count)
        for name, count in oof_frame["dataset_of_origin"].value_counts().items()
    }
    if observed_counts != dict(training_parameters["dataset_counts_after_filter"]):
        raise ValueError(
            "dataset_counts_after_filter does not match the completed OOF predictions."
        )

    try:
        dataset_metrics = pd.read_csv(
            root / "metrics_by_dataset.tsv",
            sep="\t",
            keep_default_na=False,
            dtype=str,
        )
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("Cannot read metrics_by_dataset.tsv.") from exc
    expected_dataset_columns = ("dataset_of_origin", *PERSISTED_METRIC_NAMES)
    if tuple(dataset_metrics.columns) != expected_dataset_columns or dataset_metrics.empty:
        raise ValueError(
            "metrics_by_dataset.tsv does not contain the canonical completed metric table."
        )
    if dataset_metrics["dataset_of_origin"].duplicated().any():
        raise ValueError("metrics_by_dataset.tsv contains duplicate dataset rows.")
    if set(dataset_metrics["dataset_of_origin"]) != set(observed_counts):
        raise ValueError("metrics_by_dataset.tsv sources do not match OOF predictions.")
    for _, row in dataset_metrics.iterrows():
        source = row["dataset_of_origin"]
        source_predictions = oof_frame.loc[oof_frame["dataset_of_origin"] == source]
        recorded = {
            name: None if row[name] == "" else float(row[name])
            for name in PERSISTED_METRIC_NAMES
        }
        _require_recorded_metrics(
            recorded,
            _persisted_prediction_metrics(source_predictions),
            prefix="",
            description=f"metrics_by_dataset.tsv[{source}]",
        )


def discover_completed_run(
    run_dir: str | Path,
    *,
    require_internal_evidence: bool = True,
) -> tuple[SavedRunMember, SavedRunMember]:
    """Validate two final models and optionally the raw-text internal result evidence."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Completed training run directory not found: {root}")
    root_filenames = ("training_parameters.json",)
    if require_internal_evidence:
        root_filenames += (
            "oof_metrics.json",
            "oof_predictions.tsv",
            "metrics_by_dataset.tsv",
        )
    for filename in root_filenames:
        _require_regular_artifact(
            root / filename,
            "External evaluation completed-run artifact",
        )
    training_parameters = _read_json_object(
        root / "training_parameters.json", "training parameters"
    )
    _validate_root_run_parameters(root, training_parameters)
    root_hashes = {
        filename: sha256_file(root / filename) for filename in root_filenames
    }

    members: list[SavedRunMember] = []
    fold_prediction_frames: list[pd.DataFrame] = []
    seen_indices: set[str] = set()
    total_rows = sum(training_parameters["dataset_counts_after_filter"].values())
    for held_out_fold in (1, 2):
        fold_root = root / f"heldout_fold{held_out_fold}"
        fold_filenames = ("run_manifest.json",)
        if require_internal_evidence:
            fold_filenames += ("metrics.json", "predictions.tsv")
        for filename in fold_filenames:
            _require_regular_artifact(
                fold_root / filename,
                "External evaluation completed-fold artifact",
            )
        model_dir = fold_root / "final_model"
        run_manifest_path = fold_root / "run_manifest.json"
        architecture_path = model_dir / ARCHITECTURE_MANIFEST_FILENAME
        weights_path = model_dir / SAFE_WEIGHTS_FILENAME
        run_manifest = _read_json_object(run_manifest_path, "fold run manifest")
        if _saved_redistribution_contract(
            run_manifest, training_parameters, label=str(run_manifest_path)
        ) != _saved_redistribution_contract(
            training_parameters, training_parameters, label="training_parameters.json"
        ):
            raise ValueError(
                f"Fold manifest disagrees with training_parameters.json for "
                f"gaze_redistribution: {run_manifest_path}"
            )
        _require_regular_artifact(architecture_path, "architecture manifest")
        architecture = _read_json_object(architecture_path, "architecture manifest")
        if not weights_path.is_file():
            raise FileNotFoundError(
                "Results-only archives omit model weights. Evaluate on the server run "
                f"directory containing: {weights_path}"
            )
        if weights_path.is_symlink():
            raise ValueError(f"Saved model weights must not be a symlink: {weights_path}")
        expected_training_fold = 2 if held_out_fold == 1 else 1
        for field, expected in training_parameters.items():
            if field not in run_manifest:
                raise ValueError(
                    f"Fold manifest is missing root training parameter {field!r}: "
                    f"{run_manifest_path}"
                )
            if run_manifest[field] != expected:
                raise ValueError(
                    f"Fold manifest disagrees with training_parameters.json for "
                    f"{field!r}: {run_manifest_path}"
                )
        if type(run_manifest.get("held_out_fold")) is not int or run_manifest.get(
            "held_out_fold"
        ) != held_out_fold:
            raise ValueError(
                f"Fold manifest held_out_fold mismatch in {run_manifest_path}."
            )
        if type(run_manifest.get("training_fold")) is not int or run_manifest.get(
            "training_fold"
        ) != expected_training_fold:
            raise ValueError(
                f"Fold manifest training_fold mismatch in {run_manifest_path}."
            )
        expected_fold_seed = int(training_parameters["seed"]) + held_out_fold - 1
        if run_manifest.get("fold_seed") != expected_fold_seed:
            raise ValueError(
                f"Fold manifest fold_seed mismatch in {run_manifest_path}."
            )
        for field in ("training_rows", "evaluation_rows"):
            if type(run_manifest.get(field)) is not int or run_manifest[field] <= 0:
                raise ValueError(
                    f"Fold manifest {field} must be a positive integer: {run_manifest_path}"
                )
        if run_manifest["training_rows"] != total_rows - run_manifest["evaluation_rows"]:
            raise ValueError(
                f"Fold manifest row counts do not reconstruct the complete corpus: "
                f"{run_manifest_path}"
            )

        if require_internal_evidence:
            prediction_path = fold_root / "predictions.tsv"
            prediction_frame = _read_persisted_prediction_table(
                prediction_path,
                description=f"held-out fold {held_out_fold} predictions",
            )
            if len(prediction_frame) != run_manifest["evaluation_rows"]:
                raise ValueError(
                    f"Held-out fold {held_out_fold} predictions do not match "
                    "evaluation_rows."
                )
            if set(prediction_frame["held_out_fold"]) != {str(held_out_fold)}:
                raise ValueError(
                    f"Held-out fold {held_out_fold} predictions record a different fold."
                )
            duplicate_indices = seen_indices.intersection(prediction_frame["index"])
            if duplicate_indices:
                raise ValueError(
                    "Internal prediction indices are duplicated across held-out folds: "
                    + ", ".join(sorted(duplicate_indices)[:10])
                )
            seen_indices.update(prediction_frame["index"])
            fold_prediction_frames.append(prediction_frame)
            fold_metrics = _read_json_object(fold_root / "metrics.json", "fold metrics")
            _require_recorded_metrics(
                fold_metrics,
                _persisted_prediction_metrics(prediction_frame),
                prefix="test_",
                description=f"heldout_fold{held_out_fold}/metrics.json",
            )

        _validate_architecture_provenance(
            architecture,
            training_parameters,
            run_manifest,
            held_out_fold=held_out_fold,
        )
        tokenizer_hashes = _tokenizer_artifact_hashes(model_dir)
        members.append(
            SavedRunMember(
                name=f"heldout_fold{held_out_fold}",
                held_out_fold=held_out_fold,
                training_fold=expected_training_fold,
                model_dir=model_dir,
                run_manifest_path=run_manifest_path,
                run_manifest=run_manifest,
                architecture_manifest_path=architecture_path,
                architecture_manifest=architecture,
                weights_path=weights_path,
                file_sha256={
                    **root_hashes,
                    "run_manifest": sha256_file(run_manifest_path),
                    "architecture_manifest": sha256_file(architecture_path),
                    "model_weights": sha256_file(weights_path),
                    **tokenizer_hashes,
                },
            )
        )

    if require_internal_evidence:
        _validate_root_internal_evidence(
            root,
            training_parameters,
            fold_prediction_frames,
        )

    first, second = members
    mismatches = []
    for field in RUN_CONTRACT_FIELDS:
        if first.run_manifest.get(field) != second.run_manifest.get(field):
            mismatches.append(
                f"{field}: {first.run_manifest.get(field)!r} != "
                f"{second.run_manifest.get(field)!r}"
            )
    first_reconstruction = first.architecture_manifest.get("reconstruction")
    second_reconstruction = second.architecture_manifest.get("reconstruction")
    if first_reconstruction != second_reconstruction:
        mismatches.append("architecture reconstruction contracts differ")
    first_tokenizer = {
        key: value
        for key, value in first.file_sha256.items()
        if key.startswith("tokenizer/")
    }
    second_tokenizer = {
        key: value
        for key, value in second.file_sha256.items()
        if key.startswith("tokenizer/")
    }
    if first_tokenizer != second_tokenizer:
        mismatches.append("saved tokenizer artifact inventories or contents differ")
    if mismatches:
        raise ValueError(
            "The two fold members are not one matched training condition:\n- "
            + "\n- ".join(mismatches)
        )
    return first, second


def reject_benchmark_training_sources(
    members: Sequence[SavedRunMember], benchmark_name: str
) -> None:
    """Reject manifests that explicitly name the requested external corpus as training data."""

    aliases = {
        OMG_NAME: ("omg", "omgemotion", "oneminutegradualemotion"),
        MSP_NAME: ("msppodcast", "msp podcast"),
        IDEST_NAME: (
            "idest",
            "internationaldatabaseofemotionalshorttexts",
            "international database of emotional short texts",
        ),
        SEMEVAL_NAME: (
            "semeval",
            "semeval2026",
            "semeval2026task2",
            "emotionvalaroutimevariation",
            "ecological essays",
        ),
    }[benchmark_name]
    matched = []
    for member in members:
        counts = member.run_manifest.get("dataset_counts_after_filter", {})
        if not isinstance(counts, Mapping):
            raise ValueError("Run manifest dataset_counts_after_filter must be an object.")
        for source in counts:
            compact = re.sub(r"[^a-z0-9]+", "", str(source).casefold())
            spaced = str(source).casefold()
            if any(alias.replace(" ", "") in compact or alias in spaced for alias in aliases):
                matched.append(str(source))
    if matched:
        raise ValueError(
            f"Requested benchmark {benchmark_name} appears in fine-tuning sources: "
            + ", ".join(sorted(set(matched)))
        )


def audit_finetuning_text_overlap(
    benchmark: ExternalBenchmarkData,
    members: Sequence[SavedRunMember],
    training_data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[int, pd.DataFrame]]:
    """Verify fold hashes and flag exact normalized external/fine-tuning text overlap."""

    directory = Path(training_data_dir).expanduser().resolve()
    expected_hashes = members[0].run_manifest.get("fold_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ValueError("Run manifest is missing fold_sha256 provenance.")
    fold_frames: dict[int, pd.DataFrame] = {}
    actual_hashes: dict[str, str] = {}
    for fold, filename in enumerate(FOLD_FILENAMES, start=1):
        path = directory / filename
        actual = sha256_file(path)
        expected = expected_hashes.get(filename)
        if actual != expected:
            raise ValueError(
                f"Training fold SHA256 mismatch for {path}: {actual}; expected {expected}."
            )
        actual_hashes[filename] = actual
        fold_frames[fold] = read_fold(path)

    audited = benchmark.frame.copy()
    external_normalized = audited["text"].map(normalize_overlap_text)
    member_reports: dict[str, Any] = {}
    union_training_texts: set[str] = set()
    for member in members:
        frame = fold_frames[member.training_fold]
        excluded = set(map(str, member.run_manifest.get("excluded_dataset_names", ())))
        filtered = frame.loc[~frame["dataset_of_origin"].astype(str).isin(excluded)].copy()
        expected_rows = int(member.run_manifest.get("training_rows", -1))
        if len(filtered) != expected_rows:
            raise ValueError(
                f"Training-row provenance mismatch for {member.name}: "
                f"manifest {expected_rows}, reconstructed {len(filtered)}."
            )
        normalized_training = {
            value
            for value in filtered["text"].map(normalize_overlap_text)
            if value
        }
        union_training_texts.update(normalized_training)
        column = f"overlap_with_{member.name}_training"
        audited[column] = [
            bool(value and value in normalized_training) for value in external_normalized
        ]
        member_reports[member.name] = {
            "training_fold": int(member.training_fold),
            "training_rows": int(len(filtered)),
            "unique_nonempty_normalized_training_texts": int(len(normalized_training)),
            "overlapping_external_rows": int(audited[column].sum()),
        }
    audited["overlap_with_any_finetuning_text"] = [
        bool(value and value in union_training_texts) for value in external_normalized
    ]
    report = {
        "status": "completed",
        "normalization": "Unicode NFKC + casefold + whitespace collapse; empty text excluded",
        "training_data_directory": str(directory),
        "verified_fold_sha256": actual_hashes,
        "members": member_reports,
        "external_rows": int(len(audited)),
        "overlapping_external_rows_union": int(
            audited["overlap_with_any_finetuning_text"].sum()
        ),
        "novel_external_rows_union": int(
            (~audited["overlap_with_any_finetuning_text"]).sum()
        ),
        "base_pretraining_contamination": "not auditable from this repository",
        "frozen_et2_training_contamination": (
            "not auditable from this repository; consult the pinned ET2 model card"
        ),
    }
    return audited, report, fold_frames


def _finite_metric_mean(values: Sequence[float]) -> float:
    """Average defined group metrics without converting undefined CCC to zero."""

    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def macro_group_metrics(
    labels_native: np.ndarray,
    predictions_native: np.ndarray,
    groups: Sequence[object],
) -> dict[str, Any]:
    """Calculate a clearly secondary macro average over independent groups."""

    labels = np.asarray(labels_native, dtype=np.float64)
    predictions = np.asarray(predictions_native, dtype=np.float64)
    group_series = pd.Series(list(groups), dtype=str)
    if len(group_series) != len(labels):
        raise ValueError("Group IDs must align with labels and predictions.")
    group_metrics = []
    for _, indices in group_series.groupby(group_series, sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        group_metrics.append(calculate_va_metrics(labels[positions], predictions[positions]))
    result: dict[str, Any] = {
        "n_groups": int(len(group_metrics)),
        "definition": "unweighted macro mean of per-group metrics; secondary/non-official",
    }
    for name in (
        "mse_valence",
        "mse_arousal",
        "mae_valence",
        "mae_arousal",
        "pearson_corr_valence",
        "pearson_corr_arousal",
        "ccc_valence",
        "ccc_arousal",
    ):
        values = [metrics[name] for metrics in group_metrics]
        result[name] = _finite_metric_mean(values)
        result[f"n_defined_{name}"] = int(sum(np.isfinite(value) for value in values))
    result["ccc_mean"] = _finite_metric_mean(
        [result["ccc_valence"], result["ccc_arousal"]]
    )
    return result


def _semeval_subtask1_dimension_metrics(
    labels: Sequence[float],
    predictions: Sequence[float],
    user_ids: Sequence[object],
) -> dict[str, Any]:
    """Reproduce the official Subtask 1 scorer for one affect dimension."""

    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    estimate = np.asarray(predictions, dtype=np.float64).reshape(-1)
    users = np.asarray(list(user_ids), dtype=str).reshape(-1)
    if target.size == 0:
        raise ValueError("SemEval Subtask 1 metrics require at least one row.")
    if target.shape != estimate.shape or target.shape != users.shape:
        raise ValueError("SemEval labels, predictions, and user IDs must align.")
    if not np.isfinite(target).all() or not np.isfinite(estimate).all():
        raise ValueError("SemEval labels and predictions must be finite.")
    if bool(pd.Series(users).str.strip().eq("").any()):
        raise ValueError("SemEval user IDs must be nonblank.")

    unique_users = np.unique(users)
    within_correlations: list[float] = []
    within_p_values: list[float] = []
    within_mae: list[float] = []
    user_prediction_means: list[float] = []
    user_label_means: list[float] = []
    skipped_short = 0
    skipped_constant_gold = 0
    constant_prediction_users = 0
    for user in unique_users:
        mask = users == user
        user_target = target[mask]
        user_estimate = estimate[mask]
        within_mae.append(float(np.mean(np.abs(user_estimate - user_target))))
        user_prediction_means.append(float(np.mean(user_estimate)))
        user_label_means.append(float(np.mean(user_target)))
        if int(mask.sum()) < 2:
            skipped_short += 1
            continue
        if float(np.ptp(user_target)) == 0.0:
            skipped_constant_gold += 1
            continue
        if float(np.ptp(user_estimate)) == 0.0:
            constant_prediction_users += 1
            within_correlations.append(0.0)
            within_p_values.append(1e-10)
            continue
        correlation = stats.pearsonr(user_estimate, user_target)
        within_correlations.append(float(correlation.statistic))
        within_p_values.append(float(correlation.pvalue))

    r_within = (
        float(np.mean(within_correlations))
        if within_correlations
        else float("nan")
    )
    p_within = (
        float(
            len(within_p_values)
            / sum(1.0 / max(value, 1e-10) for value in within_p_values)
        )
        if within_p_values
        else None
    )

    between_predictions = np.asarray(user_prediction_means, dtype=np.float64)
    between_labels = np.asarray(user_label_means, dtype=np.float64)
    if (
        len(unique_users) < 2
        or float(np.ptp(estimate)) == 0.0
        or float(np.ptp(between_predictions)) == 0.0
        or float(np.ptp(between_labels)) == 0.0
    ):
        r_between = float("nan")
        p_between = None
    else:
        between = stats.pearsonr(between_predictions, between_labels)
        r_between = float(between.statistic)
        p_between = float(between.pvalue)

    mae_within = float(np.mean(within_mae))
    mae_between = float(np.mean(np.abs(between_predictions - between_labels)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_composite = float(
            np.tanh(0.5 * (np.arctanh(r_within) + np.arctanh(r_between)))
        )
        official_mae_composite = float(
            np.tanh(0.5 * (np.arctanh(mae_within) + np.arctanh(mae_between)))
        )
    return {
        "r_within": r_within,
        "p_within": p_within,
        "r_between": r_between,
        "p_between": p_between,
        "r_composite": r_composite,
        "mae_within": mae_within,
        "mae_between": mae_between,
        "mae_composite_official_implementation": official_mae_composite,
        "n_users": int(len(unique_users)),
        "n_users_defined_within_correlation": int(len(within_correlations)),
        "n_users_skipped_fewer_than_two_rows": int(skipped_short),
        "n_users_skipped_constant_gold": int(skipped_constant_gold),
        "n_users_constant_prediction_scored_zero": int(constant_prediction_users),
    }


def semeval_subtask1_official_metrics(
    labels_native: np.ndarray,
    predictions_native: np.ndarray,
    user_ids: Sequence[object],
) -> dict[str, Any]:
    """Calculate the released SemEval Subtask 1 V/A ranking score exactly."""

    labels = np.asarray(labels_native, dtype=np.float64)
    predictions = np.asarray(predictions_native, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(f"SemEval labels must have shape [examples, 2], got {labels.shape}.")
    if predictions.shape != labels.shape:
        raise ValueError(
            "SemEval predictions must have the same [examples, 2] shape as labels."
        )
    if labels.shape[0] != len(user_ids):
        raise ValueError("SemEval user IDs must align with labels and predictions.")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("SemEval labels and predictions must contain finite values.")

    dimensions = {
        name: _semeval_subtask1_dimension_metrics(
            labels[:, index], predictions[:, index], user_ids
        )
        for index, name in enumerate(MODEL_OUTPUT_NAMES)
    }
    r_values = [dimensions[name]["r_composite"] for name in MODEL_OUTPUT_NAMES]
    mae_values = [
        dimensions[name]["mae_composite_official_implementation"]
        for name in MODEL_OUTPUT_NAMES
    ]
    return {
        "definition": (
            "released SemEval-2026 Task 2 Subtask 1 scorer: mean of valence and "
            "arousal r_composite; each r_composite is Fisher-z mean of within-user "
            "and between-user Pearson r"
        ),
        "dimensions": dimensions,
        "r_composite_mean_va": float(np.mean(r_values)),
        "mae_composite_mean_va_official_implementation": float(np.mean(mae_values)),
        "official_mae_warning": (
            "the released scorer applies Fisher atanh/tanh to MAE; ordinary flat "
            "native-scale MAE is reported separately under subsets"
        ),
    }


def _aligned_audited_frame(
    benchmark: ExternalBenchmarkData,
    audited_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """Require audit annotations to preserve the exact benchmark identity and order."""

    frame = benchmark.frame if audited_frame is None else audited_frame
    required = ("benchmark_id", "split", "text_sha256", "is_empty_text")
    _require_columns(benchmark.frame, required, "canonical benchmark frame")
    _require_columns(frame, required, "audited benchmark frame")
    if len(frame) != len(benchmark.frame):
        raise ValueError("Audited benchmark frame changed the official row universe.")
    canonical_ids = benchmark.frame["benchmark_id"]
    audited_ids = frame["benchmark_id"]
    if canonical_ids.isna().any() or canonical_ids.duplicated().any():
        raise ValueError("Canonical benchmark_id values must be complete and unique.")
    if audited_ids.isna().any() or audited_ids.duplicated().any():
        raise ValueError("Audited benchmark_id values must be complete and unique.")
    if audited_ids.tolist() != canonical_ids.tolist():
        raise ValueError(
            "Audited benchmark_id values/order do not align with the inference rows."
        )
    for column in ("split", "text_sha256", "is_empty_text"):
        if frame[column].tolist() != benchmark.frame[column].tolist():
            raise ValueError(
                f"Audited benchmark frame changed canonical {column} values/order."
            )
    for column in benchmark.prediction_metadata_columns:
        _require_columns(benchmark.frame, (column,), "canonical benchmark frame")
        _require_columns(frame, (column,), "audited benchmark frame")
        canonical = benchmark.frame[column].reset_index(drop=True)
        audited = frame[column].reset_index(drop=True)
        if not audited.equals(canonical):
            raise ValueError(
                f"Audited benchmark frame changed canonical {column} values/order."
            )
    return frame


def calculate_external_metrics(
    benchmark: ExternalBenchmarkData,
    predictions_model_scale,
    *,
    audited_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Report official-flat native metrics plus prespecified diagnostic subsets."""

    predictions_model = np.asarray(predictions_model_scale, dtype=np.float64)
    predictions_native = benchmark.predictions_to_native(predictions_model)
    if len(predictions_model) != len(benchmark.frame):
        raise ValueError("External predictions do not align with benchmark rows.")
    frame = _aligned_audited_frame(benchmark, audited_frame)
    labels_native = benchmark.labels_native_scale()
    labels_model = benchmark.labels_model_scale()

    masks: dict[str, np.ndarray] = {
        "official_all": np.ones(len(frame), dtype=bool),
        "nonempty_text": ~frame["is_empty_text"].to_numpy(dtype=bool),
    }
    if "overlap_with_any_finetuning_text" in frame:
        overlap = frame["overlap_with_any_finetuning_text"].to_numpy(dtype=bool)
        masks["finetuning_novel_text"] = ~overlap
        masks["finetuning_overlap_text"] = overlap
    if benchmark.name == SEMEVAL_NAME:
        seen_user = frame["is_seen_user"].to_numpy(dtype=bool)
        is_words = frame["is_words"].to_numpy(dtype=bool)
        masks["official_seen_user"] = seen_user
        masks["official_unseen_user"] = ~seen_user
        masks["essay_text"] = ~is_words
        masks["feeling_word_text"] = is_words
    subsets: dict[str, Any] = {}
    for subset_name, mask in masks.items():
        count = int(mask.sum())
        if count == 0:
            subsets[subset_name] = {
                "n_examples": 0,
                "native_scale": None,
                "model_0_1_scale": None,
            }
            continue
        native_metrics = calculate_va_metrics(
            labels_native[mask], predictions_native[mask]
        )
        model_metrics = calculate_va_metrics(labels_model[mask], predictions_model[mask])
        if benchmark.name == MSP_NAME:
            native_metrics["ccc_mean_va"] = native_metrics["ccc_mean"]
            model_metrics["ccc_mean_va"] = model_metrics["ccc_mean"]
        subsets[subset_name] = {
            "n_examples": count,
            "native_scale": _json_ready(native_metrics),
            "model_0_1_scale": _json_ready(model_metrics),
        }

    result: dict[str, Any] = {
        "benchmark": benchmark.name,
        "benchmark_version": benchmark.version,
        "split": benchmark.split,
        "output_order": list(MODEL_OUTPUT_NAMES),
        "primary": "subsets.official_all.native_scale",
        "subsets": subsets,
    }
    if benchmark.name == OMG_NAME and benchmark.official_group_column:
        result["secondary_macro_video"] = _json_ready(
            macro_group_metrics(
                labels_native,
                predictions_native,
                benchmark.frame[benchmark.official_group_column],
            )
        )
    if benchmark.name == MSP_NAME:
        result["score_scope"] = (
            "VA-only mean CCC; not the official MSP V/A/D leaderboard score because "
            "the current model has no dominance output"
        )
    elif benchmark.name == IDEST_NAME:
        result["score_scope"] = (
            "all 250 English translations; no official train/test split or leaderboard "
            "score is defined, so native-scale global VA regression metrics are primary"
        )
    elif benchmark.name == SEMEVAL_NAME:
        result["official_subtask1"] = semeval_subtask1_official_metrics(
            labels_native,
            predictions_native,
            frame["user_id"],
        )
        diagnostic_strata: dict[str, Any] = {}
        for subset_name in (
            "official_seen_user",
            "official_unseen_user",
            "essay_text",
            "feeling_word_text",
        ):
            mask = masks[subset_name]
            diagnostic_strata[subset_name] = semeval_subtask1_official_metrics(
                labels_native[mask],
                predictions_native[mask],
                frame.loc[mask, "user_id"],
            )
        result["diagnostic_official_formula_strata"] = diagnostic_strata
        result["primary"] = "official_subtask1.r_composite_mean_va"
        result["score_scope"] = (
            "official Subtask 1 test rows and released user-aware r_composite formula; "
            "seen/unseen-user and essay/feeling-word scores are diagnostic strata"
        )
    return _json_ready(result)


def build_prediction_report(
    benchmark: ExternalBenchmarkData,
    member_predictions: Mapping[str, np.ndarray],
    ensemble_predictions: np.ndarray,
    *,
    audited_frame: pd.DataFrame | None = None,
    include_gold_labels: bool = False,
) -> pd.DataFrame:
    """Build an ordered item report without raw transcripts or default gold redistribution."""

    frame = _aligned_audited_frame(benchmark, audited_frame)
    metadata_columns = ["benchmark_id", "split", "text_sha256", "is_empty_text"]
    for optional in (
        *benchmark.prediction_metadata_columns,
        "token_count_before_truncation",
        "was_truncated",
        "was_truncated_at_checkpoint_max_length",
        "was_truncated_at_evaluation_max_length",
        "overlap_with_heldout_fold1_training",
        "overlap_with_heldout_fold2_training",
        "overlap_with_any_finetuning_text",
    ):
        if optional in frame.columns:
            metadata_columns.append(optional)
    output = frame.loc[:, metadata_columns].copy()
    if include_gold_labels:
        output["gold_valence_native"] = benchmark.frame["native_valence"].to_numpy()
        output["gold_arousal_native"] = benchmark.frame["native_arousal"].to_numpy()
    for name, predictions in member_predictions.items():
        model_array = np.asarray(predictions, dtype=np.float64)
        if len(model_array) != len(benchmark.frame):
            raise ValueError(
                f"Member predictions for {name!r} do not align with benchmark rows."
            )
        native_array = benchmark.predictions_to_native(model_array)
        output[f"{name}_pred_valence_model_0_1"] = model_array[:, 0]
        output[f"{name}_pred_arousal_model_0_1"] = model_array[:, 1]
        output[f"{name}_pred_valence_native"] = native_array[:, 0]
        output[f"{name}_pred_arousal_native"] = native_array[:, 1]
    ensemble_model = np.asarray(ensemble_predictions, dtype=np.float64)
    if len(ensemble_model) != len(benchmark.frame):
        raise ValueError("Ensemble predictions do not align with benchmark rows.")
    ensemble_native = benchmark.predictions_to_native(ensemble_model)
    output["ensemble_pred_valence_model_0_1"] = ensemble_model[:, 0]
    output["ensemble_pred_arousal_model_0_1"] = ensemble_model[:, 1]
    output["ensemble_pred_valence_native"] = ensemble_native[:, 0]
    output["ensemble_pred_arousal_native"] = ensemble_native[:, 1]
    return output


def _write_external_evaluation_payload(
    output_dir: str | Path,
    benchmark: ExternalBenchmarkData,
    members: Sequence[SavedRunMember],
    member_predictions: Mapping[str, np.ndarray],
    ensemble_predictions: np.ndarray,
    *,
    audited_frame: pd.DataFrame,
    overlap_report: Mapping[str, Any],
    evaluation_manifest: Mapping[str, Any],
    include_gold_labels: bool = False,
) -> Path:
    """Write every result artifact into an already private staging directory."""

    output = Path(output_dir)
    prediction_report = build_prediction_report(
        benchmark,
        member_predictions,
        ensemble_predictions,
        audited_frame=audited_frame,
        include_gold_labels=include_gold_labels,
    )
    prediction_report.to_csv(output / "predictions.tsv", sep="\t", index=False)

    metrics = {
        "ensemble": calculate_external_metrics(
            benchmark,
            ensemble_predictions,
            audited_frame=audited_frame,
        ),
        "members": {
            name: calculate_external_metrics(
                benchmark,
                predictions,
                audited_frame=audited_frame,
            )
            for name, predictions in member_predictions.items()
        },
    }
    protocol_metric_fields = (
        "evaluation_protocol_classification",
        "primary_benchmark_result",
        "saved_checkpoint_max_length",
        "evaluation_max_length",
        "max_length_override",
    )
    protocol_metric_metadata = {
        field: evaluation_manifest[field]
        for field in protocol_metric_fields
        if field in evaluation_manifest
    }
    if protocol_metric_metadata:
        metrics["evaluation_protocol"] = protocol_metric_metadata
    with open(output / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(_json_ready(metrics), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    with open(output / "overlap_audit.json", "w", encoding="utf-8") as output_file:
        json.dump(_json_ready(overlap_report), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    manifest = {
        "schema_version": EXTERNAL_EVALUATION_SCHEMA_VERSION,
        "benchmark": benchmark.name,
        "benchmark_version": benchmark.version,
        "split": benchmark.split,
        "rows": int(len(benchmark.frame)),
        "training_performed": False,
        "gradient_updates": 0,
        "calibration_performed": False,
        "external_model_selection": False,
        "external_checkpoint_weighting": False,
        "external_labels_use": (
            "schema/range validation and post-prediction metrics only; never model "
            "inputs, calibration, member selection, or ensemble weighting"
        ),
        "checkpoint_origin": (
            "pre-existing final_model artifacts saved by the original completed "
            "training run before external evaluation"
        ),
        "ensemble": "fixed unweighted arithmetic mean of both fold members on model [0,1] scale",
        "context_policy": benchmark.context_policy,
        "normalization_policy": "documented fixed affine scale only; no observed benchmark statistics",
        "model_output_order": list(MODEL_OUTPUT_NAMES),
        "model_output_scale": {"valence": [0.0, 1.0], "arousal": [0.0, 1.0]},
        "native_scale": benchmark.native_scale,
        "model_to_native_scale": list(benchmark.model_to_native_scale),
        "model_to_native_offset": list(benchmark.model_to_native_offset),
        "join_report": benchmark.join_report,
        "source_manifest": benchmark.source_manifest,
        "item_report_contains_raw_text": False,
        "item_report_contains_gold_labels": bool(include_gold_labels),
        "members": [
            {
                "name": member.name,
                "held_out_fold": member.held_out_fold,
                "training_fold": member.training_fold,
                "model_dir": str(member.model_dir),
                "file_sha256": member.file_sha256,
                "finetuning_mode": member.run_manifest.get("finetuning_mode"),
                "dtype": member.run_manifest.get("dtype"),
                "gaze_fusion": member.run_manifest.get("gaze_fusion"),
                "gaze_features": member.run_manifest.get("gaze_features"),
                "gaze_redistribution": _saved_redistribution_contract(
                    member.run_manifest,
                    member.run_manifest,
                    label=f"{member.name} run manifest",
                ),
                "et_revision": member.run_manifest.get("et_revision"),
                "max_length": member.run_manifest.get("max_length"),
            }
            for member in members
        ],
    }
    reserved_overrides = sorted(set(manifest).intersection(evaluation_manifest))
    if reserved_overrides:
        raise ValueError(
            "evaluation_manifest cannot override authoritative result field(s): "
            + ", ".join(reserved_overrides)
        )
    manifest.update(dict(evaluation_manifest))
    with open(
        output / "external_evaluation_manifest.json", "w", encoding="utf-8"
    ) as output_file:
        json.dump(_json_ready(manifest), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    if benchmark.name == OMG_NAME:
        for name, predictions in {
            **dict(member_predictions),
            "ensemble": ensemble_predictions,
        }.items():
            native = benchmark.predictions_to_native(predictions)
            official = benchmark.frame.loc[:, ["video", "utterance"]].copy()
            official["arousal"] = native[:, 1]
            official["valence"] = native[:, 0]
            official.to_csv(output / f"omg_official_predictions_{name}.csv", index=False)
    elif benchmark.name == MSP_NAME:
        for name, predictions in {
            **dict(member_predictions),
            "ensemble": ensemble_predictions,
        }.items():
            native = benchmark.predictions_to_native(predictions)
            va_only = pd.DataFrame(
                {
                    "FileName": benchmark.frame["file_name"],
                    "EmoAct": native[:, 1],
                    "EmoVal": native[:, 0],
                }
            )
            va_only.to_csv(output / f"msp_va_only_predictions_{name}.csv", index=False)
    elif benchmark.name == IDEST_NAME:
        for name, predictions in {
            **dict(member_predictions),
            "ensemble": ensemble_predictions,
        }.items():
            native = benchmark.predictions_to_native(predictions)
            ide_st = pd.DataFrame(
                {
                    "code": benchmark.frame["idest_code"],
                    "pred_valence": native[:, 0],
                    "pred_arousal": native[:, 1],
                }
            )
            ide_st.to_csv(output / f"idest_predictions_{name}.csv", index=False)
    elif benchmark.name == SEMEVAL_NAME:
        for name, predictions in {
            **dict(member_predictions),
            "ensemble": ensemble_predictions,
        }.items():
            native = benchmark.predictions_to_native(predictions)
            submission = pd.DataFrame(
                {
                    "user_id": benchmark.frame["user_id"],
                    "text_id": benchmark.frame["text_id"],
                    "pred_valence": native[:, 0],
                    "pred_arousal": native[:, 1],
                }
            )
            submission.to_csv(
                output / f"semeval_subtask1_predictions_{name}.csv", index=False
            )
    return output


def write_external_evaluation(
    output_dir: str | Path,
    benchmark: ExternalBenchmarkData,
    members: Sequence[SavedRunMember],
    member_predictions: Mapping[str, np.ndarray],
    ensemble_predictions: np.ndarray,
    *,
    audited_frame: pd.DataFrame,
    overlap_report: Mapping[str, Any],
    evaluation_manifest: Mapping[str, Any],
    include_gold_labels: bool = False,
) -> Path:
    """Stage a complete evaluation and atomically publish it without overwriting."""

    member_names = tuple(member.name for member in members)
    if (
        len(member_names) != 2
        or len(set(member_names)) != 2
        or {member.held_out_fold for member in members} != {1, 2}
    ):
        raise ValueError(
            "External evaluation output requires exactly the two held-out fold members."
        )
    if set(member_predictions) != set(member_names):
        raise ValueError(
            "member_predictions keys must exactly match the SavedRunMember names."
        )
    expected_ensemble = fixed_unweighted_ensemble(
        [member_predictions[name] for name in member_names]
    )
    supplied_ensemble = np.asarray(ensemble_predictions, dtype=np.float64)
    if not np.array_equal(supplied_ensemble, expected_ensemble):
        raise ValueError(
            "ensemble_predictions must be the exact fixed unweighted member mean."
        )
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"External evaluation output already exists; refusing to overwrite: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        _write_external_evaluation_payload(
            temporary,
            benchmark,
            members,
            member_predictions,
            ensemble_predictions,
            audited_frame=audited_frame,
            overlap_report=overlap_report,
            evaluation_manifest=evaluation_manifest,
            include_gold_labels=include_gold_labels,
        )
        with open(
            temporary / EXTERNAL_EVALUATION_COMPLETION_MARKER,
            "x",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                {
                    "schema_version": EXTERNAL_EVALUATION_SCHEMA_VERSION,
                    "status": "completed",
                },
                output_file,
                indent=2,
                sort_keys=True,
            )
            output_file.write("\n")
        if output.exists():
            raise FileExistsError(
                f"External evaluation output already exists; refusing to overwrite: {output}"
            )
        _atomic_publish_directory_no_replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


__all__ = [
    "EXTERNAL_BENCHMARKS",
    "ExternalBenchmarkData",
    "IDEST_NAME",
    "MSP_NAME",
    "OMG_NAME",
    "OMG_REVISION",
    "SEMEVAL_NAME",
    "SavedRunMember",
    "TextBatchCollator",
    "TokenizedTextDataset",
    "audit_finetuning_text_overlap",
    "build_prediction_report",
    "calculate_external_metrics",
    "discover_completed_run",
    "download_pinned_idest_english",
    "download_pinned_omg_test",
    "download_pinned_semeval_subtask1_test",
    "fixed_unweighted_ensemble",
    "load_idest_english",
    "load_msp_podcast_test",
    "load_msp_transcript_mapping",
    "load_omg_emotion_test",
    "load_semeval_2026_subtask1_test",
    "macro_group_metrics",
    "model_predictions_to_native",
    "normalize_overlap_text",
    "reject_benchmark_training_sources",
    "semeval_subtask1_official_metrics",
    "write_external_evaluation",
]

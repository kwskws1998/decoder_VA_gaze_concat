"""Immutable source-aware text-group splits and run ownership checks."""

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trt_pipeline.config import DATA_DIR, DATA_HASHES, ROOT
from trt_pipeline.core.filters import load_filtered_folds
from trt_pipeline.data import contract_sha256, frame_sha256, source_counts, text_groups

from .config import ARMS, CONDITIONS, EXPOSURE_NOTE, Settings, updates_per_epoch


PARTITIONS = ("train", "val", "test")
COLUMNS = ["index", "text", "dataset_of_origin", "valence", "arousal"]


def utc_now():
    """Record a timezone-qualified timestamp for receipts and progress."""

    return datetime.now(timezone.utc).isoformat()


def digest(path):
    """Hash files incrementally, including large safetensors checkpoints."""

    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value, *, exclusive=False):
    """Write strict JSON, using exclusive creation for irreversible stage claims."""

    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def source_manifest():
    """Fingerprint both the new runner and all shared numerical implementations."""

    paths = [ROOT / name for name in ("requirements.txt", "run_experiments.py",
                                      "setup_environment.py", "environment_runtime.py")]
    for folder in ("holdout_experiments", "trt_pipeline"):
        paths.extend(sorted((ROOT / folder).rglob("*.py")))
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def validate_frame(frame):
    """Require unique Unicode identities, explicit sources and bounded VA."""

    if frame.empty or set(COLUMNS).difference(frame.columns):
        raise ValueError("A nonempty canonical VA table is required.")
    if frame["index"].astype(str).duplicated().any() or frame["index"].astype(str).eq("").any():
        raise ValueError("Row IDs must be nonempty and globally unique.")
    if frame["dataset_of_origin"].astype(str).eq("").any():
        raise ValueError("Every row needs a source identity.")
    gold = frame[["valence", "arousal"]].to_numpy(dtype=float)
    if not np.isfinite(gold).all() or (gold < 0).any() or (gold > 1).any():
        raise ValueError("VA targets must be finite and in [0,1].")


def grouped_split(frame, settings):
    """Greedily balance source row counts while keeping entire normalized texts together.

    Descending group size with seeded ties fills three source-count targets by
    minimum incremental squared error. Labels never participate in assignment.
    Duplicate groups spanning sources are assigned as one indivisible unit.
    """

    validate_frame(frame)
    groups, inverse = np.unique(text_groups(frame), return_inverse=True)
    if len(groups) < 3:
        raise ValueError("At least three distinct text groups are needed.")
    sources, source_index = np.unique(frame["dataset_of_origin"].astype(str), return_inverse=True)
    counts = np.zeros((len(groups), len(sources)), dtype=np.int64)
    np.add.at(counts, (inverse, source_index), 1)
    sizes = counts.sum(axis=1)
    rng = np.random.default_rng(settings.split_seed)
    order = np.lexsort((rng.random(len(groups)), -sizes))
    target = np.asarray(settings.fractions)[:, None] * counts.sum(axis=0)[None, :]
    assigned = np.zeros_like(target)
    group_parts = np.full(len(groups), -1, dtype=np.int64)
    tie_order = rng.permutation(3)
    for group in order:
        change = ((assigned + counts[group] - target)**2 - (assigned - target)**2)
        costs = (change / np.maximum(target, 1.0)).sum(axis=1)
        part = int(tie_order[np.argmin(costs[tie_order])])
        group_parts[group] = part
        assigned[part] += counts[group]
    result = {name: frame.loc[group_parts[inverse] == i, COLUMNS].reset_index(drop=True).copy()
              for i, name in enumerate(PARTITIONS)}
    validate_partitions(result)
    return result


def validate_partitions(frames):
    """Verify complete pairwise row and normalized-text disjointness."""

    ids, groups = [], []
    for name in PARTITIONS:
        validate_frame(frames[name])
        ids.append(set(frames[name]["index"].astype(str)))
        groups.append(set(text_groups(frames[name])))
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if ids[first] & ids[second] or groups[first] & groups[second]:
            raise ValueError(f"Data leakage between {PARTITIONS[first]} and {PARTITIONS[second]}.")


def prepare(output, data_dir=DATA_DIR, settings=None):
    """Create an exclusive run with locked data, sources and all training settings."""

    settings = Settings() if settings is None else settings
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Run directory already exists: {output}")
    original = {name: digest(data_dir / name) for name in DATA_HASHES}
    if original != DATA_HASHES:
        raise ValueError("Canonical input SHA-256 mismatch.")
    filtered = load_filtered_folds(data_dir, no_iemocap=True)
    frame = pd.concat((filtered.fold1, filtered.fold2), ignore_index=True)[COLUMNS]
    frame["index"] = frame["index"].astype(str)
    frames = grouped_split(frame, settings)
    output.mkdir(parents=True, exist_ok=False)
    (output / "splits").mkdir()
    partitions = {}
    for name, selected in frames.items():
        path = output / "splits" / f"{name}.tsv"
        selected.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
        loaded = read_partition(path)
        if (any(not np.array_equal(loaded[name].to_numpy(str), selected[name].to_numpy(str))
                for name in ("index", "text", "dataset_of_origin"))
                or not np.array_equal(loaded[["valence", "arousal"]].to_numpy(float),
                                      selected[["valence", "arousal"]].to_numpy(float))):
            raise ValueError("Split serialization changed identities, text, sources or exact gold values.")
        partitions[name] = {
            "rows": len(loaded), "groups": len(set(text_groups(loaded))),
            "sources": source_counts(loaded), "sha256": digest(path),
            "frame_sha256": frame_sha256(loaded),
        }
    lock = {
        "schema": 1, "created_utc": utc_now(), "settings": settings.as_dict(),
        "conditions": list(CONDITIONS), "training_arms": list(ARMS),
        "original_sha256": original, "combined_rows": len(frame), "partitions": partitions,
        "split_algorithm": "source_count_greedy_normalized_text_groups_v1",
        "group_normalization": "NFKC+casefold+whitespace-collapse",
        "blank_text_policy": "retain_as_one_group_and_use_shared_tokenizer_EOS_fallback",
        "labels_used_for_splitting": False, "iemocap": "excluded",
        "checkpoint_selection": "minimum_validation_mean_MSE_first_tie_wins",
        "test_policy": "evaluate_once_after_all_three_selection_receipts_are_sealed",
        "sample_order": "uniform_seeded_permutation_same_across_arms_each_epoch",
        "updates_per_epoch": updates_per_epoch(len(frames["train"]), settings),
        "total_updates_per_fit": updates_per_epoch(len(frames["train"]), settings) * settings.epochs,
        "companion": "train_on_all_train_rows_clean_and_noisy_MSE_clean_eval",
        "historical_exposure": EXPOSURE_NOTE, "source_sha256": source_manifest(),
    }
    lock["protocol_sha256"] = contract_sha256(lock)
    write_json(output / "protocol_lock.json", lock, exclusive=True)
    return lock


def read_partition(path):
    """Round-trip escaped TSV without numeric coercion of Unicode row IDs."""

    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\",
                       keep_default_na=False, dtype={"index": str, "text": str, "dataset_of_origin": str},
                       float_precision="round_trip")


def load_lock(run):
    """Reject edited config, data, or code before touching a model or a stage claim."""

    run = Path(run)
    lock = json.loads((run / "protocol_lock.json").read_text())
    body = {key: value for key, value in lock.items() if key != "protocol_sha256"}
    if contract_sha256(body) != lock.get("protocol_sha256"):
        raise ValueError("Protocol lock has changed.")
    settings = Settings(**lock["settings"])
    if lock["source_sha256"] != source_manifest():
        raise ValueError("Source changed since prepare; create a new run before training.")
    for name in PARTITIONS:
        if digest(run / "splits" / f"{name}.tsv") != lock["partitions"][name]["sha256"]:
            raise ValueError(f"Locked {name} data changed.")
    return lock, settings

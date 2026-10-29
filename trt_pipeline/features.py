"""Sequential encoder extraction and fingerprinted, non-pickle feature caches."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .data import frame_sha256


def _validate_arrays(arrays):
    """Require aligned finite arrays with safe numeric and Unicode storage."""

    expected = {"features", "predictions", "labels", "row_ids"}
    if set(arrays) != expected:
        raise ValueError(f"Feature cache needs exactly {sorted(expected)}.")
    values = {key: np.asarray(value) for key, value in arrays.items()}
    if values["features"].ndim != 2:
        raise ValueError("features must be a nonempty [rows, hidden] matrix.")
    count = len(values["features"])
    if count == 0 or values["features"].shape[1] == 0:
        raise ValueError("features must be a nonempty [rows, hidden] matrix.")
    for name in ("predictions", "labels"):
        if values[name].shape != (count, 2):
            raise ValueError(f"{name} must have shape [rows, 2].")
    if values["row_ids"].shape != (count,):
        raise ValueError("row_ids must be a row-aligned vector.")
    for name in ("features", "predictions", "labels"):
        if not np.issubdtype(values[name].dtype, np.number) or np.iscomplexobj(values[name]) or not np.isfinite(values[name]).all():
            raise ValueError(f"{name} must contain finite numeric values.")
        values[name] = values[name].astype(np.float32, copy=False)
        if not np.isfinite(values[name]).all():
            raise ValueError(f"{name} is outside the finite float32 range.")
    values["row_ids"] = values["row_ids"].astype(str)
    return values


def feature_fingerprint(frame, *, model_fingerprint, max_length, compute_dtype, tokenizer_fingerprint=None):
    """Bind cache values to exact ordered data and encoder reconstruction settings."""

    return {"schema_version": 1, "frame_sha256": frame_sha256(frame), "model": model_fingerprint,
            "max_length": int(max_length), "compute_dtype": str(compute_dtype),
            "tokenizer": tokenizer_fingerprint}


def save_feature_cache(path, arrays, *, fingerprint):
    """Atomically save NPZ arrays and a checksum guarded JSON sidecar."""

    from .progress import atomic_write_json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _validate_arrays(arrays)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        digest = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    metadata = {"fingerprint": fingerprint, "sha256": digest,
                "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()}}
    atomic_write_json(path.with_suffix(".json"), metadata)
    return metadata


def load_feature_cache(path, *, expected_fingerprint=None):
    """Reject changed model/data ownership, corrupted bytes, or unsafe array schemas."""

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if expected_fingerprint is not None and metadata["fingerprint"] != expected_fingerprint:
        raise ValueError("Feature cache fingerprint does not match this model and dataset.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Feature cache checksum mismatch.")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays = _validate_arrays(arrays)
    for name, value in arrays.items():
        if list(value.shape) != metadata["arrays"][name]["shape"] or str(value.dtype) != metadata["arrays"][name]["dtype"]:
            raise ValueError("Feature cache array schema mismatch.")
    return arrays


def extract_features(model, tokenizer, frame, *, max_length, batch_size, device, compute_dtype, progress=None):
    """Extract one-pass pooled states and baseline predictions in exact frame order."""

    import torch
    from torch.utils.data import DataLoader
    from trt_pipeline.core.dataset import TokenizedVADataset, VABatchCollator

    if frame.empty or int(batch_size) <= 0:
        raise ValueError("Feature extraction needs nonempty rows and positive batch size.")
    device = torch.device(device)
    dataset = TokenizedVADataset(frame, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=VABatchCollator(tokenizer, pad_to_multiple_of=8))
    dtype = compute_dtype
    if isinstance(dtype, str):
        dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
                 "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}[dtype]
    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    feature_batches, prediction_batches, label_batches = [], [], []
    completed = 0
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for batch in loader:
                labels = batch.pop("labels")
                inputs = {key: value.to(device) for key, value in batch.items() if key != "token_type_ids"}
                context = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext()
                with context:
                    pooled = model.encode(**inputs).pooled
                    predictions = model.predict_pooled(pooled)
                feature_batches.append(pooled.float().cpu().numpy())
                prediction_batches.append(predictions.float().cpu().numpy())
                label_batches.append(labels.float().numpy())
                completed += len(labels)
                if progress is not None:
                    progress(completed, len(dataset), time.monotonic() - started)
    finally:
        for module, was_training in training_states:
            module.training = was_training
    return _validate_arrays({"features": np.concatenate(feature_batches), "predictions": np.concatenate(prediction_batches),
                             "labels": np.concatenate(label_batches), "row_ids": frame["index"].astype(str).to_numpy()})

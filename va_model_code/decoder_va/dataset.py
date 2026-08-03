"""Tokenizer-agnostic PyTorch datasets and padding for VA regression."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .filters import load_filtered_folds, read_fold


class TokenizedVADataset(Dataset):
    """Lazily tokenize a VA fold with any Hugging Face-compatible tokenizer."""

    def __init__(
        self,
        data: pd.DataFrame | str | os.PathLike[str],
        tokenizer,
        *,
        max_length: int,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if isinstance(data, (str, os.PathLike, Path)):
            frame = read_fold(data)
        elif isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            raise TypeError("data must be a fold path or pandas DataFrame.")

        required = {"index", "text", "dataset_of_origin", "valence", "arousal"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"VA data is missing required column(s): {', '.join(missing)}")
        valence = pd.to_numeric(frame["valence"], errors="coerce").to_numpy(dtype=float)
        arousal = pd.to_numeric(frame["arousal"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(valence).all() or not np.isfinite(arousal).all():
            raise ValueError("VA labels must all be finite numbers.")

        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.index = frame["index"].tolist()
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.dataset_of_origin = frame["dataset_of_origin"].astype(str).tolist()
        self.valence = valence.tolist()
        self.arousal = arousal.tolist()

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
        output["labels"] = torch.tensor(
            [self.valence[item], self.arousal[item]], dtype=torch.float32
        )
        return output


class VABatchCollator:
    """Pad decoder or encoder token batches and stack two-dimensional VA labels."""

    def __init__(self, tokenizer, *, pad_to_multiple_of: int | None = None) -> None:
        if pad_to_multiple_of is not None and int(pad_to_multiple_of) <= 0:
            raise ValueError("pad_to_multiple_of must be positive when provided.")
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            eos_token = getattr(tokenizer, "eos_token", None)
            if (
                eos_token_id is not None
                and eos_token is not None
                and hasattr(tokenizer, "pad_token")
            ):
                tokenizer.pad_token = eos_token
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _manual_pad(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            maximum = ((maximum + multiple - 1) // multiple) * multiple
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id; configure a pad token."
            )
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        batch: dict[str, list[torch.Tensor]] = {}
        sequence_keys = set().union(*(feature.keys() for feature in features))
        sequence_keys.discard("labels")
        for feature in features:
            for key in sequence_keys:
                value = feature.get(key)
                if value is None:
                    if key == "attention_mask":
                        value = torch.ones_like(feature["input_ids"])
                    elif key == "token_type_ids":
                        value = torch.zeros_like(feature["input_ids"])
                    else:
                        raise ValueError(f"Feature is missing sequence field: {key}")
                amount = maximum - len(value)
                fill = pad_id if key == "input_ids" else 0
                padding = torch.full((amount,), fill, dtype=torch.long)
                padded = (
                    torch.cat((padding, value))
                    if padding_side == "left"
                    else torch.cat((value, padding))
                )
                batch.setdefault(key, []).append(padded)
        return {key: torch.stack(values) for key, values in batch.items()}

    def __call__(self, features: Iterable[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        items = [dict(feature) for feature in features]
        if not items:
            raise ValueError("Cannot collate an empty VA batch.")
        labels = torch.stack(
            [torch.as_tensor(item.pop("labels"), dtype=torch.float32) for item in items]
        )
        pad_method = getattr(self.tokenizer, "pad", None)
        if callable(pad_method):
            serializable = [
                {key: value.tolist() if torch.is_tensor(value) else value for key, value in item.items()}
                for item in items
            ]
            batch = pad_method(
                serializable,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            batch = {key: torch.as_tensor(value) for key, value in dict(batch).items()}
        else:
            batch = self._manual_pad(items)
        batch["labels"] = labels
        return batch


def load_auto_tokenizer(checkpoint: str, **kwargs):
    """Explicit tokenizer loader; no model or tokenizer is downloaded on module import."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer {checkpoint!r} has no pad or EOS token for batched training."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_fold_datasets(
    data_dir: str | os.PathLike[str],
    tokenizer,
    *,
    max_length: int,
    exclude_dataset: str | Iterable[str] | None = None,
    no_iemocap: bool = False,
    no_ieomcap: bool = False,
) -> tuple[TokenizedVADataset, TokenizedVADataset, tuple[str, ...]]:
    filtered = load_filtered_folds(
        data_dir,
        exclude_dataset=exclude_dataset,
        no_iemocap=no_iemocap,
        no_ieomcap=no_ieomcap,
    )
    return (
        TokenizedVADataset(filtered.fold1, tokenizer, max_length=max_length),
        TokenizedVADataset(filtered.fold2, tokenizer, max_length=max_length),
        filtered.excluded_names,
    )


class MyDataset(TokenizedVADataset):
    """Compatibility adapter for the original filename/checkpoint/maxlen API."""

    def __init__(
        self,
        filename: str | os.PathLike[str] | pd.DataFrame,
        checkpoint: str | None = None,
        maxlen: int = 256,
        *,
        tokenizer=None,
        tokenizer_kwargs: dict | None = None,
    ) -> None:
        if tokenizer is None:
            if checkpoint is None:
                raise ValueError("Provide either tokenizer or checkpoint.")
            tokenizer = load_auto_tokenizer(checkpoint, **(tokenizer_kwargs or {}))
        super().__init__(filename, tokenizer, max_length=maxlen)

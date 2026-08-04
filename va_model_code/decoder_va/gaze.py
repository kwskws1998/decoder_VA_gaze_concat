"""Safe, lazy ET2 inference and exact target-token gaze alignment."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import unicodedata
from typing import Sequence

import torch
from torch import nn

from .alignment import align_words_to_tokens


ET2_FEATURE_NAMES = ("nFix", "FFD", "GPT", "TRT", "fixProp")
DEFAULT_ET2_REPO_ID = "skboy/et_prediction_2"
DEFAULT_ET2_REVISION = "5785e77309d9fce8b88e908a9db100c1a0a63456"
DEFAULT_ET2_FILENAME = "et_predictor2_seed123.safetensors"
ET2_MAX_INPUT_TOKENS = 512


def _et2_roberta_config():
    """Recreate the exact roberta-base configuration used to train ET2."""

    from transformers import RobertaConfig

    return RobertaConfig(
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
    )


class _ET2RegressionModel(nn.Module):
    """Implement the published RoBERTa-base plus five-output linear architecture."""

    def __init__(self):
        super().__init__()
        from transformers import RobertaModel

        config = _et2_roberta_config()
        self.roberta = RobertaModel(config)
        self.decoder = nn.Linear(config.hidden_size, len(ET2_FEATURE_NAMES))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the five raw ET channels for every RoBERTa token."""

        hidden = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.decoder(hidden)


def _is_cjk(character: str) -> bool:
    """Return whether a character belongs to the supported CJK/Hangul ranges."""

    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xAC00 <= code <= 0xD7AF
    )


def segment_text_for_et2(text: str) -> list[str]:
    """Split text on whitespace and isolate CJK, punctuation, and symbols."""

    segments: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            segments.append("".join(buffer))
            buffer.clear()

    for character in str(text or ""):
        category = unicodedata.category(character)
        if character.isspace():
            flush_buffer()
        elif _is_cjk(character) or category.startswith(("P", "S")):
            flush_buffer()
            segments.append(character)
        else:
            buffer.append(character)
    flush_buffer()
    return segments


class ET2GazeProvider:
    """Provide frozen TRT-only ET2 features aligned to target first subwords."""

    def __init__(
        self,
        tokenizer,
        repo_id: str = DEFAULT_ET2_REPO_ID,
        revision: str = DEFAULT_ET2_REVISION,
        filename: str = DEFAULT_ET2_FILENAME,
        feature_index: int = 3,
        cache_size: int = 20000,
        max_length: int = ET2_MAX_INPUT_TOKENS,
        device: str | torch.device | None = None,
    ):
        if tokenizer is None:
            raise ValueError("A target-model tokenizer is required.")
        if not str(repo_id).strip():
            raise ValueError("repo_id must be a non-empty Hugging Face repository ID.")
        if not str(revision).strip():
            raise ValueError("revision must be a non-empty branch, tag, or commit.")
        if not str(filename).endswith(".safetensors"):
            raise ValueError("ET2 weights must be loaded from a .safetensors file.")
        if not 0 <= int(feature_index) < len(ET2_FEATURE_NAMES):
            raise ValueError(
                f"feature_index must be in [0, {len(ET2_FEATURE_NAMES) - 1}]."
            )
        if int(cache_size) < 0:
            raise ValueError("cache_size cannot be negative.")
        if int(max_length) <= 2:
            raise ValueError("max_length must leave room for lexical tokens.")
        if int(max_length) > ET2_MAX_INPUT_TOKENS:
            raise ValueError(
                f"ET2 max_length cannot exceed {ET2_MAX_INPUT_TOKENS} input tokens."
            )

        self.tokenizer = tokenizer
        self.repo_id = str(repo_id)
        self.revision = str(revision)
        self.filename = str(filename)
        self.feature_index = int(feature_index)
        self.cache_size = int(cache_size)
        self.max_length = int(max_length)
        self.device = torch.device(device) if device is not None else None

        self._et_tokenizer = None
        self._model = None
        self._model_device = None
        self._cache: OrderedDict[
            tuple[object, ...],
            tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()

    @property
    def is_loaded(self) -> bool:
        """Report whether the ET tokenizer and frozen model have been initialized."""

        return self._et_tokenizer is not None and self._model is not None

    def _load_assets(self):
        """Download only safe tokenizer assets and the requested safetensors file."""

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        et_tokenizer = AutoTokenizer.from_pretrained(
            self.repo_id,
            revision=self.revision,
            use_fast=True,
            trust_remote_code=False,
            add_prefix_space=True,
        )
        weights_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            revision=self.revision,
        )
        model = _ET2RegressionModel()
        state_dict = load_file(weights_path, device="cpu")
        model.load_state_dict(state_dict, strict=True)
        return et_tokenizer, model

    def _ensure_loaded(self, execution_device: torch.device) -> None:
        """Lazily initialize, strictly freeze, and place ET2 on one explicit device."""

        if not self.is_loaded:
            et_tokenizer, model = self._load_assets()
            model.eval()
            model.requires_grad_(False)
            self._et_tokenizer = et_tokenizer
            self._model = model
        if self._model_device != execution_device:
            self._model.to(execution_device)
            self._model_device = execution_device
        self._model.eval()
        self._model.requires_grad_(False)

    def _cache_key(self, valid_token_ids: Sequence[int]) -> tuple[object, ...]:
        """Key a target token sequence by the complete ET artifact identity."""

        return (
            self.repo_id,
            self.revision,
            self.filename,
            self.feature_index,
            tuple(int(token_id) for token_id in valid_token_ids),
        )

    def _validate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[int]:
        """Validate a right-padded token batch and return its active lengths."""

        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask,
            torch.Tensor,
        ):
            raise TypeError("input_ids and attention_mask must be torch tensors.")
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must both be rank-2.")
        if tuple(input_ids.shape) != tuple(attention_mask.shape):
            raise ValueError("input_ids and attention_mask must have identical shape.")
        if input_ids.shape[0] == 0:
            raise ValueError("The token batch must contain at least one sample.")
        if input_ids.shape[1] == 0:
            raise ValueError("The token sequence must contain at least one position.")
        if input_ids.device != attention_mask.device:
            raise ValueError("input_ids and attention_mask must be on the same device.")
        if input_ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("input_ids must use an integer dtype.")

        if attention_mask.dtype != torch.bool:
            is_binary = torch.logical_or(attention_mask == 0, attention_mask == 1)
            if not bool(is_binary.all().item()):
                raise ValueError("attention_mask must contain only zero and one values.")
        mask_bool = attention_mask.to(dtype=torch.bool)
        valid_lengths = mask_bool.sum(dim=1)
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0)
        expected_mask = positions < valid_lengths.unsqueeze(1)
        if not torch.equal(mask_bool, expected_mask):
            raise ValueError("attention_mask must use contiguous right padding.")
        return [int(value) for value in valid_lengths.detach().cpu().tolist()]

    def _decode_target_rows(self, token_rows: Sequence[Sequence[int]]) -> list[str]:
        """Decode uncached target token rows without retaining target special tokens."""

        batch_decoder = getattr(self.tokenizer, "batch_decode", None)
        if callable(batch_decoder):
            try:
                decoded = batch_decoder(
                    token_rows,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                decoded = batch_decoder(token_rows, skip_special_tokens=True)
            return [str(text) for text in decoded]

        decoder = getattr(self.tokenizer, "decode", None)
        if not callable(decoder):
            raise TypeError("The target tokenizer must expose decode or batch_decode.")
        decoded_rows = []
        for token_row in token_rows:
            try:
                text = decoder(
                    token_row,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                text = decoder(token_row, skip_special_tokens=True)
            decoded_rows.append(str(text))
        return decoded_rows

    def _word_ids_for_row(
        self,
        encoded,
        batch_index: int,
        words: Sequence[str],
        token_ids: Sequence[int],
        attention_mask: Sequence[int],
    ) -> list[int | None]:
        """Read fast-tokenizer word IDs or reconstruct them by exact alignment."""

        word_ids_method = getattr(encoded, "word_ids", None)
        if callable(word_ids_method):
            try:
                word_ids = word_ids_method(batch_index=batch_index)
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                word_ids = None
            if word_ids is not None:
                return list(word_ids)

        alignment = align_words_to_tokens(
            words,
            token_ids,
            attention_mask,
            self._et_tokenizer,
        )
        word_ids: list[int | None] = [None] * len(token_ids)
        for word_index, indices in enumerate(alignment.word_to_token_indices):
            for token_index in indices:
                word_ids[token_index] = word_index
        return word_ids

    def _map_predictions_to_target(
        self,
        words: Sequence[str],
        word_features: torch.Tensor,
        word_feature_mask: torch.Tensor,
        target_ids: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place finite ET word predictions on exact target first subwords."""

        output = torch.zeros(len(target_ids), 1, dtype=torch.float32)
        mapped_mask = torch.zeros(len(target_ids), dtype=torch.bool)
        if not words:
            return output, mapped_mask

        alignment = align_words_to_tokens(
            words,
            target_ids,
            [1] * len(target_ids),
            self.tokenizer,
        )
        for word_index, indices in enumerate(alignment.word_to_token_indices):
            if (
                not indices
                or word_index >= word_features.shape[0]
                or not bool(word_feature_mask[word_index].item())
            ):
                continue
            feature = word_features[word_index]
            if not bool(torch.isfinite(feature).item()):
                continue
            first_subword = indices[0]
            output[first_subword, 0] = feature
            mapped_mask[first_subword] = True
        return output, mapped_mask

    def _predict_uncached(
        self,
        target_rows: Sequence[Sequence[int]],
        execution_device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Batch uncached texts through ET2 and return CPU-aligned TRT rows."""

        decoded_rows = self._decode_target_rows(target_rows)
        segmented_rows = [segment_text_for_et2(text) for text in decoded_rows]
        results: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None
        ] * len(target_rows)
        active_indices = [
            index for index, words in enumerate(segmented_rows) if len(words) > 0
        ]
        for index, words in enumerate(segmented_rows):
            if words:
                continue
            results[index] = (
                torch.zeros(len(target_rows[index]), 1, dtype=torch.float32),
                torch.zeros(len(target_rows[index]), dtype=torch.bool),
            )
        if not active_indices:
            return [result for result in results if result is not None]

        self._ensure_loaded(execution_device)
        active_words = [segmented_rows[index] for index in active_indices]
        encoded = self._et_tokenizer(
            active_words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        et_input_ids = encoded["input_ids"].to(execution_device)
        et_attention_mask = encoded["attention_mask"].to(execution_device)
        autocast_context = (
            torch.autocast(device_type=execution_device.type, enabled=False)
            if execution_device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            predictions = self._model(
                input_ids=et_input_ids,
                attention_mask=et_attention_mask,
            )
        if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3:
            raise RuntimeError("ET2 must return a rank-3 prediction tensor.")
        if predictions.shape[:2] != et_input_ids.shape:
            raise RuntimeError("ET2 prediction and token sequence shapes do not match.")
        if predictions.shape[2] != len(ET2_FEATURE_NAMES):
            raise RuntimeError(
                f"ET2 must return {len(ET2_FEATURE_NAMES)} gaze channels."
            )

        predictions_cpu = predictions.detach().to(device="cpu", dtype=torch.float32)
        et_ids_cpu = et_input_ids.detach().cpu()
        et_mask_cpu = et_attention_mask.detach().cpu()
        for active_batch_index, result_index in enumerate(active_indices):
            words = segmented_rows[result_index]
            ids = et_ids_cpu[active_batch_index].tolist()
            mask = et_mask_cpu[active_batch_index].tolist()
            word_ids = self._word_ids_for_row(
                encoded,
                active_batch_index,
                words,
                ids,
                mask,
            )
            if len(word_ids) != len(ids):
                raise RuntimeError("ET tokenizer returned a malformed word-id sequence.")

            word_features = torch.zeros(len(words), dtype=torch.float32)
            word_feature_mask = torch.zeros(len(words), dtype=torch.bool)
            for token_index, word_index in enumerate(word_ids):
                if (
                    word_index is None
                    or word_index < 0
                    or word_index >= len(words)
                    or word_feature_mask[word_index]
                    or not bool(mask[token_index])
                ):
                    continue
                feature = predictions_cpu[
                    active_batch_index,
                    token_index,
                    self.feature_index,
                ]
                if not bool(torch.isfinite(feature).item()):
                    continue
                word_features[word_index] = feature
                word_feature_mask[word_index] = True

            results[result_index] = self._map_predictions_to_target(
                words,
                word_features,
                word_feature_mask,
                target_rows[result_index],
            )

        if any(result is None for result in results):
            raise RuntimeError("ET2 failed to produce every requested gaze row.")
        return [result for result in results if result is not None]

    def _store_cache(
        self,
        key: tuple[object, ...],
        value: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Store one detached CPU row in the bounded LRU cache."""

        if self.cache_size == 0:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        while len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def compute(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw selected ET2 features and mapped first-subword masks."""

        valid_lengths = self._validate_batch(input_ids, attention_mask)
        batch_size, sequence_length = input_ids.shape
        token_rows = [
            input_ids[row_index, :valid_length].detach().cpu().tolist()
            for row_index, valid_length in enumerate(valid_lengths)
        ]
        keys = [self._cache_key(token_row) for token_row in token_rows]

        resolved: dict[
            tuple[object, ...],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        missing_keys = []
        missing_rows = []
        for key, token_row in zip(keys, token_rows):
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                resolved[key] = cached
            elif key not in resolved and key not in missing_keys:
                missing_keys.append(key)
                missing_rows.append(token_row)

        if missing_rows:
            execution_device = self.device or input_ids.device
            predicted_rows = self._predict_uncached(
                missing_rows,
                execution_device,
            )
            for key, value in zip(missing_keys, predicted_rows):
                cpu_value = (
                    value[0].detach().to(device="cpu", dtype=torch.float32),
                    value[1].detach().to(device="cpu", dtype=torch.bool),
                )
                resolved[key] = cpu_value
                self._store_cache(key, cpu_value)

        features = torch.zeros(
            batch_size,
            sequence_length,
            1,
            dtype=torch.float32,
            device=input_ids.device,
        )
        mapped_mask = torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=input_ids.device,
        )
        for row_index, (key, valid_length) in enumerate(zip(keys, valid_lengths)):
            row_features, row_mask = resolved[key]
            copy_length = min(
                valid_length,
                row_features.shape[0],
                row_mask.shape[0],
            )
            if copy_length == 0:
                continue
            features[row_index, :copy_length] = row_features[:copy_length].to(
                input_ids.device
            )
            mapped_mask[row_index, :copy_length] = row_mask[:copy_length].to(
                input_ids.device
            )
        features = features.masked_fill(~mapped_mask.unsqueeze(-1), 0.0)
        return features, mapped_mask

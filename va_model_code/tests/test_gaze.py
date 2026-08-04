from __future__ import annotations

import pytest
import torch
from torch import nn

from va_model_code.decoder_va import gaze as gaze_module
from va_model_code.decoder_va.alignment import align_words_to_tokens
from va_model_code.decoder_va.gaze import (
    ET2_FEATURE_NAMES,
    ET2GazeProvider,
    _ET2RegressionModel,
    _et2_roberta_config,
    et2_feature_indices_from_names,
    et2_feature_names_from_indices,
    et2_features_used_from_indices,
    normalize_et2_feature_indices,
    segment_text_for_et2,
)
from va_model_code.decoder_va.packing import pack_prefix_gaze


class FakeTargetTokenizer:
    all_special_ids = [0, 1, 2]
    all_special_tokens = ["<s>", "<pad>", "</s>"]

    def __init__(self):
        self.id_to_token = {
            0: "<s>",
            1: "<pad>",
            2: "</s>",
            10: "ĠHel",
            11: "lo",
            12: "Ġworld",
            13: "!",
            14: "Ġlater",
            20: "Ġ你",
            21: "好",
            22: "，",
        }

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.id_to_token[ids]
        return [self.id_to_token[int(token_id)] for token_id in ids]

    def convert_tokens_to_string(self, tokens):
        pieces = []
        for token in tokens:
            if token.startswith("Ġ"):
                pieces.append(" " + token[1:])
            else:
                pieces.append(token)
        return "".join(pieces)

    def batch_decode(
        self,
        rows,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return [
            self.decode(
                row,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for row in rows
        ]

    def decode(
        self,
        row,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        tokens = self.convert_ids_to_tokens(row)
        if skip_special_tokens:
            tokens = [
                token
                for token in tokens
                if token not in self.all_special_tokens
            ]
        return self.convert_tokens_to_string(tokens).strip()


class FakeBatchEncoding(dict):
    def __init__(self, input_ids, attention_mask, word_id_rows):
        super().__init__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        self.word_id_rows = word_id_rows

    def word_ids(self, batch_index=0):
        return self.word_id_rows[batch_index]


class FakeETTokenizer:
    all_special_ids = [0, 1, 2]
    all_special_tokens = ["<s>", "<pad>", "</s>"]

    def __call__(
        self,
        word_rows,
        is_split_into_words,
        return_tensors,
        padding,
        truncation,
        max_length,
    ):
        encoded_rows = []
        word_id_rows = []
        for words in word_rows:
            ids = [0]
            word_ids = [None]
            for word_index, word in enumerate(words):
                piece_count = 2 if word.lower() == "hello" else 1
                for piece_index in range(piece_count):
                    ids.append(10 + word_index * 3 + piece_index)
                    word_ids.append(word_index)
            ids.append(2)
            word_ids.append(None)
            encoded_rows.append(ids[:max_length])
            word_id_rows.append(word_ids[:max_length])

        padded_length = max(len(row) for row in encoded_rows)
        input_rows = []
        mask_rows = []
        for ids, word_ids in zip(encoded_rows, word_id_rows):
            padding_length = padded_length - len(ids)
            input_rows.append(ids + [1] * padding_length)
            mask_rows.append([1] * len(ids) + [0] * padding_length)
            word_ids.extend([None] * padding_length)
        return FakeBatchEncoding(
            torch.tensor(input_rows, dtype=torch.long),
            torch.tensor(mask_rows, dtype=torch.long),
            word_id_rows,
        )


class FakeETModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.batch_sizes = []

    def forward(self, input_ids, attention_mask):
        self.batch_sizes.append(input_ids.shape[0])
        output = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            5,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for feature_index in range(5):
            output[:, :, feature_index] = (
                input_ids.to(dtype=torch.float32) + 100.0 * feature_index
            ) * self.scale
        output[:, :, 3] = input_ids.to(dtype=torch.float32) * self.scale
        return output


class FakeET2GazeProvider(ET2GazeProvider):
    def __init__(self, tokenizer, feature_indices=None):
        super().__init__(
            tokenizer=tokenizer,
            repo_id="fake/et2",
            revision="fixed-revision",
            cache_size=8,
            device="cpu",
            feature_indices=feature_indices,
        )
        self.load_count = 0
        self.fake_model = FakeETModel()

    def _load_assets(self):
        self.load_count += 1
        return FakeETTokenizer(), self.fake_model


def test_et2_config_matches_the_roberta_base_checkpoint_contract():
    config = _et2_roberta_config()

    assert config.vocab_size == 50265
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert config.intermediate_size == 3072
    assert config.max_position_embeddings == 514
    assert config.type_vocab_size == 1
    assert config.layer_norm_eps == pytest.approx(1e-5)
    assert config.pad_token_id == 1
    assert config.bos_token_id == 0
    assert config.eos_token_id == 2


def test_et2_model_embedding_shapes_match_the_pinned_checkpoint():
    with torch.device("meta"):
        model = _ET2RegressionModel()

    assert model.roberta.embeddings.token_type_embeddings.weight.shape == (1, 768)
    assert model.roberta.embeddings.position_embeddings.weight.shape == (514, 768)
    assert model.decoder.weight.shape == (5, 768)


def test_et2_provider_rejects_lengths_beyond_roberta_capacity():
    with pytest.raises(ValueError, match="cannot exceed 512"):
        ET2GazeProvider(
            tokenizer=FakeTargetTokenizer(),
            max_length=513,
        )


def test_named_et2_feature_subsets_are_validated_and_canonicalized():
    assert ET2_FEATURE_NAMES == ("nFix", "FFD", "GPT", "TRT", "fixProp")
    assert et2_feature_indices_from_names(["TRT", "nFix"]) == (0, 3)
    assert et2_feature_names_from_indices([3, 0]) == ("nFix", "TRT")
    assert et2_features_used_from_indices([0, 3]) == (1, 0, 0, 1, 0)
    assert normalize_et2_feature_indices() == (3,)

    with pytest.raises(ValueError, match="duplicates"):
        et2_feature_indices_from_names(["TRT", "TRT"])
    with pytest.raises(ValueError, match="Unknown"):
        et2_feature_indices_from_names(["TRT", "unknown"])
    with pytest.raises(ValueError, match="duplicates"):
        normalize_et2_feature_indices([3, 3])


def test_et2_asset_loader_preserves_pinned_identity_and_prefix_space(
    monkeypatch,
):
    calls = {}
    fake_tokenizer = object()

    class FakeModel:
        def load_state_dict(self, state_dict, strict):
            calls["state_dict"] = state_dict
            calls["strict"] = strict

    def fake_tokenizer_loader(repo_id, **kwargs):
        calls["tokenizer"] = (repo_id, kwargs)
        return fake_tokenizer

    def fake_hub_download(**kwargs):
        calls["download"] = kwargs
        return "/tmp/fake-et2.safetensors"

    def fake_load_file(path, device):
        calls["load_file"] = (path, device)
        return {"decoder.weight": torch.ones(1)}

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        fake_tokenizer_loader,
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        fake_hub_download,
    )
    monkeypatch.setattr("safetensors.torch.load_file", fake_load_file)
    monkeypatch.setattr(gaze_module, "_ET2RegressionModel", FakeModel)
    provider = ET2GazeProvider(
        tokenizer=FakeTargetTokenizer(),
        repo_id="skboy/et_prediction_2",
        revision="fixed-commit",
        filename="et_predictor2_seed123.safetensors",
    )

    tokenizer, model = provider._load_assets()

    assert tokenizer is fake_tokenizer
    assert isinstance(model, FakeModel)
    assert calls["tokenizer"] == (
        "skboy/et_prediction_2",
        {
            "revision": "fixed-commit",
            "use_fast": True,
            "trust_remote_code": False,
            "add_prefix_space": True,
        },
    )
    assert calls["download"] == {
        "repo_id": "skboy/et_prediction_2",
        "filename": "et_predictor2_seed123.safetensors",
        "revision": "fixed-commit",
    }
    assert calls["load_file"] == ("/tmp/fake-et2.safetensors", "cpu")
    assert calls["state_dict"]["decoder.weight"].tolist() == [1.0]
    assert calls["strict"] is True


def test_exact_alignment_does_not_consume_tokens_after_mismatch():
    tokenizer = FakeTargetTokenizer()
    alignment = align_words_to_tokens(
        ["absent", "Hello", "world"],
        [0, 10, 11, 12, 2],
        [1, 1, 1, 1, 1],
        tokenizer,
    )

    assert alignment.word_to_token_indices == ((), (1, 2), (3,))
    assert alignment.first_subword_mask == (False, True, False, True, False)


def test_segmenter_handles_whitespace_cjk_and_punctuation_together():
    assert segment_text_for_et2("Hello, 世界! 안녕.") == [
        "Hello",
        ",",
        "世",
        "界",
        "!",
        "안",
        "녕",
        ".",
    ]


def test_provider_is_lazy_frozen_batched_cached_and_first_subword_aligned():
    tokenizer = FakeTargetTokenizer()
    provider = FakeET2GazeProvider(tokenizer)
    input_ids = torch.tensor(
        [
            [0, 10, 11, 12, 13, 2],
            [0, 10, 11, 12, 13, 2],
            [0, 14, 2, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    assert not provider.is_loaded
    features, mapped_mask = provider.compute(input_ids, attention_mask)

    assert provider.is_loaded
    assert provider.load_count == 1
    assert provider.fake_model.batch_sizes == [2]
    assert not provider.fake_model.training
    assert all(
        not parameter.requires_grad
        for parameter in provider.fake_model.parameters()
    )
    assert features.shape == (3, 6, 1)
    assert mapped_mask.dtype == torch.bool
    assert mapped_mask[0].tolist() == [False, True, False, True, True, False]
    assert features[0, :, 0].tolist() == [0.0, 10.0, 0.0, 13.0, 16.0, 0.0]
    assert torch.equal(features[0], features[1])

    cached_features, cached_mask = provider.compute(input_ids, attention_mask)
    assert provider.fake_model.batch_sizes == [2]
    assert torch.equal(cached_features, features)
    assert torch.equal(cached_mask, mapped_mask)
    assert all(cache_value[0].device.type == "cpu" for cache_value in provider._cache.values())
    assert all(cache_value[1].device.type == "cpu" for cache_value in provider._cache.values())


def test_provider_returns_every_selected_channel_in_canonical_order():
    provider = FakeET2GazeProvider(
        FakeTargetTokenizer(),
        feature_indices=(4, 0, 3),
    )
    input_ids = torch.tensor([[0, 10, 11, 12, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    features, mapped_mask = provider.compute(input_ids, attention_mask)

    assert provider.feature_indices == (0, 3, 4)
    assert provider.feature_names == ("nFix", "TRT", "fixProp")
    assert provider.features_used == (1, 0, 0, 1, 1)
    assert provider.feature_index is None
    assert features.shape == (1, 5, 3)
    assert mapped_mask.tolist() == [[False, True, False, True, False]]
    assert features[0, 1].tolist() == [10.0, 10.0, 410.0]
    assert features[0, 3].tolist() == [13.0, 13.0, 413.0]


def test_provider_rejects_non_contiguous_attention_masks():
    provider = FakeET2GazeProvider(FakeTargetTokenizer())

    with pytest.raises(ValueError, match="contiguous right padding"):
        provider.compute(
            torch.tensor([[0, 10, 11]], dtype=torch.long),
            torch.tensor([[1, 0, 1]], dtype=torch.long),
        )

    assert provider.load_count == 0


def test_prefix_packing_compacts_per_sample_and_pools_last_text():
    text_embeddings = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [99.0, 99.0]],
            [[4.0, 4.0], [5.0, 5.0], [98.0, 98.0], [97.0, 97.0]],
        ]
    )
    text_attention_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 0, 0]],
        dtype=torch.long,
    )
    gaze_embeddings = torch.tensor(
        [
            [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [19.0, 19.0]],
            [[20.0, 20.0], [21.0, 21.0], [29.0, 29.0], [28.0, 28.0]],
        ]
    )
    gaze_mask = torch.tensor(
        [[0, 1, 1, 0], [1, 0, 0, 0]],
        dtype=torch.bool,
    )

    packed = pack_prefix_gaze(
        text_embeddings,
        text_attention_mask,
        gaze_embeddings,
        gaze_mask,
        eye_start=torch.tensor([-1.0, -1.0]),
        eye_end=torch.tensor([-2.0, -2.0]),
    )

    assert packed.inputs_embeds.shape == (2, 7, 2)
    assert packed.inputs_embeds[0].tolist() == [
        [-1.0, -1.0],
        [11.0, 11.0],
        [12.0, 12.0],
        [-2.0, -2.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]
    assert packed.inputs_embeds[1].tolist() == [
        [-1.0, -1.0],
        [20.0, 20.0],
        [-2.0, -2.0],
        [4.0, 4.0],
        [5.0, 5.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    assert packed.attention_mask.tolist() == [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0],
    ]
    assert packed.position_ids.tolist() == [
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 0, 0],
    ]
    assert packed.pooling_positions.tolist() == [6, 4]
    assert torch.equal(
        packed.inputs_embeds[
            torch.arange(2),
            packed.pooling_positions,
        ],
        torch.tensor([[3.0, 3.0], [5.0, 5.0]]),
    )


def test_prefix_packing_preserves_singleton_gaze_dimension():
    packed = pack_prefix_gaze(
        text_embeddings=torch.ones(1, 1, 3),
        text_attention_mask=torch.ones(1, 1, dtype=torch.long),
        gaze_embeddings=torch.full((1, 1, 3), 2.0),
        gaze_mask=torch.ones(1, 1, dtype=torch.bool),
        eye_start=torch.full((3,), 3.0),
        eye_end=torch.full((3,), 4.0),
    )

    assert packed.inputs_embeds.shape == (1, 4, 3)
    assert packed.inputs_embeds[0, :, 0].tolist() == [3.0, 2.0, 4.0, 1.0]
    assert packed.pooling_positions.tolist() == [3]


def test_prefix_packing_keeps_boundaries_when_no_gaze_tokens_are_mapped():
    packed = pack_prefix_gaze(
        text_embeddings=torch.tensor([[[1.0], [2.0]]]),
        text_attention_mask=torch.ones(1, 2, dtype=torch.long),
        gaze_embeddings=torch.tensor([[[10.0], [20.0]]]),
        gaze_mask=torch.zeros(1, 2, dtype=torch.bool),
        eye_start=torch.tensor([3.0]),
        eye_end=torch.tensor([4.0]),
    )

    assert packed.inputs_embeds[0, :, 0].tolist() == [3.0, 4.0, 1.0, 2.0]
    assert packed.attention_mask.tolist() == [[1, 1, 1, 1]]
    assert packed.pooling_positions.tolist() == [3]


def test_prefix_packing_rejects_gaze_on_text_padding():
    with pytest.raises(ValueError, match="padded text position"):
        pack_prefix_gaze(
            text_embeddings=torch.ones(1, 2, 3),
            text_attention_mask=torch.tensor([[1, 0]]),
            gaze_embeddings=torch.ones(1, 2, 3),
            gaze_mask=torch.tensor([[0, 1]]),
            eye_start=torch.ones(3),
            eye_end=torch.ones(3),
        )


@pytest.mark.parametrize(
    "attention_mask",
    (
        torch.tensor([[0, 1, 1]]),
        torch.tensor([[1, 0, 1]]),
    ),
)
def test_prefix_packing_rejects_non_right_padded_text(attention_mask):
    with pytest.raises(ValueError, match="contiguous right padding"):
        pack_prefix_gaze(
            text_embeddings=torch.ones(1, 3, 2),
            text_attention_mask=attention_mask,
            gaze_embeddings=torch.ones(1, 3, 2),
            gaze_mask=torch.zeros(1, 3, dtype=torch.bool),
            eye_start=torch.ones(2),
            eye_end=torch.ones(2),
        )


def test_prefix_packing_rejects_non_binary_gaze_mask():
    with pytest.raises(ValueError, match="gaze_mask"):
        pack_prefix_gaze(
            text_embeddings=torch.ones(1, 1, 2),
            text_attention_mask=torch.ones(1, 1, dtype=torch.long),
            gaze_embeddings=torch.ones(1, 1, 2),
            gaze_mask=torch.tensor([[2]]),
            eye_start=torch.ones(2),
            eye_end=torch.ones(2),
        )

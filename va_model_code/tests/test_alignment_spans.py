"""Occurrence and feature-conservation regression tests for gaze alignment."""

from __future__ import annotations

import pytest
import torch

from va_model_code.decoder_va.alignment import (
    align_words_to_tokens,
    remap_word_features_to_tokens,
)
from va_model_code.decoder_va.gaze import ET2GazeProvider, segment_text_for_et2


class OffsetTokenizerFixture:
    """Expose fixed character offsets without implementing the alignment algorithm."""

    all_special_ids = [90, 91, 92]
    all_special_tokens = ["<bos>", "<eos>", "<pad>"]
    is_fast = True

    def __init__(self, text, offsets, *, reencoded_ids=None):
        """Record an independent text/offset fixture and optional roundtrip defect."""

        self.text = text
        self.offsets = list(offsets)
        self.ids = list(range(len(offsets)))
        self.reencoded_ids = (
            list(self.ids) if reencoded_ids is None else list(reencoded_ids)
        )
        self.pieces = {
            index: text[start:end]
            for index, (start, end) in enumerate(self.offsets)
        }
        self.pieces.update(dict(zip(self.all_special_ids, self.all_special_tokens)))

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        """Decode the recorded lexical sequence with optional surrounding specials."""

        active = [int(value) for value in ids if int(value) not in self.all_special_ids]
        if active != self.ids:
            raise ValueError("Fixture decode requires the recorded lexical IDs.")
        return self.text

    def __call__(self, text, **kwargs):
        """Return the fixture's offsets only for its exact source string."""

        if text != self.text:
            raise ValueError("Tokenizer must receive the exact decoded source text.")
        return {
            "input_ids": list(self.reencoded_ids),
            "attention_mask": [1] * len(self.reencoded_ids),
            "offset_mapping": list(self.offsets),
        }

    def convert_ids_to_tokens(self, ids):
        """Resolve fixture IDs for tokenizer compatibility checks."""

        if isinstance(ids, int):
            return self.pieces[ids]
        return [self.pieces[int(value)] for value in ids]

    def convert_tokens_to_string(self, tokens):
        """Support legacy tokenizers without stripping or normalizing characters."""

        return "".join(tokens)


def punctuation_fixture():
    """Reproduce the repeated punctuation sentence that triggered the audit."""

    return OffsetTokenizerFixture(
        "I love this!!! I hate that!",
        [(0, 1), (1, 6), (6, 11), (11, 14), (14, 16), (16, 21), (21, 26), (26, 27)],
    )


def test_repeated_punctuation_preserves_each_source_occurrence():
    """Merged punctuation cannot steal the final matching punctuation token."""

    tokenizer = punctuation_fixture()
    words = segment_text_for_et2(tokenizer.text)

    result = align_words_to_tokens(words, tokenizer.ids, [1] * 8, tokenizer)

    assert result.word_to_token_indices == (
        (0,), (1,), (2,), (3,), (3,), (3,), (4,), (5,), (6,), (7,),
    )
    assert result.first_subword_mask == (True,) * 8


def test_provider_sums_all_merged_trt_and_keeps_final_punctuation_identity():
    """TRT conservation requires adding collisions instead of overwriting them."""

    tokenizer = punctuation_fixture()
    words = segment_text_for_et2(tokenizer.text)
    provider = ET2GazeProvider(tokenizer=tokenizer)
    word_features = torch.arange(1, 11, dtype=torch.float32).unsqueeze(-1)

    output, mask = provider._map_predictions_to_target(
        words, word_features, torch.ones(10, dtype=torch.bool), tokenizer.ids,
    )

    torch.testing.assert_close(
        output[:, 0], torch.tensor([1.0, 2.0, 3.0, 15.0, 7.0, 8.0, 9.0, 10.0]),
    )
    assert mask.tolist() == [True] * 8
    assert output.sum().item() == word_features.sum().item()


def test_contraction_shared_token_retains_later_repeated_word():
    """An apostrophe and its suffix may belong to one target token."""

    tokenizer = OffsetTokenizerFixture(
        "It's here. It isn't!",
        [(0, 2), (2, 4), (4, 9), (9, 10), (10, 13), (13, 17), (17, 19), (19, 20)],
    )
    words = segment_text_for_et2(tokenizer.text)

    result = align_words_to_tokens(words, tokenizer.ids, [1] * 8, tokenizer)

    assert words == ["It", "'", "s", "here", ".", "It", "isn", "'", "t", "!"]
    assert result.word_to_token_indices == (
        (0,), (1,), (1,), (2,), (3,), (4,), (5,), (6,), (6,), (7,),
    )


@pytest.mark.parametrize("excluded", ["masked", "nonfinite"])
def test_collision_aggregation_uses_only_finite_valid_words(excluded):
    """Durations/counts sum while nonadditive feature means use valid rows only."""

    tokenizer = OffsetTokenizerFixture("!!! a", [(0, 3), (3, 5)])
    provider = ET2GazeProvider(tokenizer=tokenizer, feature_indices=(0, 1, 2, 3, 4))
    words = segment_text_for_et2(tokenizer.text)
    features = torch.tensor(
        [[1, 10, 100, 1000, 0.1], [2, 20, 200, 2000, 0.2],
         [3, 30, 300, 3000, 0.3], [4, 40, 400, 4000, 0.4]],
        dtype=torch.float32,
    )
    valid = torch.ones(4, dtype=torch.bool)
    if excluded == "masked":
        valid[1] = False
    else:
        features[1, 3] = float("nan")

    output, mask = provider._map_predictions_to_target(words, features, valid, tokenizer.ids)

    torch.testing.assert_close(
        output, torch.tensor([[4, 20, 200, 4000, 0.2], [4, 40, 400, 4000, 0.4]]),
    )
    assert mask.tolist() == [True, True]


def test_no_valid_collisions_leave_position_unmapped():
    """An invalid source occurrence must not create a redistribution destination."""

    tokenizer = OffsetTokenizerFixture("!!! a", [(0, 3), (3, 5)])
    provider = ET2GazeProvider(tokenizer=tokenizer)

    output, mask = provider._map_predictions_to_target(
        segment_text_for_et2(tokenizer.text), torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        torch.tensor([False, False, False, True]), tokenizer.ids,
    )

    torch.testing.assert_close(output[:, 0], torch.tensor([0.0, 4.0]))
    assert mask.tolist() == [False, True]


def test_cjk_segments_share_one_target_token_without_dropping_mass():
    """The actual Qwen vocabulary can represent multiple CJK segments in one token."""

    tokenizer = OffsetTokenizerFixture("你好!", [(0, 2), (2, 3)])
    words = segment_text_for_et2(tokenizer.text)

    output, mask = remap_word_features_to_tokens(
        [[2.0], [3.0], [7.0]], words, tokenizer.ids, [1, 1], tokenizer, 1,
    )

    torch.testing.assert_close(output[:, 0], torch.tensor([5.0, 7.0]))
    assert mask.tolist() == [True, True]


def test_generic_mapping_supports_explicit_mean_channels_and_mask():
    """Generic remapping and provider remapping must share collision semantics."""

    tokenizer = OffsetTokenizerFixture("!!!", [(0, 3)])

    output, mask = remap_word_features_to_tokens(
        [[2.0, 10.0], [100.0, 1000.0], [3.0, 30.0]], ["!", "!", "!"],
        tokenizer.ids, [1], tokenizer, 2,
        word_feature_mask=torch.tensor([True, False, True]), sum_feature_indices=(0,),
    )

    torch.testing.assert_close(output, torch.tensor([[5.0, 20.0]]))
    assert mask.tolist() == [True]


@pytest.mark.parametrize(
    ("text", "offsets", "expected"),
    [
        ("é café", [(0, 1), (1, 6)], ((0,), (1,))),
        ("e\u0301 e\u0301", [(0, 1), (1, 2), (2, 4), (4, 5)], ((0, 1), (2, 3))),
        ("Ａ A", [(0, 1), (1, 3)], ((0,), (1,))),
        ("x\u00a0x", [(0, 1), (1, 2), (2, 3)], ((0,), (2,))),
    ],
)
def test_unicode_source_coordinates_preserve_occurrence_identity(text, offsets, expected):
    """Source spans are based on decoded Unicode, without NFKC conflation."""

    tokenizer = OffsetTokenizerFixture(text, offsets)

    result = align_words_to_tokens(
        segment_text_for_et2(text), tokenizer.ids, [1] * len(offsets), tokenizer,
    )

    assert result.word_to_token_indices == expected


def test_overlapping_byte_token_offsets_choose_first_token_once_per_segment():
    """Several byte-level tokens may cover one Unicode character."""

    tokenizer = OffsetTokenizerFixture(
        "👩\u200d🔬 x",
        [(0, 1), (0, 1), (0, 1), (1, 2), (1, 2), (2, 3), (2, 3), (2, 3), (3, 5)],
    )
    words = segment_text_for_et2(tokenizer.text)

    result = align_words_to_tokens(words, tokenizer.ids, [1] * 9, tokenizer)

    assert result.word_to_token_indices == ((0, 1, 2), (3, 4), (5, 6, 7), (8,))
    output, mask = remap_word_features_to_tokens(
        [[1.0], [2.0], [3.0], [4.0]], words, tokenizer.ids, [1] * 9, tokenizer, 1,
    )
    assert mask.tolist() == [True, False, False, True, False, True, False, False, True]
    assert output.sum().item() == 10.0


def test_whitespace_and_special_tokens_cannot_anchor_gaze():
    """Keep original token indices while excluding padding and lexical whitespace."""

    tokenizer = OffsetTokenizerFixture(" x", [(0, 1), (1, 2)])

    result = align_words_to_tokens(["x"], [90, 0, 1, 91, 92], [1, 1, 1, 1, 0], tokenizer)

    assert result.word_to_token_indices == ((2,),)
    assert result.first_subword_mask == (False, False, True, False, False)


def test_alignment_preserves_target_indices_across_inactive_positions():
    """Filtering inactive input IDs must not compact the redistribution coordinates."""

    tokenizer = OffsetTokenizerFixture("a b", [(0, 1), (1, 3)])

    result = align_words_to_tokens(["a", "b"], [0, 999, 1], [1, 0, 1], tokenizer)

    assert result.word_to_token_indices == ((0,), (2,))
    assert result.first_subword_mask == (True, False, True)


@pytest.mark.parametrize("words", [["absent", "x"], ["x", "x"], ["X"]])
def test_source_segmentation_mismatch_raises_instead_of_searching_later(words):
    """Unprovable occurrence identity must fail closed, including repeated words."""

    tokenizer = OffsetTokenizerFixture("x", [(0, 1)])

    with pytest.raises(ValueError):
        align_words_to_tokens(words, tokenizer.ids, [1], tokenizer)


def test_incomplete_utf8_roundtrip_is_rejected_before_offsets_are_used():
    """A truncated byte token decoded as U+FFFD cannot inherit replacement-token offsets."""

    tokenizer = OffsetTokenizerFixture("�", [(0, 1)], reencoded_ids=[42])

    with pytest.raises(ValueError):
        align_words_to_tokens(["�"], tokenizer.ids, [1], tokenizer)


def test_special_token_removal_that_changes_tokenization_is_rejected():
    """Removing an internal special token may merge formerly distinct lexical tokens."""

    tokenizer = OffsetTokenizerFixture("aa", [(0, 1), (1, 2)], reencoded_ids=[42])

    with pytest.raises(ValueError):
        align_words_to_tokens(["aa"], [0, 90, 1], [1, 1, 1], tokenizer)


def test_explicit_text_must_match_target_decode():
    """Offsets from unrelated text must not be accepted just because IDs are supplied."""

    tokenizer = OffsetTokenizerFixture("x", [(0, 1)])

    with pytest.raises(ValueError):
        align_words_to_tokens(["y"], tokenizer.ids, [1], tokenizer, text="y")


@pytest.mark.parametrize("offset", [(-1, 1), (1, 0), (0, 2), (0, 0)])
def test_unusable_offsets_raise_instead_of_fabricating_positions(offset):
    """Bounds and visible-character coverage must be proven before assigning gaze."""

    tokenizer = OffsetTokenizerFixture("x", [offset])

    with pytest.raises(ValueError):
        align_words_to_tokens(["x"], tokenizer.ids, [1], tokenizer)


def test_whitespace_only_text_does_not_create_a_gaze_position():
    """A lexical whitespace token is valid text input but has no ET source segment."""

    tokenizer = OffsetTokenizerFixture(" \t\n", [(0, 3)])

    result = align_words_to_tokens([], tokenizer.ids, [1], tokenizer)

    assert result.word_to_token_indices == ()
    assert result.first_subword_mask == (False,)


def test_signed_trt_is_preserved_when_segments_share_a_token():
    """Unconstrained ET predictions must not be silently clamped during alignment."""

    tokenizer = OffsetTokenizerFixture("!!!", [(0, 3)])
    provider = ET2GazeProvider(tokenizer=tokenizer)

    output, mask = provider._map_predictions_to_target(
        ["!", "!", "!"], torch.tensor([[-3.0], [2.0], [-4.0]]),
        torch.ones(3, dtype=torch.bool), tokenizer.ids,
    )

    assert output[0, 0].item() == -5.0
    assert mask.tolist() == [True]


@pytest.mark.parametrize("text,offsets", [("ab", [(0, 1)]), ("abc", [(0, 1), (2, 3)])])
def test_partial_word_coverage_is_not_treated_as_exact_alignment(text, offsets):
    """Having one overlapping token does not establish full source-span coverage."""

    tokenizer = OffsetTokenizerFixture(text, offsets)

    with pytest.raises(ValueError):
        align_words_to_tokens([text], tokenizer.ids, [1] * len(offsets), tokenizer)

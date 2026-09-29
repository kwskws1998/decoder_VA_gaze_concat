"""Regression cases discovered by the audit, now required to pass after the fix."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from va_model_code.decoder_va.alignment import align_words_to_tokens
from va_model_code.decoder_va.gaze import ET2GazeProvider, segment_text_for_et2


class PinnedTokenizationFixture:
    """Replay the token strings observed from the pinned Qwen3.5 tokenizer."""

    all_special_ids = []
    all_special_tokens = []
    pieces = ['I', 'Ġlove', 'Ġthis', '!!!', 'ĠI', 'Ġhate', 'Ġthat', '!']

    def convert_ids_to_tokens(self, ids):
        """Resolve local fixture indices, which are not the model's vocabulary IDs."""

        return [self.pieces[i] for i in ids]

    def convert_tokens_to_string(self, tokens):
        """Decode the ASCII token strings in this fixed regression example."""

        return ''.join(tokens).replace('Ġ', ' ')

    def decode(self, ids, **kwargs):
        """Decode the exact tokenized sentence used by the original reproduction."""

        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

    def __call__(self, text, **kwargs):
        """Return the independently recorded offsets from the pinned Qwen tokenizer."""

        assert text == 'I love this!!! I hate that!'
        return {'input_ids': list(range(8)), 'offset_mapping': [(0, 1), (1, 6), (6, 11), (11, 14), (14, 16), (16, 21), (21, 26), (26, 27)]}


def test_unmatched_merged_punctuation_does_not_steal_later_occurrence():
    """Require later words to retain gaze after punctuation tokenization differs."""

    tokenizer = PinnedTokenizationFixture()
    words = segment_text_for_et2('I love this!!! I hate that!')
    alignment = align_words_to_tokens(words, range(8), [1] * 8, tokenizer)
    assert alignment.word_to_token_indices[3] != (7,)
    assert alignment.word_to_token_indices[6:9] == ((4,), (5,), (6,))


def test_provider_preserves_the_identity_of_the_final_punctuation_feature():
    """Trace distinguishable synthetic ET features through the actual provider mapping."""

    tokenizer = PinnedTokenizationFixture()
    words = segment_text_for_et2('I love this!!! I hate that!')
    provider = ET2GazeProvider(tokenizer=tokenizer)
    features, mask = provider._map_predictions_to_target(
        words, torch.arange(1, 11, dtype=torch.float32).unsqueeze(-1),
        torch.ones(10, dtype=torch.bool), list(range(8)),
    )
    assert features[7, 0].item() == 10.0
    assert mask[4:7].all()

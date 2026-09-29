"""Validate fixed ET-to-Qwen mapping against independent character-offset evidence.

Only tokenizer assets and archived prediction text are loaded. Synthetic uniquely
numbered TRT values exercise the actual ET2 provider without downloading ET weights
or relying on trained model predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from va_model_code.decoder_va.alignment import align_words_to_tokens
from va_model_code.decoder_va.gaze import ET2GazeProvider, segment_text_for_et2
from va_model_code.decoder_va.redistribution import AsymGaussianRedistributor


def file_hash(path: Path) -> str:
    """Record the exact input and production-source files used for this validation."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_spans(text: str, words: list[str]) -> list[tuple[int, int]]:
    """Locate source occurrences and require every skipped character to be whitespace."""

    cursor = 0
    spans = []
    for word in words:
        start = text.index(word, cursor)
        assert all(character.isspace() for character in text[cursor:start])
        end = start + len(word)
        spans.append((start, end))
        cursor = end
    assert all(character.isspace() for character in text[cursor:])
    return spans


def independent_mapping(tokenizer, text: str, ids: list[int], words: list[str]):
    """Use the tokenizer's raw offset metadata, never production alignment helpers."""

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == ids, 'Decoded text changed the target tokenization.'
    offsets = encoded['offset_mapping']
    special_ids = set(tokenizer.all_special_ids)
    mapping = []
    for start, end in independent_spans(text, words):
        overlaps = tuple(
            index for index, (a, b) in enumerate(offsets)
            if ids[index] not in special_ids
            and a < end and b > start
            and any(not character.isspace() for character in text[max(a, start):min(b, end)])
        )
        assert overlaps, f'Source segment has no visible target token: {text[start:end]!r}.'
        mapping.append(overlaps)
    return tuple(mapping), offsets


def provider_checks(provider, words, ids, expected_mapping, filtered=False):
    """Check feature identity, collision sums, masks, and exact synthetic TRT conservation."""

    source = torch.arange(1, len(words) + 1, dtype=torch.float32).unsqueeze(-1)
    source_mask = torch.ones(len(words), dtype=torch.bool)
    if filtered:
        source_mask[::7] = False
        source[::11] = float('nan')
    expected = torch.zeros(len(ids), 1, dtype=torch.float32)
    expected_mask = torch.zeros(len(ids), dtype=torch.bool)
    included_total = 0.0
    for word_index, indices in enumerate(expected_mapping):
        if not source_mask[word_index] or not bool(torch.isfinite(source[word_index]).all()):
            continue
        expected[indices[0]] += source[word_index]
        expected_mask[indices[0]] = True
        included_total += float(source[word_index, 0])
    actual, actual_mask = provider._map_predictions_to_target(words, source, source_mask, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_mask, expected_mask)
    assert float(actual.sum()) == included_total
    assert bool((actual[~actual_mask] == 0).all())
    return actual, actual_mask


def kernel_checks(mapped, mask):
    """Check corrected mappings through both kernel directions and CPU BF16 autocast."""

    values = mapped.T
    records = []
    for left, right in ((0.5, 2.0), (2.0, 0.5)):
        kernel = AsymGaussianRedistributor(left, right)
        for use_autocast in (False, True):
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=use_autocast):
                output = kernel(values, mask.unsqueeze(0))
            assert bool(torch.isfinite(output).all())
            assert bool((output[:, ~mask] == 0).all())
            torch.testing.assert_close(output.sum(), values.sum(), rtol=3e-6, atol=2e-4)
            records.append({
                'sigma_left': left,
                'sigma_right': right,
                'cpu_bf16_autocast': use_autocast,
                'sum_abs_error': abs(float((output.sum() - values.sum()).detach())),
                'output_dtype': str(output.dtype),
            })
    return records


def mixed_feature_checks(tokenizer, words, ids, expected_mapping):
    """Independently check additive and mean channels at shared-token destinations."""

    provider = ET2GazeProvider(tokenizer=tokenizer, feature_indices=(0, 1, 2, 3, 4), cache_size=0)
    features = torch.arange(1, len(words) + 1, dtype=torch.float32).unsqueeze(1) * torch.arange(1, 6, dtype=torch.float32)
    source_mask = torch.ones(len(words), dtype=torch.bool)
    source_mask[::7] = False
    features[::11, 1] = float('nan')
    valid_sources = {}
    for index, indices in enumerate(expected_mapping):
        if source_mask[index] and bool(torch.isfinite(features[index]).all()):
            valid_sources.setdefault(indices[0], []).append(index)
    expected = torch.zeros(len(ids), 5, dtype=torch.float32)
    expected_mask = torch.zeros(len(ids), dtype=torch.bool)
    for destination, sources in valid_sources.items():
        for channel in range(5):
            total = sum(float(features[source, channel]) for source in sources)
            expected[destination, channel] = total if channel in (0, 3) else total / len(sources)
        expected_mask[destination] = True
    actual, actual_mask = provider._map_predictions_to_target(words, features, source_mask, ids)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-6)
    assert torch.equal(actual_mask, expected_mask)
    assert not provider.is_loaded
    return {'all_five_channels_checked': True, 'invalid_features_filtered_before_aggregation': True}


def validate_row(tokenizer, provider, original: str, sample_kernel=False):
    """Validate the same decode-after-200-token-truncation path used by training."""

    ids = tokenizer(original, max_length=200, truncation=True)['input_ids']
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    words = segment_text_for_et2(text)
    expected, offsets = independent_mapping(tokenizer, text, ids, words)
    actual = align_words_to_tokens(words, ids, [1] * len(ids), tokenizer)
    assert actual.word_to_token_indices == expected, {
        'text': text, 'expected': expected, 'actual': actual.word_to_token_indices,
    }
    destinations = [indices[0] for indices in expected]
    expected_mask = tuple(index in destinations for index in range(len(ids)))
    assert actual.first_subword_mask == expected_mask
    for index in destinations:
        start, end = offsets[index]
        assert text[start:end].strip()
        assert ids[index] not in tokenizer.all_special_ids
    mapped, mask = provider_checks(provider, words, ids, expected)
    provider_checks(provider, words, ids, expected, filtered=True)
    collision_counts = Counter(destinations)
    counts = Counter({
        'rows': 1,
        'source_segments': len(words),
        'mapped_segments': sum(bool(indices) for indices in actual.word_to_token_indices),
        'missing_segments': sum(not indices for indices in actual.word_to_token_indices),
        'wrong_occurrence_segments': 0,
        'wrong_occurrence_rows': 0,
        'rows_with_missing_segments': 0,
        'whitespace_destinations': 0,
        'special_destinations': 0,
        'collision_rows': int(any(count > 1 for count in collision_counts.values())),
        'collision_destinations': sum(count > 1 for count in collision_counts.values()),
        'collision_extra_segments': sum(count - 1 for count in collision_counts.values()),
        'unique_gaze_positions': len(collision_counts),
        'rows_with_no_gaze_position': int(not collision_counts),
        'provider_identity_checks': 2,
        'exact_trt_sum_checks': 2,
    })
    return counts, {
        'text': text,
        'words': words,
        'target_tokens': tokenizer.convert_ids_to_tokens(ids),
        'mapping': expected,
        'synthetic_mapped_trt': mapped.flatten().tolist(),
        'kernel_checks': kernel_checks(mapped, mask) if sample_kernel and ids else [],
        'mixed_feature_checks': mixed_feature_checks(tokenizer, words, ids, expected) if sample_kernel else None,
    }


def main():
    """Write fixed-pipeline evidence separately, preserving the original defect audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-path', type=Path, default=Path('/private/tmp/decoder-va-audit-tokenizer'))
    parser.add_argument('--results-zip', type=Path, default=Path('/Users/wansookim/Downloads/qwen3.5-0.8b_full_gaze_TRT_redistribution_asym-gaussian_sentence_only_no_iemocap_seed42_results_only (1).zip'))
    args = parser.parse_args()
    from transformers import AutoTokenizer

    baseline_path = ROOT / 'diagnostics/redistribution_audit_20260922/alignment_results.json'
    baseline = json.loads(baseline_path.read_text())['alignment']
    assert file_hash(args.results_zip) == baseline['results_zip_sha256']
    assert file_hash(args.tokenizer_path / 'tokenizer.json') == baseline['tokenizer_json_sha256']
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    provider = ET2GazeProvider(tokenizer=tokenizer, feature_indices=(3,), cache_size=0)
    sources = [ROOT / 'va_model_code/decoder_va' / name for name in ('alignment.py', 'gaze.py', 'redistribution.py')]
    source_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    constructed = []
    for text in (
        'I love this!!! I hate that!', "I don't like this.", "I'm sad, but it's okay.",
        'I am sad... Really sad.', 'The price is $12.50.', 'Hello?! How are you?',
        'U.S. and U.S. citizens.', 'P.S. I love you. P.S. hello!',
        '  Hello\tworld!\n\n Goodbye.', '你好，世界！こんにちは。안녕하세요!',
        'Cafe\u0301 café ½ ＡＢＣ 😀!!!',
    ):
        counts, details = validate_row(tokenizer, provider, text, sample_kernel=True)
        constructed.append({'counts': dict(counts), **details})
    with zipfile.ZipFile(args.results_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith('/oof_predictions.tsv')]
        assert len(names) == 1
        raw_tsv = archive.read(names[0])
        rows = list(csv.DictReader(io.StringIO(raw_tsv.decode()), delimiter='\t'))
    assert len(rows) == baseline['summary']['rows'] == 14352
    summary = Counter()
    datasets = {}
    folds = {}
    examples = []
    empty_text_rows = []
    started = time.monotonic()
    for row_index, row in enumerate(rows):
        counts, details = validate_row(tokenizer, provider, row['text'], sample_kernel=row_index % 500 == 0)
        summary.update(counts)
        datasets.setdefault(row['dataset_of_origin'], Counter()).update(counts)
        folds.setdefault(row['held_out_fold'], Counter()).update(counts)
        if counts['rows_with_no_gaze_position']:
            assert not details['text'].strip()
            empty_text_rows.append({'row_index': row_index, 'index': row['index'], 'dataset': row['dataset_of_origin'], 'fold': row['held_out_fold'], 'text': details['text']})
        if details['kernel_checks'] or counts['collision_rows'] and len(examples) < 20:
            examples.append({'row_index': row_index, 'index': row['index'], 'dataset': row['dataset_of_origin'], 'fold': row['held_out_fold'], **details})
        if (row_index + 1) % 1000 == 0:
            print(f'Validated {row_index + 1}/{len(rows)} rows; {summary["source_segments"]} segments.', flush=True)
    assert not provider.is_loaded
    assert source_hashes == {str(path.relative_to(ROOT)): file_hash(path) for path in sources}, 'Production files changed during validation; rerun.'
    output = {
        'all_assertions_passed': True,
        'elapsed_seconds': time.monotonic() - started,
        'torch_version': torch.__version__,
        'tokenizer_path': str(args.tokenizer_path),
        'tokenizer_json_sha256': file_hash(args.tokenizer_path / 'tokenizer.json'),
        'results_zip': str(args.results_zip),
        'results_zip_sha256': file_hash(args.results_zip),
        'prediction_tsv_sha256': hashlib.sha256(raw_tsv).hexdigest(),
        'source_sha256': source_hashes,
        'validation_script_sha256': file_hash(Path(__file__)),
        'baseline_audit_sha256': file_hash(baseline_path),
        'baseline_summary': baseline['summary'],
        'summary': dict(summary),
        'by_dataset': {name: dict(counts) for name, counts in datasets.items()},
        'by_fold': {name: dict(counts) for name, counts in folds.items()},
        'empty_text_rows': empty_text_rows,
        'constructed_examples': constructed,
        'corpus_examples': examples,
        'et_model_loaded': provider.is_loaded,
        'scope': 'Alignment and mapping correctness with synthetic features; no ET2 prediction coverage, learned checkpoints, or task-performance claims.',
    }
    output_path = Path(__file__).with_name('results.json')
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(output['summary'], indent=2), flush=True)
    print(output_path)


if __name__ == '__main__':
    main()

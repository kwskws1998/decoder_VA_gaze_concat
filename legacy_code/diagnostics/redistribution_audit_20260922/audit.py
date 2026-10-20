"""Independently audit redistribution arithmetic, Qwen gradients and token alignment."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from va_model_code.decoder_va.alignment import align_words_to_tokens
from va_model_code.decoder_va.gaze import ET2GazeProvider, segment_text_for_et2
from va_model_code.decoder_va.model import DecoderVARegressor
from va_model_code.decoder_va.redistribution import AsymGaussianRedistributor, redistribution_contract
from va_model_code.tests.test_redistribution_transformer import FixedET2


def oracle(values, mask, logs):
    """Evaluate source-to-destination transfer in float64 using scalar loops."""

    result = np.zeros_like(values, dtype=np.float64)
    sigmas = np.exp(np.asarray(logs, dtype=np.float64)) + 1e-6
    for b in range(values.shape[0]):
        valid = np.flatnonzero(mask[b])
        for source in valid:
            mass = np.array([
                math.exp(-0.5 * ((target - source) / sigmas[int(target >= source)]) ** 2)
                for target in valid
            ])
            result[b, valid] += values[b, source] * mass / mass.sum()
    return result


def kernel_audit():
    """Compare forward values and autograd to an independent float64 finite difference."""

    reference_root = ROOT / 'diagnostics/sigma_reference_20260916'
    manifest = json.loads((reference_root / 'source_manifest.json').read_text())
    reference_path = reference_root / 'reference_source/models/asym_gaussian_redistributor.py'
    reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    assert reference_hash == manifest['files']['models/asym_gaussian_redistributor.py']['sha256']
    spec = importlib.util.spec_from_file_location('original_redistribution_reference', reference_path)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    rng = np.random.default_rng(7301)
    records = []
    for length in (2, 7, 31, 200):
        for left, right in ((1, 1), (.5, 2), (2, .5), (.219748974, .173730701), (1.136265278, .356579512), (.01, 50)):
            for density in (0.3, 1.0):
                values = rng.normal(size=(2, length)).astype(np.float32)
                mask = rng.random((2, length)) < density
                mask[:, 0] = True
                readout = rng.normal(size=values.shape)
                kernel = AsymGaussianRedistributor(left, right)
                actual = kernel(torch.from_numpy(values), torch.from_numpy(mask))
                loss = (actual.double() * torch.from_numpy(readout)).sum()
                actual_gradient = torch.autograd.grad(loss, tuple(kernel.parameters()))
                with redirect_stdout(io.StringIO()):
                    reference_kernel = reference_module.AsymGaussianRedistributor(left, right)
                reference_output = reference_kernel(torch.from_numpy(values), torch.from_numpy(mask))
                reference_gradient = torch.autograd.grad((reference_output.double() * torch.from_numpy(readout)).sum(), tuple(reference_kernel.parameters()))
                reference_forward_error = float((reference_output - actual).abs().max().detach())
                reference_gradient_error = max(abs(float(a - b)) for a, b in zip(actual_gradient, reference_gradient))
                torch.testing.assert_close(reference_output, actual, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(torch.stack(reference_gradient), torch.stack(actual_gradient), atol=5e-6, rtol=2e-4)
                logs = np.array([float(p.detach()) for p in kernel.parameters()])
                expected = oracle(values, mask, logs)
                numerical = []
                for side in (0, 1):
                    plus, minus = logs.copy(), logs.copy()
                    plus[side] += 1e-4
                    minus[side] -= 1e-4
                    numerical.append(float(((oracle(values, mask, plus) - oracle(values, mask, minus)) * readout).sum() / 2e-4))
                observed = np.array([float(g) for g in actual_gradient])
                forward_error = float(np.max(np.abs(actual.detach().numpy() - expected)))
                gradient_error = float(np.max(np.abs(observed - numerical)))
                np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-6, rtol=2e-5)
                np.testing.assert_allclose(observed, numerical, atol=5e-6, rtol=2e-4)
                np.testing.assert_allclose(actual.detach().numpy().sum(1), (values * mask).sum(1), atol=5e-6, rtol=2e-5)
                records.append({'length': length, 'widths': [left, right], 'density': density, 'forward_max_abs_error': forward_error, 'gradient_max_abs_error': gradient_error, 'reference_forward_error': reference_forward_error, 'reference_gradient_error': reference_gradient_error})
    return {'cases': len(records), 'reference_sha256': reference_hash, 'max_forward_error': max(r['forward_max_abs_error'] for r in records), 'max_gradient_error': max(r['gradient_max_abs_error'] for r in records), 'max_reference_forward_error': max(r['reference_forward_error'] for r in records), 'max_reference_gradient_error': max(r['reference_gradient_error'] for r in records), 'records': records}


def qwen_gradient_audit():
    """Use the native hybrid Qwen3.5 layers with and without non-reentrant checkpointing."""

    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    records = []
    for precision in (torch.float32, torch.bfloat16):
        paired = []
        for checkpointing in (False, True):
            torch.manual_seed(17)
            config = Qwen3_5TextConfig(
                vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                linear_num_value_heads=4, linear_num_key_heads=2,
                linear_key_head_dim=8, linear_value_head_dim=8,
                layer_types=['linear_attention', 'full_attention'], max_position_embeddings=512,
                rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': 1., 'mrope_section': [1, 1, 2]},
            )
            config._attn_implementation = 'eager'
            model = DecoderVARegressor(
                Qwen3_5TextModel(config).to(precision),
                gaze_provider=FixedET2(feature_indices=(3,), repo_id='offline', revision='offline', filename='offline'),
                gaze_redistribution=redistribution_contract('asym-gaussian'),
                gaze_projection_dim=8, gaze_projection_dropout=(0., 0.), classifier_dropout=0.,
            )
            if checkpointing:
                model.gradient_checkpointing_enable()
            model.train()
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision == torch.bfloat16):
                prediction = model(input_ids=torch.tensor([[1, 5, 2, 7, 3, 4], [1, 6, 2, 3, 4, 0]]), attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])).logits
                loss = (prediction.float() - torch.tensor([[.1, .9], [.8, .2]])).square().mean()
            loss.backward()
            gradients = [float(p.grad) for p in model.gaze_redistributor.parameters()]
            assert all(math.isfinite(g) and g != 0 for g in gradients)
            assert all(p.dtype == torch.float32 for p in model.gaze_redistributor.parameters())
            paired.append((prediction.detach(), gradients))
            records.append({'dtype': str(precision), 'checkpointing': checkpointing, 'gradients': gradients})
        torch.testing.assert_close(paired[0][0], paired[1][0], atol=0, rtol=0)
        assert paired[0][1] == paired[1][1]
    return records


def segmented_spans(text, words):
    """Recover exact source character spans without searching beyond their occurrence."""

    cursor = 0
    spans = []
    for word in words:
        start = text.index(word, cursor)
        spans.append((start, start + len(word)))
        cursor = start + len(word)
    return spans


def audit_text(tokenizer, original):
    """Compare the current search-based mapping against token character offsets."""

    ids = tokenizer(original, max_length=200, truncation=True)['input_ids']
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != ids:
        return {'roundtrip_mismatch': True, 'text': text}
    words = segment_text_for_et2(text)
    spans = segmented_spans(text, words)
    mapping = align_words_to_tokens(words, ids, [1] * len(ids), tokenizer).word_to_token_indices
    offsets = encoded['offset_mapping']
    wrong, missing, leading_space = [], [], []
    exact_available = []
    for word_index, ((start, end), assigned) in enumerate(zip(spans, mapping)):
        overlap = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and a != b]
        if overlap:
            a, b = offsets[overlap[0]][0], offsets[overlap[-1]][1]
            exact = text[a:b].strip() == words[word_index]
        else:
            exact = False
        if not assigned:
            missing.append({'word_index': word_index, 'word': words[word_index], 'exact_token_span_available': exact})
            if exact:
                exact_available.append(word_index)
        else:
            visible = [i for i in assigned if text[offsets[i][0]:offsets[i][1]].strip()]
            if any(not (offsets[i][0] < end and offsets[i][1] > start) for i in visible):
                wrong.append({'word_index': word_index, 'word': words[word_index], 'source_span': [start, end], 'assigned_tokens': list(assigned), 'assigned_offsets': [offsets[i] for i in assigned]})
            elif assigned and not text[offsets[assigned[0]][0]:offsets[assigned[0]][1]].strip():
                leading_space.append(word_index)
    positions = [indices[0] for indices in mapping if indices]
    return {'text': text, 'tokens': tokenizer.convert_ids_to_tokens(ids), 'words': words, 'mapping': mapping, 'wrong': wrong, 'missing': missing, 'recoverable_missing_count': len(exact_available), 'leading_space_anchor_count': len(leading_space), 'positions': positions, 'gaps': np.diff(positions).tolist()}


def alignment_audit(tokenizer_path, results_zip):
    """Measure Qwen alignment errors on the exact texts archived in the completed run."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    examples = [audit_text(tokenizer, text) for text in (
        'I love this!!! I hate that!', "I don't like this.", "I'm sad, but it's okay.",
        'I am sad... Really sad.', 'The price is $12.50.', 'Hello?! How are you?',
    )]
    stats = Counter()
    by_dataset = {}
    cases = []
    with zipfile.ZipFile(results_zip) as archive:
        entry = next(name for name in archive.namelist() if name.endswith('/oof_predictions.tsv'))
        rows = list(csv.DictReader(io.StringIO(archive.read(entry).decode()), delimiter='\t'))
    assert len(rows) == 14352
    for fold in (1, 2):
        for row in rows:
            if int(row['held_out_fold']) != fold:
                continue
            result = audit_text(tokenizer, row['text'])
            delta = Counter(rows=1)
            if result.get('roundtrip_mismatch'):
                delta['roundtrip_mismatch'] += 1
            else:
                delta.update({'words': len(result['words']), 'mapped_words': sum(bool(x) for x in result['mapping']), 'missing_words': len(result['missing']), 'wrong_occurrence_words': len(result['wrong']), 'rows_with_wrong_occurrence': int(bool(result['wrong'])), 'rows_with_missing': int(bool(result['missing'])), 'recoverable_missing_words': result['recoverable_missing_count'], 'leading_space_anchor_count': result['leading_space_anchor_count'], 'rows_with_no_mapped_position': int(len(result['positions']) == 0), 'rows_with_one_mapped_position': int(len(result['positions']) == 1), 'gaps': len(result['gaps']), 'gaps_above_one': sum(g > 1 for g in result['gaps'])})
                if result['wrong'] and len(cases) < 30:
                    cases.append({'fold': fold, 'index': row['index'], 'dataset': row['dataset_of_origin'], **result})
            stats.update(delta)
            by_dataset.setdefault(row['dataset_of_origin'], Counter()).update(delta)
        print(f'Alignment audit completed fold {fold}: {dict(stats)}', flush=True)
    return {'tokenizer_path': str(tokenizer_path), 'tokenizer_json_sha256': hashlib.sha256((Path(tokenizer_path) / 'tokenizer.json').read_bytes()).hexdigest(), 'results_zip': str(results_zip), 'results_zip_sha256': hashlib.sha256(Path(results_zip).read_bytes()).hexdigest(), 'summary': dict(stats), 'by_dataset': by_dataset, 'constructed_examples': examples, 'corpus_examples': cases}


def main():
    """Write independent audit evidence without changing the training implementation."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-path')
    parser.add_argument('--results-zip', default='/Users/wansookim/Downloads/qwen3.5-0.8b_full_gaze_TRT_redistribution_asym-gaussian_sentence_only_no_iemocap_seed42_results_only (1).zip')
    parser.add_argument('--alignment-only', action='store_true')
    args = parser.parse_args()
    output = {'torch_version': torch.__version__, 'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT / 'va_model_code/decoder_va').glob('*.py')}}
    if not args.alignment_only:
        output['kernel'] = kernel_audit()
        output['native_qwen'] = qwen_gradient_audit()
    if args.tokenizer_path:
        output['alignment'] = alignment_audit(args.tokenizer_path, args.results_zip)
    path = Path(__file__).with_name('alignment_results.json' if args.alignment_only else 'results.json')
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(path)


if __name__ == '__main__':
    main()

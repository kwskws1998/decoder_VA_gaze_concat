"""Verify packaged predictions and compare matched alignv3 and historical runs."""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
DOWNLOADS = Path('/Users/wansookim/Downloads')
INPUTS = {
    'raw': 'qwen3.5-0.8b_full_gaze_TRT_sentence_only_no_iemocap_seed42_results_only.zip',
    'right': 'qwen3.5-0.8b_full_gaze_TRT_redistribution_fixed-gaussian_sentence_only_no_iemocap_seed42_results_only (1).zip',
    'left': 'qwen3.5-0.8b_full_gaze_TRT_redistribution_fixed-gaussian_sentence_only_no_iemocap_seed42_results_only.zip',
}
IDENTITY = ('held_out_fold', 'index', 'text', 'dataset_of_origin', 'valence', 'arousal')
METADATA = ('held_out_fold', 'index', 'text', 'dataset_of_origin')
PERMUTATION = Path('/private/tmp/alignv3_eval_permutation_20261014.json')


def read_rows(data):
    """Read predictions in stable fold and sample order."""
    return sorted(csv.DictReader(io.StringIO(data.decode()), delimiter='\t'),
                  key=lambda row: (int(row['held_out_fold']), int(row['index'])))


def calculate(rows):
    """Recompute all regression metrics in float64 from original predictions."""
    result = {'n_examples': len(rows)}
    for side in ('valence', 'arousal'):
        y = np.array([float(row[side]) for row in rows], dtype=np.float64)
        p = np.array([float(row['pred_' + side]) for row in rows], dtype=np.float64)
        assert np.isfinite(y).all() and np.isfinite(p).all()
        covariance = np.mean((y - y.mean()) * (p - p.mean()))
        mse = np.mean((p - y) ** 2)
        result.update({
            'mse_' + side: float(mse),
            'rmse_' + side: float(np.sqrt(mse)),
            'mae_' + side: float(np.abs(p - y).mean()),
            'pearson_corr_' + side: float(covariance / np.sqrt(y.var() * p.var())),
            'ccc_' + side: float(2 * covariance / (y.var() + p.var() + (y.mean() - p.mean()) ** 2)),
        })
    for metric in ('mse', 'mae', 'pearson_corr', 'ccc'):
        result[metric + '_mean'] = (result[metric + '_valence'] + result[metric + '_arousal']) / 2
    return result


def verify_metrics(stored, computed, prefix=''):
    """Compare stored metrics with independent recomputation without display rounding."""
    error = max(abs(stored[prefix + key] - value) for key, value in computed.items())
    assert error < 1e-12, error
    return error


def load(data):
    """Verify manifest membership, CRC and hashes before reading run evidence."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) and archive.testzip() is None
        roots = {name.split('/')[0] for name in names}
        assert len(roots) == 1
        root = next(iter(roots))
        manifest = json.loads(archive.read(root + '/results_manifest.json'))
        assert set(names) == {root + '/' + n for n in manifest['files']} | {root + '/results_manifest.json'}
        files = {}
        for name, info in manifest['files'].items():
            value = archive.read(root + '/' + name)
            assert len(value) == info['size_bytes']
            assert hashlib.sha256(value).hexdigest() == info['sha256']
            files[name] = value
    config = json.loads(files['training_parameters.json'])
    assert config['run_name'] == root
    rows = read_rows(files['oof_predictions.tsv'])
    metrics = calculate(rows)
    summary = {'run_name': root, 'config': config, 'metrics': metrics, 'folds': {}, 'recorded_datasets_invalid_metadata': {},
               'recomputation_max_error': verify_metrics(json.loads(files['oof_metrics.json']), metrics)}
    for fold in (1, 2):
        prefix = f'heldout_fold{fold}/'
        subset = [row for row in rows if int(row['held_out_fold']) == fold]
        assert read_rows(files[prefix + 'predictions.tsv']) == subset
        computed = calculate(subset)
        verify_metrics(json.loads(files[prefix + 'metrics.json']), computed, 'test_')
        states = [json.loads(value) for name, value in files.items()
                  if name.startswith(prefix + 'checkpoints/') and name.endswith('trainer_state.json')]
        state = max(states, key=lambda item: item['global_step'])
        selected_path = prefix + 'checkpoints/' + Path(state['best_model_checkpoint']).name + '/trainer_state.json'
        selected = json.loads(files[selected_path])
        assert abs(state['best_metric'] - computed['mse_mean']) < 1e-12
        fold_manifest = json.loads(files[prefix + 'run_manifest.json'])
        architecture = json.loads(files[prefix + 'final_model/decoder_va_architecture.json'])
        history = [row for row in state['log_history'] if 'eval_mse_mean' in row]
        summary['folds'][str(fold)] = {
            'metrics': computed, 'selected_epoch': selected['epoch'],
            'completed_epoch_in_state': state['epoch'], 'step': state['global_step'],
            'eval_history': history, 'training_rows': fold_manifest['training_rows'],
            'overlap': fold_manifest.get('training_text_overlap'),
            'gaze_redistribution': architecture['gaze_redistribution'],
            'gpu_memory': json.loads(files[prefix + 'gpu_memory.json']),
        }
    stored_groups = list(csv.DictReader(io.StringIO(files['metrics_by_dataset.tsv'].decode()), delimiter='\t'))
    for group in stored_groups:
        dataset = group['dataset_of_origin']
        computed = calculate([row for row in rows if row['dataset_of_origin'] == dataset])
        verify_metrics({key: float(group[key]) for key in computed}, computed)
        summary['recorded_datasets_invalid_metadata'][dataset] = computed
    return summary, rows


def identify(rows):
    """Keep exact sample identity and labels for matched comparisons."""
    return [tuple(row[key] for key in IDENTITY) for row in rows]


def recover_metadata(rows, permutation):
    """Repair result metadata and independently verify labels against hashed source folds."""
    recovered = []
    for fold, mapping in permutation['folds'].items():
        original = Path(mapping['source_file']).read_bytes()
        assert hashlib.sha256(original).hexdigest() == mapping['source_sha256']
        sources = {row['index']: row for row in csv.DictReader(io.StringIO(original.decode()), delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')}
        members = {row['index']: row for row in rows if row['held_out_fold'] == fold}
        indices = [str(i) for i in mapping['metadata_indices']]
        order = mapping['prediction_position_to_metadata_position']
        assert set(members) == set(indices) and len(members) == len(indices)
        assert sorted(order) == list(range(len(members)))
        for j, metadata_position in enumerate(order):
            corrected = dict(members[indices[j]])
            metadata = members[indices[metadata_position]]
            for key in METADATA:
                corrected[key] = metadata[key]
            gold = sources[corrected['index']]
            assert corrected['text'] == gold['text']
            assert corrected['dataset_of_origin'] == gold['dataset_of_origin']
            for side in ('valence', 'arousal'):
                assert float(corrected[side]) == float(np.float32(gold[side]))
            recovered.append(corrected)
    assert len(recovered) == len(rows)
    return sorted(recovered, key=lambda row: (int(row['held_out_fold']), int(row['index'])))


def compare_versions(old, new):
    """Distinguish evaluation membership from the known misassigned metadata labels."""
    assert [tuple(r[k] for k in METADATA) for r in old] == [tuple(r[k] for k in METADATA) for r in new]
    result = {}
    for fold in ('1', '2'):
        a = [r for r in old if r['held_out_fold'] == fold]
        b = [r for r in new if r['held_out_fold'] == fold]
        label_pairs_a = [(float(r['valence']), float(r['arousal'])) for r in a]
        label_pairs_b = [(float(r['valence']), float(r['arousal'])) for r in b]
        assert Counter(label_pairs_a) == Counter(label_pairs_b)
        result[fold] = {'n_rows': len(a), 'evaluation_membership_identical': True,
                        'label_pair_multisets_identical': True,
                        'recorded_row_label_mismatches': sum(x != y for x, y in zip(label_pairs_a, label_pairs_b))}
    return result


def pairwise(a, b):
    """Describe paired prediction differences, without a multi-seed significance claim."""
    assert identify(a) == identify(b)
    y = np.array([[float(row[k]) for k in ('valence', 'arousal')] for row in a])
    pa = np.array([[float(row['pred_' + k]) for k in ('valence', 'arousal')] for row in a])
    pb = np.array([[float(row['pred_' + k]) for k in ('valence', 'arousal')] for row in b])
    delta = np.mean((pa - y) ** 2 - (pb - y) ** 2, axis=1)
    return {'mean_mse_delta_a_minus_b': float(delta.mean()),
            'a_lower_error_fraction': float(np.mean(delta < 0)),
            'median_mse_delta': float(np.median(delta)),
            'prediction_rms_difference': float(np.sqrt(np.mean((pa - pb) ** 2))),
            'relative_mse_change_percent': float(100 * delta.mean() / np.mean((pb - y) ** 2))}


def main():
    """Write reproducible JSON evidence for matched and historical comparisons."""
    report = {'inputs_sha256': {}, 'current': {}, 'historical': {}, 'paired_current': {}, 'historical_deltas': {},
              'cross_version_evaluation': {}}
    permutation = json.loads(PERMUTATION.read_text())
    report['metadata_recovery'] = permutation
    current_rows = {}
    for name, filename in INPUTS.items():
        data = (DOWNLOADS / filename).read_bytes()
        report['inputs_sha256'][filename] = hashlib.sha256(data).hexdigest()
        report['current'][name], current_rows[name] = load(data)
    allowed = {'run_name', 'effective_output_dir', 'gaze_redistribution', 'redistribution_sigma_left', 'redistribution_sigma_right'}
    reference = report['current']['raw']['config']
    for name, run in report['current'].items():
        assert identify(current_rows[name]) == identify(current_rows['raw'])
        changed = {key for key in reference.keys() | run['config'].keys() if reference.get(key) != run['config'].get(key)}
        assert changed <= allowed, changed
        run['config_differences_from_raw'] = sorted(changed)
        recovered = recover_metadata(current_rows[name], permutation)
        verify_metrics(run['metrics'], calculate(recovered))
        run['datasets_recovered_verified'] = {
            dataset: calculate([row for row in recovered if row['dataset_of_origin'] == dataset])
            for dataset in sorted({row['dataset_of_origin'] for row in recovered})
        }
    for a, b in (('right', 'raw'), ('left', 'raw'), ('right', 'left')):
        report['paired_current'][a + '_vs_' + b] = pairwise(current_rows[a], current_rows[b])
    old_sources = [
        ('sigma_sentence_only_alignv2_results_only.zip', {'raw_results_only.zip': 'raw'}),
        ('sigma_sentence_only_l0p25r1_l1r0p25_results_only_20261007_135016.zip',
         {'right_l0p25_r1_results_only.zip': 'right', 'left_l1_r0p25_results_only.zip': 'left'}),
    ]
    for filename, inner_names in old_sources:
        data = (DOWNLOADS / filename).read_bytes()
        report['inputs_sha256'][filename] = hashlib.sha256(data).hexdigest()
        with zipfile.ZipFile(io.BytesIO(data)) as outer:
            for inner_name, label in inner_names.items():
                old, old_rows = load(outer.read(inner_name))
                report['cross_version_evaluation'][label] = compare_versions(old_rows, current_rows[label])
                report['historical'][label] = old
                new_metrics = report['current'][label]['metrics']
                report['historical_deltas'][label] = {
                    'mse_relative_change_percent': 100 * (new_metrics['mse_mean'] / old['metrics']['mse_mean'] - 1),
                    'metric_deltas': {key: new_metrics[key] - old['metrics'][key] for key in new_metrics},
                }
    report['limitations'] = ['One seed schedule, fold seeds42/43, not independent multi-seed replications.',
                             'Historical comparisons change ET2 preprocessing, precision, physical batches and training overlaps.',
                             'Fixed sigma values are configuration/log evidence; no model weights were supplied.',
                             'All original result files have incorrect text/source metadata alignment. Global and fold metrics remain valid because labels and predictions move together.',
                             'Only current dataset metrics under datasets_recovered_verified have corrected source identities; do not use recorded dataset metrics.']
    (HERE / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    for group in ('current', 'historical'):
        for name, run in report[group].items():
            print(group, name, json.dumps({'metrics': run['metrics'],
                  'folds': {k: {'mse': v['metrics']['mse_mean'], 'epoch': v['selected_epoch']} for k, v in run['folds'].items()},
                  'datasets_recovered': {k: v['mse_mean'] for k, v in run.get('datasets_recovered_verified', {}).items()}}))
    print('PAIRS', json.dumps(report['paired_current']))
    print('HISTORY', json.dumps({k: v['mse_relative_change_percent'] for k, v in report['historical_deltas'].items()}))


if __name__ == '__main__':
    main()

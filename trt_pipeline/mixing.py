"""Fixed-coefficient prediction mixing with strict row/label alignment."""
from pathlib import Path
import json
import numpy as np
from trt_pipeline.config import PREDICTION_ALPHA

KEYS = ('pearson_corr_valence', 'pearson_corr_arousal', 'rmse_valence', 'rmse_arousal', 'mae_valence', 'mae_arousal')


def mix_predictions(first, second, alpha=PREDICTION_ALPHA):
    """Apply the measured shared convex coefficient in Float64 to both VA dimensions."""
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if not 0 <= alpha <= 1 or a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('Invalid convex prediction mixture')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(a > 1) or np.any(b < 0) or np.any(b > 1):
        raise ValueError('Mixture inputs must be finite normalized VA predictions')
    return (1-alpha)*a + alpha*b


def read_predictions(path, kind):
    """Read only one declared prediction array with unique Unicode row identities."""
    with np.load(path, allow_pickle=False) as saved:
        choices = ['predictions', 'predictions_'+kind]
        if kind == 'companion':
            choices.append('predictions_matched_mse')
        matches = [key for key in choices if key in saved]
        if len(matches) != 1:
            raise ValueError('Ambiguous or missing '+kind+' predictions')
        values, labels, ids = saved[matches[0]].copy(), saved['labels'].copy(), saved['row_ids'].copy()
    if not len(values) or ids.dtype.kind != 'U' or labels.shape != values.shape or ids.shape != (len(values),) or len(set(ids.tolist())) != len(ids):
        raise ValueError('Invalid prediction identities or shape')
    if not np.isfinite(labels).all() or (labels < 0).any() or (labels > 1).any():
        raise ValueError('Invalid normalized gold labels')
    return values, labels, ids


def mix_files(baseline, companion, output):
    """Mix already computed outputs after checking exact row order and labels."""
    first, labels, ids = read_predictions(baseline, 'baseline')
    second, other_labels, other_ids = read_predictions(companion, 'companion')
    if not np.array_equal(labels, other_labels) or not np.array_equal(ids, other_ids):
        raise ValueError('Baseline/companion row order or labels differ')
    values = mix_predictions(first, second)
    from trt_pipeline.core.evaluation import calculate_va_metrics
    metrics = calculate_va_metrics(labels, values)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as target:
        np.savez_compressed(target, predictions=values, labels=labels, row_ids=ids)
    output.with_suffix('.json').write_text(json.dumps({'alpha': PREDICTION_ALPHA, 'metrics': metrics,
        'baseline': str(baseline), 'companion': str(companion), 'scope': 'Development evaluation; alpha selected on Fold1'}, indent=2)+'\n')
    return metrics


def fold_metric_mean(first, second):
    """Average separately computed fold metrics with equal fold weights."""
    return {key: (float(first[key])+float(second[key]))/2 for key in KEYS}

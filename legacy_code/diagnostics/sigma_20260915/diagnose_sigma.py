"""Audit saved sigma trajectories and run offline probes of the active kernel.

The probes use synthetic TRT and untrained projection weights. They establish
mechanisms and gradient connectivity, not causes in a trained Qwen checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from va_model_code.decoder_va.gaze import segment_text_for_et2
from va_model_code.decoder_va.model import DecoderVARegressor
from va_model_code.decoder_va.redistribution import AsymGaussianRedistributor
from transformers import LlamaConfig, LlamaModel


def describe(values):
    """Summarize a nonempty numeric vector without printing individual texts."""
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def audit_archive(path, output_dir):
    """Read the original result ZIP and preserve its manifest and logged rows."""
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure: {bad_member}")
        names = archive.namelist()
        parameter_name = next(n for n in names if n.endswith('/training_parameters.json'))
        parameters = json.loads(archive.read(parameter_name))
        result = {
            "archive": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "crc_valid": True,
            "has_model_weights": any(n.endswith(('.safetensors', '.bin')) for n in names),
            "training_parameters": parameters,
            "folds": [],
        }
        for name in names:
            if not name.endswith('/trainer_state.json'):
                continue
            state = json.loads(archive.read(name))
            fold = name.split('/')[1]
            (output_dir / f'{fold}_trainer_state.json').write_text(
                json.dumps(state, indent=2) + '\n', encoding='utf-8'
            )
            records = state['log_history']
            trajectory = {
                "fold": fold,
                "archive_member": name,
                "saved_epoch": state['epoch'],
                "saved_step": state['global_step'],
                "max_steps": state['max_steps'],
                "best_step": state['best_global_step'],
                "logs": records,
                "widths": {},
            }
            for side in ('left', 'right'):
                key = f'redistribution_sigma_{side}'
                values = [parameters[f'redistribution_sigma_{side}'] + 1e-6]
                values += [row[key] for row in records if key in row]
                theta = np.log(np.asarray(values) - 1e-6)
                variation = float(np.abs(np.diff(theta)).sum())
                trajectory['widths'][side] = {
                    "start": values[0],
                    "last_saved": values[-1],
                    "min_logged": min(values),
                    "max_logged": max(values),
                    "logged_log_sigma_total_variation": variation,
                    "net_over_logged_variation": float(abs(theta[-1] - theta[0]) / variation),
                }
            result['folds'].append(trajectory)
        prediction_name = next(n for n in names if n.endswith('/oof_predictions.tsv'))
        rows = list(csv.DictReader(io.StringIO(archive.read(prediction_name).decode()), delimiter='\t'))
        datasets = {}
        for source in sorted({row['dataset_of_origin'] for row in rows}):
            texts = [row['text'] for row in rows if row['dataset_of_origin'] == source]
            datasets[source] = {
                "whitespace_word_count": describe([len(text.split()) for text in texts]),
                "et_segment_count_before_token_alignment": describe([len(segment_text_for_et2(text)) for text in texts]),
                "at_most_one_whitespace_word": sum(len(text.split()) <= 1 for text in texts),
                "at_most_one_et_segment_before_alignment": sum(len(segment_text_for_et2(text)) <= 1 for text in texts),
            }
        result['sentence_data'] = datasets
        result['prediction_boundaries'] = {}
        for column in ('pred_valence', 'pred_arousal'):
            values = np.asarray([float(row[column]) for row in rows])
            result['prediction_boundaries'][column] = {
                "min": float(values.min()), "max": float(values.max()),
                "exact_zero_or_one_count": int(((values == 0) | (values == 1)).sum()),
                "count": int(values.size),
            }
        return result


def kernel_probes(audit):
    """Measure normalized mass, sparse-coordinate effects and exact invariances."""
    result = {"scope": "synthetic inputs through the current production kernel"}
    profiles = {}
    configurations = {'initial': (1.0, 1.0)}
    for fold in audit['folds']:
        configurations[fold['fold'].replace('heldout_', '') + '_saved'] = tuple(
            fold['widths'][side]['last_saved'] - 1e-6 for side in ('left', 'right')
        )
    impulse = torch.zeros(1, 41)
    impulse[0, 20] = 1
    mask = torch.ones_like(impulse, dtype=torch.bool)
    for name, widths in configurations.items():
        with torch.no_grad():
            profile = AsymGaussianRedistributor(*widths)(impulse, mask)[0]
        profiles[name] = profile.tolist()
    result['interior_impulse'] = {
        name: {
            "left_mass": sum(profile[:20]),
            "self_mass": profile[20],
            "right_mass": sum(profile[21:]),
            "total_variation_from_initial": 0.5 * sum(abs(a - b) for a, b in zip(profile, profiles['initial'])),
        }
        for name, profile in profiles.items()
    }
    sparse = []
    for gap in (1, 2, 3, 4, 5):
        x = torch.zeros(1, gap + 1)
        x[0, 0] = 1
        valid = torch.zeros_like(x, dtype=torch.bool)
        valid[0, 0] = valid[0, -1] = True
        kernel = AsymGaussianRedistributor()
        y = kernel(x, valid)
        gradient = torch.autograd.grad(y[0, -1], kernel.log_sigma_right)[0]
        sparse.append({"qwen_token_gap": gap, "neighbor_mass": y[0, -1].item(), "derivative_wrt_log_sigma_right": gradient.item()})
    result['two_valid_tokens'] = sparse
    gradients = {}
    x = torch.tensor([[0.3, 2.0, 0.1, 1.2, 0.4]])
    for name, readout in {'sum': torch.ones_like(x), 'position_sensitive': torch.tensor([[0., 1., -2., 0.5, 3.]])}.items():
        kernel = AsymGaussianRedistributor()
        y = kernel(x, torch.ones_like(x, dtype=torch.bool))
        values = torch.autograd.grad((y * readout).sum(), tuple(kernel.parameters()))
        gradients[name] = [value.item() for value in values]
    kernel = AsymGaussianRedistributor()
    singleton = kernel(torch.tensor([[2.0]]), torch.tensor([[True]]))
    gradients['single_valid_token'] = [value.item() for value in torch.autograd.grad(singleton.sum(), tuple(kernel.parameters()))]
    result['readout_gradients_left_right'] = gradients
    return result, profiles


def fit_known_kernel():
    """Check recovery when asymmetric target redistribution is directly observed."""
    torch.manual_seed(42)
    values = torch.rand(16, 12) * 3
    mask = torch.ones_like(values, dtype=torch.bool)
    with torch.no_grad():
        target = AsymGaussianRedistributor(0.5, 2.0)(values, mask)
    learned = AsymGaussianRedistributor()
    optimizer = torch.optim.AdamW(learned.parameters(), lr=1e-3, weight_decay=0)
    initial_loss = ((learned(values, mask) - target) ** 2).mean().item()
    for _ in range(3000):
        optimizer.zero_grad(set_to_none=True)
        loss = ((learned(values, mask) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    final_loss = ((learned(values, mask) - target) ** 2).mean().item()
    return {"scope": "synthetic direct kernel supervision; no Qwen and no VA labels", "steps": 3000, "learning_rate": 1e-3, "target_sigmas": [0.500001, 2.000001], "learned_sigmas": [learned.sigma_left.item(), learned.sigma_right.item()], "initial_mse": initial_loss, "final_mse": final_loss}


def projector_probes():
    """Measure scalar TRT sensitivity using the real, freshly initialized projector."""
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=8, hidden_size=128, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
    provider = type('Provider', (), {'feature_indices': (3,)})()
    model = DecoderVARegressor(LlamaModel(config), gaze_provider=provider, gaze_projection_dim=128)
    projector = model.gaze_projector.eval()
    measurements = []
    for scale in (0.1, 1.0, 3.0, 10.0, 100.0, 1000.0):
        x = torch.tensor([[scale]], requires_grad=True)
        y = projector(x)
        derivative = torch.autograd.functional.jacobian(projector, x).reshape(-1)
        with torch.no_grad():
            doubled = projector(2 * x)
        measurements.append({"synthetic_trt": scale, "relative_change_when_trt_doubles": ((doubled - y).norm() / y.norm()).item(), "relative_local_sensitivity": (scale * derivative.norm() / y.norm()).item()})
    first_linear = projector[0]
    with torch.no_grad():
        first_linear.bias.zero_()
    zero_bias = []
    for scale in (1.0, 10.0, 100.0):
        x = torch.tensor([[scale]])
        with torch.no_grad():
            y, doubled = projector(x), projector(2 * x)
        zero_bias.append({"synthetic_trt": scale, "relative_change_when_trt_doubles": ((doubled - y).norm() / y.norm()).item()})
    return {"scope": "synthetic scalar inputs; random untrained real projector; actual TRT scale and trained weights unavailable", "default_nonzero_bias": measurements, "zero_first_layer_bias_mechanism_control": zero_bias}


def make_figure(audit, profiles, destination):
    """Render observed trajectories and a clearly labeled synthetic impulse response."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, fold in zip(axes[:2], audit['folds']):
        for side, color in [('left', '#2775b6'), ('right', '#cc6540')]:
            logs = fold['logs']
            epochs = [0] + [r['epoch'] for r in logs]
            widths = [1.000001] + [r[f'redistribution_sigma_{side}'] for r in logs]
            ax.plot(epochs, widths, marker='.', label=side, color=color)
        ax.axhline(1, color='gray', lw=1, linestyle='--')
        ax.set(title=fold['fold'] + ' (saved logs)', xlabel='Epoch', ylabel='Sigma (Qwen token units)', ylim=(0.92, 1.09))
        ax.legend(frameon=False)
    for name, profile in profiles.items():
        axes[2].plot(range(-5, 6), profile[15:26], marker='.', label=name)
    axes[2].set(title='Interior impulse: synthetic dense mask', xlabel='Destination minus source token', ylabel='Normalized mass')
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    """Write a reproducible diagnosis bundle without modifying training code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    audit = audit_archive(args.archive, args.output_dir)
    kernel, profiles = kernel_probes(audit)
    report = {
        "source_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        "torch_version": torch.__version__,
        "archive_audit": audit,
        "kernel_probes": kernel,
        "synthetic_identification": fit_known_kernel(),
        "projector_probes": projector_probes(),
    }
    target = args.output_dir / 'diagnosis.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    make_figure(audit, profiles, args.output_dir / 'sigma_trajectory.png')
    print(json.dumps({"output": str(target.resolve()), "kernel": kernel, "synthetic_fit": report['synthetic_identification'], "projector": report['projector_probes'], "data": audit['sentence_data']}, indent=2))


if __name__ == '__main__':
    main()

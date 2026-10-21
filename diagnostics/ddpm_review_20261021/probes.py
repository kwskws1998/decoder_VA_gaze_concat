"""Reproduce DDPM review probes without changing training code or downloading weights."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from diffusion_code.config import DiffusionConfig
from diffusion_code.ddpm import GaussianDDPM
from diffusion_code.model import ConditionalDenoiser
from va_model_code.decoder_va.filters import load_filtered_folds
from va_model_code.decoder_va.model import EncodedVAFeatures


def schedule_and_shortcut():
    """Measure objective weights and a label-prior-only linear denoiser analytically."""

    ddpm = GaussianDDPM(DiffusionConfig())
    a = ddpm.alpha_bar.double()
    snr = a / (1 - a)
    rows = []
    for lo, hi in ((0, 25), (25, 50), (50, 75), (75, 100)):
        rows.append({
            "code_indices": [lo, hi - 1],
            "mean_snr": snr[lo:hi].mean().item(),
            "mean_va_mse_weight": (4 * snr[lo:hi]).mean().item(),
        })
    folds = load_filtered_folds(ROOT / "va_model_code/data/paper7_seed42", no_iemocap=True).folds
    train = torch.tensor(folds["full_dataset_fold2.csv"][["valence", "arousal"]].to_numpy()).double()
    test = torch.tensor(folds["full_dataset_fold1.csv"][["valence", "arousal"]].to_numpy()).double()
    mean, variance = train.mean(0), train.var(0, unbiased=False)
    zvar = 4 * variance
    linear_epsilon_risk = a[:, None] * zvar / (a[:, None] * zvar + 1 - a[:, None])
    return {
        "fold2_rows": len(train), "fold1_rows": len(test),
        "train_va_mean": mean.tolist(), "train_va_variance": variance.tolist(),
        "constant_train_mean_heldout_mse": (test - mean).square().mean(0).tolist(),
        "unconditional_linear_epsilon_mse": linear_epsilon_risk.mean(0).tolist(),
        "unconditional_linear_epsilon_mse_mean": linear_epsilon_risk.mean().item(),
        "unconditional_linear_epsilon_mse_by_quartile": [
            linear_epsilon_risk[lo:hi].mean().item() for lo, hi in ((0, 25), (25, 50), (50, 75), (75, 100))
        ],
        "objective_weights": rows,
        "note": "Exact expected risk of an affine predictor of noise using only noisy VA and training-label moments; no text/TRT.",
    }


def exact_oracle():
    """Test the complete 100-step chain with deliberately known clean synthetic VA."""

    ddpm = GaussianDDPM(DiffusionConfig())
    labels = torch.tensor([[.05, .95], [.3, .8], [.5, .5], [.9, .1]])
    clean = 2 * labels[:, None] - 1

    def denoise(noisy, time):
        """Return the synthetic oracle's true corruption noise at each queried state."""

        a = ddpm.alpha_bar[time, None, None]
        return (noisy - a.sqrt() * clean) / (1 - a).sqrt()

    samples = ddpm.sample(denoise, batch_size=len(labels))
    error = (samples - labels[:, None]).abs().max().item()
    assert error < 1e-5
    return {"max_va_absolute_error": error, "purpose": "Sampler arithmetic only, not empirical efficacy."}


def finite_sampling():
    """Isolate finite sample mean error using a known Gaussian conditional denoiser."""

    labels = torch.tensor([[.35, .6], [.6, .4]])
    clean_mean = 2 * labels[:, None] - 1
    clean_variance = (.1 * 2) ** 2
    rows = []
    for count in (16, 64, 256):
        for seed in (2026, 2027, 2028):
            ddpm = GaussianDDPM(DiffusionConfig(samples=count, sampling_seed=seed, clip_clean=False))

            def denoise(noisy, time):
                """Evaluate the known Gaussian's exact conditional mean epsilon."""

                a = ddpm.alpha_bar[time, None, None]
                variance = a * clean_variance + 1 - a
                return (1 - a).sqrt() / variance * (noisy - a.sqrt() * clean_mean)

            samples = ddpm.sample(denoise, batch_size=len(labels))
            rows.append({
                "samples": count, "sampling_seed": seed,
                "mean_error_per_axis": (samples.mean(1) - labels).mean(0).tolist(),
                "mse_against_known_conditional_mean": (samples.mean(1) - labels).square().mean().item(),
                "sample_variance": samples.var(1, unbiased=True).mean(0).tolist(),
            })
    return {
        "rows": rows,
        "note": "Synthetic unbounded Gaussian oracle; production reverse/final transform retained including final [0,1] clamp. Not measured model uncertainty.",
    }


def select_features(features, indices):
    """Slice all optional synthetic encoder fields together."""

    return EncodedVAFeatures(**{
        name: None if getattr(features, name) is None else getattr(features, name)[indices]
        for name in ("pooled", "text_hidden", "raw_gaze", "gaze_mask")
    })


def train_positive_control(conditioning, steps):
    """Train the production head on true DDPM corruptions with recoverable conditions."""

    torch.manual_seed(42)
    settings = DiffusionConfig(conditioning=conditioning)
    head = ConditionalDenoiser(16, settings)
    ddpm = GaussianDDPM(settings)
    n = 16
    if conditioning == "pooled":
        features = EncodedVAFeatures(pooled=torch.eye(n))
        labels = .15 + .7 * torch.rand(n, 2)
    else:
        trt = torch.linspace(.1, .9, n)
        features = EncodedVAFeatures(
            pooled=torch.ones(n, 16), text_hidden=torch.ones(n, 1, 16),
            raw_gaze=trt[:, None, None], gaze_mask=torch.ones(n, 1, dtype=torch.bool),
        )
        labels = torch.stack((.2 + .6 * trt, .8 - .6 * trt), -1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
    checkpoints = []
    losses = []
    for step in range(steps + 1):
        if step in (0, 500, 1682, steps):
            head.eval()
            with torch.no_grad():
                condition = head.prepare(features)
                generated = ddpm.sample(lambda x, t: head(x, t, condition), batch_size=n).mean(1)
                permuted = head.prepare(select_features(features, torch.arange(n).flip(0)))
                wrong = ddpm.sample(lambda x, t: head(x, t, permuted), batch_size=n).mean(1)
            checkpoints.append({
                "step": step, "generated_va_mse": F.mse_loss(generated, labels).item(),
                "reversed_condition_va_mse": F.mse_loss(wrong, labels).item(),
                "recent_epsilon_mse": sum(losses[-100:]) / len(losses[-100:]) if losses else None,
            })
            print(conditioning, checkpoints[-1], flush=True)
            head.train()
        if step == steps:
            break
        indices = torch.randint(n, (32,))
        noisy, time, epsilon = ddpm.noise_targets(labels[indices])
        loss = F.mse_loss(head(noisy[:, None], time, head.prepare(select_features(features, indices)))[:, 0], epsilon)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
        optimizer.step()
        losses.append(loss.item())
    return {"conditioning": conditioning, "checkpoints": checkpoints,
            "note": "CPU synthetic fixed features, constant head LR, no pretrained Qwen; a trainability test only."}


def main():
    """Save source fingerprints and deterministic CPU probe evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("probes.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = {
        "torch": torch.__version__,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "diffusion_code").glob("*.py"))
        },
        "schedule_and_shortcut": schedule_and_shortcut(),
        "exact_oracle": exact_oracle(),
        "finite_sampling": finite_sampling(),
        "positive_controls": [train_positive_control(mode, args.steps) for mode in ("pooled", "token-trt")],
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

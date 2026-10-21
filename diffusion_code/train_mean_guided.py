"""Train only a residual DDPM head above a verified, frozen Raw TRT mean predictor."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_code import train
from diffusion_code.diagnostic_data import require_new_output
from diffusion_code.matched_raw import MATCHED_FIELDS
from diffusion_code.mean_reference import RawMeanReference
from va_model_code.decoder_va.paths import resolve_run_directory


def build_parser():
    """Inherit the mean model's data/encoder settings and expose only new head-training choices."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-run", required=True)
    p.add_argument("--data-dir", default=str(ROOT / "va_model_code/data/paper7_seed42"))
    p.add_argument("--run-name")
    p.add_argument("--held-out-folds", type=int, nargs="+", choices=(1, 2))
    p.add_argument("--conditioning", choices=("pooled", "token-trt"), default="token-trt")
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--head-learning-rate", type=float, default=1e-4)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--diffusion-samples", type=int, default=16)
    p.add_argument("--diffusion-width", type=int, default=128)
    p.add_argument("--diffusion-blocks", type=int, default=3)
    p.add_argument("--sampling-seed", type=int, default=2026)
    p.add_argument("--seed", type=int, help="Head initialization/order seed; defaults to source base seed.")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true")
    return p


def run(args):
    """Require original weights and data, preserve reference ownership, and start a distinct run."""
    reference = RawMeanReference(resolve_run_directory(args.raw_run), args.data_dir, args.held_out_folds)
    configured = train.build_parser().parse_args([])
    for field in MATCHED_FIELDS:
        setattr(configured, field, reference.manifest[field])
    for field in ("epochs", "head_learning_rate", "diffusion_steps", "diffusion_samples", "diffusion_width",
                  "diffusion_blocks", "sampling_seed", "conditioning", "max_steps", "dry_run"):
        setattr(configured, field, getattr(args, field))
    configured.data_dir = str(args.data_dir)
    configured.held_out_folds = reference.folds
    configured.gaze_fusion = "prefix-concat"
    configured.gradient_checkpointing = False
    configured.seed = reference.manifest["seed"] if args.seed is None else args.seed
    configured.run_name = args.run_name or f"mean_guided_trt_seed{configured.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    require_new_output(resolve_run_directory(configured.run_name), reference.directory)
    print(f"Frozen mean: {reference.directory}; selected folds={reference.folds}; data and precision inherited.")
    return train.run(configured, mean_reference=reference)


if __name__ == "__main__":
    run(build_parser().parse_args())

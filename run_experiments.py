"""Edit the explicit Python settings below or supply detailed Python-script arguments.

Use the active conda environment's Python. Importing this file never starts a fit.
Settings can change only before prepare/run creates a new protocol lock.
"""

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path

from holdout_experiments.config import ARMS, CONDITIONS, Settings
from trt_pipeline.config import DATA_DIR, ROOT


OUTPUT = ROOT / "results/holdout_seed42"
INPUT_DATA = DATA_DIR
STAGE = "run"
TRAIN_ARM = "baseline"
GPU = None
EXPERIMENT_SETTINGS = Settings(
    split_seed=42,
    seed=42,
    fractions=(0.8, 0.1, 0.1),
    epochs=10,
    batch_size=16,
    eval_batch_size=16,
    max_length=200,
    learning_rate=6e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    redistribution_learning_rate=1e-3,
    sigma_left=1.0,
    sigma_right=1.0,
    min_sigma=1e-6,
    prediction_alpha=0.307649721979222,
    noise_std_fraction=0.01,
    model_id="Qwen/Qwen3.5-0.8B-Base",
    model_revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    et_model_id="skboy/et_prediction_2",
    et_revision="5785e77309d9fce8b88e908a9db100c1a0a63456",
    et_filename="et_predictor2_seed123.safetensors",
)


def build_parser():
    """Expose every supported numerical setting while keeping fixed operators explicit."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=("run", "prepare", "train", "estimate", "seal", "evaluate", "verify", "check-runtime"), default=STAGE)
    parser.add_argument("--data-dir", type=Path, default=INPUT_DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=ARMS, default=TRAIN_ARM, help="Used only by --stage train.")
    parser.add_argument("--conditions", nargs=4, choices=CONDITIONS, default=CONDITIONS,
                        help="Full run order: baseline companion prediction_mixture redistribution.")
    parser.add_argument("--gpu", type=int, default=GPU, help="Set CUDA_VISIBLE_DEVICES before importing Torch; otherwise preserve it.")
    for name in ("epochs", "seed", "split_seed", "batch_size", "eval_batch_size", "max_length"):
        options = ["--" + name.replace("_", "-")]
        if name == "batch_size":
            options.append("--train-batch-size")
        parser.add_argument(*options, dest=name, type=int, default=None,
                            help=f"Python settings default: {getattr(EXPERIMENT_SETTINGS, name)}")
    parser.add_argument("--fractions", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                        default=None, help="Group split fractions; Python settings default: 0.8 0.1 0.1")
    for name in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm",
                 "redistribution_learning_rate", "sigma_left", "sigma_right", "min_sigma"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=None,
                            help=f"Python settings default: {getattr(EXPERIMENT_SETTINGS, name)}")
    for name in ("prediction_alpha", "noise_std_fraction"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=None,
                            choices=(getattr(EXPERIMENT_SETTINGS, name),),
                            help="Fixed coefficient/operator for this comparison; no search.")
    for name in ("model_id", "model_revision", "et_model_id", "et_revision", "et_filename"):
        parser.add_argument("--" + name.replace("_", "-"), default=None,
                            help=f"Python settings default: {getattr(EXPERIMENT_SETTINGS, name)}")
    parser.add_argument("--gradient-accumulation-steps", type=int, choices=(1,), default=1)
    parser.add_argument("--validation-every-epochs", type=int, choices=(1,), default=1)
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--loss", choices=("mse",), default="mse")
    return parser


def resolve_settings(args):
    """Apply explicit arguments to Python defaults or require equality with an existing lock."""

    if args.stage == "check-runtime":
        return None
    overrides = {field.name: getattr(args, field.name) for field in fields(Settings)
                 if field.name != "schema" and getattr(args, field.name) is not None}
    if args.stage in ("prepare", "run"):
        values = EXPERIMENT_SETTINGS.as_dict()
    else:
        from holdout_experiments.protocol import load_lock
        _, locked = load_lock(args.output)
        values = locked.as_dict()
    values.update(overrides)
    return Settings(**values)


def main(argv=None):
    """Run the explicit Python experiment configuration after applying device visibility."""

    args = build_parser().parse_args(argv)
    if args.gpu is not None:
        if args.gpu < 0:
            raise ValueError("--gpu must be nonnegative.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    settings = resolve_settings(args)
    from holdout_experiments.execution import execute
    result = execute(output=args.output, data_dir=args.data_dir, settings=settings,
                     stage=args.stage, arm=args.arm, conditions=tuple(args.conditions))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()

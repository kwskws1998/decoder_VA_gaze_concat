"""One entry point for preparing, training and finally evaluating the locked suite."""

import argparse
import json
from pathlib import Path

from trt_pipeline.config import DATA_DIR

from .config import ARMS, CONDITIONS, Settings
from .execution import execute


def parser():
    """Expose explicit stages with no search, resume, overwrite or test-based tuning flags."""

    result = argparse.ArgumentParser(description="Locked train/val/test VA experiments, one CUDA GPU.")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name, help="Create a new run" + (" and execute all four comparisons" if name == "run" else " without GPU use"))
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--data-dir", type=Path, default=DATA_DIR)
        command.add_argument("--epochs", type=int, default=10, help="Lock before training; default 10 for every fit.")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--split-seed", type=int, default=42)
        command.add_argument("--conditions", nargs=4, choices=CONDITIONS, default=CONDITIONS)
    command = commands.add_parser("train", help="Train one declared arm using train and validation only")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--arm", choices=ARMS, required=True)
    for name in ("estimate", "seal", "evaluate", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
    commands.add_parser("check-runtime", help="Check pinned packages, CUDA BF16 and FLA dispatch; no model download")
    return result


def main(argv=None):
    """Perform only the user-selected stage and print its concrete receipt."""

    args = parser().parse_args(argv)
    settings = None
    if args.command in ("prepare", "run"):
        settings = Settings(epochs=args.epochs, seed=args.seed, split_seed=args.split_seed)
    result = execute(output=getattr(args, "output", getattr(args, "run", None)),
                     data_dir=getattr(args, "data_dir", DATA_DIR), settings=settings,
                     stage=args.command, arm=getattr(args, "arm", "baseline"),
                     conditions=tuple(getattr(args, "conditions", CONDITIONS)))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return result

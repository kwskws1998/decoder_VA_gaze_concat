"""Run matched raw/fixed/learned TRT experiments using Python only."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess

from sigma_runtime import (
    DEFAULT_ENVIRONMENT, PROJECT_ROOT, check_gpu, child_environment, run, select_python,
)


CONDITIONS = {
    "raw": ["--gaze-redistribution", "none"],
    "fixed": ["--gaze-redistribution", "fixed-gaussian", "--redistribution-sigma-left", "1", "--redistribution-sigma-right", "1"],
    "learned": ["--gaze-redistribution", "asym-gaussian", "--redistribution-sigma-left", "1", "--redistribution-sigma-right", "1", "--redistribution-learning-rate", "1e-3"],
}


def build_parser():
    """Keep dataset and experiment settings explicit and shared between conditions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "smoke", "dry-run"), default="full")
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "va_model_code/data/paper7_seed42")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--diagnostics-steps", type=int, default=50)
    parser.add_argument("--probe-batch-size", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--python", type=Path, default=None, help="Optional Python executable in an already prepared environment.")
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_ENVIRONMENT)
    return parser


def build_commands(args, python):
    """Build reproducible Python argument lists and distinct output names."""

    positive = (args.batch_size, args.gradient_accumulation_steps, args.epochs, args.diagnostics_steps, args.probe_batch_size)
    if any(not math.isfinite(value) or value <= 0 for value in positive) or (args.eval_batch_size is not None and args.eval_batch_size <= 0):
        raise ValueError("Batch sizes, epochs, accumulation and diagnostics intervals must be positive.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be an integer in [0, 2**32).")
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu must be nonnegative.")
    suite = args.suite_name or datetime.now().strftime("sigma_%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", suite):
        raise ValueError("Suite name must contain only letters, digits, dots, underscores and hyphens.")
    common = [
        str(python), "-u", str(PROJECT_ROOT / "va_model_code/train_model.py"),
        "qwen3.5-0.8b", "mse", "--data-dir", str(args.data_dir.expanduser().resolve()),
        "--finetuning-mode", "full", "--precision", args.precision,
        "--gaze-fusion", "prefix-concat", "--gaze-features", "TRT",
        "--sentence-only", "--no-iemocap", "--held-out-folds", "1",
    ]
    if args.mode != "smoke":
        common.append("2")
    common += [
        "--seed", str(args.seed), "--epochs", str(args.epochs), "--max-length", "200",
        "--train-batch-size", str(args.batch_size), "--eval-batch-size", str(args.eval_batch_size or args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--gradient-checkpointing", "--learning-rate", "6e-6", "--weight-decay", "0.01",
        "--warmup-ratio", "0.1", "--logging-steps", "50", "--et-cache-size", "70000",
        "--save-total-limit", "1", "--sigma-diagnostics-steps", str(args.diagnostics_steps),
        "--sigma-diagnostics-batch-size", str(args.probe_batch_size),
    ]
    if args.mode == "smoke":
        common += ["--max-steps", "20"]
    selected = CONDITIONS if args.condition == "all" else (args.condition,)
    plans = []
    for condition in selected:
        run_name = f"{suite}_{args.mode}_{condition}_sentence_only_no_iemocap_qwen_full_gaze_TRT_{args.precision}_seed{args.seed}"
        destination = PROJECT_ROOT / "results" / run_name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Existing run: {destination}. Choose another --suite-name.")
        plans.append({"condition": condition, "run_name": run_name, "command": common + CONDITIONS[condition] + ["--run-name", run_name]})
    return plans


def prepare_data(directory, python, environment):
    """Reuse a complete fold pair or prepare the checksum-pinned source bundle."""

    directory = directory.expanduser().resolve()
    present = [(directory / f"full_dataset_fold{fold}.csv").is_file() for fold in (1, 2)]
    if any(present) and not all(present):
        raise FileNotFoundError(f"Only one fold file exists in {directory}. Supply a complete pair.")
    if not any(present):
        run([
            python, PROJECT_ROOT / "va_model_code/prepare_english_data.py", "--download-default",
            "--download-path", PROJECT_ROOT / "va_model_code/data/external/english_va_bundle.zip",
            "--paper-protocol", "--seed", "42", "--output-dir", directory,
        ], environment=environment)
    if not all((directory / f"full_dataset_fold{fold}.csv").is_file() for fold in (1, 2)):
        raise FileNotFoundError(f"Data preparation did not produce both folds in {directory}.")


def main(argv=None):
    """Check the current GPU and run selected conditions sequentially with failure propagation."""

    args = build_parser().parse_args(argv)
    python = select_python(args.python, args.venv_dir)
    plans = build_commands(args, python)
    if args.mode == "dry-run":
        print(json.dumps(plans, indent=2))
        return
    runtime = check_gpu(python, gpu=args.gpu)
    if args.precision == "bf16" and not runtime["bf16_supported"]:
        raise RuntimeError("This GPU does not support BF16. Select --precision fp32 for all compared conditions.")
    print(json.dumps(runtime, indent=2))
    environment = child_environment(args.gpu)
    prepare_data(args.data_dir, python, environment)
    for plan in plans:
        run(plan["command"], environment=environment)
        if args.mode == "full":
            run([python, PROJECT_ROOT / "va_model_code/package_results.py", "--run-name", plan["run_name"]], environment=environment)
    print("Completed:", ", ".join(plan["run_name"] for plan in plans))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Sigma experiment failed: {exc}") from exc

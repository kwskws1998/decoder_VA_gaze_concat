"""Create a Python environment and install the sigma experiment dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import venv

from sigma_runtime import (
    DEFAULT_ENVIRONMENT, PROJECT_ROOT, check_gpu, choose_torch_index,
    detect_nvidia_driver, environment_python, run,
)


def build_parser():
    """Expose Python-native setup with optional interpreter and wheel overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--current-env", action="store_true", help="Install into the Python running this script instead of creating .venv.")
    parser.add_argument("--torch-index", choices=("auto", "cu126", "cu130"), default="auto")
    parser.add_argument("--gpu", type=int, default=None, help="Optional physical GPU index; otherwise preserve CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--dry-run", action="store_true", help="Print installation steps without creating an environment or installing packages.")
    return parser


def main(argv=None):
    """Install with the selected Python and verify CUDA on the current machine."""

    args = build_parser().parse_args(argv)
    if sys.version_info < (3, 11):
        raise RuntimeError("Run setup_environment.py with Python 3.11 or newer.")
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu must be nonnegative.")
    driver = detect_nvidia_driver() if args.torch_index == "auto" else None
    index = choose_torch_index(driver["driver_cuda"]) if driver else args.torch_index
    python = Path(sys.executable) if args.current_env else environment_python(args.venv_dir)
    commands = [
        [python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [python, "-m", "pip", "install", "--upgrade", f"torch==2.12.1+{index}", "--index-url", f"https://download.pytorch.org/whl/{index}"],
        [python, "-m", "pip", "install", "-r", PROJECT_ROOT / "requirements.txt"],
        [python, "-m", "pip", "check"],
    ]
    print(json.dumps({"python": str(python), "torch_index": index, "nvidia_driver": driver}, indent=2))
    if args.dry_run:
        for command in commands:
            print(" ".join(str(part) for part in command))
        return
    if not args.current_env:
        destination = args.venv_dir.expanduser().resolve()
        if destination.exists() and not (destination / "pyvenv.cfg").is_file():
            raise ValueError(f"Refusing to reuse a directory that is not a virtual environment: {destination}")
        if not destination.exists():
            venv.EnvBuilder(with_pip=True).create(destination)
    for command in commands:
        run(command)
    runtime = check_gpu(python, gpu=args.gpu)
    record = {"python": str(python), "torch_index": index, "driver": driver, "runtime": runtime}
    (PROJECT_ROOT / "environment_setup.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runtime, indent=2))
    print("Environment ready. Run: python run_sigma_experiments.py --mode smoke")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Environment setup failed: {exc}") from exc

"""Optional checks for an active conda environment; requirements.txt is the only dependency list.

Normal installation is: python -m pip install -r requirements.txt
This helper does not create environments or choose a separate package profile.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from environment_runtime import PROJECT_ROOT, check_gpu, child_environment, require_active_conda, run


def install_commands(python):
    """Use the exact same single requirements-file installation documented for manual use."""

    return [[python, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
            [python, "-m", "pip", "check"]]


def main(argv=None):
    """Optionally install requirements and check CUDA without starting any experiment fit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--check-only", action="store_true", help="Do not install; only check the active environment.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without installation or GPU execution.")
    args = parser.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu must be nonnegative.")
    ownership = require_active_conda()
    if ownership["python_version"][:2] < [3, 11]:
        raise RuntimeError("Use Python 3.11 or newer in the active conda environment.")
    python = sys.executable
    commands = [] if args.check_only else install_commands(python)
    kernel_check = [python, "-m", "trt_pipeline.check_training_kernels", "--require-fla", "--smoke-test",
                    "--output", str(PROJECT_ROOT / "results/training_kernel_check.json")]
    runtime_check = [python, "run_experiments.py", "--stage", "check-runtime"]
    if args.dry_run:
        for command in [*commands, kernel_check, runtime_check]:
            print(shlex.join(command))
        return {"dry_run": True, "conda": ownership}
    for command in commands:
        run(command)
    runtime = check_gpu(python, gpu=args.gpu)
    run(kernel_check, environment=child_environment(args.gpu))
    run(runtime_check, environment=child_environment(args.gpu))
    record = {"environment_manager": "conda", "conda": ownership, "runtime": runtime,
              "dependency_specification": "requirements.txt"}
    Path(PROJECT_ROOT / "environment_setup.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Environment check failed: {error}") from error

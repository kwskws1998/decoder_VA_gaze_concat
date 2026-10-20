"""Run the frozen experimental stack with isolated imports and shared results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


ARCHIVE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ARCHIVE_ROOT.parent
CODE_ROOT = ARCHIVE_ROOT / "va_model_code"
ENTRYPOINTS = {
    "train": "train_model.py",
    "package": "package_results.py",
    "evaluate-external": "evaluate_external.py",
    "evaluate-relation": "evaluate_gaze_relation.py",
    "diagnose-relation": "run_gaze_relation_diagnostic.py",
    "benchmark-rotation": "benchmark_gaze_rotation.py",
}


def verify_snapshot():
    """Verify every archived source against its pre-refactor SHA-256."""

    manifest = json.loads((ARCHIVE_ROOT / "SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        source = ARCHIVE_ROOT / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing or unsafe archived source: {name}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archived source changed: {name}")
    return manifest


def main(argv=None):
    """Select only archived modules, while keeping original data/results in place."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(*ENTRYPOINTS, "verify"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    manifest = verify_snapshot()
    if args.command == "verify":
        print(f"Verified {len(manifest['files'])} files from {manifest['source_commit']}.")
        return 0
    if any(name == "decoder_va" or name.startswith("decoder_va.") for name in sys.modules):
        raise RuntimeError("Start this launcher in a fresh Python process to isolate archived imports.")
    sys.path[:0] = [str(CODE_ROOT), str(ARCHIVE_ROOT)]
    from decoder_va import paths

    original_resolve = paths.resolve_run_directory
    shared_results = PROJECT_ROOT / "results"

    def resolve_run_directory(run_name, *, results_root=shared_results):
        """Keep the archive's original safety checks with the canonical results root."""

        return original_resolve(run_name, results_root=results_root)

    paths.PROJECT_ROOT = PROJECT_ROOT
    paths.RESULTS_ROOT = shared_results
    paths.resolve_run_directory = resolve_run_directory
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if args.command == "train" and not any(arg.split("=")[0] == "--data-dir" for arg in arguments):
        arguments = [*arguments, "--data-dir", str(PROJECT_ROOT / "va_model_code/data/paper7_seed42")]
    entrypoint = CODE_ROOT / ENTRYPOINTS[args.command]
    sys.argv = [str(entrypoint), *arguments]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

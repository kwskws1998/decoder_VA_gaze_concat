# Legacy experiment code

This directory freezes the experimental Qwen VA stack before the baseline-only
refactor. It contains redistribution, TRT attention rotation, learned readout,
heteroscedastic Gaussian NLL and their original dependencies, tests and diagnostics.
The older encoder/GazeReward archive remains separately in `../legacy/`.

## Module map

| Experiment | Archived modules under `va_model_code/decoder_va/` |
| --- | --- |
| Fixed/learned Gaussian redistribution | `redistribution.py`, `sigma_diagnostics.py` |
| TRT relative/readout attention rotation | `gaze_relation.py`, `gaze_attention.py`, `gaze_relation_inputs.py`, `gaze_relation_reporting.py`, `gaze_relation_diagnostics.py` |
| Learned prediction token | `readout.py`, integration in `packing.py` and `model.py` |
| Heteroscedastic variance head and NLL | `uncertainty.py`, integration in `model.py`, `losses.py` and `evaluation.py` |

The shared model/trainer/evaluator source is frozen alongside these modules.
This duplication is deliberate: future baseline changes cannot silently alter
an old experiment. There are no imports from this directory in active code.

## Run from the repository root

Use the configured Python environment and the wrapper, which isolates imports
and retains the canonical `results/` location:

```bash
python legacy_code/run.py verify
python legacy_code/run.py train --help
python legacy_code/run.py package --run-name YOUR_EXISTING_EXPERIMENT_RUN
python legacy_code/run.py evaluate-relation --help
python legacy_code/run.py evaluate-external --help
python legacy_code/run.py diagnose-relation --help
python legacy_code/run.py benchmark-rotation --help
```

To rerun an experiment, take its original training arguments and replace
`python train_model.py` with `python legacy_code/run.py train`. Supply a new run
name. Data remains in `va_model_code/data/paper7_seed42` at the repository root;
the wrapper uses that absolute directory when `--data-dir` is omitted. Explicit
paths are resolved relative to your terminal's current directory. Existing
results and model weights are not moved or copied by the wrapper.

Do not run the archived training/packaging entry points directly if you want
shared results: their unmodified relative-root logic points to `legacy_code/results/`.
The original `run_sigma_experiments.py` and `setup_environment.py` are historical
snapshots, not the current setup workflow. Use the root installer for the active
pinned environment. Future environment changes may require the archived
`requirements.txt` in a separate environment to reproduce these experiments.

## Provenance and tests

`SOURCE_MANIFEST.json` records the original Git commit and SHA-256 of every
archived source/report. `run.py` verifies this snapshot before dispatch. The
pre-refactor root README is preserved byte-for-byte as `SNAPSHOT_README.md`.
Historical experiment instructions and diagnostic reports contain the paths
and assumptions used at that time; some diagnostic scripts require historical
local input files. They are reference material, not new verified experiment runs.

Run archived tests in a separate Python process and directory:

```bash
cd legacy_code
python -m pytest -q -c /dev/null va_model_code/tests
```

Do not collect active and archived tests in one pytest process: both frozen and
active suites intentionally use their own `va_model_code` package namespace.
The baseline regression suite separately loads the archived model under a
qualified namespace only to verify numerical parity.

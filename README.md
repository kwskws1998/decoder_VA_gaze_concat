# Decoder-based VA prediction

The repository has one active code directory, one legacy directory, and one
generated-results directory:

```text
va_model_code/   active Qwen VA training, data preparation, and tests
legacy/          archived encoder VA code, GazeReward reference, paper evidence
results/         every current training run and its results-only ZIP
```

Only the repository-root `requirements.txt` defines the active environment.
Nothing under `legacy/` is imported or installed.

## Install on a 24 GB NVIDIA machine

Run installation from the repository root:

```bash
conda create -n decoder-va python=3.11 pip -y
conda activate decoder-va

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

## Prepare data

Keep generated datasets below the active directory's single `data/` root:

```bash
python va_model_code/prepare_english_data.py \
  --download-default \
  --download-path va_model_code/data/external/english_va_bundle.zip \
  --output-dir va_model_code/data

python va_model_code/prepare_english_data.py \
  --download-default \
  --download-path va_model_code/data/external/english_va_bundle.zip \
  --paper-protocol \
  --seed 42 \
  --output-dir va_model_code/data/paper7_seed42
```

## Verify

Run tests from the repository root:

```bash
python -m pytest -q va_model_code/tests

python va_model_code/train_model.py \
  --list-datasets \
  --data-dir va_model_code/data/paper7_seed42
python va_model_code/train_model.py --dry-run \
  --data-dir va_model_code/data/paper7_seed42 \
  --finetuning-mode full \
  --gaze-fusion prefix-concat \
  --gaze-features TRT
```

## Result-path contract

Training accepts a single directory name through `--run-name`. Regardless of
whether the command is launched from the repository root or `va_model_code`,
the output is always:

```text
<repository>/results/<run-name>/
```

Arbitrary training `--output-dir` paths and the old `Preds/` layout are
intentionally unsupported. The data-preparation CLI still uses `--output-dir`
for its dataset destination. A condition-aware run name is generated when
`--run-name` is omitted. Names containing `baseline` or `gaze` are checked
against the actual `--gaze-fusion` setting before any model is downloaded.

After both held-out folds complete, create a small verified ZIP inside the same
run directory:

```bash
python va_model_code/package_results.py --run-name <run-name>
```

The packager accepts only the complete two-fold OOF result. The archive name is
derived from `training_parameters.json`, not from the directory name. Model
weights, optimizer states, and other large checkpoint files are excluded by
construction.

## One-time migration of an old Vast.ai checkout

Stop active training before moving completed runs. From the repository root,
inspect the old locations and move each run by its exact name:

```bash
find . -maxdepth 3 -type d -name Preds -print

mkdir -p results
mv va_model_code/Preds/<completed-run-name> results/<completed-run-name>

mkdir -p va_model_code/data
mv va_model_code/data_paper7_seed42 va_model_code/data/paper7_seed42
```

Do not use a wildcard for these moves: an existing destination must be
inspected rather than overwritten. New runs no longer create either legacy
location.

See `va_model_code/README.md` for model design and complete baseline/gaze
commands.

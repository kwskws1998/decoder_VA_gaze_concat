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

## Optional gaze redistribution

Gaze concat and gaze redistribution are separate settings. Existing commands
still use raw ET2 features because `--gaze-redistribution none` is the default.
To learn an asymmetric Gaussian TRT transformation before the same gaze-prefix
projection, add:

```bash
python va_model_code/train_model.py --dry-run \
  --data-dir va_model_code/data/paper7_seed42 \
  --finetuning-mode full \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
  --gaze-redistribution asym-gaussian \
  --redistribution-sigma-left 1.0 \
  --redistribution-sigma-right 1.0
```

The independent `decoder_va/redistribution.py` module uses the mapped gaze mask
for both source and destination tokens, including interior unmapped positions.
It introduces two learned Gaussian-width parameters and leaves frozen ET2,
the other selected gaze channels, data preprocessing, and evaluation unchanged.
Configuration is recorded in training and architecture manifests; learned
parameters are stored with the model weights. Existing checkpoints keep their
original raw-gaze behavior: train a new condition to enable redistribution.

See [the implementation details and full BF16 command](va_model_code/README.md#gaze-redistribution)
for mask semantics, source provenance, and the FP32 variant. GPU memory for the
new condition has not been measured; 24 GB is not a verified memory guarantee.

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

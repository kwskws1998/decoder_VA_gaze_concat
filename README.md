# Decoder-based VA prediction

The active model is Qwen VA regression, with either text only or the original
frozen ET2 raw gaze prefix (TRT by default). Both predict valence/arousal from
the **last valid text token** using the original two-output regression head.

```text
va_model_code/          active baseline, raw TRT concat, data tools and tests
diffusion_code/         isolated conditional DDPM VA experiment and tests
setup_environment.py   Python environment installer and CUDA/kernel checks
environment_runtime.py shared environment helpers
requirements.txt       pinned active dependencies
results/               existing and new run outputs
legacy_code/           frozen redistribution, rotation, readout and NLL experiments
legacy/                older encoder model, GazeReward reference and paper evidence
```

Active code never imports either archive. The refactor preserves current ET2
alignment, FP32 trainable parameters/head, BF16/FP16 autocast, sequential evaluation,
fold filtering, configurable training-text overlap, checkpoint selection and FLA
kernel support. It does not revert to an older preprocessing or precision policy.
Existing data and trained results stay in place.

## Environment and data

Run from the repository root using Python 3.11 or newer. On a fresh server, clone
this repository first. The setup script detects the NVIDIA driver and verifies
CUDA execution; it does not assume a particular GPU model.

```bash
python setup_environment.py --current-env
python va_model_code/prepare_english_data.py \
  --download-default \
  --download-path va_model_code/data/external/english_va_bundle.zip \
  --paper-protocol --seed 42 \
  --output-dir va_model_code/data/paper7_seed42
```

Omit `--current-env` to create the project `.venv`; then use `.venv/bin/python`
for subsequent commands. An already configured environment can repair only its
FLA installation with `python setup_environment.py --current-env --kernels-only`.
No environment reinstall is needed solely for this source-code separation.
The kernel smoke check exercises native hybrid Qwen forward/backward and gradient
checkpoint replay, without any archived attention modifications.

## Train the original raw TRT concat

Run from the repository root in the configured environment:

```bash
python -u va_model_code/train_model.py qwen3.5-0.8b mse \
  --data-dir va_model_code/data/paper7_seed42 \
  --finetuning-mode full \
  --precision bf16 \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
  --et-model-id skboy/et_prediction_2 \
  --no-iemocap \
  --training-text-overlap keep \
  --held-out-folds 1 2 \
  --max-length 200 \
  --train-batch-size 8 --eval-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 10 --max-steps -1 \
  --learning-rate 6e-6 --weight-decay 0.01 --warmup-ratio 0.1 \
  --logging-steps 200 --save-total-limit 1 \
  --group-by-length --gradient-checkpointing \
  --seed 42
```

The automatically generated run name prevents reusing an old result directory.
For a text-only control, use `--gaze-fusion none`. The original `mse`, `ccc` and
`mse+ccc` losses remain available. `--sentence-only` is still an explicit option;
this example includes all non-IEMOCAP sources. Overlapping training texts are
retained by the explicit `--training-text-overlap keep` argument.

Use the printed run name to package both completed folds without weights:

```bash
python va_model_code/package_results.py --run-name YOUR_RUN_NAME
```

The packager still verifies fold completeness, metadata, OOF predictions,
recomputed metrics, hashes and ZIP integrity. Existing baseline checkpoint
metadata is supported when alignment and precision contracts match.

## Archived experiments

Redistribution, added TRT rotations, learned readout and heteroscedastic NLL are
**absent from the active CLI and forward pass**. Experimental weights and metadata
are rejected by the active loader; they are never silently treated as Raw TRT.
`baseline_contracts.py` only keeps disabled metadata identities for compatibility.

Use [legacy_code/README.md](legacy_code/README.md) for module locations and the
isolated Python launcher. For example, to package an existing experimental run:

```bash
python legacy_code/run.py package --run-name YOUR_EXPERIMENT_RUN_NAME
```

Both launchers use repository-root `results/`. Experimental source, tests,
requirements, commands and research diagnostics are preserved in the archive,
with pre-refactor SHA-256 hashes and the original Git commit.

## Conditional DDPM experiment

The new [diffusion_code/README.md](diffusion_code/README.md) documents the exact
epsilon-prediction objective, stepwise text/TRT conditioning, stochastic sampler,
matched controls, setup, training and results-only packaging. Its entry point is
`python diffusion_code/train.py`. The baseline has no diffusion imports; only
its encoder feature extraction is shared. The frozen archive remains unchanged.

## Validation

```bash
python -m pytest -q
python legacy_code/run.py verify
```

Default pytest discovery runs baseline and new diffusion tests. To verify the frozen stack in
its own import context:

```bash
cd legacy_code
python -m pytest -q -c /dev/null va_model_code/tests
```

The refactor tests compare initial weights, dropout RNG, FP32/BF16 predictions,
gradients and optimizer updates against the frozen implementation, including a
small native hybrid Qwen model with gradient checkpointing. These are local
regression checks, not a new full-dataset GPU experiment.

# Conditional DDPM for VA with TRT

Completed-run diagnostics, a current-code Raw TRT control, and the separate
frozen-mean residual DDPM experiment are documented in
[FOLLOWUP_EXPERIMENTS.md](FOLLOWUP_EXPERIMENTS.md). The original DDPM below
retains its version-1 training and sampling semantics.

This is a new experiment, isolated from `va_model_code/train_model.py` and the
frozen `legacy_code/` experiments. The baseline still uses its original raw TRT
prefix and last-valid-text-token regression head. This experiment reuses that
encoder and replaces the regression head with a small conditional denoiser.

## Exact method

- **Target:** the two normalized labels `(valence, arousal)`, transformed from
  `[0, 1]` to `[-1, 1]` by `z = 2 * labels - 1`.
- **Forward path:** `x_t = sqrt(alpha_bar_t) * z + sqrt(1-alpha_bar_t) * epsilon`.
  There are 100 discrete timesteps by default, using the cosine cumulative
  schedule with offset `0.008`, clipping each beta at `0.999`. The finite
  terminal distribution is approximately standard Gaussian.
- **Training:** independently sample one timestep and Gaussian epsilon per
  example; minimize mean squared epsilon prediction error across both VA axes.
  This is not Gaussian NLL, flow matching, or a direct VA MSE training objective.
- **Sampling:** all discrete DDPM reverse steps, using analytical posterior
  coefficients and fixed posterior variance. Estimated clean VA is clipped to
  `[-1, 1]` each step. The final step adds no random noise.
- **Prediction:** inverse-transform generated samples, clamp to `[0, 1]`,
  then average 16 samples per input by default. Every epoch uses this same
  free-generation procedure for VA evaluation and checkpoint selection.
- **Precision:** Qwen uses the selected autocast precision. Trainable parameters,
  the entire diffusion head, noise MSE and reverse-process arithmetic stay FP32.
- **Optimization:** the new head defaults to LR `1e-4`; Qwen and the gaze
  projector use `--learning-rate` (full tuning defaults to `6e-6`). Both groups
  use the existing warmup/decay schedule and ordinary bias/normalization decay
  exclusions. No additional pretrained weights or diffusion library are needed.

In the supplied *An Introduction to Flow Matching and Diffusion Models*,
this corresponds to §4.3, p.32, Algorithm 4's epsilon-prediction alternative,
combined with §5.1, p.34, vanilla conditioning. The notes describe a predictor
that “learns to predict the noise that was used to corrupt a data sample”.
Their time direction is noise at zero to data at one; this implementation uses
the conventional DDPM noising indices and iterates them in reverse at generation.
The continuous SDE description in §4.2 is an interpretation, not an additional
SDE solver used by this code. See also [DDPM](https://arxiv.org/abs/2006.11239),
equations 4, 7 and 11–14 and Algorithms 1–2.

## How TRT enters

`--conditioning token-trt` is the proposed method:

1. Frozen ET2 predicts raw TRT with the existing alignment. The existing
   `[eye_start, gaze prefix, eye_end, text]` sequence goes through Qwen once.
2. The final valid text representation supplies a global context token.
3. Each valid TRT anchor pairs its contextual **text-token** representation
   with its aligned scalar TRT. Separate learned projections combine them into
   memory vectors. Padding and non-anchor positions are excluded.
4. At every reverse step, a query formed from noisy VA, timestep and global
   context attends to that memory. TRT affects both keys and values; a single
   anchor can contribute. With no anchors, the global context remains valid.
5. Keys/values are projected once and reused; Qwen is not rerun for each
   denoising step or Monte Carlo sample.

There is no fixed assumption that larger TRT must get larger attention.
TRT conditions the epsilon predictor; it does not set the diffusion noise scale.
These are frozen ET2 predictions derived from the text, not new measured gaze.

## Controls

| Condition | Entry point/options | Purpose |
| --- | --- | --- |
| Existing Raw TRT | `va_model_code/train_model.py ... mse` | Direct-regression baseline |
| DDPM with pooled Raw TRT context | `diffusion_code/train.py --conditioning pooled` | Effect of replacing the regression head |
| DDPM with stepwise word/TRT memory | `diffusion_code/train.py --conditioning token-trt` | Additional contribution of direct word/TRT conditioning |

For an optional text-only DDPM, combine `--gaze-fusion none --conditioning pooled`.
The pooled and token-TRT conditions have different parameter counts.
Use identical data, code, training precision, seeds and encoder settings for
comparisons. Historical results alone do not isolate the new head's effect.

## Environment and data

From the repository root, in the intended Python environment:

```bash
python setup_environment.py --current-env

python va_model_code/prepare_english_data.py \
  --download-default \
  --download-path va_model_code/data/external/english_va_bundle.zip \
  --paper-protocol --seed 42 \
  --output-dir va_model_code/data/paper7_seed42
```

The existing installer checks CUDA and FLA. An already installed environment
needs no new dependencies for DDPM. Data must be prepared on a fresh GPU machine.
This first implementation supports one process and one visible GPU; select one
GPU through `CUDA_VISIBLE_DEVICES` when the machine exposes multiple GPUs.

## Proposed run: one seed, both held-out folds

The following starts only the proposed token-TRT DDPM arm. It retains all
non-IEMOCAP sources and overlapping training texts, following the current
requested protocol. It does not activate sentence-only filtering.

```bash
python -u diffusion_code/train.py \
  --data-dir va_model_code/data/paper7_seed42 \
  --finetuning-mode full \
  --precision bf16 \
  --gaze-fusion prefix-concat \
  --conditioning token-trt \
  --diffusion-steps 100 \
  --diffusion-samples 16 \
  --diffusion-width 128 \
  --diffusion-blocks 3 \
  --head-learning-rate 1e-4 \
  --sampling-seed 2026 \
  --et-model-id skboy/et_prediction_2 \
  --no-iemocap \
  --training-text-overlap keep \
  --held-out-folds 1 2 \
  --max-length 200 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 10 \
  --learning-rate 6e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --logging-steps 200 \
  --save-total-limit 1 \
  --group-by-length \
  --gradient-checkpointing \
  --seed 42
```

An automatic timestamped run name is printed. Add `--dry-run` to validate data,
options and TrainingArguments without downloading Qwen or ET2. A short
`--max-steps 10 --held-out-folds 1` run checks the real GPU lifecycle, not efficacy.
The default does not force eager attention; existing Qwen/FLA support remains.
Measured GPU runtime and VA improvement are not established by local CPU tests.

## Logs, reproducibility and checkpoint selection

- `loss` is training **epsilon MSE**, not comparable in scale to the old VA MSE.
- `eval_loss` and `eval_mse_mean` are **generated-VA MSE in original [0,1] units**.
  Pearson, RMSE, MAE and CCC retain the baseline definitions.
- Best-checkpoint selection uses minimum `eval_mse_mean`, never training noise
  MSE. Gold evaluation labels are not passed into the generative model.
- The evaluator uses a private CPU-generated common Gaussian noise bank with
  a saved seed. Every input uses the same bank, so changing batch sizes/order
  does not assign different random draws to examples. It does not consume the
  training RNG. Floating-point backend differences can still change results.
- Sampling variability is not automatically calibrated uncertainty. One VA
  annotation per input does not establish a distribution of human ratings.
- Configuration, code hashes, input-fold hashes, overlap policy, package/kernel
  versions, precision, timestep schedule and sampling seed are persisted.
- `trainer_state.json` records epoch metrics and the best checkpoint. Every
  checkpoint contains its own DDPM architecture and training contract.

Full checkpoint resume requires identical semantic/source contracts and
optimizer/scheduler/RNG files. Pass the original `--run-name`, all original
options, a single `--held-out-folds N`, and
`--resume-from-checkpoint results/RUN/heldout_foldN/checkpoints/checkpoint-STEP`.
Previously completed other folds are retained when rebuilding OOF reports.
Final-model weights alone are inference artifacts, not full resume checkpoints.

## Reload and package

```bash
python diffusion_code/evaluate.py \
  --model-dir results/RUN_NAME/heldout_fold1/final_model \
  --data-file va_model_code/data/paper7_seed42/full_dataset_fold1.csv \
  --output-dir results/RUN_NAME/reloaded_fold1 \
  --precision bf16

python diffusion_code/package_results.py --run-name RUN_NAME
```

The reload evaluator defaults to the saved training truncation length. An
external evaluation file is evaluated as supplied; use the same filtered data
to reproduce training-fold metrics. Its report marks this as external evaluation,
not a new OOF experiment.

The dedicated packager requires all folds originally requested by that run to
finish, checks checkpoint state, prediction hashes, OOF union and recomputed
metrics, then archives reports/configuration only. Weights and optimizer files
are excluded. The original baseline packager deliberately does not handle DDPM.

## Validation

```bash
python -m pytest -q diffusion_code/tests
python -m pytest -q
```

Tests cover analytical posterior coefficients, final-step noise handling,
normalization, label isolation, batch/order-invariant sampling noise,
single-anchor TRT gradients, masking, FP32/BF16 and native small-Qwen backward
with gradient checkpointing, optimizer coverage, paired baseline parity, and
real Trainer save/reload/report/ZIP lifecycle with small local fixtures.
Synthetic tests validate execution and learnability only; they do not establish
real-data performance or superiority over direct regression.

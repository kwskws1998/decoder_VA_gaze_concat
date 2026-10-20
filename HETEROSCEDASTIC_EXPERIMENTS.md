# Heteroscedastic VA experiments

## Requested comparison

| Condition | Decoder input | Prediction position | Objective |
|---|---|---|---|
| VA + heteroscedastic | Text only | Last valid text token | Gaussian NLL |
| TRT gaze concat + heteroscedastic | `eye_start`, compact raw TRT gaze, `eye_end`, text | Last valid text token | Same Gaussian NLL |

Both use the existing Qwen VA mean head and original last-text-token pooling.
The second arm uses the existing frozen ET2 TRT prefix. Neither arm enables
redistribution, TRT rotation, or a learned readout token. Adding a variance head
is necessary for input-dependent uncertainty; replacing MSE alone cannot create
an input-dependent variance prediction.

The commands retain **all non-IEMOCAP data**, including word ratings, and keep
training texts that also occur in the other fold. There is no `--sentence-only`
filter or new validation split. The existing two-fold checkpoint-selection
protocol is retained as requested. Base seed42 means fold1 uses seed42 and
fold2 uses seed43, following the existing code; it is one two-fold experiment
per condition, not two independent base-seed experiments.

## Model and objective

The output columns are exactly:

```text
[mean_valence, mean_arousal, log_variance_valence, log_variance_arousal]
```

The existing `LayerNorm -> Dropout -> Linear(2) -> hard_sigmoid` mean path stays
unchanged. A separate FP32 `Linear(hidden_size, 2)` uses the same normalized
pooled features to predict variance, without adding dropout to that branch.
Its weights start at zero and its bias gives initial variance `0.01` (standard
deviation `0.1`) in normalized VA units. This initialization does not consume
the existing mean/projector initialization RNG.

For each VA dimension, the implementation uses:

```text
variance = softplus(raw_variance) + 1e-6
s = log(variance)
NLL = 0.5 * (log(2*pi) + s + (target - mean)^2 * exp(-s))
loss = mean(NLL over examples and both VA dimensions)
```

There are no uncertainty labels. The head learns conditional residual variance
from the VA supervision. This is not model-weight/epistemic uncertainty, an ET2
confidence score, or a measurement of human rating variance. In particular,
systematic mean errors can also contribute to the learned residual variance.
The two VA dimensions use a diagonal Gaussian: no covariance is learned.

BF16 remains the decoder compute precision in the commands below. Trainable
parameters, both heads, and the NLL calculations remain FP32 under the existing
mixed-precision policy. Only the two means use hard sigmoid. The log variances
can be negative; **a negative Gaussian NLL is also valid**. Do not compare its
numeric magnitude with a previous MSE training loss.

The variance head uses the same base optimizer learning rate as the existing
head (`6e-6` here), with the existing weight/bias decay grouping. The initial
variance changes the mean-gradient scale relative to MSE; this is part of the
new objective, not evidence that the old learning rate is optimal for it.

## Selection, evaluation and saved results

- The best checkpoint still minimizes **`eval_mse_mean`**, not NLL.
- `loss` and `eval_loss` are Gaussian NLL. Evaluation also reports the existing
  VA MSE, RMSE, MAE, Pearson and CCC from the mean predictions only.
- Additional metrics include `gaussian_nll_mean`, per-dimension NLL, average
  predicted variance/std, `standardized_mse`, variance/error Spearman, 90%/95%
  coverage, and average interval widths. Undefined correlations are JSON null.
- Coverage uses the untruncated intervals `mean +/- z * std`. Intervals are not
  clipped to `[0,1]`; the Gaussian is an approximation on the bounded VA scale.
  Coverage alone can be inflated by very broad intervals, so inspect NLL,
  interval widths and mean accuracy together.
- Each fold's `predictions.tsv` and root `oof_predictions.tsv` include both log
  variances, both variances and both standard deviations. Existing mean columns
  retain their names and order. Dataset metrics include uncertainty metrics.
- `package_results.py` recomputes metrics and validates derived variance/std
  columns, fold consistency and OOF row identity. ZIP names include
  `heteroscedastic`; weights and optimizer binaries are excluded as before.
- Architecture schema11 records the complete prediction distribution in the
  root, fold and model metadata. Earlier point-prediction models retain their
  original meaning. A point-prediction checkpoint cannot resume as a Gaussian
  model; use a fresh run directory. Matching Gaussian resumes restore both
  variance parameters and optimizer moments.
- External benchmark inference uses the two conditional means. External
  uncertainty calibration and mixture/ensemble variance are not evaluated by
  that existing benchmark command.

This pair compares the effect of TRT under Gaussian NLL. To isolate the effect
of replacing MSE, compare with matching current-code MSE runs using the same
input condition and all other settings. Historical runs can provide context
but do not establish that the new loss caused a performance difference.

## Environment and data setup

First publish the local implementation to your GitHub `main` yourself. The
following block is for a fresh Linux GPU server. On an existing matching
`decoder-va` environment it reuses the environment and cached downloads. The
`cu130` index follows the previously used 5090/CUDA13-compatible driver setup;
`setup_environment.py` verifies the actual CUDA and FLA forward/backward path.
Check disk space before retaining four full fold trainings and optimizer state;
`--save-total-limit 1` limits checkpoints per fold, not total disk usage.

```bash
(
set -e
set -o pipefail

cd /workspace
if [ ! -d decoder_VA_gaze_concat/.git ]; then
  git clone https://github.com/kwskws1998/decoder_VA_gaze_concat.git
fi
cd decoder_VA_gaze_concat
git pull --ff-only origin main

python - <<'PY'
from pathlib import Path
if not Path("va_model_code/decoder_va/uncertainty.py").is_file():
    raise SystemExit("Heteroscedastic code is missing. Publish the local changes, then pull again.")
PY

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda run -n decoder-va python --version >/dev/null 2>&1; then
  conda create -n decoder-va python=3.11 pip -y
fi
conda activate decoder-va
mkdir -p results
nvidia-smi
df -h /workspace

python -u setup_environment.py --current-env --torch-index cu130 \
  2>&1 | tee "results/heteroscedastic_setup_$(date +%Y%m%d_%H%M%S).log"

cd va_model_code
python -u prepare_english_data.py \
  --download-default \
  --paper-protocol \
  --seed 42 \
  --output-dir data/paper7_seed42
)
```

## Train both conditions sequentially, single base seed42

Paste this entire block once. It runs text-only folds1/2, packages their results,
then runs TRT concat folds1/2 and packages those results. Any failed command
stops the sequence. Both arms use physical batch8, accumulation2, effective
batch16 on one GPU; the prepared dataset split stays seed42.

```bash
(
set -e
set -o pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate decoder-va
cd /workspace/decoder_VA_gaze_concat/va_model_code
mkdir -p ../results

SEED=42
for FUSION in none prefix-concat
do
  if [ "$FUSION" = "none" ]; then
    CONDITION="text_only"
  else
    CONDITION="gaze_TRT_concat"
  fi
  RUN_NAME="paper7_no_iemocap_qwen_full_${CONDITION}_heteroscedastic_lasttoken_overlapkeep_bf16_gc_b8_ga2_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

  python -u train_model.py qwen3.5-0.8b heteroscedastic \
    --data-dir data/paper7_seed42 \
    --finetuning-mode full \
    --precision bf16 \
    --gaze-fusion "$FUSION" \
    --gaze-features TRT \
    --gaze-redistribution none \
    --gaze-relation none \
    --readout last-token \
    --attn-implementation eager \
    --et-model-id skboy/et_prediction_2 \
    --et-cache-size 70000 \
    --no-iemocap \
    --training-text-overlap keep \
    --held-out-folds 1 2 \
    --max-length 200 \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --epochs 10 \
    --max-steps -1 \
    --learning-rate 6e-6 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --logging-steps 200 \
    --save-total-limit 1 \
    --group-by-length \
    --gradient-checkpointing \
    --seed "$SEED" \
    --run-name "$RUN_NAME" \
    2>&1 | tee "../results/${RUN_NAME}.log"

  python package_results.py --run-name "$RUN_NAME"
done
)
```

The shared TRT/cache flags do not load or use ET2 in the text-only arm.
Each successful packaging command prints its ZIP path beneath that run's
`/workspace/decoder_VA_gaze_concat/results/<RUN_NAME>/` directory.

## Local validation

Tests cover the NLL against an independent Gaussian density and gradient,
unchanged initial mean predictions/RNG, CPU BF16 autocast, variance positivity,
native hybrid-Qwen training with accumulation and gradient checkpointing,
checkpoint reload and Adam resume, output order, both training conditions,
legacy checkpoints, and tamper-resistant results-only packaging. A fixed-mean
synthetic probe checks recovery of two known residual variance scales; this is
a correctness test, not evidence of improvement on the research datasets.

```bash
python -m pytest -q va_model_code/tests
```

No full-data GPU training or new empirical accuracy/calibration claim is part
of this implementation handoff.

Verified locally on 2026-10-20: **1370 passed, 5 skipped** in the complete test
suite (the skipped device cases require unavailable CUDA/MPS). Both documented
training arms also passed a CPU/FP32 dry run using the real prepared folds:
26,898 training / 26,893 evaluation rows for fold1, and the reverse for fold2.
The documented shell blocks passed `bash -n`. These checks do not measure 5090
training performance or real-data accuracy.

# Fixed left/right redistribution experiments

Run both directions with frozen sigma values, keeping the previous sentence-only,
TRT-only, full Qwen, BF16, seed-42 training settings. The right condition uses
left/right 0.5/2.0; the left condition uses 2.0/0.5. Both run held-out folds 1 and 2
for 10 epochs. Sentence-only includes EmoTales sentences, Emobank and fb; it
excludes IEMOCAP.

`fixed-gaussian` now accepts unequal widths and keeps both log-sigma parameters
outside the optimizer. Do not pass `--redistribution-learning-rate`: it applies
only to learned `asym-gaussian`. The existing `min_sigma=1e-6` remains, so effective
widths are approximately 0.500001 and 2.000001. Diagnostics record zero sigma
updates; the rest of the model is trained normally.

## Server commands

Publish the updated repository code to GitHub main before running this block.
It reuses the existing `decoder-va` conda environment and `data/paper7_seed42`.
Paste the entire block into the server terminal. The right condition completes
both folds before the left condition starts. A failed command stops this block.

```bash
conda activate decoder-va &&
cd /workspace/decoder_VA_gaze_concat/va_model_code &&
(
set -e
set -o pipefail
git pull --ff-only origin main
mkdir -p ../results

RUN_NAME="paper7_sentence_only_qwen_full_gaze_TRT_fixed_right_l0p5_r2_bf16_gc_b16_seed42_$(date +%Y%m%d_%H%M%S)"

python train_model.py qwen3.5-0.8b mse \
  --data-dir data/paper7_seed42 \
  --finetuning-mode full \
  --precision bf16 \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
  --gaze-redistribution fixed-gaussian \
  --redistribution-sigma-left 0.5 \
  --redistribution-sigma-right 2.0 \
  --et-model-id skboy/et_prediction_2 \
  --et-cache-size 70000 \
  --sentence-only \
  --held-out-folds 1 2 \
  --max-length 200 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --epochs 10 \
  --max-steps -1 \
  --learning-rate 6e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --logging-steps 200 \
  --sigma-diagnostics-steps 50 \
  --sigma-diagnostics-batch-size 2 \
  --save-total-limit 1 \
  --group-by-length \
  --gradient-checkpointing \
  --seed 42 \
  --run-name "$RUN_NAME" \
  2>&1 | tee "../results/${RUN_NAME}.log"

RUN_NAME="paper7_sentence_only_qwen_full_gaze_TRT_fixed_left_l2_r0p5_bf16_gc_b16_seed42_$(date +%Y%m%d_%H%M%S)"

python train_model.py qwen3.5-0.8b mse \
  --data-dir data/paper7_seed42 \
  --finetuning-mode full \
  --precision bf16 \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
  --gaze-redistribution fixed-gaussian \
  --redistribution-sigma-left 2.0 \
  --redistribution-sigma-right 0.5 \
  --et-model-id skboy/et_prediction_2 \
  --et-cache-size 70000 \
  --sentence-only \
  --held-out-folds 1 2 \
  --max-length 200 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --epochs 10 \
  --max-steps -1 \
  --learning-rate 6e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --logging-steps 200 \
  --sigma-diagnostics-steps 50 \
  --sigma-diagnostics-batch-size 2 \
  --save-total-limit 1 \
  --group-by-length \
  --gradient-checkpointing \
  --seed 42 \
  --run-name "$RUN_NAME" \
  2>&1 | tee "../results/${RUN_NAME}.log"
)
```

Results and logs are in repository-root `results/`, in separately named
`fixed_right_l0p5_r2` and `fixed_left_l2_r0p5` runs. Each completed run prints
`Completed. OOF reports:` and writes root `oof_metrics.json`.


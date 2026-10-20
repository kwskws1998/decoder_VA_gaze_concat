# Readout-connected TRT rotation

## What is being compared

| Condition | Prediction token | TRT score correction |
|---|---|---|
| Matched Raw | Learned token after text | None |
| Readout rotation | Same learned token after text | Text pairs and readout-to-text pairs; readout TRT reference is zero |

Both arms retain the frozen ET2 raw TRT prefix. A single positive-TRT word can
now influence the final readout through the rotation. Missing gaze and true zero
TRT are not converted into artificial nonzero values. Earlier checkpoints keep
their original last-token architecture; start new run directories for this variant.

Use the same prepared `paper7_seed42` folds for both training seeds. The commands
keep every non-IEMOCAP row, including shared training/evaluation texts, and retain
the existing two-fold evaluation/checkpoint-selection protocol. They do not use
`--sentence-only`. Physical batch 8 and accumulation 2 give effective batch 16.

The alpha learning rate stays at the previous explicit `1e-3` so this first
matched comparison does not also introduce an unrecorded optimizer change. A
separate `1e-4` follow-up is possible; it is not a demonstrated improvement.
Best-checkpoint selection is retained. Neither successful CPU tests nor the
new structural path establish a real-data performance gain or a measured GPU ETA.

## Paired diagnostics for the already-trained rotation model

Run this on the server where `final_model/model.safetensors` still exists.
Results-only ZIPs cannot run inference. This checks the original completed
base-seed42 run; it does not retrain it or modify any weights.

```bash
(
set -e
set -o pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate decoder-va
cd /workspace/decoder_VA_gaze_concat/va_model_code

RUN_NAME="paper7_no_iemocap_qwen_full_gaze_TRT_rotation_compact_overlapkeep_5090_bf16_gc_b8_ga2_seed42_20261019_141907"

python -u evaluate_gaze_relation.py \
  --run-dir "../results/${RUN_NAME}" \
  --data-dir data/paper7_seed42 \
  --held-out-folds 1 2 \
  --precision checkpoint \
  --eval-batch-size 8 \
  --shuffle-seed 42 \
  --output-dir "../diagnostics/${RUN_NAME}_paired_$(date +%Y%m%d_%H%M%S)"
)
```

The evaluator validates both completed fold models, data hashes, source row
identities and labels. It checks that the learned arm reproduces stored
predictions within a recorded precision-dependent tolerance. It then publishes
`paired_predictions.tsv`, `paired_metrics.tsv`, and `diagnostics.json` in a new
directory; existing output directories are never overwritten. These files
contain row IDs and text hashes, not raw text.

For each arm it reports overall/fold/source and structural slices. A positive
`mse_delta_vs_learned` means that control has higher error than learned rotation.
The report records how many actual TRT values changed during shuffling; one
anchor cannot be shuffled. Phase counts and maxima are exact, and phase quantiles
are marked approximate if the bounded causal-pair reservoir is used. These are
unweighted angles, not observed attention probabilities or logit-correction ratios.

This comparison asks whether the already-trained model uses its rotation at
inference. It does not replace training the matched Raw arm below.

## New matched training, seeds 42 and 43

Publish the readout changes to the GitHub branch before using `git pull` on the
server. The block below stops before installing dependencies if that code is
missing. On the previously verified 5090/CUDA13 driver configuration it creates
the conda environment if absent, installs the pinned dependencies and FLA,
checks actual CUDA/FLA forward and backward execution, and prepares the data.
Existing matching downloads and prepared data are reused.

Retaining checkpoints, Adam state, and final models for eight fold trainings can
accumulate approximately 100 GB plus temporary checkpoints and caches. Check
available disk space; `--save-total-limit 1` is not a limit across all runs.

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
if not Path("va_model_code/decoder_va/readout.py").is_file():
    raise SystemExit("Readout changes are missing. Publish the updated code to GitHub, then pull again.")
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
  2>&1 | tee "results/readout_environment_setup_$(date +%Y%m%d_%H%M%S).log"

cd va_model_code
python -u prepare_english_data.py \
  --download-default \
  --paper-protocol \
  --seed 42 \
  --output-dir data/paper7_seed42
)
```

The following is one pasteable block. It runs Raw then rotation for seed42,
followed by Raw then rotation for seed43: four runs, two folds each, eight fold
trainings in total. Failure stops the sequence. Each successful run is packaged
without weights by the existing results validator.

```bash
(
set -e
set -o pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate decoder-va
cd /workspace/decoder_VA_gaze_concat/va_model_code
mkdir -p ../results

for SEED in 42 43
do
  for RELATION in none trt-readout-rotation
  do
    EXTRA_ARGS=()
    if [ "$RELATION" = "none" ]; then
      CONDITION="raw_readout"
    else
      CONDITION="rotation_readout_lr1e-3"
      EXTRA_ARGS=(--gaze-relation-learning-rate 1e-3)
    fi

    RUN_NAME="paper7_no_iemocap_qwen_full_gaze_TRT_${CONDITION}_overlapkeep_bf16_gc_b8_ga2_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

    python -u train_model.py qwen3.5-0.8b mse \
      --data-dir data/paper7_seed42 \
      --finetuning-mode full \
      --precision bf16 \
      --gaze-fusion prefix-concat \
      --gaze-features TRT \
      --gaze-redistribution none \
      --readout learned-token \
      --gaze-relation "$RELATION" \
      --gaze-relation-scale 1.0 \
      "${EXTRA_ARGS[@]}" \
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
done
)
```

For a one-arm setup smoke test, use the same flags with `--max-steps 2
--held-out-folds 1` and a fresh run name. Such a run is partial evidence and
must not be packaged or compared as a complete two-fold experiment.

## Interpretation and compatibility

- Compare the new learned-readout Raw and rotated arms directly. Historical
  last-token Raw runs do not isolate the rotation effect.
- Preserve all non-IEMOCAP data, source folds, overlap policy, precision, eager
  backend, batch/accumulation, seed, learning schedule, and checkpoint selection.
- A common additive TRT offset no longer cancels for readout-to-text pairs,
  because the reference remains zero. The reference is an explicit modeling
  choice, not evidence about human fixation duration.
- The last-layer path is now available; useful learned attention still depends
  on Q/K content and data. Nonzero alpha alone does not establish usefulness.
- Schema10 metadata distinguishes readout geometry. Schema8/9 checkpoints load
  with original semantics; legacy relative rotation cannot be switched to the
  learned-readout method by renaming a command or checkpoint.
- Existing schema9 runs can resume with the unchanged last-token architecture
  and matching data/optimizer contracts. The resumed fold records its source
  schema. Adding learned readout still requires a fresh run.

## Local verification

```bash
python -m pytest -q va_model_code/tests
```

The new tests include native hybrid Qwen singleton and punctuation cases,
alpha-zero equality with matched Raw, final-layer alpha/readout gradients,
padding and batch invariance, causal isolation, gradient checkpoint replay,
safe checkpoint roundtrips, legacy metadata, and result contract enforcement.
A native hybrid-Qwen Trainer regression also verifies actual readout and alpha
updates, separate optimizer groups, exact prediction equality after reload, and
restoration of Adam moments when resuming training.

Verified locally on 2026-10-20: **1329 passed, 5 skipped** across the full
`va_model_code/tests` suite. Both documented training arms also passed local
FP32/CPU dry runs on the actual `paper7_seed42` files: 26,898 training / 26,893
held-out rows for fold1, and the reverse for fold2. These dry runs validate
arguments and data, not 5090 execution. All shell blocks passed `bash -n`.
No new full-data GPU training or real-checkpoint paired evaluation was run.

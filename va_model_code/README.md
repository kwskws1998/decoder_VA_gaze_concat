# Active Qwen VA baseline

This directory contains only the original VA regression model and raw ET2 gaze
prefix concat. See the [root README](../README.md) for environment installation,
data download and a complete training command.

## Model path

```text
Text only: text -> Qwen -> last valid text token -> VA head
Raw TRT:   eye_start + projected TRT + eye_end + text
           -> Qwen -> last valid text token -> VA head
```

The frozen ET2 provider predicts gaze features from text. TRT is the default
channel; the original raw-feature selection API is preserved for existing
checkpoints. No Gaussian redistribution, added attention rotation, learned
prediction token or uncertainty head is executed. Qwen's native position encoding
and native attention remain part of the backbone.

The head is LayerNorm, dropout, Linear(hidden_size, 2), then hard sigmoid. Outputs
are normalized valence/arousal means. Original objectives: `mse`, `ccc`, `mse+ccc`.
Trainable weights and head computation remain FP32; `--precision bf16` or `fp16`
selects mixed-precision autocast. Full fine-tuning and LoRA remain available.

## Modules

| Module | Responsibility |
| --- | --- |
| `decoder_va/model.py` | Qwen backbone, raw gaze projection, VA head, strict save/load |
| `decoder_va/gaze.py`, `alignment.py` | Frozen ET2 inference and offset-based alignment v3 |
| `decoder_va/packing.py` | Compact gaze prefix, causal positions and text-token pooling |
| `decoder_va/losses.py`, `trainer.py` | Original VA objectives, ordinary optimizer, sequential evaluation |
| `decoder_va/baseline_contracts.py` | Disabled experiment metadata compatibility and rejection guards |
| `decoder_va/dataset.py`, `filters.py`, `preprocessing.py` | Data loading, folds, scope and overlap policy |
| `decoder_va/evaluation.py`, `external_benchmarks.py` | VA metrics, ordered reports and external benchmark protocols |
| `decoder_va/training_kernels.py` | Qwen kernel dispatch inspection |

`train_model.py`, `prepare_english_data.py`, `download_data.py`,
`filter_datasets.py`, `evaluate_external.py`, `package_results.py` and
`check_training_kernels.py` remain the active entry points.

## Data and results contracts

- Prepare paper-protocol seed-42 data once; data is not copied into the archive.
- `--no-iemocap` excludes IEMOCAP; `--sentence-only` retains EmoTales sentences,
  Emobank and Facebook. These remain separate explicit flags.
- Use `--training-text-overlap keep` for the user's original row-level protocol.
  `exclude` remains an explicit alternative. This refactor does not change either.
- `--held-out-folds 1 2` trains the two opposite-fold models. Best checkpoint
  selection remains minimum evaluation mean MSE.
- Evaluation is sequential even with length-grouped training, preserving row
  metadata correspondence.
- Every training run writes to `../results/<run-name>/`; `--output-dir` is not
  supported. Automatic run names include condition and timestamp.
- Packaging requires complete two-fold evidence and excludes weights/optimizer
  state. It verifies metrics, manifest consistency, safety, SHA-256 and ZIP CRC.

## Old checkpoints and experimental results

Baseline architecture schema remains 11: removing inactive branches does not
change the two-output state dict or numerical computation. Compatible baseline
schemas 8–10 remain loadable under their existing alignment/precision checks;
older schemas may be packageable but are not silently upgraded for inference.
Disabled `gaze_redistribution`, `gaze_relation`, `readout` and
`prediction_distribution` metadata are retained to identify old Raw runs exactly.
These fields are validation metadata, not active experimental modules.

Enabled experiment metadata, uncertainty prediction tables and experimental
weight keys are rejected. Use the isolated archived implementation instead:

```bash
python ../legacy_code/run.py train --help
python ../legacy_code/run.py package --run-name YOUR_EXPERIMENT_RUN
```

Experiment flags have moved, including `--gaze-redistribution`, all sigma options,
`--gaze-relation`, `--readout` and the `heteroscedastic` objective. A stale command
fails rather than silently changing the experiment. See
[archive module map and launcher](../legacy_code/README.md).

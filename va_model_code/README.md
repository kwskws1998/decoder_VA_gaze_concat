# Decoder-based valence/arousal prediction

This directory now trains a decoder-only Qwen model for two-dimensional
valence/arousal (VA) regression and can prepend predicted eye-tracking features
to the decoder input.

## Model decision

The default is
[`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base),
loaded through the text-only `Qwen3_5ForCausalLM` class. The official model card
lists a 0.8B language model, hidden size 1024, 24 decoder layers, and Apache-2.0
licensing. It also states, word for word:

> “The intended use cases are fine-tuning, in-context learning experiments,
> and other research or development purposes, not direct interaction.”

The same card states:

> “Global Linguistic Coverage: Expanded support to 201 languages and dialects”

This makes the Base checkpoint a better starting point for a regression
fine-tune than an instruction/chat checkpoint. Its sub-billion-parameter size
also makes both LoRA and carefully measured full fine-tuning plausible on a
24GB GPU; full fine-tuning still requires the smoke test below rather than a
memory guarantee.
The code pins the currently verified model revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

This selection is a capacity, language-coverage, task-formulation, and memory
decision. It is not evidence that Qwen already matches XLM-R on this VA dataset.
That claim can only be made after running both models on the same source bundle,
same folds, same exclusions, and same metrics.

The implementation:

- loads only the Qwen text backbone and discards the LM head and vision tower;
- exposes an explicit `--finetuning-mode {lora,full}` contract;
- defaults to rank-16 all-linear LoRA for backward-compatible, lower-memory
  runs;
- makes every Qwen text-backbone parameter trainable in `full` mode;
- trains the gaze projector, boundary embeddings, and VA head fully;
- keeps ET2 frozen in both fine-tuning modes;
- uses BF16 on supported NVIDIA GPUs;
- enables gradient checkpointing;
- defaults to batch size 4 with four-step gradient accumulation.

The learning-rate default is mode-aware: `1e-4` for LoRA and `6e-6` for full
fine-tuning. An explicit `--learning-rate` always overrides it. Run manifests
and saved-model manifests record the selected mode, and checkpoints from
different modes cannot be resumed or reloaded as one another.

The official
[`Qwen3.5` Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
explicitly supports `inputs_embeds`; this is the interface used to insert gaze
embeddings.

## Gaze concat

The ET model is
[`skboy/et_prediction_2`](https://huggingface.co/skboy/et_prediction_2).
Its model card describes a `roberta-base` regression model with a five-output
linear head. This implementation:

- pins ET revision `5785e77309d9fce8b88e908a9db100c1a0a63456`;
- downloads its tokenizer and `et_predictor2_seed123.safetensors`;
- reconstructs the declared architecture locally;
- uses `trust_remote_code=False` and never executes the repository's
  `model.py`;
- freezes the complete ET model;
- selects any non-empty subset of the five raw channels
  `(nFix, FFD, GPT, TRT, fixProp)`; the default remains TRT at index 3;
- aligns ET words monotonically to the first exact Qwen subword;
- caches detached results by model revision, selected channels, and the complete
  token sequence.

Each sample is packed before batch padding:

```text
[eye_start] [projected valid gaze vectors] [eye_end] [valid Qwen text] [right padding]
```

This preserves the prefix order in the official
[`gaze_reward` GazeConcat implementation](https://github.com/Telefonica-Scientific-Research/gaze_reward/blob/main/rlhf_rw/models/reward_model_general_sp.py#L154-L210):
eye-start boundary, projected gaze sequence, eye-end boundary, then text. The
Qwen adaptation uses trainable boundary parameters and compact selected gaze
vectors mapped to exact Qwen first-subword positions rather than adding tokenizer
vocabulary items or retaining every predictor position.

The causal decoder is pooled at each sample's last valid text token. That token
can attend to the complete gaze prefix and all preceding text, whereas the
prefix-side `eye_end` cannot causally attend to later text. Explicit per-sample
pooling indices avoid selecting physical right padding, and position IDs are
rebuilt after packing.

The regression head emits exactly:

```text
[valence, arousal]
```

Both outputs are constrained to `[0, 1]` with the original VA paper's hard
sigmoid output activation. Training requires an explicit choice
of `mse`, `ccc`, or the legacy-compatible 50:50 `mse+ccc`; no uncertainty or
log-variance head is used. The commands below use MSE as the simplest baseline.

## Gaze redistribution

`--gaze-redistribution` is an independent optional transformation, not a new
concat architecture. The default is `none`, which preserves the existing raw
gaze path and adds no learned parameters. `asym-gaussian` requires
`--gaze-fusion prefix-concat` and TRT among `--gaze-features`; invalid
combinations fail before model downloads.

The active implementation is `decoder_va/redistribution.py`. Its position in
the model is:

```text
frozen ET2 -> Qwen alignment + gaze mask -> raw-aligned-feature cache
          -> optional TRT redistribution -> unchanged gaze projector
          -> unchanged gaze-prefix packing -> Qwen -> VA head
```

The cache contains only raw ET2 predictions. Redistribution is outside ET2's
inference-only context and executes on every forward pass, so its two learned
log-width parameters receive VA-loss gradients in both full and LoRA modes when
a row has at least two valid gaze positions. A one-position row is an identity
mapping and contributes exactly zero gradient to both widths.
`--redistribution-sigma-left` and `--redistribution-sigma-right` default to
`1.0`; the positive width is `exp(log_sigma) + min_sigma`, with
`--redistribution-min-sigma 1e-6`. Kernel calculations use FP32 even during BF16
autocast. Only the selected TRT channel is replaced at valid gaze positions;
other selected channels are unchanged there. Masked rows are zeroed before
projection, including non-TRT channels, so invalid NaN/Inf padding cannot poison
projector gradients. No gaze-value clipping, renormalization of the input features,
or ET2 fine-tuning is introduced.

The two log-widths use their own AdamW parameter group. Its learning rate is
set by `--redistribution-learning-rate` (default `1e-3` when redistribution is
enabled), while its weight decay is fixed at zero. All other trainable
parameters retain `--learning-rate` and the normal Trainer decay/no-decay
policy. The common scheduler multiplies both learning rates by the same factor.
Regular Trainer logs include `redistribution_sigma_left`,
`redistribution_sigma_right`, and the scheduled
`redistribution_learning_rate`, so movement can be audited during training.

Mask handling is mandatory. The model passes the provider's explicit
`gaze_mask`, which marks valid mapped first subwords, rather than deriving a
mask from `TRT != 0` or using only the text padding mask. Both sources and
destinations are masked. Padding, special tokens, continuation subwords, and
unmapped positions neither contribute nor receive TRT. Valid zero-valued TRT
positions remain eligible destinations. Each valid source's Gaussian weights
sum to one across valid destinations, preserving the total valid signed TRT
up to numerical roundoff; an all-masked row returns zeros.

The recorded geometry is `aligned_qwen_tokens`: Gaussian distance uses the
original aligned Qwen token indices, retaining gaps at masked positions.
Redistribution happens before gaze-prefix compaction. It does not use compact
word ranks, character distances, or the ET2 tokenizer's positions. This is an
explicit adaptation to this model's alignment, not a claim of numerical
equivalence to redistribution in a different tokenizer.

Source audit (2026-09-04): the supplied
`34755_Leveraging_Psychophysica_Supplementary Material (1).zip` has SHA256
`6d8fef30b7f8a0ad9cd2f6d81220a7d5b16c73ea1f3523896762de2be5228e61`;
its `models/asym_gaussian_redistributor.py` member has SHA256
`284db3253abf8f9d399bd314c394eff67109c0ef0a72c721582321d99044b714`.
That inspected kernel already implements source/destination masking. The
`MyRewardBase.process_fixations` ET2 branch in `models/reward_model_base.py`
checks whether TRT is selected but does not check
`self.use_asym_gaussian_redistributor`, unlike its ET1 branch. Thus its ET2
path can apply redistribution even when the enable flag is false, a separate
integration issue from the kernel's masking support. This implementation
explicitly wires the requested option and the original sparse gaze mask into
the model; it does not assume that an unobserved historical run used the
inspected mask-aware code.

Schema-7 run and architecture manifests record the canonical
`gaze_redistribution` configuration; `model.safetensors` stores the two learned
width parameters alongside the existing model weights. Initialization values
in JSON are not the learned final widths. New default run names and archive
condition names distinguish raw gaze from redistribution. Legacy schema-5/6
models retain disabled redistribution; a schema-7 checkpoint must record the
setting explicitly. Resume requires the same configuration and cannot add or
remove redistribution. Enabled resume supports ordinary single-file Trainer
`model.safetensors` checkpoints, not adapter-only, pickle, sharded, or symlinked
weight layouts. Before replacing any fold manifest, it safely inspects the two
named Gaussian-width tensors and requires finite FP32 scalars, preventing
silent initialization when width state is absent or incompatible. External
evaluation reconstructs and freezes the saved
condition: there is no benchmark-time switch for retrofitting redistribution
onto a previously trained raw-TRT checkpoint.
Final-model reload uses the same finite-FP32-scalar validator before loading the
backbone; corrupted or silently cast Gaussian-width state is rejected.
Run manifests additionally record `redistribution_learning_rate` and the fixed
`redistribution_weight_decay=0.0`. Resuming an older redistribution checkpoint
that lacks this optimizer-group contract is rejected because its saved optimizer
has a different parameter-group layout.

Run a new full-fine-tuning BF16 condition from `va_model_code`, reusing the
existing paper-protocol data and all matched training settings:

```bash
SEED=43
RUN_NAME="paper7_sentence_only_no_iemocap_qwen_full_gaze_TRT_redistribution_asym-gaussian_bf16_gc_b16_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p ../results
set -o pipefail

python train_model.py qwen3.5-0.8b mse \
  --data-dir data/paper7_seed42 \
  --finetuning-mode full \
  --precision bf16 \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
  --gaze-redistribution asym-gaussian \
  --redistribution-sigma-left 1.0 \
  --redistribution-sigma-right 1.0 \
  --redistribution-min-sigma 1e-6 \
  --redistribution-learning-rate 1e-3 \
  --et-model-id skboy/et_prediction_2 \
  --et-cache-size 70000 \
  --sentence-only \
  --no-iemocap \
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
  --save-total-limit 1 \
  --group-by-length \
  --gradient-checkpointing \
  --seed "$SEED" \
  --run-name "$RUN_NAME" \
  2>&1 | tee "../results/${RUN_NAME}.log"
```

For the matched raw-TRT control, use `--gaze-redistribution none`, omit all four
redistribution-only options (the three sigma options plus
`--redistribution-learning-rate`), and remove `redistribution_asym-gaussian`
from its new run name.
Do not regenerate `data/paper7_seed42` when changing only the training seed.

FP32 uses the same redistribution interface: change `--precision bf16` to
`--precision fp32` and the run-name precision tag. If memory requires a smaller
physical batch, `--train-batch-size 8 --eval-batch-size 8
--gradient-accumulation-steps 2` retains effective batch size 16; record this
change and match it in the corresponding control. The new redistribution
condition has not been trained or GPU-memory-profiled here. Two additional
parameters do not imply zero activation-memory cost: the kernel uses
quadratic token-pair weights. Start with the smoke test below and inspect
`heldout_fold*/gpu_memory.json`; fitting a 24 GB device is not guaranteed.

## Environment

Qwen3.5 requires a newer Transformers build than the original code:

```bash
# Run from the repository root, before entering va_model_code.
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
cd va_model_code
```

The current local CPU environment can run the unit tests but has an older
Transformers version. Full Qwen training is intended for the 24GB NVIDIA
machine.

## Data download and preprocessing

The authorized seven-source English bundle is configured as:

```text
Google Drive file ID: 1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj
SHA256: 5db750ededfd9717dcca465b34fd7e6c348e50e563ad2c0814c458b04441e81d
```

It has already been downloaded and validated at
`data/external/english_va_bundle.zip`.

To reproduce the download:

```bash
python download_data.py
```

To download and preprocess in one command:

```bash
python prepare_english_data.py --download-default
```

The command above preserves the repository's legacy preprocessing behavior.
For the original VA paper's source-wise two-fold protocol, build a separate
dataset directory:

```bash
python prepare_english_data.py \
  --download-default \
  --paper-protocol \
  --seed 42 \
  --output-dir data/paper7_seed42
```

Paper protocol mode retains every row with finite VA labels, preserves the
source text, normalizes against the original source scale, independently
shuffles and halves each source, and then combines the corresponding halves.
The split is generated once and reused by every model condition.

Verified paper-protocol output for the bundled seven sources is 63,823 rows:
31,909 in fold 1 and 31,914 in fold 2. Verify the intended directory before
training:

```bash
python train_model.py --list-datasets \
  --data-dir data/paper7_seed42
```

This reproduces the paper's preprocessing and two-fold procedure on the
available seven-source English bundle; it is not the paper's unavailable
34-source multilingual training artifact. The
[authors' repository](https://github.com/gmendes9/multilingual_va_prediction#dataset)
explains that the original combined dataset cannot be publicly provided. There
is also a small source-version mismatch even within the English subset: Table 1
lists IEMOCAP at 10,039 and Facebook Posts at 2,894, whereas this validated
bundle contains 10,032 and 2,895 respectively. Therefore, use 63,823 as this
experiment's expected sample count and compare baseline against gaze on exactly
the same generated folds; do not compare the absolute score directly against
the paper's table.

The `data` directory is intentionally excluded from Git, so this command must
be run once after cloning the repository onto a new training machine.

To preprocess the existing validated ZIP:

```bash
python prepare_english_data.py \
  --archive data/external/english_va_bundle.zip \
  --sha256 5db750ededfd9717dcca465b34fd7e6c348e50e563ad2c0814c458b04441e81d \
  --output-dir data
```

The preprocessing reproduces the legacy semantics used by this code:

1. sort source TSVs;
2. clean whitespace while retaining blank text;
3. drop rows with missing/non-finite VA;
4. deduplicate text within each source;
5. leave a source unchanged when both dimensions are already in `[0, 1]`;
6. otherwise apply observed per-source min-max normalization;
7. concatenate sources;
8. shuffle once with seed 42;
9. add a stable global row index;
10. split into two contiguous row halves.

Verified output:

| Dataset | Rows |
|---|---:|
| EmoTales sentences | 1,369 |
| Emobank | 9,906 |
| GlasgowNorms | 5,553 |
| IEMOCAP sentences | 8,013 |
| fb | 2,887 |
| nrc-vad | 19,971 |
| word ratings ENG | 13,915 |
| Total | 61,614 |

The generated files are:

```text
data/full_dataset_fold1.csv
data/full_dataset_fold2.csv
data/full_dataset_english_all.csv
data/english_dataset_manifest.json
```

Despite the `.csv` suffix, fold files are tab-separated to preserve the
original loader/evaluation convention.

Important scope limitation: the downloaded artifact contains seven English
sources. The legacy README describes a separate 34-source multilingual dataset
that was not publicly bundled. Results from that 34-source experiment are not a
valid direct baseline for this seven-source build.

## Dataset exclusions

Exclusions are applied in memory to both training and held-out evaluation. Fold
files are never rewritten.

List available dataset names:

```bash
python train_model.py --list-datasets
```

Preview IEMOCAP removal:

```bash
python filter_datasets.py --data-dir data --no-iemocap
```

Both requested spellings work:

```bash
python train_model.py --no-iemocap
python train_model.py --no-ieomcap
```

Any dataset can be excluded with repeated or comma-separated values:

```bash
python train_model.py \
  --exclude-dataset Emobank \
  --exclude-dataset fb

python train_model.py \
  --exclude-dataset Emobank,fb
```

Patterns are resolved against actual `dataset_of_origin` values. An unmatched
pattern fails with the available names instead of silently doing nothing.

For the exact non-IEMOCAP sentence-only experiment, use `--sentence-only`. It
keeps the provenance allowlist `EmoTales sentences`, `Emobank`, and `fb` in both
training and held-out folds; it does not infer scope from token count. The
paper-protocol folds contain 7,175 and 7,177 retained rows respectively (14,352
OOF rows). All three sources must exist in each fold, and separately excluding
one of them is rejected. `--no-iemocap` may also be supplied to make the intended
condition explicit.

```bash
python train_model.py --dry-run \
  --data-dir data/paper7_seed42 \
  --sentence-only \
  --no-iemocap \
  --finetuning-mode full \
  --precision bf16 \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
  --gaze-redistribution asym-gaussian \
  --redistribution-learning-rate 1e-3
```

The matched raw-TRT control must also use `--sentence-only`; comparing a
sentence-only redistribution model against an older seven-source raw-TRT model
does not isolate redistribution.

Verified no-IEMOCAP counts:

```text
fold 1: 30,807 -> 26,885
fold 2: 30,807 -> 26,716
total: 53,601
```

## Training and evaluation

Default Qwen + TRT prefix concat with LoRA:

```bash
python train_model.py qwen3.5-0.8b mse \
  --finetuning-mode lora
```

The same model with full Qwen text-backbone fine-tuning:

```bash
python train_model.py qwen3.5-0.8b mse \
  --finetuning-mode full
```

Choose one raw ET2 feature:

```bash
python train_model.py qwen3.5-0.8b mse \
  --finetuning-mode lora \
  --gaze-features FFD
```

Choose several features. They are canonicalized to the published ET2 order:

```bash
python train_model.py qwen3.5-0.8b mse \
  --finetuning-mode lora \
  --gaze-features nFix FFD GPT TRT fixProp
```

Every run manifest records the canonical feature names, zero-based indices, and
five-bit mask in `(nFix, FFD, GPT, TRT, fixProp)` order. For example,
`--gaze-features TRT nFix` is stored as names `[nFix, TRT]`, indices `[0, 3]`,
and mask `[1, 0, 0, 1, 0]`.

No-IEMOCAP run:

```bash
python train_model.py qwen3.5-0.8b mse \
  --finetuning-mode lora \
  --no-iemocap
```

Text-only Qwen ablation:

```bash
python train_model.py qwen3.5-0.8b mse \
  --finetuning-mode lora \
  --gaze-fusion none
```

Explicit precision is selected with `--precision {auto,bf16,fp16,fp32}`. The
default `auto` uses BF16 on a BF16-capable CUDA GPU. An FP32 gaze run must pass
`--precision fp32`; this loads Qwen, the trainable gaze projector, and the VA
head in FP32, disables Trainer BF16/FP16 autocast, and explicitly disables TF32.
ET2 inference and its raw features are already FP32.

Use a smaller physical batch for the higher-memory FP32 condition while keeping
effective batch size 16:

```bash
python train_model.py qwen3.5-0.8b mse \
  --data-dir data \
  --finetuning-mode lora \
  --precision fp32 \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
  --no-iemocap \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --run-name english7_no_iemocap_qwen_lora_gaze_TRT_fp32_seed42
```

Dataset and Trainer-API validation without tokenizer or model downloads:

```bash
python train_model.py --dry-run --no-iemocap
```

### Paper-protocol splits, no-IEMOCAP single-seed Qwen A/B

Generate `data/paper7_seed42` first with the paper-protocol command above. Then
run the text-only baseline. `--no-iemocap` is an explicit experiment exclusion,
not part of the quoted paper protocol:

Relevant word-for-word excerpts from the paper are:

> “randomly split in half”

> “hard sigmoid activation function”

> “batch size was fixed at 16”

> “models were trained during 10 epochs”

Source: [Mendes and Martins (2023), Section 5](https://arxiv.org/pdf/2302.14021#page=7).

```bash
python train_model.py qwen3.5-0.8b mse \
  --data-dir data/paper7_seed42 \
  --finetuning-mode full \
  --gaze-fusion none \
  --no-iemocap \
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
  --save-total-limit 1 \
  --group-by-length \
  --seed 42 \
  --run-name paper7_no_iemocap_qwen_full_baseline_seed42
```

Run the matching gaze condition, choosing the desired raw feature subset:

```bash
python train_model.py qwen3.5-0.8b mse \
  --data-dir data/paper7_seed42 \
  --finetuning-mode full \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
  --no-iemocap \
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
  --save-total-limit 1 \
  --group-by-length \
  --seed 42 \
  --run-name paper7_no_iemocap_qwen_full_gaze_TRT_seed42
```

These two commands form the primary full-fine-tuning A/B. To run the matching
LoRA A/B, use `--finetuning-mode lora --learning-rate 1e-4` in both commands
and use new run names. Never compare a full baseline against a LoRA
gaze condition as the gaze ablation.

The top-level seed is 42. Internally, held-out folds 1 and 2 use fold seeds 42
and 43 respectively. The same fold seed is reused across baseline and gaze so
their shared Qwen and regression-head initialization is paired within the same
fine-tuning mode. This is an ablation-control choice, not a requirement stated
by the original paper.

Compare the two root `oof_metrics.json` files, not the arithmetic mean of fold
metrics. With this exact no-IEMOCAP bundle and split, both must report
`n_examples == 53791`. The paper-facing metrics are
Pearson, RMSE, and MAE for valence and arousal. The two
`oof_predictions.tsv` files must have identical `index`, `held_out_fold`,
`dataset_of_origin`, `valence`, and `arousal` columns; only predictions should
differ. One top-level seed supports a descriptive A/B result, not a
seed-variance or significance claim.

The evaluation protocol remains fixed two-fold out-of-fold:

- train fold 2, predict held-out fold 1;
- load a fresh model, train fold 1, predict held-out fold 2;
- combine predictions once into the OOF report.

Run one held-out fold for a sentence-only redistribution smoke test:

```bash
python train_model.py qwen3.5-0.8b mse \
  --data-dir data/paper7_seed42 \
  --finetuning-mode full \
  --precision bf16 \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
  --gaze-redistribution asym-gaussian \
  --redistribution-learning-rate 1e-3 \
  --sentence-only \
  --no-iemocap \
  --held-out-folds 1 \
  --max-length 200 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --epochs 1 \
  --max-steps 3 \
  --learning-rate 6e-6 \
  --save-total-limit 1 \
  --gradient-checkpointing \
  --seed 42 \
  --run-name smoke_sentence_only_no_iemocap_qwen_full_gaze_TRT_redistribution_bf16_seed42
```

Three optimizer steps ensure that AdamW state exists while a later
forward/backward pass is measured. Inspect
`../results/<run-name>/heldout_fold1/gpu_memory.json`; it records peak allocated
and reserved CUDA memory. If batch 16 runs out of memory or peak reserved memory
leaves too little headroom, use `--train-batch-size 8` and
`--gradient-accumulation-steps 2` in both baseline and gaze runs. This preserves
effective batch size 16 for the MSE experiment.

Each run writes:

```text
../results/<run-name>/
  heldout_fold1/
    checkpoints/
    final_model/
    gpu_memory.json
    metrics.json
    predictions.tsv
    run_manifest.json
  heldout_fold2/
    ...
  oof_predictions.tsv
  oof_metrics.json
  metrics_by_dataset.tsv
  training_parameters.json
```

The run root is always anchored to the repository-root `results/` directory.
`--run-name` accepts exactly one directory name; paths and traversal components
are rejected. The old CWD-relative `Preds/` output contract and arbitrary
training `--output-dir` paths are intentionally unsupported. If a user-supplied
name contains `baseline` or `gaze`, the name is checked against the actual
fusion setting before model download.

`final_model/` contains a complete safe state dict, the locally saved tokenizer,
and a versioned architecture manifest with the fine-tuning mode, exact
decoder/ET revisions, conditional LoRA settings, gaze projector dimensions,
and output contract. Full Trainer checkpoint directories include AdamW state
and are substantially larger than LoRA checkpoints; retain
`--save-total-limit 1`. Reload strictly with:

```python
import torch

from decoder_va import load_saved_decoder_va_model

model, tokenizer = load_saved_decoder_va_model(
    "../results/<run-name>/heldout_fold1/final_model",
    dtype=torch.bfloat16,
)
model.to("cuda")
```

Create a small review archive after both held-out folds finish:

```bash
python package_results.py \
  --run-name paper7_no_iemocap_qwen_full_baseline_seed42
```

The ZIP is written inside that same run directory. Its filename is derived from
the recorded model, fine-tuning mode, gaze condition, feature selection, and
seed. The packager requires the complete two-fold OOF result, validates all
fold/OOF files, writes SHA-256 hashes, tests the completed ZIP, and excludes
model weights and optimizer states.

The reload path executes no repository-supplied Python. It reconstructs the
recorded raw-Qwen or Qwen/LoRA architecture, validates
`decoder_va_architecture.json`, and loads `model.safetensors` with strict key
checking. The pinned Qwen checkpoint must be available locally or from Hugging
Face during reconstruction; ET2 weights remain external and are fetched lazily
only when gaze inference starts.

The selectable-mode, selectable-feature, hard-sigmoid two-output head uses
architecture manifest schema version 6. Version 5 is accepted only as the
legacy LoRA-only form and is narrowly migrated to `finetuning_mode=lora`;
versions 4 and earlier remain incompatible. `--resume-from-checkpoint` requires
a checkpoint under the selected held-out fold and a matching run manifest.
Version-5 run manifests without a mode can resume only as LoRA. A LoRA
checkpoint can never resume as full fine-tuning, or vice versa.

Legacy metric names and semantics are retained:

- `mse_valence`, `mae_valence`, `pearson_corr_valence`;
- `mse_arousal`, `mae_arousal`, `pearson_corr_arousal`.

CCC and mean metrics are also reported. Per-dataset reporting is generated only
for datasets still present after filtering, so exclusions cannot trigger
hard-coded source lookup failures.

## Frozen external generalization benchmarks

`evaluate_external.py` evaluates an already completed two-fold run on external
corpora. It has no Trainer, optimizer, scheduler, backward pass, checkpoint
selection, calibration, or fine-tuning path. OMG-Emotion, MSP-Podcast, IDEST
English, and SemEval-2026 Task 2 Subtask 1 rows are used only after both saved
models and their inference contract have already been fixed.

The default, strongest input contract is the original server run directory,
not a results-only review ZIP. The ZIP deliberately omits `model.safetensors`;
external inference always requires both of these complete directories:

```text
<run-dir>/heldout_fold1/final_model/
<run-dir>/heldout_fold2/final_model/
```

### Hugging Face model upload contract

A raw-text-free Hugging Face bundle can omit every internal prediction table
that contains fine-tuning text. Its minimum supported layout is:

```text
<bundle>/
  training_parameters.json
  heldout_fold1/
    run_manifest.json
    final_model/
      model.safetensors
      decoder_va_architecture.json
      tokenizer_config.json
      tokenizer payload and companion files saved by Trainer
  heldout_fold2/
    run_manifest.json
    final_model/
      model.safetensors
      decoder_va_architecture.json
      tokenizer_config.json
      tokenizer payload and companion files saved by Trainer
```

Copy each `final_model/` directory as a whole. Do not select tokenizer files by
guessing: `tokenizer_config.json` and at least one real tokenizer payload such
as `tokenizer.json`, `tokenizer.model`, or `vocab.json` are required, and the
two fold inventories must match byte for byte. Do not mix folds from different
seeds, precisions, gaze conditions, or runs.

The raw-text-free bundle deliberately excludes these raw-text or otherwise
unnecessary artifacts:

```text
oof_predictions.tsv
heldout_fold1/predictions.tsv
heldout_fold2/predictions.tsv
full_dataset_fold1.csv
full_dataset_fold2.csv
checkpoints/
optimizer and scheduler state
training_args.bin
gpu_memory.json
training logs
```

The three prediction TSVs contain original fine-tuning text and gold labels.
Do not place them in a public model repository unless every source license has
been reviewed for redistribution. A private full-run backup may retain them;
the default preflight then also requires `oof_metrics.json`,
`metrics_by_dataset.tsv`, and both fold `metrics.json` files.

A raw-text-free bundle is evaluated explicitly with `--no-preflight-check`. Keep
the original fold CSVs in an authorized local directory so the independent
hash and exact-text-overlap audit can still run:

```bash
python evaluate_external.py omg-emotion \
  --run-dir /models/<raw-text-free-hf-bundle> \
  --training-data-dir /secure/paper7_seed42 \
  --raw-dir data/external_benchmarks/omg-emotion \
  --no-preflight-check \
  --device cuda \
  --output-dir ../results/external/<run-name>/omg-emotion/test
```

If the fold CSVs cannot be retained, add `--no-require-overlap-audit`; that is
a weaker portability run and the result manifest records both missing gates.
The retained JSON manifests can still contain local paths and machine metadata;
review those fields before a public release. The evaluator rejects symlinked
model artifacts, so materialize a Hugging Face
download as ordinary files before passing it as `--run-dir`. The local bundle
directory may be renamed; its recorded `run_name` must still agree internally
with `effective_output_dir`. Verify materialization with:

```bash
find /models/<raw-text-free-hf-bundle> -type l
```

The command must print nothing. A user checkpoint repository does not need to
duplicate the pinned Qwen base or ET2 artifacts. Reconstruction still fetches
`Qwen/Qwen3.5-0.8B-Base` at the saved revision, and gaze runs additionally
fetch `skboy/et_prediction_2` at the saved revision. An offline machine must
have those exact snapshots in its local Hugging Face cache.

This custom model is not a standalone `AutoModel.from_pretrained()` package.
Keep this repository at a committed/tagged revision, record that revision in
the Hugging Face model card, and install the matching `requirements.txt` before
evaluation.

The first model was trained on fold 2 and the second on fold 1. The evaluator
reports both members and a prespecified, unweighted arithmetic mean of their
`[valence, arousal]` predictions. It never selects a better member or learns
ensemble weights from external labels. Each `final_model` is the artifact saved
by the original completed training run; the external evaluator neither
re-selects it nor uses an external benchmark label to alter it.

For a saved gaze-fusion run, “gaze” in this external evaluation means frozen
ET2 pseudo-TRT predicted solely from each benchmark transcript. It is not eye
tracking measured from any external-benchmark participant, and no external gold
label is supplied to ET2.

The evaluator also reconstructs the original fine-tuning folds, verifies their
recorded SHA-256 hashes, and reports exact normalized-text overlap. The official
full test score remains the primary benchmark result; the fine-tuning-novel-text
subset is a separately labeled contamination diagnostic. Base-model pretraining
contamination cannot be established from this repository.

### Current seed-43 BF16 checkpoint paths on the Vast.ai server

The following commands use the two raw-text-free Hugging Face runs already
materialized on the benchmark server. Set the paths once from the repository's
`va_model_code` directory:

```bash
cd /workspace/decoder_VA_gaze_concat/va_model_code

BASE_RUN=/workspace/models/seed43_bf16/runs/paper7_no_iemocap_qwen_full_baseline_bf16_gc_b16_seed43_20260829_195414
TRT_RUN=/workspace/models/seed43_bf16/runs/paper7_no_iemocap_qwen_full_gaze_TRT_bf16_gc_b16_seed43_20260830_104257
TRAIN_DATA=/workspace/decoder_VA_gaze_concat/va_model_code/data/paper7_seed42
IDEST_RAW=/workspace/data/external_benchmarks/idest-english
SEMEVAL_RAW=/workspace/data/external_benchmarks/semeval-2026-task2-subtask1

mkdir -p /workspace/results/external
```

These bundles intentionally omit the internal prediction tables, so the
commands below use `--no-preflight-check`. They retain
`--training-data-dir` and the default overlap audit. Do not add
`--no-require-overlap-audit` unless the original fold files truly are
unavailable, because doing so weakens the contamination audit.

### IDEST English: all 250 stories

The IDEST paper states, word for word:

> “We introduce a database (IDEST) of 250 short stories rated for valence,
> arousal, and comprehensibility in two languages.”

Source: [International Database of Emotional Short Texts](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274480).

The evaluator downloads only the public `IDEST_Database.csv` from OSF file
`xh4kv` in project `9tga3`. The fixed file SHA-256 is
`91e3e05a9495833cafe6fcb59a445b3f248c7c2a3881549e9d8741a7f94bc0b7`.
`--download` is on-demand and idempotent: an existing file is reused only after
its hash matches. All 250 nonblank source rows are evaluated in source-file
order; IDEST has no official held-out split, so this is an all-items frozen
zero-shot test rather than a leaderboard split.

Only `text_english` enters the model. The published English mean valence and
arousal ratings stay on the fixed 1–9 native scale and are mapped into the
model scale without dataset statistics:

```text
model target = (IDEST English mean rating - 1) / 8
native-scale prediction = 1 + 8 * model prediction
```

No IDEST example is trained on, and IDEST labels are not used for calibration,
checkpoint selection, member weighting, or model input. The primary report is
global native-scale VA regression over all 250 stories; IDEST does not define
an official leaderboard score for this protocol.

Validate data, hashes, checkpoints, and fine-tuning-text overlap for each run
without loading either model:

```bash
python evaluate_external.py idest-english \
  --run-dir "$BASE_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$IDEST_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --output-dir /workspace/results/external/seed43_bf16_baseline/idest-english-all-250 \
  --dry-run
```

```bash
python evaluate_external.py idest-english \
  --run-dir "$TRT_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$IDEST_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --output-dir /workspace/results/external/seed43_bf16_trt/idest-english-all-250 \
  --dry-run
```

Run the two real frozen GPU evaluations:

```bash
python evaluate_external.py idest-english \
  --run-dir "$BASE_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$IDEST_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --output-dir /workspace/results/external/seed43_bf16_baseline/idest-english-all-250
```

```bash
python evaluate_external.py idest-english \
  --run-dir "$TRT_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$IDEST_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --output-dir /workspace/results/external/seed43_bf16_trt/idest-english-all-250
```

IDEST stories are substantially longer than the utterances in the original
benchmarks. On a real run, the evaluator tokenizes every story once without
truncation, records its token count, and reports how many exceed the checkpoint's
saved `max_length` (currently 200). Inference still uses that immutable saved
limit. The audit is label-free and is not performed by `--dry-run`, because a
dry run deliberately does not load the saved tokenizer.

If the primary 200-token run shows substantial truncation, run a separately
labeled post-hoc sensitivity analysis at 512 tokens. This changes inference
preprocessing relative to training, so the evaluator marks
`primary_benchmark_result=false` in both `metrics.json` and
`external_evaluation_manifest.json`. It does not change model weights, labels,
member selection, or ensemble weights. The 512-token ceiling matches the frozen
ET2 input limit and keeps the baseline/TRT comparison symmetric.

```bash
LENGTH_STAMP="$(date +%Y%m%d_%H%M%S)"
LENGTH_ROOT="/workspace/results/external/seed43_bf16_idest_maxlen512_${LENGTH_STAMP}"
mkdir -p "$LENGTH_ROOT"

python evaluate_external.py idest-english \
  --run-dir "$BASE_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$IDEST_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --batch-size 8 \
  --max-length 512 \
  --output-dir "$LENGTH_ROOT/baseline"
```

```bash
python evaluate_external.py idest-english \
  --run-dir "$TRT_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$IDEST_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --batch-size 8 \
  --max-length 512 \
  --output-dir "$LENGTH_ROOT/trt"
```

Use a fresh `LENGTH_ROOT` for every attempt because completed and partial
output directories are never overwritten. If 512-token TRT inference exceeds
available memory, rerun both conditions with `--batch-size 4`; batch size does
not change predictions under evaluation mode except for possible negligible
floating-point kernel differences.

### SemEval-2026 Task 2 Subtask 1: official test only

The SemEval paper states, word for word:

> “Systems are ranked using the average of rcomposite across V & A scores for
> Subtask 1”

Source: [SemEval-2026 Task 2 paper](https://aclanthology.org/2026.semeval-1.451/).

The evaluator pins the official repository revision
`50abd23fb884d3dd693c2df479124bcf6c153086` and downloads exactly these two
released files:

```text
datasets/TEST_RELEASE_5JAN2026/test_subtask1.csv
SHA256 61500316be2d5fcd88979e7f12885e4a42d3b9e71e1e2feb15deeda6134ff5fd

datasets/TEST_LABELS_RELEASE_23FEB2026/test_labels_subtask1.csv
SHA256 9d4734b93112c9db07144f404e013f55abb3f847c96b3cfd2550d8340cf5be3c
```

The released files contain 1,737 official test rows from 91 users. The loader
joins one-to-one on `(user_id, text_id)`, preserves official input order, and
rejects missing, duplicate, or metadata-inconsistent rows. The pinned released
CSV—not an observed min/max—fixes native valence to `[-2,2]` and arousal to
`[0,2]`:

```text
model valence target = (native valence + 2) / 4
model arousal target = native arousal / 2
native valence prediction = 4 * model valence - 2
native arousal prediction = 2 * model arousal
```

No SemEval train or development item is loaded, and no target example is used
for training, calibration, checkpoint selection, or ensemble weighting. Gold
test labels are joined only for schema/range validation and post-prediction
metrics. The primary score is the official user-aware Subtask 1
`r_composite`, reported for valence and arousal and averaged across V/A;
seen/unseen-user and essay/feeling-word results are diagnostics only.

Validate both runs without model loading:

```bash
python evaluate_external.py semeval-2026-task2-subtask1 \
  --run-dir "$BASE_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$SEMEVAL_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --output-dir /workspace/results/external/seed43_bf16_baseline/semeval-2026-task2-subtask1-test \
  --dry-run
```

```bash
python evaluate_external.py semeval-2026-task2-subtask1 \
  --run-dir "$TRT_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$SEMEVAL_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --output-dir /workspace/results/external/seed43_bf16_trt/semeval-2026-task2-subtask1-test \
  --dry-run
```

Run the two real frozen GPU evaluations:

```bash
python evaluate_external.py semeval-2026-task2-subtask1 \
  --run-dir "$BASE_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$SEMEVAL_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --output-dir /workspace/results/external/seed43_bf16_baseline/semeval-2026-task2-subtask1-test
```

```bash
python evaluate_external.py semeval-2026-task2-subtask1 \
  --run-dir "$TRT_RUN" \
  --training-data-dir "$TRAIN_DATA" \
  --raw-dir "$SEMEVAL_RAW" \
  --download \
  --no-preflight-check \
  --precision checkpoint \
  --device cuda \
  --output-dir /workspace/results/external/seed43_bf16_trt/semeval-2026-task2-subtask1-test
```

The present model receives one current row's `text` only. It does not receive
`user_id`, timestamps, prior user texts, collection phase, seen-user status, or
essay/feeling-word flags; those fields are retained only for alignment and
post-prediction reporting. This is therefore a strict current-text-only
zero-shot test, not a reproduction of history-aware SemEval systems. A real run
performs the same label-free, checkpoint-`max_length` truncation audit described
for IDEST and records it in `external_evaluation_manifest.json`.

### OMG-Emotion test

The implementation pins the official repository commit
`5931b237e92d68d04931bb932854fac6d9cd6a41` and exact hashes for
`omg_TestTranscripts.tsv` and `omg_TestVideos_WithLabels.csv`. Despite its
suffix, the transcript file is parsed as comma-separated CSV. Gold rows are
left-joined one-to-one on `(video, utterance)` in official label order. All
2,229 gold rows are retained, including 90 blank transcripts; the seven
transcript-only rows are recorded but cannot be scored.

The dataset paper states, word for word:

> “The intervals, [0,1] for arousal and [-1,1] for valence”

It also states:

> “each annotation is based on multimodal information”

Source: [The OMG-Emotion Behavior Dataset](https://www2.informatik.uni-hamburg.de/wtm/publications/2018/BCLSSW18/Barros_OMG.pdf).

The current model emits both dimensions on `[0,1]`. The only permitted mapping
is therefore fixed before evaluation:

```text
OMG valence prediction = 2 * model valence - 1
OMG arousal prediction = model arousal
```

No observed OMG minimum, maximum, mean, validation score, or label distribution
is used. Each input remains one current utterance transcript; previous OMG
utterances are not added as context.

First validate everything without loading either model:

```bash
python evaluate_external.py omg-emotion \
  --run-dir ../results/<completed-run-name> \
  --training-data-dir data/paper7_seed42 \
  --raw-dir data/external_benchmarks/omg-emotion \
  --download \
  --dry-run
```

Then run frozen inference with the checkpoint-recorded precision and batch
size:

```bash
python evaluate_external.py omg-emotion \
  --run-dir ../results/<completed-run-name> \
  --training-data-dir data/paper7_seed42 \
  --raw-dir data/external_benchmarks/omg-emotion \
  --precision checkpoint \
  --device cuda
```

Before the external pass, each reloaded model must reproduce 32 persisted
internal held-out predictions within a dtype-aware tolerance. BF16/FP16 uses
the same CUDA autocast family as Trainer evaluation; FP32 disables autocast and
TF32. Failure aborts before external scores are written. The external evaluator
accepts only `--precision checkpoint`; a post-hoc precision sensitivity run
cannot occupy or be mistaken for the canonical external-benchmark result.

The primary OMG metric is the official-script-compatible global utterance CCC
on the native scale. A per-video macro CCC is also written explicitly as a
secondary, non-official diagnostic.

### MSP-Podcast 2.0 Test1 and Test2

MSP-Podcast is not downloaded by this repository. Obtain version 2.0 through
the institutional Academic License procedure on the
[official corpus page](https://www.lab-msp.com/MSP/MSP-Podcast.html), then pass
the authorized local files directly. No MSP item data should be committed or
redistributed.

The corpus paper states:

> “We use a Likert scale from 1 to 7”

Source: [The MSP-Podcast Corpus](https://www.lab-msp.com/MSP/publications/Busso_2025.pdf).

The fixed prediction mapping is:

```text
MSP valence prediction = 1 + 6 * model valence
MSP arousal prediction = 1 + 6 * model arousal
```

The loader fixes the official label fields `FileName`, `Split_Set`, `EmoAct`,
and `EmoVal`; they cannot be changed by the canonical CLI. It accepts an
authorized transcript CSV/TSV table, a directory of per-utterance TXT files, or
the local transcript ZIP. Missing IDs, duplicate normalized IDs, blank matched
transcripts, values outside `[1,7]`, and incomplete version-2.0 split counts
fail without dropping rows. Only transcript ID/text columns can be overridden
because transcript packaging can vary across licensed releases; an explicit
override must exist and every resolved field is recorded.

The corpus paper states, word for word:

> “The test 2 set was collected without the retrieval-based protocol presented in Section III-C.”

Source: [The MSP-Podcast Corpus](https://www.lab-msp.com/MSP/publications/Busso_2025.pdf).

Therefore, run Test2 as the primary retrieval-bias-reduced cross-corpus
evaluation:

```bash
python evaluate_external.py msp-podcast \
  --run-dir ../results/<completed-run-name> \
  --training-data-dir data/paper7_seed42 \
  --split test2 \
  --labels-file /secure/MSP-Podcast-2.0/labels_consensus.csv \
  --transcripts /secure/MSP-Podcast-2.0/Transcripts.zip \
  --precision checkpoint \
  --device cuda
```

Run Test1 separately as the secondary external evaluation:

```bash
python evaluate_external.py msp-podcast \
  --run-dir ../results/<completed-run-name> \
  --training-data-dir data/paper7_seed42 \
  --split test1 \
  --labels-file /secure/MSP-Podcast-2.0/labels_consensus.csv \
  --transcripts /secure/MSP-Podcast-2.0/Transcripts.zip \
  --precision checkpoint \
  --device cuda
```

Train and Development are intentionally unavailable in this evaluator. Test3
is also rejected because the official page states:

> “The labels, speaker information, transcription, and forced alignment information have been hidden.”

Test1 and Test2 are never pooled. The reported `ccc_mean_va` averages only
valence and arousal. It is not the official MSP V/A/D leaderboard score because
this model has no dominance output.

Each completed external evaluation writes:

```text
<run-dir>/external_benchmarks/<benchmark>/<split>/
  predictions.tsv
  metrics.json
  overlap_audit.json
  external_evaluation_manifest.json
  COMPLETED
  *_predictions_heldout_fold1.csv
  *_predictions_heldout_fold2.csv
  *_predictions_ensemble.csv
```

`predictions.tsv` excludes transcript text and, by default, gold labels. Add
`--include-gold-labels` only for a protected local item-level audit. Existing
external result directories are never overwritten. Files are first written to
a private sibling staging directory and published only after the `COMPLETED`
marker is present, so a failed write cannot look like a completed evaluation.

## Methodological warning

The default split intentionally preserves the old row-level shuffle-and-halves
procedure for comparison. The seven-source data has repeated text across the
two halves, so this protocol can leak lexical items across train and held-out
folds. Treat it as legacy-compatible evaluation, not a strict unseen-text
generalization estimate. A grouped-by-normalized-text split should be a separate
experiment rather than silently replacing the historical folds.

## Tests

The tests do not download Qwen or ET2:

```bash
# Run from the repository root so va_model_code is importable.
python -m pytest -q va_model_code/tests
```

They cover safe ZIP handling, exact legacy counts, exclusions and typo aliases,
token batching, ET2 freezing/cache/alignment, prefix packing, last-text pooling,
loss stability, and dynamic OOF reporting.

## Legacy files

The original encoder implementation is preserved together under
`../legacy/original_va_model/`. The GazeReward source reference is preserved
under `../legacy/gaze_reward_reference/`. Neither directory is imported by the
active `train_model.py`; use the `decoder_va/` package and the commands in this
README for every current run.

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
- makes every Qwen text-backbone parameter trainable and stores it in FP32 in `full` mode;
- trains the gaze projector, boundary embeddings, and VA head fully;
- keeps ET2 frozen in both fine-tuning modes;
- uses BF16 autocast on supported NVIDIA GPUs while keeping trainable parameters in FP32;
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
- aligns ET segments by their exact character occurrences to the first
  overlapping visible Qwen token;
- caches detached results by model revision, selected channels, alignment
  version, and the complete token sequence.

### Occurrence-preserving alignment

Alignment version 3 uses the decoded valid Qwen text as the shared coordinate
system. ET2 receives `text.strip().split()` whitespace words, matching the pinned predictor's published inference preprocessing. Punctuation stays attached to its word.
ET segments must cover that text in order, skipping only whitespace.
The fast Qwen tokenizer supplies character offsets, and re-encoding the decoded
text must reproduce the original lexical token IDs exactly. Alignment never
searches ahead for a later matching word or punctuation mark. Unsupported
tokenizers, incomplete text coverage, or a failed token-ID round trip raise an
error instead of silently assigning gaze to another occurrence.

Only the first active ET2 subword supplies each word's prediction. Selected
channels must be finite; NaN or infinity raises immediately without substituting
a later subword or caching a partial result. Negative predictions are clipped to
zero before alignment, matching the pinned predictor's inference code.
Each valid ET prediction contributes at the first Qwen token overlapping a
non-whitespace character of its segment. The generic remapper supports multiple
segments sharing one Qwen token. These collisions sum `nFix` and `TRT` and take the
arithmetic mean of `FFD`, `GPT`, and `fixProp`, using only valid, finite ET
predictions. The generic remapper itself conserves signed input values; ET2
production inputs have already been clipped to nonnegative values. A token with a valid contribution
stays in the gaze mask even when its resulting TRT is zero.

The full policy is recorded as `gaze_alignment` in the model config, saved
architecture and reconstruction metadata, and run manifests. Reload and resume
require the exact version-3 contract for every gaze condition, including raw TRT.
Old gaze checkpoints lack this contract and cannot be evaluated or resumed with
the corrected alignment; historical evaluation requires their original code.
Start new matched raw and redistribution runs to compare the corrected pipeline.
Schema 8 also records the corrected numerical precision contract, so older
text-only checkpoints likewise require their original code for evaluation.

Each sample is packed before batch padding:

```text
[eye_start] [projected valid gaze vectors] [eye_end] [valid Qwen text] [right padding]
```

This preserves the prefix order in the official
[`gaze_reward` GazeConcat implementation](https://github.com/Telefonica-Scientific-Research/gaze_reward/blob/main/rlhf_rw/models/reward_model_general_sp.py#L154-L210):
eye-start boundary, projected gaze sequence, eye-end boundary, then text. The
Qwen adaptation uses trainable boundary parameters and compact selected gaze
vectors anchored at the verified Qwen token positions rather than adding tokenizer
vocabulary items or retaining every predictor position.

By default, the causal decoder is pooled at each sample's last valid text token. That token
can attend to the complete gaze prefix and all preceding text, whereas the
prefix-side `eye_end` cannot causally attend to later text. Explicit per-sample
pooling indices avoid selecting physical right padding, and position IDs are
rebuilt after packing.

`--readout learned-token` instead appends one trainable prediction token after
the valid text and pools that token. This mode is shared by the new readout
rotation and its matched Raw control; see [readout experiments](../READOUT_EXPERIMENTS.md).

The regression head emits exactly:

```text
[valence, arousal]
```

Both outputs are constrained to `[0, 1]` with the original VA paper's hard
sigmoid output activation. Training requires an explicit choice
of `mse`, `ccc`, or the legacy-compatible 50:50 `mse+ccc`; no uncertainty or
log-variance head is used. The commands below use MSE as the simplest baseline.

## Training text overlap

`--training-text-overlap exclude` (the default) excludes training rows whose text
also appears in the evaluation fold after Unicode NFKC normalization, case
folding, and whitespace collapse. `--training-text-overlap keep` retains every
original training row, including different-source ratings for the same text.
Use `keep` in both arms when comparing against the original dataset split;
use the same policy for Raw TRT concat and TRT rotation.

Labels and dataset names do not affect the overlap audit. Both policies preserve
evaluation rows, order, and labels; the same evaluation fold still selects the
best checkpoint and supplies the reported predictions. No extra split is made.
Original fold files are never rewritten. An empty training set raises an error.

Run manifests record `training_text_overlap_contract`; each fold additionally
records its `training_text_overlap` counts and actual `training_rows`. The root
`dataset_counts_after_filter` remains the full filtered corpus used for OOF
reporting. Two-fold result packaging recomputes the selected policy's audit from
the OOF texts, and resuming a checkpoint with a different policy is rejected.

For the existing `paper7_seed42` folds with only IEMOCAP excluded, `keep` uses
26,898 training rows and 26,893 evaluation rows for held-out fold 1, and 26,893
training rows and 26,898 evaluation rows for held-out fold 2. With one GPU,
batch size 8, gradient accumulation 2, and 10 epochs, the progress totals are
16,820 and 16,810 optimizer updates respectively. These are per-fold totals.
The original full-data concat runs predate other alignment and precision fixes;
matching the rows alone does not make those old results a controlled ablation.

## Gaze redistribution

`--gaze-redistribution` is an independent optional transformation, not a new
concat architecture. The default is `none`, which uses raw gaze after the
alignment described above and adds no learned parameters. `asym-gaussian` requires
`--gaze-fusion prefix-concat` and TRT among `--gaze-features`; invalid
combinations fail before model downloads.

The active implementation is `decoder_va/redistribution.py`. Its position in
the model is:

```text
frozen ET2 -> Qwen alignment + gaze mask -> raw-aligned-feature cache
          -> optional TRT redistribution -> unchanged gaze projector
          -> unchanged gaze-prefix packing -> Qwen -> VA head
```

The cache contains aligned ET2 predictions before redistribution, including the
collision aggregation described above. Redistribution is outside ET2's
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
`gaze_mask`, which marks token anchors with valid ET contributions, rather than
deriving a mask from `TRT != 0` or using only the text padding mask. Both sources and
destinations are masked. Padding, special tokens, and positions without a valid
ET anchor neither contribute nor receive TRT. Valid zero-valued TRT
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

Current run and architecture manifests record the canonical
`gaze_redistribution` configuration; `model.safetensors` stores the two learned
width parameters alongside the existing model weights. Initialization values
in JSON are not the learned final widths. New default run names and archive
condition names distinguish raw gaze from redistribution. Schema-8 checkpoints
record this setting explicitly. Schema-6/7 results remain packageable as historical
results; their weights cannot be resumed or loaded under the corrected precision
and preprocessing contracts. Resume requires the same configuration and
cannot add or remove redistribution. Enabled resume supports ordinary single-file Trainer
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
# This CUDA 13.0 wheel is the tested RTX 5090 / sm_120 configuration.
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu130
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
default `auto` uses BF16 autocast on a BF16-capable CUDA GPU. Full fine-tuning
always stores the trainable backbone in FP32; LoRA may keep its frozen base in
BF16/FP16 but stores trainable adapters in FP32. The regression head and hard
sigmoid run in FP32 outside autocast. `compute_dtype` records the compute mode;
`dtype` and `backbone_parameter_dtype` record backbone storage, and
`trainable_parameter_dtype` records FP32. External evaluation reuses
`compute_dtype` for autocast. This increases full-tuning memory relative to the
historical BF16-parameter implementation; recheck GPU memory with a short smoke
run on the actual machine. An FP32 gaze run must pass
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

The current architecture manifest uses schema version 9. It requires the exact
`precision_contract` and, for gaze models, version-3 `gaze_alignment` contract.
It also records an explicit `gaze_relation` contract, including when disabled.
Schema-8 models without a relation remain loadable for evaluation. Earlier
models require their original code for historical evaluation; converting
old BF16 tensors to FP32 does not recover lost training updates. Start fresh
matched runs. Existing schema-6/7/8 reports remain packageable.
`--resume-from-checkpoint` requires a schema-9 checkpoint under the selected
held-out fold and a matching run manifest, including the text-overlap exclusion
audit. A LoRA checkpoint cannot resume as full fine-tuning, or vice versa.

Legacy metric names and semantics are retained:

- `mse_valence`, `mae_valence`, `pearson_corr_valence`;
- `mse_arousal`, `mae_arousal`, `pearson_corr_arousal`.

CCC and mean metrics are also reported. Per-dataset reporting is generated only
for datasets still present after filtering, so exclusions cannot trigger
hard-coded source lookup failures.

## TRT-relative attention experiment

### Hypothesis and interpretation

The hypothesis is that directly conditioning text-pair attention on differences
in predicted TRT can help the VA model learn useful contextual interactions
with fewer examples than relying on the gaze prefix alone. The prefix and this
branch use the same frozen ET2 predictions; the branch provides a different
route for using those predictions, not new measurements of human gaze.

For mapped tokens, the extra phase depends on `TRT_j - TRT_i`. Q/K still supply
the content of the interaction, and a learned coefficient can increase or
decrease its score. This does not assume that high TRT means emotional
importance, that equal TRT means semantic similarity, or that the rotation
models left-to-right reading. Equal TRT gives exactly zero direct correction.
Effectiveness remains untested on real VA data.

### Operator and numerical behavior

Enable with `--gaze-relation trt-relative-rotation`. The existing frozen ET2
provider and raw TRT gaze-prefix embeddings are retained. Only Qwen3.5
full-attention layers receive a new per-query-head scalar `alpha`, initialized
to zero. Linear-attention layers remain native. Defaults remain unchanged.

After Qwen's native RoPE, the extra logit correction is:

```text
phase_i[h] = alpha[h] * raw_TRT_i / scale
delta_ij[h] = mapped_text_i * mapped_text_j
              * ((R(phase_i) q_i)^T (R(phase_j) k_j) - q_i^T k_j) / sqrt(head_dim)
logits = native_logits + delta + native_causal_and_padding_mask
```

The implementation evaluates this equivalent expression in FP32, avoiding
subtraction of two nearly equal attention matrices for small angles:

```text
angle_ij[h] = alpha[h] * (TRT_j - TRT_i) / scale
J(k) = concat(-k_second_half, k_first_half)
delta_ij[h] = (-2 * sin(angle_ij / 2)^2 * dot(q_i, k_j)
              + sin(angle_ij) * dot(q_i, J(k_j))) / sqrt(head_dim)
```

It is an extra rotation after native RoPE. The new full-head rotation and
Qwen's partial native RoPE need not commute; this is not a replacement of
native position IDs with TRT. Alpha remains in the gradient graph even at
zero, so a zero initial correction does not freeze learning.

The rotation uses the same phase in every split-half plane and the original
Qwen head dimension. Each full-attention layer has its own head coefficients.
`--gaze-relation-scale` is a fixed positive divisor (default `1.0`) in ET2's
output units; it is not estimated from labels or evaluation data. There is no
claim that the ET2 output unit represents milliseconds.

Only original text tokens selected by the provider's alignment mask participate.
Their positions are shifted by each example's compact gaze-prefix length plus
the two boundary tokens. Missing anchors, continuation subtokens without an
anchor, prefix tokens, and padding are excluded; a valid TRT of zero remains an
anchor. No Gaussian redistribution is allowed alongside this initial condition.
Text-to-prefix logits receive no direct correction. Softmax normalization and
changes propagated through earlier layers can still change their weights or
later-layer logits.

The correction and alpha storage use FP32. A separate alpha optimizer group
uses `--gaze-relation-learning-rate` (default `1e-3`) with zero weight decay.
Logs record alpha norm, maximum magnitude, mean, and current learning rate.
The branch requires `--attn-implementation eager`, set automatically when
enabled. Force eager in the raw comparison too. The relation uses extra score
matrices and pairwise trigonometric computations. Production GPU memory and
full-model throughput require measurement on the target machine.
Native Qwen query gating, grouped keys/values, masks, and one shared softmax are
preserved. Explicit per-forward gaze arguments support checkpoint recomputation;
cached generation is unsupported for this regression-only branch.

Train a fresh run to enable the method. Architecture/run/result manifests record
the relation, scale, and optimizer policy; safe reload and resume reject absent,
malformed, nonfinite, or incompatible alpha weights. Evaluation always preserves
input row order, even when training groups batches by length. Prediction export
also checks returned labels against the original metadata when labels exist.

### Compact execution and performance verification

`compact_anchors_v1` retains the same equation, FP32 correction, alpha parameters,
checkpoint keys, and masks. It gathers only mapped text anchors before FP32
conversion and computes the two extra matrix products over those anchors. If
the packed sequence length is `L` and a batch has at most `M` anchors per sample,
the extra matrix products operate on `M × M` instead of `L × L`. A correction
tensor is still scattered back into `L × L`; native eager attention, its dense
logits, and its softmax remain unchanged. This is not a linear-memory attention
implementation.

The packed anchor layout is built once per forward and passed explicitly through
checkpoint replay. Prefix packing uses batched gathers and one host metadata
read instead of per-sample scalar reads and dynamic Boolean gathers. Rotation
combines successful-path tensor validation into one host check per layer call.
Zero/one-anchor batches return an exact zero correction while retaining the
gradient graph. Zero alpha with two or more anchors still computes the full
correction graph, preserving the gradient that learns alpha.

Forward and backward comparisons cover dense/compact computation, native hybrid
Qwen attention, BF16, grouped keys, partial native RoPE, padding, LoRA, multiple
outstanding forwards, and checkpoint recomputation. Floating-point reduction
order can differ; this preserves the mathematical operator, not a promise of
bitwise identical training trajectories on every device. Run manifests record
`gaze_relation_implementation` separately from the unchanged relation contract.
Running Python processes retain their loaded code; updating Git alone does not
apply this optimization to an already-running training process.

An offline Python benchmark compares native eager attention, a frozen dense
rotation reference from commit `1f95381`, and the optimized production path. It
checks outputs and Q/K/V/alpha gradients before timing, including nonzero alpha.
From `va_model_code` on the training GPU:

```bash
python benchmark_gaze_rotation.py \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --gradient-checkpointing \
  --output ../results/gaze_rotation_benchmark_4090.json
```

The default head dimensions are an explicit synthetic template. Pass
`--config /path/to/local/config.json` to use the model's stored dimensions without
downloads. Timings include two attention forward/backward microbatches per
measurement, with checkpoint replay. JSON records wall time, CUDA event time,
peak allocated memory, numerical errors, shapes, and device/precision settings.
TF32 is recorded and is not enabled by the script. Optional `--profile` writes
Chrome traces and increases runtime.

This is a single full-attention-layer microbenchmark. It excludes ET2, packing,
projections, native RoPE, linear-attention layers, MLPs, optimizer updates, and
evaluation. Its speed ratio must not be presented as a full-training speedup or
used directly to divide a training ETA. For an offline CPU correctness check:

```bash
python benchmark_gaze_rotation.py \
  --device cpu --dtype float32 --batch-size 2 \
  --heads 4 --kv-heads 2 --head-dim 8 \
  --lengths 8 --anchor-counts 0 1 half \
  --warmup 1 --iterations 2 \
  --output ../results/gaze_rotation_benchmark_cpu_smoke.json
```

### Matched controls

Keep data, fold use, initialization seed, precision, effective batch size,
optimizer, training steps, checkpoint selection, and eager attention identical.
The existing fold protocol and checkpoint selection are unchanged.

| Arm | Options in addition to the common training settings | Purpose |
| --- | --- | --- |
| Raw TRT prefix | `--gaze-relation none --attn-implementation eager` | Current input-feature baseline |
| Aligned TRT rotation | `--gaze-relation trt-relative-rotation --gaze-relation-input aligned --gaze-relation-learning-rate 1e-3 --attn-implementation eager` | Test direct TRT-conditioned interactions |
| Shuffled TRT rotation | `--gaze-relation trt-relative-rotation --gaze-relation-input shuffled --gaze-relation-shuffle-seed 42 --gaze-relation-learning-rate 1e-3 --attn-implementation eager` | Same module and TRT distribution, changed word-to-TRT correspondence in the relation only |

All three arms retain the genuine, aligned raw-TRT prefix. The shuffled arm
permutes only the relation values among the original mapped text anchors.
It preserves the exact within-example multiset, including valid zeros, and
never includes padding, prefix positions, or unmapped subtokens. Its permutation
is fixed by the experiment seed, active token IDs, and original anchor positions
using `blake2b_rank_v1`; labels and TRT magnitudes are not part of the hash.
Batch order, batch padding, global RNG state, train/eval mode, and checkpoint
replay cannot change it. Zero/one-anchor examples and identical values cannot
provide an informative shuffle; identity permutations are permitted.

If omitted, `--gaze-relation-shuffle-seed` uses `--seed`, fixed across the two
folds. The resolved seed and algorithm are stored in a version-2 relation
contract. Existing version-1 aligned contracts remain loadable. Condition names
distinguish aligned and shuffled runs and include the shuffle seed. Resuming
with a different input mode or shuffle seed fails explicitly.

The hypothesis gains support only if aligned TRT rotation improves on both
the matched raw arm and the shuffled arm across repeated seeds. An improvement
over raw alone could reflect the added module; no improvement over shuffled
does not establish a benefit from correct word-to-TRT correspondence. These
controls do not establish cognitive validity or separate TRT from correlated
lexical properties such as word length; those need further controls.

Each completed relation fold writes `gaze_relation_parameters.json` from the
selected model used for final prediction, including every layer/head's alpha
and `alpha / scale`. `package_results.py` includes this small file in the
results-only ZIP when present. It contains no model weights or original text.
Older runs without the report still package normally.

The CPU diagnostic in `run_gaze_relation_diagnostic.py` uses tiny randomly
initialized Qwen, synthetic tokens/TRT, and planted teacher targets. Its raw,
relation, and shuffled-relation arms use the production input-control path and
check mechanism learnability with paired
initialization and a held-out synthetic set. This is neither pretrained ET2+Qwen
VA evidence nor a novelty demonstration. A real comparison still requires
matched pretrained runs, several seeds, and an independent development split
for checkpoint selection before final testing.

### Readout-connected rotation

The legacy `trt-relative-rotation` condition retains its original mapped-text
pair scope and last-text-token pooling. It gives zero direct correction for
zero/one-anchor inputs. Its final full-attention layer can directly alter the
regression readout only when that last text token is itself an eligible anchor.

Use `--readout learned-token --gaze-relation trt-readout-rotation` for the new
condition. A learned embedding is appended after each sample's valid text,
before padding. Its extra TRT phase is fixed at zero; this is an explicit
reference position, not a predicted fixation duration. The original text
anchors keep their raw ET2 TRT values. Existing text-to-text corrections remain,
and the new readout query receives corrections against preceding mapped text
keys. Prefix tokens, unmapped text, and padding stay outside the rotation mask.
Native positional RoPE, causal attention, and the original raw TRT prefix remain.

For readout query `r` and text key `j`, the extra phase difference is
`alpha[h] * TRT[j] / scale`. Thus a single positive-TRT anchor can affect the
pooled state and the final full-attention layer's alpha gradient. Zero anchors
or all-zero TRT still give zero correction. This establishes an information
path, not a guaranteed performance gain or a monotonic attention preference
for larger TRT. The absolute TRT reference changes the mechanism: it is no
longer invariant to adding a constant to all text TRT values.

The matched Raw condition is `--readout learned-token --gaze-relation none`,
also with eager attention. Both arms share the same seeded readout/head/projector
initialization; alpha starts at zero, giving identical initial outputs. Changing
only from the historical last-token Raw model would confound the readout change
with the rotation change. A learned readout adds one packed position and
`hidden_size` trainable parameters (1024 for the pinned 0.8B backbone) in both
arms; the rotated arm additionally has its existing 48 head coefficients.

Schema 10 records the readout contract in root/fold/architecture/reconstruction
metadata. Package validation, resume, and model loading reject mixed contracts
and missing, malformed, nonfinite, or non-FP32 learned-readout weights. Schema
8/9 models continue to load with their original last-token pooling. They cannot
be resumed as learned-readout models; train new conditions in fresh run folders.

`evaluate_gaze_relation.py` evaluates a completed checkpoint with learned alpha,
zero alpha, and deterministically shuffled relation-only TRT. It retains the raw
prefix and uses identical ordered batches for every arm. Reports include source,
sentence/word group, anchor count, original last-token eligibility, actual shuffle
changes, and causal-pair phase statistics. Singleton TRT cannot be permuted
within its sample, so the shuffle control is uninformative for those rows;
zero-alpha remains useful for the new readout mechanism. See the complete
[commands and interpretation](../READOUT_EXPERIMENTS.md).

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

### Historical seed-43 BF16 checkpoint paths on the Vast.ai server

The following paths refer to two historical raw-text-free Hugging Face runs
materialized on the benchmark server. Their gaze checkpoint predates alignment
version 2 and requires its original code for evaluation; the current loader
rejects it. For new comparisons, replace both run paths with matched corrected
runs. Set the paths once from the repository's
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

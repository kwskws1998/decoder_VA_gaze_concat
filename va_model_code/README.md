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
fine-tune than an instruction/chat checkpoint. It is also much safer under a
24GB VRAM ceiling than moving directly to a multi-billion-parameter decoder.
The code pins the currently verified model revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

This selection is a capacity, language-coverage, task-formulation, and memory
decision. It is not evidence that Qwen already matches XLM-R on this VA dataset.
That claim can only be made after running both models on the same source bundle,
same folds, same exclusions, and same metrics.

The implementation:

- loads only the Qwen text backbone and discards the LM head and vision tower;
- applies rank-16 LoRA to all linear layers of the text decoder;
- trains the gaze projector, boundary embeddings, and VA head fully;
- uses BF16 on supported NVIDIA GPUs;
- enables gradient checkpointing;
- defaults to batch size 4 with four-step gradient accumulation.

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
  --output-dir data_paper7_seed42
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
  --data-dir data_paper7_seed42
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

Verified no-IEMOCAP counts:

```text
fold 1: 30,807 -> 26,885
fold 2: 30,807 -> 26,716
total: 53,601
```

## Training and evaluation

Default Qwen + TRT prefix concat:

```bash
python train_model.py qwen3.5-0.8b mse
```

Choose one raw ET2 feature:

```bash
python train_model.py qwen3.5-0.8b mse \
  --gaze-features FFD
```

Choose several features. They are canonicalized to the published ET2 order:

```bash
python train_model.py qwen3.5-0.8b mse \
  --gaze-features nFix FFD GPT TRT fixProp
```

Every run manifest records the canonical feature names, zero-based indices, and
five-bit mask in `(nFix, FFD, GPT, TRT, fixProp)` order. For example,
`--gaze-features TRT nFix` is stored as names `[nFix, TRT]`, indices `[0, 3]`,
and mask `[1, 0, 0, 1, 0]`.

No-IEMOCAP run:

```bash
python train_model.py qwen3.5-0.8b mse \
  --no-iemocap
```

Text-only Qwen ablation:

```bash
python train_model.py qwen3.5-0.8b mse \
  --gaze-fusion none
```

Dataset and Trainer-API validation without tokenizer or model downloads:

```bash
python train_model.py --dry-run --no-iemocap
```

### Original-paper protocol, single-seed Qwen A/B

Generate `data_paper7_seed42` first with the paper-protocol command above. Then
run the text-only baseline:

Relevant word-for-word excerpts from the paper are:

> “randomly split in half”

> “hard sigmoid activation function”

> “batch size was fixed at 16”

> “models were trained during 10 epochs”

Source: [Mendes and Martins (2023), Section 5](https://arxiv.org/pdf/2302.14021#page=7).

```bash
python train_model.py qwen3.5-0.8b mse \
  --data-dir data_paper7_seed42 \
  --gaze-fusion none \
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
  --output-dir Preds/paper7_qwen_baseline_seed42
```

Run the matching gaze condition, choosing the desired raw feature subset:

```bash
python train_model.py qwen3.5-0.8b mse \
  --data-dir data_paper7_seed42 \
  --gaze-fusion prefix-concat \
  --gaze-features TRT \
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
  --output-dir Preds/paper7_qwen_gaze_TRT_seed42
```

The top-level seed is 42. Internally, held-out folds 1 and 2 use fold seeds 42
and 43 respectively. The same fold seed is reused across baseline and gaze so
their shared Qwen/LoRA and regression-head initialization is paired. This is an
ablation-control choice, not a requirement stated by the original paper.

Compare the two root `oof_metrics.json` files, not the arithmetic mean of fold
metrics. Both must report `n_examples == 63823`. The paper-facing metrics are
Pearson, RMSE, and MAE for valence and arousal. The two
`oof_predictions.tsv` files must have identical `index`, `held_out_fold`,
`dataset_of_origin`, `valence`, and `arousal` columns; only predictions should
differ. One top-level seed supports a descriptive A/B result, not a
seed-variance or significance claim.

The evaluation protocol remains fixed two-fold out-of-fold:

- train fold 2, predict held-out fold 1;
- load a fresh model, train fold 1, predict held-out fold 2;
- combine predictions once into the OOF report.

Run one held-out fold for recovery or a smoke run:

```bash
python train_model.py \
  --held-out-folds 1 \
  --max-steps 1 \
  --epochs 1
```

If the exact machine still runs out of memory, lower
`--train-batch-size` to 2 or 1 and raise
`--gradient-accumulation-steps` proportionally.

Each run writes:

```text
Preds/<run>/
  heldout_fold1/
    checkpoints/
    final_model/
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

`final_model/` contains a complete safe state dict, the locally saved tokenizer,
and a versioned architecture manifest with the exact decoder/ET revisions,
LoRA settings, gaze projector dimensions, and output contract. Reload it
strictly with:

```python
import torch

from decoder_va import load_saved_decoder_va_model

model, tokenizer = load_saved_decoder_va_model(
    "Preds/<run>/heldout_fold1/final_model",
    dtype=torch.bfloat16,
)
model.to("cuda")
```

The reload path executes no repository-supplied Python. It reconstructs the
pinned Qwen/LoRA architecture, validates `decoder_va_architecture.json`, and
loads `model.safetensors` with strict key checking. The pinned Qwen checkpoint
must be available locally or from Hugging Face during reconstruction; ET2
weights remain external and are fetched lazily only when gaze inference starts.

The selectable-feature, hard-sigmoid two-output head uses architecture manifest
schema version 5. Version 4 was TRT-only and used a different output activation,
version 3 allowed the removed four-output uncertainty head, and version 2 used
the incompatible postfix gaze contract. Older schemas are rejected on strict
reload rather than silently reinterpreting their weights. Do not resume an
older Trainer checkpoint. `--resume-from-checkpoint` requires a checkpoint
under the selected held-out fold and a matching version-5 `run_manifest.json`.

Legacy metric names and semantics are retained:

- `mse_valence`, `mae_valence`, `pearson_corr_valence`;
- `mse_arousal`, `mae_arousal`, `pearson_corr_arousal`.

CCC and mean metrics are also reported. Per-dataset reporting is generated only
for datasets still present after filtering, so exclusions cannot trigger
hard-coded source lookup failures.

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

`fold1.py`, `fold2.py`, `models.py`, `custom_trainer.py`, `metrics.py`,
`data_loader.py`, and `utils.py` are retained unchanged only as provenance for
the original implementation. The new `train_model.py` does not import them.
Use the `decoder_va/` package and the commands in this README for all new runs.

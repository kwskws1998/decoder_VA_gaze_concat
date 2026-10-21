# Qwen 0.5B ExPeL-inspired VA pilot

Run date: 2026-10-20 (Asia/Seoul).

The pipeline runs end to end using only one frozen Qwen2.5-0.5B-Instruct model.
The results do **not** support recommending this configuration for VA prediction.
Retrieval plus insights improves pooled MSE over the fixed few-shot model, but
the effect reverses between folds, insights alone is better overall, and a
training-mean predictor is much more accurate than every Qwen condition.

## Matched results

Scores use the repository's normalized [0, 1] labels. Lower MSE and MAE are better.
Every condition predicts the same 24 rows per fold, 48 rows total.

| Condition | Fold 1 MSE | Fold 2 MSE | Pooled MSE | Pooled MAE |
|---|---:|---:|---:|---:|
| Fixed few-shot Qwen baseline | 0.144638 | 0.109994 | 0.127316 | 0.308228 |
| Retrieval only | 0.118925 | 0.155907 | 0.137416 | 0.342425 |
| Insights only | 0.138576 | 0.074581 | 0.106579 | 0.272082 |
| Retrieval + insights | 0.101053 | 0.123649 | 0.112351 | 0.295675 |
| Training-label mean | **0.015340** | **0.015827** | **0.015583** | **0.085164** |
| Retrieved memory score mean | 0.017771 | 0.017209 | 0.017490 | 0.092875 |

This is an exploratory feasibility pilot, not a full-data or multi-seed efficacy
result. It does not compare against the existing fine-tuned Qwen3.5/TRT model.
No failed VA predictions were excluded: all 192 Qwen evaluations produced valid
scores, and the fallback count was zero in every condition.

## Configuration and data

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, 494,032,768 parameters.
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Frozen weights, CPU FP32, four threads, greedy generation, context limit 4096.
- No external LLM, ET2, TRT, neural retriever, server, or gradient updates.
- Existing checksum-verified seven-source archive; source-scale preprocessing.
- Pilot sources: EmoTales sentences, Emobank, Facebook.
- Per evaluation fold: 3 EmoTales, 16 Emobank, and 5 Facebook rows.
- Per fold: 3 fixed gold demonstrations and 12 separate experience tasks.
- Success tolerance: absolute error at most 0.10 on **each** VA dimension.
- One training retry with reference-score feedback; assisted success is logged
  separately and does not constitute a held-out prediction.
- Fold 1 excludes 120 training rows matching the full evaluation fold.
- Fold 2 excludes 102 such training rows. Exclusion occurs before subsampling.
- Memory sizes including fixed examples: 6 for fold 1 and 15 for fold 2.

## Rule quality and negative evidence

The first run's original structured extraction rejected all six rule-update
batches. It is preserved under
`results/qwen05_expel_sentence_pilot_seed42/`, including its original source
snapshot. No manually supplied rules were substituted.

The revised implementation simplifies extraction and, on schema failure, asks
the same Qwen for prose that Python serializes as ADD. The revision was motivated
by training-stage output-format failures. The same held-out pilot set was reused;
these remain exploratory results, not an independent confirmation set.

The revised run produced 1 rule for fold 1 and 3 rules for fold 2. Two of the six
update batches still failed validation. Accepted rules included:

> Add your prediction after label feedback.

> Use the feedback provided in the label to inform future predictions about valence and arousal.

These are not useful general instructions for a new text whose labels are
unknown. Syntactically accepted extraction therefore does not establish that
meaningful, transferable knowledge was learned. The evaluator did not provide
target labels despite these generated instructions.

Additionally, 41 of 48 baseline predictions satisfy `valence + arousal = 1`
within `1e-8`. That is an observed output pattern, not evidence about the model's
internal reasoning. Its ability to estimate the two continuous dimensions needs
substantial improvement before interpreting prompt changes as affective reasoning.

## Artifacts and verification

Revised run:
`results/qwen05_expel_sentence_pilot_v2_seed42/`.

- `summary.json`: full-precision per-fold and pooled metrics.
- `predictions.json`: row-level predictions for all six conditions.
- `fold1/insights.json`, `fold2/insights.json`: raw outputs, fallback responses,
  operation histories, and final rules.
- `manifest.json`, `backend.json`, `source/`: data, code, model, runtime provenance.
- `validation.json`: independently recomputed metrics, source/data hash checks,
  memory isolation, replayed insight operations, and cache identity validation.
- The revised run recorded 248 unique generations, 5,640 generated tokens,
  and 305.25 seconds inside model generation. This excludes preparation,
  the first failed-extraction run, and development probes.

Verification command:

```bash
python -m pytest -q expel_code/tests \
  va_model_code/tests/test_filters.py \
  va_model_code/tests/test_data.py
```

Result: **97 passed**, including 28 ExPeL-specific checks. Pyflakes and
`git diff --check` passed. An exact resume reused 248 cached responses, performed
zero new generations, and produced byte-identical prediction artifacts.

An additional existing Qwen3.5 overlap-training suite was attempted. Eight tests
failed because the installed Transformers 4.49.0 does not provide
`transformers.models.qwen3_5`. That environment limitation is separate from the
Qwen2.5 run; the complete repository suite is not claimed to pass.

## Reproduce

Use a new run name unless resuming the exact original run:

```bash
python -u expel_code/run.py \
  --run-name qwen05_expel_sentence_reproduction \
  --download-data --sentence-only \
  --experience-samples 12 --eval-samples 24 \
  --model-path artifacts/expel/Qwen2.5-0.5B-Instruct \
  --device cpu --threads 4 --seed 42
```

For a fresh machine without the local weights, omit `--model-path` to download
the pinned revision. See [README.md](README.md) for full-fold evaluation.

# Qwen 0.5B versus 3B: matched experiential VA pilot

Run date: 2026-10-20 (Asia/Seoul).

The 3B model greatly reduces the errors seen with 0.5B. On this pilot, adding
retrieval plus insights to 3B increases MSE by 8.17% over fixed few-shot prompting.
Insights alone produces the best 3B result, reducing MSE by 2.11%, with small
improvements in both folds. The complete retrieval-plus-insights condition
improves fold 1 but worsens fold 2.

Both models were actually run on the same Mac GPU using FP16. These are
frozen text-generation experiments: no fine-tuning, regression head, TRT/ET2,
external LLM, or inference server is involved. The implementation is a supervised
VA adaptation inspired by ExPeL, not an original-benchmark reproduction.

## Matched results

Each arm scores the same 24 held-out rows per fold, 48 total. MSE and MAE average
the valence and arousal dimensions on the normalized [0, 1] scale; lower is better.

| Condition | 0.5B MSE | 3B MSE | 0.5B MAE | 3B MAE |
|---|---:|---:|---:|---:|
| Fixed few-shot | 0.125647 | 0.011998 | 0.306353 | 0.073818 |
| Retrieval only | 0.136203 | 0.012957 | 0.341175 | 0.074599 |
| Insights only | 0.102279 | **0.011745** | 0.264165 | **0.072047** |
| Retrieval + insights | 0.110341 | 0.012978 | 0.292175 | 0.073167 |
| Training-label mean | **0.015583** | 0.015583 | **0.085164** | 0.085164 |
| Retrieved-memory score mean | 0.017490 | 0.017517 | 0.092875 | 0.097125 |

The 3B fixed few-shot baseline changes pooled MSE by -90.45%
relative to the 0.5B baseline. Within 3B, retrieval plus insights changes pooled
MSE by +8.17% relative to fixed few-shot prompting.

The complete method slightly improves MAE relative to the 3B baseline despite
worsening MSE. Insights alone also has the highest 3B mean Pearson correlation
(0.551571) and mean CCC (0.493347); the baseline values are 0.517568 and 0.458228.

| 3B condition | Fold 1 MSE | Fold 2 MSE |
|---|---:|---:|
| Fixed few-shot | 0.015638 | 0.008358 |
| Retrieval only | 0.017107 | 0.008806 |
| Insights only | 0.015265 | 0.008226 |
| Retrieval + insights | 0.015101 | 0.010856 |
| Training-label mean | 0.015340 | 0.015827 |
| Retrieved-memory score mean | 0.019570 | 0.015465 |

Full per-dimension MSE, MAE, Pearson, and CCC are in each run's `summary.json`.
The training-mean control uses all eligible training labels; it has a larger
label budget than the 3 fixed demonstrations plus 12 experience tasks per fold.

## Experience and rule extraction

Success requires absolute error at most 0.10 on each dimension. Retry feedback
reveals training labels, so assisted successes are not held-out generalization.

| Model | Fold | Initial successes | Additional assisted successes | Memory rows | Final rules | Rejected batches |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 1 | 0/12 | 3/12 | 6 | 1 | 2/3 |
| 0.5B | 2 | 3/12 | 9/12 | 15 | 3 | 0/3 |
| 3B | 1 | 3/12 | 9/12 | 15 | 2 | 0/3 |
| 3B | 2 | 6/12 | 6/12 | 15 | 1 | 0/3 |

Final 3B rules, copied exactly from the saved model outputs:

Fold 1:

> Text interpretation should consider the overall sentiment expressed in the context and the intensity of language used to predict emotional valence and arousal accurately.

> Text interpretation should focus on the presence of positive or negative keywords and their intensity relative to neutral terms to predict emotional valence and arousal.

Fold 2:

> Texts with higher positive word counts tend to have higher valence, while detailed descriptions or lists can indicate lower arousal.

Accepted syntax is not evidence of useful or causal knowledge. Each model
regenerates its own experiences, memory, and rules. The size comparison therefore
measures the complete pipeline; it does not hold generated memory content fixed.

All 192 Qwen evaluation attempts per model remain in the scores. Final
training-mean fallbacks: 0.5B = 0, 3B = 0. Evaluation
format-repair attempts: 0.5B = 1,
3B = 0.

One of six 3B extraction batches needed the same-model prose fallback after an
invalid JSON response. All six batches were ultimately accepted; this does not
mean that all initial structured responses were valid.

## Controlled configuration

- 0.5B: `Qwen/Qwen2.5-0.5B-Instruct`, 494,032,768 parameters,
  revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- 3B: `Qwen/Qwen2.5-3B-Instruct`, 3,085,938,688 parameters,
  revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Same MPS/FP16 runtime, PyTorch 2.12.1, Transformers 4.49.0, four CPU threads.
- Same source/data hashes, tokenizer files, seeds, selected rows, and prompts.
- Greedy decoding, one beam, explicit repetition penalty 1.1, context limit 4096.
- Same character TF-IDF retriever, top 3 examples, maximum 6 rules, batches of 4.
- Sources: EmoTales sentences, Emobank, Facebook. Each evaluation fold contains
  3, 16, and 5 rows from those sources, respectively.
- Full outer-fold text exclusion precedes sampling: 120 training rows excluded
  for fold 1 and 102 for fold 2.

The checkpoint defaults differ in repetition penalty (0.5B: 1.1; 3B: 1.05).
A preliminary 3B run was interrupted after this was discovered, and both models
were rerun with an explicit shared policy. The incomplete run is preserved under
`results/qwen3b_expel_sentence_mps_seed42/` and is excluded from this comparison.
The explicit-policy 0.5B rerun produced byte-identical predictions to its previous
MPS run. The older CPU/FP32 pilot remains separately documented in
[PILOT_RESULTS.md](PILOT_RESULTS.md).

## Verification and artifacts

- 101 relevant tests passed: ExPeL, data filtering, and source preprocessing.
- Independent audit checked source snapshots, data hashes, and local weight
  hashes against pinned download metadata.
- Cached full requests replayed the complete pipeline byte for byte.
- All-row MSE/MAE independently recomputed; identical rows across arms verified.
- Memory isolation, self-exclusion, and insight-operation histories verified.
- Model-size comparisons require equal effective decoding policies.
- Exact 3B resume reused 238 cached responses with zero new generations and
  byte-identical predictions. SHA-256:
  `9fbfd67503bbe94c07e462d16d3cb39ba2b167dcb5d76c8ed602a3bdad443275`.

Generation accounting:

- 0.5B: 248 unique generations, 5,638 generated tokens, 151.83 seconds inside generation.
- 3B: 238 unique generations, 5,348 generated tokens, 825.27 seconds inside generation.

Generation timing excludes loading, downloading, retrieval, auditing, and earlier
runs. It is execution accounting, not a controlled hardware benchmark.

Runs:

- `results/qwen05_expel_sentence_mps_matched_seed42/`
- `results/qwen3b_expel_sentence_mps_matched_seed42/`

Each includes `manifest.json`, `backend.json`, `predictions.json`, `summary.json`,
`validation.json`, `source/`, and per-fold experiences, insights, and selections.
The 3B run also contains `comparison.json`.
`audit_source.py` preserves the audit implementation; execution and resume logs
are retained with the corresponding run.

The full repository suite is not claimed to pass: the installed Transformers
version lacks the module needed by the separate Qwen3.5 overlap-training tests.

## Reproduce

From the repository root, use a new run name:

```bash
python -u expel_code/run.py \
  --run-name qwen3b_expel_reproduction \
  --model-size 3b --download-data --sentence-only \
  --experience-samples 12 --eval-samples 24 \
  --model-path artifacts/expel/Qwen2.5-3B-Instruct \
  --device mps --threads 4 --seed 42
```

Omit `--model-path` to download the pinned model on a fresh machine. For 0.5B,
change size, local model path, and run name together. Use the same device and
precision for both models. See [README.md](README.md) for full-fold evaluation
and the independent comparison-audit command.

These are single-seed, 48-row exploratory results on a previously examined pilot
set. They do not establish full-data efficacy, improvements over the existing
fine-tuned Qwen/TRT model, or a general benefit from increasing parameter count.

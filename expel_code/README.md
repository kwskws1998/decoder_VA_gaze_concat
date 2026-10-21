# Single-Qwen experiential VA prediction

This experiment uses **one frozen Qwen2.5-Instruct model** for VA
prediction, supervised retries, and insight extraction. There is no ET2/TRT,
external LLM, embedding model, inference server, LoRA, or regression-head
training. Retrieval uses character TF-IDF cosine similarity. Select a pinned
model with `--model-size`; the default remains `0.5b`.

| Size | Model | Pinned revision |
|---|---|---|
| `0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| `3b` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |

Local checkpoint architecture must match the declared size. Model identity,
weight hashes, precision, and device are included in provenance and cache keys.
Both sizes use the same explicit decoding policy: greedy generation, one beam,
and repetition penalty 1.1. This overrides their different checkpoint defaults
(0.5B: 1.1; 3B: 1.05). The complete explicit policy is recorded in `backend.json`.

The completed CPU pilot and its negative findings are recorded in
[PILOT_RESULTS.md](PILOT_RESULTS.md).
The matched 0.5B/3B GPU comparison is recorded in
[MODEL_SIZE_RESULTS.md](MODEL_SIZE_RESULTS.md).

## Method and scope

This is a **supervised VA adaptation inspired by
[ExPeL](https://arxiv.org/abs/2308.10144)**, not a reproduction of its original
interactive-agent benchmarks.

1. Start each outer fold with fresh memory and insights.
2. Exclude training texts matching any text in the complete evaluation fold,
   using NFKC, case folding, and whitespace normalization, before sampling.
3. Select fixed gold-labelled training demonstrations and separate experience
   tasks, reproducibly and without looking at label values.
4. Predict each experience task. Success requires both absolute VA errors to
   be within `--success-tolerance` on the normalized [0, 1] scale.
5. On failure, expose training reference scores and retry. Preserve initial
   predictions, outputs, references, and retries. Label-assisted successes are
   counted separately and are not evidence of generalization.
6. Store successful generated predictions as retrievable examples, alongside
   fixed gold demonstrations. Failed attempts remain available for insights.
   The retrieval index contains only training text, not labels.
7. Let the same Qwen iteratively ADD, EDIT, UPVOTE, or DOWNVOTE insights from
   training experiences. Invalid operation batches are rejected atomically
   and recorded. If structured output fails, ask the same model for one prose
   guideline and serialize it as ADD in Python. Verbatim training-text copies
   are rejected. This fallback is explicitly logged; no manually invented
   rules replace failed extraction.
8. Freeze memory and insights. Evaluate identical held-out rows in every arm.

Continuous reference scores replace environment feedback. Experiences contain
predictions and supervised corrections rather than ReAct action trajectories.
TF-IDF replaces the original neural embedding retriever. No weights are updated.

## Run

Use the existing project environment, or install the standalone dependencies:

```bash
python -m pip install -r expel_code/requirements.txt
```

From the repository root, run a pilot using the three sentence datasets
(EmoTales sentences, Emobank, Facebook), both held-out folds, and fixed samples:

```bash
python -u expel_code/run.py \
  --run-name qwen05_expel_sentence_trial \
  --download-data \
  --sentence-only \
  --experience-samples 12 \
  --eval-samples 24 \
  --seed 42
```

`--download-data` invokes the existing checksum-verified seven-source ZIP
download and source-scale preprocessing. A valid local ZIP is reused.
By default, the pinned Qwen revision is downloaded through Hugging Face.
To reuse the model downloaded for the local pilot, append
`--model-path artifacts/expel/Qwen2.5-0.5B-Instruct`.

The default device is CUDA, then MPS, then CPU, according to availability.
`--device cpu --threads 4` explicitly selects CPU. GPU uses FP16; CPU uses FP32.
The maximum context defaults to 4096 tokens.

For the same pilot with 3B:

```bash
python -u expel_code/run.py \
  --run-name qwen3b_expel_sentence_trial \
  --model-size 3b \
  --download-data --sentence-only \
  --experience-samples 12 --eval-samples 24 \
  --seed 42
```

When using local 3B weights, add
`--model-path artifacts/expel/Qwen2.5-3B-Instruct`. For a size comparison, use the
same explicit device for both models. Regenerate experiences and rules with
each model; keep the task selection, prompts, and hyperparameters unchanged.
The original 0.5B CPU result is retained separately from comparisons on MPS.

For full held-out evaluation with a larger experience collection:

```bash
python -u expel_code/run.py \
  --run-name qwen05_expel_sentence_full_seed42 \
  --download-data \
  --sentence-only \
  --experience-samples 256 \
  --eval-samples 0 \
  --model-path artifacts/expel/Qwen2.5-0.5B-Instruct \
  --seed 42
```

Zero sample count means all available rows. `--experience-samples 0` additionally
collects all eligible training texts and can be expensive. Omit `--sentence-only`
to include word-rating sources; IEMOCAP is excluded unless `--include-iemocap`
is passed. Full evaluation is more expensive than regression inference because
every Qwen condition generates text.

Append `--dry-run` to validate inputs without loading a model. Append `--resume`
with the same arguments to reuse a partial run. Changed data, code, dependencies,
model files, runtime, or configuration are rejected. Outputs are cached by full
request and model identity, so restarting reuses completed generations.
Use a new run name for changed configurations.

Old runs retain their original sources in `source/`; changing model support
changes the code hash and therefore requires a new run rather than resuming
under revised code.

## Controls and output

| Arm | Demonstrations | Extracted rules |
|---|---|---|
| `baseline` | Fixed training examples | No |
| `retrieval` | Retrieved successful experiences | No |
| `insights` | Same fixed examples | Yes |
| `expel` | Same retrieved examples | Yes |
| `train_mean` | Training-label mean, without Qwen | No |
| `retrieval_mean` | Unweighted mean of retrieved memory scores | No |

The baseline is **fixed few-shot generation**, not the existing fine-tuned
Qwen3.5 regression baseline and not zero-shot generation. Historical results
must not be treated as a matched comparison.

Outputs are saved under `results/<run-name>/`:

- `manifest.json`: configuration, exclusions, code/data hashes, dependencies,
  and local model hashes when a local directory is supplied.
- `backend.json`: model revision, parameter count, dtype, device, and runtime.
- `foldN/selection.json`: fixed examples and sampled row identities.
- `foldN/experiences.json`: predictions, errors, and training-only retries.
- `foldN/memory.json`: retrievable demonstrations with provenance.
- `foldN/insights.json`: every update, rejected output, and final rules.
- `foldN/predictions.json` and `predictions.json`: predictions and row identities.
- `foldN/summary.json` and `summary.json`: MSE, MAE, Pearson, CCC,
  parse-failure counts, and extraction success.
- `cache/responses/`: full prompts, outputs, token counts, and generation times.
- `source/`: a copy of the exact Python sources covered by the manifest hashes.

One format-repair attempt is allowed for invalid VA JSON, without revealing
the target label. If both attempts fail, use the fold's training mean. Failures
remain in metrics and are counted explicitly. Nonfinite, out-of-range,
duplicate-key, and ambiguous outputs are rejected.

Optional examples and rules are removed from the end if a prompt is too long;
actual IDs used are recorded. Target text is never silently truncated. Oversized
targets or insight batches raise an error: adjust context or batch size in a new
run. Empty source text rows are preserved in evaluation; matching empty training
rows are excluded.

Successful extraction does not establish that the rules are correct. Zero
rules are reported as extraction failure even if evaluation completes.
Sampled pilot metrics do not establish full-dataset improvement.

## Verification

```bash
python -m pytest -q expel_code/tests
```

Tests cover outer-fold isolation, target-label invariance, text-only retrieval,
self-exclusion, malformed outputs, atomic rule updates, retry provenance,
fallback accounting, context handling, model-size validation, matching decoding
policies, and complete output generation.

For completed local runs, audit a matched size comparison without loading model
weights or generating new responses:

```bash
python expel_code/audits/compare_runs.py \
  results/qwen05_expel_sentence_mps_matched_seed42 \
  results/qwen3b_expel_sentence_mps_matched_seed42
```

This verifies data, current and snapshotted sources, local weight hashes against
download metadata, exact cached request identities, and byte-identical pipeline
replay. It independently recomputes all-row MSE/MAE, checks fold isolation, and
requires identical selection and decoding policies between models. Run it
against the matching source checkout, without Python `-O`. It writes each run's
`validation.json` and the second run's `comparison.json`.

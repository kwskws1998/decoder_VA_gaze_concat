# TRT rotation review — 2026-10-20

## Scope and conclusion

Reviewed commit `f4829a388711426b3b4e048956bf377ea8a96a39`, the supplied first-fold log through epoch 6.064, local paper-protocol seed-42 folds, and the pinned Qwen/ET2 tokenizers. No trained weights from the current run were available locally. Tokenizer coverage does not measure actual ET2 values or learned attention effects.

The strongest verified issue is a mismatch between the implemented **mapped-text-pair scope** and the input/readout geometry. The arithmetic and masking implement their stated contract, but the additional rotation is structurally inactive on most inputs, and its final layer usually cannot directly affect the prediction. Separately, the visible training/evaluation trajectory is consistent with overfitting, especially for arousal. These findings do not establish that rotation alone caused the historical performance gap.

## 1. Verified coverage restrictions

| Dataset | Rows | At least two TRT anchors | At least two anchors and final pooled token is anchored |
|---|---:|---:|---:|
| Emobank | 10,062 | 9,774 | 1,286 |
| EmoTales | 1,395 | 1,370 | 25 |
| Facebook | 2,895 | 2,812 | 781 |
| NRC-VAD | 19,971 | 132 | 123 |
| GlasgowNorms | 5,553 | 871 | 0 |
| Word ratings ENG | 13,915 | 102 | 92 |
| Total | 53,791 | 15,061 (28.00%) | 2,307 (4.29%) |

There are 38,727 one-anchor rows and three zero-anchor rows. A word split into several Qwen subtokens still has only one TRT anchor. Rotation requires distinct pairs with both endpoints anchored, so these 38,730 rows have exactly zero correction and zero direct alpha gradient. Raw TRT prefix embeddings remain active. Shared backbone updates learned on other rows can still affect singleton predictions across training.

The regressor reads the final valid text token. That token is frequently punctuation or a continuation subtoken, excluded from the relation mask. The pinned Qwen has full-attention layers 3, 7, 11, 15, 19 and 23, with layer 23 last in the model. Only 4.29% of rows satisfy the necessary geometry for that last layer's alpha to affect the readout. Earlier layers can still influence unanchored readouts through subsequent layers; this is not a claim that all rotation effects vanish on 95.71% of rows.

Relevant code:

- `va_model_code/decoder_va/alignment.py:76`: first-subword anchor policy.
- `va_model_code/decoder_va/gaze_relation.py:282`: singleton shortcut.
- `va_model_code/decoder_va/gaze_relation.py:290`: both-endpoint mask, diagonal exclusion, TRT difference.
- `va_model_code/decoder_va/packing.py:178`: packed anchors and final-text pooling position.
- `va_model_code/decoder_va/model.py:550`: readout used by the VA head.

Independent native-Qwen CPU reproduction changed final-layer alpha from 0 to 0.4: some hidden states changed by 0.101315, while the unanchored pooled final token and all final-layer alpha gradients changed by exactly zero. A two-full-layer control retained a nonzero earlier-layer gradient while the last-layer gradient remained zero. See `last_token_probe.json` and `structural_probe.json`.

The actual-tokenizer audit found zero alignment errors, zero ET2 truncation/missing-word rows, zero anchor-collision rows, and only three Qwen-truncated Facebook rows. Widespread truncation does not explain the current result.

## 2. What the partial log shows

Numbers below are rounded console values, not final OOF metrics.

| Epoch | Eval mean MSE | Valence MSE | Arousal MSE | Valence Pearson | Arousal Pearson |
|---|---:|---:|---:|---:|---:|
| 1 | 0.01465 | 0.01244 | 0.01685 | 0.7758 | 0.5901 |
| 2 | 0.01220 | 0.01079 | 0.01361 | 0.8089 | 0.6667 |
| 3 | 0.01217 | 0.01033 | 0.01402 | 0.8203 | 0.6598 |
| 4 | 0.01253 | 0.01026 | 0.01480 | 0.8204 | 0.6403 |
| 5 | 0.01279 | 0.01032 | 0.01525 | 0.8180 | 0.6302 |
| 6 | 0.01298 | 0.01042 | 0.01554 | 0.8176 | 0.6264 |

Epoch 3 to 6 mean MSE worsens by approximately 6.66%, arousal MSE by 10.84%, and valence MSE by 0.87%, while recent training losses continue decreasing. This is consistent with overfitting, but does not identify its cause. The current configuration selects the lowest evaluation mean MSE and loads that checkpoint before final saving/prediction; epoch 3 is best among the six visible evaluations.

Alpha norm grows from 0.2295 at epoch 1 to 0.3632 at epoch 6. Absolute maximum grows from 0.0793 to 0.1502. A near-zero alpha mean does not mean no learning: signed coefficients cancel in the mean. There is no evidence here for frozen alpha or runaway coefficient magnitude.

## 3. Mechanism and optimization interpretation

The phase is `alpha[head] * (TRT[key] - TRT[query]) / scale`. ET2 features are raw selected outputs clamped at zero, without sentence-wise z-scoring. A common additive TRT offset cancels in this additional relation, while absolute TRT remains in the prefix branch.

This operator does not guarantee that a higher-TRT word receives more attention. The first-order correction also depends on the content-dependent bilinear term `q dot J(k)`, which can have either sign. The measured alpha magnitude alone cannot establish actual phase magnitude, effective attention change or harmful saturation. Those require actual TRT differences and per-layer score diagnostics.

The extra rotation operates over the full head after native partial RoPE; it is not a rescaling of native positional distances. This is documented behavior. Changing the pairing/subspace would be another mechanism variant, not a correction justified by the current log.

Alpha has a separate FP32 optimizer group, peak LR 1e-3 and zero weight decay. Backbone LR is 6e-6. Their ratio is 166.7, but the parameter roles differ, so this ratio alone does not prove a bad learning rate. Warmup of 1,682 updates and subsequent decay match the observed log.

Initial gradient norms 1023 and 1161 are single-update, pre-clipping whole-model norms. Default clipping is 1.0. They are neither average norms across 200 updates nor alpha-only gradients. Later norms fall below one. Additional group-specific gradient measurements would be needed to identify the source of those early spikes.

A minor display inconsistency exists: the base LR is logged from before the scheduler step, while custom alpha LR reads the optimizer after that step. This creates a one-update timestamp difference in the displayed rates; it is not an optimizer or performance bug.

## 4. Historical Raw comparison

Read the actual archive `qwen_full_gaze_TRT_results_only_20260805_130426.zip`, including fold-1 `trainer_state.json` and `run_manifest.json`.

- Historical Raw best evaluation mean MSE: 0.009843639882743001 at epoch 3.
- Current visible rotation best: approximately 0.01217 at epoch 3, about 23.6% higher.
- Both report fold-1 evaluation rows 26,893 and opposite-fold training rows 26,898; historical fold hashes match the local prepared folds audited here.
- Historical metadata: architecture schema 6, Transformers 5.14.1, batch 16 / accumulation 1, attention implementation unspecified, RTX 3090.
- Current code/run settings: schema 9, Transformers 5.16.1, batch 8 / accumulation 2, explicit eager attention plus FLA, RTX 5090, later alignment and precision implementation.

Thus the older Raw result is relevant historical context but cannot isolate the rotation effect. Equivalent effective batch size does not make the complete code and training paths matched.

## 5. Recommended resolution order

1. Evaluate a saved current checkpoint on the same held-out rows with learned alpha, alpha set temporarily to zero, and relation-only TRT shuffled. Report metrics separately by source and by anchor/readout geometry. This tests current inference-time contribution; it does not undo the backbone's training history or replace a matched Raw training control.
2. Measure eligible/nonzero causal-pair phase quantiles, correction/native-logit RMS, per-layer alpha gradients, and especially final-layer activity. Log actual changed TRT under the shuffle control. A singleton-preserving shuffle is uninformative on most rows; for seed 42, 73.77% of all rows have the identity permutation, even before equal-TRT ties.
3. If retaining all no-IEMOCAP data, address the readout connection and singleton limitation before a broad LR sweep. A separate readout query with an explicitly defined reference phase attending to TRT-bearing keys could use absolute TRT even for single-word inputs. This changes the mechanism and needs a new versioned contract and an equally modified Raw baseline.
4. Copying a word's TRT to every continuation subtoken alone does not solve singleton inputs: their TRT differences still vanish. Merely removing masks would also incorrectly assign missing/special-token gaze values.
5. For the existing mechanism, a matched current-code Raw control and a smaller alpha LR such as 1e-4 are focused follow-ups. If current Raw also overfits, reduce backbone LR or use LoRA while retaining the requested data. These are hypotheses to test, not promised improvements. Existing best-checkpoint selection already avoids automatically choosing the final epoch.

No evaluation-fold or text-overlap policy change is proposed here.

## Validation and limits

- Rotation/attention/packing/model review: 183 targeted tests passed; two unavailable hardware checks skipped.
- Trainer/loss/evaluation/kernel-report review: 76 targeted tests passed.
- Independent explicit 2D rotation comparison: head dimension 256, GQA, native partial RoPE factors 0.25/0.5/1, alpha 0/0.15/1. Maximum score error versus FP64 reference approximately 2.14e-6; maximum Q/K-gradient error approximately 1.51e-7. See `rotation_math_probe.json`.
- The earlier supplied server setup log showed successful FLA CUDA forward/backward and checkpoint smoke tests. These do not establish full-model real-data efficacy or eliminate all GPU-specific numerical questions.
- Current trained weights, complete fold histories, per-dataset predictions, actual ET2 TRT distributions and paired current-code Raw results were unavailable. Those are the remaining evidence needed for causal performance attribution.

`audit.json` records the source-log hash, code commit, fold hashes, parsed partial logs and full tokenizer-coverage summary. The three `probe_*.py` files reproduce structural/numerical checks using the project's installed dependencies with `PYTHONPATH=va_model_code`.

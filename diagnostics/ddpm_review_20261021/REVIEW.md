# DDPM + TRT implementation review — 2026-10-21

## Scope and conclusion

Reviewed the active `diffusion_code` and shared VA encoder/Trainer against local
commit `d57d3566b41913972f470bb710a72450e557d285`. The production DDPM files were
unchanged during this review. Their exact fingerprints are in `probes.json`.
Unrelated existing ExpeL and root-documentation changes were left intact.

Available run evidence is the user's pasted held-out fold 1 epoch-1 and epoch-8
logs. The complete learning curve, actual server source fingerprints, saved
model, and final two-fold OOF predictions are not yet available. Local data
counts match the pasted run, but that alone does not certify identical files.
Fold reuse and retention of overlapping training texts are intentional and
are not findings.

No arithmetic, label leakage, missing-gradient, optimizer membership, or
prediction-order defect was reproduced in the tested paths. This is not a claim
that all bugs have been excluded or that the pretrained CUDA run is validated.
The main concerns are the choice of objective, replacing the direct VA
prediction path, and estimating the conditional mean from only 16 common
random trajectories. Their contribution to this run's error is still unknown.

## 1. Direct VA supervision and the old prediction head are removed

Source:

- `diffusion_code/model.py:109`: `encoder.regression_head = nn.Identity()`.
- `diffusion_code/model.py:153`: sample corrupted gold VA and predict epsilon.
- `diffusion_code/model.py:155`: the sole training loss is epsilon MSE.
- `diffusion_code/model.py:163`: prediction is the mean of generated samples.
- `diffusion_code/train.py:236`: construct the pretrained Qwen base encoder;
  no earlier fitted Raw-TRT checkpoint is loaded.

This is a replacement of the regression task head. It does not preserve a
trained Raw-TRT predictor and add a correction to it. Both the new denoiser and
the VA-specific representations must be learned through the noise objective.
There is no auxiliary direct-VA loss or deterministic VA mean head anchoring
optimization. These are intentional implementation choices, not a broken
backpropagation path, but they remove any reason to assume the old predictor's
accuracy will carry over.

Additional changed regularization: the baseline regression head contains
dropout 0.1; the replacement denoiser has no dropout. The gaze projector retains
its baseline dropout. This is a secondary generalization hypothesis, not
evidence of overfitting from the incomplete epoch log.

## 2. Low epsilon MSE does not establish good conditional VA prediction

Let `z = 2*y - 1`, `a = alpha_bar[t]`, and let `z_hat` be reconstructed from the
predicted epsilon before clipping. Direct algebra gives:

```text
epsilon_error² = a/(1-a) * (z_hat-z)²
              = 4*a/(1-a) * (y_hat-y)²
```

This is a per-timestep training reconstruction identity, not an expression for
the error of a complete generated reverse trajectory.

For the actual T=100 cosine schedule:

| Code timestep indices | Mean multiplier on VA reconstruction squared error |
|---|---:|
| 0–24 | 519.2420 |
| 25–49 | 9.6536 |
| 50–74 | 1.8613 |
| 75–99 | 0.2022 |

Equal VA reconstruction errors receive very different epsilon losses across
noise levels. These multipliers alone do not measure actual parameter-gradient
contributions; the errors and network Jacobians also vary with timestep.

### A concrete context-free shortcut

Using only fold-2 label means/variances, construct an affine epsilon predictor:

```text
mu = E[z]
v  = Var[z]
epsilon_hat = sqrt(1-a)/(a*v + 1-a) * (x_t - sqrt(a)*mu)
E[epsilon_error²] = a*v/(a*v + 1-a)
```

It observes noisy VA and timestep but no text, TRT, or individual clean target.
The risk identity uses the first two moments and independent Gaussian noise;
it does not require the actual label distribution to be Gaussian.

For the local 26,898-row no-IEMOCAP training fold:

- Expected all-timestep epsilon MSE: **0.2379075**.
- Expected epsilon MSE in indices 75–99: **0.0055130**.
- Constant training-mean prediction on held-out fold 1 has VA MSE:
  V **0.0309128**, A **0.0242628**, mean **0.0275878**.

The run's early epsilon loss around 0.23 is therefore not evidence by itself
of successful semantic conditioning. Its later noise loss around 0.10–0.12
and nonzero VA correlations do not justify saying it ignores all context.
The actual fitted model needs timestep-binned and condition-ablation tests.

### Paper check

The original DDPM paper, §3.4, Eq. (14), does use unweighted epsilon MSE with
uniform timestep sampling. Its exact stated rationale includes:

> “we found it beneficial to sample quality”

Source: [Ho et al., Denoising Diffusion Probabilistic Models](https://arxiv.org/html/2006.11239),
§3.4 / Eq. (14), with Algorithms 1–2.

Thus this objective is not an accidental transcription error. That image
sample-quality finding does not establish superiority for two-dimensional
conditional-mean VA regression. The weighting analysis above is our own
calculation for this implementation, not a quote or result from that paper.

## 3. Fixed 16-sample evaluation has a measurable finite-sample sensitivity

`diffusion_code/ddpm.py:31` creates one CPU-generated noise bank, and
`:76` expands it across every input. This is deliberate: evaluation is
reproducible and invariant to batching/order. However, using the same small
bank can make finite-sample deviations correlated across examples. Increasing
the number of evaluation rows does not necessarily average that component away.

For independent generated paths and a fixed input, the sample mean has variance
`Var(Y_generated | input) / N` over noise-bank draws. This statement is about
the model-generated distribution, not calibrated human uncertainty, and does
not remove bias from the learned model or discretization.

A synthetic Gaussian conditional oracle with VA standard deviation 0.1,
the actual reverse-step equations, and seed 2026 produced:

| Samples | MSE versus known conditional mean |
|---|---:|
| 16 | 0.00114907 |
| 64 | 0.00050248 |
| 256 | 0.00003875 |

At 16 samples, seeds 2027 and 2028 instead gave 0.00030700 and 0.00074721.
These are synthetic sensitivity examples, **not estimates of the actual model's
sampling error**. This oracle probe disabled intermediate clean clipping to
match the unbounded Gaussian oracle, retained the production fixed posterior
variance and final [0,1] clamp, and therefore includes those sampler effects.
Its realized error is not a pure unbiased Monte Carlo variance estimate.

With a deterministic point-mass oracle, the same production sampler recovers
the point essentially exactly for different seeds. There is no unavoidable
0.001 error floor in every DDPM model.

The current result artifacts save only sample means. They cannot reveal actual
sample dispersion or seed sensitivity. Re-evaluate the same retained checkpoint
using more samples and several preselected seeds; report the spread instead of
selecting the best sampling seed.

## 4. Confirmed functioning paths and new evidence

| Check | Evidence |
|---|---|
| Forward/noising formula, posterior coefficients and final-step noise omission | Existing independent analytic tests pass |
| Entire 100-step sampler with exact conditional epsilon | Maximum VA absolute error **1.12e-8** |
| Entire 250-step sampler | Additional exact-oracle test passes |
| Single TRT anchor can affect prediction and receive gradient | Existing gradient/masking test passes |
| Single-anchor TRT actually learnable through true DDPM training | Synthetic generated VA MSE **0.00001294** at update 1,682 |
| Same synthetic model with reversed conditions | VA MSE **0.08793254** at update 1,682 |
| No-anchor fallback, padding, target/text alignment | Existing shared encoder and DDPM tests pass |
| BF16 head/loss FP32 and gradient checkpoint replay | Tiny native-Qwen CPU tests pass |
| Gradient accumulation normalization | Actual Trainer backward for two microbatches matches full-batch gradients with identical corruption |
| Optimizer membership, separate LRs and scheduler | All trainable weights covered once; head/backbone LR ratio maintained through warmup and decay |
| Evaluation label isolation and ordering | Labels removed before generation; sequential evaluation tests pass |
| Save/reload/resume/results packaging | Existing complete-lifecycle and tamper-detection tests pass |

The synthetic learning probes use the production 128-wide, 3-block head,
T=100, 16 output samples and LR 1e-4. They use small synthetic fixed encoder
features, no pretrained Qwen, no real ET2, constant LR and CPU FP32. They
establish conditional trainability only, not real-data efficacy.

For the token-TRT synthetic probe, epsilon loss continued to fall between
updates 1,682 and 3,000 while generated-VA MSE increased from 0.00001294 to
0.00006573. Both errors are tiny; this is another illustration that the
two objectives are not interchangeable, not evidence of real-data overfitting.

### Gradient norms

Installed Transformers 5.16.1 / Torch 2.12.1 / Accelerate 1.14.0 were inspected
locally. The selected TrainingArguments have `max_grad_norm=1.0`.
Trainer logs the pre-clip norm and clips the whole model before the optimizer
step. The review test independently checks the post-clip norm is at most 1.

Large initial logged norms are therefore not proof of an unbounded optimizer
update. They are still an optimization signal worth investigating. The current
logs cannot identify whether Qwen, the gaze projector, or the new head dominates
that norm. Do not infer a proportional effective Adam learning rate from the
clip factor: its moment normalization also matters.

## 5. What the pasted performance establishes

| Held-out fold 1 | Epoch 1 | Epoch 8 |
|---|---:|---:|
| VA MSE mean | 0.02026 | 0.01349 |
| Valence Pearson | 0.6703 | 0.8042 |
| Arousal Pearson | 0.4754 | 0.6388 |

The model is learning; this does not establish an advantage over Raw TRT.
Without epochs 2–7/9–10 and the actual best checkpoint, it is also premature to
label epoch 8 as the final/best result or diagnose an overfitting trajectory.
Historical baseline numbers cannot isolate this new head's effect when code,
precision, data, and attention kernels are not matched.

The final exported model is intended to be the lowest generated-VA MSE
checkpoint, since the shared TrainingArguments set `load_best_model_at_end=True`
and `metric_for_best_model="mse_mean"`. The trainer state and final predictions
are needed to verify the completed server run.

## 6. Next actions in order

1. **Analyze completed artifacts first:** both folds, best epoch, learning
   curves, per-dataset and word/sentence errors, residual bias, predicted
   standard deviation and correlation. These need the results-only ZIP.
2. **Use the same saved model to isolate evaluation noise:** 16/64/256 samples
   with several predeclared seeds, same precision/data, same T=100. This needs
   retained server weights; the results-only ZIP cannot regenerate samples.
3. **Measure actual conditional use:** retain prefix and pooled representations
   while changing only the extra direct TRT memory; separately ablate the full
   conditioning to diagnose the denoising shortcut. Use fixed timestep/noise
   probes and report per-noise-bin epsilon and reconstructed-VA errors.
   Zeroing TRT is an out-of-distribution intervention, and inference ablation
   does not replace a retrained matched control.
4. **If poor mean prediction remains:** test a direct-MSE VA head together
   with a separately identifiable diffusion residual objective. Start from a
   matched Raw model, retain an explicit deterministic VA prediction path, and
   compare the residual contribution. This changes the method; superiority
   is a hypothesis, not an automatic repair. An auxiliary VA loss can also
   change the encoder's probabilistic objective and must be documented.
5. **Only then tune T/epochs:** the existing T flag changes both training and
   generation schedule. A 100→250 comparison requires a separate training
   condition; do not reinterpret an old checkpoint by changing T.

No change to the active model's loss, sampler, or training defaults was made.
This review adds only the evidence report, probes and independent tests.

## Reproduction

With the existing project dependencies installed, from the repository root:

```bash
python diagnostics/ddpm_review_20261021/probes.py

python -m pytest \
  diffusion_code/tests/test_diffusion.py \
  diagnostics/ddpm_review_20261021/test_review.py \
  va_model_code/tests/test_alignment_training_integration.py \
  va_model_code/tests/test_model_loss_evaluation.py \
  va_model_code/tests/test_train_model_contract.py -q
```

Result: **149 passed**. `probes.json` records all numerical outputs, not only
selected successful cases. These CPU checks do not test real pretrained
weights on the server's CUDA/FLA kernels.

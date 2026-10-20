# Redistribution implementation audit

Date: 2026-09-22. Checkout inspected: `670cf0c`. This audit changes no production
training code and does not rerun the trained GPU models.

## Conclusion

**A real upstream alignment bug is confirmed.** The Gaussian kernel itself passed
independent arithmetic and gradient checks, but its input TRT and source/target
mask can already be wrong. Therefore the earlier kernel-only audit was insufficient
to clear the full mechanism. Fix alignment before drawing another conclusion about
left-versus-right redistribution. This does not establish that the bug caused the
observed sigma direction or quantify its effect on VA performance.

## 1. Confirmed wrong-occurrence alignment

In [alignment.py:95](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/alignment.py:95),
the aligner searches all remaining token positions for a matching string. After a
match it sets `cursor = end`. The ET segmenter splits punctuation into separate
pieces; Qwen can merge punctuation, contractions, and abbreviations into larger
tokens. A segment that cannot match its own occurrence can match a later one,
advancing the cursor over unrelated words.

Reproduction using the pinned Qwen tokenizer:

```text
Text:        I love this!!! I hate that!
Qwen tokens: I | Ġlove | Ġthis | !!! | ĠI | Ġhate | Ġthat | !
ET segments: I | love | this | ! | ! | ! | I | hate | that | !
```

The first `!` segment, at character 11, is assigned to the last `!` token, at
character 26. `I`, `hate`, `that`, and the actual last `!` segment then fail to map.
The gaze mask becomes `[1, 1, 1, 0, 0, 0, 0, 1]`.

With distinct synthetic ET features numbered 1 through 10, the actual
[provider mapping](/Users/wansookim/Documents/decoder_based_va_prediction/va_model_code/decoder_va/gaze.py:459)
places value **4**, rather than value **10**, at the final `!`. This reproduces
feature misplacement through production mapping code, not only a tokenizer mismatch.
The synthetic features establish routing identity; they are not measured TRT values.

An actual Emobank experiment row (`index=222`) has the same failure:

> The numbers of children who deserve our services are rising... as quickly as our programs can grow to serve them.

The first dot of `...` maps to the final dot, and 11 later segments with available
exact token spans remain unmapped. Other actual examples involve `U.S.`, `P.S.`,
and an early apostrophe in `it's` stealing a later apostrophe.

## 2. Prevalence in the actual completed experiment texts

The audit uses all 14,352 texts from the result ZIP's `oof_predictions.tsv`, with
the same 200-token truncation, decoding, segmentation and current alignment code.
Character offsets from the pinned fast tokenizer provide an independent occurrence
check. Every audited token sequence round-tripped exactly; no rows were excluded.

| Dataset | Rows | Rows with a wrong-occurrence match | Percentage |
|---|---:|---:|---:|
| Emobank | 10,062 | 614 | 6.10% |
| EmoTales sentences | 1,395 | 22 | 1.58% |
| fb | 2,895 | 313 | 10.81% |
| Total | 14,352 | **949** | **6.61%** |

- Wrong-occurrence segment assignments: **1,947**.
- Unmapped ET segments: **40,651 / 270,714 (15.02%)**. This includes merged-token
  segmentation incompatibilities and is not all attributed to wrong-occurrence jumps.
- Unmapped segments despite an available exact token span: **14,880**.
- At least one unmapped segment: **6,678** sentences.
- No mapped gaze position: **54** sentences; exactly one position: **149**.
  A zero/one-position source set cannot provide a meaningful redistribution-width
  signal. Some very short inputs naturally have one position, so these counts are
  not all classified as alignment errors.

These are alignment-stage measurements. They do not require trained model weights
or assumptions about the magnitude of ET predictions. Actual ET truncation or
nonfinite predictions can remove additional valid features; full ET inference was
not rerun. The report does not infer downstream MSE changes from these counts.

The initial progress estimate of 2,511 affected rows was too broad: it counted
standalone leading-space tokens inside otherwise correct word spans. The final
occurrence check excludes whitespace-only pieces and gives 949 rows. Leading-space
anchors are recorded separately, not classified as wrong occurrences.

The local `va_model_code/data` folds have different hashes and row counts from the
server experiment. They were therefore not used for these prevalence numbers.
Current `alignment.py` and `gaze.py` are unchanged from commit `654f195`, which
preceded the recorded experiments. The result ZIP does not independently establish
the exact server source commit.

## 3. Consequences for redistribution

The kernel receives `gaze_mask` for both sources and destinations. A routing error
therefore changes:

1. Which token receives each raw TRT prediction.
2. Which locations can send and receive redistributed TRT.
3. Distances between valid positions, source-normalization denominators, and the
   sigma gradient computed through these values.
4. The compact gaze prefix length and hence subsequent text positions.

For the constructed example, the last valid-position gap grows to five Qwen tokens.
With sigma 1, the unnormalized weight across distance five is `exp(-12.5)`, about
`3.73e-6`; distance one gives about `0.6065`. Removing intermediate destinations
can therefore strongly change which mixtures are possible. This illustrates a
mechanism, not a measured explanation of the recorded right-sigma values.

Raw, fixed and learned modes share this alignment path. Sharing the bug does not
make the redistribution contrast immune: redistribution additionally depends on
the corrupted mask and distances. The existing results measure performance of
that pipeline; they do not isolate redistribution on correctly aligned gaze.

## 4. Gaussian arithmetic and gradients passed independent checks

The following checks used CPU PyTorch 2.12.1 and Transformers 5.16.1.

| Check | Result |
|---|---|
| Existing redistribution, model, Trainer, diagnostics, gaze and CLI tests | **248 passed, 1 skipped** |
| New independent float64 scalar-loop oracle | 48 dense/sparse cases, lengths 2/7/31/200 |
| Maximum output error against the oracle | `3.4811e-7` |
| Maximum log-sigma gradient error against float64 central differences | `2.2632e-6` |
| Maximum output error against the unmodified supplied kernel | `2.3842e-7` |
| Maximum gradient error against the supplied kernel | `1.4305e-6` |
| Native tiny Qwen3.5 hybrid decoder, FP32 and BF16 | Both sigma gradients finite and nonzero |
| Non-reentrant checkpointing on/off for that Qwen test | Identical predictions and sigma gradients within each precision |
| New regression cases for the alignment defect | **2 expected failures**, explicitly marked strict xfail |

The Gaussian checks include symmetric widths, both fixed directional pairs, both
recorded high-LR selected widths, and an extreme `.01/50` pair. Kernel checks also
verify total signed TRT conservation. The native Qwen test uses both linear- and
full-attention layers, random small weights and deterministic synthetic gaze. It
does not validate the trained 0.8B CUDA model or reproduce a server training run.

No error was found in the tested left/right branch, source-wise normalization,
padding exclusion, raw-cache separation, gradient path, separate sigma optimizer
group, frozen-width handling, or checkpoint restoration. Existing tests also
check valid zero-valued destinations and masked nonfinite values. These passing
checks do not establish universal correctness beyond their scope.

## 5. Additional implementation choices needing interpretation

The current kernel's sources and destinations are mapped first-subword positions,
but distance is measured in uncompressed Qwen token coordinates. The gaze prefix
is compacted afterward. Thus sigma is a width over a sparse, tokenizer-dependent
coordinate system, not a uniform word or character distance. About 10.33% of the
215,765 successive valid-position gaps in this audit exceed one; this statistic
also includes gaps produced by the alignment bug.

The supplied kernel and ours agree for equal inputs and masks. That equivalence
does not make the surrounding pipelines identical. Using a mapped-gaze mask for
destinations excludes unmapped lexical subwords as well as padding. This is an
explicit current design choice, not proof of equivalence to masking padding alone.

The small-width insensitivity previously diagnosed remains mathematically real:
if width is much smaller than the distance to any other valid position, both
redistribution and its sigma derivative become tiny. The alignment defect can
alter those distances, but its contribution to the observed stagnation has not
been isolated. LayerNorm/BF16 sensitivity and boundary effects likewise remain
possible contributors, not newly established root causes.

## 6. Required correction and validation before interpreting new runs

Use source character spans and target tokenizer offsets to preserve occurrence
identity. A repeated string later in a sentence must not substitute for the
current occurrence. Explicitly handle target tokens that cover multiple ET
segments: merely removing the search-ahead loop leaves contractions and merged
punctuation unsupported, while assigning multiple features to the same token with
the existing `output[first_subword] = feature` would silently overwrite values.

Then verify full-corpus coverage, occurrence identity, and the chosen many-to-one
aggregation rule before rerunning matched raw and redistribution controls. Record
the revised alignment policy in run metadata so corrected results are not mixed
with previous runs. Keep the Gaussian unchanged while isolating the alignment fix.

Production code is unchanged in this audit. No trained checkpoint was available;
the effect on final sigma or VA metrics is still unmeasured. The old test suite's
success should not have been taken as evidence that this real-tokenizer edge case
was covered.

## Reproduction and provenance

- [Audit code](audit.py)
- [Kernel and Qwen evidence](results.json)
- [Full alignment counts and examples](alignment_results.json)
- [Explicit failing regression cases](test_alignment_regression.py)
- Pinned tokenizer: `Qwen/Qwen3.5-0.8B-Base`, revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Downloaded `tokenizer.json` SHA-256:
  `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.
- Audited result ZIP SHA-256:
  `d6c8db31ae4956bbccedc1b94f264133cc25da48645442ffed5d1f17e589f4a8`.
- Supplied reference kernel SHA-256:
  `284db3253abf8f9d399bd314c394eff67109c0ef0a72c721582321d99044b714`.

```text
PYTHONPATH=/private/tmp/decoder-va-audit-20260922 python diagnostics/redistribution_audit_20260922/audit.py
PYTHONPATH=/private/tmp/decoder-va-audit-20260922 python diagnostics/redistribution_audit_20260922/audit.py --alignment-only --tokenizer-path /private/tmp/decoder-va-audit-tokenizer
PYTHONPATH=/private/tmp/decoder-va-audit-20260922 python -m pytest -q -rx diagnostics/redistribution_audit_20260922/test_alignment_regression.py
```

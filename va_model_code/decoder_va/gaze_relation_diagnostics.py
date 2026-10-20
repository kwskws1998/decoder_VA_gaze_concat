"""Paired, inference-only ablations of a saved TRT rotation model.

These controls measure dependence of one trained model on its relation branch.
They do not estimate the effect of training with rotation instead of Raw TRT.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from .evaluation import calculate_va_metrics
from .filters import FOLD_FILENAMES, SENTENCE_ONLY_DATASET_NAMES, read_fold
from .gaze_relation import TRTRelativeRotation, gaze_relation_contract, validate_gaze_relation_contract
from .gaze_relation_inputs import shuffle_relation_trt

ARMS = ("learned", "zero_alpha", "shuffled_trt")


def relation_modules(model):
    """Return only the explicit rotation modules, rejecting a disabled checkpoint."""

    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, TRTRelativeRotation)]
    if not modules:
        raise ValueError("Saved model has no TRT rotation modules to ablate.")
    return modules


@contextmanager
def relation_arm(model, arm: str, *, shuffle_seed: int = 42):
    """Temporarily change only alpha or relation TRT; restore exact state on error."""

    if arm not in ARMS:
        raise ValueError(f"Unknown diagnostic arm: {arm!r}.")
    original_contract = model.gaze_relation
    saved_contract = validate_gaze_relation_contract(original_contract)
    if saved_contract["method"] == "none":
        raise ValueError("Ablations require a trained TRT relation checkpoint.")
    if saved_contract.get("input_mode", "aligned") != "aligned":
        raise ValueError("These aligned-versus-shuffled diagnostics require an aligned checkpoint.")
    modules = relation_modules(model)
    snapshots = [(module.alpha, module.alpha.detach().clone()) for _, module in modules]
    modes = [(module, module.training) for module in model.modules()]
    has_config = hasattr(model, "config")
    config_has_relation = has_config and hasattr(model.config, "gaze_relation")
    original_config = getattr(model.config, "gaze_relation", None) if has_config else None
    try:
        model.eval()
        if arm == "zero_alpha":
            with torch.no_grad():
                for parameter, _ in snapshots:
                    parameter.zero_()
        if arm == "shuffled_trt":
            model.gaze_relation = gaze_relation_contract(
                saved_contract["method"], scale=saved_contract["scale"],
                input_mode="shuffled", shuffle_seed=shuffle_seed,
            )
        if has_config:
            model.config.gaze_relation = copy.deepcopy(model.gaze_relation)
        with torch.inference_mode():
            yield model
    finally:
        with torch.no_grad():
            for parameter, value in snapshots:
                parameter.copy_(value)
        model.gaze_relation = original_contract
        if config_has_relation:
            model.config.gaze_relation = original_config
        elif has_config and hasattr(model.config, "gaze_relation"):
            delattr(model.config, "gaze_relation")
        for module, mode in modes:
            module.training = mode


class PhaseAccumulator:
    """Count all causal anchor pairs and sample their actual phase magnitudes.

    A bounded random-priority reservoir samples pairs uniformly. Crossing the
    retained pairs with every layer/head coefficient gives phase quantiles;
    quantiles are explicitly approximate once the reservoir limit is exceeded.
    Counts are exact, include all pairs, exclude the always-zero diagonal, and
    are not weighted by attention probability or task loss.
    """

    def __init__(self, coefficients, *, sample_pairs: int = 100_000, seed: int = 0):
        values = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if not len(values) or not np.isfinite(values).all() or sample_pairs < 1:
            raise ValueError("Finite coefficients and a positive pair sample size are required.")
        self.coefficients = values
        self.sample_pairs = int(sample_pairs)
        self.rng = np.random.default_rng(seed)
        self._deltas = []
        self._keys = []
        self._buffer_size = 0
        self.pair_count = 0
        self.nonzero_delta_count = 0
        self.max_absolute_delta = 0.0

    def update(self, values):
        """Accumulate a single row's ordered mapped anchors, including readout if present."""

        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Valid relation TRT must be finite.")
        query, key = np.tril_indices(len(values), k=-1)
        delta = values[key] - values[query]
        self.pair_count += len(delta)
        self.nonzero_delta_count += int(np.count_nonzero(delta))
        if len(delta):
            self.max_absolute_delta = max(self.max_absolute_delta, float(np.abs(delta).max()))
            self._deltas.append(delta)
            self._keys.append(self.rng.random(len(delta)))
            self._buffer_size += len(delta)
        if self._buffer_size >= self.sample_pairs * 2:
            self._compact()

    def _compact(self):
        """Keep uniform random-priority pair samples without growing with corpus size."""

        if not self._deltas:
            return
        values, keys = np.concatenate(self._deltas), np.concatenate(self._keys)
        if len(values) > self.sample_pairs:
            selected = np.argpartition(keys, self.sample_pairs - 1)[:self.sample_pairs]
            values, keys = values[selected], keys[selected]
        self._deltas, self._keys = [values], [keys]
        self._buffer_size = len(values)

    def report(self):
        """Return exact eligibility counts and clearly labeled radian phase quantiles."""

        self._compact()
        deltas = self._deltas[0] if self._deltas else np.array([], dtype=np.float64)
        phases = np.abs(self.coefficients[:, None] * deltas[None, :]).reshape(-1)
        nonzero = phases[phases > 0]
        if not np.isfinite(phases).all():
            raise ValueError("Computed relation phases must be finite.")
        quantiles = {str(q): float(np.quantile(nonzero, q)) for q in (0, .25, .5, .75, .95, .99, 1)} if len(nonzero) else {}
        return {
            "eligible_causal_pairs_excluding_diagonal": self.pair_count,
            "layer_head_count": len(self.coefficients),
            "nonzero_phase_count": self.nonzero_delta_count * int(np.count_nonzero(self.coefficients)),
            "maximum_absolute_phase_radians": self.max_absolute_delta * float(np.abs(self.coefficients).max()),
            "sampled_pairs": len(deltas),
            "quantiles_exact": self.pair_count <= self.sample_pairs,
            "absolute_nonzero_phase_radian_quantiles": quantiles,
            "sampling": "uniform random-priority pair reservoir; every layer/head; seed 0",
            "weighting": "unweighted causal pairs and layer/heads; not attention probabilities",
        }


def describe_batch(model, batch, *, shuffle_seed: int, accumulators):
    """Audit actual provider output; shuffle relation values without modifying raw prefix."""

    with torch.inference_mode():
        raw, mask = model.gaze_provider.compute(batch["input_ids"], batch["attention_mask"])
        feature = tuple(model.gaze_provider.feature_indices).index(3)
        values = raw[..., feature]
        shuffled = shuffle_relation_trt(values, batch["input_ids"], batch["attention_mask"], mask, seed=shuffle_seed)
    eligible = mask.bool()
    if bool((eligible & ~batch["attention_mask"].bool()).any()):
        raise ValueError("Gaze anchors must be active text positions.")
    is_readout = model.gaze_relation["method"] == "trt-readout-rotation"
    rows = []
    for row in range(len(values)):
        valid = eligible[row]
        original = values[row, valid].detach().cpu().double().numpy()
        permuted = shuffled[row, valid].detach().cpu().double().numpy()
        count = len(original)
        length = int(batch["attention_mask"][row].sum())
        changed = int(np.count_nonzero(original != permuted))
        rows.append({
            "anchor_count": count,
            "anchor_group": "0" if count == 0 else "1" if count == 1 else "many",
            "last_original_token_anchored": bool(valid[length - 1]),
            "shuffled_anchors_changed": changed,
            "shuffled_any_changed": changed > 0,
        })
        if is_readout:
            original, permuted = np.append(original, 0.), np.append(permuted, 0.)
        accumulators["learned"].update(original)
        accumulators["zero_alpha"].update(original)
        accumulators["shuffled_trt"].update(permuted)
    return rows


def evaluate_paired_batches(model, loader, *, device, dtype=torch.float32, shuffle_seed=42):
    """Use identical ordered batches for all arms, restoring model state after each arm."""

    relation = validate_gaze_relation_contract(model.gaze_relation)
    coefficients = np.concatenate([module.alpha.detach().cpu().double().numpy() / module.scale
                                   for _, module in relation_modules(model)])
    accumulators = {arm: PhaseAccumulator(np.zeros_like(coefficients) if arm == "zero_alpha" else coefficients)
                    for arm in ARMS}
    predictions, descriptions = {arm: [] for arm in ARMS}, []
    for cpu_batch in loader:
        batch = {name: value.to(device) for name, value in cpu_batch.items()
                 if name in {"input_ids", "attention_mask", "token_type_ids"}}
        with relation_arm(model, "learned", shuffle_seed=shuffle_seed):
            descriptions.extend(describe_batch(model, batch, shuffle_seed=shuffle_seed, accumulators=accumulators))
        for arm in ARMS:
            with relation_arm(model, arm, shuffle_seed=shuffle_seed):
                context = (torch.autocast(device_type=device.type, dtype=dtype)
                           if device.type == "cuda" and dtype != torch.float32 else nullcontext())
                with context:
                    output = model(**batch).logits.detach().cpu().double().numpy()
            if output.shape != (len(batch["input_ids"]), 2) or not np.isfinite(output).all():
                raise ValueError(f"Invalid finite two-output predictions in {arm}.")
            if np.any((output < 0) | (output > 1)):
                raise ValueError(f"Predictions outside [0,1] in {arm}.")
            predictions[arm].append(output)
    if not descriptions:
        raise ValueError("Cannot evaluate an empty held-out fold.")
    return ({arm: np.concatenate(values) for arm, values in predictions.items()},
            pd.DataFrame(descriptions), {arm: accumulator.report() for arm, accumulator in accumulators.items()})


def validate_heldout_data(members, data_dir):
    """Verify exact source hashes and stable held-out identities before any model loads."""

    frames, hashes = {}, {}
    for number, filename in enumerate(FOLD_FILENAMES, start=1):
        path = Path(data_dir) / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        for member in members:
            expected = member.run_manifest.get("fold_sha256", {}).get(filename)
            if actual != expected:
                raise ValueError(f"Source fold SHA256 mismatch: {filename}.")
        frames[number], hashes[filename] = read_fold(path), actual
    selected = {}
    for member in members:
        frame = frames[member.held_out_fold]
        excluded = set(member.run_manifest.get("excluded_dataset_names", ()))
        frame = frame.loc[~frame["dataset_of_origin"].isin(excluded)].reset_index(drop=True)
        if len(frame) != member.run_manifest.get("evaluation_rows"):
            raise ValueError(f"Held-out row count mismatch: {member.name}.")
        persisted = pd.read_csv(member.model_dir.parent / "predictions.tsv", sep="\t", keep_default_na=False)
        for column in ("index", "text", "dataset_of_origin"):
            if frame[column].astype(str).tolist() != persisted[column].astype(str).tolist():
                raise ValueError(f"Held-out metadata order mismatch for {column}: {member.name}.")
        source_labels = frame[["valence", "arousal"]].to_numpy(dtype=np.float32).astype(np.float64)
        if not np.allclose(source_labels, persisted[["valence", "arousal"]].to_numpy(float), rtol=0, atol=1e-12):
            raise ValueError(f"Held-out label mismatch: {member.name}.")
        frame[["valence", "arousal"]] = source_labels
        selected[member.held_out_fold] = frame
    return selected, hashes


def paired_prediction_frame(frame, predictions, descriptions, *, held_out_fold):
    """Build one stable row per example with paired predictions and structural groups."""

    if len(frame) != len(descriptions) or any(len(predictions[arm]) != len(frame) for arm in ARMS):
        raise ValueError("Paired prediction row counts disagree.")
    result = frame[["index", "dataset_of_origin", "valence", "arousal"]].reset_index(drop=True).copy()
    result.insert(0, "held_out_fold", held_out_fold)
    result["text_sha256"] = frame["text"].map(lambda text: hashlib.sha256(str(text).encode()).hexdigest()).to_numpy()
    result["dataset_group"] = np.where(result["dataset_of_origin"].isin(SENTENCE_ONLY_DATASET_NAMES), "sentence", "word_or_other")
    for column in descriptions:
        result[column] = descriptions[column].to_numpy()
    for arm in ARMS:
        for dimension, name in enumerate(("valence", "arousal")):
            result[f"{arm}_pred_{name}"] = predictions[arm][:, dimension]
    return result


def paired_metrics(frame):
    """Compute pooled and stratified metrics, plus within-row MSE changes from learned."""

    groups = [("overall", "all", frame)]
    for column in ("held_out_fold", "dataset_of_origin", "dataset_group", "anchor_group", "last_original_token_anchored", "shuffled_any_changed"):
        groups.extend((column, str(key), group) for key, group in frame.groupby(column, sort=True))
    groups.extend(("fold_dataset", f"{fold}:{source}", group)
                  for (fold, source), group in frame.groupby(["held_out_fold", "dataset_of_origin"], sort=True))
    output = []
    for grouping, key, group in groups:
        labels = group[["valence", "arousal"]].to_numpy(float)
        reference = group[["learned_pred_valence", "learned_pred_arousal"]].to_numpy(float)
        reference_errors = np.square(reference - labels).mean(axis=1)
        for arm in ARMS:
            predictions = group[[f"{arm}_pred_valence", f"{arm}_pred_arousal"]].to_numpy(float)
            metrics = calculate_va_metrics(labels, predictions)
            delta = np.square(predictions - labels).mean(axis=1) - reference_errors
            output.append({"grouping": grouping, "group": key, "arm": arm, **metrics,
                           "mse_delta_vs_learned": float(delta.mean()),
                           "fraction_rows_lower_mse_vs_learned": float(np.mean(delta < 0)),
                           "mean_absolute_prediction_change_vs_learned": float(np.abs(predictions - reference).mean()),
                           "shuffle_changed_row_fraction": float(group["shuffled_any_changed"].mean()),
                           "shuffle_changed_anchor_fraction": float(group["shuffled_anchors_changed"].sum() / group["anchor_count"].sum()) if group["anchor_count"].sum() else 0.})
    return pd.DataFrame(output)


def write_diagnostic_results(output_dir, frame, manifest):
    """Publish a new complete report directory; never replace any existing output."""

    from .external_benchmarks import _atomic_publish_directory_no_replace

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Diagnostic output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        frame.to_csv(staging / "paired_predictions.tsv", sep="\t", index=False)
        paired_metrics(frame).to_csv(staging / "paired_metrics.tsv", sep="\t", index=False)
        (staging / "diagnostics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        _atomic_publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target

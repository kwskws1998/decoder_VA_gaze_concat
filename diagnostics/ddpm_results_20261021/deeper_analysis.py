"""Audit source-specific residuals, fold consistency and explanatory limits.

All calculations use already verified OOF predictions. Source-mean residual
decompositions are descriptive, not fitted correction models or new scores.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import ARCHIVES, read_archive

OUT = Path(__file__).resolve().parent
TARGETS = ["valence", "arousal"]
PREDICTIONS = ["pred_valence", "pred_arousal"]
SENTENCES = ["EmoTales sentences", "Emobank", "fb"]


def residuals(frame):
    """Return per-row errors in the original normalized VA units."""
    return frame[PREDICTIONS].to_numpy() - frame[TARGETS].to_numpy()


def bias_decomposition(frame):
    """Verify MSE equals source-weighted residual variance plus mean residual squared."""
    e = residuals(frame)
    source_bias_squared = np.zeros(2)
    within_source_variance = np.zeros(2)
    for _, subset in frame.groupby("dataset_of_origin"):
        error = residuals(subset)
        weight = len(subset) / len(frame)
        source_bias_squared += weight * error.mean(0) ** 2
        within_source_variance += weight * error.var(0)
    mse = np.mean(e ** 2, axis=0)
    np.testing.assert_allclose(mse, source_bias_squared + within_source_variance, atol=1e-14)
    return {d: {"mse": float(mse[i]), "global_bias_squared": float(e[:, i].mean() ** 2),
                "source_weighted_bias_squared": float(source_bias_squared[i]),
                "within_source_residual_variance": float(within_source_variance[i]),
                "source_bias_fraction": float(source_bias_squared[i] / mse[i])}
            for i, d in enumerate(TARGETS)}


def same_text_audit(frame):
    """Describe identical-text label disagreement without excluding any data."""
    grouped = frame.groupby(["held_out_fold", "text"], sort=False)
    size = grouped["index"].transform("size")
    sources = grouped["dataset_of_origin"].transform("nunique")
    label_disagreement = grouped[TARGETS].transform("max") - grouped[TARGETS].transform("min")
    nonidentical = label_disagreement.gt(1e-7).any(axis=1)
    predicted_range = grouped[PREDICTIONS].max() - grouped[PREDICTIONS].min()
    gold_variation = frame[TARGETS] - grouped[TARGETS].transform("mean")
    return {
        "repeated_rows": int(size.gt(1).sum()),
        "cross_source_repeated_rows": int(sources.gt(1).sum()),
        "cross_source_repeated_rows_with_different_label_pairs": int((sources.gt(1) & nonidentical).sum()),
        "per_fold_exact_text_within_group_gold_variance": gold_variation.pow(2).mean().to_dict(),
        "maximum_identical_text_prediction_range": predicted_range.max().to_dict(),
        "interpretation": "Empirical ambiguity among exact same texts within a fold; not a population error floor. GPU/batch-dependent numerical differences can affect saved predictions.",
    }


def main():
    """Compute independent identities and save source/fold diagnostics for review."""
    _, ddpm, _, _ = read_archive(ARCHIVES["ddpm"])
    _, rotation, _, _ = read_archive(ARCHIVES["readout_rotation"])
    _, raw, _, _ = read_archive(ARCHIVES["historical_raw"])
    pd.testing.assert_frame_equal(ddpm.drop(columns=PREDICTIONS), rotation.drop(columns=PREDICTIONS))
    total_excess_sse = (residuals(ddpm) ** 2 - residuals(rotation) ** 2).sum()
    source_rows, fold_rows, overlap_rows = [], [], []
    for name, group in ddpm.groupby("dataset_of_origin"):
        other = rotation.loc[group.index]
        de, re = residuals(group), residuals(other)
        for i, dimension in enumerate(TARGETS):
            delta_sse = (de[:, i] ** 2 - re[:, i] ** 2).sum()
            source_rows.append({
                "dataset": name, "dimension": dimension, "n": len(group),
                "gold_mean": float(group[dimension].mean()),
                "ddpm_mean": float(group[PREDICTIONS[i]].mean()),
                "rotation_mean": float(other[PREDICTIONS[i]].mean()),
                "ddpm_bias": float(de[:, i].mean()), "rotation_bias": float(re[:, i].mean()),
                "ddpm_mse": float(np.mean(de[:, i] ** 2)), "rotation_mse": float(np.mean(re[:, i] ** 2)),
                "excess_sse": float(delta_sse), "fraction_of_net_total_excess_sse": float(delta_sse / total_excess_sse),
                "bias_squared_difference": float(de[:, i].mean() ** 2 - re[:, i].mean() ** 2),
                "centered_residual_variance_difference": float(de[:, i].var() - re[:, i].var()),
            })
    for fold, group in ddpm.groupby("held_out_fold"):
        for name, mask in [("sentence_sources", group.dataset_of_origin.isin(SENTENCES)),
                           ("word_sources", ~group.dataset_of_origin.isin(SENTENCES))]:
            sub = group.loc[mask]
            a = np.mean(residuals(sub) ** 2)
            b = np.mean(residuals(rotation.loc[sub.index]) ** 2)
            fold_rows.append({"fold": int(fold), "source_type": name, "n": len(sub),
                              "ddpm_mse": float(a), "rotation_mse": float(b), "change_percent": float(100 * (a / b - 1))})
            seen = sub.text.isin(set(ddpm.loc[ddpm.held_out_fold.ne(fold), "text"]))
            for overlap, selected in [("exact_text_seen_in_training", seen), ("exact_text_unseen", ~seen)]:
                part = sub.loc[selected]
                overlap_rows.append({"fold": int(fold), "source_type": name, "overlap": overlap,
                                     "n": len(part), "ddpm_mse": float(np.mean(residuals(part) ** 2)),
                                     "rotation_mse": float(np.mean(residuals(rotation.loc[part.index]) ** 2))})
    decomposition = {"ddpm": bias_decomposition(ddpm), "rotation": bias_decomposition(rotation)}
    excess_mean_mse = total_excess_sse / (2 * len(ddpm))
    bias_delta = np.mean([decomposition["ddpm"][d]["source_weighted_bias_squared"] -
                         decomposition["rotation"][d]["source_weighted_bias_squared"] for d in TARGETS])
    assert np.isclose(sum(row["excess_sse"] for row in source_rows), total_excess_sse)
    sampling_scenarios = {}
    for name, comparator in [("historical_raw", raw), ("readout_rotation", rotation)]:
        gap = np.mean(residuals(ddpm) ** 2) - np.mean(residuals(comparator) ** 2)
        sampling_scenarios[name] = {
            "observed_mse_gap": float(gap), "required_average_variance_if_expected_gap_were_all_mc": float(16 * gap),
            "rms_conditional_sd_for_that_scenario": float(np.sqrt(16 * gap)),
            "warning": "Hypothetical expectation over independent noise-bank draws, assuming the infinite-sample DDPM mean matches comparator risk. One observed bank does not identify this variance.",
        }
    summary = {
        "bias_decomposition": decomposition,
        "source_bias_difference_fraction_of_ddpm_rotation_mse_gap": float(bias_delta / excess_mean_mse),
        "same_text": same_text_audit(ddpm),
        "sampling_scenarios": sampling_scenarios,
        "fold_source_type": fold_rows,
        "source_contributions": source_rows,
    }
    (OUT / "deeper_analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(source_rows).to_csv(OUT / "source_residual_decomposition.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_source_type.tsv", sep="\t", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "exact_text_overlap_diagnostics.tsv", sep="\t", index=False)
    ddpm.loc[ddpm.text.eq("anxiety")].to_csv(OUT / "same_input_example.tsv", sep="\t", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "source_contributions"}, indent=2))
    print(pd.DataFrame(source_rows).groupby("dataset")["fraction_of_net_total_excess_sse"].sum())


if __name__ == "__main__":
    main()

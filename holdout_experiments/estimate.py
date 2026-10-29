"""Planning bounds from the completed RTX4090 experiment, not a benchmark claim."""

import math

from .config import conditions_from_lock, training_arms


def estimate_runtime(lock):
    """Scale actual update and evaluation costs, with explicit padding/operator uncertainty."""

    rows = lock["partitions"]
    steps = lock["total_updates_per_fit"]
    epochs = lock["settings"]["epochs"]
    validation_rows = rows["val"]["rows"]
    baseline_train_seconds_per_update = (6014.5223 - 986.8879) / 16820
    companion_seconds_per_update = 4967.136422556825 / 16820
    validation_seconds_per_row = 209.55322674801573 / 26898
    validation = validation_rows * epochs * validation_seconds_per_row
    checkpoint_overhead = epochs * 12.0
    baseline = steps * baseline_train_seconds_per_update + validation + checkpoint_overhead
    companion = steps * companion_seconds_per_update + validation + checkpoint_overhead
    redistribution = baseline
    arms = training_arms(conditions_from_lock(lock))
    nominal = {"baseline": baseline, "companion": companion, "redistribution": redistribution}
    evaluation = rows["test"]["rows"] * validation_seconds_per_row * len(arms) + 120
    lower = (sum(nominal[arm] * (1.05 if arm == "redistribution" else 0.95) for arm in arms) + evaluation) / 3600
    upper = (sum(nominal[arm] * (1.50 if arm == "redistribution" else 1.30) for arm in arms) + evaluation) / 3600
    settings = lock["settings"]
    calibrated = all(settings.get(name, expected) == expected for name, expected in {
        "batch_size": 16, "eval_batch_size": 16, "max_length": 200,
        "model_id": "Qwen/Qwen3.5-0.8B-Base",
        "model_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "et_model_id": "skboy/et_prediction_2",
        "et_revision": "5785e77309d9fce8b88e908a9db100c1a0a63456",
    }.items())
    return {
        "gpu": "one RTX4090, sequential fits", "training_runs": len(arms), "prediction_mixture_fits": 0,
        "rows": {name: value["rows"] for name, value in rows.items()},
        "actual_updates_per_fit": steps, "epochs": epochs,
        "calibration_matches_compute_settings": calibrated,
        "estimated_hours_without_extra_buffer": [round(lower, 2), round(upper, 2)] if calibrated else None,
        "planning_hours_with_25pct_buffer": math.ceil(upper * 1.25) if calibrated else None,
        "per_arm_nominal_hours": {arm: round(nominal[arm] / 3600, 2) for arm in arms} if calibrated else None,
        "measured_basis": {
            "baseline_26893_train_26898_eval_seconds": 6210.140790019184,
            "companion_16820_updates_total_seconds": 5227.3651582710445,
            "companion_fit_validation_audits_seconds": 4967.136422556825,
            "source": "completed 2026-10-28 RTX4090 reverse-direction receipts",
        },
        "limits": [
            "No real-data GPU benchmark has been run for this new protocol.",
            "Uniform batches change padding versus the historical grouped baseline.",
            *(["Redistribution adds an unbenchmarked O(batch * text_length**2) FP32 kernel."]
              if "redistribution" in arms else []),
            "Downloads, installation, queueing and machine interruptions are excluded.",
            "10 epochs are fixed; validation results do not shorten or increase the budget.",
            "Changing batch size, sequence length or pretrained artifacts invalidates this calibration; no ETA is asserted then.",
        ],
    }

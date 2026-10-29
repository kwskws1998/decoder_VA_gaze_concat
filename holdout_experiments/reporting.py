"""Seal validation selections before one final test evaluation of the declared outputs."""

import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from trt_pipeline.core.evaluation import calculate_va_metrics
from trt_pipeline.data import contract_sha256
from trt_pipeline.mixing import mix_predictions

from .config import CONDITIONS, EXPOSURE_NOTE, METRICS, conditions_from_lock, training_arms
from .engine import data_loader, finite_metrics, predict, runtime_check, seed_everything
from .model import load_saved
from .protocol import digest, load_lock, read_partition, utc_now, write_json
from .suite import condition_event


def completion_receipts(run, lock):
    """Require all fixed budgets, the first validation minimum and strict reload evidence."""

    receipts = {}
    for arm in training_arms(conditions_from_lock(lock)):
        folder = Path(run) / arm
        receipt = json.loads((folder / "train_complete.json").read_text())
        history = json.loads((folder / "validation_history.json").read_text())
        reload_proof = json.loads((folder / "strict_reload.json").read_text())
        model = json.loads((folder / "model/model_manifest.json").read_text())
        if (receipt.get("status") != "completed" or receipt.get("arm") != arm
                or receipt.get("protocol_sha256") != lock["protocol_sha256"]
                or receipt.get("actual_optimizer_updates") != lock["total_updates_per_fit"]
                or receipt.get("epochs") != lock["settings"]["epochs"]
                or receipt.get("test_evaluations_during_training") != 0
                or receipt.get("selection_partition") != "val"
                or len(history) != lock["settings"]["epochs"]):
            raise ValueError(f"Incomplete fit or incompatible selection: {arm}")
        for epoch, event in enumerate(history, 1):
            if (event.get("partition") != "val" or event.get("arm") != arm
                    or event.get("epoch") != epoch
                    or event.get("actual_optimizer_updates") != epoch * lock["updates_per_epoch"]
                    or not np.isfinite(event["metrics"]["mse_mean"])):
                raise ValueError(f"Invalid validation trajectory for {arm}")
        best = min(history, key=lambda event: event["metrics"]["mse_mean"])
        weights = digest(folder / "model/model.safetensors")
        if (receipt.get("selected_epoch") != best["epoch"]
                or receipt.get("selected_step") != best["actual_optimizer_updates"]
                or receipt.get("selected_validation_mse_mean") != best["metrics"]["mse_mean"]
                or receipt.get("selected_weights_sha256") != weights
                or model.get("weights_sha256") != weights
                or model.get("settings") != lock["settings"]
                or model.get("protocol_sha256") != lock["protocol_sha256"]
                or model.get("arm") != arm
                or reload_proof.get("status") != "passed"
                or reload_proof.get("probe_partition") != "val"
                or reload_proof.get("weights_sha256") != weights
                or receipt.get("validation_history_sha256") != digest(folder / "validation_history.json")
                or receipt.get("strict_reload_sha256") != digest(folder / "strict_reload.json")):
            raise ValueError(f"Selected model or proof drift: {arm}")
        for name in ("optimizer_audit.json", "final_optimizer_audit.json"):
            audit = json.loads((folder / name).read_text())
            if audit.get("status") != "passed" or not audit.get("parameters"):
                raise ValueError(f"Missing live optimizer proof for {arm}")
        receipts[arm] = {"train_complete_sha256": digest(folder / "train_complete.json"),
                         "weights_sha256": weights, "selected_epoch": best["epoch"]}
    return receipts


def seal_selection(run):
    """Freeze all declared validation-selected weights and alpha before opening test rows."""

    run = Path(run)
    lock, settings = load_lock(run)
    conditions = conditions_from_lock(lock)
    receipts = completion_receipts(run, lock)
    path = run / "selection_lock.json"
    if path.exists():
        saved = json.loads(path.read_text())
        body = {key: value for key, value in saved.items() if key != "selection_sha256"}
        if (saved.get("protocol_sha256") != lock["protocol_sha256"] or saved.get("models") != receipts
                or saved.get("prediction_alpha") != settings.prediction_alpha
                or saved.get("conditions", list(CONDITIONS)) != list(conditions)
                or contract_sha256(body) != saved.get("selection_sha256")):
            raise ValueError("Sealed selection has changed.")
        return saved
    if (run / "evaluation_started.json").exists():
        raise RuntimeError("Test evaluation was started without a valid selection lock.")
    selection = {"utc": utc_now(), "protocol_sha256": lock["protocol_sha256"],
                 "models": receipts, "prediction_alpha": settings.prediction_alpha,
                 "conditions": list(conditions)}
    selection["selection_sha256"] = contract_sha256(selection)
    write_json(path, selection, exclusive=True)
    return selection


def prediction_table(path, frame, predictions, *, protocol_sha256, selection_sha256):
    """Preserve exact ordered identities, sources, texts, gold and predictions without pickle."""

    if predictions.shape != (len(frame), 2) or not np.isfinite(predictions).all():
        raise ValueError("Predictions do not match the test frame.")
    with Path(path).open("xb") as stream:
        np.savez_compressed(
            stream, row_ids=frame["index"].to_numpy(dtype=str), text=frame["text"].to_numpy(dtype=str),
            source=frame["dataset_of_origin"].to_numpy(dtype=str),
            labels=frame[["valence", "arousal"]].to_numpy(dtype=float), predictions=predictions,
            protocol_sha256=np.array(protocol_sha256), selection_sha256=np.array(selection_sha256))


def prepare_prediction_mixture(run):
    """Declare the fixed mixture from two selected weights without fitting or test access."""

    run = Path(run).resolve()
    lock, settings = load_lock(run)
    if "prediction_mixture" not in conditions_from_lock(lock):
        raise ValueError("Prediction mixture was not declared in this run.")
    dependencies = {}
    for arm in ("baseline", "companion"):
        path = run / arm
        receipt = json.loads((path / "train_complete.json").read_text())
        weights = digest(path / "model/model.safetensors")
        if (receipt.get("status") != "completed" or receipt.get("arm") != arm
                or receipt.get("protocol_sha256") != lock["protocol_sha256"]
                or receipt.get("actual_optimizer_updates") != lock["total_updates_per_fit"]
                or receipt.get("test_evaluations_during_training") != 0
                or receipt.get("selection_partition") != "val"
                or receipt.get("selected_weights_sha256") != weights):
            raise ValueError(f"Prediction mixture requires a completed validation-selected {arm}.")
        dependencies[arm] = weights
    expected = {"schema": 1, "condition": "prediction_mixture", "status": "prepared",
                "protocol_sha256": lock["protocol_sha256"], "requires_training": False,
                "actual_optimizer_updates": 0, "test_evaluated_at_preparation": False,
                "prediction_alpha": settings.prediction_alpha, "dependency_weights_sha256": dependencies,
                "dependency_model_paths": {name: f"{name}/model" for name in ("baseline", "companion")},
                "formula": "(1-alpha)*baseline + alpha*companion; same alpha for both VA axes"}
    folder = run / "prediction_mixture"
    path = folder / "preparation.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if {key: value for key, value in saved.items() if key != "utc"} != expected:
            raise ValueError("Prepared mixture no longer matches its fixed coefficient and selected dependencies.")
        return saved
    if (run / "evaluation_started.json").exists():
        raise RuntimeError("Mixture preparation must precede test evaluation.")
    folder.mkdir(exist_ok=False)
    result = {**expected, "utc": utc_now()}
    write_json(path, result, exclusive=True)
    return result


def evaluate(run):
    """Consume test once after all fits; form the fixed mixture from those same cached arrays."""

    run = Path(run).resolve()
    lock, settings = load_lock(run)
    conditions = conditions_from_lock(lock)
    arms = training_arms(conditions)
    selection = seal_selection(run)
    mixture_plan = prepare_prediction_mixture(run)
    if (run / "evaluation_started.json").exists():
        raise FileExistsError("Test evaluation already started; no automatic repeated test pass.")
    runtime = runtime_check()
    write_json(run / "evaluation_started.json", {
        "utc": utc_now(), "selection_sha256": selection["selection_sha256"]}, exclusive=True)
    started = time.perf_counter()
    output = run / "test_results"
    output.mkdir(exist_ok=False)
    current = None
    try:
        frame = read_partition(run / "splits/test.tsv")
        gold = frame[["valence", "arousal"]].to_numpy(dtype=float)
        predictions, metrics, timings, condition_results = {}, {}, {}, {}
        for condition in conditions:
            current = condition
            condition_event(run, condition, "test_evaluation", "running")
            condition_started = time.perf_counter()
            if condition == "prediction_mixture":
                predictions[condition] = mix_predictions(
                    predictions["baseline"], predictions["companion"], settings.prediction_alpha)
            else:
                seed_everything(settings.seed)
                model, tokenizer, manifest = load_saved(run / condition / "model", settings,
                                                       lock["protocol_sha256"], condition)
                if manifest["weights_sha256"] != selection["models"][condition]["weights_sha256"]:
                    raise ValueError("Test model differs from the sealed selection.")
                model.to("cuda:0")
                predictions[condition] = predict(model, data_loader(frame, tokenizer, settings), torch.device("cuda:0"))
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
            metrics[condition] = finite_metrics(calculate_va_metrics(gold, predictions[condition]))
            prediction_table(output / f"{condition}.npz", frame, predictions[condition],
                             protocol_sha256=lock["protocol_sha256"],
                             selection_sha256=selection["selection_sha256"])
            timings[condition] = time.perf_counter() - condition_started
            receipt = {"status": "completed", "utc": utc_now(), "condition": condition,
                       "partition": "test", "requires_training": condition != "prediction_mixture",
                       "actual_optimizer_updates": 0 if condition == "prediction_mixture" else lock["total_updates_per_fit"],
                       "protocol_sha256": lock["protocol_sha256"], "selection_sha256": selection["selection_sha256"],
                       "metrics": metrics[condition], "test_seconds": timings[condition],
                       "predictions_sha256": digest(output / f"{condition}.npz")}
            condition_results[condition] = receipt
            write_json(run / condition / "condition_complete.json", receipt, exclusive=True)
            condition_event(run, condition, "test_evaluation", "completed", metrics=metrics[condition])
        deltas = {arm: {key: (None if metrics[arm][key] is None or metrics["baseline"][key] is None
                             else metrics[arm][key] - metrics["baseline"][key]) for key in METRICS}
                  for arm in conditions}
        training = {arm: json.loads((run / arm / "train_complete.json").read_text()) for arm in arms}
        report = {
            "status": "completed", "utc": utc_now(), "partition": "test", "n_test": len(frame),
            "conditions": list(conditions), "condition_count": len(conditions), "fit_count": len(arms),
            "condition_results": condition_results,
            "mixture_preparation": mixture_plan,
            "protocol_sha256": lock["protocol_sha256"], "selection": selection,
            "prediction_alpha_companion_weight": settings.prediction_alpha,
            "formula": "(1-alpha)*baseline + alpha*companion; same alpha for both VA axes",
            "metrics": metrics, "delta_from_baseline": deltas,
            "metric_direction": "Pearson higher is better; RMSE and MAE lower are better",
            "training": training, "test_wall_seconds_including_reload": timings,
            "evaluation_total_seconds": time.perf_counter() - started, "environment": runtime,
            "historical_exposure": EXPOSURE_NOTE,
        }
        write_json(output / "report.json", report, exclusive=True)
        metric_names = tuple(metrics[conditions[0]])
        lines = ["arm\t" + "\t".join(metric_names)]
        for arm in conditions:
            lines.append(arm + "\t" + "\t".join(str(metrics[arm][key]) for key in metric_names))
        (output / "metrics.tsv").write_text("\n".join(lines) + "\n")
        write_json(run / "complete.json", {"status": "completed", "utc": utc_now(),
                   "report_sha256": digest(output / "report.json"),
                   "prediction_sha256": {path.name: digest(path) for path in sorted(output.glob("*.npz"))},
                   "condition_receipt_sha256": {name: digest(run / name / "condition_complete.json") for name in conditions},
                   "mixture_preparation_sha256": digest(run / "prediction_mixture/preparation.json"),
                   "selection_sha256": selection["selection_sha256"]}, exclusive=True)
    except BaseException as error:
        if current is not None:
            condition_event(run, current, "test_evaluation", "failed", error=str(error))
        write_json(run / "evaluation_failure.json", {"utc": utc_now(),
                   "error": type(error).__name__, "message": str(error)})
        raise
    return report


def verify_results(run):
    """Recompute all final scores and mixtures on CPU from exact test identities and gold."""

    run = Path(run).resolve()
    lock, settings = load_lock(run)
    conditions = conditions_from_lock(lock)
    arms = training_arms(conditions)
    complete = json.loads((run / "complete.json").read_text())
    selection = seal_selection(run)
    output = run / "test_results"
    report = json.loads((output / "report.json").read_text())
    if (complete.get("status") != "completed"
            or digest(output / "report.json") != complete.get("report_sha256")
            or complete.get("selection_sha256") != selection["selection_sha256"]
            or report.get("selection") != selection
            or report.get("conditions") != list(conditions)
            or report.get("condition_count") != len(conditions) or report.get("fit_count") != len(arms)
            or set(report.get("condition_results", {})) != set(conditions)
            or set(report.get("training", {})) != set(arms)
            or digest(run / "prediction_mixture/preparation.json") != complete.get("mixture_preparation_sha256")
            or report.get("mixture_preparation") != json.loads((run / "prediction_mixture/preparation.json").read_text())
            or report.get("protocol_sha256") != lock["protocol_sha256"]):
        raise ValueError("Final result receipt does not match the sealed run.")
    frame = read_partition(run / "splits/test.tsv")
    gold = frame[["valence", "arousal"]].to_numpy(float)
    arrays = {}
    for arm in conditions:
        receipt_path = run / arm / "condition_complete.json"
        if digest(receipt_path) != complete["condition_receipt_sha256"].get(arm):
            raise ValueError(f"Condition receipt SHA drift: {arm}")
        receipt = json.loads(receipt_path.read_text())
        if (receipt != report["condition_results"][arm] or receipt.get("status") != "completed"
                or receipt.get("condition") != arm or receipt.get("partition") != "test"
                or receipt.get("metrics") != report["metrics"][arm]
                or receipt.get("predictions_sha256") != complete["prediction_sha256"].get(f"{arm}.npz")
                or receipt.get("requires_training") is not (arm != "prediction_mixture")
                or receipt.get("actual_optimizer_updates") != (0 if arm == "prediction_mixture" else lock["total_updates_per_fit"])):
            raise ValueError(f"Invalid completed condition receipt: {arm}")
        path = output / f"{arm}.npz"
        if digest(path) != complete["prediction_sha256"].get(path.name):
            raise ValueError(f"Prediction archive SHA drift: {arm}")
        with np.load(path, allow_pickle=False) as saved:
            for key, column in (("row_ids", "index"), ("text", "text"), ("source", "dataset_of_origin")):
                if saved[key].dtype.kind != "U" or not np.array_equal(saved[key], frame[column].to_numpy(str)):
                    raise ValueError(f"Prediction identities changed for {arm}: {key}")
            if (not np.array_equal(saved["labels"], gold)
                    or saved["protocol_sha256"].item() != lock["protocol_sha256"]
                    or saved["selection_sha256"].item() != selection["selection_sha256"]):
                raise ValueError(f"Prediction ownership or gold changed: {arm}")
            arrays[arm] = saved["predictions"].copy()
        recomputed = finite_metrics(calculate_va_metrics(gold, arrays[arm]))
        for key, expected in recomputed.items():
            actual = report["metrics"][arm][key]
            if expected is None:
                if actual is not None:
                    raise ValueError("Undefined metric changed.")
            elif actual is None or not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Metric mismatch: {arm}/{key}")
    expected = mix_predictions(arrays["baseline"], arrays["companion"], settings.prediction_alpha)
    if not np.array_equal(expected, arrays["prediction_mixture"]):
        raise ValueError("Fixed prediction mixture differs from the declared formula.")
    result = {"status": "passed", "utc": utc_now(), "rows_per_arm": len(frame), "arms": len(conditions),
              "verified_prediction_rows": len(conditions) * len(frame), "protocol_sha256": lock["protocol_sha256"],
              "selection_sha256": selection["selection_sha256"],
              "scope": "CPU NPZ identity/gold/hash/metric/mixing recomputation; weights checked by SHA"}
    write_json(output / "verification.json", result)
    return result

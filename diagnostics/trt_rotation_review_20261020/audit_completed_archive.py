"""Verify the completed rotation archive and compare its row identities to source folds."""

import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "va_model_code"))
from decoder_va.filters import read_fold
from package_results import _validate_result_contract, _validate_training_parameters


def audit(source: Path) -> dict:
    """Validate both ZIP layers, hashes, result contracts, source labels and completion."""
    with zipfile.ZipFile(source) as outer:
        assert outer.testzip() is None
        selected = json.loads(outer.read("selected_runs.json"))
        names = [name for name in outer.namelist() if name.endswith(".zip")]
        runs = []
        for name in names:
            run_name = PurePosixPath(name).parts[0]
            log = outer.read(f"{run_name}/training.log").decode("utf-8")
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                assert inner.testzip() is None
                assert len(inner.namelist()) == len(set(inner.namelist()))
                snapshots = {}
                for member in inner.namelist():
                    parts = PurePosixPath(member).parts
                    assert parts[0] == run_name and ".." not in parts
                    assert not member.endswith((".safetensors", ".pt", ".bin"))
                    snapshots["/".join(parts[1:])] = inner.read(member)
                manifest = json.loads(snapshots["results_manifest.json"])
                assert set(manifest["files"]) == set(snapshots) - {"results_manifest.json"}
                for member, expected in manifest["files"].items():
                    data = snapshots[member]
                    assert len(data) == expected["size_bytes"]
                    assert hashlib.sha256(data).hexdigest() == expected["sha256"]
                parameters = json.loads(snapshots["training_parameters.json"])
                _validate_training_parameters(parameters)
                _validate_result_contract(Path(parameters["effective_output_dir"]), parameters, snapshots)
                assert f"Completed. OOF reports: {parameters['effective_output_dir']}" in log
                fold_results = []
                for fold in (1, 2):
                    data_path = ROOT / "va_model_code/data/paper7_seed42" / f"full_dataset_fold{fold}.csv"
                    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == parameters["fold_sha256"][data_path.name]
                    source_rows = read_fold(data_path)
                    source_rows = source_rows.loc[~source_rows.dataset_of_origin.isin(parameters["excluded_dataset_names"])].reset_index(drop=True)
                    predictions = pd.read_csv(io.BytesIO(snapshots[f"heldout_fold{fold}/predictions.tsv"]), sep="\t", keep_default_na=False)
                    assert len(source_rows) == len(predictions)
                    for column in ("index", "text", "dataset_of_origin"):
                        assert np.array_equal(source_rows[column].astype(str).to_numpy(), predictions[column].astype(str).to_numpy()), (fold, column)
                    label_error = float(np.max(np.abs(source_rows[["valence", "arousal"]].to_numpy(dtype=np.float32).astype(np.float64) - predictions[["valence", "arousal"]].to_numpy())))
                    assert label_error < 1e-12, label_error
                    state = json.loads(next(value for key, value in snapshots.items() if key.startswith(f"heldout_fold{fold}/checkpoints/") and key.endswith("trainer_state.json")))
                    relation = json.loads(snapshots[f"heldout_fold{fold}/gaze_relation_parameters.json"])
                    alpha = np.concatenate([np.asarray(layer["alpha"]) for layer in relation["layers"]])
                    fold_results.append({
                        "fold": fold,
                        "rows": len(predictions),
                        "source_labels_max_absolute_error": label_error,
                        "source_row_order_matches": True,
                        "selected_epoch": state["epoch"],
                        "selected_step": state["global_step"],
                        "metrics": json.loads(snapshots[f"heldout_fold{fold}/metrics.json"]),
                        "alpha_l2": float(np.linalg.norm(alpha)),
                        "alpha_absmax": float(np.max(np.abs(alpha))),
                        "alpha_min": float(alpha.min()),
                        "alpha_max": float(alpha.max()),
                        "alpha_nonzero_count": int(np.count_nonzero(alpha)),
                    })
                records = []
                for item in re.findall(r"\{[^{}\n]*\}", log):
                    try:
                        record = ast.literal_eval(item)
                    except (ValueError, SyntaxError):
                        continue
                    if "eval_mse_mean" in record or "train_runtime" in record:
                        records.append(record)
                complete = [r for r in records if "train_runtime" in r]
                assert len(complete) == 2 and all(float(r["epoch"]) == 10 for r in complete)
                runs.append({
                    "run_name": run_name,
                    "package_contract_validation": "passed",
                    "sha256_verified_files": len(manifest["files"]),
                    "parameters": parameters,
                    "oof_metrics": json.loads(snapshots["oof_metrics.json"]),
                    "by_dataset": pd.read_csv(io.BytesIO(snapshots["metrics_by_dataset.tsv"]), sep="\t").to_dict(orient="records"),
                    "folds": fold_results,
                    "epoch_records_console_rounded": records,
                })
    return {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "selected_runs": selected, "runs": runs}


if __name__ == "__main__":
    result = audit(Path(sys.argv[1]))
    output = Path(__file__).with_name("completed_results_audit.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for run in result["runs"]:
        print(run["run_name"], run["package_contract_validation"])
        print("OOF", run["oof_metrics"])
        print("FOLDS", run["folds"])
        print("EPOCHS", [{k: v for k, v in row.items() if k in ("epoch", "eval_mse_mean", "eval_mse_valence", "eval_mse_arousal", "train_runtime", "gaze_relation_alpha_norm")} for row in run["epoch_records_console_rounded"]])
    print(output)

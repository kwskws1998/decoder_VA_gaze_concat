"""Verify frozen-Qwen runs from data, source hashes, and complete request caches."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from expel_code.core import InsightBook, digest, isolate_train, text_key
from expel_code.experiment import ARMS, run_fold, save_json
from va_model_code.decoder_va.downloads import sha256_file
from va_model_code.decoder_va.filters import load_filtered_folds


def read(path):
    """Load an artifact without changing its contents."""

    return json.loads(path.read_text())


class ReplayBackend:
    """Require every generated response to match a saved full request exactly."""

    def __init__(self, run):
        self.identity = read(run / "backend.json")
        self.context_tokens = self.identity["context_tokens"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.identity["resolved_source"], local_files_only=True,
        )
        self.cache = run / "cache/responses"
        self.calls = 0
        self.used = set()

    def token_count(self, messages):
        """Recount the actual chat template using the saved local tokenizer."""

        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        ))

    def generate(self, messages, max_new_tokens=48):
        """Fail rather than invent a response when provenance does not match."""

        request = {
            "backend": self.identity, "messages": messages,
            "max_new_tokens": max_new_tokens,
        }
        key = digest(request)
        saved = read(self.cache / f"{key}.json")
        assert saved["request"] == request
        assert saved["input_tokens"] == self.token_count(messages)
        assert saved["input_tokens"] + max_new_tokens <= self.context_tokens
        self.calls += 1
        self.used.add(key)
        return saved["output"]


def audit(run):
    """Rebuild the experiment from cached model outputs and audit all held-out rows."""

    manifest = read(run / "manifest.json")
    config = manifest["configuration"]
    backend = read(run / "backend.json")
    summary = read(run / "summary.json")
    predictions = read(run / "predictions.json")
    for filename, expected in manifest["code_sha256"].items():
        assert sha256_file(ROOT / filename) == expected
        assert sha256_file(run / "source" / filename) == expected
    for filename, expected in manifest["data_sha256"].items():
        assert sha256_file(Path(config["data_dir"]) / filename) == expected
    assert backend["local_model_sha256"] == manifest["local_model_sha256"]
    for filename, expected in manifest["local_model_sha256"].items():
        assert sha256_file(Path(backend["resolved_source"]) / filename) == expected
        if filename.endswith(".safetensors"):
            metadata = Path(backend["resolved_source"]) / ".cache/huggingface/download" / f"{filename}.metadata"
            revision, object_hash = metadata.read_text().splitlines()[:2]
            assert revision == backend["revision"]
            assert object_hash == expected
    filtered = load_filtered_folds(
        Path(config["data_dir"]), no_iemocap=not config["include_iemocap"],
        sentence_only=config["sentence_only"],
    )
    folds = {1: filtered.fold1, 2: filtered.fold2}
    replay = ReplayBackend(run)
    fold_reports = {}
    rebuilt = []
    with tempfile.TemporaryDirectory(prefix="qwen-replay-") as temporary:
        for fold in config["held_out_folds"]:
            train, evaluation, exclusions = isolate_train(folds[3 - fold], folds[fold])
            assert exclusions == manifest["fold_audits"][str(fold)]
            original = run / f"fold{fold}"
            output = Path(temporary) / f"fold{fold}"
            rows, _ = run_fold(
                replay, train, evaluation, output, SimpleNamespace(**config),
            )
            rebuilt.extend({**row, "held_out_fold": fold} for row in rows)
            for filename in (
                "selection.json", "experiences.json", "memory.json",
                "insights.json", "predictions.json", "summary.json",
            ):
                assert (original / filename).read_bytes() == (output / filename).read_bytes()
            memory = read(original / "memory.json")
            selection = read(original / "selection.json")
            insights = read(original / "insights.json")
            assert set(row["row_id"] for row in memory) <= set(train["row_id"])
            assert not set(text_key(row["text"]) for row in memory) & set(evaluation["text_key"])
            assert not set(selection["experience_row_ids"]) & set(selection["evaluation_row_ids"])
            book = InsightBook(config["max_rules"])
            for update in insights["updates"]:
                assert book.snapshot() == update["before"]
                if update["error"] is None:
                    book.apply(json.dumps({"operations": update["operations"]}))
                assert book.snapshot() == update["after"]
            assert book.snapshot() == insights["rules"]
            for row in rows:
                if row["arm"] in ARMS:
                    assert row["row_id"] not in row["audit"]["example_ids"]
            eval_rows = [row for row in rows if row["arm"] == "baseline"]
            fold_reports[str(fold)] = {
                "evaluation_rows": len(eval_rows), "memory_rows": len(memory),
                "source_counts": dict(Counter(row["source"] for row in eval_rows)),
                "rules": insights["rules"],
                "invalid_rule_batches": sum(x["error"] is not None for x in insights["updates"]),
                "plain_rule_fallback_batches": sum(x["repair_output"] is not None for x in insights["updates"]),
            }
    assert json.loads(json.dumps(rebuilt)) == predictions
    for fold in (None, *config["held_out_folds"]):
        metrics = summary["pooled"] if fold is None else summary["folds"][str(fold)]["metrics"]
        selected = [row for row in predictions if fold is None or row["held_out_fold"] == fold]
        arm_ids = []
        for arm in (*ARMS, "train_mean", "retrieval_mean"):
            rows = [row for row in selected if row["arm"] == arm]
            arm_ids.append([(row["held_out_fold"], row["row_id"]) for row in rows])
            truth = np.array([[r["valence"], r["arousal"]] for r in rows])
            values = np.array([r["prediction"] for r in rows])
            assert np.isfinite(values).all()
            assert ((values >= 0) & (values <= 1)).all()
            squared = np.mean((truth - values) ** 2, axis=0)
            absolute = np.mean(np.abs(truth - values), axis=0)
            expected = metrics[arm]
            assert expected["n_examples"] == len(rows)
            for dimension, suffix in enumerate(("valence", "arousal")):
                assert abs(expected[f"mse_{suffix}"] - squared[dimension]) < 1e-12
                assert abs(expected[f"mae_{suffix}"] - absolute[dimension]) < 1e-12
            assert abs(expected["mse_mean"] - np.mean(squared)) < 1e-12
            assert abs(expected["mae_mean"] - np.mean(absolute)) < 1e-12
            assert expected["fallback_count"] == sum(row["used_fallback"] for row in rows)
        assert all(ids == arm_ids[0] for ids in arm_ids)
    cached = []
    for path in sorted((run / "cache/responses").glob("*.json")):
        entry = read(path)
        assert digest(entry["request"]) == path.stem
        assert entry["request"]["backend"] == backend
        cached.append(entry)
    assert len(replay.used) == len(cached)
    report = {
        "status": "passed",
        "audit_script_sha256": sha256_file(Path(__file__)),
        "data_source_snapshot_and_weight_hashes_verified": True,
        "weight_hashes_match_pinned_download_metadata": True,
        "complete_pipeline_replayed_from_full_request_cache": True,
        "outer_fold_memory_and_self_exclusion_verified": True,
        "rule_operations_replayed": True,
        "all_row_mse_and_mae_independently_recomputed": True,
        "matched_arm_row_identities_verified": True,
        "folds": fold_reports,
        "unique_generations": len(cached), "replay_calls": replay.calls,
        "generated_tokens": sum(x["output_tokens"] for x in cached),
        "model_generation_seconds": sum(x["elapsed_seconds"] for x in cached),
        "prediction_sha256": sha256_file(run / "predictions.json"),
        "evaluation_format_repairs": sum(
            len(x["audit"]["outputs"]) > 1 for x in predictions if x["arm"] in ARMS
        ),
    }
    validation_path = run / "validation.json"
    if validation_path.exists() and "resume" in read(validation_path):
        report["resume"] = read(validation_path)["resume"]
    save_json(validation_path, report)
    return manifest, backend, summary, report


def main():
    """Audit one run or additionally verify a matched pair of model-size experiments."""

    if sys.flags.optimize:
        raise RuntimeError("Run this audit without Python -O; its assertions must stay enabled.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    assert len(args.runs) in (1, 2)
    audited = [audit(path) for path in args.runs]
    if len(audited) == 2:
        first, second = audited
        allowed = {"run_name", "model_path", "model_size"}
        configs = [
            {key: value for key, value in item[0]["configuration"].items() if key not in allowed}
            for item in audited
        ]
        assert configs[0] == configs[1]
        for key in ("data_sha256", "code_sha256", "dependencies", "fold_audits"):
            assert first[0][key] == second[0][key]
        for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            assert first[0]["local_model_sha256"][filename] == second[0]["local_model_sha256"][filename]
        for key in (
            "device", "dtype", "torch", "transformers", "context_tokens",
            "threads", "generation", "generation_parameters",
        ):
            assert first[1][key] == second[1][key]
        for fold in (1, 2):
            assert read(args.runs[0] / f"fold{fold}/selection.json") == read(
                args.runs[1] / f"fold{fold}/selection.json"
            )
        comparison = {
            "status": "passed", "matched_configuration": configs[0],
            "changed_configuration_fields": sorted(allowed),
            "same_data_code_environment_and_selection": True,
            "experiences_memory_and_rules_regenerated_by_each_model": True,
            "runs": {
                path.name: {"backend": item[1], "pooled": item[2]["pooled"], "validation": item[3]}
                for path, item in zip(args.runs, audited)
            },
        }
        save_json(args.runs[-1] / "comparison.json", comparison)
    print(json.dumps({str(path): item[3] for path, item in zip(args.runs, audited)}, indent=2))


if __name__ == "__main__":
    main()

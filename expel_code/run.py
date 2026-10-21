"""Run a single-Qwen ExPeL-inspired VA experiment without a server or weight updates."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from va_model_code.decoder_va.downloads import sha256_file
from va_model_code.decoder_va.filters import load_filtered_folds
from va_model_code.prepare_english_data import main as prepare_data
from expel_code.backend import MODEL_SPECS, QwenBackend
from expel_code.core import isolate_train
from expel_code.experiment import run_fold, save_json, score_predictions


ROOT = Path(__file__).resolve().parents[1]


def parser():
    """Expose reproducible sampling, full evaluation, and data-download settings."""

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-name", required=True)
    result.add_argument("--data-dir", type=Path, default=ROOT / "va_model_code/data/paper7_seed42")
    result.add_argument("--download-data", action="store_true")
    result.add_argument("--model-path", default=None)
    result.add_argument("--model-size", choices=tuple(MODEL_SPECS), default="0.5b")
    result.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    result.add_argument("--held-out-folds", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    result.add_argument("--sentence-only", action="store_true")
    result.add_argument("--include-iemocap", action="store_true")
    result.add_argument("--experience-samples", type=int, default=32, help="Per fold; 0 uses all eligible training texts.")
    result.add_argument("--eval-samples", type=int, default=64, help="Per fold; 0 evaluates the full filtered outer fold.")
    result.add_argument("--top-k", type=int, default=3)
    result.add_argument("--success-tolerance", type=float, default=0.10)
    result.add_argument("--max-retries", type=int, default=1)
    result.add_argument("--max-rules", type=int, default=6)
    result.add_argument("--insight-batch-size", type=int, default=4)
    result.add_argument("--context-tokens", type=int, default=4096)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    """Validate inputs before loading weights and isolate every experiment output."""

    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", args.run_name):
        raise ValueError("run-name must be a simple directory name.")
    if len(set(args.held_out_folds)) != len(args.held_out_folds):
        raise ValueError("Do not repeat a held-out fold.")
    if min(args.experience_samples, args.eval_samples, args.max_retries) < 0:
        raise ValueError("Sample counts and retry count cannot be negative.")
    if min(args.top_k, args.max_rules, args.insight_batch_size, args.threads) <= 0:
        raise ValueError("k, rule limit, insight batch size, and threads must be positive.")
    if not 0 < args.success_tolerance <= 1:
        raise ValueError("success-tolerance must be in (0, 1].")
    if args.download_data:
        prepare_data([
            "--download-default", "--download-path", str(ROOT / "va_model_code/data/external/english_va_bundle.zip"),
            "--paper-protocol", "--seed", str(args.seed), "--output-dir", str(args.data_dir),
        ])
    filtered = load_filtered_folds(
        args.data_dir, no_iemocap=not args.include_iemocap, sentence_only=args.sentence_only,
    )
    folds = {1: filtered.fold1, 2: filtered.fold2}
    prepared, audits = {}, {}
    for heldout in args.held_out_folds:
        train, evaluation, audit = isolate_train(folds[3 - heldout], folds[heldout])
        prepared[heldout], audits[str(heldout)] = (train, evaluation), audit
    source_paths = sorted(Path(__file__).resolve().parent.glob("*.py"))
    source_paths += [
        ROOT / "va_model_code/decoder_va" / name
        for name in ("downloads.py", "filters.py", "preprocessing.py", "evaluation.py", "baseline_contracts.py")
    ]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in {"resume", "dry_run", "download_data"}}
    manifest = {
        "schema": 1, "method": "supervised ExPeL-inspired VA prompting",
        "configuration": config, "fold_audits": audits,
        "data_sha256": {
            name: sha256_file(args.data_dir / name)
            for name in ("full_dataset_fold1.csv", "full_dataset_fold2.csv", "english_dataset_manifest.json")
        },
        "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
        "dependencies": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "numpy", "pandas", "scikit-learn", "scipy",
        )},
        "evaluation": "sampled_outer_folds" if args.eval_samples else "full_outer_folds",
        "retrieval": "character TF-IDF cosine, training successful experiences only plus fixed gold seeds",
        "feedback": "training reference scores revealed on retry; assisted successes are not generalization",
        "weights_updated": False, "gaze_used": False, "external_llm_used": False,
    }
    if args.model_path:
        model_dir = Path(args.model_path)
        model_files = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.json"))
        model_files += [path for path in (model_dir / "merges.txt",) if path.is_file()]
        if not any(path.suffix == ".safetensors" for path in model_files):
            raise ValueError("model-path must contain local safetensors weights.")
        manifest["local_model_sha256"] = {p.name: sha256_file(p) for p in model_files}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    output = ROOT / "results" / args.run_name
    manifest_path = output / "manifest.json"
    if output.exists():
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError("Run directory exists; use --resume with exactly matching inputs.")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume refused: code, data, environment, or configuration changed.")
    else:
        output.mkdir(parents=True)
        save_json(manifest_path, manifest)
    for path in source_paths:
        saved_source = output / "source" / path.relative_to(ROOT)
        saved_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved_source)
    backend = QwenBackend(
        output / "cache", model_path=args.model_path, device=args.device,
        context_tokens=args.context_tokens, threads=args.threads,
        model_size=args.model_size,
    )
    if "local_model_sha256" in manifest:
        backend.identity["local_model_sha256"] = manifest["local_model_sha256"]
    backend_path = output / "backend.json"
    if backend_path.exists() and json.loads(backend_path.read_text()) != backend.identity:
        raise ValueError("Resume refused: model runtime identity changed.")
    save_json(backend_path, backend.identity)
    print(json.dumps(backend.identity, indent=2), flush=True)
    predictions, summaries = [], {}
    for fold, (train, evaluation) in prepared.items():
        print(f"Starting held-out fold {fold}", flush=True)
        rows, summary = run_fold(backend, train, evaluation, output / f"fold{fold}", args)
        predictions.extend({**row, "held_out_fold": fold} for row in rows)
        summaries[str(fold)] = summary
    summary = {
        "folds": summaries, "pooled": score_predictions(predictions),
        "evaluation": manifest["evaluation"],
        "all_folds_have_rules": all(x["rule_extraction_succeeded"] for x in summaries.values()),
        "generation_calls_this_invocation": backend.generated_calls,
        "cached_calls_this_invocation": backend.cached_calls,
    }
    save_json(output / "predictions.json", predictions)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary["pooled"], indent=2), flush=True)
    print(f"Saved experiment: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

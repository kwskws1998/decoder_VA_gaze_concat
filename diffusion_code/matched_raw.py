"""Run direct-MSE Raw TRT with the original DDPM data and encoder training settings."""

import argparse
import copy
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_code.train import baseline, source_hashes, write_json
from diffusion_code.diagnostic_data import verified_data, require_new_output
from va_model_code.decoder_va.paths import resolve_run_directory


MATCHED_FIELDS = (
    "model_id", "model_revision", "finetuning_mode", "precision", "et_model_id",
    "et_revision", "et_filename", "et_cache_size", "exclude_dataset", "no_iemocap",
    "sentence_only", "training_text_overlap_policy", "held_out_folds", "max_length",
    "train_batch_size", "eval_batch_size", "gradient_accumulation_steps", "epochs",
    "max_steps", "learning_rate", "weight_decay", "warmup_ratio", "logging_steps",
    "save_total_limit", "group_by_length", "gradient_checkpointing", "seed",
    "lora_rank", "lora_alpha", "lora_dropout", "attn_implementation", "use_cpu",
)


def build_parser():
    """Accept a source run and location, inheriting scientific settings from its manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", required=True)
    p.add_argument("--data-dir", default=str(ROOT / "va_model_code/data/paper7_seed42"))
    p.add_argument("--run-name")
    p.add_argument("--dry-run", action="store_true")
    return p


def matched_arguments(manifest, *, data_dir, run_name, dry_run):
    """Copy recorded common settings explicitly, never importing DDPM head options."""
    source = manifest["arguments"]
    if source["gaze_fusion"] != "prefix-concat":
        raise ValueError("This Raw TRT control requires a gaze-prefix DDPM source.")
    args = baseline._build_parser().parse_args(["qwen3.5-0.8b", "mse"])
    for name in MATCHED_FIELDS:
        setattr(args, name, copy.deepcopy(source[name]))
    args.gaze_fusion, args.gaze_features = "prefix-concat", ["TRT"]
    args.gaze_redistribution, args.gaze_relation, args.readout = "none", "none", "last-token"
    args.data_dir, args.run_name, args.dry_run = str(data_dir), run_name, dry_run
    args.resume_from_checkpoint = None
    args.report_to = []
    return args


def run(args):
    """Train the same encoder from its original initialization with a direct MSE head."""
    source = resolve_run_directory(args.reference_run)
    manifest, _ = verified_data(source, args.data_dir)
    name = args.run_name or f"raw_trt_matched_seed{manifest['arguments']['seed']}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output = require_new_output(resolve_run_directory(name), source)
    training = matched_arguments(manifest, data_dir=args.data_dir, run_name=name, dry_run=args.dry_run)
    print(f"Matched Raw TRT: seed={training.seed}, precision={training.precision}, "
          f"batch={training.train_batch_size}, accumulation={training.gradient_accumulation_steps}, "
          f"epochs={training.epochs}, sentence_only={training.sentence_only}, no_iemocap={training.no_iemocap}.")
    result = baseline.run(training)
    if not args.dry_run:
        if Path(result).resolve() != output:
            raise ValueError("Baseline output does not match the reserved run directory.")
        write_json(output / "comparison_reference.json", {
            "kind": "current_code_raw_trt_direct_mse", "reference_run": str(source),
            "reference_manifest": manifest, "matched_fields": list(MATCHED_FIELDS),
            "source_sha256": source_hashes(),
            "limits": "Matched data and common hyperparameters; current code/runtime, head and optimization objective differ.",
        })
    return result


if __name__ == "__main__":
    run(build_parser().parse_args())

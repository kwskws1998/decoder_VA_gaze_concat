"""Reload a completed DDPM checkpoint and evaluate a normalized VA TSV."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from va_model_code.decoder_va.dataset import TokenizedVADataset, VABatchCollator
from va_model_code.decoder_va.filters import read_fold
from va_model_code.decoder_va.evaluation import prediction_frame, write_oof_reports
from diffusion_code.model import load_diffusion_model
from diffusion_code.train import baseline


def main(argv=None):
    """Run label-free generation and write normal VA metrics without modifying training."""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, help="Defaults to the saved training truncation length.")
    p.add_argument("--precision", choices=baseline.PRECISION_MODES, default="bf16")
    p.add_argument("--use-cpu", action="store_true")
    args = p.parse_args(argv)
    contract = json.loads((args.model_dir / "diffusion_training_contract.json").read_text())
    if args.max_length is None:
        args.max_length = contract["arguments"]["max_length"]
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Evaluation output must be empty.")
    dtype, bf16, fp16 = baseline._runtime_precision(args.use_cpu, args.precision)
    model, tokenizer = load_diffusion_model(args.model_dir, dtype=dtype)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    model.to(device).eval()
    frame = read_fold(args.data_file)
    dataset = TokenizedVADataset(frame, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=VABatchCollator(tokenizer))
    predictions, labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels.append(batch.pop("labels").numpy())
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=dtype, enabled=bf16 or fp16):
                predictions.append(model(**batch).logits.float().cpu().numpy())
    result = prediction_frame(frame, np.concatenate(labels), np.concatenate(predictions), fold=0)
    write_oof_reports([result], args.output_dir, {
        "evaluation_kind": "saved_model_external_evaluation_not_oof",
        "model_dir": str(args.model_dir.resolve()), "diffusion": model.settings.to_dict(),
        "max_length": args.max_length, "precision": args.precision,
    })
    print(f"Saved-model evaluation: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

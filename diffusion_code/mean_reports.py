"""Compare generated VA against the unchanged mean predictor on exactly the same rows."""

import numpy as np
import pandas as pd
import torch

from diffusion_code.train import write_json
from va_model_code.decoder_va.evaluation import prediction_frame, metrics_from_prediction_frame, write_oof_reports


@torch.no_grad()
def evaluate_frozen_mean(trainer, dataset, metadata, fold, output, reference):
    """Re-evaluate the frozen mean without using targets and archive paired baseline predictions."""
    model = trainer.model
    model.eval()
    means = []
    for batch in trainer.get_eval_dataloader(dataset):
        inputs = trainer._prepare_inputs({key: batch[key] for key in ("input_ids", "attention_mask")})
        dtype = torch.bfloat16 if trainer.args.bf16 else torch.float16
        with torch.autocast(inputs["input_ids"].device.type, dtype=dtype,
                            enabled=trainer.args.bf16 or trainer.args.fp16):
            prediction = model.encoder(**inputs).logits
        means.append(prediction.detach().float().cpu().numpy())
    means = np.concatenate(means)
    frame = prediction_frame(metadata, metadata[["valence", "arousal"]].to_numpy(dtype=np.float32), means, fold=fold)
    directory = output / "frozen_mean"
    directory.mkdir()
    frame.to_csv(directory / "predictions.tsv", sep="\t", index=False)
    write_json(directory / "metrics.json", metrics_from_prediction_frame(frame))
    previous = pd.read_csv(reference.directory / f"heldout_fold{fold}/predictions.tsv", sep="\t")
    maximum = float(np.max(np.abs(means - previous[["pred_valence", "pred_arousal"]].to_numpy())))
    write_json(directory / "reproduction.json", {
        "max_abs_difference_from_source": maximum,
        "source_predictions_sha256": reference.records[fold]["predictions_sha256"],
        "comparison": "Use these re-evaluated frozen means for paired comparison with generated VA.",
    })
    print(f"Frozen Raw reference fold {fold}: MSE={metrics_from_prediction_frame(frame)['mse_mean']:.8f}; "
          f"max prediction change from archived Raw={maximum:.8g}.", flush=True)


def write_mean_comparison(output, folds):
    """Publish both full OOF predictions and paired per-fold/pooled comparison metrics."""
    references = [pd.read_csv(output / f"heldout_fold{fold}/frozen_mean/predictions.tsv", sep="\t", keep_default_na=False)
                  for fold in folds]
    write_oof_reports(references, output / "frozen_mean_oof", run_parameters={
        "kind": "reevaluated_frozen_raw_mean", "held_out_folds": folds,
    })
    rows = []
    for fold in [*folds, "oof"]:
        if fold == "oof":
            reference_path, generated_path = output / "frozen_mean_oof/oof_predictions.tsv", output / "oof_predictions.tsv"
        else:
            reference_path = output / f"heldout_fold{fold}/frozen_mean/predictions.tsv"
            generated_path = output / f"heldout_fold{fold}/predictions.tsv"
        reference = pd.read_csv(reference_path, sep="\t", keep_default_na=False)
        generated = pd.read_csv(generated_path, sep="\t", keep_default_na=False)
        identity = ["index", "held_out_fold", "text", "dataset_of_origin", "valence", "arousal"]
        pd.testing.assert_frame_equal(reference[identity], generated[identity], check_dtype=False, rtol=1e-7, atol=1e-9)
        for name, frame in [("frozen_raw_mean", reference), ("mean_guided_ddpm", generated)]:
            rows.append({"held_out_fold": fold, "condition": name, **metrics_from_prediction_frame(frame)})
    pd.DataFrame(rows).to_csv(output / "mean_comparison.tsv", sep="\t", index=False)

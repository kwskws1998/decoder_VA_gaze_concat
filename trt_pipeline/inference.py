"""Sequential saved-model inference for the two retained native precision profiles."""
import json
from pathlib import Path
import time
import numpy as np


def predict_saved(kind, model_dir, frame, fold, output):
    """Strictly reload and evaluate the requested profile without changing its native storage."""
    import torch
    from torch.utils.data import DataLoader
    from trt_pipeline.core.dataset import TokenizedVADataset, VABatchCollator
    from trt_pipeline.core.evaluation import calculate_va_metrics, prediction_frame
    from trt_pipeline.historical.recipe import verify_runtime
    verify_runtime()
    if kind == 'baseline':
        from trt_pipeline.historical.model import load_model
    elif kind == 'companion':
        from trt_pipeline.precision import load_saved_bf16_model as load_model
    else:
        raise ValueError('Unsupported model kind')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Saved evaluation requires a BF16-capable CUDA GPU')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output/'predictions.npz'
    if target.exists():
        raise FileExistsError(target)
    model, tokenizer = load_model(model_dir)
    model.to('cuda').eval()
    dataset = TokenizedVADataset(frame, tokenizer, max_length=200)
    loader = DataLoader(dataset, batch_size=16 if kind == 'baseline' else 8, shuffle=False, num_workers=0,
                        collate_fn=VABatchCollator(tokenizer, pad_to_multiple_of=8))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    gold, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            gold.append(batch.pop('labels').numpy())
            inputs = {key: value.to('cuda') for key, value in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                predictions.append(model(**inputs).logits.float().cpu().numpy())
    torch.cuda.synchronize()
    labels, values = np.concatenate(gold), np.concatenate(predictions)
    table = prediction_frame(frame, labels, values, fold=fold)
    table.to_csv(output/'predictions.tsv', sep='\t', index=False)
    with target.open('xb') as stream:
        np.savez_compressed(stream, labels=labels, predictions=values, row_ids=frame['index'].astype(str).to_numpy(dtype=str))
    record = {'metrics': calculate_va_metrics(labels, values), 'kind': kind, 'fold': fold,
              'model_dir': str(model_dir), 'inference_seconds': time.monotonic()-started,
              'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30}
    (output/'evaluation.json').write_text(json.dumps(record, indent=2)+'\n')
    return record

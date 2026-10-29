"""The historical ten-epoch baseline with passive save/optimizer auditing."""
import json
import math
from pathlib import Path
from trt_pipeline.config import DATA_DIR


def train(fold, data_dir=DATA_DIR, run_name=None):
    """Preserve seed42+fold-offset, native BF16 and held-out-development checkpoint selection."""
    from trt_pipeline import training
    from trt_pipeline.audit import IntegrityCallback
    from trt_pipeline.historical import recipe
    from trt_pipeline.historical.train_raw import build_parser
    from trt_pipeline.protocol import verify_data
    from trt_pipeline.core.filters import load_filtered_folds
    from trt_pipeline.inference import predict_saved
    from trt_pipeline.io_utils import digest, write_json
    verify_data(data_dir)
    name = run_name or f'baseline_fold{fold}'
    args = build_parser().parse_args(['qwen3.5-0.8b', 'mse', '--data-dir', str(data_dir),
        '--held-out-folds', str(fold), '--seed', '42', '--run-name', name, '--save-total-limit', '1', '--logging-steps', '50'])
    training._validate_args(args)
    recipe.validate_args(args)
    original = training.VARegressionTrainer

    class ObservedTrainer(original):
        """Add only the original passive native-state audits."""
        def __init__(self, *values, **options):
            super().__init__(*values, **options)
            self.add_callback(IntegrityCallback(fold))

    training.VARegressionTrainer = ObservedTrainer
    try:
        output = training.run(args, model_profile=recipe)
    finally:
        training.VARegressionTrainer = original
    location = output/f'heldout_fold{fold}'
    state = json.loads((location/'trainer_state.json').read_text())
    folds = load_filtered_folds(data_dir, no_iemocap=True, sentence_only=False, exclude_dataset=[])
    train_frame = folds.fold2 if fold == 1 else folds.fold1
    outer = (folds.fold1 if fold == 1 else folds.fold2).reset_index(drop=True)
    if state['global_step'] != 10*math.ceil(len(train_frame)/16):
        raise ValueError('Baseline optimizer budget changed')
    model = location/'final_model'
    selected = Path(state['best_model_checkpoint'])/'model.safetensors'
    if digest(model/'model.safetensors') != digest(selected):
        raise ValueError('Final baseline differs from the selected checkpoint')
    record = predict_saved('baseline', model, outer, fold, location)
    record.update(actual_updates=state['global_step'], weights_sha256=digest(model/'model.safetensors'),
                  best_checkpoint=state['best_model_checkpoint'], best_metric=state['best_metric'], actual_seed=42+fold-1)
    write_json(location/'complete.json', record)
    from trt_pipeline.retention import retain_checkpoints
    retain_checkpoints(location, model, record['weights_sha256'], 10, 'historical')
    return record

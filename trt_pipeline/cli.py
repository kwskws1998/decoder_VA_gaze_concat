"""Expose only baseline, companion and fixed prediction-mixing workflows."""
import argparse
import json
from pathlib import Path
from trt_pipeline.config import DATA_DIR, ROOT, CompanionConfig


def build_parser():
    """Keep retired experiments out of the supported command surface."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    baseline = sub.add_parser('baseline', help='Train the historical ten-epoch baseline')
    baseline.add_argument('--fold', type=int, choices=(1, 2), required=True)
    baseline.add_argument('--data-dir', type=Path, default=DATA_DIR)
    baseline.add_argument('--run-name')
    companion = sub.add_parser('companion', help='Train only the matched clean/noisy MSE model')
    companion.add_argument('--fold', type=int, choices=(1, 2), required=True)
    companion.add_argument('--data-dir', type=Path, default=DATA_DIR)
    companion.add_argument('--output', type=Path)
    companion.add_argument('--preflight', action='store_true', help='Only two training-only validation updates')
    mix = sub.add_parser('mix', help='Apply the frozen common prediction coefficient')
    mix.add_argument('--baseline', type=Path, required=True)
    mix.add_argument('--companion', type=Path, required=True)
    mix.add_argument('--output', type=Path, required=True)
    infer = sub.add_parser('predict', help='Evaluate a saved baseline or companion')
    infer.add_argument('--kind', choices=('baseline', 'companion'), required=True)
    infer.add_argument('--model-dir', type=Path, required=True)
    infer.add_argument('--fold', type=int, choices=(1, 2), required=True)
    infer.add_argument('--data-dir', type=Path, default=DATA_DIR)
    infer.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None):
    """Load heavy training dependencies only after parsing a supported command."""
    args = build_parser().parse_args(argv)
    if args.command == 'baseline':
        from trt_pipeline.baseline import train
        result = train(args.fold, args.data_dir, args.run_name)
    elif args.command == 'companion':
        from trt_pipeline.companion import train_encoder_stage, finish_companion
        from trt_pipeline.protocol import build_plan
        config = CompanionConfig(data_dir=args.data_dir, fold=args.fold)
        frames = build_plan(config)['frames']
        if args.preflight:
            import numpy as np

            def sample(role, count):
                """Use the same training-only length-stratified preflight subsets."""
                frame = frames[role]
                order = np.argsort(frame['text'].fillna('').astype(str).str.len().to_numpy(), kind='stable')
                return frame.iloc[order[np.linspace(0, len(order)-1, count).astype(int)]].reset_index(drop=True).copy()

            a, validation, outer = (sample(role, count) for role, count in [('A', 32), ('B_validation', 16), ('B_fit', 8)])
        else:
            a, validation, outer = (frames[role] for role in ['A', 'B_validation', f'outer_fold{args.fold}'])
        output = args.output or ROOT/'results'/f'companion_fold{args.fold}{"_preflight" if args.preflight else ""}'
        output = output.resolve()
        if not output.is_relative_to((ROOT/'results').resolve()):
            raise ValueError('Companion output must be below the repository results directory')
        result = train_encoder_stage(config, a, validation, outer, output_dir=output, condition='matched_mse', preflight=args.preflight)
        result = finish_companion(result, outer, output, args.preflight)
    elif args.command == 'mix':
        from trt_pipeline.mixing import mix_files
        result = mix_files(args.baseline, args.companion, args.output)
    else:
        from trt_pipeline.protocol import verify_data
        from trt_pipeline.core.filters import load_filtered_folds
        from trt_pipeline.inference import predict_saved
        verify_data(args.data_dir)
        folds = load_filtered_folds(args.data_dir, no_iemocap=True, sentence_only=False, exclude_dataset=[])
        frame = (folds.fold1 if args.fold == 1 else folds.fold2).reset_index(drop=True)
        result = predict_saved(args.kind, args.model_dir, frame, args.fold, args.output)
    print(json.dumps(result, indent=2))
    return result

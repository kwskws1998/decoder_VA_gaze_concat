"""Shared argument, prediction and alignment helpers for the companion."""
import math
import numpy as np

def score(labels, predictions):
    """Reuse the established VA metric definitions and make undefined values JSON safe."""

    from trt_pipeline.core.evaluation import calculate_va_metrics

    return {name: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for name, value in calculate_va_metrics(labels, predictions).items()}


def baseline_arguments(config, seed):
    """Build original Trainer options without mutating any active baseline source."""

    from trt_pipeline import training as baseline

    settings = config.baseline
    args = baseline._build_parser().parse_args([settings.model, settings.loss])
    for name, value in vars(settings).items():
        if hasattr(args, name):
            setattr(args, name, value)
    args.seed = int(seed)
    args.no_iemocap = config.no_iemocap
    args.sentence_only = config.sentence_only
    args.training_text_overlap_policy = config.training_text_overlap
    args.gaze_features = list(settings.gaze_features)
    baseline._validate_args(args)
    return args


def write_prediction_table(frame, labels, predictions, fold, path):
    """Preserve the baseline's checked row-to-label alignment in every result table."""

    from trt_pipeline.core.evaluation import prediction_frame

    output = prediction_frame(frame, labels, predictions, fold=fold)
    output.to_csv(path, sep="\t", index=False)
    return output


def validate_feature_rows(arrays, frame):
    """Reject reordered feature rows or labels before associating residuals with text."""

    expected_ids = frame["index"].astype(str).to_numpy()
    expected_labels = frame[["valence", "arousal"]].to_numpy(dtype=np.float32)
    if not np.array_equal(arrays["row_ids"], expected_ids):
        raise ValueError("Feature row identities differ from current data order.")
    actual_labels = np.asarray(arrays["labels"], dtype=np.float32)
    if actual_labels.shape != expected_labels.shape or not np.array_equal(actual_labels, expected_labels):
        raise ValueError("Feature labels differ from the current row metadata.")

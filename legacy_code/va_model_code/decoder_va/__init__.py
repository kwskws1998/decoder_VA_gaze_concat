"""Data and model building blocks for decoder-based VA regression."""

from .dataset import TokenizedVADataset, VABatchCollator
from .evaluation import calculate_va_metrics, write_oof_reports
from .external_benchmarks import (
    audit_finetuning_text_overlap,
    discover_completed_run,
    download_pinned_idest_english,
    download_pinned_semeval_subtask1_test,
    fixed_unweighted_ensemble,
    load_idest_english,
    load_msp_podcast_test,
    load_omg_emotion_test,
    load_semeval_2026_subtask1_test,
    semeval_subtask1_official_metrics,
)
from .filters import collect_exclude_patterns, load_filtered_folds
from .gaze import ET2GazeProvider
from .model import (
    DecoderVARegressor,
    build_qwen_va_model,
    load_saved_decoder_va_model,
)
from .preprocessing import build_english_dataset

__all__ = [
    "DecoderVARegressor",
    "ET2GazeProvider",
    "TokenizedVADataset",
    "VABatchCollator",
    "build_english_dataset",
    "build_qwen_va_model",
    "calculate_va_metrics",
    "collect_exclude_patterns",
    "audit_finetuning_text_overlap",
    "discover_completed_run",
    "download_pinned_idest_english",
    "download_pinned_semeval_subtask1_test",
    "fixed_unweighted_ensemble",
    "load_idest_english",
    "load_msp_podcast_test",
    "load_omg_emotion_test",
    "load_semeval_2026_subtask1_test",
    "load_saved_decoder_va_model",
    "load_filtered_folds",
    "semeval_subtask1_official_metrics",
    "write_oof_reports",
]

"""Build only the frozen historical baseline arguments."""
from trt_pipeline import training as train_model
from trt_pipeline.config import ROOT

def build_parser():
    """Expose familiar flags while assigning explicit historical recipe defaults."""
    parser = train_model._build_parser()
    parser.description = __doc__
    parser.epilog = "Historical-only recipe; unsupported condition changes fail before training."
    parser.set_defaults(
        finetuning_mode="full", precision="bf16", loss="mse", gaze_features=["TRT"],
        train_batch_size=16, eval_batch_size=16, gradient_accumulation_steps=1,
        no_iemocap=True, sentence_only=False, training_text_overlap_policy="keep",
        epochs=10, learning_rate=6e-6, gradient_checkpointing=True, group_by_length=True,
        data_dir=str(ROOT / "data/paper7_seed42"),
    )
    return parser

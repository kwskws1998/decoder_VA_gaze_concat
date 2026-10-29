"""Frozen settings for baseline, companion and a single shared prediction coefficient."""
from dataclasses import dataclass,field,asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import math
ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=ROOT/'data/paper7_seed42'
FOLD_FILENAMES=('full_dataset_fold1.csv','full_dataset_fold2.csv')
PREDICTION_ALPHA=0.307649721979222
DATA_HASHES={'full_dataset_fold1.csv':'86c882ca8b895f1008863a0dad909a1dd02c2c5e9007af2a4c3dc864e2d988b5','full_dataset_fold2.csv':'47ad2496e7c20ddb214684a748086c4d54b983b1655c3e8fe53ce4cbbb7e899a','english_dataset_manifest.json':'48e35d168fa958c3a3d9f999450c50d28ab68f18672081e40d041d72281c2d31'}

def _integer(value: Any, name: str, minimum: int = 1) -> None:
    """Reject booleans and fractional values for discrete experiment choices."""

    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _finite(value: Any, name: str, minimum: float, maximum: float | None = None,
            *, lower_inclusive: bool = True) -> None:
    """Validate numerical settings before downloading weights or allocating a GPU."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    invalid_lower = value < minimum if lower_inclusive else value <= minimum
    if invalid_lower or (maximum is not None and value > maximum):
        lower = ">=" if lower_inclusive else ">"
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{name} must be {lower} {minimum}{upper}.")


def _sequence(value: Any, name: str) -> tuple:
    """Normalize explicit Python lists or tuples without accepting scalar strings."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list or tuple.")
    return tuple(value)


@dataclass(frozen=True)
class BaselineConfig:
    """Retain the active Raw TRT model and a common effective batch of sixteen."""

    model: str = "qwen3.5-0.8b"
    model_id: str = "Qwen/Qwen3.5-0.8B-Base"
    model_revision: str = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    et_model_id: str = "skboy/et_prediction_2"
    et_revision: str = "5785e77309d9fce8b88e908a9db100c1a0a63456"
    et_filename: str = "et_predictor2_seed123.safetensors"
    et_cache_size: int = 70000
    finetuning_mode: str = "full"
    precision: str = "bf16"
    backbone_weight_dtype: str = "float32"
    gaze_fusion: str = "prefix-concat"
    gaze_features: tuple[str, ...] = ("TRT",)
    loss: str = "mse"
    epochs: float = 10.0
    learning_rate: float = 6e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 200
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    logging_steps: int = 50
    save_total_limit: int = 1
    max_steps: int = -1
    use_cpu: bool = False
    attn_implementation: str | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def __post_init__(self) -> None:
        """Enforce valid shared baseline settings while allowing declared overrides."""

        for name in ("model", "model_id", "model_revision", "et_model_id", "et_revision", "et_filename"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string.")
        if self.model not in ("qwen3.5-0.8b", "qwen"):
            raise ValueError("model must be the established qwen3.5-0.8b or qwen CLI alias.")
        if self.finetuning_mode != "full":
            raise ValueError("Only full fine-tuning is retained.")
        if self.precision not in ("auto", "bf16", "fp16", "fp32"):
            raise ValueError("precision must be auto, bf16, fp16, or fp32.")
        if self.backbone_weight_dtype not in ("float32", "bfloat16"):
            raise ValueError("backbone_weight_dtype must be float32 or bfloat16.")
        if self.backbone_weight_dtype == "bfloat16" and (
            self.precision != "bf16" or self.finetuning_mode != "full" or self.use_cpu
        ):
            raise ValueError("BF16 backbone weights require CUDA, precision bf16, and full fine-tuning.")
        if self.gaze_fusion not in ("none", "prefix-concat"):
            raise ValueError("gaze_fusion must be none or prefix-concat.")
        if self.loss != "mse":
            raise ValueError("Only the MSE objective is retained.")
        gaze_features = _sequence(self.gaze_features, "gaze_features")
        if len(set(gaze_features)) != len(gaze_features) or any(
            feature not in ("nFix", "FFD", "GPT", "TRT", "fixProp") for feature in gaze_features
        ):
            raise ValueError("gaze_features must contain unique supported ET2 feature names.")
        object.__setattr__(self, "gaze_features", gaze_features)
        for name in ("max_length", "train_batch_size", "eval_batch_size", "gradient_accumulation_steps",
                     "logging_steps", "save_total_limit", "lora_rank", "lora_alpha"):
            _integer(getattr(self, name), name)
        _integer(self.et_cache_size, "et_cache_size", 0)
        if type(self.max_steps) is not int or (self.max_steps != -1 and self.max_steps < 1):
            raise ValueError("max_steps must be -1 or a positive integer.")
        for name in ("epochs", "learning_rate"):
            _finite(getattr(self, name), name, 0, lower_inclusive=False)
        _finite(self.weight_decay, "weight_decay", 0)
        _finite(self.warmup_ratio, "warmup_ratio", 0, 1)
        if self.warmup_ratio == 1:
            raise ValueError("warmup_ratio must be < 1, matching the baseline Trainer contract.")
        _finite(self.lora_dropout, "lora_dropout", 0, 1)
        if self.lora_dropout == 1:
            raise ValueError("lora_dropout must be < 1.")
        for name in ("gradient_checkpointing", "group_by_length", "use_cpu"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean.")
        if self.use_cpu and self.precision not in ("auto", "fp32"):
            raise ValueError("CPU baseline runs require precision auto or fp32.")
        if self.attn_implementation is not None and (
            not isinstance(self.attn_implementation, str) or not self.attn_implementation.strip()
        ):
            raise ValueError("attn_implementation must be None or a nonempty string.")

    @property
    def effective_batch_size(self) -> int:
        """Return the single-GPU examples per optimizer update."""

        return self.train_batch_size * self.gradient_accumulation_steps

@dataclass(frozen=True)
class CompanionConfig:
    """Allow either original fold while retaining the measured companion recipe."""
    data_dir: Path=DATA_DIR
    fold: int=1
    seed: int=1
    optimizer_updates: int=16820
    eval_interval: int=1682
    max_gradient_norm: float=1.0
    conditions: tuple=('matched_mse',)
    baseline: BaselineConfig=field(default_factory=lambda: BaselineConfig(backbone_weight_dtype='bfloat16',train_batch_size=16,gradient_accumulation_steps=1,group_by_length=False))

    def __post_init__(self):
        """Reject changed seeds, budgets, losses or training policies."""
        expected=BaselineConfig(backbone_weight_dtype='bfloat16',train_batch_size=16,gradient_accumulation_steps=1,group_by_length=False)
        if self.fold not in (1,2) or self.seed!=1 or self.optimizer_updates!=16820 or self.eval_interval!=1682 or self.max_gradient_norm!=1.0 or self.conditions!=('matched_mse',) or self.baseline!=expected:
            raise ValueError('Only the fixed two-fold matched-MSE companion is supported.')

    def encoder_config(self):
        """Expose only fields used by the unchanged encoder argument helpers."""
        return SimpleNamespace(baseline=self.baseline,no_iemocap=True,sentence_only=False,training_text_overlap='keep')

    def to_dict(self):
        """Record the exact retained settings without retired method options."""
        value=asdict(self);value['data_dir']=str(self.data_dir);value['conditions']=list(self.conditions);value['baseline']['gaze_features']=list(self.baseline.gaze_features);value['official_repositories']={}
        return value

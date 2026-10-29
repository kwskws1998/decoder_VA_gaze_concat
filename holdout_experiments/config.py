"""Predeclared settings shared by all three fits and the fixed prediction mixture."""

from dataclasses import asdict, dataclass
import math

from trt_pipeline.config import PREDICTION_ALPHA


ARMS = ("baseline", "companion", "redistribution")
CONDITIONS = ("baseline", "companion", "prediction_mixture", "redistribution")
METRICS = tuple(f"{metric}_{axis}" for metric in ("pearson_corr", "rmse", "mae")
                for axis in ("valence", "arousal"))
EXPOSURE_NOTE = (
    "These original data were used in previous development experiments. The fixed "
    "mixture coefficient was selected previously on these data. This run enforces "
    "train/validation/test separation internally; its test is not a historically "
    "untouched dataset. No coefficient or hyperparameter search is performed."
)


@dataclass(frozen=True)
class Settings:
    """Freeze settings before any training or test prediction is produced."""

    schema: int = 1
    split_seed: int = 42
    seed: int = 42
    fractions: tuple = (0.8, 0.1, 0.1)
    epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 16
    max_length: int = 200
    learning_rate: float = 6e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    redistribution_learning_rate: float = 1e-3
    sigma_left: float = 1.0
    sigma_right: float = 1.0
    min_sigma: float = 1e-6
    prediction_alpha: float = PREDICTION_ALPHA
    noise_std_fraction: float = 0.01
    model_id: str = "Qwen/Qwen3.5-0.8B-Base"
    model_revision: str = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    et_model_id: str = "skboy/et_prediction_2"
    et_revision: str = "5785e77309d9fce8b88e908a9db100c1a0a63456"
    et_filename: str = "et_predictor2_seed123.safetensors"

    def __post_init__(self):
        """Reject malformed settings instead of coercing invalid budgets or coefficients."""

        for name in ("schema", "epochs", "batch_size", "eval_batch_size", "max_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.schema != 1:
            raise ValueError("Unsupported protocol schema.")
        for name in ("split_seed", "seed"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32).")
        if len(self.fractions) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
            for x in self.fractions
        ) or not math.isclose(sum(self.fractions), 1.0, abs_tol=1e-12):
            raise ValueError("Three positive train/val/test fractions must sum to one.")
        object.__setattr__(self, "fractions", tuple(self.fractions))
        for name in ("learning_rate", "max_grad_norm", "redistribution_learning_rate",
                     "sigma_left", "sigma_right", "min_sigma"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("weight_decay", "warmup_ratio", "prediction_alpha", "noise_std_fraction"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x < 1:
                raise ValueError(f"{name} must be finite and in [0,1).")
        if self.noise_std_fraction != 0.01 or self.prediction_alpha != PREDICTION_ALPHA:
            raise ValueError("The companion noise and previously selected mixture coefficient are fixed.")
        for name in ("model_id", "model_revision", "et_model_id", "et_revision", "et_filename"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string.")

    def as_dict(self):
        """Serialize the complete settings with explicit JSON array semantics."""

        value = asdict(self)
        value["fractions"] = list(self.fractions)
        return value


def updates_per_epoch(rows, settings):
    """Count actual optimizer updates, including the last partial physical batch."""

    return math.ceil(rows / settings.batch_size)

"""Versioned, explicit DDPM semantics independent of the baseline manifest."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class DiffusionConfig:
    """Fix the loss, target transform, conditioning, and reproducible sampler."""

    version: int = 1
    timesteps: int = 100
    hidden_dim: int = 128
    blocks: int = 3
    conditioning: str = "token-trt"
    samples: int = 16
    sampling_seed: int = 2026
    schedule: str = "cosine_s0.008_beta_max0.999"
    objective: str = "epsilon_mse"
    target_transform: str = "normalized_va_0_1_to_minus1_1"
    sampler: str = "ddpm_fixed_posterior_variance"
    clip_clean: bool = True
    randomness: str = "cpu_common_random_numbers_v1"

    def __post_init__(self):
        """Reject ambiguous, unsupported, or numerically invalid configurations."""

        for key in ("version", "timesteps", "hidden_dim", "blocks", "samples", "sampling_seed"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer.")
        if self.version != 1 or self.timesteps < 2:
            raise ValueError("Version must be 1 and timesteps must be at least 2.")
        if self.hidden_dim < 8 or self.hidden_dim % 2 or self.blocks < 1 or self.samples < 1:
            raise ValueError("Use an even hidden_dim >= 8 and positive blocks/samples.")
        if not 0 <= self.sampling_seed < 2**63:
            raise ValueError("sampling_seed must be in [0, 2**63).")
        if self.conditioning not in ("pooled", "token-trt"):
            raise ValueError("conditioning must be pooled or token-trt.")
        fixed = {
            "schedule": "cosine_s0.008_beta_max0.999",
            "objective": "epsilon_mse",
            "target_transform": "normalized_va_0_1_to_minus1_1",
            "sampler": "ddpm_fixed_posterior_variance",
            "randomness": "cpu_common_random_numbers_v1",
        }
        if any(getattr(self, key) != value for key, value in fixed.items()):
            raise ValueError("Unsupported DDPM semantics; use the documented version-1 contract.")
        if type(self.clip_clean) is not bool:
            raise ValueError("clip_clean must be boolean.")

    def to_dict(self):
        """Serialize every scientifically relevant DDPM and sampling choice."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        """Require a complete manifest instead of silently supplying missing fields."""

        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Incomplete or unknown diffusion configuration fields.")
        return cls(**value)


def positive_finite(value, name):
    """Validate scalar optimizer settings before starting a costly run."""

    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return value

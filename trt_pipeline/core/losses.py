"""The retained FP32 two-axis mean-squared-error objective."""
from dataclasses import dataclass
from torch import Tensor
from torch.nn import functional as F

@dataclass(frozen=True)
class LossBreakdown:
    """Expose the scalar mean loss used by the original baseline Trainer."""
    total: Tensor
    point: Tensor

def va_regression_loss(logits: Tensor, labels: Tensor, loss_name: str) -> LossBreakdown:
    """Keep the original MSE value and gradient while rejecting retired objectives."""
    if loss_name.lower() != 'mse':
        raise ValueError('Only MSE training is retained')
    if labels.ndim != 2 or labels.shape[-1] != 2 or logits.shape != labels.shape or not len(labels):
        raise ValueError('VA logits and labels must have matching nonempty [batch,2] shapes')
    mse=F.mse_loss(logits[:,:2].float(),labels.float())
    return LossBreakdown(total=mse,point=mse)

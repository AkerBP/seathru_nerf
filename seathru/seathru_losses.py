import torch
from torch import Tensor
from jaxtyping import Float


def acc_loss(
    transmittance_object: Float[Tensor, "*bs num_samples 1"], beta: float
) -> torch.Tensor:
    """Compute the acc_loss.

    Args:
        transmittance_object: Transmittances of object.
        factor: factor to control the weight of the two distributions

    Returns:
            Acc Loss.
    """
    P = torch.exp(-torch.abs(transmittance_object) / 0.1) + beta * torch.exp(
        -torch.abs(1 - transmittance_object) / 0.1
    )
    loss = -torch.log(P)
    return loss.mean()


def recon_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute the reconstruction loss.

    Args:
        gt: Ground truth.
        pred: RGB prediction.

    Returns:
        Reconstruction loss.
    """
    inner = torch.square((pred - gt) / (pred.detach() + 1e-3))
    return torch.mean(inner)

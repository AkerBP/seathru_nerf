from jaxtyping import Float

import torch
import os
from torch import Tensor


def get_transmittance(
    deltas: Tensor, densities: Float[Tensor, "*bs num_samples 1"]
) -> Float[Tensor, "*bs num_samples 1"]:
    """Compute transmittance for each ray sample.

    Args:
        deltas: Distance between each ray sample.
        densities: Densities of each ray sample.

    Returns:
        Transmittance for each ray sample.
    """
    delta_density = deltas * densities
    transmittance_object = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance_object = torch.cat(
        [
            torch.zeros(
                (*transmittance_object.shape[:1], 1, transmittance_object.shape[-1]),
                device=transmittance_object.device,
            ),
            transmittance_object,
        ],
        dim=-2,
    )
    transmittance_object = torch.exp(-transmittance_object)

    return transmittance_object


def get_bayer_mask(indices: torch.Tensor) -> torch.Tensor:
    """Get bayer mask for rgb loss.

    Args:
        indices: tensor of shape ([num_rays, 2]) containing the row and column of the
        pixel corresponding to the ray.

    Returns:
        Tensor of shape ([num_rays, 3]) containing the bayer mask.
    """

    # Red is top left (0, 0).
    r = (indices[:, 0] % 2 == 0) * (indices[:, 1] % 2 == 0)
    # Green is top right (0, 1) and bottom left (1, 0).
    g = (indices[:, 0] % 2 == 1) * (indices[:, 1] % 2 % 2 == 0) + (
        indices[:, 0] % 2 == 0
    ) * (indices[:, 1] % 2 == 1)
    # Blue is bottom right (1, 1).
    b = (indices[:, 0] % 2 == 1) * (indices[:, 1] % 2 == 1)
    return torch.stack([r, g, b], dim=-1).float()


def save_debug_info(
    weights: torch.Tensor,
    transmittance: torch.Tensor,
    depth: torch.Tensor,
    prop_depth: torch.Tensor,
    step: int,
) -> None:
    """Save output tensors for debugging purposes.

    Args:
        weights: weights tensor.
        transmittance: transmittance tensor.
        depth: depth tensor.
        prop_depth: prop_depth tensor.
        step: current step of training.

    Returns:
        None
    """

    rel_path = "debugging/"

    # if first step create direcrory
    if not os.path.exists(f"{rel_path}debug"):
        os.makedirs(f"{rel_path}debug")

    # save transmittance
    torch.save(transmittance.cpu(), f"{rel_path}debug/transmittance_{step}.pt")

    # save weights
    torch.save(weights.cpu(), f"{rel_path}debug/weights_{step}.pt")

    # save depth
    torch.save(depth.cpu(), f"{rel_path}debug/depth_{step}.pt")

    # save prop depth
    torch.save(prop_depth.cpu(), f"{rel_path}debug/prop_depth_{step}.pt")


def add_water(
    img: Tensor, depth: Tensor, beta_D: Tensor, beta_B: Tensor, B_inf: Tensor
) -> Tensor:
    """Add water effect to image.
       Image formation model from https://openaccess.thecvf.com/content_cvpr_2018/papers/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.pdf (Eq. 20).

    Args:
        img: image to add water effect to.
        beta_D: depth map.
        beta_B: background map.
        B_inf: background image.

    Returns:
        Image with water effect.
    """  # noqa: E501

    depth = depth.repeat_interleave(3, dim=-1)
    I_out = img * torch.exp(-beta_D * depth) + B_inf * (1 - torch.exp(-beta_B * depth))

    return I_out

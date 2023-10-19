from typing import Literal, Union, Tuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples

from seathru.seathru_utils import get_transmittance


class SeathruRGBRenderer(nn.Module):
    """Volumetric RGB rendering of an unnderwater scene.

    Args:
        use_new_rendering_eqs: Whether to use the new rendering equations.
    """

    def __init__(self, use_new_rendering_eqs: bool = True) -> None:
        super().__init__()
        self.use_new_rendering_eqs = use_new_rendering_eqs

    def combine_rgb(
        self,
        object_rgb: Float[Tensor, "*bs num_samples 3"],
        medium_rgb: Float[Tensor, "*bs num_samples 3"],
        medium_bs: Float[Tensor, "*bs num_samples 3"],
        medium_attn: Float[Tensor, "*bs num_samples 3"],
        densities: Float[Tensor, "*bs num_samples 1"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Render pixel colour along rays using volumetric rendering.

        Args:
            object_rgb: RGB values of object.
            medium_rgb: RGB values of medium.
            medium_bs:  sigma backscatter of medium.
            medium_attn: sigma attenuation of medium.
            densities: Object densities.
            weights: Object weights.
            ray_samples: Set of ray samples.

        Returns:
            Rendered pixel colour (and direct, bs, and J if using new rendering
            equations and not training).
        """

        # Old rendering equations
        if not self.use_new_rendering_eqs:
            s = ray_samples.frustums.starts

            # Object RGB
            attn_component = torch.exp(-medium_attn * s)
            comp_object_rgb = torch.sum(weights * attn_component * object_rgb, dim=-2)

            # Medium RGB
            # Ignore type error that occurs because ray_samples can be initialized without deltas
            transmittance_object = get_transmittance(ray_samples.deltas, densities)  # type: ignore
            bs_comp1 = torch.exp(-medium_bs * s)
            bs_comp2 = 1 - torch.exp(-medium_bs * ray_samples.deltas)
            comp_medium_rgb = torch.sum(
                transmittance_object * bs_comp1 * bs_comp2 * medium_rgb, dim=-2
            )
            comp_medium_rgb = torch.nan_to_num(comp_medium_rgb)

            comp_rgb = comp_object_rgb + comp_medium_rgb

            return comp_rgb

        # New rendering equations (adapted from https://github.com/deborahLevy130/seathru_NeRF/blob/c195ff3384632058d56aef0cddb8057b538c1511/internal/render.py#L288C8-L288C8)
        # Lead to the same comp_rgb as the old rendering equations, but also return
        # direct, bs, and J. (and detach deltas for medium contributions as it showed
        # to enhance stability)
        else:
            # Ignore type error that occurs because ray_samples can be initialized without deltas
            transmittance_object = get_transmittance(ray_samples.deltas, densities)  # type: ignore

            # Get transmittance_attn
            # Ignore type error that occurs because ray_samples can be initialized without deltas
            deltas_detached = ray_samples.deltas.detach()  # type: ignore
            transmittance_attn = get_transmittance(
                deltas=deltas_detached, densities=medium_attn
            )

            # Get bs_weights
            transmittance_bs = get_transmittance(
                deltas=deltas_detached, densities=medium_bs
            )
            alphas_bs = 1 - torch.exp(-medium_bs * deltas_detached)
            bs_weights = alphas_bs * transmittance_bs

            # Get direct and bs
            direct = torch.sum(weights * transmittance_attn * object_rgb, dim=-2)
            bs = torch.sum(transmittance_object * bs_weights * medium_rgb, dim=-2)
            comp_rgb = direct + bs
            J = (torch.sum(weights * object_rgb, dim=-2)).detach()

            # Only return direct, bs, and J if not training to save memory and time
            if not self.training:
                return comp_rgb, direct, bs, J
            else:
                return comp_rgb

    def forward(
        self,
        object_rgb: Float[Tensor, "*bs num_samples 3"],
        medium_rgb: Float[Tensor, "*bs num_samples 3"],
        medium_bs: Float[Tensor, "*bs num_samples 3"],
        medium_attn: Float[Tensor, "*bs num_samples 3"],
        densities: Float[Tensor, "*bs num_samples 1"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Render pixel colour along rays using volumetric rendering.

        Args:
            object_rgb: RGB values of object.
            medium_rgb: RGB values of medium.
            medium_bs:  sigma backscatter of medium.
            medium_attn: sigma attenuation of medium.
            densities: Object densities.
            weights: Object weights.
            ray_samples: Set of ray samples.

        Returns:
            Rendered pixel colour (and direct, bs, and J if using new rendering
            equations and not training).
        """

        if not self.training:
            object_rgb = torch.nan_to_num(object_rgb)
            medium_rgb = torch.nan_to_num(medium_rgb)
            medium_bs = torch.nan_to_num(medium_bs)
            medium_attn = torch.nan_to_num(medium_attn)

            if self.use_new_rendering_eqs:
                rgb, direct, bs, J = self.combine_rgb(
                    object_rgb,
                    medium_rgb,
                    medium_bs,
                    medium_attn,
                    densities,
                    weights,
                    ray_samples=ray_samples,
                )

                torch.clamp_(rgb, min=0.0, max=1.0)
                torch.clamp_(direct, min=0.0, max=1.0)
                torch.clamp_(bs, min=0.0, max=1.0)
                torch.clamp_(J, min=0.0, max=1.0)

                return rgb, direct, bs, J

            else:
                rgb = self.combine_rgb(
                    object_rgb,
                    medium_rgb,
                    medium_bs,
                    medium_attn,
                    densities,
                    weights,
                    ray_samples=ray_samples,
                )

                if isinstance(rgb, torch.Tensor):
                    torch.clamp_(rgb, min=0.0, max=1.0)

                return rgb

        else:
            rgb = self.combine_rgb(
                object_rgb,
                medium_rgb,
                medium_bs,
                medium_attn,
                densities,
                weights,
                ray_samples=ray_samples,
            )
            return rgb


class SeathruDepthRenderer(nn.Module):
    """Calculate depth along rays.

    Args:
        far_plane: Far plane of rays.
        method: Depth calculation method.
    """

    def __init__(
        self, far_plane: float, method: Literal["median", "expected"] = "median"
    ) -> None:
        super().__init__()
        self.far_plane = far_plane
        self.method = method

    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
    ) -> Float[Tensor, "*batch 1"]:
        """Calculate depth along rays.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.

        Returns:
            Depht values.
        """
        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        acc_weights = torch.sum(
            weights, dim=-2, keepdim=True
        )  # Shape: [num_rays, 1, 1]

        # As object weights are not guaranteed to sum to 1 (e.g. when in a region of
        # only water without an object), we need to add an additional sample at the end
        # of each ray to ensure that the weights sum to 1.

        # Compute the weight for the additional sample
        bg_weight = 1.0 - acc_weights

        # Concatenate this new weight to the original weights tensor
        weights_ext = torch.cat([weights, bg_weight], dim=1)
        # Concatenate the far plane to the original steps tensor
        steps_ext = torch.cat(
            [
                steps,
                torch.ones((*steps.shape[:1], 1, 1), device=steps.device)
                * self.far_plane,
            ],
            dim=1,
        )

        if self.method == "expected":
            eps = 1e-10
            depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
            depth = torch.clip(depth, steps.min(), steps.max())
            return depth

        if self.method == "median":
            # Code snippet from nerfstudio DepthRenderer
            cumulative_weights_ext = torch.cumsum(
                weights_ext[..., 0], dim=-1
            )  # [..., num_samples]
            split = (
                torch.ones((*weights_ext.shape[:-2], 1), device=weights_ext.device)
                * 0.5
            )  # [..., 1]
            median_index = torch.searchsorted(
                cumulative_weights_ext, split, side="left"
            )  # [..., 1]
            median_index = torch.clamp(
                median_index, 0, steps_ext.shape[-2] - 1
            )  # [..., 1]
            median_depth = torch.gather(
                steps_ext[..., 0], dim=-1, index=median_index
            )  # [..., 1]
            return median_depth

        raise NotImplementedError(f"Method {self.method} not implemented")

from dataclasses import dataclass, field
from typing import Dict, List, Type, Literal, Tuple

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.renderers import AccumulationRenderer
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss, interlevel_loss
from nerfstudio.utils import colormaps
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)


from seathru.seathru_field import SeathruField
from seathru.seathru_fieldheadnames import SeathruHeadNames
from seathru.seathru_renderers import SeathruRGBRenderer
from seathru.seathru_losses import acc_loss, recon_loss
from seathru.seathru_utils import get_bayer_mask, save_debug_info, get_transmittance
from seathru.seathru_renderers import SeathruDepthRenderer


@dataclass
class SeathruModelConfig(ModelConfig):
    """SeaThru-NeRF Config."""

    _target: Type = field(default_factory=lambda: SeathruModel)
    near_plane: float = 0.05
    """Near plane of rays."""
    far_plane: float = 10.0
    """Far plane of rays."""
    num_levels: int = 16
    """Number of levels of the hashmap for the object base MLP."""
    min_res: int = 16
    """Minimum resolution of the hashmap for the object base MLP."""
    max_res: int = 8192
    """Maximum resolution of the hashmap for the object base MLP."""
    log2_hashmap_size: int = 21
    """Size of the hashmap for the object base MLP."""
    features_per_level: int = 2
    """Number of features per level of the hashmap for the object base MLP."""
    num_layers: int = 2
    """Number of hidden layers for the object base MLP."""
    hidden_dim: int = 256
    """Dimension of hidden layers for the object base MLP."""
    bottleneck_dim: int = 63
    """Bottleneck dimension between object base MLP and object head MLP."""
    num_layers_colour: int = 3
    """Number of hidden layers for colour MLP."""
    hidden_dim_colour: int = 256
    """Dimension of hidden layers for colour MLP."""
    num_layers_medium: int = 2
    """Number of hidden layers for medium MLP."""
    hidden_dim_medium: int = 128
    """Dimension of hidden layers for medium MLP."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Implementation of the MLPs (tcnn or torch)."""
    use_viewing_dir_obj_rgb: bool = False
    """Whether to use viewing direction in object rgb MLP."""
    object_density_bias: float = 0.0
    """Bias for object density."""
    medium_density_bias: float = 0.0
    """Bias for medium density (sigma_bs and sigma_attn)."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 128)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup."""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps."""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Whether to use the same proposal network."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 512,
                "use_linear": False,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 7,
                "max_res": 2048,
                "use_linear": False,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing (this gives an exploration at the \
        beginning of training)."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 15000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to \
        the camera."""
    initial_acc_loss_mult: float = 0.0001
    """Initial accuracy loss multiplier."""
    final_acc_loss_mult: float = 0.0001
    """Final accuracy loss multiplier."""
    acc_decay: int = 10000
    """Decay of the accuracy loss multiplier. (After this many steps, acc_loss_mult = \
        final_acc_loss_mult.)"""
    rgb_loss_use_bayer_mask: bool = False
    """Whether to use a Bayer mask for the RGB loss."""
    prior_on: Literal["weights", "transmittance"] = "transmittance"
    """Prior on the proposal weights or transmittance."""
    debug: bool = False
    """Whether to save debug information."""
    beta_prior: float = 100.0
    """Beta hyperparameter for the prior used in the acc_loss."""
    use_viewing_dir_obj_rgb: bool = False
    """Whether to use viewing direction in object rgb MLP."""
    use_new_rendering_eqs: bool = True
    """Whether to use the new rendering equations."""


class SeathruModel(Model):
    """Seathru model

    Args:
        config: SeaThru-NeRF configuration to instantiate the model with.
    """

    config: SeathruModelConfig  # type: ignore

    def populate_modules(self):
        """Setup the fields and modules."""
        super().populate_modules()

        # Scene contraction
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Initialize SeaThru field
        self.field = SeathruField(
            aabb=self.scene_box.aabb,
            num_levels=self.config.num_levels,
            min_res=self.config.min_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            bottleneck_dim=self.config.bottleneck_dim,
            num_layers_colour=self.config.num_layers_colour,
            hidden_dim_colour=self.config.hidden_dim_colour,
            num_layers_medium=self.config.num_layers_medium,
            hidden_dim_medium=self.config.hidden_dim_medium,
            spatial_distortion=scene_contraction,
            implementation=self.config.implementation,
            use_viewing_dir_obj_rgb=self.config.use_viewing_dir_obj_rgb,
            object_density_bias=self.config.object_density_bias,
            medium_density_bias=self.config.medium_density_bias,
        )

        # Initialize proposal network(s) (this code snippet is taken from from nerfacto)
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert (
                len(self.config.proposal_net_args_list) == 1
            ), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )

        def update_schedule(step):
            return np.clip(
                np.interp(
                    step,
                    [0, self.config.proposal_warmup],
                    [0, self.config.proposal_update_every],
                ),
                1,
                self.config.proposal_update_every,
            )

        # Initial sampler
        initial_sampler = None  # None is for piecewise as default
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(
                single_jitter=self.config.use_single_jitter
            )

        # Proposal sampler
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )

        # Renderers
        self.renderer_rgb = SeathruRGBRenderer(
            use_new_rendering_eqs=self.config.use_new_rendering_eqs
        )
        self.renderer_depth = SeathruDepthRenderer(
            far_plane=self.config.far_plane, method="median"
        )
        self.renderer_accumulation = AccumulationRenderer()

        # Losses
        self.rgb_loss = MSELoss(reduction="none")

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Step member variable to keep track of the training step
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the parameter groups for the optimizer. (Code snippet from nerfacto)

        Returns:
            The parameter groups.
        """
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def step_cb(self, step) -> None:
        """Function for training callbacks to use to update training step.

        Args:
            step: The training step.
        """
        self.step = step

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Get the training callbacks.
           (Code of this function is from nerfacto but added step tracking for debugging.)

        Args:
            training_callback_attributes: The training callback attributes.

        Returns:
            List with training callbacks.
        """
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        # Additional callback to track the training step for decaying and
        # debugging purposes
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.step_cb,
            )
        )

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:  # type: ignore
        """Get outputs from the model.

        Args:
            ray_bundle: RayBundle containing the input rays to compute and render.

        Returns:
            Dict containing the outputs of the model.
        """

        ray_samples: RaySamples

        # Get output from proposal network(s)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )

        # Get output from Seathru field
        field_outputs = self.field.forward(ray_samples)
        field_outputs[FieldHeadNames.DENSITY] = torch.nan_to_num(
            field_outputs[FieldHeadNames.DENSITY], nan=1e-3
        )
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        # Render rgb (only rgb in training and rgb, direct, bs, J in eval for
        # performance reasons as we do not optimize with respect to direct, bs, J)
        # ignore types to avoid unnecesarry pyright errors
        if self.training or not self.config.use_new_rendering_eqs:
            rgb = self.renderer_rgb(
                object_rgb=field_outputs[FieldHeadNames.RGB],
                medium_rgb=field_outputs[SeathruHeadNames.MEDIUM_RGB],  # type: ignore
                medium_bs=field_outputs[SeathruHeadNames.MEDIUM_BS],  # type: ignore
                medium_attn=field_outputs[SeathruHeadNames.MEDIUM_ATTN],  # type: ignore
                densities=field_outputs[FieldHeadNames.DENSITY],
                weights=weights,
                ray_samples=ray_samples,
            )
            direct = None
            bs = None
            J = None
        else:
            rgb, direct, bs, J = self.renderer_rgb(
                object_rgb=field_outputs[FieldHeadNames.RGB],
                medium_rgb=field_outputs[SeathruHeadNames.MEDIUM_RGB],  # type: ignore
                medium_bs=field_outputs[SeathruHeadNames.MEDIUM_BS],  # type: ignore
                medium_attn=field_outputs[SeathruHeadNames.MEDIUM_ATTN],  # type: ignore
                densities=field_outputs[FieldHeadNames.DENSITY],
                weights=weights,
                ray_samples=ray_samples,
            )

        # Render depth and accumulation
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # Calculate transmittance and add to outputs for acc_loss calculation
        # Ignore type error that occurs because ray_samples can be initialized without deltas
        transmittance = get_transmittance(
            ray_samples.deltas, field_outputs[FieldHeadNames.DENSITY]  # type: ignore
        )
        outputs = {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "transmittance": transmittance,
            "weights": weights,
            "direct": direct if not self.training else None,
            "bs": bs if not self.training else None,
            "J": J if not self.training else None,
        }

        # Add outputs from proposal network(s) to outputs if training for proposal loss
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        # Add proposed depth to outputs
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        return outputs

    def get_metrics_dict(self, outputs, batch):
        """Get evaluation metrics dictionary.
        (Compared to get_image_metrics_and_images(), this function does not render
        images and is executed at each training step.)

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Dict containing the metrics to log.
        """
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Calculate loss dictionary.

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Dict containing the loss values.
        """
        loss_dict = {}
        image = batch["image"].to(self.device)

        # RGB loss
        if self.config.rgb_loss_use_bayer_mask:
            # Cut out camera/image indices and pass to get_bayer_mask
            bayer_mask = get_bayer_mask(batch["indices"][:, 1:].to(self.device))
            squared_error = self.rgb_loss(image, outputs["rgb"])  # clip or not clip?
            scaling_grad = 1 / (outputs["rgb"].detach() + 1e-3)
            loss = squared_error * torch.square(scaling_grad)
            denom = torch.sum(bayer_mask)
            loss_dict["rgb_loss"] = torch.sum(loss * bayer_mask) / denom
        else:
            loss_dict["rgb_loss"] = recon_loss(gt=image, pred=outputs["rgb"])

        if self.training:
            # Accumulation loss
            if self.step < self.config.acc_decay:
                acc_loss_mult = self.config.initial_acc_loss_mult
            else:
                acc_loss_mult = self.config.final_acc_loss_mult

            if self.config.prior_on == "weights":
                loss_dict["acc_loss"] = acc_loss_mult * acc_loss(
                    transmittance_object=outputs["weights"], beta=self.config.beta_prior
                )
            elif self.config.prior_on == "transmittance":
                loss_dict["acc_loss"] = acc_loss_mult * acc_loss(
                    transmittance_object=outputs["transmittance"],
                    beta=self.config.beta_prior,
                )
            else:
                raise ValueError(f"Unknown prior_on: {self.config.prior_on}")

            # Proposal loss
            loss_dict[
                "interlevel_loss"
            ] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Get evaluation metrics dictionary and images to log for eval batch.
        (extended from nerfacto)

        Args:
            outputs: Dict containing the outputs of the model.
            batch: Dict containing the gt data.

        Returns:
            Tuple containing the metrics to log (as scalars) and the images to log.
        """
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]

        # Accumulation and depth maps
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Log the images
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        if self.config.use_new_rendering_eqs:
            # J (clean image), direct and bs images
            direct = outputs["direct"]
            bs = outputs["bs"]
            J = outputs["J"]

            combined_direct = torch.cat([direct], dim=1)
            combined_bs = torch.cat([bs], dim=1)
            combined_J = torch.cat([J], dim=1)

            # log the images
            images_dict["direct"] = combined_direct
            images_dict["bs"] = combined_bs
            images_dict["J"] = combined_J

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # Compute metrics
        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # Log the metrics (as scalars)
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        # Log the proposal depth maps
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(outputs[key])
            images_dict[key] = prop_depth_i

        # Debugging
        if self.config.debug:
            save_debug_info(
                weights=outputs["weights"],
                transmittance=outputs["transmittance"],
                depth=outputs["depth"],
                prop_depth=outputs["prop_depth_0"],
                step=self.step,
            )

        return metrics_dict, images_dict

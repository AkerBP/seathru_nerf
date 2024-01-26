from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from seathru.seathru_model import SeathruModelConfig


# Base method configuration
seathru_method = MethodSpecification(
    config=TrainerConfig(
        method_name="seathru-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
            ),
            model=SeathruModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "camera_opt": {
                "mode": "off",
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=6e-6, max_steps=500000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="SeaThru-NeRF for underwater scenes.",
)

# Lite method configuration
seathru_method_lite = MethodSpecification(
    config=TrainerConfig(
        method_name="seathru-nerf-lite",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=50000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=4096,
            ),
            model=SeathruModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_nerf_samples_per_ray=64,
                num_proposal_samples_per_ray=(256, 128),
                max_res=2048,
                log2_hashmap_size=19,
                hidden_dim=64,
                bottleneck_dim=31,
                hidden_dim_colour=64,
                hidden_dim_medium=64,
                proposal_net_args_list=[
                    {
                        "hidden_dim": 16,
                        "log2_hashmap_size": 17,
                        "num_levels": 5,
                        "max_res": 128,
                        "use_linear": False,
                    },
                    {
                        "hidden_dim": 16,
                        "log2_hashmap_size": 17,
                        "num_levels": 5,
                        "max_res": 256,
                        "use_linear": False,
                    },
                ],
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "camera_opt": {
                "mode": "off",
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=6e-6, max_steps=500000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Light SeaThru-NeRF for underwater scenes.",
)

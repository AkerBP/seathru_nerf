import pytest
import torch

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

from seathru.seathru_field import SeathruField
from seathru.seathru_fieldheadnames import SeathruHeadNames


class TestSeathruField:
    # Create fixture for the classes that are reused in the tests
    @pytest.fixture(scope="class")
    def fields(self):
        scene_contraction = SceneContraction(order=float("inf"))
        scene_box = SceneBox(torch.Tensor([[-1, -1, -1], [1, 1, 1]]))
        field1 = SeathruField(
            aabb=scene_box.aabb,
            num_levels=2,
            min_res=4,
            max_res=8,
            log2_hashmap_size=17,
            features_per_level=2,
            num_layers=2,
            hidden_dim=16,
            bottleneck_dim=8,
            num_layers_colour=2,
            hidden_dim_colour=16,
            num_layers_medium=2,
            hidden_dim_medium=16,
            spatial_distortion=scene_contraction,
            implementation="torch",
            use_viewing_dir_obj_rgb=False,
            object_density_bias=0.0,
            medium_density_bias=0.0,
        )

        scene_contraction2 = None
        scene_box2 = SceneBox(torch.Tensor([[-1, -1, -1], [1, 1, 1]]))
        field2 = SeathruField(
            aabb=scene_box2.aabb,
            num_levels=2,
            min_res=4,
            max_res=8,
            log2_hashmap_size=17,
            features_per_level=2,
            num_layers=2,
            hidden_dim=16,
            bottleneck_dim=8,
            num_layers_colour=2,
            hidden_dim_colour=16,
            num_layers_medium=2,
            hidden_dim_medium=16,
            spatial_distortion=scene_contraction2,
            implementation="torch",
            use_viewing_dir_obj_rgb=True,
            object_density_bias=0.0,
            medium_density_bias=0.0,
        )

        return [field1, field2]

    @pytest.fixture(scope="class")
    def ray_samples(self):
        origins = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        )  # *bs, 3
        directions = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1]]]
        )  # *bs, 3
        pixel_area = torch.tensor([[[1.0], [1.0], [1.0]]])  # *bs, 1
        intervals = torch.tensor([[[0.0], [2.0], [4.0], [6]]])
        starts = intervals[..., :-1, :]
        ends = intervals[..., 1:, :]
        frustums = Frustums(
            origins=origins,
            directions=directions,
            starts=starts,
            ends=ends,
            pixel_area=pixel_area,
        )
        return RaySamples(frustums=frustums)

    def test_get_density(self, fields, ray_samples):
        """Test that basic foward pass and get_density function works."""
        for field in fields:
            density, bottleneck_vector = field.get_density(ray_samples)
            # Check that the outputs are tensors and have the correct shape
            assert isinstance(density, torch.Tensor)
            assert isinstance(bottleneck_vector, torch.Tensor)
            assert density.shape == torch.Size([1, 3, 1])
            assert bottleneck_vector.shape == torch.Size([1, 3, 8])

    def test_get_outputs(self, fields, ray_samples):
        """Test that basic foward pass and get_outputs function works."""
        for field in fields:
            _, bottleneck_vector = field.get_density(ray_samples)
            outputs = field.get_outputs(ray_samples, bottleneck_vector)
            # Check that the outputs dict contains the right entries, they are tensors and have the correct shape
            assert isinstance(outputs, dict)
            assert FieldHeadNames.RGB in outputs.keys()
            assert SeathruHeadNames.MEDIUM_RGB in outputs.keys()
            assert SeathruHeadNames.MEDIUM_BS in outputs.keys()
            assert SeathruHeadNames.MEDIUM_ATTN in outputs.keys()
            assert isinstance(outputs[FieldHeadNames.RGB], torch.Tensor)
            assert isinstance(outputs[SeathruHeadNames.MEDIUM_RGB], torch.Tensor)
            assert isinstance(outputs[SeathruHeadNames.MEDIUM_BS], torch.Tensor)
            assert isinstance(outputs[SeathruHeadNames.MEDIUM_ATTN], torch.Tensor)
            assert outputs[FieldHeadNames.RGB].shape == torch.Size([1, 3, 3])
            assert outputs[SeathruHeadNames.MEDIUM_RGB].shape == torch.Size([1, 3, 3])
            assert outputs[SeathruHeadNames.MEDIUM_BS].shape == torch.Size([1, 3, 3])
            assert outputs[SeathruHeadNames.MEDIUM_ATTN].shape == torch.Size([1, 3, 3])

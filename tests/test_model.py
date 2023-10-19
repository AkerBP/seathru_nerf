import pytest
import torch

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler

from seathru.seathru_model import SeathruModel, SeathruModelConfig
from seathru.seathru_field import SeathruField


class TestSeathruModel:
    # Create fixture for the classes that are reused in the tests
    @pytest.fixture(scope="class")
    def model(self):
        return SeathruModel(
            config=SeathruModelConfig(),
            scene_box=SceneBox(torch.Tensor([[-1, -1, -1], [1, 1, 1]])),
            num_train_data=1,
        )

    @pytest.fixture(scope="class")
    def ray_bundle(self):
        origins = torch.tensor(
            [[0.0, 0.1, 0.0], [0.0, 0.0, 0.4], [0.2, 0.0, 0.0]]
        )  # *bs, 3
        directions = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )  # *bs, 3
        pixel_area = torch.tensor([[1.0], [1.0], [1.0]])  # *bs, 1
        nears = torch.zeros(3, 1)
        fars = torch.ones(3, 1) * 50
        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            nears=nears,
            fars=fars,
        )

    # @dataclass
    # class RayBundle(TensorDataclass):
    #     """A bundle of ray parameters."""

    #     # TODO(ethan): make sure the sizes with ... are correct
    #     origins: Float[Tensor, "*batch 3"]
    #     """Ray origins (XYZ)"""
    #     directions: Float[Tensor, "*batch 3"]
    #     """Unit ray direction vector"""
    #     pixel_area: Float[Tensor, "*batch 1"]
    #     """Projected area of pixel a distance 1 away from origin"""
    #     camera_indices: Optional[Int[Tensor, "*batch 1"]] = None
    #     """Camera indices"""
    #     nears: Optional[Float[Tensor, "*batch 1"]] = None
    #     """Distance along ray to start sampling"""
    #     fars: Optional[Float[Tensor, "*batch 1"]] = None
    #     """Rays Distance along ray to stop sampling"""
    #     metadata: Dict[str, Shaped[Tensor, "num_rays latent_dims"]] = field(default_factory=dict)
    #     """Additional metadata or data needed for interpolation, will mimic shape of rays"""
    #     times: Optional[Float[Tensor, "*batch 1"]] = None
    #     """Times at which rays are sampled"""

    def test_populate_modules(self, model):
        """Test populate_modules function."""
        model.populate_modules()
        # check if field is initialized
        assert isinstance(model.field, SeathruField)
        # check if proposal networks(s) are initialized
        assert len(model.proposal_networks) > 0
        # check if proposal sampler is initialized
        assert isinstance(model.proposal_sampler, ProposalNetworkSampler)

    def test_get_param_groups(self, model):
        """Test whether we can retreive parameter lists with get_param_groups function."""
        model.populate_modules()
        param_groups = model.get_param_groups()
        # check if param_groups is a dictionary
        assert isinstance(param_groups, dict)
        # check if param_groups contains the right number of elements (one for network and one for proposal networks)
        assert len(param_groups) == 2

    def test_get_outputs(self, model, ray_bundle):
        """Test that basic foward pass and get_outputs function works."""
        outputs = model.get_outputs(ray_bundle)

        # check that the outputs dict contains the right entries, they are tensors and have the correct shape
        assert isinstance(outputs, dict)
        outputs_keys_expected = [
            "rgb",
            "depth",
            "accumulation",
            "transmittance",
            "weights",
            "weights_list",
            "ray_samples_list",
            "prop_depth_0",
            "prop_depth_1",
        ]
        for key in outputs_keys_expected:
            assert key in outputs.keys()
        for key in outputs_keys_expected[:5]:
            print(key)
            assert isinstance(outputs[key], torch.Tensor)
        assert outputs["rgb"].shape == torch.Size([3, 3])  # [bs, 3]
        assert outputs["depth"].shape == torch.Size([3, 1])  # [bs, 1]
        assert outputs["accumulation"].shape == torch.Size([3, 1])  # [bs, 1]
        assert outputs["transmittance"].shape == torch.Size(
            [3, 64, 1]
        )  # [bs, num_samples, 1]
        assert outputs["weights"].shape == torch.Size(
            [3, 64, 1]
        )  # [bs, num_samples, 1]

    def test_get_metrics_dict(self, model, ray_bundle):
        """Test that get_metrics_dict function works."""
        # create dummy outputs and batch
        outputs = model.get_outputs(ray_bundle)
        batch = {"image": torch.ones((3, 3))}
        metrics_dict = model.get_metrics_dict(outputs, batch)
        # check that the metrics_dict contains the right entries, they are tensors and have the correct shape
        assert "psnr" in metrics_dict.keys()

    def test_get_loss_dict(self, model, ray_bundle):
        """Basic test for get_loss_dict function."""
        outputs = model.get_outputs(ray_bundle)
        # create dummy batch (gt values)
        batch = {"image": torch.ones((3, 3))}
        loss_dict = model.get_loss_dict(outputs, batch)
        # check that the loss_dict contains the right entries, they are tensors and have the correct shape
        assert isinstance(loss_dict, dict)
        losses_expected = ["rgb_loss", "acc_loss", "interlevel_loss"]
        for key in losses_expected:
            assert key in loss_dict.keys()
            assert isinstance(loss_dict[key], torch.Tensor)
            assert loss_dict[key].shape == torch.Size([])

    def test_get_image_metrics_and_images(self, model, ray_bundle):
        """Test that get_image_metrics_and_images function works."""
        # create dummy outputs and batch (gt values)
        outputs = {
            "rgb": torch.ones((1280, 720, 3)),
            "accumulation": torch.ones((1280, 720, 1)),
            "depth": torch.ones((1280, 720, 1)),
            "prop_depth_0": torch.ones((1280, 720, 1)),
            "prop_depth_1": torch.ones((1280, 720, 1)),
            "direct": torch.ones((1280, 720, 3)),
            "J": torch.ones((1280, 720, 3)),
            "bs": torch.ones((1280, 720, 3)),
        }
        batch = {"image": torch.ones((1280, 720, 3))}
        image_metrics, images = model.get_image_metrics_and_images(outputs, batch)
        assert True

    # @pytest.fixture(scope="class")
    # def ray_samples(self):
    #     origins = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]) # *bs, 3
    #     directions = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1]]]) # *bs, 3
    #     pixel_area = torch.tensor([[[1.0], [1.0], [1.0]]]) # *bs, 1
    #     intervals = torch.tensor([[[0.0], [2.0], [4.0], [6]]])
    #     starts = intervals[..., :-1, :]
    #     ends = intervals[..., 1:, :]
    #     frustums = Frustums(
    #         origins=origins, directions=directions, starts=starts, ends=ends, pixel_area=pixel_area
    #     )
    #     return RaySamples(frustums=frustums)

    # def test_get_density(self, fields, ray_samples):
    #     '''Test that basic foward pass and get_density function works.'''
    #     for field in fields:
    #         density, bottleneck_vector = field.get_density(ray_samples)
    #         # check that the outputs are tensors and have the correct shape
    #         assert isinstance(density, torch.Tensor)
    #         assert isinstance(bottleneck_vector, torch.Tensor)
    #         assert density.shape == torch.Size([1, 3, 1])
    #         assert bottleneck_vector.shape == torch.Size([1, 3, 8])

    # def test_get_outputs(self, fields, ray_samples):
    #     '''Test that basic foward pass and get_outputs function works.'''
    #     for field in fields:
    #         _, bottleneck_vector = field.get_density(ray_samples)
    #         outputs = field.get_outputs(ray_samples, bottleneck_vector)
    #         # check that the outputs dict contains the right entries, they are tensors and have the correct shape
    #         assert isinstance(outputs, dict)
    #         assert FieldHeadNames.RGB in outputs.keys()
    #         assert SeathruHeadNames.MEDIUM_RGB in outputs.keys()
    #         assert SeathruHeadNames.MEDIUM_BS in outputs.keys()
    #         assert SeathruHeadNames.MEDIUM_ATTN in outputs.keys()
    #         assert isinstance(outputs[FieldHeadNames.RGB], torch.Tensor)
    #         assert isinstance(outputs[SeathruHeadNames.MEDIUM_RGB], torch.Tensor)
    #         assert isinstance(outputs[SeathruHeadNames.MEDIUM_BS], torch.Tensor)
    #         assert isinstance(outputs[SeathruHeadNames.MEDIUM_ATTN], torch.Tensor)
    #         assert outputs[FieldHeadNames.RGB].shape == torch.Size([1, 3, 3])
    #         assert outputs[SeathruHeadNames.MEDIUM_RGB].shape == torch.Size([1, 3, 3])
    #         assert outputs[SeathruHeadNames.MEDIUM_BS].shape == torch.Size([1, 3, 3])
    #         assert outputs[SeathruHeadNames.MEDIUM_ATTN].shape == torch.Size([1, 3, 3])

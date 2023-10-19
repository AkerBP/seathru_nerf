import pytest
import torch
import math

from nerfstudio.cameras.rays import Frustums, RaySamples

from seathru.seathru_renderers import SeathruRGBRenderer, SeathruDepthRenderer


class TestDepthRenderer:
    # create fixtures for the different renderers
    @pytest.fixture(scope="class")
    def renderer_expected(self):
        return SeathruDepthRenderer(far_plane=10, method="expected")

    @pytest.fixture(scope="class")
    def renderer_median(self):
        return SeathruDepthRenderer(far_plane=10, method="median")

    @pytest.mark.parametrize(
        "input_intervals, input_weights, expected",
        [
            (
                torch.tensor([[[0.0], [2.0], [4.0], [6.0]]]),
                torch.tensor([[[1 / 3], [1 / 3], [1 / 3]]]),
                torch.tensor([[[3.0]]]),
            ),
            (
                torch.tensor(
                    [[[0.0], [2.0], [4.0], [6.0]], [[0.0], [2.0], [4.0], [6.0]]]
                ),
                torch.tensor(
                    [[[1 / 3], [1 / 3], [1 / 3]], [[1 / 3], [1 / 3], [1 / 3]]]
                ),
                torch.tensor([[[3.0]], [[3.0]]]),
            ),
            (
                torch.tensor([[[1.0], [5.0], [9.0], [13.0]]]),
                torch.tensor([[[1 / 3], [1 / 3], [1 / 3]]]),
                torch.tensor([[[7.0]]]),
            ),
        ],
    )
    def test_median(self, renderer_median, input_intervals, input_weights, expected):
        """Test the median depth renderer for known cases, where weights add up to 1."""
        # create frustrums (start and ends are important for the depth renderer)
        _ = torch.ones((1, 3, 1))
        starts = input_intervals[..., :-1, :]
        ends = input_intervals[..., 1:, :]
        frustums = Frustums(
            origins=_, directions=_, starts=starts, ends=ends, pixel_area=_
        )
        input_ray_samples = RaySamples(frustums=frustums)

        # calculate depth
        depth = renderer_median(weights=input_weights, ray_samples=input_ray_samples)
        assert torch.allclose(depth, torch.tensor(expected))

    @pytest.mark.parametrize(
        "input_intervals, input_weights, expected",
        [
            (
                torch.tensor([[[0.0], [2.0], [4.0], [6]]]),
                torch.tensor([[[1 / 3], [1 / 3], [1 / 3]]]),
                torch.tensor([[[3.0]]]),
            ),
            (
                torch.tensor([[[0.0], [2.0], [4.0], [6]]]),
                torch.tensor([[[0.1], [0.2], [0.7]]]),
                torch.tensor([[[4.2]]]),
            ),
            (
                torch.tensor([[[0.0], [2.0], [4.0], [6]], [[0.0], [2.0], [4.0], [6]]]),
                torch.tensor([[[0.1], [0.2], [0.7]], [[0.1], [0.2], [0.7]]]),
                torch.tensor([[[4.2]], [[4.2]]]),
            ),
        ],
    )
    def test_expected(
        self, renderer_expected, input_intervals, input_weights, expected
    ):
        """Test the expected depth renderer for known cases."""
        # create frustrums and ray samples (start and ends are important for the depth renderer)
        _ = torch.ones((1, 3, 1))
        starts = input_intervals[..., :-1, :]
        ends = input_intervals[..., 1:, :]
        frustums = Frustums(
            origins=_, directions=_, starts=starts, ends=ends, pixel_area=_
        )
        input_ray_samples = RaySamples(frustums=frustums)

        # calculate depth
        depth = renderer_expected(weights=input_weights, ray_samples=input_ray_samples)
        assert torch.allclose(depth, expected)


class TestRGBRenderer:
    # create fixtures for the different renderers
    @pytest.fixture(scope="class")
    def renderer_old(self):
        return SeathruRGBRenderer(use_new_rendering_eqs=False)

    @pytest.fixture(scope="class")
    def renderer_new(self):
        return SeathruRGBRenderer(use_new_rendering_eqs=True)

    @pytest.mark.parametrize(
        "input_intervals, input_deltas, input_med_attn, input_c_obj, input_densities, expected",
        [
            (
                torch.tensor([[[0.0], [1.0]]]),
                torch.ones((1, 1, 1)),
                torch.ones((1, 1, 3)),
                torch.ones((1, 1, 3)),
                torch.ones((1, 1, 1)),
                torch.tensor([[[1 - 1 / math.e, 1 - 1 / math.e, 1 - 1 / math.e]]]),
            ),
            (
                torch.tensor([[[0.0], [1.0], [0.2]]]),
                torch.ones((1, 2, 1)),
                torch.ones((1, 2, 3)),
                torch.ones((1, 2, 3)),
                torch.ones((1, 2, 1)),
                torch.tensor(
                    [
                        [
                            (1 + 1 / math.e**2) * (1 - 1 / math.e),
                            (1 + 1 / math.e**2) * (1 - 1 / math.e),
                            (1 + 1 / math.e**2) * (1 - 1 / math.e),
                        ]
                    ]
                ),
            ),
        ],
    )
    def test_c_obj(
        self,
        renderer_old,
        renderer_new,
        input_intervals,
        input_deltas,
        input_med_attn,
        input_c_obj,
        input_densities,
        expected,
    ):
        """Test renderers when c_med is 0. (I.e. only test eq (7) in report)"""
        # create frustrums and ray samples (start and ends are important for the depth renderer)
        _ = torch.ones_like(input_deltas)
        starts = input_intervals[..., :-1, :]
        ends = input_intervals[..., 1:, :]
        frustums = Frustums(
            origins=_, directions=_, starts=starts, ends=ends, pixel_area=_
        )
        input_ray_samples = RaySamples(frustums=frustums, deltas=input_deltas)

        # calculate weights
        input_weights = input_ray_samples.get_weights(input_densities)

        # set medium rgb to 0
        input_c_med = torch.zeros_like(input_med_attn)

        # calculate rgb
        rgb_old = renderer_old(
            input_c_obj,
            input_c_med,
            _,
            input_med_attn,
            input_densities,
            input_weights,
            input_ray_samples,
        )
        rgb_new = renderer_new(
            input_c_obj,
            input_c_med,
            _,
            input_med_attn,
            input_densities,
            input_weights,
            input_ray_samples,
        )
        assert torch.allclose(rgb_old, expected), "Old rendering equations failed!"
        assert torch.allclose(rgb_new, expected), "New rendering equaions failed!"

    @pytest.mark.parametrize(
        "input_intervals, input_deltas, input_med_bs, input_c_med, input_densities, expected",
        [
            (
                torch.tensor([[[0.0], [1.0]]]),
                torch.ones((1, 1, 1)),
                torch.ones((1, 1, 3)),
                torch.ones((1, 1, 3)),
                torch.ones((1, 1, 1)),
                torch.tensor([[[1 - 1 / math.e, 1 - 1 / math.e, 1 - 1 / math.e]]]),
            ),
            (
                torch.tensor([[[0.0], [1.0], [0.2]]]),
                torch.ones((1, 2, 1)),
                torch.ones((1, 2, 3)),
                torch.ones((1, 2, 3)),
                torch.ones((1, 2, 1)),
                torch.tensor(
                    [
                        [
                            (1 + 1 / math.e**2) * (1 - 1 / math.e),
                            (1 + 1 / math.e**2) * (1 - 1 / math.e),
                            (1 + 1 / math.e**2) * (1 - 1 / math.e),
                        ]
                    ]
                ),
            ),
        ],
    )
    def test_c_med(
        self,
        renderer_old,
        renderer_new,
        input_intervals,
        input_deltas,
        input_med_bs,
        input_c_med,
        input_densities,
        expected,
    ):
        """Test renderers when c_obj is 0. (I.e. only test eq (8) in report)"""
        # create frustrums and ray samples (start and ends are important for the depth renderer)
        _ = torch.ones_like(input_deltas)
        starts = input_intervals[..., :-1, :]
        ends = input_intervals[..., 1:, :]
        frustums = Frustums(
            origins=_, directions=_, starts=starts, ends=ends, pixel_area=_
        )
        input_ray_samples = RaySamples(frustums=frustums, deltas=input_deltas)

        # calculate weights
        input_weights = input_ray_samples.get_weights(input_densities)

        # set object rgb to 0
        input_c_obj = torch.zeros_like(input_med_bs)

        # calculate rgb
        rgb_old = renderer_old(
            input_c_obj,
            input_c_med,
            input_med_bs,
            _,
            input_densities,
            input_weights,
            input_ray_samples,
        )
        rgb_new = renderer_new(
            input_c_obj,
            input_c_med,
            input_med_bs,
            _,
            input_densities,
            input_weights,
            input_ray_samples,
        )
        assert torch.allclose(rgb_old, expected), "Old rendering equations failed!"
        assert torch.allclose(rgb_new, expected), "New rendering equaions failed!"

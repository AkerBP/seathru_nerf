import pytest
import torch
import numpy as np
import os
import math

from seathru.seathru_utils import (
    get_transmittance,
    get_bayer_mask,
    save_debug_info,
    add_water,
)


@pytest.mark.parametrize("dimensions", [((100, 50, 1)), ((20, 1000, 1)), ((1, 1, 1))])
def test_get_transmittance_dimensions(dimensions):
    """Test get_transmittance for correct output dimensions."""
    input_deltas = torch.ones(dimensions)
    input_densities = torch.ones(dimensions)
    output = get_transmittance(input_deltas, input_densities)
    assert output.shape == input_deltas.shape


@pytest.mark.parametrize(
    "input_deltas, input_densities",
    [
        (torch.zeros((10, 20, 1)), torch.zeros((10, 20, 1))),
        (torch.ones((10, 20, 1)), torch.zeros((10, 20, 1))),
        (torch.zeros((10, 20, 1)), torch.ones((10, 20, 1))),
    ],
)
def test_get_transmittance_one(input_deltas, input_densities):
    """Test get_transmittance function for cases where the output should tensor of 1."""
    output = get_transmittance(input_deltas, input_densities)
    assert torch.allclose(output, torch.ones_like(output))


@pytest.mark.parametrize(
    "input_deltas, input_densities, expected",
    [
        (
            torch.ones((1, 3, 1)),
            torch.tensor([[[2.0], [4.0], [1.0]]]),
            torch.tensor(
                [[[1], [1 / np.exp(2)], [1 / np.exp(6)]]], dtype=torch.float32
            ),
        ),
        (
            torch.tensor([[[0.2], [3], [0.5]]]),
            torch.tensor([[[1], [0.5], [5]]]),
            torch.tensor(
                [[[1], [1 / np.exp(0.2)], [1 / np.exp(1.7)]]], dtype=torch.float32
            ),
        ),
    ],
)
def test_get_transmittance(input_deltas, input_densities, expected):
    """Test get_transmittance function for cases where the output is known."""
    output = get_transmittance(input_deltas, input_densities)
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("dimensions", [((1, 2)), ((1000, 2))])
def test_get_bayer_mask_dimensions(dimensions):
    """Test get_bayer_mask for correct output dimensions."""
    input_indices = torch.ones(dimensions)
    output = get_bayer_mask(input_indices)
    assert output.shape == torch.Size([dimensions[0], 3])


@pytest.mark.parametrize(
    "input_indices, expected",
    [
        (
            torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        ),
        (
            torch.tensor([[0, 0], [0, 1], [0, 2], [1, 1], [2, 2], [3, 3]]),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
        (
            torch.tensor([[0, 0], [0, 0], [0, 0]]),
            torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ),
    ],
)
def test_get_bayer_mask(input_indices, expected):
    """Test get_bayer_mask function for cases where the output is known."""
    output = get_bayer_mask(input_indices)
    assert torch.equal(output, expected)


def test_save_debug_info(tmp_path):
    """Test if save_debug_info function correctly saves the debug info."""
    # Create some dummy data
    weights = torch.ones((50, 20, 1))
    transmittance = torch.ones((50, 20, 1))
    depth = torch.ones((50, 20, 1))
    prop_depth = torch.ones((50, 20, 1))
    step = 500

    # Change the current working directory to the tmp_path directory (a temporary directory by pytest that will be deleted after the test)
    original_path = os.getcwd()
    os.chdir(tmp_path)

    # Call the function
    save_debug_info(weights, transmittance, depth, prop_depth, step)

    # Change back to the original directory
    os.chdir(original_path)

    # Check if correct directory was created
    debug_dir = tmp_path / "debugging" / "debug"
    assert debug_dir.exists()

    # Check if all files were saved
    assert (debug_dir / f"transmittance_{step}.pt").exists()
    assert (debug_dir / f"weights_{step}.pt").exists()
    assert (debug_dir / f"depth_{step}.pt").exists()
    assert (debug_dir / f"prop_depth_{step}.pt").exists()

    # Check if the files contain the correct data
    saved_weights = torch.load(debug_dir / f"weights_{step}.pt")
    saved_transmittance = torch.load(debug_dir / f"transmittance_{step}.pt")
    saved_depth = torch.load(debug_dir / f"depth_{step}.pt")
    saved_prop_depth = torch.load(debug_dir / f"prop_depth_{step}.pt")

    assert torch.equal(saved_weights, weights.cpu())
    assert torch.equal(saved_transmittance, transmittance.cpu())
    assert torch.equal(saved_depth, depth.cpu())
    assert torch.equal(saved_prop_depth, prop_depth.cpu())


@pytest.mark.parametrize(
    "input_image, input_depth, input_beta_D, input_beta_B, input_B_inf, expected",
    [
        (
            torch.ones((400, 500, 3)),
            torch.ones((400, 500, 1)),
            torch.ones(3),
            torch.ones(3),
            torch.ones(3),
            torch.ones((400, 500, 3)),
        ),  # All ones should return e+(1-e) = 1
        (
            torch.ones((400, 500, 3)),
            torch.ones((400, 500, 1)),
            torch.ones(3),
            torch.zeros(3),
            torch.ones(3),
            torch.ones((400, 500, 3)) * (1 / math.e),
        ),  # All ones should and beta_B=0 should return 1/e
        (
            torch.ones((400, 500, 3)),
            torch.ones((400, 500, 1)),
            torch.zeros(3),
            torch.ones(3),
            torch.ones(3),
            torch.ones((400, 500, 3)) * (2 - 1 / math.e),
        ),  # All ones should and beta_D=0 should return 2-1/e
    ],
)
def test_add_water(
    input_image, input_depth, input_beta_D, input_beta_B, input_B_inf, expected
):
    """Test add_water function for known cases."""
    output = add_water(
        input_image, input_depth, input_beta_D, input_beta_B, input_B_inf
    )
    assert torch.allclose(output, expected)

import pytest
import torch
import numpy as np

from seathru.seathru_losses import acc_loss, recon_loss


@pytest.mark.parametrize(
    "dimensions",
    [
        ((100, 50, 3)),
        ((20, 1000, 3)),
        ((1, 1, 3)),
    ],
)
def test_acc_loss_dim(dimensions):
    """Test the acc_loss function for correct dimensions."""
    input = torch.zeros(dimensions)
    output = acc_loss(input, beta=1)
    # output should be a scalar
    assert output.shape == torch.Size([])


@pytest.mark.parametrize(
    "input", [(torch.zeros((10, 10, 1))), (torch.ones((10, 10, 1)))]
)
def test_acc_loss_zero(input):
    """Test the acc_loss function for cases that should return 0."""
    output = acc_loss(input, beta=1).item()
    assert np.isclose(output, 0.0, atol=1e-4)


def test_acc_loss_negative():
    """Test acc_loss for cases that should return negative values."""
    input = torch.ones((10, 10, 1))
    output = acc_loss(input, beta=100).item()
    assert output < 0.0


@pytest.mark.parametrize(
    "dimensions", [((100, 50, 3)), ((20, 1000, 3)), ((1, 1, 3)), ((1, 3)), ((50, 3))]
)
def test_recon_loss_dim(dimensions):
    """Test recon_loss for correct dimensions."""
    output = recon_loss(torch.zeros(dimensions), torch.zeros(dimensions))
    # output should be a scalar
    assert output.shape == torch.Size([])


@pytest.mark.parametrize(
    "input_gt, input_pred, expected",
    [
        (torch.ones((50, 50, 3)), torch.ones((50, 50, 3)), 0.0),
        (torch.zeros((20, 20, 3)), torch.zeros((20, 20, 3)), 0.0),
        (torch.ones((1, 1, 3)), torch.zeros((1, 1, 3)), 1e6),
        (
            torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
            torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
            0.0,
        ),
    ],
)
def test_recon_loss(input_gt, input_pred, expected):
    """Test recon_loss for known cases."""
    output = recon_loss(input_gt, input_pred).item()
    assert np.isclose(output, expected)

"""Pytest test suite for comparing numpy and PyTorch module implementations.

This module contains tests to verify the correctness of the numpy implementation
by comparing forward pass values and gradients with equivalent PyTorch implementations.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from modules.numpy.linear import Linear
from modules.numpy.activations import ReLU
from modules.numpy.objectives import MSELoss


class NumpyMLP:
    """Simple Multi-Layer Perceptron implementation using numpy modules.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden layer.
        output_size (int): Size of output layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.layer1 = Linear(input_size, hidden_size)
        self.activation = ReLU()
        self.layer2 = Linear(hidden_size, output_size)
        self.loss_fn = MSELoss()

        # Store intermediate values for backward pass
        self._layer1_output = None
        self._activation_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the MLP.

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, output_size).
        """
        self._layer1_output = self.layer1.forward(x)
        self._activation_output = self.activation.forward(self._layer1_output)
        output = self.layer2.forward(self._activation_output)
        return output

    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Backward pass through the MLP.

        Args:
            predictions (np.ndarray): Model predictions.
            targets (np.ndarray): Target values.

        Returns:
            float: Loss value.
        """
        # Compute loss
        loss = self.loss_fn.forward(predictions, targets)

        # Compute gradients starting from loss
        loss_grad = self.loss_fn.backward(predictions, targets)

        # Backward through second layer
        layer2_grad = self.layer2.backward(loss_grad)

        # Backward through activation
        activation_grad = self.activation.backward(layer2_grad)

        # Backward through first layer
        self.layer1.backward(activation_grad)

        return loss

    def get_parameters(self):
        """Get model parameters.

        Returns:
            dict: Dictionary containing weights and biases.
        """
        return {
            "layer1_weights": self.layer1.weights.copy(),
            "layer1_biases": self.layer1.biases.copy(),
            "layer2_weights": self.layer2.weights.copy(),
            "layer2_biases": self.layer2.biases.copy(),
        }

    def get_gradients(self):
        """Get computed gradients.

        Returns:
            dict: Dictionary containing gradients.
        """
        return {
            "layer1_weight_grads": self.layer1._w_grads.copy()
            if self.layer1._w_grads is not None
            else None,
            "layer1_bias_grads": self.layer1._b_grads.copy()
            if self.layer1._b_grads is not None
            else None,
            "layer2_weight_grads": self.layer2._w_grads.copy()
            if self.layer2._w_grads is not None
            else None,
            "layer2_bias_grads": self.layer2._b_grads.copy()
            if self.layer2._b_grads is not None
            else None,
        }

    def set_parameters(self, params: dict):
        """Set model parameters.

        Args:
            params (dict): Dictionary containing weights and biases.
        """
        self.layer1.weights = params["layer1_weights"].copy()
        self.layer1.biases = params["layer1_biases"].copy()
        self.layer2.weights = params["layer2_weights"].copy()
        self.layer2.biases = params["layer2_biases"].copy()


class PyTorchMLP(nn.Module):
    """Equivalent PyTorch MLP implementation.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden layer.
        output_size (int): Size of output layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss and perform backward pass.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Loss value.
        """
        return self.loss_fn(predictions, targets)

    def get_parameters_dict(self):
        """Get model parameters as numpy arrays.

        Returns:
            dict: Dictionary containing weights and biases as numpy arrays.
        """
        return {
            "layer1_weights": self.layer1.weight.data.numpy(),
            "layer1_biases": self.layer1.bias.data.numpy(),
            "layer2_weights": self.layer2.weight.data.numpy(),
            "layer2_biases": self.layer2.bias.data.numpy(),
        }

    def get_gradients_dict(self):
        """Get computed gradients as numpy arrays.

        Returns:
            dict: Dictionary containing gradients as numpy arrays.
        """
        return {
            "layer1_weight_grads": self.layer1.weight.grad.numpy()
            if self.layer1.weight.grad is not None
            else None,
            "layer1_bias_grads": self.layer1.bias.grad.numpy()
            if self.layer1.bias.grad is not None
            else None,
            "layer2_weight_grads": self.layer2.weight.grad.numpy()
            if self.layer2.weight.grad is not None
            else None,
            "layer2_bias_grads": self.layer2.bias.grad.numpy()
            if self.layer2.bias.grad is not None
            else None,
        }

    def set_parameters_from_dict(self, params: dict):
        """Set model parameters from numpy arrays.

        Args:
            params (dict): Dictionary containing weights and biases as numpy arrays.
        """
        with torch.no_grad():
            self.layer1.weight.copy_(torch.from_numpy(params["layer1_weights"]))
            self.layer1.bias.copy_(torch.from_numpy(params["layer1_biases"]))
            self.layer2.weight.copy_(torch.from_numpy(params["layer2_weights"]))
            self.layer2.bias.copy_(torch.from_numpy(params["layer2_biases"]))


@pytest.fixture
def setup_models():
    """Fixture to set up synchronized numpy and PyTorch models.

    Returns:
        tuple: (numpy_model, pytorch_model, test_input, test_targets)
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Model dimensions
    input_size = 10
    hidden_size = 8
    output_size = 5
    batch_size = 4

    # Create models
    numpy_model = NumpyMLP(input_size, hidden_size, output_size)
    pytorch_model = PyTorchMLP(input_size, hidden_size, output_size)

    # Synchronize parameters (copy from numpy to pytorch)
    numpy_params = numpy_model.get_parameters()
    pytorch_model.set_parameters_from_dict(numpy_params)

    # Create test data
    np.random.seed(123)
    test_input = np.random.randn(batch_size, input_size).astype(np.float32)
    test_targets = np.random.randn(batch_size, output_size).astype(np.float32)

    return numpy_model, pytorch_model, test_input, test_targets


def test_forward_pass_equivalence(setup_models):
    """Test that forward pass outputs are equivalent between numpy and PyTorch implementations.

    Args:
        setup_models: Fixture providing synchronized models and test data.
    """
    numpy_model, pytorch_model, test_input, _ = setup_models

    # Forward pass through numpy model
    numpy_output = numpy_model.forward(test_input)

    # Forward pass through PyTorch model
    pytorch_input = torch.from_numpy(test_input)
    pytorch_output = pytorch_model(pytorch_input).detach().numpy()

    # Compare outputs
    np.testing.assert_allclose(
        numpy_output,
        pytorch_output,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Forward pass outputs do not match between numpy and PyTorch implementations",
    )


def test_loss_computation_equivalence(setup_models):
    """Test that loss computation is equivalent between numpy and PyTorch implementations.

    Args:
        setup_models: Fixture providing synchronized models and test data.
    """
    numpy_model, pytorch_model, test_input, test_targets = setup_models

    # Forward pass
    numpy_output = numpy_model.forward(test_input)
    pytorch_input = torch.from_numpy(test_input)
    pytorch_output = pytorch_model(pytorch_input)

    # Compute loss
    numpy_loss = numpy_model.loss_fn.forward(numpy_output, test_targets)
    pytorch_targets = torch.from_numpy(test_targets)
    pytorch_loss = pytorch_model.compute_loss(pytorch_output, pytorch_targets).item()

    # Compare losses
    np.testing.assert_allclose(
        numpy_loss,
        pytorch_loss,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Loss values do not match between numpy and PyTorch implementations",
    )


def test_gradient_computation_equivalence(setup_models):
    """Test that gradient computation is equivalent between numpy and PyTorch implementations.

    Args:
        setup_models: Fixture providing synchronized models and test data.
    """
    numpy_model, pytorch_model, test_input, test_targets = setup_models

    # Forward and backward pass through numpy model
    numpy_output = numpy_model.forward(test_input)
    numpy_loss = numpy_model.backward(numpy_output, test_targets)
    numpy_grads = numpy_model.get_gradients()

    # Forward and backward pass through PyTorch model
    pytorch_input = torch.from_numpy(test_input)
    pytorch_targets = torch.from_numpy(test_targets)
    pytorch_output = pytorch_model(pytorch_input)
    pytorch_loss = pytorch_model.compute_loss(pytorch_output, pytorch_targets)
    pytorch_loss.backward()
    pytorch_grads = pytorch_model.get_gradients_dict()

    # Compare gradients for each parameter
    for param_name in [
        "layer1_weight_grads",
        "layer1_bias_grads",
        "layer2_weight_grads",
        "layer2_bias_grads",
    ]:
        numpy_grad = numpy_grads[param_name]
        pytorch_grad = pytorch_grads[param_name]

        assert numpy_grad is not None, f"Numpy gradient for {param_name} is None"
        assert pytorch_grad is not None, f"PyTorch gradient for {param_name} is None"

        np.testing.assert_allclose(
            numpy_grad,
            pytorch_grad,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Gradients for {param_name} do not match between numpy and PyTorch implementations",
        )


def test_multiple_forward_backward_cycles(setup_models):
    """Test consistency over multiple forward-backward cycles.

    Args:
        setup_models: Fixture providing synchronized models and test data.
    """
    numpy_model, pytorch_model, test_input, test_targets = setup_models

    num_cycles = 3

    for cycle in range(num_cycles):
        # Clear PyTorch gradients
        pytorch_model.zero_grad()

        # Forward pass
        numpy_output = numpy_model.forward(test_input)
        pytorch_input = torch.from_numpy(test_input)
        pytorch_output = pytorch_model(pytorch_input)

        # Backward pass
        numpy_loss = numpy_model.backward(numpy_output, test_targets)
        pytorch_targets = torch.from_numpy(test_targets)
        pytorch_loss = pytorch_model.compute_loss(pytorch_output, pytorch_targets)
        pytorch_loss.backward()

        # Compare outputs and losses
        np.testing.assert_allclose(
            numpy_output,
            pytorch_output.detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Forward pass outputs do not match at cycle {cycle}",
        )

        np.testing.assert_allclose(
            numpy_loss,
            pytorch_loss.item(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Loss values do not match at cycle {cycle}",
        )

        # Compare gradients
        numpy_grads = numpy_model.get_gradients()
        pytorch_grads = pytorch_model.get_gradients_dict()

        for param_name in [
            "layer1_weight_grads",
            "layer1_bias_grads",
            "layer2_weight_grads",
            "layer2_bias_grads",
        ]:
            np.testing.assert_allclose(
                numpy_grads[param_name],
                pytorch_grads[param_name],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Gradients for {param_name} do not match at cycle {cycle}",
            )


def test_different_batch_sizes():
    """Test that models work correctly with different batch sizes."""
    input_size = 6
    hidden_size = 4
    output_size = 3

    batch_sizes = [1, 5, 10]

    for batch_size in batch_sizes:
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Create models
        numpy_model = NumpyMLP(input_size, hidden_size, output_size)
        pytorch_model = PyTorchMLP(input_size, hidden_size, output_size)

        # Synchronize parameters
        numpy_params = numpy_model.get_parameters()
        pytorch_model.set_parameters_from_dict(numpy_params)

        # Create test data
        test_input = np.random.randn(batch_size, input_size).astype(np.float32)
        test_targets = np.random.randn(batch_size, output_size).astype(np.float32)

        # Forward pass
        numpy_output = numpy_model.forward(test_input)
        pytorch_input = torch.from_numpy(test_input)
        pytorch_output = pytorch_model(pytorch_input)

        # Compare outputs
        np.testing.assert_allclose(
            numpy_output,
            pytorch_output.detach().numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Forward pass outputs do not match for batch size {batch_size}",
        )


def test_single_sample_forward_pass():
    """Test forward pass with a single sample (1D input)."""
    input_size = 5
    hidden_size = 3
    output_size = 2

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Create models
    numpy_model = NumpyMLP(input_size, hidden_size, output_size)
    pytorch_model = PyTorchMLP(input_size, hidden_size, output_size)

    # Synchronize parameters
    numpy_params = numpy_model.get_parameters()
    pytorch_model.set_parameters_from_dict(numpy_params)

    # Create single sample (1D input)
    single_input = np.random.randn(input_size).astype(np.float32)

    # Forward pass - numpy model should handle 1D input
    numpy_output = numpy_model.forward(single_input.reshape(1, -1))

    # Forward pass - PyTorch model with reshaped input
    pytorch_input = torch.from_numpy(single_input.reshape(1, -1))
    pytorch_output = pytorch_model(pytorch_input)

    # Compare outputs
    np.testing.assert_allclose(
        numpy_output,
        pytorch_output.detach().numpy(),
        rtol=1e-6,
        atol=1e-6,
        err_msg="Single sample forward pass outputs do not match",
    )


if __name__ == "__main__":
    pytest.main([__file__])

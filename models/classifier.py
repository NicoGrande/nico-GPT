"""Neural network classifier implementation.

This module provides a configurable multi-layer perceptron classifier
with batch normalization, ReLU activations, and L2 regularization.
Originally designed for the Abalone dataset from the UCI Machine Learning
Repository: http://archive.ics.uci.edu/ml/datasets/Abalone

Classes:
    Classifier: A multi-layer neural network classifier with configurable architecture.

Example:
    >>> import numpy as np
    >>> from classifier import Classifier

    >>> # Create a classifier for 8 input features and 3 output classes
    >>> model = Classifier(input_dim=8, output_dim=3, hidden_dim=64)
    >>>
    >>> # Forward pass
    >>> x = np.random.randn(32, 8)  # batch_size=32, features=8
    >>> predictions = model.forward(x)
    >>>
    >>> # Training step
    >>> labels = np.random.randint(0, 3, (32,))
    >>> loss = model.compute_loss(predictions, labels)
    >>> model.backward(predictions, labels)
    >>> model.parameter_update(lr=0.001)
"""

import numpy as np

from modules.numpy.linear import Linear
from modules.numpy.activations import Softmax, ReLU
from modules.numpy.normalization import BatchNormalization
from modules.numpy.objectives import CrossEntropyLoss


class Classifier:
    """Multi-layer perceptron classifier with batch normalization.

    A configurable neural network classifier that supports multiple hidden layers,
    batch normalization, ReLU activations, and L2 regularization. The architecture
    consists of fully connected layers with batch normalization and ReLU activations,
    followed by a softmax output layer.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output classes.
        num_hidden: Number of hidden layers (default: 2).
        hidden_dim: Number of neurons in each hidden layer (default: 128).
        use_regularization: Whether to apply L2 regularization (default: True).
        regularization_lambda: L2 regularization strength (default: 1e-4).

    Attributes:
        num_hidden: Number of hidden layers in the network.
        hidden_dim: Number of neurons in each hidden layer.
        use_regularization: Whether L2 regularization is enabled.
        regularization_lambda: L2 regularization coefficient.
        model: List of network layers (Linear, BatchNorm, Activation).
        loss: Cross-entropy loss function.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden: int = 2,
        hidden_dim: int = 128,
        use_regularization: bool = True,
        regularization_lambda: float = 1e-4,
    ):
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim
        self.use_regularization = use_regularization
        self.regularization_lambda = regularization_lambda

        # Construct model architecture
        self.model = [
            Linear(input_dim, hidden_dim),
            BatchNormalization(hidden_dim),
            ReLU(),
        ]
        for _ in range(self.num_hidden):
            self.model.extend(
                [Linear(hidden_dim, hidden_dim), BatchNormalization(hidden_dim), ReLU()]
            )
        self.model.extend([Linear(hidden_dim, output_dim), Softmax()])

        # Initialize loss function
        self.loss = CrossEntropyLoss()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Makes the classifier callable.

        Args:
            x: Input array of shape (batch_size, input_dim).

        Returns:
            Class probabilities of shape (batch_size, output_dim).
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network.

        Args:
            x: Input array of shape (batch_size, input_dim).

        Returns:
            Class probabilities of shape (batch_size, output_dim) after softmax.
        """
        for module in self.model:
            x = module(x)

        return x

    def backward(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Backward pass through the network.

        Computes gradients for all parameters using backpropagation and applies
        L2 regularization gradients to layers with weights.

        Args:
            preds: Model predictions of shape (batch_size, output_dim).
            labels: True labels of shape (batch_size,) with integer class indices.

        Returns:
            Input gradients of shape (batch_size, input_dim).
        """
        grads = self.loss.backward(preds, labels)
        for module in reversed(self.model):
            grads = module.backward(grads)

            # inject regularization gradient if the module has weights
            if self.use_regularization and hasattr(module, "weights"):
                grads_w = getattr(module, "_w_grads", None)
                if grads_w is not None:
                    module._w_grads += 2 * self.regularization_lambda * module.weights

        return grads

    def compute_loss(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute the total loss including regularization.

        Calculates cross-entropy loss between predictions and true labels,
        plus L2 regularization penalty if enabled.

        Args:
            preds: Model predictions of shape (batch_size, output_dim).
            labels: True labels of shape (batch_size,) with integer class indices.

        Returns:
            Total loss value (cross-entropy + L2 regularization).
        """
        reg = 0.0
        if self.use_regularization:
            for module in self.model:
                if hasattr(module, "weights"):
                    # L2 Regularization
                    reg += self.regularization_lambda * np.sum(module.weights**2)

        return self.loss(preds, labels) + reg

    def parameter_update(self, optimizer: str = "sgd", **kwargs):
        """Update parameters for all modules in the network.

        Args:
            optimizer: Optimization algorithm name (e.g. "sgd", "adam").
            **kwargs: Optimizer-specific parameters (e.g. lr, beta1, beta2, eps).
        """
        for module in self.model:
            if hasattr(module, "parameter_update"):
                module.parameter_update(optimizer=optimizer, **kwargs)

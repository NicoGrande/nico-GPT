"""Linear (fully connected) layer implementation using NumPy.

This module provides a Linear layer class that performs the fundamental
linear transformation y = xW^T + b with support for multiple weight
initialization methods and optimization algorithms including SGD and Adam.

Classes:
    Linear: Fully connected linear transformation layer.

Example:
    >>> import numpy as np
    >>> from linear import Linear

    >>> # Create a linear layer
    >>> layer = Linear(in_features=784, out_features=128)
    >>>
    >>> # Forward pass
    >>> x = np.random.randn(32, 784)  # batch_size=32, input_dim=784
    >>> output = layer.forward(x)  # shape: (32, 128)
    >>>
    >>> # Backward pass
    >>> grad_output = np.random.randn(32, 128)
    >>> grad_input = layer.backward(grad_output)
    >>>
    >>> # Parameter update
    >>> layer.parameter_update(lr=0.001, optimizer="adam")
"""

import numpy as np
from modules.numpy.optimizers import SGD, Adam, AdaGrad, RMSProp, AdaDelta, SGDMomentum


class Linear:
    """Fully connected linear layer implementation.

    Performs the linear transformation y = xW^T + b where W is the weight matrix
    and b is the bias vector. Supports He and Glorot weight initialization methods
    and SGD/Adam optimizers.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include bias terms (default: True).
        init_method: Weight initialization method, either "He" or "Glorot" (default: "He").

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether bias is enabled.
        weights: Weight matrix of shape (out_features, in_features).
        biases: Bias vector of shape (out_features,).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_method: str = "He",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if init_method == "He":
            self.weights = np.random.normal(
                0, np.sqrt(2 / in_features), (out_features, in_features)
            )
        elif init_method == "Glorot":
            d_avg = (in_features + out_features) / 2
            self.weights = np.random.normal(
                0, np.sqrt(1 / d_avg), (out_features, in_features)
            )
        else:
            raise ValueError(
                "Invalid weight initialization method. Valid methods include 'He' and 'Glorot'."
            )

        # Initialize biases (out_features)
        self.biases = np.zeros(out_features)

        # Optimizers for weights and biases
        self._weight_optimizer = None
        self._bias_optimizer = None

        # Placeholder to store input for back-propagation
        self._last_input: np.ndarray | None = None
        self._w_grads: np.ndarray | None = None
        self._b_grads: np.ndarray | None = None

    def __repr__(self) -> str:
        """Returns string representation of the Linear layer.

        Returns:
            String representation showing layer dimensions and bias setting.
        """
        return f"Linear({self.in_features}, {self.out_features}, bias={self.use_bias})"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Makes the Linear layer callable.

        Args:
            x: Input array of shape (batch_size, in_features).

        Returns:
            Output array of shape (batch_size, out_features).
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the linear layer.

        Computes the linear transformation y = xW^T + b.

        Args:
            x: Input array of shape (batch_size, in_features).

        Returns:
            Output array of shape (batch_size, out_features).
        """
        # Save input for use during the backward pass
        self._last_input = x

        # y = x W^T + b
        out = x @ self.weights.T + self.biases
        return out

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass for the linear layer.

        Computes gradients with respect to weights, biases, and inputs using
        the chain rule of backpropagation.

        Args:
            grads: Gradient of loss w.r.t. output, shape (batch_size, out_features).

        Returns:
            Gradient of loss w.r.t. input, shape (batch_size, in_features).

        Raises:
            ValueError: If forward() hasn't been called before backward().
        """
        if self._last_input is None:
            raise ValueError("forward must be called before backward.")

        # (D_out x B) x (B x D_in) -> (D_out x D_in)
        d_W = grads.T @ self._last_input

        # (B x D_out) -> (D_out)
        d_b = grads.sum(axis=0)

        # Save values for weight updates
        self._w_grads = d_W
        self._b_grads = d_b

        # (B x D_out) * (D_out x D_in) -> (B x D_in)
        d_x = grads @ self.weights

        return d_x

    def parameter_update(self, optimizer: str = "sgd", **kwargs):
        """Update parameters using the specified optimization algorithm.

        Creates optimizer instances on first call and reuses them for subsequent updates.
        The same optimizer type is used for both weights and biases.

        Args:
            optimizer: Optimization algorithm name ("sgd", "adam", "adagrad", "rmsprop", "adadelta", "sgd_momentum").
            **kwargs: Optimizer-specific parameters (e.g., lr, beta1, beta2, eps).

        Raises:
            ValueError: If backward() hasn't been called or unsupported optimizer specified.
        """
        if self._w_grads is None or self._b_grads is None:
            raise ValueError(
                "parameter_update called before backward – gradients are missing."
            )

        # Initialize optimizers on first call
        if self._weight_optimizer is None:
            optimizer_map = {
                "sgd": SGD,
                "adam": Adam,
                "adagrad": AdaGrad,
                "rmsprop": RMSProp,
                "adadelta": AdaDelta,
                "sgd_momentum": SGDMomentum
            }
            
            if optimizer.lower() not in optimizer_map:
                raise ValueError(f"Unsupported optimizer '{optimizer}'.")
            
            optimizer_class = optimizer_map[optimizer.lower()]
            self._weight_optimizer = optimizer_class(**kwargs)
            self._bias_optimizer = optimizer_class(**kwargs)

        # Update weights using the weight optimizer
        self.weights = self._weight_optimizer.update(self.weights, self._w_grads)
        
        # Update biases using the bias optimizer (if bias is enabled)
        if self.use_bias:
            self.biases = self._bias_optimizer.update(self.biases, self._b_grads)

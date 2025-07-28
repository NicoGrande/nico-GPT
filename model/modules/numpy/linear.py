"""Implementation of a linear layer in numpy."""

import numpy as np


class Linear:
    """Linear layer in numpy.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool): Whether to use bias. Defaults to True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # initialize weights and biases (out_features x in_features)
        self.weights = np.random.randn(out_features, in_features)

        # initialize biases (out_features)
        self.biases = np.random.randn(out_features)

        # placeholder to store input for back-propagation
        self._last_input: np.ndarray | None = None
        self._w_grads: np.ndarray | None = None
        self._b_grads: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"Linear({self.in_features}, {self.out_features}, bias={self.use_bias})"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the linear layer.

        Args:
            x (np.ndarray): The input tensor of shape (batch_size, seq_len, in_features).
        """
        # Save input for use during the backward pass
        self._last_input = x

        # y = x W^T + b
        out = x @ self.weights.T + self.biases
        return out

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass for the linear layer.

        Args:
            grads (np.ndarray): Gradient of the loss with respect to the output of this layer
                                (same shape as the output tensor).
        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        if self._last_input is None:
            raise ValueError("forward must be called before backward.")

        # (D_in x B) x (B x D_out) -> (D_in x D_out)
        d_W = self._last_input.T @ grads

        # (B x D_out) -> (D_out)
        d_b = grads.sum(axis=0)

        # Save values for weight updates
        self._w_grads = d_W
        self._b_grads = d_b

        # (B x D_out) * (D_out x D_in) -> (B x D_in)
        d_x = grads @ self.weights

        return d_x

    def parameter_update(self, lr: float):
        """Performs Stochastic Gradient Descent Update.

        Args:
            lr (float): The learning rate used to upate model params
        """

        self.weights -= lr * self._w_grads
        self.biases -= lr * self._b_grads

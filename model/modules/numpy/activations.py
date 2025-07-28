import abc
import numpy as np


class ActivationBase(abc.ABC):
    def __init__(self, **kwargs):
        """Initialize the ActivationBase object"""
        super().__init__()

    def __call__(self, x: np.ndarray):
        """Apply the activation function to an input"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    @abc.abstractmethod
    def forward(self, x: np.ndarray):
        """Apply the activation function to an input"""
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, grads: np.ndarray, **kwargs):
        """Compute the gradient of the activation function w.r.t the input"""
        raise NotImplementedError


class ReLU(ActivationBase):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return "ReLU"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the ReLU activation.

        Args:
            x (np.ndarray): Input tensor of shape (B, D) or (D,) where B is the batch
                size and D is the feature dimension.

        Returns:
            np.ndarray: Tensor of the same shape as x with the ReLU function applied
                element-wise.
        """
        return x * (x > 0)

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """ReLU backward pass.

        Args:
            grads (np.ndarray): Upstream gradients dL/dy of shape matching the forward
                input (B, D) or (D,).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input dL/dx with the
                same shape as grads.
        """
        return 1.0 * (grads > 0)


class LeakyRelu(ActivationBase):
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the Leaky-ReLU activation.

        Args:
            x (np.ndarray): Input tensor of shape (B, D) or (D,).

        Returns:
            np.ndarray: Activated output with the same shape as x where negative values
                are scaled by alpha.
        """
        x_copy = x.copy()
        x_copy[x < 0] = x_copy[x < 0] * self.alpha
        return x_copy

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Leaky-ReLU backward pass.

        Args:
            grads (np.ndarray): Upstream gradients dL/dy of shape (B, D) or (D,).

        Returns:
            np.ndarray: dL/dx with the same shape as grads. Gradients flowing
                through negative activations are scaled by alpha.
        """
        g = np.ones_like(grads)
        g[grads < 0] = self.alpha
        return g


class Sigmoid(ActivationBase):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return "Sigmoid"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the Sigmoid activation.

        Args:
            x (np.ndarray): Input tensor of shape (B, D) or (D,).

        Returns:
            np.ndarray: Tensor of the same shape as x with the Sigmoid function applied
                element-wise.
        """
        return 1 / (1 + (np.exp(-x)))

    def backward(self, grads: np.ndarray):
        """Sigmoid backward pass.

        Args:
            grads (np.ndarray): Upstream gradients dL/dy of shape (B, D) or (D,).

        Returns:
            np.ndarray: dL/dx with the same shape as grads.
        """
        return (1 / (1 + (np.exp(-grads)))) * (1 - (1 / (1 + (np.exp(-grads)))))


class Softmax(ActivationBase):
    def __init__(self):
        super().__init__()
        self._last_sm = None

    def __repr__(self) -> str:
        return "Softmax"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the Softmax activation along the last axis.

        Args:
            x (np.ndarray): Input tensor of shape (B, D) where B is the batch size and
                D is the number of classes / features.

        Returns:
            np.ndarray: Softmax probabilities of shape (B, D) where each row sums to 1.
        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self._last_sm = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self._last_sm

    def backward(self, grads: np.ndarray):
        """Softmax backward pass.

        Args:
            grads (np.ndarray): Upstream gradients dL/dy of shape (B, D).

        Returns:
            np.ndarray: dL/dx of shape (B, D).
        """
        s = self._last_sm  # (B, D)
        dot = np.sum(grads * s, axis=-1, keepdims=True)  # (B, 1)
        return s * (grads - dot)

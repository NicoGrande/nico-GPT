"""Activation function implementations using NumPy.

This module provides implementations of common activation functions used in
neural networks, including ReLU, Leaky ReLU, Sigmoid, and Softmax. All
activations follow a consistent interface with forward and backward methods
for use in backpropagation.

Classes:
    ActivationBase: Abstract base class defining the activation interface.
    ReLU: Rectified Linear Unit activation function.
    LeakyRelu: Leaky ReLU activation with configurable negative slope.
    Sigmoid: Sigmoid activation function with numerical stability.
    Softmax: Softmax activation for probability distributions.

Example:
    >>> import numpy as np
    >>> from activations import ReLU, Sigmoid, Softmax

    >>> # ReLU activation
    >>> relu = ReLU()
    >>> x = np.array([[-1, 0, 1], [2, -3, 4]])
    >>> output = relu.forward(x)  # [[0, 0, 1], [2, 0, 4]]

    >>> # Sigmoid activation
    >>> sigmoid = Sigmoid()
    >>> output = sigmoid.forward(x)

    >>> # Softmax for classification
    >>> softmax = Softmax()
    >>> logits = np.array([[1, 2, 3], [1, 1, 1]])
    >>> probs = softmax.forward(logits)  # Each row sums to 1

Note:
    This implementation is for educational purposes and ML interview preparation.
    For production use, consider optimized frameworks like PyTorch or TensorFlow.
"""

import abc
import numpy as np


class ActivationBase(abc.ABC):
    """Abstract base class for activation functions.

    Defines the interface that all activation functions must implement,
    including forward pass computation and backward pass gradient computation.

    Attributes:
        Subclasses should define their own attributes as needed.
    """

    def __init__(self, **kwargs):
        """Initialize the activation function.

        Args:
            **kwargs: Additional keyword arguments for subclass initialization.
        """
        super().__init__()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Makes the activation function callable.

        Provides a convenient interface to the forward method with automatic
        shape handling for 1D inputs.

        Args:
            x: Input array of shape (batch_size, features) or (features,).

        Returns:
            Activated output with same shape as input (after normalization).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the activation function.

        Args:
            x: Input array of shape (batch_size, features).

        Returns:
            Activated output array of same shape as input.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, grads: np.ndarray, **kwargs) -> np.ndarray:
        """Backward pass of the activation function.

        Args:
            grads: Gradient of loss w.r.t. activation output.
            **kwargs: Additional arguments for specific activation functions.

        Returns:
            Gradient of loss w.r.t. activation input.
        """
        raise NotImplementedError


class ReLU(ActivationBase):
    """Rectified Linear Unit (ReLU) activation function.

    Applies the element-wise function ReLU(x) = max(0, x). ReLU is widely
    used in deep learning due to its simplicity and effectiveness in
    mitigating the vanishing gradient problem.

    Attributes:
        _last_input: Stores input from forward pass for backward computation.
    """

    def __init__(self):
        """Initialize ReLU activation function."""
        super().__init__()
        self._last_input = None

    def __repr__(self) -> str:
        """Returns string representation of ReLU.

        Returns:
            String representation of the activation function.
        """
        return "ReLU"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of ReLU activation.

        Applies ReLU(x) = max(0, x) element-wise.

        Args:
            x: Input array of shape (batch_size, features).

        Returns:
            Output array of same shape with ReLU applied element-wise.
        """
        # Store input for backward pass
        self._last_input = x
        return x * (x > 0)

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass of ReLU activation.

        Computes gradient using the derivative: d/dx ReLU(x) = 1 if x > 0, else 0.

        Args:
            grads: Upstream gradients of shape (batch_size, features).

        Returns:
            Input gradients of same shape as grads.

        Raises:
            ValueError: If forward() hasn't been called before backward().
        """
        if self._last_input is None:
            raise ValueError("forward must be called before backward.")
        # Derivative of ReLU: 1 where input > 0, 0 elsewhere
        return grads * (self._last_input > 0)


class LeakyRelu(ActivationBase):
    """Leaky Rectified Linear Unit (Leaky ReLU) activation function.

    Applies the element-wise function: LeakyReLU(x) = max(αx, x) where α is
    a small positive constant. This helps mitigate the "dying ReLU" problem
    by allowing small gradients to flow through negative inputs.

    Args:
        alpha: Negative slope coefficient (default: 0.01).

    Attributes:
        alpha: The negative slope parameter.
        _last_input: Stores input from forward pass for backward computation.
    """

    def __init__(self, alpha: float = 0.01):
        """Initialize Leaky ReLU activation function.

        Args:
            alpha: Slope for negative values, typically small (default: 0.01).
        """
        super().__init__()
        self.alpha = alpha
        self._last_input = None

    def __repr__(self) -> str:
        """Returns string representation of Leaky ReLU.

        Returns:
            String representation showing the alpha parameter.
        """
        return f"LeakyReLU(alpha={self.alpha})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of Leaky ReLU activation.

        Applies LeakyReLU(x) = max(αx, x) element-wise.

        Args:
            x: Input array of shape (batch_size, features).

        Returns:
            Output array where negative values are scaled by alpha.
        """
        self._last_input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass of Leaky ReLU activation.

        Computes gradient using derivative: d/dx LeakyReLU(x) = 1 if x > 0, else α.

        Args:
            grads: Upstream gradients of shape (batch_size, features).

        Returns:
            Input gradients where negative regions are scaled by alpha.

        Raises:
            ValueError: If forward() hasn't been called before backward().
        """
        if self._last_input is None:
            raise ValueError("forward must be called before backward.")

        return grads * np.where(self._last_input > 0, 1.0, self.alpha)


class Sigmoid(ActivationBase):
    """Sigmoid activation function with numerical stability.

    Applies the sigmoid function: σ(x) = 1 / (1 + e^(-x)). Uses numerically
    stable computation to avoid overflow for large positive and negative values.

    Attributes:
        _last_output: Stores sigmoid output for efficient backward computation.
    """

    def __init__(self):
        """Initialize Sigmoid activation function."""
        super().__init__()
        self._last_output = None

    def __repr__(self) -> str:
        """Returns string representation of Sigmoid.

        Returns:
            String representation of the activation function.
        """
        return "Sigmoid"

    def _positive_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid for positive values.

        Args:
            x: Input array with positive values.

        Returns:
            Sigmoid output for positive inputs.
        """
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid for negative values.

        Args:
            x: Input array with negative values.

        Returns:
            Sigmoid output for negative inputs.
        """
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of Sigmoid activation.

        Uses numerically stable computation to avoid overflow.

        Args:
            x: Input array of shape (batch_size, features).

        Returns:
            Sigmoid output in range (0, 1) with same shape as input.
        """
        positives = x > 0
        negatives = ~positives
        out = np.empty_like(x)
        out[positives] = self._positive_sigmoid(x[positives])
        out[negatives] = self._negative_sigmoid(x[negatives])
        self._last_output = out  # Store output for backward pass
        return out

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass of Sigmoid activation.

        Uses the property that d/dx σ(x) = σ(x)(1 - σ(x)) for efficiency.

        Args:
            grads: Upstream gradients of shape (batch_size, features).

        Returns:
            Input gradients of same shape as grads.

        Raises:
            ValueError: If forward() hasn't been called before backward().
        """
        if self._last_output is None:
            raise ValueError("forward must be called before backward.")

        # Use stored output for efficiency: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        return grads * self._last_output * (1 - self._last_output)


class Softmax(ActivationBase):
    """Softmax activation function for probability distributions.

    Applies the softmax function: softmax(x_i) = e^(x_i) / ∑_j e^(x_j).
    Produces a probability distribution where all outputs sum to 1.
    Uses numerical stability by subtracting the maximum value.

    Attributes:
        _last_output: Stores softmax output for efficient backward computation.
    """

    def __init__(self):
        """Initialize Softmax activation function."""
        super().__init__()
        self._last_output = None

    def __repr__(self) -> str:
        """Returns string representation of Softmax.

        Returns:
            String representation of the activation function.
        """
        return "Softmax"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of Softmax activation.

        Applies softmax along the last axis with numerical stability.

        Args:
            x: Input logits of shape (batch_size, num_classes).

        Returns:
            Probability distribution where each row sums to 1.
        """
        # Numerical stability: subtract max to prevent overflow
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self._last_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self._last_output

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass of Softmax activation.

        Uses the efficient vectorized formula: s * (grads - (grads • s))
        where s is the softmax output and • denotes dot product.

        Args:
            grads: Upstream gradients of shape (batch_size, num_classes).

        Returns:
            Input gradients of same shape as grads.

        Raises:
            ValueError: If forward() hasn't been called before backward().
        """
        if self._last_output is None:
            raise ValueError("forward must be called before backward.")

        s = self._last_output  # Softmax output
        # Efficient computation: s * (grads - sum(grads * s))
        dot_product = np.sum(grads * s, axis=-1, keepdims=True)
        return s * (grads - dot_product)

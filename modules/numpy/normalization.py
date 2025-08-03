"""Normalization layers implementation using NumPy.

This module provides implementations of common normalization techniques used in
deep learning, including batch normalization and layer normalization. These
layers help stabilize training, reduce internal covariate shift, and often
allow for higher learning rates.

Classes:
    BaseNormalization: Abstract base class defining the normalization interface.
    BatchNormalization: Normalizes across the batch dimension during training.
    LayerNormalization: Normalizes across the feature dimension for each sample.

Example:
    >>> import numpy as np
    >>> from normalization import BatchNormalization, LayerNormalization

    >>> # Batch normalization example
    >>> x = np.random.randn(32, 64)  # batch_size=32, features=64
    >>> bn = BatchNormalization(feature_dim=64)
    >>> output = bn.forward(x)

    >>> # Layer normalization example
    >>> ln = LayerNormalization(feature_dim=64)
    >>> output = ln.forward(x)

Note:
    This implementation is for educational purposes and ML interview preparation.
    For production use, consider optimized frameworks like PyTorch or TensorFlow.
"""

import abc
import numpy as np
from modules.numpy.optimizers import SGD, Adam, AdaGrad, RMSProp, AdaDelta, SGDMomentum


class BaseNormalization(abc.ABC):
    """Abstract base class for normalization layers.

    This class defines the interface for normalization layers including
    batch normalization and layer normalization. All concrete implementations
    must implement forward, backward, and update methods.

    Attributes:
        feature_dim: The feature dimension size.
        gamma: Learnable scale parameter.
        beta: Learnable shift parameter.
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.gamma = None
        self.beta = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Makes the normalization layer callable.

        Args:
            x: Input array of shape (batch_size, feature_dim).

        Returns:
            Normalized output array.
        """
        return self.forward(x)

    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the normalization layer.

        Args:
            x: Input array of shape (batch_size, feature_dim).

        Returns:
            Normalized output array.
        """
        pass

    @abc.abstractmethod
    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass of the normalization layer.

        Args:
            grads: Gradient of loss with respect to output.

        Returns:
            Gradient of loss with respect to input.
        """
        pass

    @abc.abstractmethod
    def parameter_update(self, lr: float, optimizer: str = "sgd", **kwargs) -> None:
        """Update learnable parameters using computed gradients.

        Args:
            lr: Learning rate for parameter updates.
            optimizer: Optimization algorithm to use (default: "sgd").
            **kwargs: Additional optimizer-specific parameters.
        """
        pass


class BatchNormalization(BaseNormalization):
    """Batch normalization layer implementation.

    Normalizes inputs across the batch dimension. During training, computes
    statistics from the current batch and maintains running statistics for
    inference. Each feature is normalized independently.

    Args:
        feature_dim: The feature dimension size.
        momentum: Momentum for updating running statistics (default: 0.9).

    Attributes:
        gamma: Learnable scale parameter of shape (feature_dim,).
        beta: Learnable shift parameter of shape (feature_dim,).
        eps: Small constant for numerical stability.
        momentum: Momentum for exponential moving average.
        running_mean: Running mean for inference.
        running_var: Running variance for inference.
    """

    def __init__(self, feature_dim: int, momentum: float = 0.9):
        super().__init__(feature_dim)

        self.gamma = np.ones(feature_dim)
        self.beta = np.zeros(feature_dim)
        self.eps = 1e-8

        # Keep running statistics for inference time
        self.momentum = momentum
        self.running_mean = np.zeros(feature_dim)
        self.running_var = np.ones(feature_dim)

        # Keep variables needed for gradient computation
        self._x = None
        self._x_norm = None
        self._batch_mean = None
        self._batch_var = None
        self._d_gamma = None
        self._d_beta = None
        
        # Optimizers for gamma and beta
        self._gamma_optimizer = None
        self._beta_optimizer = None

    def forward(self, x: np.ndarray, inference: bool = False) -> np.ndarray:
        """Forward pass of batch normalization.

        Args:
            x: Input array of shape (batch_size, feature_dim).
            inference: If True, use running statistics; if False, use batch statistics.

        Returns:
            Normalized output array of shape (batch_size, feature_dim).
        """
        if inference:
            x_norm = (x - self.running_mean) / (np.sqrt(self.running_var + self.eps))

        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Update running statistics for inference time
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            x_norm = (x - batch_mean) / (np.sqrt(batch_var + self.eps))

            # Store variables needed for backward pass
            self._x = x
            self._x_norm = x_norm
            self._batch_mean = batch_mean
            self._batch_var = batch_var

        return self.gamma * x_norm + self.beta

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass of batch normalization.

        Computes gradients with respect to input, accounting for the fact that
        each input affects the batch statistics used to normalize all samples.

        Args:
            grads: Gradient of loss with respect to output, shape (batch_size, feature_dim).

        Returns:
            Gradient of loss with respect to input, shape (batch_size, feature_dim).
        """
        # Gradients with respect to gamma and beta
        self._d_gamma = np.sum(self._x_norm * grads, axis=0)
        self._d_beta = np.sum(grads, axis=0)

        # Standard deviation
        std = np.sqrt(self._batch_var + self.eps)

        # Gradient with respect to normalized input
        d_x_norm = grads * self.gamma

        # Gradient with respect to variance
        d_var = (
            np.sum(d_x_norm * (self._x - self._batch_mean), axis=0)
            * (-0.5)
            * (self._batch_var + self.eps) ** (-1.5)
        )

        # Gradient with respect to mean
        d_mean = (
            np.sum(d_x_norm * (-1.0 / std), axis=0)
            + d_var
            * np.sum(-2.0 * (self._x - self._batch_mean), axis=0)
            / grads.shape[0]
        )

        # Final gradient with respect to input
        d_x = (
            (d_x_norm / std)
            + (d_var * 2.0 * (self._x - self._batch_mean) / grads.shape[0])
            + (d_mean / grads.shape[0])
        )

        return d_x

    def parameter_update(self, optimizer: str = "sgd", **kwargs) -> None:
        """Update gamma and beta parameters using computed gradients.

        Creates optimizer instances on first call and reuses them for subsequent updates.
        The same optimizer type is used for both gamma and beta parameters.

        Args:
            optimizer: Optimization algorithm name ("sgd", "adam", "adagrad", "rmsprop", "adadelta", "sgd_momentum").
            **kwargs: Optimizer-specific parameters (e.g., lr, beta1, beta2, eps).

        Raises:
            ValueError: If unsupported optimizer is specified.
        """
        # Initialize optimizers on first call
        if self._gamma_optimizer is None:
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
            self._gamma_optimizer = optimizer_class(**kwargs)
            self._beta_optimizer = optimizer_class(**kwargs)

        # Update gamma and beta using their respective optimizers
        self.gamma = self._gamma_optimizer.update(self.gamma, self._d_gamma)
        self.beta = self._beta_optimizer.update(self.beta, self._d_beta)


class LayerNormalization(BaseNormalization):
    """Layer normalization implementation.

    Normalizes inputs across the feature dimension for each sample independently.
    Unlike batch normalization, statistics are computed per sample rather than
    across the batch, making it more suitable for variable sequence lengths.

    Args:
        feature_dim: The feature dimension size.

    Attributes:
        gamma: Learnable scale parameter of shape (feature_dim,).
        beta: Learnable shift parameter of shape (feature_dim,).
        eps: Small constant for numerical stability.
    """

    def __init__(self, feature_dim: int):
        super().__init__(feature_dim)

        self.gamma = np.ones(feature_dim)
        self.beta = np.zeros(feature_dim)
        self.eps = 1e-8

        # Keep variables needed for gradient computation
        self._x = None
        self._x_norm = None
        self._layer_mean = None
        self._layer_var = None
        self._d_gamma = None
        self._d_beta = None
        
        # Optimizers for gamma and beta
        self._gamma_optimizer = None
        self._beta_optimizer = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of layer normalization.

        Args:
            x: Input array of shape (batch_size, feature_dim).

        Returns:
            Normalized output array of shape (batch_size, feature_dim).
        """
        layer_mean = np.mean(x, axis=1, keepdims=True)
        layer_var = np.var(x, axis=1, keepdims=True)

        x_norm = (x - layer_mean) / (np.sqrt(layer_var + self.eps))

        # Store variables needed for backward pass
        self._x = x
        self._x_norm = x_norm
        self._layer_mean = layer_mean
        self._layer_var = layer_var

        return self.gamma * x_norm + self.beta

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass of layer normalization.

        Computes gradients with respect to input. Since each sample is normalized
        independently, the gradients are simpler than batch normalization.

        Args:
            grads: Gradient of loss with respect to output, shape (batch_size, feature_dim).

        Returns:
            Gradient of loss with respect to input, shape (batch_size, feature_dim).
        """
        # Gradients with respect to gamma and beta accumulated over batch
        self._d_gamma = np.sum(self._x_norm * grads, axis=0)
        self._d_beta = np.sum(grads, axis=0)

        # Standard deviation
        std = np.sqrt(self._layer_var + self.eps)

        # Gradient with respect to normalized input
        d_x_norm = grads * self.gamma

        # Gradient with respect to variance
        d_var = (
            np.sum(d_x_norm * (self._x - self._layer_mean), axis=1, keepdims=True)
            * (-0.5)
            * (self._layer_var + self.eps) ** (-1.5)
        )

        # Gradient with respect to mean
        d_mean = (
            np.sum(d_x_norm * (-1.0 / std), axis=1, keepdims=True)
            + d_var
            * np.sum(-2.0 * (self._x - self._layer_mean), axis=1, keepdims=True)
            / self.feature_dim
        )

        # Final gradient with respect to input
        d_x = (
            (d_x_norm / std)
            + (d_var * 2.0 * (self._x - self._layer_mean) / self.feature_dim)
            + (d_mean / self.feature_dim)
        )

        return d_x

    def parameter_update(self, optimizer: str = "sgd", **kwargs) -> None:
        """Update gamma and beta parameters using computed gradients.

        Creates optimizer instances on first call and reuses them for subsequent updates.
        The same optimizer type is used for both gamma and beta parameters.

        Args:
            optimizer: Optimization algorithm name ("sgd", "adam", "adagrad", "rmsprop", "adadelta", "sgd_momentum").
            **kwargs: Optimizer-specific parameters (e.g., lr, beta1, beta2, eps).

        Raises:
            ValueError: If unsupported optimizer is specified.
        """
        # Initialize optimizers on first call
        if self._gamma_optimizer is None:
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
            self._gamma_optimizer = optimizer_class(**kwargs)
            self._beta_optimizer = optimizer_class(**kwargs)

        # Update gamma and beta using their respective optimizers
        self.gamma = self._gamma_optimizer.update(self.gamma, self._d_gamma)
        self.beta = self._beta_optimizer.update(self.beta, self._d_beta)

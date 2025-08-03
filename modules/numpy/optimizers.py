"""Optimizer implementations for gradient-based learning algorithms.

This module provides various optimization algorithms commonly used in machine learning,
including SGD, Adam, AdaGrad, RMSProp, and others. All optimizers inherit from a common
BaseOptimizer interface and can be used interchangeably.
"""

import abc
import numpy as np


class BaseOptimizer(abc.ABC):
    """Abstract base class for all optimizers.
    
    This class defines the interface that all optimizers must implement.
    Optimizers are responsible for updating model parameters based on gradients.
    """
    
    def __init__(self):
        """Initialize the optimizer.
        
        Sets the time step counter to 0.
        """
        self.time_step = 0

    @abc.abstractmethod
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters based on gradients.
        
        Args:
            params: Current parameter values.
            gradients: Gradients with respect to the parameters.
            
        Returns:
            Updated parameter values.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset the optimizer's internal state.
        
        This should be called when starting training from scratch or
        when resetting the optimizer between training runs.
        """
        pass


class SGD(BaseOptimizer):
    """Stochastic Gradient Descent optimizer.
    
    Implements the basic SGD algorithm: params = params - lr * gradients.
    """
    
    def __init__(self, lr: float = 1e-4):
        """Initialize SGD optimizer.
        
        Args:
            lr: Learning rate (step size). Defaults to 1e-4.
        """
        super().__init__()
        self.lr = lr

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using SGD rule.
        
        Args:
            params: Current parameter values.
            gradients: Gradients with respect to the parameters.
            
        Returns:
            Updated parameter values.
        """
        self.time_step += 1
        return params - self.lr * gradients

    def reset(self):
        """Reset the optimizer state.
        
        Resets the time step counter to 0.
        """
        self.time_step = 0


class SGDMomentum(BaseOptimizer):
    """SGD with momentum optimizer.
    
    Implements SGD with momentum to accelerate training and reduce oscillations.
    The momentum term accumulates gradients from previous steps.
    """
    
    def __init__(self, lr: float = 1e-4, beta: float = 0.9):
        """Initialize SGD with momentum optimizer.
        
        Args:
            lr: Learning rate (step size). Defaults to 1e-4.
            beta: Momentum coefficient. Defaults to 0.9.
        """
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.velocity = None

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using SGD with momentum.
        
        Args:
            params: Current parameter values.
            gradients: Gradients with respect to the parameters.
            
        Returns:
            Updated parameter values.
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.time_step += 1
        self.velocity = self.beta * self.velocity + gradients
        return params - self.lr * self.velocity

    def reset(self):
        """Reset the optimizer state.
        
        Resets the time step counter and velocity to initial values.
        """
        self.time_step = 0
        self.velocity = None


class AdaGrad(BaseOptimizer):
    """AdaGrad optimizer.
    
    Adapts the learning rate for each parameter based on the historical gradients.
    Parameters that receive large gradients will have their learning rate reduced,
    while parameters with small gradients will have their learning rate increased.
    """
    
    def __init__(self, lr: float = 1e-4, eps=1e-8):
        """Initialize AdaGrad optimizer.
        
        Args:
            lr: Initial learning rate. Defaults to 1e-4.
            eps: Small constant to prevent division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.gradient_accumulation = None

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using AdaGrad algorithm.
        
        Args:
            params: Current parameter values.
            gradients: Gradients with respect to the parameters.
            
        Returns:
            Updated parameter values.
        """
        if self.gradient_accumulation is None:
            self.gradient_accumulation = np.zeros_like(params)

        self.time_step += 1
        self.gradient_accumulation += gradients**2
        return params - (self.lr * gradients) / (
            np.sqrt(self.gradient_accumulation) + self.eps
        )

    def reset(self):
        """Reset the optimizer state.
        
        Resets the time step counter and gradient accumulation to initial values.
        """
        self.time_step = 0
        self.gradient_accumulation = None


class RMSProp(BaseOptimizer):
    """RMSProp optimizer.
    
    Maintains a moving average of the squared gradients to normalize the gradient.
    This helps to resolve AdaGrad's radically diminishing learning rates by using
    an exponentially decaying average instead of accumulating all past gradients.
    """
    
    def __init__(self, lr: float = 1e-4, beta: float = 0.9, eps=1e-8):
        """Initialize RMSProp optimizer.
        
        Args:
            lr: Learning rate. Defaults to 1e-4.
            beta: Decay rate for the moving average. Defaults to 0.9.
            eps: Small constant to prevent division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.gradient_accumulation = None

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using RMSProp algorithm.
        
        Args:
            params: Current parameter values.
            gradients: Gradients with respect to the parameters.
            
        Returns:
            Updated parameter values.
        """
        if self.gradient_accumulation is None:
            self.gradient_accumulation = np.zeros_like(params)

        self.time_step += 1
        self.gradient_accumulation = (
            self.beta * self.gradient_accumulation + (1 - self.beta) * gradients**2
        )
        return params - (self.lr * gradients) / (
            np.sqrt(self.gradient_accumulation) + self.eps
        )

    def reset(self):
        """Reset the optimizer state.
        
        Resets the time step counter and gradient accumulation to initial values.
        """
        self.time_step = 0
        self.gradient_accumulation = None


class AdaDelta(BaseOptimizer):
    """AdaDelta optimizer.
    
    An extension of AdaGrad that seeks to reduce its aggressive, monotonically
    decreasing learning rate. Instead of accumulating all past squared gradients,
    AdaDelta restricts the window of accumulated past gradients to some fixed size.
    """
    
    def __init__(self, beta: float = 0.9, eps=1e-8):
        """Initialize AdaDelta optimizer.
        
        Args:
            beta: Decay rate for the moving averages. Defaults to 0.9.
            eps: Small constant to prevent division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.gradient_accumulation = None
        self.update_accumulation = None

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using AdaDelta algorithm.
        
        Args:
            params: Current parameter values.
            gradients: Gradients with respect to the parameters.
            
        Returns:
            Updated parameter values.
        """
        if self.gradient_accumulation is None:
            self.gradient_accumulation = np.zeros_like(params)

        if self.update_accumulation is None:
            self.update_accumulation = np.zeros_like(params)

        self.time_step += 1
        self.gradient_accumulation = (
            self.beta * self.gradient_accumulation + (1 - self.beta) * gradients**2
        )

        rms_gradients = np.sqrt(self.gradient_accumulation + self.eps)
        rms_updates = np.sqrt(self.update_accumulation + self.eps)
        param_update = rms_updates * gradients / rms_gradients

        self.update_accumulation = (
            self.beta * self.update_accumulation + (1 - self.beta) * param_update**2
        )

        return params - param_update

    def reset(self):
        """Reset the optimizer state.
        
        Resets the time step counter and all accumulation arrays to initial values.
        """
        self.time_step = 0
        self.gradient_accumulation = None
        self.update_accumulation = None


class Adam(BaseOptimizer):
    """Adam optimizer.
    
    Combines the advantages of two other extensions of stochastic gradient descent:
    AdaGrad and RMSProp. Adam computes adaptive learning rates for each parameter
    by keeping an exponentially decaying average of past gradients and squared gradients.
    """
    
    def __init__(
        self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps=1e-8
    ):
        """Initialize Adam optimizer.
        
        Args:
            lr: Learning rate. Defaults to 1e-3.
            beta1: Decay rate for the first moment estimates. Defaults to 0.9.
            beta2: Decay rate for the second moment estimates. Defaults to 0.999.
            eps: Small constant to prevent division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using Adam algorithm.
        
        Args:
            params: Current parameter values.
            gradients: Gradients with respect to the parameters.
            
        Returns:
            Updated parameter values.
        """
        if self.m is None:
            self.m = np.zeros_like(params)

        if self.v is None:
            self.v = np.zeros_like(params)

        self.time_step += 1
        
        # Calculate first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2

        # Apply bias correction
        m_h = self.m / (1 - self.beta1**self.time_step)
        v_h = self.v / (1 - self.beta2**self.time_step)

        return params - (self.lr * m_h) / (np.sqrt(v_h) + self.eps)

    def reset(self):
        """Reset the optimizer state.
        
        Resets the time step counter and moment estimates to initial values.
        """
        self.time_step = 0
        self.m = None
        self.v = None

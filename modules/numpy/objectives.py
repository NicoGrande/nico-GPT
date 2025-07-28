import abc
import enum
import numpy as np


class Reduction(enum.Enum):
    """Enum for specifying reduction methods in loss functions.

    Attributes:
        MEAN: Average the loss over all elements.
        SUM: Sum the loss over all elements.
    """

    MEAN = "mean"
    SUM = "sum"


class ObjectiveBase(abc.ABC):
    """Abstract base class for all objective functions.

    This class defines the interface that all objective functions must implement,
    including forward pass (loss computation) and backward pass (gradient computation).
    """

    def __init__(self):
        """Initialize the ObjectiveBase object."""
        super().__init__()

    def __call__(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Apply the objective function to compute loss.

        This method serves as a convenient interface to the forward method,
        allowing the loss function to be called directly like a function.

        Args:
            preds (np.ndarray): Model predictions as a numpy array.
            labels (np.ndarray): Ground truth labels as a numpy array.

        Returns:
            float: The computed loss value.
        """
        return self.forward(preds, labels)

    @abc.abstractmethod
    def forward(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Apply the objective function to compute loss.

        Args:
            preds: Model predictions as a numpy array.
            labels: Ground truth labels as a numpy array.

        Returns:
            The computed loss value as a float.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function w.r.t the predictions.

        Args:
            preds: Model predictions as a numpy array.
            labels: Ground truth labels as a numpy array.

        Returns:
            Gradient of the loss with respect to predictions as a numpy array.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class MAELoss(ObjectiveBase):
    """Mean Absolute Error (L1) loss function.

    Computes the mean absolute error between predictions and labels:
    MAE = mean(|labels - preds|)
    """

    def __init__(self):
        """Initialize the MAELoss object."""
        super().__init__()

    def __repr__(self) -> str:
        """Return string representation of the MAELoss object.

        Returns:
            String representation of the loss function.
        """
        return "MAELoss"

    def forward(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute the Mean Absolute Error loss.

        Args:
            preds: Model predictions as a numpy array.
            labels: Ground truth labels as a numpy array.

        Returns:
            The computed MAE loss value as a float.
        """
        return np.mean(np.abs(labels - preds))

    def backward(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute the gradient of MAE loss w.r.t predictions.

        Args:
            preds: Model predictions as a numpy array.
            labels: Ground truth labels as a numpy array.

        Returns:
            Gradient of MAE loss with respect to predictions as a numpy array.
            The gradient is the sign of (preds - labels) / N.
        """
        return np.sign(preds - labels) / preds.size


class MSELoss(ObjectiveBase):
    """Mean Squared Error (L2) loss function.

    Computes the mean squared error between predictions and labels:
    MSE = mean((labels - preds)^2)
    """

    def __init__(self):
        """Initialize the MSELoss object."""
        super().__init__()

    def __repr__(self) -> str:
        """Return string representation of the MSELoss object.

        Returns:
            String representation of the loss function.
        """
        return "MSELoss"

    def forward(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute the Mean Squared Error loss.

        Args:
            preds: Model predictions as a numpy array.
            labels: Ground truth labels as a numpy array.

        Returns:
            The computed MSE loss value as a float.
        """
        return np.mean((labels - preds) ** 2)

    def backward(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute the gradient of MSE loss w.r.t predictions.

        Args:
            preds: Model predictions as a numpy array.
            labels: Ground truth labels as a numpy array.

        Returns:
            Gradient of MSE loss with respect to predictions as a numpy array.
            The gradient is 2 * (preds - labels) / N.
        """
        return 2 * (preds - labels) / preds.size


class CrossEntropyLoss(ObjectiveBase):
    """Cross-entropy loss function for classification tasks.

    Computes the cross-entropy loss between predicted probabilities and true labels:
    CE = -sum(labels * log(preds)) / denominator

    The denominator depends on the reduction method:
    - 'mean': divides over batch size (preds.shape[0])
    - 'sum': no division (denominator = 1)
    """

    def __init__(self, reduction: str | Reduction = "mean"):
        """Initialize the CrossEntropyLoss object.

        Args:
            reduction: Method for reducing the loss across samples. Can be 'mean', 'sum',
                or a Reduction enum value. Defaults to 'mean'.
        """
        super().__init__()
        if isinstance(reduction, str):
            self._reduction = Reduction(reduction)
        else:
            self._reduction = reduction

    def __repr__(self) -> str:
        """Return string representation of the CrossEntropyLoss object.

        Returns:
            String representation of the loss function.
        """
        return "CrossEntropyLoss"

    def forward(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute the cross-entropy loss.

        Args:
            preds: Model predictions (probabilities) as a numpy array.
            labels: Ground truth labels as a numpy array.

        Returns:
            The computed cross-entropy loss value as a float.
        """
        denominator = (
            1 if self._reduction == Reduction.SUM else preds.shape[0]
        )  # batch size
        eps = 1e-12
        return -np.sum(labels * np.log(np.clip(preds, eps, 1.0))) / denominator

    def backward(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute the gradient of cross-entropy loss w.r.t predictions.

        Args:
            preds: Model predictions (probabilities) as a numpy array.
            labels: Ground truth labels as a numpy array.

        Returns:
            Gradient of cross-entropy loss with respect to predictions as a numpy array.
            The gradient is -labels / (preds * denominator).
        """
        denominator = (
            1 if self._reduction == Reduction.SUM else preds.shape[0]
        )  # batch size
        eps = 1e-12
        return -labels / (np.clip(preds, eps, 1.0) * denominator)

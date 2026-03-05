import numpy as np

class Loss:
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError
        
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class MeanSquaredError(Loss):
    """
    MSE: L = (1/N) * SUM( (y_pred - y_true)^2 )
    """
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.power(y_true - y_pred, 2))
        
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        samples = y_true.shape[0]
        # DERIVATIVE: -2/N * (y_true - y_pred)
        return -2 * (y_true - y_pred) / samples

class BinaryCrossEntropy(Loss):
    """
    BCE: L = -(1/N) * SUM( y_true * log(y_pred) + (1-y_true) * log(1-y_pred) )
    """
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses)
        
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        samples = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # DERIVATIVE: -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        return -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped)) / samples

class CategoricalCrossEntropy(Loss):
    """
    CCE: L = -SUM( y_true * log(y_pred) )
    """
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        samples = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # HANDLE BOTH 1D (SPARSE) AND 2D (ONE-HOT) LABELS
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true.astype(int)]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)
        
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        samples = y_true.shape[0]
        
        # CONVERT SPARSE TO ONE-HOT IF NECESSARY
        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[range(samples), y_true.astype(int)] = 1
            y_true = y_true_one_hot
            
        # DERIVATIVE: -y_true / y_pred
        return -y_true / np.clip(y_pred, 1e-7, 1 - 1e-7) / samples
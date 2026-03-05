import numpy as np

def _format_labels(y_pred: np.ndarray, y_true: np.ndarray):
    """
    HELPER FUNCTION TO CONVERT PROBABILITIES/ONE-HOT TO DISCRETE CLASS LABELS.
    """
    y_pred_labels = y_pred
    y_true_labels = y_true
    
    # FORMAT PREDICTIONS
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    elif len(y_pred.shape) == 2:
        y_pred_labels = (y_pred > 0.5).astype(int)
        
    # FORMAT TRUE LABELS
    if len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
        
    return y_pred_labels.flatten(), y_true_labels.flatten()


# ==========================================
# CLASSIFICATION METRICS
# ==========================================

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred_labels, y_true_labels = _format_labels(y_pred, y_true)
    return np.mean(y_pred_labels == y_true_labels)

def precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred_labels, y_true_labels = _format_labels(y_pred, y_true)
    
    true_positives = np.sum((y_pred_labels == 1) & (y_true_labels == 1))
    false_positives = np.sum((y_pred_labels == 1) & (y_true_labels == 0))
    
    # ADD 1e-7 TO PREVENT DIVISION BY ZERO
    return true_positives / (true_positives + false_positives + 1e-7)

def recall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred_labels, y_true_labels = _format_labels(y_pred, y_true)
    
    true_positives = np.sum((y_pred_labels == 1) & (y_true_labels == 1))
    false_negatives = np.sum((y_pred_labels == 0) & (y_true_labels == 1))
    
    # ADD 1e-7 TO PREVENT DIVISION BY ZERO
    return true_positives / (true_positives + false_negatives + 1e-7)

def f1_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    
    # 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
    return 2 * (prec * rec) / (prec + rec + 1e-7)


# ==========================================
# REGRESSION METRICS
# ==========================================

def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # MEAN SQUARED ERROR
    return np.mean(np.power(y_true - y_pred, 2))
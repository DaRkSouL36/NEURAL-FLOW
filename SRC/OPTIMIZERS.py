class SGD:
    """
    STOCHASTIC/MINI-BATCH GRADIENT DESCENT OPTIMIZER.
    """
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
    def update(self, layer):
        # UPDATE RULE: W = W - ETA * dW
        if hasattr(layer, 'weights'):
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases
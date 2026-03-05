import numpy as np

class Dense:
    """
    FULLY CONNECTED NEURAL NETWORK LAYER.
    """
    def __init__(self, n_inputs: int, n_neurons: int, seed: int = 36):
        np.random.seed(seed)
        # HE/XAVIER INITIALIZATION SCALING
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        
        self.inputs = None
        self.dweights = None
        self.dbiases = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # STORE INPUTS FOR BACKPROPAGATION
        self.inputs = inputs
        # Z = W.X + B
        return np.dot(inputs, self.weights) + self.biases
        
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # GRADIENTS WITH RESPECT TO PARAMETERS
        self.dweights = np.dot(self.inputs.T, d_out)
        self.dbiases = np.sum(d_out, axis=0, keepdims=True)
        
        # GRADIENT WITH RESPECT TO INPUTS (TO PASS DOWN TO PREVIOUS LAYER)
        d_inputs = np.dot(d_out, self.weights.T)
        return d_inputs
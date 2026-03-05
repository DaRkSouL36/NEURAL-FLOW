import numpy as np

class Activation:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ReLU(Activation):
    """
    RECTIFIED LINEAR UNIT: f(x) = max(0, x)
    """
    def __init__(self):
        self.inputs = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)
        
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # DERIVATIVE IS 1 IF X > 0, ELSE 0
        d_inputs = d_out.copy()
        d_inputs[self.inputs <= 0] = 0
        return d_inputs

class Sigmoid(Activation):
    """
    SIGMOID ACTIVATION: f(x) = 1 / (1 + e^-x)
    """
    def __init__(self):
        self.output = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # CLIP FOR NUMERICAL STABILITY
        inputs = np.clip(inputs, -500, 500)
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
        
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # DERIVATIVE: f(x) * (1 - f(x))
        return d_out * self.output * (1 - self.output)

class Tanh(Activation):
    """
    HYPERBOLIC TANGENT ACTIVATION: f(x) = (e^x - e^-x) / (e^x + e^-x)
    """
    def __init__(self):
        self.output = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.tanh(inputs)
        return self.output
        
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # DERIVATIVE: 1 - f(x)^2
        return d_out * (1 - np.power(self.output, 2))

class Softmax(Activation):
    """
    SOFTMAX ACTIVATION FOR MULTI-CLASS CLASSIFICATION.
    f(x_i) = e^(x_i) / SUM(e^(x_j))
    """
    def __init__(self):
        self.output = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # SHIFT FOR NUMERICAL STABILITY (PREVENT EXP OVERFLOW)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
        
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # JACOBIAN MATRIX COMPUTATION FOR EACH SAMPLE IN BATCH
        d_inputs = np.empty_like(d_out)
        for i, (single_output, single_d_out) in enumerate(zip(self.output, d_out)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            d_inputs[i] = np.dot(jacobian_matrix, single_d_out)
        return d_inputs
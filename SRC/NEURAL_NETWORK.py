import numpy as np
from typing import List, Any

class NeuralNetwork:
    """
    CORE NEURAL NETWORK ORCHESTRATOR.
    """
    def __init__(self):
        self.layers: List[Any] = []
        self.loss_function = None
        self.optimizer = None
        
    def add(self, layer: Any):
        """
        ADDS A LAYER OR ACTIVATION TO THE NETWORK.
        """
        self.layers.append(layer)
        
    def compile(self, loss_function: Any, optimizer: Any):
        """
        COMPILES THE NETWORK WITH LOSS AND OPTIMIZER.
        """
        self.loss_function = loss_function
        self.optimizer = optimizer
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        EXECUTES FULL FORWARD PASS.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
        
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        EXECUTES FULL BACKWARD PASS USING CHAIN RULE.
        """
        # GET INITIAL GRADIENT FROM LOSS FUNCTION
        d_out = self.loss_function.backward(y_pred, y_true)
        
        # BACKPROPAGATE IN REVERSE ORDER
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
            
    def update_weights(self):
        """
        UPDATES WEIGHTS USING THE DEFINED OPTIMIZER.
        """
        for layer in self.layers:
            self.optimizer.update(layer)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        MAKES INFERENCE ON NEW DATA.
        """
        return self.forward(X)
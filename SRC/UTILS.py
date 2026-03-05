import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# SET RANDOM SEED FOR REPRODUCIBILITY
np.random.seed(36)

def generate_classification_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    GENERATES NON-LINEAR CIRCULAR/SPIRAL CLASSIFICATION DATA.
    """
    # GENERATE SPIRAL DATA
    N = n_samples // 2
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype='uint8')
    
    for j in range(2):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1.0, N)
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
        
    df = pd.DataFrame(X, columns=['FEATURE_1', 'FEATURE_2'])
    df['TARGET'] = y
    
    # SAVE TO CSV
    os.makedirs('DATA', exist_ok=True)
    df.to_csv('DATA/CLASSIFICATION_DATA.csv', index=False)
    return df

def generate_regression_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    GENERATES NON-LINEAR SINUSOIDAL REGRESSION DATA.
    """
    X = np.random.uniform(-5, 5, (n_samples, 1))
    # NON-LINEAR FUNCTION WITH NOISE
    y = np.sin(X) + 0.5 * X + np.random.normal(0, 0.2, (n_samples, 1))
    
    df = pd.DataFrame(np.hstack((X, y)), columns=['FEATURE_1', 'TARGET'])
    
    # SAVE TO CSV
    os.makedirs('DATA', exist_ok=True)
    df.to_csv('DATA/REGRESSION_DATA.csv', index=False)
    return df

def configure_plots():
    """
    SETS MATPLOTLIB PLOT CONFIGURATION TO STRICT REQUIREMENTS.
    """
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
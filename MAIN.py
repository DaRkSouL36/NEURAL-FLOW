import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# IMPORT CUSTOM MODULES FROM DIRECTORY STRUCTURE
from SRC.UTILS import generate_classification_data, configure_plots
from SRC.PREPROCESSING import StandardScaler, train_test_split
from SRC.LAYERS import Dense
from SRC.ACTIVATIONS import ReLU, Sigmoid
from SRC.LOSSES import BinaryCrossEntropy
from SRC.OPTIMIZERS import SGD
from SRC.NEURAL_NETWORK import NeuralNetwork
from SRC.METRICS import accuracy, precision, recall, f1_score
from EXPERIMENTS.TRAIN_MODEL import train_network

def main():
    # 1. INITIALIZE SETTINGS
    configure_plots()
    print("=== NEURAL-FLOW: FROM SCRATCH ANN IMPLEMENTATION ===\n")
    
    # 2. DATA GENERATION & PREPROCESSING
    print("[*] GENERATING NON-LINEAR CLASSIFICATION DATASET...")
    df = generate_classification_data(n_samples=1000)
    X = df[['FEATURE_1', 'FEATURE_2']].values
    y = df['TARGET'].values.reshape(-1, 1)

    print("[*] SPLITTING AND SCALING DATA...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=36)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. BUILD NEURAL NETWORK ARCHITECTURE
    print("\n[*] BUILDING NEURAL NETWORK ARCHITECTURE...")
    model = NeuralNetwork()
    
    # HIDDEN LAYER 1
    model.add(Dense(n_inputs=2, n_neurons=16, seed=36))
    model.add(ReLU())
    
    # HIDDEN LAYER 2
    model.add(Dense(n_inputs=16, n_neurons=16, seed=36))
    model.add(ReLU())
    
    # OUTPUT LAYER (BINARY CLASSIFICATION)
    model.add(Dense(n_inputs=16, n_neurons=1, seed=36))
    model.add(Sigmoid())

    # 4. COMPILATION
    print("[*] COMPILING MODEL...")
    model.compile(
        loss_function=BinaryCrossEntropy(),
        optimizer=SGD(learning_rate=0.1)
    )

    # 5. TRAINING PIPELINE
    print("\n[*] INITIATING TRAINING PIPELINE...")
    EPOCHS = 100
    BATCH_SIZE = 32
    METRICS_LIST = [accuracy, precision, recall, f1_score]
    
    history = train_network(
        model=model, 
        X_train=X_train, 
        y_train=y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        metric_fns=METRICS_LIST
    )

    # 6. EVALUATION ON UNSEEN TEST SET
    print("\n[*] EVALUATING ON TEST SET...")
    y_pred_test = model.predict(X_test)
    
    test_loss = model.loss_function.calculate(y_pred_test, y_test)
    test_acc = accuracy(y_pred_test, y_test)
    test_prec = precision(y_pred_test, y_test)
    test_rec = recall(y_pred_test, y_test)
    test_f1 = f1_score(y_pred_test, y_test)
    
    print(f"TEST LOSS:      {test_loss:.4f}")
    print(f"TEST ACCURACY:  {test_acc:.4f}")
    print(f"TEST PRECISION: {test_prec:.4f}")
    print(f"TEST RECALL:    {test_rec:.4f}")
    print(f"TEST F1 SCORE:  {test_f1:.4f}")

    # 7. VISUALIZATIONS (STRICTLY FORMATTED)
    print("\n[*] GENERATING VISUALIZATIONS...")
    
    # PLOT 1: EPOCH VS LOSS
    plt.figure(figsize=(8, 6), dpi=500)
    plt.plot(history['loss'], color='red', linewidth=2.5)
    plt.title('TRAINING CONVERGENCE: EPOCH VS LOSS', fontweight='bold', fontsize=12)
    plt.xlabel('EPOCH', fontweight='bold', fontsize=10)
    plt.ylabel('BINARY CROSS ENTROPY LOSS', fontweight='bold', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig('LOSS_CURVE.png')
    plt.show()

    # PLOT 2: EPOCH VS METRICS
    plt.figure(figsize=(8, 6), dpi=500)
    plt.plot(history['accuracy'], label='ACCURACY', color='blue', linewidth=2)
    plt.plot(history['f1_score'], label='F1 SCORE', color='green', linewidth=2)
    plt.title('MODEL PERFORMANCE: EPOCH VS METRICS', fontweight='bold', fontsize=12)
    plt.xlabel('EPOCH', fontweight='bold', fontsize=10)
    plt.ylabel('METRIC SCORE', fontweight='bold', fontsize=10)
    plt.legend(prop={'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig('METRICS_CURVE.png')
    plt.show()

if __name__ == '__main__':
    main()
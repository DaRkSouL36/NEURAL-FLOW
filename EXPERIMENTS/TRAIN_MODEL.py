import numpy as np
import time

def train_network(model, X_train: np.ndarray, y_train: np.ndarray, 
                  epochs: int, batch_size: int, metric_fn) -> dict:
    """
    FULL TRAINING LOOP IMPLEMENTING MINI-BATCH GRADIENT DESCENT.
    """
    history = {'loss': [], 'metric': []}
    n_samples = X_train.shape[0]
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # SHUFFLE DATA AT START OF EACH EPOCH
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_losses = []
        epoch_metrics = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # 1. FORWARD PASS
            y_pred = model.forward(X_batch)
            
            # 2. COMPUTE LOSS AND METRIC
            loss = model.loss_function.calculate(y_pred, y_batch)
            metric_val = metric_fn(y_pred, y_batch)
            
            # 3. BACKPROPAGATION
            model.backward(y_pred, y_batch)
            
            # 4. OPTIMIZATION
            model.update_weights()
            
            epoch_losses.append(loss)
            epoch_metrics.append(metric_val)
            
        avg_loss = np.mean(epoch_losses)
        avg_metric = np.mean(epoch_metrics)
        
        history['loss'].append(avg_loss)
        history['metric'].append(avg_metric)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"EPOCH {epoch+1}/{epochs} | LOSS: {avg_loss:.4f} | METRIC: {avg_metric:.4f}")
            
    end_time = time.time()
    print(f"TRAINING COMPLETE IN {end_time - start_time:.2f} SECONDS")
    return history
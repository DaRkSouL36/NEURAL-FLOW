import numpy as np
import time

def train_network(model, X_train: np.ndarray, y_train: np.ndarray, 
                  epochs: int, batch_size: int, metric_fns: list = None) -> dict:
    """
    MAIN TRAINING PIPELINE WITH MINI-BATCH LOGIC AND HISTORY TRACKING.
    """
    if metric_fns is None:
        metric_fns = []
        
    # INITIALIZE HISTORY TRACKING
    history = {'loss': []}
    for fn in metric_fns:
        history[fn.__name__] = []
        
    n_samples = X_train.shape[0]
    start_time = time.time()
    
    for epoch in range(epochs):
        # 1. SHUFFLE DATA AT THE START OF EACH EPOCH FOR STOCHASTICITY
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_losses = []
        epoch_metrics = {fn.__name__: [] for fn in metric_fns}
        
        # 2. MINI-BATCH BATCHING LOGIC
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # 3. FORWARD PASS
            y_pred = model.forward(X_batch)
            
            # 4. LOSS COMPUTATION
            loss = model.loss_function.calculate(y_pred, y_batch)
            epoch_losses.append(loss)
            
            # 5. METRIC COMPUTATION
            for fn in metric_fns:
                metric_val = fn(y_pred, y_batch)
                epoch_metrics[fn.__name__].append(metric_val)
            
            # 6. BACKPROPAGATION
            model.backward(y_pred, y_batch)
            
            # 7. WEIGHT UPDATE (OPTIMIZATION LOOP)
            model.update_weights()
            
        # AGGREGATE EPOCH RESULTS
        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)
        
        metric_print_str = ""
        for fn in metric_fns:
            avg_metric = np.mean(epoch_metrics[fn.__name__])
            history[fn.__name__].append(avg_metric)
            metric_print_str += f" | {fn.__name__.upper()}: {avg_metric:.4f}"
        
        # MONITORING OUTPUT
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"EPOCH {epoch+1:03d}/{epochs} | LOSS: {avg_loss:.4f}{metric_print_str}")
            
    end_time = time.time()
    print(f"\nTRAINING COMPLETE IN {end_time - start_time:.2f} SECONDS")
    
    return history
import numpy as np

def gini(data):
    data = np.asarray(data)
    if len(data) == 0:
        return 0.0
    
    # shift to non-negative
    min_val = np.min(data)
    if min_val < 0:
        data = data - min_val
        
    if np.sum(data) == 0:
        return 0.0

    data = np.sort(data)
    n = len(data)
    index = np.arange(1, n + 1)
    
    return (2 * np.sum(index * data)) / (n * np.sum(data)) - (n + 1) / n

def sen_welfare(data):
    return np.mean(data) * (1-gini(data))
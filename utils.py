import numpy as np
from scipy import stats

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

def polarization(beliefs):
    """Mean pairwise absolute distance, O(N log N)."""
    b = np.sort(np.asarray(beliefs, dtype=float))
    n = len(b)
    weights = 2 * np.arange(1, n + 1) - n - 1
    return (4 / n**2) * np.dot(weights, b)


def bimodality_coefficient(beliefs):
    """
    BC = (skewness² + 1) / kurtosis  (with small-sample correction).
    BC > 0.555 suggests bimodality.
    """
    n = len(beliefs)
    skew = stats.skew(beliefs) * np.sqrt((n - 1) / n) * (n / (n - 2))
    kurt = stats.kurtosis(beliefs, fisher=True)
    return (skew**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
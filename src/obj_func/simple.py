import numpy as np
__all__ = [
    "quad", "almost_twin_peaks"
]

def quad(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)

def almost_twin_peaks(x: np.ndarray) -> np.ndarray:
    peak1 = -20 * np.exp(-np.sum((x + 2)**2, axis=1) / 3.0)
    peak2 = -1 * np.exp(-np.sum((x - 2)**2, axis=1) / 10.0)
    return peak1 + peak2
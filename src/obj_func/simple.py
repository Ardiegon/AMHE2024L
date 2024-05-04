import numpy as np
__all__ = [
    "simple_obj_wrapper"
]

def quad(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)

def almost_twin_peaks(x: np.ndarray) -> np.ndarray:
    peak1 = -20 * np.exp(-np.sum((x + 2)**2, axis=1) / 3.0)
    peak2 = -1 * np.exp(-np.sum((x - 2)**2, axis=1) / 10.0)
    return peak1 + peak2

SIMPLE_OBJECTIVES = {
    "quad": quad,
    "almost-twin-peaks": almost_twin_peaks,
}

def objective_wrapper(func_desc: str):
    return SIMPLE_OBJECTIVES[func_desc]

def get_names():
    return list(SIMPLE_OBJECTIVES.keys())   

import os 
import numpy as np
from functools import partial
from src.obj_func.cec_from_authors import cec22_test_func
from pathlib import Path

def create_evaluate_cec_function(function_number):
    return partial(cec_wrapper, function_number=function_number)

def cec_wrapper(x, function_number):
    mx, nx = np.shape(x)
    value = np.zeros((mx,))
    for i in range(mx):
        value[i] = cec22_test_func(x[i, :], nx, 1, function_number)
    # return cec22_test_func(x, len(x), 1, function_number)
    return value

def clear_console()->None:
    if os.name == 'nt':  
        os.system('cls')
    else:
        os.system('clear')

def print_clean(string: str)->None:
    clear_console()
    print(string)

def are_any_arrays_equal(arrays: list[np.ndarray])->bool:
    """
    Checks if given list of ndarrays contains two arrays with same values.
    """
    num_arrays = len(arrays)
    for i in range(num_arrays):
        for j in range(i + 1, num_arrays):
            if np.array_equal(arrays[i], arrays[j]):
                return True
    return False
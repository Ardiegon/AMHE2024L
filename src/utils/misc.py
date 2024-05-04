import os 
import numpy as np

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
from math import sqrt

def default_config(pop_dims=2):
    maxFES = pop_dims*10000
    pop_size = pop_dims*5
    offspring_size = pop_size//2
    # window_size = int(6+3*sqrt(pop_dims))
    window_size = int(6+5*sqrt(pop_dims))
    delta_change_rate = 4/(pop_dims+4) 

    return {
    "pop_dims": pop_dims,
    "find_max": False,
    
    # init population params 
    "new_population_mean": 5.0,
    "new_population_variance": 3.0,

    # run params
    "max_n_epochs": maxFES,
    "pop_size": pop_size,
    "offspring_size": offspring_size,
    "history_window_size": window_size,
    "delta": 0,
    # "scalling_factor": 1/sqrt(2),
    "scalling_factor": 1/sqrt(2),
    "small_delta": 0.797885,
    "noise_intensity": 1e-8*0.797885,
    "delta_change_rate": delta_change_rate
    } 
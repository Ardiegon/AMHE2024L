def default_config(pop_dims=2):
    pop_size = pop_dims*5
    return {
    "pop_size": pop_size,
    "pop_dims": pop_dims,
    #whatever goes here
    #
    # init population params 
    "new_population_mean": 5.0,
    "new_population_variance": 3.0,
    }
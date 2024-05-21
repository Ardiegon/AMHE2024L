def default_config(pop_dims=10):
    population_max = 30 * pop_dims
    population_min = 0.25 * population_max
    population_size = population_max
    max_iterations = 1000*pop_dims
    window_size = 20 * pop_dims
    memory_F = 0.2
    memory_Cr = 0.2
    archive_size = int(2.1 * population_size)
    return {
    "population_max": population_max,
    "population_min": population_min,
    "pop_dims": pop_dims,
    "pop_size": population_size,
    "max_iters": max_iterations,
    "window_size": window_size,
    "memory_F": memory_F,
    "memory_Cr": memory_Cr,
    "archive_size": archive_size,
    "archive_probability": 0.5,
    "best_part_of_pop": 0.2, 

    # init population params 
    "new_population_mean": 5.0,
    "new_population_variance": 3.0,
    }
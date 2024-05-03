import numpy as np

def pop_mean(population: np.ndarray)->np.ndarray:
    """
    General
    Calculates mean specimen of given population
    """
    return np.mean(population, axis=0)

def pop_mean(sorted_population: np.ndarray, best_part: int)->np.ndarray:
    """
    General
    Calculates mean specimen of best part of sorted population.
    best_part = 1 will return best specimen
    """
    return np.mean(sorted_population[:best_part, :], axis=0)

def sort_pop(population:np.ndarray, agent_eval_callable: callable)->np.ndarray:
    """
    General
    Sorts population accordingly to fitness value of objective given to agent.
    """
    pass



def calc_end_value(population, best_mean):
    """
    DES
    Calculates value that is further used in end condition evaluation
    """
    sum = 0.0
    for dimension in range(population.shape[1]):
        sum += np.sqrt(np.sum([(a-best_mean[dimension])**2 for a in population[:,dimension]]))
    return sum/population.shape[1]
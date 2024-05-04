import numpy as np

EPSILON = 1e-8

####
# General
####

def pop_mean(population: np.ndarray)->np.ndarray:
    """
    General
    Calculates mean specimen of given population
    """
    return np.mean(population, axis=0)

def best_pop_mean(sorted_population: np.ndarray, best_part: int)->np.ndarray:
    """
    General
    Calculates mean specimen of best part of sorted population.
    best_part = 1 will return best specimen
    """
    return np.mean(sorted_population[:best_part, :], axis=0)

def sort_pop(population:np.ndarray, agent_eval_callable: callable)->tuple[np.ndarray,np.ndarray]:
    """
    General
    Sorts population accordingly to fitness value of objective given to agent.
    """
    q_population = agent_eval_callable(population)
    sorted_args = np.argsort(q_population)
    return q_population[sorted_args], population[sorted_args]

def nonstandard_cauchy(mean: float = 0.0, gamma: float = 1.0)->float:
    """
    General
    Returns realisation of random variable from non-standard Cauchy Distribution.
    Implementation is possible due to Cauchy is the same in terms of addition and 
    multiplication as for example Normal distribution.
    """
    return gamma * np.random.standard_cauchy(1) + mean

####
# DES
####

def calc_end_value(population, best_mean):
    """
    DES
    Calculates value that is further used in end condition evaluation
    """
    sum = 0.0
    for dimension in range(population.shape[1]):
        sum += np.sqrt(np.sum([(a-best_mean[dimension])**2 for a in population[:,dimension]]))
    return sum/population.shape[1]

####
# NL-SHADE-RSP
####

def calculate_archive_probability(deltas: np.ndarray)->float:
    """
    NL-SHADE-RSP
    read about in article: equation 14
    deltas shape should be (:, 2) where:
    First column of deltas are values between evaluation of new specimen and it's parent
    Second column of deltas tells if in mutation archive was used.
    """
    deltas_archive = deltas[deltas[:,1]==True, 0]
    deltas_pop = deltas[deltas[:,1]==False, 0]
    sum_da = np.sum(deltas_archive)
    sum_dp = np.sum(deltas_pop)
    ratio_da = len(deltas_archive)/len(deltas)
    ratio_dp = len(deltas_pop)/len(deltas)
    return np.clip(sum_da/(ratio_da+EPSILON)/(sum_da/(ratio_da+EPSILON) + sum_dp/(ratio_dp+EPSILON)), 0.1, 0.9)

def calculate_new_memory(k:int, memory: np.ndarray, samples: np.ndarray, deltas: np.ndarray)->np.ndarray:
    """
    NL-SHADE-RSP
    read about in article: equation 7, 8, 9
    Calculates new F and Cr values for given k in H-sized window of memory.
    memory: H-sized memory of F or Cr values 
    samples: the F or Cr modifiers that was succesfull in creation of better specimen than it's parrent
    deltas shape should be (:, 2) where:
    First column of deltas are values between evaluation of new specimen and it's parent
    Second column of deltas tells if in mutation archive was used.
    """
    deltas = deltas[:,0]
    weights = deltas/np.sum(deltas)
    mean_w_L = np.sum(weights*(samples**2))/np.sum(weights*samples)
    memory[k] = (0.5 * memory[k] + 0.5 * mean_w_L).item()
    return memory

def binomial_crossover(x: np.ndarray, u: np.ndarray, Crb: float)->np.ndarray:
    """
    NL-SHADE-RSP
    read about in article: equation 2
    Randomly ejecting genes from mutant u to parent x at probability Crb. 
    """
    new_specimen = np.copy(x)
    jrand = np.random.randint(0,len(new_specimen))
    for i in range(len(new_specimen)):
        if np.random.rand() < Crb or i == jrand:
            new_specimen[i] = u[i]
    return new_specimen

def exponential_crossover(x: np.ndarray, u: np.ndarray, Cr_i: float)->np.ndarray:
    """
    NL-SHADE-RSP
    read about in article: equation 3
    Ejecting some range of genes from n1 to n1+n2 from mutant u to parent x, 
    where Cr_i is probability for n2 to increase +1 in next loop steps. 
    """
    new_specimen = np.copy(x)
    n1 = np.random.randint(0,len(new_specimen)-1)
    n2 = 1
    while True:
        if np.random.rand()<Cr_i and n1+n2<len(new_specimen):
            n2+=1
        else:
            break
    new_specimen[n1:n1+n2] = u[n1:n1+n2]
    return new_specimen

def nlpsr(g: int, max_iters: int, NPmin: int, NPmax: int)->int:
    """
    NL-SHADE-RSP
    read about in article: equation 13
    Calculates new size of populations accordingly to min pop, max pop, and current timestep g.
    It's non linear and author's are explaining why exactly it is better.
    """
    nfe_r = g/max_iters
    return round((NPmin-NPmax)*(nfe_r**(1-nfe_r))+NPmax)

def update_Crb(g: int, max_iters: int)->int:
    """
    NL-SHADE-RSP
    read about in article: equation 15
    Returns probability of ejecting genes from mutant to parent at binomial crossover.
    It is 0 until half of epochs passed, then linearly gets to 1.
    """
    if g < 0.5 * max_iters:
        return 0
    else: 
        return 2*(g/max_iters-0.5) # W artykule błąd we wzorze, w tekście jest mowa o ciągłości której we wzorze nie ma 
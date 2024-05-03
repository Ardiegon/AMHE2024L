import numpy as np
from dataclasses import dataclass

from configs.AMALGATIONconfig import default_config
from utils.misc import print_clean

@dataclass
class StateAMALGATION:
    timestep: int
    population: np.ndarray
    # pewnie jeszcze coś

class AMALGATION:
    def __init__(self, objective, population=None, config = {}) -> None:
        self.config = default_config() | config
        self.objective = objective
        population = population if population is not None else self._init_population()
        self.history = [self._init_state(population)]

    def _init_population(self):
        n_dims = self.config["pop_size"]
        n_samples = self.config["pop_dims"]
        mu = self.config["new_population_mean"]
        sigma = self.config["new_population_variance"]
        population = np.random.normal(mu, sigma, (n_dims, n_samples))
        q_population = self._eval(population)
        sorted_args = np.argsort(q_population)
        return population[sorted_args]
    
    def _init_state(self, population):
        return StateAMALGATION(0, population)

    def _make_step(self, state):
        pass
        # Do uzupełnienia

    def _eval(self, pop):
        if len(pop.shape) == 1: # single specimen
            pop = pop[None,:]
        return self.objective(pop)
    
    def run(self):
        # pewnie mało się zmieni
        max_iters = self.config["max_iters"]
        state = self.history[-1]

        while state.timestep < max_iters:
            step_result = self._make_step(state) # self._make_step(state, i inne...)
            state = StateAMALGATION(*step_result)
            self.history.append(state)
            if state.timestep%100==0:
                print_clean(f"Timestep: {state.timestep}\nCurrent Mean: {np.mean(state.population, axis=0)}\nEval: {self._eval(np.mean(state.population, axis=0))}")
        self.dump_history_to_file(f"src/checkpoints/lastamalgation.npy")

    def get_history_means(self):
        means = []
        for state in self.history:
            means.append(np.mean(state.population, axis=0))
        return np.array(means)
    
    def dump_history_to_file(self, file_path):
        np.save(file_path, np.array(self.history))

    def load_history_from_file(self, file_path):
        self.history = np.load(file_path, allow_pickle=True)
    
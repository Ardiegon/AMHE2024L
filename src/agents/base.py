import numpy as np
import utils.algorithmic as alg
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BaseAgentState(ABC):
    timestep: int
    population: np.ndarray

class BaseAgent(ABC):
    def __init__(self, objective: callable, population: np.ndarray=None, config: dict={}) -> None:
        self.config = config
        self.objective = objective
        population = population if population is not None else self._init_population()
        self.history = [self._init_state(population)]

    def _init_population(self):
        n_dims = self.config["pop_size"]
        n_samples = self.config["pop_dims"]
        mu = self.config["new_population_mean"]
        sigma = self.config["new_population_variance"]
        population = np.random.normal(mu, sigma, (n_dims, n_samples))
        _, population = alg.sort_pop(population, self._eval)
        return population
    
    def _eval(self, pop: np.ndarray)->np.ndarray:
        if len(pop.shape) == 1: # single specimen
            pop = pop[None,:]
        return self.objective(pop)

    def get_history_means(self)->np.ndarray:
        means = []
        for state in self.history:
            means.append(np.mean(state.population, axis=0))
        return np.array(means)
    
    def dump_history_to_file(self, file_path:str)->None:
        np.save(file_path, np.array(self.history))

    def load_history_from_file(self, file_path:str)->None:
        self.history = np.load(file_path, allow_pickle=True)
    
    @abstractmethod
    def _init_state(self, population:np.ndarray)->BaseAgentState:
        """
        Define AgentState dataclass, which will hold timestep dependent parameters (among others timestep and population).
        Here, return object of this AgentState class, initialized for timestep 0.  
        """
        pass
        
    @abstractmethod
    def _make_step(self, *args, **kwargs)->tuple:
       """
       Single Step of your future BBO algorithm
       It should begin from about sorting population, and end by returning all info needed to create next AgentState object.
       Please add arguments here that are constant in timesteps.
       """
       pass

    @abstractmethod
    def run(self)->np.ndarray:
        """
        Loads constant in time parameters from config and passes them to next timesteps until finished.
        Should return best specimen after end conditions are met.
        Try to use dump_history_to_file and clean_print for maintaining homogenity.
        """
        pass

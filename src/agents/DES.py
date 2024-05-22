import numpy as np
from dataclasses import dataclass

import src.utils.algorithmic as alg
from src.agents.base import BaseAgent, BaseAgentState
from src.configs.DESconfig import default_config
from src.utils.misc import print_clean

@dataclass
class StateDES(BaseAgentState):
    delta: np.ndarray

class DES(BaseAgent):
    def __init__(self, objective, population=None, config = {}) -> None:
        super().__init__(objective, population, config = default_config() | config)
    
    def _init_state(self, population):
        return StateDES(0, population, np.ones_like(population[0]) * self.config["delta"])

    def _make_step(self, state, mi, c, f, d, h, e):
        pop_mean = alg.pop_mean(state.population)
        q_pop_mean = self._eval(pop_mean) # TODO log
        _, sorted_pop = alg.sort_pop(state.population, self._eval)
        best_mean = alg.best_pop_mean(sorted_pop, mi)
        next_delta = (1 - c) * state.delta + c * (best_mean - pop_mean)

        next_pop = []
        for _ in range(state.population.shape[0]):
            hw = np.random.randint(1, min(h,len(self.history))) if len(self.history) > 1 else 1
            j,k = np.random.randint(0,mi), np.random.randint(0,mi)
            sample = f * (self.history[-hw].population[j,:] - self.history[-hw].population[k,:]) + next_delta * d * np.random.normal(0,1)
            next_pop.append( (best_mean+ sample + e * np.random.normal(0,1, sample.shape))[None, :])
        return np.concatenate(next_pop, axis=0), next_delta, best_mean

    def run(self):
        max_n_epochs = self.config["max_n_epochs"]
        c = self.config["delta_change_rate"]
        f = self.config["scalling_factor"]
        d = self.config["small_delta"] # in paper small_delta = E||N(0,I)||, is it weirdly written? Norm of single random variable realisation should be equal to its absolute value?.......
        h = self.config["history_window_size"]
        mi = self.config["mi"]
        e = self.config["noise_intensity"]

        state = self.history[-1]

        t = 1  # TODO zmigrować się na state
        end_cond_value = float("inf")
        while end_cond_value >= e and t < max_n_epochs:
            next_pop, next_delta, best_mean = self._make_step(state, mi, c, f, d, h, e)
            end_cond_value = alg.calc_end_value(next_pop, best_mean)
            t = t+1
            state = StateDES(t, next_pop, next_delta)
            self.history.append(state)
            if t%100==0:
                print_clean(f"Timestep: {t}\nCurrent Mean: {best_mean}\nEval: {self._eval(best_mean)}\nCurrent End Value: {end_cond_value}\nEnd Value: {e}")
        self.dump_history_to_file(f"src/checkpoints/lastDES.npy")
        _, sorted_pop = alg.sort_pop(state.population, self._eval)
        return alg.best_pop_mean(sorted_pop, 1)
    
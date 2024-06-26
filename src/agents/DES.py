import numpy as np
from dataclasses import dataclass

import src.utils.algorithmic as alg
from src.agents.base import BaseAgent, BaseAgentState
from src.configs.DESConfig import DESConfig
from src.utils.misc import print_clean


@dataclass
class StateDES(BaseAgentState):
    delta: np.ndarray


class DES(BaseAgent):
    def __init__(self, objective, config: DESConfig, population=None) -> None:
        self.max_n_epochs = config.max_iterations
        self.c = config.delta_change_rate
        self.f = config.scaling_factor
        self.d = config.small_delta  # in paper small_delta = E||N(0,I)||, is it weirdly written? Norm of single random variable realisation should be equal to its absolute value?.......
        self.h = config.history_window_size
        self.mu = config.mu
        self.e = config.noise_intensity
        self.delta = config.delta
        super().__init__(objective, config, population)

    def _init_state(self, population):
        return StateDES(0, population, np.ones_like(population[0]) * self.delta)

    def _make_step(self, state):
        pop_mean = alg.pop_mean(state.population)
        q_pop_mean = self._eval(pop_mean) # TODO log
        _, sorted_pop = alg.sort_pop(state.population, self._eval)
        best_mean = alg.best_pop_mean(sorted_pop, self.mu)
        next_delta = (1 - self.c) * state.delta + self.c * (best_mean - pop_mean)

        next_pop = []
        for _ in range(state.population.shape[0]):
            hw = np.random.randint(1, min(self.h, len(self.history))) if len(self.history) > 1 else 1
            j,k = np.random.randint(0, self.mu), np.random.randint(0, self.mu)
            sample = self.f * (self.history[-hw].population[j,:] - self.history[-hw].population[k,:]) + next_delta * self.d * np.random.normal(0,1)
            next_pop.append((best_mean + sample + self.e * np.random.normal(0, 1, sample.shape))[None, :])
        return np.concatenate(next_pop, axis=0), next_delta, best_mean

    def run(self):
        state = self.history[-1]

        t = 1  # TODO zmigrować się na state
        end_cond_value = float("inf")
        while end_cond_value >= self.e and t < self.max_n_epochs:
            next_pop, next_delta, best_mean = self._make_step(state)
            end_cond_value = alg.calc_end_value(next_pop, best_mean)
            t = t+1
            state = StateDES(t, next_pop, next_delta)
            self.history.append(state)
            if t%100==0:
                print_clean(f"Timestep: {t}\nCurrent Mean: {best_mean}\nEval: {self._eval(best_mean)}\nCurrent End Value: {end_cond_value}\nEnd Value: {self.e}")
        self.dump_history_to_file(f"src/checkpoints/lastDES.npy")
        _, sorted_pop = alg.sort_pop(state.population, self._eval)
        return alg.best_pop_mean(sorted_pop, 1)
    
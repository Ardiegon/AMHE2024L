import numpy as np
from dataclasses import dataclass

import src.utils.algorithmic as alg
from src.configs import NLSHADERSPConfig
from src.agents.base import BaseAgent, BaseAgentState
from src.utils.misc import print_clean, are_any_arrays_equal

EPSILON = 1e-8


@dataclass
class StateHybridizationV2(BaseAgentState):
    archive: np.ndarray
    archive_probability: float
    NP: int
    NA: int
    M_F: np.ndarray
    Mk: int
    delta: np.ndarray



class Hybridization(BaseAgent):
    def __init__(self, objective, config: NLSHADERSPConfig, population=None) -> None:
        self.H = config.history_window_size
        self.NPmin = config.population_size_min
        self.NPmax = config.population_size_max
        self.max_iters = config.max_iterations
        self.best_part = config.best_part_of_pop
        self.memory_F = config.memory_F
        self.archive_probability = config.archive_probability
        self.archive_size = config.archive_size

        self.c = config.delta_change_rate
        self.d = config.small_delta
        self.e = config.noise_intensity

        super().__init__(objective, config, population)

    def _init_state(self, population):
        mF = np.ones(self.H) * self.memory_F
        archive = np.copy(population) # Miejmy częściowo zainicjowane archiwum, prościej i ładniej biorąc co piszą w artykule
        return StateHybridizationV2(
            0,
            population,
            archive,
            self.archive_probability,
            self.config.population_size,
            self.archive_size,
            mF,
            0,
            np.zeros_like(population[0])
        )

    def _make_step(self, state):
        # 6
        S_F = []
        q_deltas = []
        # 7
        q_pop, pop = alg.sort_pop(state.population, self._eval)
        # 8 - 16
        realizations_F = []
        for _ in range(state.NP):
            r = np.random.randint(0, self.H)
            while True:
                F_i = alg.nonstandard_cauchy(state.M_F[r], 0.1)
                if F_i > 0:
                    realizations_F.append(min(F_i, 1))
                    break
                    # 17
        # 18 - 42

        mean_individual = alg.pop_mean(pop)
        mean_of_best_individuals = alg.best_pop_mean(pop, int(state.NP*self.best_part))

        next_delta = (1 - self.c) * state.delta + self.c * (mean_of_best_individuals - mean_individual)


        for i in range(state.NP):
            archive_used = False
            while True:
                current_index = i
                r1_index = np.random.randint(state.NP)
                if np.random.rand() < state.archive_probability:
                    r2_index = np.random.randint(state.NP)
                else:
                    archive_used = True
                    r2_index = np.random.randint(state.archive.shape[0])

                if not are_any_arrays_equal([current_index, r1_index, r2_index]):
                    break

            current = pop[i]
            r1 = pop[r1_index]
            r2 = state.archive[r2_index]
            di = realizations_F[i]*(r1-r2) + next_delta * self.d * np.random.normal(0,1)
            new = mean_of_best_individuals + di + self.e * np.random.normal(0, 1, di.shape)
            q_new = self._eval(new)
            q_current = q_pop[i]
            if q_new < q_current:
                pop[i] = new
                if len(state.archive) >= state.NA:
                    state.archive[np.random.randint(len(state.archive))] = current[None,:]
                else:
                    state.archive = np.concatenate([state.archive, current[None, :]], axis=0)
                S_F.append(realizations_F[i])
                q_deltas.append([q_current - q_new, archive_used])
        # 43
        new_NP = alg.nlpsr(state.timestep, self.max_iters, self.NPmin, self.NPmax)
        new_NA = int(2.1*new_NP)
        # 44-49
        if len(state.archive) > new_NA:
            random_samples_args = np.random.randint(0,len(state.archive), size=(new_NA))
            new_archive = state.archive[random_samples_args]
        else:
            new_archive = state.archive
        if len(pop) > new_NP:
            pop = pop[:new_NP]
        # 50-51
        q_deltas = np.asarray(q_deltas, dtype=object)
        S_F = np.asarray(S_F, dtype=object)
        if len(q_deltas) != 0:
            new_archive_probability = alg.calculate_archive_probability(q_deltas)
            new_M_F = alg.calculate_new_memory(state.Mk, state.M_F, S_F, q_deltas)
        else:
            new_archive_probability = state.archive_probability
            new_M_F = state.M_F
        new_Mk = (state.Mk+1)%self.H
        new_timestep = state.timestep + 1
        return new_timestep, pop, new_archive, new_archive_probability, new_NP, new_NA, new_M_F, new_Mk, next_delta

    def run(self):
        state = self.history[-1]

        while state.timestep < self.max_iters:
            step_result = self._make_step(state)
            state = StateHybridizationV2(*step_result)
            self.history.append(state)
            if state.timestep%100==0:
                print_clean(f"Timestep: {state.timestep}\nCurrent Mean: {alg.pop_mean(state.population)}\nEval: {self._eval(alg.pop_mean(state.population))}")

        self.dump_history_to_file(f"src/checkpoints/lastnlshadersp.npy")
        _, sorted_pop = alg.sort_pop(state.population, self._eval)
        return alg.best_pop_mean(sorted_pop, 1)

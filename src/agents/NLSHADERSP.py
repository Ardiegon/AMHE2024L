import numpy as np
from dataclasses import dataclass

import src.utils.algorithmic as alg
from src.configs import NLSHADERSPConfig
from src.agents.base import BaseAgent, BaseAgentState
from src.utils.misc import print_clean, are_any_arrays_equal

EPSILON = 1e-8


@dataclass
class StateNLSHADERSP(BaseAgentState):
    archive: np.ndarray
    archive_probability: float
    NP: int
    NA: int
    M_F: np.ndarray 
    M_Cr: np.ndarray
    Mk: int


class NLSHADERSP(BaseAgent):
    def __init__(self, objective, config: NLSHADERSPConfig, population=None) -> None:
        self.H = config.history_window_size
        self.NPmin = config.population_size_min
        self.NPmax = config.population_size_max
        self.max_iters = config.max_iterations
        self.best_part = config.best_part_of_pop
        self.memory_F = config.memory_F
        self.memory_Cr= config.memory_Cr
        self.archive_probability = config.archive_probability
        self.archive_size = config.archive_size

        super().__init__(objective, config, population)

    def _init_state(self, population):
        mCr = np.ones(self.H) * self.memory_Cr
        mF = np.ones(self.H) * self.memory_F
        archive = np.copy(population) # Miejmy częściowo zainicjowane archiwum, prościej i ładniej biorąc co piszą w artykule
        return StateNLSHADERSP(
            0,
            population,
            archive,
            self.archive_probability,
            self.config.population_size,
            self.archive_size,
            mF,
            mCr,
            0
        )

    def _make_step(self, state):
        # 6
        S_F = []
        S_Cr = []
        q_deltas = []
        # 7
        q_pop, pop = alg.sort_pop(state.population, self._eval)
        # 8 - 16
        realizations_F = []
        realizations_Cr = []
        for _ in range(state.NP):
            r = np.random.randint(0, self.H)
            realizations_Cr.append(np.clip(np.random.normal(state.M_Cr[r], 0.1), 0.0, 1.0))
            while True:
                F_i = alg.nonstandard_cauchy(state.M_F[r], 0.1)
                if F_i > 0:
                    realizations_F.append(min(F_i, 1))
                    break
                    # 17
        realizations_Cr = np.sort(realizations_Cr) # W tekście wytłumaczone że according to fitness to w praktyce mniejsze pierwsze jeśli populację mamy posortowaną
        # 18 - 42
        for i in range(state.NP):
            archive_used = False
            while True:
                # TODO poprawić
                pbest_index = np.random.randint(int(state.NP*self.best_part))
                current_index = i
                r1_index = np.random.randint(state.NP)
                if np.random.rand() < state.archive_probability:
                    r2_index = np.random.randint(state.NP)
                else:
                    archive_used = True
                    r2_index = np.random.randint(state.archive.shape[0])

                if not are_any_arrays_equal([current_index, pbest_index, r1_index, r2_index]):
                    break
            current = pop[current_index]
            pbest = pop[pbest_index]
            r1 = pop[r1_index]
            r2 = state.archive[r2_index]
            new = current + realizations_F[i]*(pbest-current) + realizations_F[i]*(r1-r2)
            Crb = alg.update_Crb(state.timestep, self.max_iters)
            if np.random.rand() < 0.5:
                new = alg.binomial_crossover(current, new, Crb)
            else:
                new = alg.exponential_crossover(current, new, realizations_Cr[i])
            q_new = self._eval(new)
            q_current = q_pop[i]
            if q_new < q_current:
                pop[i] = new
                if len(state.archive)>=state.NA:
                    state.archive[np.random.randint(len(state.archive))] = current[None,:]
                else:
                    state.archive = np.concatenate([state.archive, current[None, :]], axis=0)
                S_F.append(realizations_F[i])
                S_Cr.append(realizations_Cr[i])
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
        S_Cr = np.asarray(S_Cr, dtype=object)
        if len(q_deltas) != 0:
            new_archive_probability = alg.calculate_archive_probability(q_deltas)
            new_M_F = alg.calculate_new_memory(state.Mk, state.M_F, S_F, q_deltas)
            new_M_Cr = alg.calculate_new_memory(state.Mk, state.M_Cr, S_Cr, q_deltas)
        else:
            new_archive_probability = state.archive_probability
            new_M_F = state.M_F
            new_M_Cr = state.M_Cr
        new_Mk = (state.Mk+1)%self.H
        new_timestep = state.timestep + 1
        return new_timestep, pop, new_archive, new_archive_probability, new_NP, new_NA, new_M_F, new_M_Cr, new_Mk

    def run(self):
        state = self.history[-1]

        while state.timestep < self.max_iters:
            step_result = self._make_step(state)
            state = StateNLSHADERSP(*step_result)
            self.history.append(state)
            if state.timestep%100==0:
                print_clean(f"Timestep: {state.timestep}\nCurrent Mean: {alg.pop_mean(state.population)}\nEval: {self._eval(alg.pop_mean(state.population))}")

        self.dump_history_to_file(f"src/checkpoints/lastnlshadersp.npy")
        _, sorted_pop = alg.sort_pop(state.population, self._eval)
        return alg.best_pop_mean(sorted_pop, 1)

import numpy as np
import utils.algorithmic as alg
from dataclasses import dataclass

from configs.NLSHADERSPconfig import default_config
from utils.misc import print_clean, are_any_arrays_equal

EPSILON = 1e-8

@dataclass
class StateNLSHADERSP:
    timestep: int
    population: np.ndarray
    archive: np.ndarray
    archive_probability: float
    NP: int
    NA: int
    M_F: np.ndarray 
    M_Cr: np.ndarray
    Mk: int

class NLSHADERSP:
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
        mCr = np.ones(self.config["window_size"]) * self.config["memory_Cr"]
        mF = np.ones(self.config["window_size"]) * self.config["memory_F"]
        archive = np.copy(population) # Miejmy częściowo zainicjowane archiwum, prościej i ładniej biorąc co piszą w artykule
        return StateNLSHADERSP(0, population, archive, self.config["archive_probability"], 
                               self.config["pop_size"], self.config["archive_size"],
                               mF, mCr, 0)
    
    def _eval(self, pop):
        if len(pop.shape) == 1: # single specimen
            pop = pop[None,:]
        return self.objective(pop)

    def _make_step(self, state, H, NPmin, NPmax, max_iters, best_part):
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
            r = np.random.randint(0, H)
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
                current = pop[i]
                pbest = pop[np.random.randint(int(state.NP*best_part))]
                r1 = pop[np.random.randint(state.NP)]
                if np.random.rand() < state.archive_probability:
                    r2 = pop[np.random.randint(state.NP)]
                else:
                    archive_used = True
                    r2 = state.archive[np.random.randint(state.archive.shape[0])]
                if not are_any_arrays_equal([current, pbest, r1, r2]):
                    break
            new = current + realizations_F[i]*(pbest-current) + realizations_F[i]*(r1-r2)
            Crb = alg.update_Crb(state.timestep, max_iters)
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
        new_NP = alg.nlpsr(state.timestep, max_iters, NPmin, NPmax)
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
        new_Mk = (state.Mk+1)%H
        new_timestep = state.timestep + 1
        return new_timestep, pop, new_archive, new_archive_probability, new_NP, new_NA, new_M_F, new_M_Cr, new_Mk
    
    def run(self):
        H = self.config["window_size"]
        NPmin = self.config["population_min"]
        NPmax = self.config["population_max"]
        max_iters = self.config["max_iters"]
        best_part = self.config["best_part_of_pop"]

        state = self.history[-1]

        while state.timestep < max_iters:
            step_result = self._make_step(state, H, NPmin, NPmax, max_iters, best_part)
            state = StateNLSHADERSP(*step_result)
            self.history.append(state)
            if state.timestep%100==0:
                print_clean(f"Timestep: {state.timestep}\nCurrent Mean: {alg.pop_mean(state.population)}\nEval: {self._eval(alg.pop_mean(state.population))}")
        self.dump_history_to_file(f"src/checkpoints/lastnlshadersp.npy")
        _, sorted_pop = alg.sort_pop(state.population, self._eval)
        return alg.best_pop_mean(sorted_pop, 1)

    def get_history_means(self):
        means = []
        for state in self.history:
            means.append(np.mean(state.population, axis=0))
        return np.array(means)
    
    def dump_history_to_file(self, file_path):
        np.save(file_path, np.array(self.history))

    def load_history_from_file(self, file_path):
        self.history = np.load(file_path, allow_pickle=True)
    
import numpy as np
from dataclasses import dataclass

from configs.NLSHADERSPconfig import default_config
from utils.misc import print_clean

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

    def _make_step(self, state, H, NPmin, NPmax, max_iters, best_part):
        # 6
        S_F = []
        S_Cr = []
        q_deltas = []
        # 7
        pop = state.population
        q_pop = self._eval(pop)
        sorted_args = np.argsort(q_pop)
        pop = pop[sorted_args]
        q_pop = q_pop[sorted_args]
        # 8 - 16
        realizations_F = []
        realizations_Cr = []
        for _ in range(state.NP):
            r = np.random.randint(0, H)
            realizations_Cr.append(np.clip(np.random.normal(state.M_Cr[r], 0.1), 0.0, 1.0))
            while True:
                F_i = self._nonstandard_cauchy(state.M_F[r], 0.1)
                if F_i > 0:
                    realizations_F.append(min(F_i, 1))
                    break        
        # 17
        realizations_Cr = np.sort(realizations_Cr) # W tekście wytłumaczne że according to fitness to w praktyce mniejsze pierwsze jeśli populację mamy posortowaną
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
                if not self._are_any_arrays_equal([current, pbest, r1, r2]):
                    break
            new = current + realizations_F[i]*(pbest-current) + realizations_F[i]*(r1-r2)
            Crb = self._update_Crb(state.timestep, max_iters)
            if np.random.rand() < 0.5:
                new = self._binomial_crossover(current, new, Crb)
            else:
                new = self._exponential_crossover(current, new, realizations_Cr[i])
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
        new_NP = self._nlpsr(state.timestep, max_iters, NPmin, NPmax)
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
            new_archive_probability = self._calculate_archive_probability(q_deltas)
            new_M_F = self._calculate_new_memory(state.Mk, state.M_F, S_F, q_deltas)
            new_M_Cr = self._calculate_new_memory(state.Mk, state.M_Cr, S_Cr, q_deltas)
        else:
            new_archive_probability = state.archive_probability
            new_M_F = state.M_F
            new_M_Cr = state.M_Cr
        new_Mk = (state.Mk+1)%H
        new_timestep = state.timestep + 1
        return new_timestep, pop, new_archive, new_archive_probability, new_NP, new_NA, new_M_F, new_M_Cr, new_Mk

    def _calculate_archive_probability(self, deltas):
        deltas_archive = deltas[deltas[:,1]==True, 0]
        deltas_pop = deltas[deltas[:,1]==False, 0]
        sum_da = np.sum(deltas_archive)
        sum_dp = np.sum(deltas_pop)
        ratio_da = len(deltas_archive)/len(deltas)
        ratio_dp = len(deltas_pop)/len(deltas)
        return np.clip(sum_da/(ratio_da+EPSILON)/(sum_da/(ratio_da+EPSILON) + sum_dp/(ratio_dp+EPSILON)), 0.1, 0.9)

    def _calculate_new_memory(self, k, memory, samples, deltas):
        deltas = deltas[:,0]
        weights = deltas/np.sum(deltas)
        mean_w_L = np.sum(weights*(samples**2))/np.sum(weights*samples)
        memory[k] = (0.5 * memory[k] + 0.5 * mean_w_L).item()
        return memory

    def _binomial_crossover(self, x, u, Crb):
        new_specimen = np.copy(x)
        jrand = np.random.randint(0,len(new_specimen))
        for i in range(len(new_specimen)):
            if np.random.rand() < Crb or i == jrand:
                new_specimen[i] = u[i]
        return new_specimen

    def _exponential_crossover(self, x, u, Cr_i):
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

    def _nlpsr(self, g, max_iters, NPmin, NPmax):
        nfe_r = g/max_iters
        return round((NPmin-NPmax)*(nfe_r**(1-nfe_r))+NPmax)
    
    def _update_Crb(self, g, max_iters):
        if g < 0.5 * max_iters:
            return 0
        else: 
            return 2*(g/max_iters-0.5) # W artykule błąd we wzorze, w tekście jest mowa o ciągłości której we wzorze nie ma 

    def _nonstandard_cauchy(self, mean = 0.0, gamma = 1.0):
        return gamma * np.random.standard_cauchy(1) + mean

    def _are_any_arrays_equal(self, arrays):
        num_arrays = len(arrays)
        for i in range(num_arrays):
            for j in range(i + 1, num_arrays):
                if np.array_equal(arrays[i], arrays[j]):
                    return True
        return False

    def _eval(self, pop):
        if len(pop.shape) == 1: # single specimen
            pop = pop[None,:]
        return self.objective(pop)
    
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
                print_clean(f"Timestep: {state.timestep}\nCurrent Mean: {np.mean(state.population, axis=0)}\nEval: {self._eval(np.mean(state.population, axis=0))}")
        self.dump_history_to_file(f"src/checkpoints/lastnlshadersp.npy")

    def get_history_means(self):
        means = []
        for state in self.history:
            means.append(np.mean(state.population, axis=0))
        return np.array(means)
    
    def dump_history_to_file(self, file_path):
        np.save(file_path, np.array(self.history))

    def load_history_from_file(self, file_path):
        self.history = np.load(file_path, allow_pickle=True)
    
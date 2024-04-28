import numpy as np
from pathlib import Path

from configs.DESconfig import default_config
from utils.misc import print_clean


class StateDES:
    def __init__(self, timestep, population, delta) -> None:
        self.timestep = timestep
        self.population = population
        self.delta = np.ones_like(population[0]) * delta

class DES:
    def __init__(self, objective, population=None, config = {}) -> None:
        self.config = default_config() | config
        self.objective = objective
        self.once = True
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
        return StateDES(1, population, self.config["delta"])

    def _make_step(self, state, mi, c, f, d, h, e):
        pop_mean = np.mean(state.population, axis=0)
        q_pop_mean = self._eval(pop_mean) # ???? in paper this was written in pseudo code but does nothing XDD
        q_population = self._eval(state.population)

        sorted_args = np.argsort(q_population)
        sorted_pop = state.population[sorted_args]

        best_mean = np.mean(sorted_pop[:mi, :], axis=0)
        next_delta = (1 - c) * state.delta + c * (best_mean - pop_mean)

        next_pop = []
        for _ in range(state.population.shape[0]):
            hw = np.random.randint(1, min(h,len(self.history))) if len(self.history) > 1 else 1
            j,k = np.random.randint(0,mi), np.random.randint(0,mi)
            sample = f * (self.history[-hw].population[j,:] - self.history[-hw].population[k,:]) + next_delta * d * np.random.normal(0,1)
            next_pop.append( (best_mean+ sample + e * np.random.normal(0,1, sample.shape))[None, :])
        return np.concatenate(next_pop, axis=0), next_delta, best_mean

    def _eval(self, pop):
        if len(pop.shape) == 1: # single specimen
            pop = pop[None,:]
        return self.objective(pop)
    
    def _calc_end_value(self, population, best_mean):
        sum = 0.0
        for dimension in range(population.shape[1]):
            sum += np.sqrt(np.sum([(a-best_mean[dimension])**2 for a in population[:,dimension]]))
        return sum/population.shape[1]


    def run(self):
        max_n_epochs = self.config["max_n_epochs"]
        c = self.config["delta_change_rate"]
        f = self.config["scalling_factor"]
        d = self.config["small_delta"] # in paper small_delta = E||N(0,I)||, is it weirdly written? Norm of single random variable realisation should be equal to its absolute value?.......
        h = self.config["history_window_size"]
        mi = self.config["offspring_size"]
        e = self.config["noise_intensity"]

        state = self.history[-1]

        t = 1
        end_cond_value = float("inf")
        while end_cond_value >= e and t < max_n_epochs:
            next_pop, next_delta, best_mean = self._make_step(state, mi, c, f, d, h, e)
            end_cond_value = self._calc_end_value(next_pop, best_mean)
            t = t+1
            state = StateDES(t, next_pop, next_delta)
            self.history.append(state)
            if t%100==0:
                print_clean(f"Timestep: {t}\nCurrent Mean: {best_mean}\nEval: {self._eval(best_mean)}\nCurrent End Value: {end_cond_value}\nEnd Value: {e}")
        self.dump_history_to_file(f"src/checkpoints/lastDES.npy")

    def get_history_means(self):
        means = []
        for state in self.history:
            means.append(np.mean(state.population, axis=0))
        return np.array(means)
    
    def dump_history_to_file(self, file_path):
        np.save(file_path, np.array(self.history))

    def load_history_from_file(self, file_path):
        self.history = np.load(file_path, allow_pickle=True)


if __name__ == "__main__":
    sorted(zip([-0.99999989, -0.99999889, -0.99999789], np.array([[2.00011378, 2.00050722, 1.99982995, 2.00065993, 2.00058674],[2.00011357, 2.00050722, 1.99983,    2.00065984, 2.00058621], [2.00011342, 2.00050738, 1.99982982, 2.00066011, 2.00058601]])), reverse=False)

        
from dataclasses import dataclass


@dataclass
class NLSHADERSPConfig:
    population_size_max: int
    population_size_min: int
    problem_dimensions: int
    population_size: int
    max_iterations: int
    history_window_size: int
    memory_F: float
    memory_Cr: float
    archive_size: int
    archive_probability: float
    best_part_of_pop: float

    def __init__(self, problem_dimensions=10):
        self.population_size_max = 30 * problem_dimensions
        self.population_size_min = int(0.25 * problem_dimensions)
        self.problem_dimensions = problem_dimensions
        self.population_size = self.population_size_max
        self.max_iterations = problem_dimensions*1000
        self.history_window_size = 20 * problem_dimensions
        self.memory_F = 0.2
        self.memory_Cr = 0.2
        self.archive_size = int(2.1 * self.population_size)
        self.archive_probability = 0.5
        self.best_part_of_pop = 0.2

        # init population params
        self.new_population_mean = 5.0,
        self.new_population_variance = 3.0

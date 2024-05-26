from dataclasses import dataclass
from math import sqrt

from src.configs.BaseAgentConfig import BaseAgentConfig


@dataclass
class DESConfig(BaseAgentConfig):
    problem_dimensions: int
    max_iterations: int
    population_size: int
    mu: int
    history_window_size: int
    delta_change_rate: float
    find_max: bool
    new_population_mean: float
    new_population_variance: float
    delta: int
    scaling_factor: float
    small_delta: float
    noise_intensity: float

    def __init__(self, problem_dimensions=10):
        self.problem_dimensions = problem_dimensions
        self.max_iterations = problem_dimensions*10000
        self.population_size = problem_dimensions * 5
        self.mu = self.population_size // 2

        self.history_window_size = int(6 + 5 * sqrt(problem_dimensions))
        self.delta_change_rate = 4 / (problem_dimensions + 4)
        self.find_max = False

        self.new_population_mean = 5.0
        self.new_population_variance = 3.0
        self.delta = 0
        self.scaling_factor = 1 / sqrt(2)
        self.small_delta = 0.797885
        self.noise_intensity = 1e-8 * 0.797885

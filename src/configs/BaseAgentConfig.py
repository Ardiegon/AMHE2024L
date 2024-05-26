from dataclasses import dataclass


@dataclass
class BaseAgentConfig:
    problem_dimensions: int
    population_size: int
    new_population_mean: float
    new_population_variance: float

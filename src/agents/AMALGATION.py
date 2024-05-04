import numpy as np
from dataclasses import dataclass

import utils.algorithmic as alg 
from agents.base import BaseAgent, BaseAgentState
from configs.AMALGATIONconfig import default_config
from utils.misc import print_clean

@dataclass
class StateAMALGATION(BaseAgentState):
    pass

class AMALGATION(BaseAgent):
    def __init__(self, objective, population=None, config = {}) -> None:
        super().__init__(objective, population, config = default_config() | config)
    
    def _init_state(self, population):
        # Do zrobienia
        return StateAMALGATION(0, population)

    def _make_step(self, state):
        # Do zrobienia
        pass
    
    def run(self):
        # Do zrobienia
        # pewnie mało się zmieni
        max_iters = self.config["max_iters"]
        state = self.history[-1]

        while state.timestep < max_iters:
            step_result = self._make_step(state) # self._make_step(state, i inne...)
            state = StateAMALGATION(*step_result)
            self.history.append(state)
            if state.timestep%100==0:
                print_clean(f"Timestep: {state.timestep}\nCurrent Mean: {np.mean(state.population, axis=0)}\nEval: {self._eval(np.mean(state.population, axis=0))}")
        self.dump_history_to_file(f"src/checkpoints/lastamalgation.npy")

    
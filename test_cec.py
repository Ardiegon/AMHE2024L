import argparse
import numpy as np
from time import time
from pathlib import Path
import yaml

from src.agents.agents_dictionary import get_agent, agent_choices, get_agent_config
from src.obj_func.function_dictionary import get_objective

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", type=str, required=True, choices=agent_choices(),
                        help="Name of an agent you want to use")
    return parser.parse_args()

def main(args):
    np.random.seed(1)
    cec_optimum = [300.0, 400.0, 600.0, 800.0, 900.0, 1800.0, 2000.0, 2200.0, 2300.0, 2400.0, 2600.0, 2700.0]

    agent_class = get_agent(args.agent)
    config_class = get_agent_config(args.agent)
    all_loses = {}
    all_best_ones = {}
    all_timesteps = {}
    for cec_name, optimum in zip([f"cec-{x}" for x in range(1,13)], cec_optimum):
        dimension = 2 if cec_name not in ["cec-7", "cec-8"] else 10
        config = config_class(dimension)
        objective = get_objective(cec_name)
        agent = agent_class(objective, config)
        best_one, final_timestep = agent.run(optimum)
        loss = objective(np.expand_dims(best_one, axis=0))
        all_loses[cec_name] = float(loss[0])
        all_best_ones[cec_name] = best_one.tolist()
        all_timesteps[cec_name] = final_timestep
    with open(Path("results") /  f'losses_{args.agent}.yaml', 'w') as file:
        yaml.dump(all_loses, file)
    with open(Path("results") /  f'best_ones_{args.agent}.yaml', 'w') as file:
        yaml.dump(all_best_ones, file)
    with open(Path("results") /  f'timesteps_{args.agent}.yaml', 'w') as file:
        yaml.dump(all_timesteps, file)
if __name__ == "__main__":
    args = parse_args()
    main(args)
import argparse
import numpy as np

from src.agents.agents_dictionary import get_agent, agent_choices, get_agent_config
from src.obj_func.function_dictionary import get_objective, objective_choices
from obj_func.visualiser import plot_population_on_objective

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", type=str, required=True, choices=agent_choices(),
                        help="Name of an agent you want to use")
    parser.add_argument("-o", "--objective", type=str, required=True, choices=objective_choices(),
                        help="Name of an objective agent will optimize")
    parser.add_argument("-s", "--history", type=str, required=True,
                        help="Path to agent history")
    parser.add_argument("-i", "--interval", type=int, default=100,
                        help="Interval between drawing next samples from history")
    parser.add_argument("-n", "--problem_dimension", type=int, required=True,
                        help="Number of problems dimensions")
    return parser.parse_args()

def main(args):
    agent_class = get_agent(args.agent)
    objective = get_objective(args.objective)
    config_class = get_agent_config(args.agent)
    config = config_class(args.problem_dimension)
    agent = agent_class(objective, config)
    agent.load_history_from_file(args.history)
    means = agent.get_history_means()
    plot_population_on_objective(objective, means[::args.interval, :], 0, 1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
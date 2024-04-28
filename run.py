import argparse
import numpy as np

from agents import get_agent, agent_choices
from obj_func import get_objective, objective_choices

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", type=str, required=True, choices=agent_choices(),
                        help="Name of an agent you want to use")
    parser.add_argument("-o", "--objective", type=str, required=True, choices=objective_choices(),
                        help="Name of an objective agent will optimize")
    return parser.parse_args()

def main(args):
    agent_class = get_agent(args.agent)
    objective = get_objective(args.objective)
    agent = agent_class(objective)
    agent.run()

if __name__ == "__main__":
    args = parse_args()
    main(args)
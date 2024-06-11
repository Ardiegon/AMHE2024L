import argparse
import numpy as np
from time import time

from src.agents.agents_dictionary import get_agent, agent_choices, get_agent_config
from src.obj_func.function_dictionary import get_objective, objective_choices


# TODO
# sensowne logowanie - zależnie od tego jakie badania chcemy przeprowadzić
# budżetowanie na liczbę ewaluacji - nice to have
# configi na klasy - nice to have

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", type=str, required=True, choices=agent_choices(),
                        help="Name of an agent you want to use")
    parser.add_argument("-o", "--objective", type=str, required=True, choices=objective_choices(),
                        help="Name of an objective agent will optimize")
    parser.add_argument("-n", "--problem_dimension", type=str, required=True,
                        help="Number of problems dimensions")
    return parser.parse_args()

def main(args):
    np.random.seed(1)
    problem_dimension = int(args.problem_dimension)  # TODO dodać do configa
    agent_class = get_agent(args.agent)
    objective = get_objective(args.objective)
    config_class = get_agent_config(args.agent)
    config = config_class(problem_dimension)
    agent = agent_class(objective, config)
    
    start = time()
    agent.run()
    print("Elapsed:", time()-start)

if __name__ == "__main__":
    args = parse_args()
    main(args)
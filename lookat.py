import argparse

from agents import get_agent, agent_choices
from obj_func import get_objective, family_choices, objective_choices, repr_choices
from obj_func.visualiser import plot_population_on_objective

class CustomHelpFormatter(argparse.HelpFormatter):
    def format_help(self):
        original_help = super().format_help()
        return original_help + repr_choices()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument("-a", "--agent", type=str, required=True, choices=agent_choices(),
                        help="Name of an agent you want to use")
    parser.add_argument("-f", "--family", type=str, required=True, choices=family_choices(),
                        help="Name of an objective family")
    args = parser.parse_known_args()[0]
    parser.add_argument("-o", "--objective", type=str, required=True, choices=objective_choices(args.family),
                        help="Name of an objective agent will optimize")
    parser.add_argument("-s", "--history", type=str, required=True,
                        help="Path to agent history")
    parser.add_argument("-i", "--interval", type=int, default=100,
                        help="Interval between drawing next samples from history")
    return parser.parse_args()

def main(args):
    agent_class = get_agent(args.agent)
    objective = get_objective(args.family, args.objective)
    agent = agent_class(objective)
    agent.load_history_from_file(args.history)
    means = agent.get_history_means()
    plot_population_on_objective(objective, means[::args.interval, :], 0, 1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
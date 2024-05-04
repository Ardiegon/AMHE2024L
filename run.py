import argparse

from agents import get_agent, agent_choices
from obj_func import get_objective, family_choices, objective_choices, repr_choices

class CustomHelpFormatter(argparse.HelpFormatter):
    def format_help(self):
        original_help = super().format_help()
        return original_help + "\n Here are possible options to use as family and objective" + repr_choices()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument("-a", "--agent", type=str, required=True, choices=agent_choices(),
                        help="Name of an agent you want to use")
    parser.add_argument("-f", "--family", type=str, required=True, choices=family_choices(),
                        help="Name of an objective family")
    args = parser.parse_known_args()[0]
    parser.add_argument("-o", "--objective", type=str, required=True, choices=objective_choices(args.family),
                        help="Name of an objective agent will optimize")
    return parser.parse_args()

def main(args):
    agent_class = get_agent(args.agent)
    objective = get_objective(args.family, args.objective)
    agent = agent_class(objective)
    best = agent.run()
    print(f"Best found specimen is: {best}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
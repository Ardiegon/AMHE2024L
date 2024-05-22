from src.agents.DES import DES
from src.agents.NLSHADERSP import NLSHADERSP
from src.agents.hybridization import Hybridization


AGENTS = {
    "des": DES,
    "nlshadersp": NLSHADERSP,
    "hybridization": Hybridization
}


def get_agent(name):
    return AGENTS[name]


def agent_choices():
    return list(AGENTS.keys())

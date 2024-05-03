from agents.DES import DES
from agents.NLSHADERSP import NLSHADERSP
from agents.AMALGATION import AMALGATION

AGENTS = {
    "des": DES,
    "nlshadersp": NLSHADERSP,
    "amalgation": AMALGATION
}

def get_agent(name):
    return AGENTS[name]

def agent_choices():
    return list(AGENTS.keys())
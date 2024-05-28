from src.agents.Hybridization import Hybridization
from src.configs.HybridizationConfig import HybridizationConfig
from src.configs.NLSHADERSPConfig import NLSHADERSPConfig
from src.configs.DESConfig import DESConfig
from src.agents.DES import DES
from src.agents.NLSHADERSP import NLSHADERSP


AGENTS = {
    "des": DES,
    "nlshadersp": NLSHADERSP,
    "hybridization": Hybridization
}

AGENTS_CONFIGS = {
    "des": DESConfig,
    "nlshadersp": NLSHADERSPConfig,
    "hybridization": HybridizationConfig
}


def get_agent(name):
    return AGENTS[name]


def get_agent_config(name):
    return AGENTS_CONFIGS[name]


def agent_choices():
    return list(AGENTS.keys())

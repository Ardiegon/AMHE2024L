from src.configs.NLSHADERSPConfig import NLSHADERSPConfig
from src.configs.DESConfig import DESConfig
from src.configs.hybridizationConfig import HybridizationConfig
from src.agents.DES import DES
from src.agents.NLSHADERSP import NLSHADERSP
from src.agents.hybridization import Hybridization


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

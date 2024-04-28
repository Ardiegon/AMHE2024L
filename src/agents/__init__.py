from agents.DES import DES

AGENTS = {
    "des": DES,
    "nl-shade-rsp": NotImplementedError
}

def get_agent(name):
    return AGENTS[name]

def agent_choices():
    return list(AGENTS.keys())
import obj_func.simple as simple
import obj_func.cec as cec

OBJECTIVES = {
    "simple": simple.objective_wrapper,
    "cec": cec.objective_wrapper
}

NAMES = {
    "simple": simple.get_names(),
    "cec": cec.get_names()
}

def get_objective(family_name, func_desc):
    return OBJECTIVES[family_name](func_desc)

def repr_choices()->str:
    representation = "Here are possible options to use as family and objective"
    for family in list(NAMES.keys()):
        representation += "\n\t"
        representation += str(family)
        representation += "\n\t\t"
        for name in NAMES[family]:
            representation += str(name)
            representation += ", "
    representation += "\n"
    return representation

def family_choices():
    return list(OBJECTIVES.keys())

def objective_choices(family_name):
    return NAMES[family_name]



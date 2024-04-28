from obj_func.simple import quad, almost_twin_peaks

OBJECTIVES = {
    "quad": quad,
    "almost-twin-peaks": almost_twin_peaks,
    "cec-standard": lambda x: NotImplementedError()
}

def get_objective(name):
    return OBJECTIVES[name]

def objective_choices():
    return list(OBJECTIVES.keys())
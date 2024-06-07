from src.obj_func.simple import quad, almost_twin_peaks
from src.utils.misc import create_evaluate_cec_function

OBJECTIVES = {
    "quad": quad,
    "almost-twin-peaks": almost_twin_peaks,
    "cec-1": create_evaluate_cec_function(1),
    "cec-2": create_evaluate_cec_function(2),
    "cec-3": create_evaluate_cec_function(3),
    "cec-4": create_evaluate_cec_function(4),
    "cec-5": create_evaluate_cec_function(5),
    "cec-6": create_evaluate_cec_function(6),
    "cec-7": create_evaluate_cec_function(7),
    "cec-8": create_evaluate_cec_function(8),
    "cec-9": create_evaluate_cec_function(9),
    "cec-10": create_evaluate_cec_function(10),
    "cec-11": create_evaluate_cec_function(11),
    "cec-12": create_evaluate_cec_function(12)
}

def get_objective(name):
    return OBJECTIVES[name]

def objective_choices():
    return list(OBJECTIVES.keys())
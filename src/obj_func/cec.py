import obj_func.cec_from_authors as orgcec
import numpy as np

FUNC_NAMES = [str(x) for x in range(1,13)]

class CecAdjusted():
    def __init__(self, func_num: int)->None:
        self.func = func_num  

    def objective(self, x:np.ndarray)->np.ndarray:
        x = np.transpose(x, (1,0))
        nx, mx = np.shape(x)
        ObjFunc = np.zeros((mx,))
        for i in range(mx):
            ObjFunc[i] = orgcec.cec22_test_func(x[:,i], nx, 1, self.func)
        return ObjFunc

def objective_wrapper(func_desc):
    CEC = CecAdjusted(func_num = int(func_desc))
    def cec_objective(x: np.ndarray)->np.ndarray:
        return CEC.objective(x)
    return cec_objective

def get_names():
    return FUNC_NAMES

if __name__ == "__main__":
    nx = 10
    mx = 30
    fx_n = 12

    x = 200.0*np.random.rand(mx,nx)*0.0-100.0
    print(x.shape)
    objective = objective_wrapper(fx_n)
    print(objective(x).shape)
    print(objective(x))
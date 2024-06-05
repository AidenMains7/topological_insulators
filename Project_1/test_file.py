from joblib import Parallel, delayed, parallel_backend
import numpy as np
from itertools import product


def disorder_single(iterations:int, num_jobs:int):
    pass

    def worker_1(i):
        pass

    data = Parallel(n_jobs=num_jobs)(delayed(worker_1)(j) for j in range(iterations))
    avg = np.average(data)
    return avg

def disorder_range(iterations:int, num_jobs:int, disorder_values:np.ndarray):
    pass

    def worker_2(i):
        value = disorder_single(iterations, num_jobs)

        return value    

    data = Parallel(n_jobs=num_jobs)(delayed(worker_2)(j) for j in range(disorder_values.size))
    return data

def bott_index(val1, val2) -> float:
    bott = np.random.rand()
    return bott


def compute(iterations:int, num_jobs:int, disorder_values:np.ndarray, M_vals:np.ndarray, B_vals:np.ndarray):
    pass

    params = tuple(product(M_vals, B_vals))

    def worker_3(i):
        M, B = params(i)
        bott = bott_index(M, B)

        if bott != 0:
            disorder_array = disorder_range(iterations, num_jobs, disorder_values)

        else:
            disorder_array = np.array([[0],[bott]])

        return disorder_array
        
        

    data = Parallel(n_jobs=num_jobs)(delayed(worker_3)(j) for j in range(len(params)))
    return data



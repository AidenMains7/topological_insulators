"""

Functions: 
init_environment: initializes the environment with specified number of cores per job as to not cannabalize
bott_from_disorder: computes the average bott index from a disorder strength 
many_disorder: iterates bott_from_disorder for a range of disorder strength values
many_lattices: compues the bott index, and if nonzero, the disorder over a range of values for a specified range of M and B_tilde
"""


#outline
import sys
sys.path.append(".")
from Week_1.project_dependencies import mass_disorder, projector, bott_index, precompute_lattice, Hamiltonian_reconstruct
from joblib import Parallel, delayed
import numpy as np
from itertools import product
from time import time
from datetime import datetime
import os
import pandas as pd

from ProjectCode.PhaseDiagram.PhaseDiagramDependencies import precompute_data

def init_environment(cores_per_job):
    ncore = str(int(cores_per_job))
    os.environ["OMP_NUM_THREADS"] = ncore
    os.environ["OPENBLAS_NUM_THREADS"] = ncore
    os.environ["MKL_NUM_THREADS"] = ncore
    os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
    os.environ["NUMEXPR_NUM_THREADS"] = ncore

def bott_from_disorder(H_init:np.ndarray, lattice:np.ndarray, W:float, iterations:int, fermi_energy:float=0.0, num_jobs:int=4, cores_per_job:int=1, progress=False):

    init_environment(cores_per_job)

    system_size = np.max(lattice) + 1
    t0 = time()

    def do_iter(i):
        try:
            disorder_operator = mass_disorder(strength=W, system_size=system_size, df=2, sparse=False, type='uniform')
            H_new = H_init + disorder_operator
            P = projector(H_new, fermi_energy)
            bott = bott_index(P, lattice)

            dt = time() - t0
            if progress:
                print(f'W = {W:.2f}: {100*(i+1)/iterations:.2f}%, {round(dt)}s')

            return bott
        except Exception as e:
            print(f"An error occurred for W={W:.3f}: {e}")
            return np.nan

    data = np.array(Parallel(n_jobs=num_jobs)(delayed(do_iter)(j) for j in range(iterations)))
    data = data[~np.isnan(data)]
    bott_mean = np.mean(data)
    return bott_mean

def many_disorder(H, lattice, W_values, iterations_per_disorder, fermi_energy, num_jobs, cores_per_job, progresses=(True, False), printparams=None):

    init_environment(cores_per_job)

    t0 = time()
    def compute_single(i):
        try:
            W = W_values[i]
            bott_final = bott_from_disorder(H, lattice, W, iterations=iterations_per_disorder, fermi_energy=fermi_energy, num_jobs=num_jobs, cores_per_job=cores_per_job, progress=progresses[0])
        
            dt = time() - t0
            if progresses[1]:
                print(f"Finished W={W}, {printparams}; {round(dt,1)}s")
    
            return W, bott_final
        
        except Exception as e:
            print(f"Error at W={W}: {e}")
            return [np.nan]*2

    data = np.array(Parallel(n_jobs=num_jobs)(delayed(compute_single)(j) for j in range(W_values.size))).T
    return data

def many_lattices(method, order, pad_w, pbc, n, M_values, B_tilde_values, W_values, iterations_per_disorder, fermi_energy, num_jobs, cores_per_job, progresses=(True,True,False)):
    
    init_environment(cores_per_job)

    if method not in ['symmetry']:
        raise ValueError("Method is incorrect.")
    if method == 'symmetry' and n is None:
        raise ValueError("When using symmetry, n must not be None.")

    pre_data, frac_lat = precompute_data(method=method, order=order, pad_width=pad_w, pbc=pbc, n=n)
    parameter_values = tuple(product(M_values, B_tilde_values))

    t00 = time()

    def compute_single(i):
        M, B_tilde = parameter_values[i]

        if method == "symmetry":
            H = Hamiltonian_reconstruct(method=method, pre_data=pre_data, M=M, B_tilde=B_tilde, sparse=False)
            P = projector(H, fermi_energy=fermi_energy)
            bott_init = bott_index(P, frac_lat)

        dt = time() - t00
        if progresses[0]:
            print(f'M={M}, B_tilde={B_tilde}; {100*(i+1)/len(parameter_values):.2f}%, {round(dt)}s.....bott={bott_init}')

        if bott_init != 0:
            bott_disorder_array = many_disorder(H, frac_lat, W_values=W_values, iterations_per_disorder=iterations_per_disorder, fermi_energy=fermi_energy, num_jobs=num_jobs, cores_per_job=cores_per_job, progresses = (progresses[1], progresses[2]), printparams=f"M={M}, B_tilde={B_tilde}")
        else:
            bott_disorder_array = np.full((2,W_values.size),np.nan)

        return M, B_tilde, bott_init, bott_disorder_array
    
    data = Parallel(n_jobs=num_jobs)(delayed(compute_single)(j) for j in range(len(parameter_values)))

    data_array = np.empty(len(parameter_values), dtype = np.dtype([('val1', 'f4'), ('val2', 'f4'), ('val3', 'f4'), ('array', 'O')]))
    for i in range(len(parameter_values)):
        data_array[i] = data[i]

    return data_array

def main():
    pass


if __name__ == "__main__":
    main()
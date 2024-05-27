#outline
import sys
sys.path.append(".")
from Week_1.project_dependencies import mass_disorder, projector, bott_index, precompute_lattice, Hamiltonian_reconstruct
from joblib import Parallel, delayed
import numpy as np
from itertools import product
from time import time
import os

from ProjectCode.PhaseDiagram.PhaseDiagramDependencies import precompute_data

def init_environment(cores_per_job):
    ncore = str(int(cores_per_job))
    os.environ["OMP_NUM_THREADS"] = ncore
    os.environ["OPENBLAS_NUM_THREADS"] = ncore
    os.environ["MKL_NUM_THREADS"] = ncore
    os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
    os.environ["NUMEXPR_NUM_THREADS"] = ncore

def bott_from_disorder(H_init:np.ndarray, lattice:np.ndarray, W:float, iterations:int, fermi_energy:float=0.0, num_jobs:int=4, cores_per_job:int=1, progress=True):

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

def many_disorder(H, lattice, W_values, iterations_per_disorder, fermi_energy, num_jobs, cores_per_job, progress, printparams=None):

    init_environment(cores_per_job)

    t0 = time()
    def compute_single(i):
        try:
            W = W_values[i]
            bott_final = bott_from_disorder(H, lattice, W, iterations=iterations_per_disorder, fermi_energy=fermi_energy, num_jobs=num_jobs, cores_per_job=cores_per_job, progress=progress)
        
            dt = time() - t0
            if True:
                print(f"Finished W={W}, {printparams}; {round(dt,1)}s")
    
            return W, bott_final
        
        except Exception as e:
            print(f"Error at W={W}: {e}")
            return [np.nan]*3

    data = np.array(Parallel(n_jobs=num_jobs)(delayed(compute_single)(j) for j in range(W_values.size))).T
    return data

def many_lattices(method, order, pad_w, pbc, n, M_values, B_tilde_values, W_values, iterations_per_disorder, fermi_energy, num_jobs, cores_per_job, progresses=(True,True)):
    
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
            print(f'{100*(i+1)/len(parameter_values):.2f}%, {round(dt)}s')

        if bott_init != 0:
            bott_disorder_array = many_disorder(H, frac_lat, W_values=W_values, iterations_per_disorder=iterations_per_disorder, fermi_energy=fermi_energy, num_jobs=num_jobs, cores_per_job=cores_per_job, progress = progresses[1], printparams=f"M={M}, B_tilde={B_tilde}")
        else:
            bott_disorder_array = None

        return M, B_tilde, bott_init, bott_disorder_array
    
    data = np.array(Parallel(n_jobs=num_jobs)(delayed(compute_single)(j) for j in range(len(parameter_values)))).T

    data = data.T
    return data

def main():
    method = "symmetry"
    order = 3
    pad_w = 0
    pbc = True
    n = 5
    M_values = np.linspace(-1, 1, 11)
    B_tilde_values = np.linspace(-1, 1, 11)
    W_values = np.linspace(0, 10, 11)
    iterations_per_disorder = 10
    fermi_energy = 0.0

    data = many_lattices(method=method, order=order, pad_w=pad_w, pbc=pbc, n=n, M_values=M_values, B_tilde_values=B_tilde_values, W_values=W_values, 
                         iterations_per_disorder=iterations_per_disorder, fermi_energy=fermi_energy, num_jobs=4, cores_per_job=1, progresses=(True, False))
    np.savetxt("data.txt",data)



if __name__ == "__main__":
    main()
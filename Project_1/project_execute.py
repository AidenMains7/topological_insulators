

import sys
sys.path.append(".")
import numpy as np
import os
from time import time
from project_dependencies import mass_disorder, projector, bott_index, precompute, Hamiltonian_reconstruct
from itertools import product
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from datetime import datetime

def init_environment(cores_per_job): #initiate cores to prevent cannibalization
    ncore = str(int(cores_per_job))
    os.environ["OMP_NUM_THREADS"] = ncore
    os.environ["OPENBLAS_NUM_THREADS"] = ncore
    os.environ["MKL_NUM_THREADS"] = ncore
    os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
    os.environ["NUMEXPR_NUM_THREADS"] = ncore

def bott_from_disorder(H_init:np.ndarray, lattice:np.ndarray, W:float, iterations:int, E_F:float=0.0, num_jobs:int=4, cores_per_job:int=1, progress=False) -> float:
    """
    Finds the average Bott index from random disorder of given strength W.

    Parameters:
    H_init (ndarray): initial Hamiltonian
    lattice (ndarray): lattice arrary
    W (float): Strength of disorder
    iterations (int): number of times to compute / average over
    E_F (float): fermi energy
    num_jobs (int): number of jobs 
    cores_per_job (int): CPU cores per job
    progress (bool): Whether to display progress

    Returns:
    bott_mean (float): average bott index over iterations
    """
    init_environment(cores_per_job)

    #size
    system_size = np.max(lattice) + 1
    t0 = time()

    #get the bott index from disorder a single time
    def do_iter(i):
        try:
            disorder_operator = mass_disorder(strength=W, system_size=system_size, df=2, sparse=False)
            H_new = H_init + disorder_operator
            P = projector(H_new, E_F)
            bott = bott_index(P, lattice)

            dt = time() - t0
            if progress:
                print(f'W = {W:.2f}: {100*(i+1)/iterations:.2f}%, {round(dt)}s')

            return bott
        except Exception as e:
            print(f"An error occurred for W={W:.2f}: {e}")
            return np.nan

    #do for the number of iterations
    data = np.array(Parallel(n_jobs=num_jobs)(delayed(do_iter)(j) for j in range(iterations)))
    
    #remove incomplete data
    data = data[~np.isnan(data)]

    #find the average
    bott_mean = np.mean(data)
    return bott_mean

def many_disorder(H:np.ndarray, lattice:np.ndarray, W_values:np.ndarray, iterations_per_disorder:int, E_F:float, num_jobs:int, cores_per_job:int, progresses:tuple[bool, bool]=(False, False), printparams:str=None) -> np.ndarray:
    """
    Calculuates the bott index after disorder over a range of disorder values.

    Parameters:
    H (ndarray): Hamiltonian
    lattice (ndarray): lattice array
    W_values (ndarray): Array of disorder values
    iterations_per_disorder (int): Iterations for each disorder
    E_F (float): fermi energy
    num_jobs (int): number of jobs 
    cores_per_job (int): CPU cores per job
    progress (tuple[bool, bool]): Whether to display progress. First index is for many_disorder, second index is for bott_from_disorder 
    printparams (str): Additional parameters to write into progress message

    Returns: 
    data (ndarray): 2xN array, where N is the length of W_values. First row is W_values, second row is final bott index. 
    """
    init_environment(cores_per_job)

    t0 = time()

    #compute disorder for a single W value
    def compute_single(i):
        try:
            W = W_values[i]
            bott_final = bott_from_disorder(H, lattice, W, iterations=iterations_per_disorder, E_F=E_F, num_jobs=num_jobs, cores_per_job=cores_per_job, progress=progresses[1])
        
            dt = time() - t0
            if progresses[0]:
                print(f"Finished W={W}, {printparams}; {round(dt,1)}s")
    
            return W, bott_final
        
        except Exception as e:
            print(f"Error at W={W}: {e}")
            return [np.nan]*2

    #compute disorder over the range of given values
    data = np.array(Parallel(n_jobs=num_jobs)(delayed(compute_single)(j) for j in range(W_values.size))).T
    return data

def many_lattices(method:str, order:int, pad_width:int, pbc:bool, n:int, M_values:np.ndarray, B_tilde_values:np.ndarray, W_values:np.ndarray, iterations_per_disorder:int, E_F:float, num_jobs:int=4, cores_per_job:int=1, sparse:bool=False, progresses:tuple[bool, bool, bool]=(False,False,False)) -> np.ndarray:
    """
    Computes many Sierpinski carpet lattices over a range of M and B_tilde values. If the bott index is not 0, compute the disorder over a range.

    Paramters:
    method (str): method to use
    order (int): order of the Sierpinski carpet
    pad_width (int): padding width of the lattice
    pbc (bool): periodic boundary conditions?
    n (int): maximum hopping distance 
    M_values (ndarray): array of values for M
    B_tilde_values (ndarray): array of values for B_tilde
    W_values (ndaray): array of values for disorder strength 
    iterations_per_disorder (int): number of times to compute each disorder value 
    E_F (float): fermi energy
    num_jobs (int): number of jobs
    cores_per_job (int): CPU cores per job
    sparse (bool): Whether to generate as a sprase matrix (currently not functional)
    progresses (tuple[bool, bool, bool]): whether to print progress update. Index 0 is for creating the lattice and finding its bott index. Index 1 is for many_disorder. Index 2 is for bott_from_disorder

    Returns: 
    data (ndarray): array containing the data for each computed lattice.
    """
    init_environment(cores_per_job)

    #precompute data and possible values
    pre_data, frac_lat = precompute(method=method, order=order, pad_width=pad_width, pbc=pbc, n=n, sparse=sparse)
    parameter_values = tuple(product(M_values, B_tilde_values))

    t0 = time()

    #compute a single bott index; if nonzero, apply disorder over a range
    def compute_single(i):
        M, B_tilde = parameter_values[i]

        #method of symmetry
        H = Hamiltonian_reconstruct(method=method, precomputed_data=pre_data, M=M, B_tilde=B_tilde, sparse=sparse)
        P = projector(H, E_F=E_F)
        bott_init = bott_index(P, frac_lat)

        dt = time() - t0
        
        #show progress?
        if progresses[0]:
            print(f'M={M:.2f}, B_tilde={B_tilde:.2f}; {100*(i+1)/len(parameter_values):.2f}%, {round(dt)}s.....bott={bott_init}')

        #if bott index is not 0
        if bott_init != 0:
            bott_disorder_array = many_disorder(H, frac_lat, W_values=W_values, iterations_per_disorder=iterations_per_disorder, E_F=E_F, num_jobs=num_jobs, cores_per_job=cores_per_job, progresses = (progresses[1], progresses[2]), printparams=f"M={M}, B_tilde={B_tilde}")
        
        #othe
        else:
            bott_disorder_array = np.full((2,W_values.size),np.nan)

        return M, B_tilde, bott_init, bott_disorder_array
    
    data = Parallel(n_jobs=num_jobs)(delayed(compute_single)(j) for j in range(len(parameter_values)))

    #array, custom dtype, each row is a different M, B_tilde
    data_array = np.empty(len(parameter_values), dtype = np.dtype([('val1', 'f4'), ('val2', 'f4'), ('val3', 'f4'), ('array', 'O')]))
    for i in range(len(parameter_values)):
        data_array[i] = data[i]

    return data_array

def plot_data(filepath) -> None:
    #load file
    read_data = np.load(filepath, allow_pickle=True)
    data = read_data['arr_0']

    #plot 
    for i in range(data.size):
        M = data[i][0]
        B_tilde = data[i][1]
        bott_init = data[i][2]
        bott_array = data[i][3]
        if (bott_init != 0):
            x = np.concatenate(([0],bott_array[0]))
            y = np.concatenate(([bott_init], bott_array[1]))
            plt.plot(x,y, label=f"M={M}, B_tilde={B_tilde}")

    #set plotting values
    plt.xlabel("W (disorder strength)")
    plt.ylabel("Bott Index")
    plt.grid()
    plt.show()

def timely_filename() -> str:
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hr = now.hour
    minute = now.minute
    time_string=f"{month}_{day}_{year}__{hr}_{minute}"
    return time_string

def main():
    methodlist= ['symmetry', 'square', 'renorm', 'site_elim']
    method = methodlist[0]
    order = 3
    pad_w = 0
    pbc = True
    n = 5
    M_values = np.linspace(-2, 12, 5)
    B_tilde_values = np.linspace(0, 2, 5)
    W_values = np.linspace(0.5, 5, 5)
    print("W_values = ",W_values)
    iterations_per_disorder = 10
    E_F = 0.0

    data = many_lattices(method=method, order=order, pad_width=pad_w, pbc=pbc, n=n, M_values=M_values, B_tilde_values=B_tilde_values, W_values=W_values,
                         iterations_per_disorder=iterations_per_disorder, E_F=E_F, progresses=(True, True, False))

    #save data

    filepath = f"Project_1\\bott_index_disorder_{timely_filename()}.npz"
    np.savez(filepath,data)
    plot_data(filepath=filepath)

def main2():
    pass
    

if __name__ == "__main__":
    main()




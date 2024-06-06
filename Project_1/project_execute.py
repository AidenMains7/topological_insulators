"""

"""

import sys
sys.path.append(".")
import numpy as np
import os
from time import time
from project_dependencies import mass_disorder, projector, bott_index, precompute, Hamiltonian_reconstruct
from itertools import product
from joblib import Parallel, delayed, parallel_backend

def _init_environment(cores_per_job): #initiate cores to prevent cannibalization
    ncore = str(int(cores_per_job))
    os.environ["OMP_NUM_THREADS"] = ncore
    os.environ["OPENBLAS_NUM_THREADS"] = ncore
    os.environ["MKL_NUM_THREADS"] = ncore
    os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
    os.environ["NUMEXPR_NUM_THREADS"] = ncore


def _many_bott(method:str, order:int, pad_width:int, pbc:bool, n:int, M_values:np.ndarray, B_tilde_values:np.ndarray, E_F:float, num_jobs:int=4, cores_per_job:int=1, sparse:bool=False, progress=True) -> np.ndarray:
    """
    Computes the Bott Index for the lattice  for every combination of M and B_tilde

    Parameters:


    Returns: 
    data (ndarray): array of shape (3, M_values.size*B_tilde_values.size)
    """
    
    pre_data, frac_lat = precompute(method=method, order=order, pad_width=pad_width, pbc=pbc, n=n, sparse=sparse)
    parameter_values = tuple(product(M_values, B_tilde_values))

    t0 = time()

    def compute_single(i):
        M, B_tilde = parameter_values[i]

        H = Hamiltonian_reconstruct(method=method, precomputed_data=pre_data, M=M, B_tilde=B_tilde, sparse=sparse)
        P = projector(H, E_F=E_F)
        bott = bott_index(P, frac_lat)



        if progress:
            dt = time() - t0
            percent_message = f"{(100*(i+1)/len(parameter_values)):.2f}%"
            message_2 = f"Completed: (M, B_tilde, bott) = ({M:.2f}, {B_tilde:.2f}, {bott:.2f})"
            message_3 = f"{dt:.1f}s"

            print(f"{percent_message.ljust(10)} {message_2.ljust(15)} {message_3.rjust(5)}")

        return np.array([M, B_tilde, bott])
    
    data = np.array(Parallel(n_jobs=num_jobs)(delayed(compute_single)(j) for j in range(len(parameter_values)))).T
    return data


def _disorder(H_init:np.ndarray, lattice:np.ndarray, W:float, iterations:int, E_F:float=0.0, num_jobs:int=4, cores_per_job:int=1, progress=False) -> float:
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

    #size
    system_size = np.max(lattice) + 1
    t0 = time()

    #get the bott index from disorder a single time
    def do_iter(i):
        disorder_operator = mass_disorder(strength=W, system_size=system_size, df=2, sparse=False)
        H_new = H_init + disorder_operator
        P = projector(H_new, E_F)
        bott = bott_index(P, lattice)


        if progress:
            dt = time() - t0
            print(f'W = {W:.2f}: {100*(i+1)/iterations:.2f}%, {round(dt)}s')

        return bott

    #do for the number of iterations

    data = np.array(Parallel(n_jobs=num_jobs)(delayed(do_iter)(j) for j in range(iterations)))
    
    #remove incomplete data
    data = data[~np.isnan(data)]

    #find the average
    bott_mean = np.mean(data) if len(data) > 0 else np.nan
    return bott_mean


def _disorder_range(H:np.ndarray, lattice:np.ndarray, W_values:np.ndarray, iterations_per_disorder:int, E_F:float, num_jobs:int, cores_per_job:int, progresses:"tuple[bool, bool]"=(False, False), printparams:str=None) -> np.ndarray:
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

    t0 = time()

    #compute disorder for a single W value
    def compute_single(i):
        try:
            W = W_values[i]
            bott_final = _disorder(H, lattice, W, iterations=iterations_per_disorder, E_F=E_F, num_jobs=num_jobs, cores_per_job=cores_per_job, progress=progresses[1])
        

            if progresses[0]:
                dt = time() - t0
                message_1 = f"{printparams}"
                message_2 = f"({100*(i+1)/W_values.size:.2f}%)"
                message_3 = f"Finished W = {W:.2f}"

                print(f"{message_1.ljust(10)} {message_2.ljust(15)} {message_3.rjust(5)}")
    
            return W, bott_final
        
        except Exception as e:
            print(f"Error at W={W:.2f}: {e}")
            return [np.nan]*2

    #compute disorder over the range of given values

    data = np.array(Parallel(n_jobs=num_jobs)(delayed(compute_single)(j) for j in range(W_values.size))).T
    
    return data


def computation(method:str, order:int, pad_width:int, pbc:bool, n:int, M_values:np.ndarray, B_tilde_values:np.ndarray, W_values:np.ndarray, iterations_per_disorder:int, E_F:float=0.0, num_jobs:int=4, cores_per_job:int=1, sparse:bool=False, progresses:"tuple[bool, bool, bool]"=(True,False,False)) -> np.ndarray:
    """
    Given array ranges of M, B_tilde, and W, find the Bott Index for a given combination of (M, B_tilde). If it is non-zero, generate disorder over a specified range and find the resulting Bott Index.

    Parameters: 
    method (str): Which method to use. WARNING: 'renorm' is currently not functioning properly
    order (int): order of the Sierpinski carpet
    pad_width (int): width of padding of the lattice
    pbc (bool): periodic boundary conditions
    n (int): maximum hop distance (for method of symmetry)
    M_values (ndarray): range of values for M
    B_tilde_values (ndarray): range of values for B_tilde
    W_values (ndarray): range of values for disorder strength
    iterations_per_disorder (int): number of times to compute a given disorder strength (then average across all)
    E_F (float): fermi energy
    num_jobs (int): number of jobs to run
    cores_per_job (int): number of cores per job
    sparse (bool): Whether to generate as a sparse matrix WARNING: not currently functional
    progresses (tuple[bool, bool, bool]): Whether to report progress. Index 0 is for primary function, index 1 is for _many_disorder(), index 2 is for _bott_from_disorder()

    Returns:
    data (ndarray): Array of data, of shape (M_values.size * B_tilde_values.size, 2, W_values.size + 1). First column is [[0],[bott_init]], where bott_init is the Bott Index without disorder.
    """

    _init_environment(cores_per_job)

    pre_data, frac_lat = precompute(method, order, pad_width, pbc, n, sparse)
    parameters = tuple(product(M_values, B_tilde_values))

    t0 = time()

    def worker(i:int) -> np.ndarray:
        try: 
            #Compute bott values for parameter
            M, B_tilde = parameters[i]

            H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, sparse)
            P = projector(H, E_F)
            bott = bott_index(P, frac_lat)

            bott_array = np.array([[0.0], [bott]])


            if progresses[0]:
                dt = time() - t0
                percent_str = f"{100*(i+1)/len(parameters):.2f}%"
                bott_str = f"bott = {bott:+.2f}"
                time_str = f"{dt:.0f}s"
                print(f"{percent_str.ljust(10)} {bott_str.ljust(15)} {time_str.rjust(5)}")

            #If nonzero, disorder over the range
            if bott != 0:
                disorder_array = _disorder_range(H, frac_lat, W_values, iterations_per_disorder, E_F, num_jobs, cores_per_job, (progresses[1], progresses[2]), printparams=f"D: {100*(i+1)/len(parameters):.2f}%")
                
                disorder_array = np.concatenate((bott_array, disorder_array), axis=1)


                if progresses[0]:
                    dt = time() - t0
                    percent_str = f"{100*(i+1)/len(parameters):.2f}%"
                    message_str = "Completed disorder calculation."
                    time_str = f"{dt:.0f}s"
                    print(f"{percent_str.ljust(10)} {message_str.ljust(15)} {time_str.rjust(5)}")

            else:
                disorder_array = np.concatenate((bott_array, np.full((2, W_values.size), np.nan)), axis=1)

            return disorder_array

        except Exception as e:
            print(f"Error occured at M, B_tilde = {parameters[i]}: {e}")
            return np.full((2, W_values.size + 1), np.nan)
        

    data = np.array(Parallel(n_jobs=num_jobs)(delayed(worker)(j) for j in range(len(parameters))))
    
    return data


def computation_alt(method:str, order:int, pad_width:int, pbc:bool, n:int, M_values:np.ndarray, B_tilde_values:np.ndarray, W_values:np.ndarray, iterations_per_disorder:int, E_F:float=0.0, num_jobs:int=4, cores_per_job:int=1, sparse:bool=False, progresses:"tuple[bool, bool, bool]"=(False, False, False, True)) -> np.ndarray:
    """
    First, computes the Bott Index of each lattice for all combinations of M_values and B_tilde_values. Then, for those which have nonzero Bott Index, compute the disorder over the specified range.

    In comparison to computation(), we split to avoid three layers of Parallelization (here, we use two)
    
    Parameters: 
    method (str): Which method to use. WARNING: 'renorm' is currently not functioning properly
    order (int): order of the Sierpinski carpet
    pad_width (int): width of padding of the lattice
    pbc (bool): periodic boundary conditions
    n (int): maximum hop distance (for method of symmetry)
    M_values (ndarray): range of values for M
    B_tilde_values (ndarray): range of values for B_tilde
    W_values (ndarray): range of values for disorder strength
    iterations_per_disorder (int): number of times to compute a given disorder strength (then average across all)
    E_F (float): fermi energy
    num_jobs (int): number of jobs to run
    cores_per_job (int): number of cores per job
    sparse (bool): Whether to generate as a sparse matrix WARNING: not currently functional
    progresses (tuple[bool, bool, bool, bool]): Whether to report progress. Index 0 is for _many_boott(), index 1 is for _many_disorder(), index 2 is for _bott_from_disorder(), index 3 is for primary function.

    Returns:
    data (ndarray): Array of data, of shape (M_values.size * B_tilde_values.size, 2, W_values.size + 1). First column is [[0],[bott_init]], where bott_init is the Bott Index without disorder.
    
    """

    _init_environment(cores_per_job)

    #First, compute the Bott index of all lattices for the specified values.
    all_bott_array = _many_bott(method, order, pad_width, pbc, n, M_values, B_tilde_values, E_F, num_jobs, cores_per_job, sparse, progresses[0])

    #Find only the columns where the Bott Index is nonzero.
    mask = all_bott_array[2, :] != 0
    nonzero_bott = all_bott_array[:, mask]

    #precompute
    pre_data, frac_lat = precompute(method, order, pad_width, pbc, n, sparse)
    t0 = time()


    def worker(i):
        #get parameters
        M, B_tilde, bott_init = nonzero_bott[0,i], nonzero_bott[1,i], nonzero_bott[2,i]
        bott_array = np.array([[0],[bott_init]])
        param_array = np.array([[M],[B_tilde]])

        #create Hamiltonian
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, sparse)
        
        #disorder over the range
        disorder_array = _disorder_range(H, frac_lat, W_values, iterations_per_disorder, E_F, num_jobs, cores_per_job, (progresses[1], progresses[2]), printparams=f"D: {100*(i+1)/nonzero_bott.shape[1]:.2f}%")
    
        #add initial bott and 'disorder=0'
        disorder_array = np.concatenate((bott_array, disorder_array), axis=1)

        #Print progress
        if progresses[3]:
            dt = time() - t0
            percent_str = f"{100*(i+1)/nonzero_bott.shape[1]:.2f}%"
            message_str = "Completed disorder calculation."
            time_str = f"{dt:.0f}s"
            print(f"{percent_str.ljust(10)} {message_str.ljust(15)} {time_str.rjust(5)}")

        return disorder_array

    data = np.array(Parallel(n_jobs=num_jobs)(delayed(worker)(j) for j in range(nonzero_bott.shape[1])))
    return data




#----------main function implementation----------
def main():
    pass

if __name__ == "__main__":
    main()



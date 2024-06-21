"""

"""

import numpy as np
from itertools import product
from joblib import Parallel, delayed
from time import time
import os

from project_dependencies import mass_disorder, projector, bott_index, precompute, Hamiltonian_reconstruct



def init_environment(cores_per_job=1):
    ncore = str(int(cores_per_job))
    os.environ["OMP_NUM_THREADS"] = ncore
    os.environ["OPENBLAS_NUM_THREADS"] = ncore
    os.environ["MKL_NUM_THREADS"] = ncore
    os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
    os.environ["NUMEXPR_NUM_THREADS"] = ncore


# Parallel
def bott_many(method, order, pad_width, pbc, n, M_values, B_tilde_values, E_F, num_jobs, cores_per_job, progress_bott=True, sparse=False):
    """
    Computes the Bott Index for every combination of M and B_tilde

    Parameters: 
    method (str): method with which to construct the Hamiltonian
    order (int): order of the fractal lattice, will determine size of square if used
    pad_width (int): width of padding on the outside of the lattice
    pbc (bool): whether to impose periodic boundary conditions
    n (int): cutoff length for hopping; infinity norm
    M_values (ndarray): Range of M values; point mass
    B_tilde_values (ndarray): Range of B_tilde values; amplitude of hopping between same orbital
    E_F (float): Fermi energy
    num_jobs (int): Number of threads to use to compute
    cores_per_job (int): Number of cores per job
    progress_bott (bool): Display progress info
    sparse (bool): whether to generate as a sparse matrix

    Returns:
    bott_arr (ndarray): Array of shape (3, N). Row 0 is M value, row 1 is B_tilde value, row 2 is Bott Index
    """

    # Initialize environment to avoid cannibalization
    init_environment(cores_per_job)

    #precompute data
    pre_data, lattice = precompute(method=method, order=order, pad_width=pad_width, pbc=pbc, n=n, sparse=sparse)
    parameter_values = tuple(product(M_values, B_tilde_values))
    t0 = time()

    def do_single(i):
        # Parameters for single lattice
        M, B_tilde = parameter_values[i]

        # Construct hamiltonian, projector, compute bott index
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, sparse)
        P = projector(H, E_F)
        bott = bott_index(P, lattice)

        # Provide progress update
        if progress_bott:
            percent_message = f"{(100*(i+1)/len(parameter_values)):.2f}%"
            bott_message = f"Completed: (M, B_tilde, bott) = ({M:.2f}, {B_tilde:.2f}, {bott:+.0f})"
            time_message = f"{time()-t0:.0f}s"
            print(f"{percent_message.ljust(10)} {bott_message.ljust(15)} {time_message.rjust(5)}")

        return np.array([M, B_tilde, bott])
    
    bott_arr = np.array(Parallel(n_jobs=num_jobs)(delayed(do_single)(j) for j in range(len(parameter_values)))).T
    return bott_arr


def disorder_avg(H_init, lattice, W, iterations, E_F, progress=False, sparse=False):
    """
    Will calculate the average Bott Index from disorder for the specified number of iterations.

    Parameters:
    H_init (ndarray): Initial Hamiltonian
    lattice (ndarray): the lattice array
    W (float): Disorder strength
    iterations (int): Number of iterations to average over
    E_F (float): Fermi energy
    progress (bool): Display progress info
    sparse (bool): whether to generate as a sparse matrix

    Returns: 
    mean_bott (float): The average Bott Index over all iterations

    """
    # System size
    system_size = np.max(lattice) + 1

    # Initial time 
    t0 = time()


    def do_iter(i):
        # Calculate a disorder operator
        disorder_operator = mass_disorder(W, system_size, 2, sparse)
        
        # Add disorder to the Hamiltonian
        H_new = H_init + disorder_operator

        # Compute projector, bott index
        P = projector(H_new, E_F)
        bott = bott_index(P, lattice)

        # Provide progress update
        if progress:
            time_message = f"{time()-t0:.0f}s"
            value_message = f"Averaging over iterations: W = {W:.2f}"
            percent_message = f"({100*(i+1)/iterations:.2f}% complete)"
            print(f"{value_message.ljust(10)} {percent_message.ljust(15)} {time_message.rjust(5)}")

        return bott
    
    data = np.array([do_iter(j) for j in range(iterations)])
    return np.average(data)


# Parallel
def disorder_range(H, lattice, W_values, iterations, E_F, num_jobs, cores_per_job, progress_disorder_iter=False, progress_disorder_range=True, sparse=False):
    """

    Will find the Bott Index after disorder for each value in the provided range. 

    Parameters: 
    H (ndarray): Hamiltonian of the lattice
    lattice (ndarray): lattice array
    W_values (ndarray): Range of disorder strengths
    iterations (int): Number of iterations to average over
    E_F (float): Fermi energy
    num_jobs (int): Number of threads to use to compute
    cores_per_job (int): Number of cores per job
    progress_disorder_iter (bool): Display progress info for disorder_avg()
    progress_disorder_range (bool) Display progress for disorder_range()
    sparse (bool): whether to generate as a sparse matrix

    Returns: 
    data (ndarray): Array of shape (2, N). Row 0 is disorder value, row 1 is resultant Bott Index.

    """
    # Initialize environment to prevent cannibalization
    init_environment(cores_per_job)


    # Initial time
    t0 = time()

    def do_single(i):
        W = W_values[i]
        bott_final = disorder_avg(H, lattice, W, iterations, E_F, progress_disorder_iter, sparse)

        if progress_disorder_range:
            dt = time() - t0
            value_message = f"Disorder Range: W = {W:.2f}"
            percent_message = f"{100*(i+1)/W_values.size:.2f}%"
            time_message = f"{dt:.0f}s"
            print(f"{value_message.ljust(len(value_message))} {percent_message.ljust(10)} {time_message.rjust(0)}")

        return W, bott_final
    
    # Parallelization
    data = np.array(Parallel(n_jobs=num_jobs)(delayed(do_single)(j) for j in range(W_values.size))).T
    return data


def disorder_many(bott_arr, method, order, pad_width, pbc, n, W_values, iterations, E_F, num_jobs, cores_per_job, amount_per_idx=None, progress_disorder_iter=False, progress_disorder_range=True, progress_disorder_many=True, doStatistic=True, sparse=False):
    """
    Will find the resultant Bott Index from disorder over the provided range for all provided (M, B_tilde, bott_init) values.

    Parameters: 
    bott_arr (ndarray): (3, N) array where row 0 is M, row 1 is B_tilde, and row 2 is Bott Index 
    method (str): method with which to construct the Hamiltonian
    order (int): order of the fractal lattice, will determine size of square if used
    pad_width (int): width of padding on the outside of the lattice
    pbc (bool): whether to impose periodic boundary conditions
    n (int): cutoff length for hopping; infinity norm
    W_values (ndarray): Range of disorder strengths
    iterations (int): Number of iterations to average over
    E_F (float): Fermi energy
    num_jobs (int): Number of threads to use to compute
    cores_per_job (int): Number of cores per job
    amount_per_idx (int): Maximum number of lattices to compute for, per each unique Bott Index. If "None", will compute all.
    progress_disorder_iter (bool): Display progress info for disorder_avg()
    progress_disorder_range (bool): Display progress info for disorder_range()
    progress_disorder_many (bool): Display progress info for disorder_many()
    doStatistic (bool): Display info regarding the percent of how many are nonzero.
    sparse (bool): whether to generate as a sparse matrix

    Returns:
    data (ndarray): Array of shape (N, 2, W_values.size+1), containing the result data from disorder averaging over a range. 
    """


    # Unique Bott Index values, not including 0
    unique_values = list(np.unique(bott_arr[2]))
    unique_values.remove(0)
    
    # Separated into a list of arrays for each unique index (without 0)
    separated_arrs = [bott_arr[:, mask] for mask in [bott_arr[2, :] == val for val in unique_values]]

    # List of respective sizes
    separated_sizes = [arr.shape[1] for arr in separated_arrs]

    # Take only specified amount
    if amount_per_idx is not None:
        # Take only specified amount from each unique index; if more exist, take random ones (without replacement)
        limited_arr = [None]*len(unique_values)

        # Random number generator
        rng = np.random.default_rng()

        for i in range(len(unique_values)):
            size = separated_sizes[i]
            if size > amount_per_idx:
                # Get random indices without replacement
                random_idx = rng.choice(size, amount_per_idx, replace=False)
                limited_arr[i] = separated_arrs[i][:, random_idx]
            else: 
                limited_arr[i] = separated_arrs[i][:, :5]

        # Concatenate into single array
        nonzero_arr = np.concatenate(limited_arr, axis=1)

    else:
        nonzero_arr = np.concatenate(separated_arrs, axis=1)


    # Print statistic info
    if doStatistic:
        num_total = bott_arr.shape[1]
        num_nonzero = sum(separated_sizes)
        percent = 100*num_nonzero/num_total
        print(f"Of {num_total} total lattices, {num_nonzero} have a nonzero Bott Index ({percent:.2f}%).")

    # Precompute data
    pre_data, lattice = precompute(method, order, pad_width, pbc, n, sparse)
    t0 = time()

    # Disorder over given range for a single lattice
    def do_single(i):
        # Get values of lattice
        M, B_tilde, bott_init = tuple(nonzero_arr[:, i])

        # Construct the Hamiltonian
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, sparse)

        # Calculate the bott after disorder over the range
        disorder_arr = disorder_range(H, lattice, W_values, iterations, E_F, num_jobs, cores_per_job, progress_disorder_iter, progress_disorder_range, sparse)

        # Add the initial to the array
        disorder_arr = np.concatenate((np.array([[0], [bott_init]]), disorder_arr), axis=1)

        if progress_disorder_many:
            time_message = f"{time()-t0:.0f}s"
            percent_message = f"{100*(i+1)/nonzero_arr.shape[1]:.2f}%"
            value_message = f"Completed range calculation: (M, B_tilde, Bott) = ({M:+.2f}, {B_tilde:+.2f}, {bott_init:+.2f})"
            print(f"{percent_message.ljust(10)} {value_message.ljust(len(value_message))} {time_message.rjust(5)}")
        
        return disorder_arr
    
    data = np.empty((nonzero_arr.shape[1], 2, W_values.size+1))
    for j in range(nonzero_arr.shape[1]):
        data[j,:,:] = do_single(j)

    return data



#---------------------
def main():
    pass

if __name__ == "__main__":
    main()
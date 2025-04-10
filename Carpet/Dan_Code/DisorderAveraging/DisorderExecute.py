"""
Module for Parallel Computation of the Mean Bott Index with Disorder

This module provides tools for parallel computation of the mean Bott index
for a given base Hamiltonian with specified disorder strength. The computation
leverages the functionalities provided by the `DisorderDependencies` and `ComputeBottIndex`
modules and utilizes parallel processing to handle multiple realizations efficiently.

Functions:
- in_regions(point, regions): Check if a point lies within any specified regions.
- task_with_timeout(task_func, timeout, *args, **kwargs): Execute a task with a timeout.
- single_disorder_bott(H_base, lattice, W, num_realizations, E_F, num_jobs, cores_per_job, KPM, N, exact_log_regions, log_order, progress, task_timeout, **kwargs): Compute the mean Bott index for a disordered system.
- main(): Placeholder for main function implementation.
"""

import sys
sys.path.append(".")

import numpy as np
import os
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from Dan_Code.ComputeBottIndex import projector_exact, projector_KPM, bott_index
from scipy.sparse import csr_matrix
from time import time
from DisorderDependencies import H_and_lattice_wrapper, uniform_mass_disorder


def in_regions(point, regions):
    """
    Check if a point lies within any of the specified regions.

    Parameters:
    point (float): The point to check.
    regions (list of tuples): List of regions specified as (min, max) tuples.

    Returns:
    bool: True if the point lies within any region, False otherwise.
    """
    regions = np.array(regions)
    return np.any((regions[:, 0] <= point) & (point <= regions[:, 1]))


def task_with_timeout(task_func, timeout, *args, **kwargs):
    """
    Execute a task function with a specified timeout.

    Parameters:
    task_func (function): The function to execute.
    timeout (float): The maximum time to allow for execution in seconds.
    *args: Arguments to pass to the task function.
    **kwargs: Keyword arguments to pass to the task function.

    Returns:
    Any: The result of the task function or np.nan if it times out.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task_func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
        except TimeoutError:
            result = np.nan
    return result


def single_disorder_bott(H_base, lattice, W, num_realizations, E_F=0., num_jobs=28, cores_per_job=1, KPM=False, N=1024, exact_log_regions='all', log_order=20, progress=True, task_timeout=None, **kwargs):
    """
    Compute the mean Bott index for a Hamiltonian with disorder.

    Parameters:
    H_base (ndarray or csr_matrix): Base Hamiltonian matrix.
    lattice (ndarray): Lattice array indicating site positions.
    W (float): Disorder strength.
    num_realizations (int): Number of disorder realizations.
    E_F (float): Fermi energy. Default is 0.
    num_jobs (int): Number of parallel jobs. Default is 28.
    cores_per_job (int): Number of cores per job. Default is 1.
    KPM (bool): Whether to use the Kernel Polynomial Method (KPM). Default is False.
    N (int): Number of moments in the Chebyshev expansion (for KPM). Default is 1024.
    exact_log_regions (str or list of tuples): Regions for exact logarithm computation. Default is 'all'.
    log_order (int): Order of the power series expansion for logm. Default is 20.
    progress (bool): Whether to print progress information. Default is True.
    task_timeout (float): Timeout for each task in seconds. Default is None.
    **kwargs: Additional keyword arguments.

    Returns:
    float: Mean Bott index over all realizations.
    """
    # Set the number of threads for various libraries to avoid oversubscription
    ncore = str(int(cores_per_job))
    os.environ["OMP_NUM_THREADS"] = ncore
    os.environ["OPENBLAS_NUM_THREADS"] = ncore
    os.environ["MKL_NUM_THREADS"] = ncore
    os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
    os.environ["NUMEXPR_NUM_THREADS"] = ncore

    system_size = np.max(lattice) + 1

    # Convert H_base to csr_matrix if using KPM
    if KPM:
        H_base = csr_matrix(H_base)

    t0 = time()

    def worker(i):
        """
        Worker function to compute the Bott index for a single disorder realization.

        Parameters:
        i (int): Index of the realization.

        Returns:
        float: Bott index for the realization.
        """
        try:
            if KPM:
                # Generate disorder and compute Hamiltonian
                W_operator = uniform_mass_disorder(W, system_size, 2, True)
                H = H_base + W_operator
                P = projector_KPM(H, E_F, N)
            else:
                # Generate disorder and compute Hamiltonian
                W_operator = uniform_mass_disorder(W, system_size, 2, False)
                H = H_base + W_operator
                P = projector_exact(H, E_F)

            # Compute the Bott index using the specified method
            if exact_log_regions == 'all':
                bott = bott_index(P, lattice)
            elif exact_log_regions == 'none':
                bott = bott_index(P, lattice, order=log_order)
            else:
                if in_regions(W, exact_log_regions):
                    bott = bott_index(P, lattice)
                else:
                    bott = bott_index(P, lattice, order=log_order)

            # Print progress information if enabled
            dt = time() - t0
            if progress:
                print(f'W = {W:.2f}: {100*(i+1)/num_realizations:.2f}%, {round(dt)}s')

            return bott

        except Exception as e:
            print(f"An error occurred for W={W:.3f}: {e}")
            return np.nan

    def worker_with_timeout(i):
        """
        Wrapper function to execute the worker function with a timeout.

        Parameters:
        i (int): Index of the realization.

        Returns:
        float: Bott index for the realization or np.nan if it times out.
        """
        return task_with_timeout(worker, task_timeout, i)

    # Execute the worker functions in parallel
    if task_timeout is None:
        data = np.array(Parallel(n_jobs=num_jobs)(delayed(worker)(j) for j in range(num_realizations)))
    else:
        data = np.array(Parallel(n_jobs=num_jobs)(delayed(worker_with_timeout)(j) for j in range(num_realizations)))

    # Filter out NaN values from the results
    data = data[~np.isnan(data)]

    # Compute the mean Bott index over all realizations
    bott_mean = np.mean(data)

    return bott_mean


def main():
    # Placeholder for main function implementation
    order = 3
    M = 9.2
    B_tilde = 2
    W=0
    n = int(10)
    iters=10
    H_init, frac_lat = H_and_lattice_wrapper(order, 'symmetry', M, B_tilde, n=n)
    mean_bott = single_disorder_bott(H_init, frac_lat, W, iters, E_F=0.0, num_jobs=8)
    print("Mean Bott Index = ", mean_bott)


if __name__ == '__main__':
    main()

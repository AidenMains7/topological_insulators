import numpy as np
from scipy.spatial import ConvexHull
import os
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from ComputeBottIndex import projector_exact, projector_KPM, bott_index
from time import time
from PhaseDiagramDependencies import precompute_data, reconstruct_hamiltonian
from itertools import product


def in_regions_1D(point, regions):
    """
    Check if a point lies within any of the specified 1D regions.

    Parameters:
    point (float): The point to check.
    regions (list of tuples): List of regions specified as (min, max) tuples.

    Returns:
    bool: True if the point lies within any region, False otherwise.
    """
    # Convert regions to a NumPy array for easier manipulation
    regions = np.array(regions)
    # Check if the point is within any of the specified regions
    return np.any((regions[:, 0] <= point) & (point <= regions[:, 1]))


def in_regions_2D(point, regions, tolerance=1e-9):
    """
    Check if a point lies within any of the specified 2D regions.

    Parameters:
    point (array-like): The point to check.
    regions (list of arrays): List of regions specified as arrays of points.
    tolerance (float): Tolerance for numerical precision.

    Returns:
    bool: True if the point lies within any region, False otherwise.
    """
    for region in regions:
        region = np.array(region)

        # Check if the region is a 1D line by checking if all x or y coordinates are constant
        is_vertical_line = np.all(np.abs(region[:, 0] - region[0, 0]) < tolerance)
        is_horizontal_line = np.all(np.abs(region[:, 1] - region[0, 1]) < tolerance)

        if is_vertical_line:
            # The x-dimension is constant, region is a vertical line
            if not np.abs(point[0] - region[0, 0]) < tolerance:
                continue  # Point does not share the same x-coordinate
            # Apply the 1D check along the y-dimension
            min_y, max_y = np.min(region[:, 1]), np.max(region[:, 1])
            if in_regions_1D(point[1], [(min_y, max_y)]):
                return True
        elif is_horizontal_line:
            # The y-dimension is constant, region is a horizontal line
            if not np.abs(point[1] - region[0, 1]) < tolerance:
                continue  # Point does not share the same y-coordinate
            # Apply the 1D check along the x-dimension
            min_x, max_x = np.min(region[:, 0]), np.max(region[:, 0])
            if in_regions_1D(point[0], [(min_x, max_x)]):
                return True
        else:
            # The region is not a 1D line, apply the 2D convex hull check
            hull = ConvexHull(region)
            # Check if the point is inside the convex hull of the region
            if all(np.dot(eq[:-1], point) + eq[-1] <= 0 for eq in hull.equations):
                return True
    return False


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
            result = [np.nan]*3
    return result


def phase_diagram_data(order, method, M_values, B_tilde_values, n=None, E_F=0., num_jobs=28, cores_per_job=1, KPM=False, N=1024, exact_log_regions='all', log_order=20, progress=True, task_timeout=None, **kwargs):
    """
    Compute phase diagram data for given M and B_tilde values.

    Parameters:
    order (int): Order of the Sierpinski carpet.
    method (str): Method to use ('symmetry', 'square', 'site_elim', 'renorm').
    M_values (array-like): Array of M values to evaluate.
    B_tilde_values (array-like): Array of B_tilde values to evaluate.
    n (int, optional): Cutoff length for distances (required for 'symmetry' method).
    E_F (float): Fermi energy. Default is 0.
    num_jobs (int): Number of parallel jobs. Default is 28.
    cores_per_job (int): Number of cores per job. Default is 1.
    KPM (bool): Whether to use the Kernel Polynomial Method (KPM). Default is False.
    N (int): Number of moments in the Chebyshev expansion (for KPM). Default is 1024.
    exact_log_regions (str or list of arrays): Regions for exact logarithm computation. Default is 'all'.
    log_order (int): Order of the power series expansion for logm. Default is 20.
    progress (bool): Whether to print progress information. Default is True.
    task_timeout (float): Timeout for each task in seconds. Default is None.
    **kwargs: Additional keyword arguments.

    Returns:
    ndarray: Phase diagram data.
    """
    # Validate method parameter
    if method not in ['symmetry', 'square', 'site_elim', 'renorm']:
        raise ValueError(f"Invalid method {method}: options are ['symmetry', 'square', 'site_elim', 'renorm'].")
    # Validate 'n' parameter for 'symmetry' method
    if method == 'symmetry' and not isinstance(n, int):
        raise ValueError(f"Parameter 'n' must be specified and must be an integer.")

    # Generate all combinations of M and B_tilde values
    parameter_sets = tuple(product(M_values, B_tilde_values))
    # Precompute Hamiltonian data for the specified method and lattice order
    precomputed_data, lattice = precompute_data(order, method, True, n=n, **kwargs)

    # Set the number of threads for various libraries to avoid oversubscription
    ncore = str(int(cores_per_job))
    os.environ["OMP_NUM_THREADS"] = ncore
    os.environ["OPENBLAS_NUM_THREADS"] = ncore
    os.environ["MKL_NUM_THREADS"] = ncore
    os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
    os.environ["NUMEXPR_NUM_THREADS"] = ncore

    t0 = time()

    def worker(i):
        """
        Worker function to compute the Bott index for a single set of M and B_tilde values.

        Parameters:
        i (int): Index of the parameter set.

        Returns:
        list: [M, B_tilde, bott] for the parameter set.
        """
        M, B_tilde = parameter_sets[i]
        try:
            if KPM:
                # Reconstruct Hamiltonian using the Kernel Polynomial Method (KPM)
                H = reconstruct_hamiltonian(method, precomputed_data, M, B_tilde, sparse=True)
                P = projector_KPM(H, E_F, N)
            else:
                # Reconstruct Hamiltonian using exact diagonalization
                H = reconstruct_hamiltonian(method, precomputed_data, M, B_tilde, sparse=False)
                P = projector_exact(H, E_F)

            # Compute the Bott index using the specified method
            if exact_log_regions == 'all':
                bott = bott_index(P, lattice)
            elif exact_log_regions == 'none':
                bott = bott_index(P, lattice, order=log_order)
            else:
                if in_regions_2D([M, B_tilde], exact_log_regions):
                    bott = bott_index(P, lattice)
                else:
                    bott = bott_index(P, lattice, order=log_order)

            dt = time() - t0
            if progress:
                print(f'{100*(i+1)/len(parameter_sets):.2f}%, {round(dt)}s')

            return [M, B_tilde, bott]

        except Exception as e:
            print(f"An error occurred for [M, B_tilde] = {[M, B_tilde]}: {e}")
            return [np.nan]*3

    def worker_with_timeout(i):
        """
        Wrapper function to execute the worker function with a timeout.

        Parameters:
        i (int): Index of the parameter set.

        Returns:
        list: [M, B_tilde, bott] for the parameter set or [np.nan]*3 if it times out.
        """
        return task_with_timeout(worker, task_timeout, i)

    # Execute the worker functions in parallel
    if task_timeout is None:
        data = np.array(Parallel(n_jobs=num_jobs)(delayed(worker)(j) for j in range(len(parameter_sets)))).T
    else:
        data = np.array(Parallel(n_jobs=num_jobs)(delayed(worker_with_timeout)(j) for j in range(len(parameter_sets)))).T

    # Filter out NaN values from the results
    data = data[:, ~np.isnan(data).any(axis=0)]

    return data


def main():
    # Placeholder for main function implementation
    pass


if __name__ == '__main__':
    main()

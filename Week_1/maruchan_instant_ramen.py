'''
Most of this is Dan Salib's code

Functions:----------
create_lattice
geometry
method_of_symmetry
Hamiltonian
wrapper

mass_disorder

projector
bott_index

in_regions
task_with_timeout
bott_from_disorder
--------------------
'''

import numpy as np
from scipy.linalg import eig, eigh, logm
from scipy.sparse import dok_matrix, csr_matrix, diags
from scipy.sparse.linalg import cg

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from joblib import Parallel, delayed
from time import time
import os


#create a Sierpinski carpet and its corresponding square lattice
def create_lattice(order:int, pad_width:int=0) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Generates a Sierpinski carpet fractal lattice and its corresponding square lattice, as well as the index locations for the empty and filled sites.
    
    Parameters:
    order (int): the order of the fractal
    pad_width (int): amount of padding on the perimeter of the lattice

    Returns:
    square_lat (np.ndarray): a square lattice of the same size
    fractal_lat (np.ndarray): the fractal lattice, with holes having value -1
    hole_indices (np.ndarray): the indices of the holes (empty sites)
    filled_indices (np.ndarray): the indices of the filled sites
    '''
    
    if (order < 0):
        raise ValueError("Order of lattice must be >= 0.")

    #A sierpinski carpet fractal is infinitely self-similar, however we may compute it only up to a certain order
    def sierpinski_carpet(order_):
        if(order_ == 0):
            return np.array([1], dtype=int)
    
        #carpet of one lower degree; recursion
        carpet_lower = sierpinski_carpet(order_-1)

        #concatenate to make current degree
        top = np.hstack((carpet_lower,carpet_lower,carpet_lower))
        mid = np.hstack((carpet_lower,carpet_lower*0,carpet_lower))
        carpet = np.vstack((top,mid,top))

        return carpet
    
    #'side length' in one dimension
    L = 3**order

    #square lattice
    square_lat = np.arange(L*L).reshape((L,L))

    carpet = sierpinski_carpet(order)

    #now, we account for padding (if wanted)
    if(pad_width > 0):
        carpet = np.pad(carpet,pad_width,mode='constant',constant_values=1)
    
    
    #Determining indecies for holes and otherwise
    #flattened array
    flat = carpet.flatten()

    #locations of the filled and empty sites
    filled_indices = np.flatnonzero(flat)
    hole_indices = np.where(flat==0)[0]

    #lattice 
    fractal_lat = np.full(flat.shape, -1, dtype=int)
    fractal_lat[filled_indices] = np.arange(filled_indices.size)
    fractal_lat = fractal_lat.reshape(carpet.shape)

    return square_lat, fractal_lat, hole_indices, filled_indices


#Get the distance  and angles between all pairs of filled sites
def geometry(lattice: np.ndarray, pbc: bool, n: int) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Finds the distance and angle between sites. Polar.

    Also provides masks for the principal axes (xy plane) and the diagonal (rotated 45 deg)
    '''
    #size of system
    side_len = lattice.shape[0]

    #ensure that cutoff distance is within expect range
    if pbc and (n >= side_len//2):
        raise ValueError("Cutoff length must be half of system size while periodic boundary conditions.")

    #where the lattice does not contain a hole
    filled_idx = np.argwhere(lattice > -1)

    #this is wizardry
    diff = filled_idx[None,:,:] - filled_idx[:,None,:]
    dy, dx = diff[..., 0], diff[..., 1]

    #if periodic, set dx and dy such that if they exit the lattice, wrap
    if pbc:
        dx = np.where(np.abs(dx) > side_len / 2, dx - np.sign(dx) * side_len, dx)
        dy = np.where(np.abs(dy) > side_len / 2, dy - np.sign(dy) * side_len, dy)

    #mask as to only consider sites in which the cutoff is not exceeded.
    dist_mask = np.maximum(np.abs(dx), np.abs(dy)) <= n

    #separate masks for principal and diagonal directions
    prin_mask = ( ((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0)) ) & dist_mask #only one diff is nonzero
    diag_mask = ( (np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0)) ) & dist_mask #45 deg diag; absolute value of diff is the same for each axes and they are both nonzero

    pre_mask = prin_mask | diag_mask

    #get distance between pairs within mask
    dr = np.where(pre_mask, np.maximum(np.abs(dx), np.abs(dy)), 0)

    #calculate angles between pairs within mask
    cos_dphi = np.where(pre_mask, np.cos(np.arctan2(dy, dx)), 0.)
    sin_dphi = np.where(pre_mask, np.sin(np.arctan2(dy, dx)), 0.)

    return dr, cos_dphi, sin_dphi, prin_mask, diag_mask


#use the method of symmetry to construct wannier matrices
def method_of_symmetry(lattice:np.ndarray, pbc:bool, n:int, r0:float=1.0)->tuple:
    '''
    Parameters: 
    lattice (ndarray): the lattice in question
    pbc (bool): periodic boundary conditions
    n (int): cuttoff distance
    r0 (float): exponential decay constant (default 1.0)
    
    Returns: 
    I (ndarray): identity matrix
    Sx (ndarray): wannier matrix, sin in x-dir
    Sy (ndarray): wannier matrix, sin in y-dir
    Cx_plus_Cy (ndarray): wannier matrix, both cos
    CxSy (ndarray): wannier matrix, cos in x and sin in y
    SxCy (ndarray): wannier matrix, sin in x and cos in y
    CxCy (ndarray): wannier matrix, cos in x and cos in y
    '''

    #get the geometry of the lattice
    dr, cos_dphi, sin_dphi, prin_mask, diag_mask = geometry(lattice, pbc, n)

    #system size / number of filled sites
    num_filled_sites = np.max(lattice)+1

    #identity matrix
    I = np.eye(num_filled_sites, dtype=np.complex128)

    #we implement an exponential decay for moving based on distance
    F_p = np.where(prin_mask, np.exp(1 - dr / r0), 0. + 0.j)
    F_d = np.where(diag_mask, np.exp(1 - dr / r0), 0. + 0.j)

    #we construct the wannier matrices
    Sx = 1j * cos_dphi * F_p / 2 
    Sy = 1j * sin_dphi * F_p / 2
    Cx_plus_Cy = F_p / 2
    CxSy = 1j * sin_dphi * F_d / (2 * np.sqrt(2))
    SxCy = 1j * cos_dphi * F_d / (2 * np.sqrt(2))
    CxCy = F_d / 4

    return I, Sx, Sy, Cx_plus_Cy, CxSy, SxCy, CxCy


#construct hamiltonian from wannier matrices
def Hamiltonian(M:float, B_tilde:float, wannier_matrices:tuple, t1:float=1.0, t2:float=1.0, B:float=1.0):
    '''
    Constructs the Hamiltonian from Wannier Matrices

    Parameters: 
    M (float): On-site mass.
    B_tilde (float): Diagonal hopping amplitude between same orbitals.
    wannier_matrices (tuple): Tuple containing Wannier matrices in expected form (I, Sx, Sy, Cx_plus_Cy, CxSy, SxCy, CxCy)
    t1 (float): Principal hopping amplitude between opposite orbitals.
    t2 (float): Diagonal hopping amplitude between opposite orbitals.
    B (float): Principal hopping amplitude between same orbitals.

    Returns: 
    H (ndarray): Hamiltonian operator
    '''
    
    #unpack tuple
    I, Sx, Sy, Cx_plus_Cy, CxSy, SxCy, CxCy = wannier_matrices

    #Pauli matrices
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)

    d1 = t1 * Sx + t2 * CxSy
    d2 = t1 * Sy + t2 * SxCy
    d3 = (M - 4 * B - 4 * B_tilde) * I + 2 * B * Cx_plus_Cy + 4 * B_tilde * CxCy
    d = np.array([d1,d2,d3], dtype=np.complex128)

    # Combine components into the full Hamiltonian matrix using Kronecker products with Pauli matrices.
    H = 0
    for i in range(3):
        H += np.kron(d[i], pauli[i])

    return H
    

#wrapper function for method of symmetry
def hamiltonian_wrapper(fractal_order:int, pad_width:int, pbc:bool, n:int, r0:float, M:float, B_tilde:float, t1:float=1.0, t2:float=1.0, B:float=1.0) -> tuple: 
    '''
    
    '''

    square_lat, fractal_lat, hole_indices, filled_indices = create_lattice(fractal_order, pad_width)
    wannier_matrices = method_of_symmetry(fractal_lat, pbc, n, r0)
    H = Hamiltonian(M, B_tilde, wannier_matrices, t1, t2, B)

    return H, fractal_lat, square_lat


#generate disorder operator
def mass_disorder(strength:float, system_size:int, df:int, sparse:bool, type:str='uniform') -> np.ndarray:
    '''
    Generates a disorder operator of random values for a given strength.


    Parameters:
    type (str): type of random distribution to sample from
    strength (float): strength of disorder
    system_size (int): number of sites
    df (int): degrees of internal freedom
    sparse (bool): generate as sparse matrix?

    Returns: 
    disorder_operator (ndarray): Diagonal operator to give disorder

    '''

    if type not in ['uniform', 'normal']:
        raise ValueError("type must be one of: ['uniform', 'normal']")

    if type == 'uniform':
        disorder_array = np.random.uniform(-strength/2, strength/2, size=system_size)
        #ensure mean = 0
        delta = np.sum(disorder_array) / system_size
        disorder_array -= delta

    elif type == 'normal':
        disorder_array = np.random.normal(loc=0, scale=strength/2, size=system_size)
        #mean is already 0

    #repeat for all freedoms
    disorder_array = np.repeat(disorder_array, df)

    #diagonal matrix operator
    disorder_operator = np.diag(disorder_array).astype(np.complex128) if not sparse else diags(disorder_array, dtype=np.complex128, format='csr')
    return disorder_operator


#construct the projector onto the eigenstates
def projector(H:np.ndarray, fermi_energy:float) -> np.ndarray:
    '''
    Constructs the projector of the Hamiltonian onto the states below the Fermi energy

    Parameters: 
    H (ndarray): Hamiltonian operator
    fermi_energy (float): Fermi energy

    Returns: 
    P (ndarray): Projector operator  
    '''
    #eigenvalues and eigenvectors of the Hamiltonian
    eigvals, eigvecs = eigh(H, overwrite_a=True)

    #diagonal matrix 
    D = np.where(eigvals < fermi_energy, 1.0 + 0.0j, 0.0 + 0.0j)
    D_dagger = np.einsum('i,ij->ij', D, eigvecs.conj().T)

    #projector given by matrix multiplaction of eigenvectors and D_dagger
    P = eigvecs @ D_dagger

    return P


#
def bott_index(P:np.ndarray, lattice:np.ndarray):
    '''
    
    '''
    Y, X = np.where(lattice >= 0)[:]
    system_size = np.max(lattice) + 1
    states_per_site = P.shape[0] // system_size
    X = np.repeat(X, states_per_site)
    Y = np.repeat(Y, states_per_site)
    Ly, Lx = lattice.shape

    #
    Ux = np.exp(1j*2*np.pi*X/Lx)
    Uy = np.exp(1j*2*np.pi*Y/Ly)

    UxP = np.einsum('i,ij->ij', Ux, P)
    UyP = np.einsum('i,ij->ij', Uy, P)
    Ux_daggerP = np.einsum('i,ij->ij', Ux.conj(), P)
    Uy_daggerP = np.einsum('i,ij->ij', Uy.conj(), P)

    A = np.eye(P.shape[0], dtype=np.complex128) - P + P.dot(UxP).dot(UyP).dot(Ux_daggerP).dot(Uy_daggerP)
   
    bott = round(np.imag(np.trace(logm(A))) / (2 * np.pi))

    return bott


#
def in_regions(point:float, regions:list):
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

#
def task_with_timeout(task_func, timeout:float, *args, **kwargs):
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


#
def bott_from_disorder(H_init:np.ndarray, lattice:np.ndarray, W:float, num_realizations:int, fermi_energy:float=0.0, num_jobs:int=4, cores_per_job:int=1, progress:bool=True, task_timeout:float=None):
    '''
    I only have 4 cores
    '''

    #set threats for librar
    ncore = str(int(cores_per_job))
    os.environ["OMP_NUM_THREADS"] = ncore
    os.environ["OPENBLAS_NUM_THREADS"] = ncore
    os.environ["MKL_NUM_THREADS"] = ncore
    os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
    os.environ["NUMEXPR_NUM_THREADS"] = ncore

    system_size = np.max(lattice)+1
    t0 = time()

    def worker(i:int) -> float:
        '''
        Compute bott index for W
        '''
        try:
            W_op = mass_disorder(W, system_size, 2, False)
            H = H_init + W_op
            P = projector(H, fermi_energy)

            bott = bott_index(P, lattice)

            dt = time() - t0
            if progress:
                print(f'W = {W:.2f}: {100*(i+1)/num_realizations:.2f}%, {round(dt)}s')

            return bott
        
        except Exception as e:
            print(f"An error occurred for W={W:.3f}: {e}")
            return np.nan

    def worker_w_timeout(i):
        '''
        use worker function, apply timeout
        '''

        return task_with_timeout(worker, task_timeout, i)

    if task_timeout is None:
        data = np.array(Parallel(n_jobs=num_jobs)(delayed(worker)(j) for j in range(num_realizations)))
    else:
        data = np.array(Parallel(n_jobs=num_jobs)(delayed(worker_w_timeout)(j) for j in range(num_realizations)))

    data = data[~np.isnan(data)]
    bott_mean = np.mean(data)

    return bott_mean


#wrapper for computing bott from a disorder
def bott_disorder_wrapper(fractal_order:int, pad_width:int, pbc:bool, n:int, r0:float, M:float, B_tilde:float, t1:float, t2:float, B:float, W:float, num_realizations:int, fermi_energy:float, num_jobs:int):
    '''
    
    Parameters: 
    fractal_order (int): 
    pad_width (int): 
    '''
    H_init, fractal_lat, square_lat = hamiltonian_wrapper(fractal_order, pad_width, pbc, n, r0, M, B_tilde, t1, t2, B)
    bott_mean = bott_from_disorder(H_init, fractal_lat, W, num_realizations, fermi_energy, num_jobs, progress=True, task_timeout=None)
    return bott_mean


def main():
    order = 3
    pad_w = 0
    n = 5
    r0 = 1
    M = 1
    B_tilde = 1
    t1 = 1
    t2 = 1
    B = 1
    W = 0
    num_r = 5
    fermi_e = 2
    num_jobs = 4

    bott_mean = bott_disorder_wrapper(order, pad_w, True, n, r0, M, B_tilde, t1, t2, B, W, num_r, fermi_e, num_jobs)
    print(bott_mean)

if __name__ == "__main__":
    main()








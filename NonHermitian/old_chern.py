import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
from itertools import product
import os, h5py
from matplotlib.colors import ListedColormap, BoundaryNorm
from nonhermitian_chern import plot_phase_diagram


def compute_d_vector(kx:np.ndarray, ky:np.ndarray, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float) -> np.ndarray:
    """
    Compute the d-vector components for a non-Hermitian system.

    This function calculates the three components of the d-vector used in the 
    Hamiltonian of a topological insulator model with non-Hermitian terms.

    Parameters
    ----------
    kx : ndarray
        Wave vector component along the x-direction with shape (N, )
    ky : ndarray
        Wave vector component along the y-direction with shape (N, )
    m0 : float
        Mass term parameter
    h_vector : ndarray
        Non-Hermitian perturbation terms [h_x, h_y, h_z] with shape (3, )
    t : float
        Hopping parameter for the sine terms in d1 and d2.
    t0 : float
        Hopping parameter for the cosine terms in d3.
    a : float
        Lattice constant.

    Returns
    -------
    d_vector : ndarray
        Array of three complex components [d1, d2, d3] representing the 
        d-vector of the Hamiltonian. Has shape (3, Nx, Ny)
    """
    h_vector = np.array(h_vector)
    assert isinstance(m0, float), "m0 must be a float"
    assert h_vector.size == 3, "h_vector must have size (3, ) or (3, 1)"

    d1 = t * np.sin(kx[:, np.newaxis] * a) + 1j * h_vector[0] + 0 * ky[np.newaxis, :]
    d2 = t * np.sin(ky[np.newaxis, :] * a) + 1j * h_vector[1] + 0 * kx[:, np.newaxis]
    d3 = m0 + t0 * (np.cos(kx[:, np.newaxis] * a) + np.cos(ky[np.newaxis, :] * a)) + 1j * h_vector[2]
    return np.array([d1, d2, d3])

def compute_d_vector_scalar(kx:np.ndarray, ky:np.ndarray, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float) -> np.ndarray:

    h_vector = np.array(h_vector)
    assert isinstance(m0, float), "m0 must be a float"
    assert h_vector.size == 3, "h_vector must have size (3, ) or (3, 1)"

    d1 = t * np.sin(kx * a) + 1j * h_vector[0]
    d2 = t * np.sin(ky * a) + 1j * h_vector[1]
    d3 = m0 + t0 * (np.cos(kx * a) + np.cos(ky * a)) + 1j * h_vector[2]
    return np.array([d1, d2, d3])


def compute_hamiltonian(d_vector:np.ndarray) -> np.ndarray:
    """
    Compute the Hamiltonians for a non-Hermitian system.

    Parameters
    ----------
    d_vector : ndarray
        Hopping vector with shape (3, Nx, Ny)
    Returns
    -------
    hamiltonians : ndarray
        Array of momentum-space Hamiltonians with shape (2, 2, Nx, Ny)
    """
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    tau = np.swapaxes(np.array((pauli_x, pauli_y, pauli_z)), 0, -1) # Shape (2, 2, 3)

    hamiltonians =  np.einsum('ijk,klm->ijlm', tau, d_vector) # Shape (2, 2, Nx, Ny)
    return hamiltonians


def compute_eigenvectors(hamiltonians:np.ndarray, band_index:int=0):
    """
    Compute the eigenvalues, left and right eigenvectors for an array of momentum-space Hamiltonians
    Parameters
    ----------

    Returns
    -------
    eigenvalues : ndarray
        Array of shape (2, Nx, Ny) containing the eigenvalues of each Hamiltonian
    left_eigenvectors : ndarray
        Array of shape (2, 2, Nx, Ny) containing the left_eigenvectors of each Hamiltonian
    right_eigenvectors : ndarray
        Array of shape (2, 2, Nx, Ny) containing the right_eigenvectors of each Hamiltonian
    """
    Nx, Ny = hamiltonians.shape[-2], hamiltonians.shape[-1]
    idxs = np.indices((Nx, Ny))
    idx_i, idx_j = idxs[0].flatten(), idxs[1].flatten()

    eigenvalues = np.full((2, Nx, Ny), np.nan, dtype=complex)
    left_eigenvectors = np.full(hamiltonians.shape, np.nan, dtype=complex)
    right_eigenvectors = np.full(hamiltonians.shape, np.nan, dtype=complex)
    
    for i, j in zip(idx_i, idx_j):
        if False:
            eigs, eigvecs = spla.eigh(hamiltonians[:, :, i, j])
            eigenvalues[:, i, j], left_eigenvectors[:, :, i, j], right_eigenvectors[:, :, i, j] = eigs, eigvecs, eigvecs
        else:
            eigenvalues[:, i, j], left_eigenvectors[:, :, i, j], right_eigenvectors[:, :, i, j] = spla.eig(hamiltonians[:, :, i, j], left=True, right=True)


    return {"eigenvalues": eigenvalues[band_index, :, :],
            "left_eigenvectors": left_eigenvectors[:, band_index, :, :], 
            "right_eigenvectors": right_eigenvectors[:, band_index, :, :]}


def compute_eigenvectors_from_momentum(kx, ky, m0, h_vector, t, t0, a, band_index):
    d_vector = compute_d_vector(kx, ky, m0, h_vector, t, t0, a)
    hamiltonians = compute_hamiltonian(d_vector)
    return compute_eigenvectors(hamiltonians, band_index)


def compute_u1_link_variable(kx:np.ndarray, ky:np.ndarray, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float, direction:str, band_index:int = 0):
    """
    Parameters
    ----------


    Returns
    -------
    u_lower : ndarray
        U(1) link variable for the lower eigenvector with shape (Nx, Ny)
    u_upper : ndarray
        U(1) link variable for the upper eigenvector with shape (Nx, Ny)
    """
    dkx = 2 * np.pi / len(kx)
    dky = 2 * np.pi / len(ky)

    _, left_eigenvectors, right_eigenvectors = compute_eigenvectors_from_momentum(kx, ky, m0, h_vector, t, t0, a, band_index).values()

    if direction == 'none':
        _ = left_eigenvectors
        shifted_right_eigenvectors = right_eigenvectors
    elif direction == 'x':
        _, _, shifted_right_eigenvectors = compute_eigenvectors_from_momentum(kx + dkx, ky, m0, h_vector, t, t0, a, band_index).values()
    elif direction == 'y':
        _, _, shifted_right_eigenvectors = compute_eigenvectors_from_momentum(kx, ky + dky, m0, h_vector, t, t0, a, band_index).values()

    product = np.einsum('ijk,ijk->jk', left_eigenvectors.conj(), shifted_right_eigenvectors)
    u = product / np.abs(product)
    return u

def compute_lattice_field_strength(kx:np.ndarray, ky:np.ndarray, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float, band_index:int=0):
    """
    
    """
    dkx = 2 * np.pi / kx.size
    dky = 2 * np.pi / ky.size

    term1 = compute_u1_link_variable(kx, ky, m0, h_vector, t, t0, a, 'x', band_index)
    term2 = compute_u1_link_variable(kx + dkx, ky, m0, h_vector, t, t0, a, 'y', band_index)
    term3 = compute_u1_link_variable(kx, ky + dky, m0, h_vector, t, t0, a, 'x', band_index).conj()
    term4 = compute_u1_link_variable(kx, ky, m0, h_vector, t, t0, a, 'y', band_index).conj()

    field_strength = np.angle(term1 * term2 * term3 * term4)
    chern = np.sum(field_strength) / (2 * np.pi)
    return chern.real


def compute_chern_number(m0, h_vector, t:float=1.0, t0:float=1.0, a:float=1.0, Nx:int=25, Ny:int=25):
    kx = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, Ny, endpoint=False)
    chern = compute_lattice_field_strength(kx, ky, m0, h_vector, t, t0, a, band_index=1)
    return chern


def compute_chern_phase_diagram(output_file:str, resolution = (51, 51)):
    m0_values = np.linspace(-2.0, 2.0, resolution[0])
    h_values = np.linspace(-1.0, 1.0, resolution[1])

    parameters = tuple(product(m0_values, h_values))

    def worker(i):
        m0, h = parameters[i]
        chern = compute_FHS_chern_fast(m0, [0., 0., h])
        return [m0, h, chern]
    
    with tqdm_joblib(tqdm(total=len(parameters), desc=f"Computing phase diagram for Chern number.")) as progress_bar:
        m0_data, h_data, chern_data = np.array(Parallel(n_jobs=-1)(delayed(worker)(i) for i in range(len(parameters))), dtype=float).T

    with h5py.File(output_file, "w") as f:
        f.create_dataset(name = "m0", data=m0_data)
        f.create_dataset(name = "h", data=h_data)
        f.create_dataset(name =  "chern", data=chern_data.reshape(resolution).T)
    return output_file


if __name__ == "__main__":
    f = compute_chern_phase_diagram('temp.h5', (15,15))

    with h5py.File(f, 'r') as f:
        m0_data = f['m0'][:]
        h_data = f['h'][:]
        chern_data = f['chern'][:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_phase_diagram(fig, ax, m0_data, h_data, chern_data)
    plt.show()
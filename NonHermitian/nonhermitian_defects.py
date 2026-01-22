import numpy as np
import scipy.linalg as spla
from scipy.sparse import dok_matrix

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
import os, h5py
from itertools import product


# region Lattice Generation
def generate_square_lattice(Lx:int, Ly:int):
    return np.arange(Lx*Ly).reshape((Ly, Lx)), []


def generate_vacancy_lattice(Lx:int, Ly:int, vacancy_radius:int=1):
    assert (Lx*Ly % 2 == 1), "Side lengths must be odd"
    assert vacancy_radius > 0, "Defect radius must be positive definite"
    assert vacancy_radius <= (min(Lx, Ly) // 2 + 1), "Defect hole must fit inside the lattice."

    lattice, _ = generate_square_lattice(Lx, Ly)

    defect_indices = []
    for i in range(-vacancy_radius, vacancy_radius):
        for j in range(-vacancy_radius, vacancy_radius):
            if abs(i) + abs(j) < vacancy_radius:
                vacancy_index = lattice[Ly // 2 + i, Lx // 2 + j]
                lattice[Ly // 2 + i, Lx // 2 + j] = -1                
                defect_indices.append(vacancy_index)

    return lattice, defect_indices


def generate_schottky_lattice(Lx:int, Ly:int, separation:int, n_pairs:int = 1):
    assert ((Lx + separation) % 2 == 1) and ((Ly + separation) % 2 == 1), "Separation must be odd for even side lengths, and even for odd side lengths"
    assert n_pairs in [1, 2], "Number of pairs must be either 1 or 2"
    assert (separation <= Lx) and (separation <= Ly), "Separation must be less than the side lengths"
    assert separation > 0, "Separation must be positive definite"
    if separation is None:
        separation = Lx % 2 + 1
    
    lattice, _ = generate_square_lattice(Lx, Ly)
    shift = (separation - 1) // 2
    
    up_parity_index1 = lattice[Ly // 2 + shift, Lx // 2 + shift] = -1
    down_parity_index1 = lattice[Ly // 2 - 1 - shift, Lx // 2 - 1 - shift] = -1
    defect_indices = [up_parity_index1, down_parity_index1]

    if n_pairs == 2:
        up_parity_index2 = lattice[Ly // 2 + shift, Lx // 2 - 1 - shift]
        down_parity_index2 = lattice[Ly // 2 - 1 - shift, Lx // 2 + shift]

        defect_indices.append(up_parity_index2)
        defect_indices.append(down_parity_index2)

    return lattice, defect_indices


def generate_substitution_lattice(Lx:int, Ly:int, substitution_radius:int=1):
    return generate_vacancy_lattice(Lx, Ly, substitution_radius)


def generate_interstitial_lattice(Lx:int, Ly:int, interstitial_radius:int=1):
    lattice, _ = generate_square_lattice(Lx, Ly)

    Y, X = np.where(lattice >= 0)
    X = X * 2
    Y = Y * 2

    large_lattice = np.arange(4 * Lx * Ly).reshape((Ly * 2, Lx * 2))
    square_indices = large_lattice[Y, X]

    defect_indices = []
    X = list(X)
    Y = list(Y)
    for i in range(-interstitial_radius, interstitial_radius):
        for j in range(-interstitial_radius, interstitial_radius):
            if abs(i) + abs(j) < interstitial_radius:
                X.append(Lx - 1 + 2 * j)
                Y.append(Ly - 1 + 2 * i)
                interstitial_index = large_lattice[Ly - 1 + 2 * i, Lx - 1 + 2 * j]
                defect_indices.append(interstitial_index)

    occupied_indices = square_indices + defect_indices
    #return large_lattice, defect_indices


def generate_frenkel_pair_lattice():
    pass


# endregion
# region Lattice Class
class DefectLattice:
    def __init__(self, Lx:int, Ly:int, method:str, defect_radius:int=1,
                 schottky_separation:int=None, schottky_n_pairs:int=1):
        match method:
            case 'square':
                self.lattice, self.defect_indices = generate_square_lattice(Lx, Ly)
            case 'vacancy':
                self.lattice, self.defect_indices = generate_vacancy_lattice(Lx, Ly, defect_radius)
            case 'schottky':
                self.lattice, self.defect_indices = generate_schottky_lattice(Lx, Ly, schottky_separation, schottky_n_pairs)
            case 'substitution':
                self.lattice, self.defect_indices = generate_substitution_lattice(Lx, Ly, defect_radius)
            case 'interstitial':
                self.lattice, self.defect_indices = generate_interstitial_lattice(Lx, Ly, defect_radius)
            case 'frenkel_pair':
                self.lattice, self.defect_indices = generate_frenkel_pair_lattice(Lx, Ly, )
# endregion
# region Geometry and Hamiltonian

def generate_wannier_matrices(lattice:np.ndarray, pbc:bool):
    Y, X = np.where(lattice >= 0)[:]
    side_length = lattice.shape[0]
    dx = X - X[:, None]
    dy = Y - Y[:, None]
    if pbc:
        multipliers = tuple(product([-1, 0, 1], repeat=2))
        shifts = [(i * side_length, j * side_length) for i, j in multipliers]

        x_shifted = np.empty((dx.shape[0], dx.shape[1], len(shifts)), dtype=dx.dtype)
        y_shifted = np.empty((dy.shape[0], dy.shape[1], len(shifts)), dtype=dy.dtype)
        for i, (dx_shift, dy_shift) in enumerate(shifts):
            x_shifted[:, :, i] = dx + dx_shift
            y_shifted[:, :, i] = dy + dy_shift

        distances = x_shifted**2 + y_shifted**2
        minimal_hop = np.argmin(distances, axis = -1)
        i_idxs, j_idxs = np.indices(minimal_hop.shape)

        dx = x_shifted[i_idxs, j_idxs, minimal_hop]
        dy = y_shifted[i_idxs, j_idxs, minimal_hop]

    xp_mask = (dx == 1) & (dy == 0)
    yp_mask = (dx == 0) & (dy == 1)


    Cx =   dok_matrix(dx.shape, dtype=complex)
    Sx =   dok_matrix(dx.shape, dtype=complex)
    Cy =   dok_matrix(dx.shape, dtype=complex)
    Sy =   dok_matrix(dx.shape, dtype=complex)
    I =    np.eye(dx.shape[0], dtype=complex)

    Sx[xp_mask] = 1j / 2
    Cx[xp_mask] = 1 / 2
    Cy[yp_mask] = 1 / 2
    Sy[yp_mask] = 1j / 2

    Sx += Sx.conj().T
    Sy += Sy.conj().T
    Cx += Cx.conj().T
    Cy += Cy.conj().T
    return I, Sx.toarray(), Sy.toarray(), Cx.toarray() + Cy.toarray()


def compute_hamiltonian(m0:float, h_vector:np.ndarray, wannier_matrices:tuple[np.ndarray], t:float, t0:float):
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    I, Sx, Sy, Cx_plus_Cy = wannier_matrices
    hx, hy, hz = h_vector

    dx = t * Sx + 1.0j * hx * I
    dy = t * Sy + 1.0j * hy * I
    dz = (m0 + 1.0j * hz) * I + t0 * Cx_plus_Cy

    hamiltonian = np.kron(dx, pauli_x) + np.kron(dy, pauli_y) + np.kron(dz, pauli_z)
    return hamiltonian

# endregion
# region Bott Index Stuff
def compute_projector(hamiltonian:np.ndarray):
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, overwrite_a=True, left=True, right=True)
    sort_idxs = np.lexsort((eigenvalues.imag, eigenvalues.real))
    eigenvalues = eigenvalues[sort_idxs]

    left_eigenvectors = left_eigenvectors[:, sort_idxs]
    right_eigenvectors = right_eigenvectors[:, sort_idxs]

    occupied_left_vectors = left_eigenvectors[:, :eigenvalues.size // 2]
    occupied_right_vectors = right_eigenvectors[:, :eigenvalues.size // 2]

    overlap = occupied_left_vectors.conj().T @ occupied_right_vectors
    inv_overlap = spla.inv(overlap)
    projector = occupied_right_vectors @ inv_overlap @ occupied_left_vectors.conj().T

    return projector

def compute_bott_index(projector:np.ndarray, lattice:np.ndarray):
    Y, X = np.where(lattice >= 0)[:]
    X, Y = X.flatten(), Y.flatten()

    # Repeated (two orbitals)
    X = np.repeat(X, 2)
    Y = np.repeat(Y, 2)

    Lx = np.max(X) - np.min(X) # length of the x-direction
    Ly = np.max(Y) - np.min(Y)

    x_unitary = np.exp(1j * 2 * np.pi * X / Lx) # unitary operator in the x-direction
    y_unitary = np.exp(1j * 2 * np.pi * Y / Ly)
    x_unitary_proj = np.einsum('i,ij->ij', x_unitary, projector) # projector in the x-direction
    y_unitary_proj = np.einsum('i,ij->ij', y_unitary, projector)
    x_unitary_dagger_proj = np.einsum('i,ij->ij', x_unitary.conj(), projector)  # projector in the x-direction (dagger)
    y_unitary_dagger_proj = np.einsum('i,ij->ij', y_unitary.conj(), projector)

    I = np.eye(projector.shape[0], dtype=np.complex128) 
    A = I - projector + projector @ x_unitary_proj @ y_unitary_proj @ x_unitary_dagger_proj @ y_unitary_dagger_proj # BI operator given in arxiv:2407.13767 [Eq. (5)]
    bott_index = np.imag(np.sum(np.log(spla.eigvals(A)))) / (2 * np.pi)
    return bott_index


def compute_bott_index_wrapper(wannier_matrices:tuple, lattice:np.ndarray, m0:float, h_vector:np.ndarray, t:float, t0:float):
    hamiltonian = compute_hamiltonian(m0, h_vector, wannier_matrices, t, t0)
    projector = compute_projector(hamiltonian)
    bott_index = compute_bott_index(projector, lattice)
    return np.round(bott_index, 3)

# endregion


if __name__ == "__main__":
    lattice, defect_indices = generate_vacancy_lattice(15, 15, 1)
    wannier_matrices = generate_wannier_matrices(lattice, True)
    hamiltonian = compute_hamiltonian(1.0, [0.25, 0., 0.], wannier_matrices, 1.0, 1.0)

    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True)
    sort_idxs = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sort_idxs]


    Y, X = np.where(lattice >= 0)
    #plt.scatter(X, Y)
    #plt.show()


    def get_normed_eigenvectors_from_idxs(idxs):
        arrs = []
        lefts = []
        rights = []
        ratios = []
        for idx in idxs:
            left_arr = np.abs((left_eigenvectors[:, idx][::2]) + (left_eigenvectors[:, idx][1::2]))**2
            right_arr = np.abs((right_eigenvectors[:, idx][::2]) + (right_eigenvectors[:, idx][1::2]))**2
            lefts.append(left_arr)
            rights.append(right_arr)
            ratios.append(left_arr / right_arr)

        left = np.sum(lefts, axis=0)
        right = np.sum(rights, axis=0)
        ratio = np.sum(ratios, axis=0)

        return left, right, ratio
    



    #plt.scatter(X, Y, c=c)
    #plt.show()
    #plt.scatter(np.arange(len(eigenvalues)), eigenvalues.real)
    #plt.scatter(np.arange(len(eigenvalues))[in_gap_idx], eigenvalues.real[in_gap_idx], c='r')
    #plt.scatter(np.arange(len(eigenvalues))[in_gap_idx_2], eigenvalues.real[in_gap_idx_2], c='r')
    in_gap_idx = len(eigenvalues) // 2
    n_center_states = 2
    center_states = [len(eigenvalues) // 2 + i for i in np.arange(-n_center_states//2, n_center_states//2)]
    left, right, ratio = get_normed_eigenvectors_from_idxs(center_states)
    __ = np.abs(right_eigenvectors[:, 2][0::2] + right_eigenvectors[:, 2][1::2])**2

    plt.scatter(X, Y, c=ratio)
    plt.colorbar()
    plt.show()

    plt.scatter(np.arange(len(eigenvalues)), eigenvalues.real)
    plt.scatter(np.arange(len(eigenvalues))[center_states], eigenvalues.real[center_states])
    plt.show()



    #Y, X = np.where(lattice >= 0)
    #plt.scatter(X, Y)
    #for idx in defect_indices:
    #    pos = np.where(lattice == idx)
    #    plt.scatter(pos[1], pos[0], c='r', zorder=2)
    #    
    #plt.show()
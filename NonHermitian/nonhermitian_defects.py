"""
Geometrically trivial defects in non-Hermitian Chern insulators
"""

import numpy as np
import scipy.linalg as spla
from scipy.sparse import dok_matrix

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
import os, h5py
from itertools import product

from cProfile import Profile
import pstats
import functools
from time import time

# =============================================================================
# =============================================================================
def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = Profile()
        pr.enable()
        try: 
            return func(*args, **kwargs)
        finally:
            pr.disable()
            stats = pstats.Stats(pr)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
    return wrapper
# =============================================================================
# =============================================================================
# region Lattice Generation
def generate_square_lattice(Lx:int, Ly:int):
    return np.arange(Lx*Ly).reshape((Ly, Lx)), None


def generate_vacancy_lattice(Lx:int, Ly:int, vacancy_radius:int=1):
    #assert (Lx*Ly % 2 == 1), "Side lengths must be odd"
    assert vacancy_radius > 0, "Defect radius must be positive definite"
    assert vacancy_radius <= (min(Lx, Ly) // 2 + 1), "Defect must fit inside the lattice."

    lattice, _ = generate_square_lattice(Lx, Ly)
    defect_indices = []
    vacancy_index = -1
    for i in range(-vacancy_radius, vacancy_radius):
        for j in range(-vacancy_radius, vacancy_radius):
            if abs(i) + abs(j) < vacancy_radius:
                lattice[Ly // 2 + i, Lx // 2 + j] = vacancy_index
                defect_indices.append(vacancy_index)
                vacancy_index -= 1
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
    
    up_parity_index1 = lattice[Ly // 2 + shift, Lx // 2 + shift]
    down_parity_index1 = lattice[Ly // 2 - 1 - shift, Lx // 2 - 1 - shift]
    defect_indices = [up_parity_index1, down_parity_index1]

    if n_pairs == 2:
        up_parity_index2 = lattice[Ly // 2 + shift, Lx // 2 - 1 - shift]
        down_parity_index2 = lattice[Ly // 2 - 1 - shift, Lx // 2 + shift]

        defect_indices.append(up_parity_index2)
        defect_indices.append(down_parity_index2)

    return lattice, defect_indices


def generate_substitution_lattice(Lx:int, Ly:int, substitution_radius:int=1):
    #assert (Lx * Ly % 2 == 1), "Side lengths must be odd"
    assert substitution_radius > 0, "Defect radius must be positive definite"
    assert substitution_radius <= (min(Lx, Ly) // 2 + 1), "Defect must fit inside the lattice."

    lattice, _ = generate_square_lattice(Lx, Ly)
    defect_indices = []
    for i in range(-substitution_radius, substitution_radius):
        for j in range(-substitution_radius, substitution_radius):
            if abs(i) + abs(j) < substitution_radius:
                defect_indices.append(lattice[Ly // 2 + i, Lx // 2 + j])
    return lattice, defect_indices


def generate_interstitial_lattice(Lx:int, Ly:int, interstitial_radius:int=1):
    assert (Lx % 2 == 0) and (Ly % 2 == 0), "Side lengths must be even"
    assert interstitial_radius > 0, "Defect radius must be positive definite"
    assert interstitial_radius <= (min(Lx, Ly) // 2 + 1), "Defect must fit inside the lattice."
    
    lattice, _ = generate_square_lattice(Lx, Ly)
    Y, X = np.where(lattice >= 0)

    displacement = 0.5
    scale = 2

    X *= scale
    Y *= scale

    large_lattice = np.full((scale * Ly - scale + 1, scale * Lx - scale + 1), np.nan)
    large_lattice[Y, X] = np.arange(len(X))


    Y_pos = []
    X_pos = []
    for i in range(-interstitial_radius, interstitial_radius):
        for j in range(-interstitial_radius, interstitial_radius):
            if abs(i) + abs(j) < interstitial_radius:
                y = int(Ly * scale / 2 - (displacement * scale) + scale * i)
                x = int(Lx * scale / 2 - (displacement * scale) + scale * j)
                Y_pos.append(y)
                X_pos.append(x)
                large_lattice[y, x] = np.inf

    large_lattice[np.where(large_lattice >= 0)] = np.arange(len(np.where(large_lattice >= 0)[0].flatten()))
    defect_indices = list(large_lattice[Y_pos, X_pos].astype(int))
    return large_lattice, defect_indices


def generate_frenkel_pair_lattice(Lx:int, Ly:int, x_disp:float, y_disp:float):
    #assert (Lx * Ly % 2 == 1), "Side lengths must be odd"
    assert ((x_disp % 1 == 0.5) and (y_disp % 1 == 0.5)), "Displacements must be odd half integer"
    assert (abs(x_disp) < Lx / 2) and (abs(y_disp) < Ly / 2), "Interstitial displacement must be within the lattice"

    # Convert x_disp, y_disp to the doubled lattice linear length
    x_disp = int(2 * x_disp)
    y_disp = int(2 * y_disp)


    lattice, _ = generate_square_lattice(Lx, Ly)
    Y, X = np.where(lattice >= 0)
    X = X * 2
    Y = Y * 2

    large_lattice = np.full((2 * Ly - 1, 2 * Lx - 1), np.nan)
    large_lattice[Y, X] = np.arange(len(X))

    vac_y, vac_x = Ly - Ly % 2, Lx - Lx % 2
    large_lattice[vac_y, vac_x] = -1 # Vacancy at center
    large_lattice[vac_y + y_disp, vac_x + x_disp] = np.inf # Interstitial site

    large_lattice[np.where(large_lattice >= 0)] = np.arange(len(np.where(large_lattice >= 0)[0].flatten()))
    defect_indices = [-1, int(large_lattice[vac_y + y_disp, vac_x + x_disp])]
    return large_lattice, defect_indices

# endregion
# =============================================================================
# =============================================================================
class DefectLattice:
    def __init__(self, Lx:int, Ly:int, defect_type:str, pbc:bool, defect_radius:int=1,
                 schottky_separation:int=None, schottky_n_pairs:int=1, frenkel_x_disp:int = -1.5, frenkel_y_disp:int = -0.5):
        assert Lx % 2 == 0
        assert Ly % 2 == 0
        self._defect_type = defect_type
        self._pbc = pbc
        self._Lx = Lx
        self._Ly = Ly
        match defect_type:
            case 'none':
                self._lattice, self._defect_indices = generate_square_lattice(Lx, Ly)
            case 'vacancy':
                self._lattice, self._defect_indices = generate_vacancy_lattice(Lx, Ly, defect_radius)
            case 'schottky':
                self._lattice, self._defect_indices = generate_schottky_lattice(Lx, Ly, schottky_separation, schottky_n_pairs)
            case 'substitution':
                self._lattice, self._defect_indices = generate_substitution_lattice(Lx, Ly, defect_radius)
            case 'interstitial':
                self._lattice, self._defect_indices = generate_interstitial_lattice(Lx, Ly, defect_radius)
            case 'frenkel_pair':
                self._lattice, self._defect_indices = generate_frenkel_pair_lattice(Lx, Ly, frenkel_x_disp, frenkel_y_disp)
                self._fp_xdisp = frenkel_x_disp
                self._fp_ydisp = frenkel_y_disp
            case _:
                raise ValueError('Defect type not properly provided')

        Y, X = np.where(self.lattice >= 0)[:]
        if defect_type in ['interstitial', 'frenkel_pair']:
            self._X, self._Y = X / 2, Y / 2
        else:
            self._X, self._Y = X, Y 

        self._dx, self._dy = self.compute_distances()
        if defect_type in ["interstitial", "frenkel_pair"]:
            self._wannier_matrices = self.compute_wannier_matrices_polar()
        else:
            self._wannier_matrices = self.compute_wannier_matrices_fourier()

    # Properties
    @property
    def lattice(self): return self._lattice
    @property
    def defect_indices(self): return self._defect_indices
    @property
    def X(self): return self._X
    @property
    def Y(self): return self._Y
    @property
    def defect_type(self): return self._defect_type
    @property
    def pbc(self): return self._pbc
    @property
    def wannier_matrices(self): return self._wannier_matrices
    @property
    def dx(self): return self._dx
    @property
    def dy(self): return self._dy
    @property
    def Lx(self): return self._Lx
    @property
    def Ly(self): return self._Ly

    def compute_distances(self):
        X = self.X
        Y = self.Y

        dx = X - X[:, None]
        dy = Y - Y[:, None]
        if self.pbc:
            multipliers = tuple(product([-1, 0, 1], repeat=2))
            shifts = [(i * self.Lx, j * self.Ly) for i, j in multipliers]

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
        return dx, dy

    def compute_wannier_matrices_fourier(self):
        dx = self.dx
        dy = self.dy

        xp_mask = np.isclose(dx, 1.0) & np.isclose(dy, 0.0)
        yp_mask = np.isclose(dx, 0.0) & np.isclose(dy, 1.0)

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
    
    def compute_wannier_matrices_polar(self):
        dx = self.dx
        dy = self.dy
        theta = np.arctan2(dy, dx)
        dr = np.sqrt(dx ** 2 + dy ** 2)

        # Create masks for different types of hopping. 
        distance_mask = ((dr <= 1.1 + 1e-6) & (dr > 1e-6)) # Mask for distances close to 1
        principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & distance_mask 
        diagonal_mask  = ((np.isclose(np.abs(dx), np.abs(dy), atol=1e-4)) & (dx != 0)) & distance_mask
        hopping_mask = principal_mask | diagonal_mask
    
        # Compute the Wannier matrices based on the masks
        d_cos = np.where(hopping_mask, np.cos(theta), 0. + 0.j)
        d_sin = np.where(hopping_mask, np.sin(theta), 0. + 0.j)
        amplitude = np.where(hopping_mask, np.exp(1. - dr), 0. + 0.j)

        # Momentum space matrices constructed from the real-space displacements based on arxiv.org/abs/2407.13767
        Cx_plus_Cy = amplitude / 2
        Sx = 1j * d_cos * amplitude / 2
        Sy = 1j * d_sin * amplitude / 2
        return np.eye(Sx.shape[0], dtype=complex), Sx, Sy, Cx_plus_Cy

    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        plt.scatter(self.X, self.Y)
        plt.show()
# =============================================================================
# =============================================================================
# region Hamiltonian
def compute_hamiltonian(Lattice:DefectLattice, m0:float, h_vector:np.ndarray, t:float, t0:float, msub:float = None):
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    I, Sx, Sy, Cx_plus_Cy = Lattice.wannier_matrices
    hx, hy, hz = h_vector

    defect_indices = Lattice.defect_indices

    onsite_mass = m0 * I
    if defect_indices is not None:
        if msub == None and Lattice.defect_type in ['substitution', 'interstitial', 'frenkel_pair']:
            raise ValueError(f"`msub` cannot be None when defect_indices are provided for 'substitution', 'interstitial', 'frenkel_pair'")
        for idx in defect_indices:
            if (idx >= 0) and (Lattice.defect_type != "schottky"):
                onsite_mass[idx, idx] = msub

    dx = t * Sx + 1.0j * hx * I
    dy = t * Sy + 1.0j * hy * I
    dz = ((1.0j * hz) * I + onsite_mass) + t0 * Cx_plus_Cy

    hamiltonian = np.kron(dx, pauli_x) + np.kron(dy, pauli_y) + np.kron(dz, pauli_z)

    if Lattice.defect_type == "schottky":
        mask = np.full(hamiltonian.shape[0], True)
        for i, idx in enumerate(defect_indices):
            mask[2 * idx + i % 2] = False
        hamiltonian = hamiltonian[np.ix_(mask, mask)]
    return hamiltonian


def get_separated_points(z, k=6, threshold=0.75):
    # Using ChatGPT 5
    z = np.asarray(z)
    N = len(z)

    # Build pairwise distance matrix
    dz = z.reshape(N, 1) - z.reshape(1, N)
    dist = np.abs(dz)

    # Ignore self-distance
    np.fill_diagonal(dist, np.inf)

    # k-th nearest neighbor distance
    knn = np.partition(dist, k-1, axis=1)[:, k-1]
    knn = (knn - np.min(knn)) / np.max(knn)

    # Add weight to smaller real and imag gap
    z_real, z_imag = np.abs(z.real), np.abs(z.imag)
    real_weight = 1.0 - (z_real - np.min(z_real)) / np.max(z_real)
    imag_weight = 1.0 - (z_imag - np.min(z_imag)) / np.max(z_imag)

    if False:
        sum_weight = knn + real_weight + imag_weight
        weights = [knn, real_weight, imag_weight, sum_weight]
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))   
        x = np.arange(len(z))
        for i, weight in enumerate(weights):
            axs[0, i].scatter(x, weight, c=knn)
            axs[1, i].scatter(z.real, z.imag, c=weight)
        plt.show()

    knn_idxs = np.argsort(knn)[-2:]
    real_idxs = np.argsort(real_weight)[-2:]
    imag_idxs = np.argsort(imag_weight)[-2:]

    idxs = np.concatenate((knn_idxs, real_idxs, imag_idxs))
    return knn_idxs



def find_defect_points(Lattice, m0, h_vector, msub):
    PristineLat = DefectLattice(Lattice.Lx, Lattice.Ly, 'none', Lattice.pbc)
    pristine_hamiltonian = compute_hamiltonian(PristineLat, m0, h_vector, 1.0, 1.0)
    defect_hamiltonian = compute_hamiltonian(Lattice, m0, h_vector, 1.0, 1.0, msub)

    pristine_eigenvalues = spla.eig(pristine_hamiltonian, left=False, right=False, overwrite_a=True)
    sort_idxs = np.argsort(pristine_eigenvalues.real)
    pristine_eigenvalues = pristine_eigenvalues[sort_idxs]

    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(defect_hamiltonian, left=True, right=True, overwrite_a=True)
    sort_idxs = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sort_idxs]
    left_eigenvectors = left_eigenvectors[:, sort_idxs]
    right_eigenvectors = right_eigenvectors[:, sort_idxs]


    tol = 1e-1

    result = np.array([
        z for z in eigenvalues
        if not np.any(np.abs(pristine_eigenvalues - z) < tol)
    ])
    plt.scatter(result.real, result.imag)
    plt.show()



def compute_eigenvectors_eigenvalues(Lattice:DefectLattice, m0:float, 
                                     h_vector:np.ndarray, msub:float = None, 
                                     n_closest_to_zero:int = 2):
    """
    Returns:
    data_dictionary (dict) : \\
    keys: ['eigenvalues', 
    'L', 'R', 
    'close_to_zero_idxs', 
    'close_to_zero_left_eigenvectors', 
    'close_to_zero_right_eigenvectors']
    """
    if n_closest_to_zero is not None:
        assert (n_closest_to_zero <= len(Lattice.X) * 2), "Number of selected indices must be <= number of indices"
    
    hamiltonian = compute_hamiltonian(Lattice, m0, h_vector, 1.0, 1.0, msub)
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True)
    sort_idxs = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sort_idxs]
    left_eigenvectors = left_eigenvectors[:, sort_idxs]
    right_eigenvectors = right_eigenvectors[:, sort_idxs]

    hcond = np.linalg.cond(hamiltonian)
    if hcond > 1e10:
        print('Hamiltonian condition number:', hcond)

    # Assuming particle-hole symmetry
    close_to_zero_idxs_negreal = np.argsort(np.abs(eigenvalues[:len(eigenvalues) // 2]))[:n_closest_to_zero // 2]
    close_to_zero_idxs_posreal = np.argsort(np.abs(eigenvalues[len(eigenvalues) // 2:]))[:n_closest_to_zero // 2] + len(eigenvalues) // 2
    close_to_zero_idxs = np.unique(np.concatenate((close_to_zero_idxs_negreal, close_to_zero_idxs_posreal)))

    # Hard-coded selection for certain parameters
    if 0:
        if h_vector[0] == 0.25 or h_vector[1] == 0.25 or h_vector[2] == 0.25:
            if all([h_vector[2], not any(h != 0 for h in h_vector[:2]), Lattice.defect_type == 'interstitial']):
                close_to_zero_idxs = get_separated_points(eigenvalues)
                if (m0, msub) in [(1.0, 2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:2]
                if (m0, msub) in [(2.5, -2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[:2]
            elif all([any(h != 0 for h in h_vector[:2]), Lattice.defect_type == "substitution"]):
                if (m0, msub) in [(2.5, -1.0)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:4]
            elif all([h_vector[2], not any(h != 0 for h in h_vector[:2]), Lattice.defect_type == 'frenkel_pair']):
                if (m0, msub) in [(1.0, -1.0), (1.0, 2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:4]
                elif (m0, msub) in [(2.5, -1.0)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[:2]
                elif (m0, msub) in [(2.5, -2.5)]:
                    arr = 10 * np.abs(eigenvalues.imag) - np.abs(eigenvalues.real)
                    close_to_zero_idxs = np.argsort(arr)[:2]
            elif all([any(h != 0 for h in h_vector[:2]), Lattice.defect_type == 'frenkel_pair']):
                if (m0, msub) in [(1.0, -1.0), (1.0, 2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:4]
        elif np.abs(h_vector[2]) >= 1 :
            if Lattice.defect_type in ['substitution', 'interstitial', 'frenkel_pair']:
                close_to_zero_idxs = get_separated_points(eigenvalues)
            else:
                if np.abs(m0) >= 2.0:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:2]
        elif np.abs(h_vector[0]) >= 1 and Lattice.defect_type != 'none':
            if Lattice.defect_type == 'interstitial':
                if (m0, msub) in [(1.0, 2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues))[:4]

    close_left = left_eigenvectors[:, close_to_zero_idxs] # Eigenstates of selected eigenvalues
    close_right = right_eigenvectors[:, close_to_zero_idxs]

    close_left = np.sum(np.abs(close_left) ** 2, axis = 1) # Modulus of the complex vectors
    close_right = np.sum(np.abs(close_right) ** 2, axis = 1)
    L = np.sum(np.abs(left_eigenvectors) ** 2, axis = 1)
    R = np.sum(np.abs(right_eigenvectors) ** 2, axis = 1)

    def sum_over_orbitals(arr:np.ndarray):
        return arr[0::2] + arr[1::2]

    def ensure_eigenvector_shape_for_schottky(eigenvector):
        mask = np.full(hamiltonian.shape[0] + len(Lattice.defect_indices), True)
        for i, idx in enumerate(Lattice.defect_indices):
            mask[2 * idx + i % 2] = False
        resized_eigenvector = np.zeros(mask.shape, dtype = L.dtype)
        resized_eigenvector[mask] = eigenvector
        return resized_eigenvector

    # Properly handle Schottky Pair parity elimination
    if (Lattice.defect_indices is not None) and (Lattice.defect_type == 'schottky'):
        L = ensure_eigenvector_shape_for_schottky(L)
        R = ensure_eigenvector_shape_for_schottky(R)
        close_left = ensure_eigenvector_shape_for_schottky(close_left)
        close_right = ensure_eigenvector_shape_for_schottky(close_right)

    data_dictionary = {
        'eigenvalues' : eigenvalues, 
        'L' : sum_over_orbitals(L), 
        'R' : sum_over_orbitals(R), 
        'selected_idxs' : close_to_zero_idxs,
        'selected_left_eigenvectors' : sum_over_orbitals(close_left),
        'selected_right_eigenvectors' : sum_over_orbitals(close_right)
        }
    return data_dictionary


def compute_gap(Lattice, m0, hvec, msub=None):
    hamiltonian = compute_hamiltonian(Lattice, m0, hvec, 1.0, 1.0, msub)
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True)
    
    lowest_energy_eigval = np.sort(np.abs(eigenvalues))[0]

    return lowest_energy_eigval
# endregion
# =============================================================================
# =============================================================================
# region Plotting
def plot_on_lattice(fig:plt.Figure, ldos_ax:plt.Axes, Lattice:DefectLattice, color_array:np.ndarray, plot_type:str, 
                    
                    cmap:str = 'cividis', title:str = None, tick_fontsize:int = 16, label_fontsize:int = 20, scatter_size:int=10,
                    rasterized:bool = True):
    
    lattice = Lattice.lattice
    X = Lattice.X
    Y = Lattice.Y

    if plot_type == 'trisurf':
        ax_pos = ldos_ax.get_position()
        ldos_ax.remove()
        ldos_ax = fig.add_axes(ax_pos, projection="3d")
        plot = ldos_ax.plot_trisurf(X, Y, color_array, cmap=cmap, linewidth=0.2, antialiased=False, rasterized=rasterized)
    elif plot_type == 'scatter':
        plot = ldos_ax.scatter(X, Y, c=color_array, cmap=cmap, s=scatter_size, marker='.', rasterized=rasterized)
    elif plot_type == 'imshow':
        Z = np.full(lattice.size, np.nan)
        filled_idxs = np.argwhere(lattice.flatten() >= 0).flatten()
        Z[filled_idxs] = color_array
        plot = ldos_ax.imshow(Z.reshape(lattice.shape), cmap=cmap, origin='lower', extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)), rasterized=rasterized)
    elif plot_type == "tripcolor":
        triang = tri.Triangulation(X, Y)
        xtri = triang.x[triang.triangles]
        ytri = triang.y[triang.triangles]
        l01 = np.sqrt((xtri[:,1] - xtri[:,0])**2 + (ytri[:,1] - ytri[:,0])**2)
        l12 = np.sqrt((xtri[:,2] - xtri[:,1])**2 + (ytri[:,2] - ytri[:,1])**2)
        l20 = np.sqrt((xtri[:,0] - xtri[:,2])**2 + (ytri[:,0] - ytri[:,2])**2)
        lmax = np.maximum.reduce([l01, l12, l20])
        mask = lmax > np.sqrt(2) + 1e-6
        triang.set_mask(mask)
        plot = ldos_ax.tripcolor(triang, color_array, cmap=cmap, shading='flat', rasterized=rasterized)
    elif plot_type == "tricontourf":
        plot = ldos_ax.tricontourf(X, Y, color_array, 10, cmap=cmap, rasterized=rasterized)
    else:
        raise ValueError("Plot type not provided correctly. It is:", plot_type)

    # Colorbar
    divider = make_axes_locatable(ldos_ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(plot, cax=cax)

    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_powerlimits((0,  0))
    formatter.format = '%.2f'
    cbar.formatter = formatter
    cbar.update_ticks()

    vmin, vmax = plot.get_clim()
    ticks = np.linspace(vmin, vmax, 3)
    cbar.set_ticks(ticks)

    cbar.ax.yaxis.offsetText.set_fontsize(tick_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Ticks
    xticks = [0, np.max(X)]
    yticks = [0, np.max(Y)]

    ldos_ax.set_xticks(xticks)
    ldos_ax.set_yticks(yticks)
    ldos_ax.set_xticklabels([1, Lattice.Lx], fontsize=tick_fontsize)
    ldos_ax.set_yticklabels([1, Lattice.Ly], fontsize=tick_fontsize)

    ldos_ax.set_xlabel("$x$", fontsize=label_fontsize, labelpad=-15)
    ldos_ax.set_ylabel("$y$", rotation=0, fontsize=label_fontsize, labelpad=-10)

    ldos_ax.set_title(title, fontsize=16)
    return ldos_ax, cax


def plot_complex_spectrum(spectrum_ax:plt.Axes, eigenvalues:np.ndarray, scatter_kwargs = {}, highlighted_idxs:int = None, 
                          zoomGap:bool = False):
    eig_real, eig_imag = eigenvalues.real, eigenvalues.imag
    scat = spectrum_ax.scatter(eig_real, eig_imag, **scatter_kwargs, rasterized = False)

    if isinstance(highlighted_idxs, (np.ndarray, list, tuple)):
        scat2 = spectrum_ax.scatter(eig_real[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=2, rasterized = False)

    xmax, ymax = np.round(np.max(eig_real), 1), np.round(np.max(eig_imag), 1)
    xticks = [-xmax, 0.0, xmax]
    yticks = [-ymax, 0.0, ymax]
    spectrum_ax.set_xticks(xticks)
    spectrum_ax.set_yticks(yticks)
    spectrum_ax.set_xticklabels(xticks, fontsize=16)
    spectrum_ax.set_yticklabels(yticks, fontsize=16)
    spectrum_ax.set_xlabel("$\\Re(E)$", fontsize=20)
    spectrum_ax.set_ylabel("$\\Im(E)$", fontsize=20, rotation=0)


    if zoomGap:
        highlighted_eigenvalues = eigenvalues[highlighted_idxs]
        min_real = np.min(highlighted_eigenvalues.real)
        max_real = np.max(highlighted_eigenvalues.real)
        min_imag = np.min(highlighted_eigenvalues.imag)
        max_imag = np.max(highlighted_eigenvalues.imag)

        width_real = max_real - min_real
        width_imag = max_imag - min_imag

        dx = width_real + 1e-3
        dy = width_imag + 1e-3
        axins = spectrum_ax.inset_axes(
            [0.7, 0.05, 0.25, 0.25],
            xlim = (min_real - dx, max_real + dx), ylim = (min_imag - dy, max_imag + dy),
            xticklabels = [], yticklabels = [])
        axins.scatter(eig_real, eig_imag, c='k', s=25, zorder=1)
        axins.scatter(eig_real[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=2)
        #axins.get_xaxis().set_visible(False)
        #axins.get_yaxis().set_visible(False)
        spectrum_ax.indicate_inset_zoom(axins, edgecolor='black')

    return spectrum_ax


def plot_spectrum_ldos(fig, axs, Lattice:DefectLattice, m0:float, h_vector:list, msub:float = None, zoomGap:bool = False):
    eigvec_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h_vector, msub)
    eigenvalues, L, R, close_idxs, close_L, close_R = eigvec_dict.values()

    if Lattice.defect_type in ["interstitial", "frenkel_pair"]:
        plot_type = "tripcolor"
    else:
        plot_type = "tripcolor"

    plot_complex_spectrum(axs[0], eigenvalues, {'c':'k'}, close_idxs, zoomGap = zoomGap)
    plot_on_lattice(fig, axs[1], Lattice, close_L, plot_type, scatter_size = 100)
    plot_on_lattice(fig, axs[2], Lattice, close_R, plot_type, scatter_size = 100)
    plot_on_lattice(fig, axs[3], Lattice,       L, plot_type, scatter_size = 100)
    plot_on_lattice(fig, axs[4], Lattice,       R, plot_type, scatter_size = 100)

    axs[0].set_title("Eigenvalue Spectra", fontsize=16)
    axs[1].set_title("Selected $|\\Psi^L_i|^2$", fontsize=16)
    axs[2].set_title("Selected $|\\Psi^R_i|^2$", fontsize=16)
    axs[3].set_title("All $|\\Psi^L_i|^2$", fontsize=16)
    axs[4].set_title("All $|\\Psi^R_i|^2$", fontsize=16)

    axs[0].set_box_aspect(0.95)

    axs[0].yaxis.set_label_coords(-0.225, 0.4625)
    for ax in axs[1:]:
        ax.set_box_aspect(1)
        ax.set_aspect('equal')

    for ax in axs[2:]:
        ax.set_ylabel("")

    plt.subplots_adjust(wspace=0.25)

    return fig, axs


def big_nvalues_probe(Lattice, m0_values, h_vector, msub_values, ext:str = '.png', overwrite:bool = False):
    if h_vector[0]:
        hdir = 'x'
        h = h_vector[0]
    elif h_vector[1]:
        hdir = 'y'
        h = h_vector[1]
    elif h_vector[2]:
        hdir = 'z'
        h = h_vector[2]
    else:
        hdir = 'NA'
        h = 0.0

    directory = "./NonHermitian/Plots/" + Lattice.defect_type.capitalize() + "/"
    basename = f"{Lattice.defect_type}_h{hdir}={h}"
    if Lattice.defect_type == "frenkel_pair":
        basename += f"_fx={Lattice._fp_xdisp}_fy={Lattice._fp_ydisp}"
    basename += f"_L={Lattice.Lx}"

    if os.path.exists(directory + basename + ext) and not overwrite:
        print(f"File '{directory + basename + ext}' exists and overwrite is False")
        return

    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc(('xtick.major', 'ytick.major'), width=2.5)

    fig, axs = plt.subplots(len(m0_values), 5, figsize=(6 * 5, 6 * len(m0_values)))

    if len(m0_values) == 1:
        axs = np.array(axs).reshape(1, 5)

    if len(msub_values) != len(m0_values):
        msub_values = [None] * len(m0_values)

    for i, (m0, msub) in enumerate(zip(m0_values, msub_values)):
        if all([
            abs(m0) == 2.5,
            msub is not None and abs(msub) == 1.0 and m0 * msub == -2.5,
            any(h == 0.25 for h in h_vector[:2]),
            any(h <= 1.0 for h in h_vector[:2]),
            Lattice.defect_type == "substitution",
            ]):
            zoomGap = True
        elif all([Lattice.defect_type == 'frenkel_pair', m0 == -2.5, any(h != 0 for h in h_vector[:2])]):
            zoomGap = True
        else:
            zoomGap = False

        fig, axs[i, :] = plot_spectrum_ldos(fig, axs[i, :], Lattice, m0, h_vector, msub, zoomGap)

        
        if Lattice.defect_type in ["vacancy", "schottky", "none"]:
            annotation_text = f'$m_0 = {m0}$'
        elif Lattice.defect_type == "substitution":
            annotation_text = f'$m_0^{{\\rm back}} = {m0}$' + f"\n$m_0^{{\\rm sub}} = {msub}$"
        elif Lattice.defect_type in ["interstitial", "frenkel_pair"]:
            annotation_text = f'$m_0^{{\\rm back}} = {m0}$' + f"\n$m_0^{{\\rm int}} = {msub}$"

        axs[i, 0].annotate(
            annotation_text,
            xy = (0.025, 0.975),
            xycoords='axes fraction', 
            ha='left', 
            fontsize=16, 
            rotation=0,
            va='top',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
        )

    for ax in axs[:-1, :].flatten():
        ax.set_xlabel("")

    for ax in axs.flatten():
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title(ax.get_title(), fontsize=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    plt.savefig(directory + basename + ext, bbox_inches='tight', pad_inches=0, dpi=96)
    #plt.savefig(directory + basename + '.png', bbox_inches='tight', pad_inches=0, dpi=300)


def plot_nhse_xyz_left_right(Lx:int, Ly:int, m0:float):
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    plt.subplots_adjust(hspace=0.33, wspace=0.33)

    Lattice_obc = DefectLattice(Lx, Ly, "none", False)
    Lattice_pbc = DefectLattice(Lx, Ly, "none", True)
    eig1_obc, L1, R1, _, _, _ = compute_eigenvectors_eigenvalues(Lattice_obc, m0, [0.25, 0.0, 0.0]).values()
    eig2_obc, L2, R2, _, _, _ = compute_eigenvectors_eigenvalues(Lattice_obc, m0, [0.0, 0.25, 0.0]).values()
    eig3_obc, L3, R3, _, _, _ = compute_eigenvectors_eigenvalues(Lattice_obc, m0, [0.0, 0.0, 0.25]).values()
    eig1_pbc, _, _, _, _, _ =   compute_eigenvectors_eigenvalues(Lattice_pbc, m0, [0.25, 0.0, 0.0]).values()
    eig2_pbc, _, _, _, _, _ =   compute_eigenvectors_eigenvalues(Lattice_pbc, m0, [0.0, 0.25, 0.0]).values()
    eig3_pbc, _, _, _, _, _ =   compute_eigenvectors_eigenvalues(Lattice_pbc, m0, [0.0, 0.0, 0.25]).values()

    eigval_obc_list = [eig1_obc, eig2_obc, eig3_obc]
    eigval_pbc_list= [eig1_pbc, eig2_pbc, eig3_pbc]
    L_list = [L1, L2, L3]
    R_list = [R1, R2, R3]

    for i in range(3):
        plot_complex_spectrum(axs[i, 0], eigval_obc_list[i], scatter_kwargs = {'c':'red', 'label':'OBC'})
        plot_complex_spectrum(axs[i, 0], eigval_pbc_list[i], scatter_kwargs = {'c':'blue', 'label':'PBC'})
        axs[i, 0].legend()
        axs[i, 0].set_title('')
        
        plot_on_lattice(fig, axs[i, 1], Lattice_obc, L_list[i], 'imshow', label_fontsize=24, tick_fontsize=20)
        plot_on_lattice(fig, axs[i, 2], Lattice_obc, R_list[i], 'imshow', label_fontsize=24, tick_fontsize=20)

    #plt.tight_layout()
    plt.savefig(f'./NonHermitian/Plots/nhse_square_lattice_m0={m0}.png')


def compute_figures(L, defect_types, h, h_directions = 'xz', fpd=-3.5,
                    m0_values =   [2.5, 1.0],
                    msub_values = [2.5, 1.0, -1.0, -2.5],
                    overwrite = False):
    
    unique_pairs = np.array(list({(xi, yi) for xi in m0_values for yi in msub_values if xi != yi}))
    sort = np.lexsort((unique_pairs[:, 1], -unique_pairs[:, 0]))
    unique_pairs = unique_pairs[sort]

    default = np.array((h, 0.0, 0.0))
    dir_map = {'x':0,'y':1,'z':2}
    h_vectors = [np.roll(default, dir_map[d]) for d in h_directions]
    # 'none', 'vacancy', 'schottky', 'substitution'
    for deftype in defect_types:
        Lattice = DefectLattice(L, L, deftype, True, schottky_separation=7, defect_radius=1)
        for hv in h_vectors:
            if deftype in ['vacancy', 'schottky', 'none']:
                big_nvalues_probe(Lattice, m0_values, hv, [], overwrite=overwrite)
            else:
                big_nvalues_probe(Lattice, unique_pairs[:, 0], hv, unique_pairs[:, 1], overwrite=overwrite)

    if 'frenkel_pair' not in defect_types:
        return

    for (fpx, fpy) in [(fpd, fpd)]:
        Lattice = DefectLattice(L, L, "frenkel_pair", True, frenkel_x_disp=fpx, frenkel_y_disp=fpy)
        for hv in h_vectors:
            big_nvalues_probe(Lattice, unique_pairs[:, 0], hv, unique_pairs[:, 1], overwrite=overwrite)

# endregion
# =============================================================================
# =============================================================================

def big_defect():
    L = 20
    dr = 1
    Lattice = DefectLattice(L, L, "interstitial", True, frenkel_x_disp=-5.5, frenkel_y_disp=-5.5, defect_radius=dr)
    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    m0, hv, msub = -1.0, [0.25, 0., 0.], -2.5
    plot_spectrum_ldos(fig, axs[1:], Lattice, m0, hv, msub, False)
    eigval_dict = compute_eigenvectors_eigenvalues(Lattice, m0, hv, msub)
    eigvals = eigval_dict['eigenvalues']
    #axs[0].scatter(np.arange(len(eigvals)), np.abs(eigvals), c='k', zorder=0)
    axs[0].scatter(np.arange(len(eigvals)), eigvals.real, c='b', zorder=1, alpha=0.5)
    axs[0].scatter(np.arange(len(eigvals)), eigvals.imag, c='r', alpha=0.1, zorder=2)
    points = np.array([(L / 2 - dr + 0.5, L / 2 - 0.5), (L / 2 - 0.5, L / 2 - dr + 0.5), (L / 2 + dr - 1.5, L / 2 - 0.5), (L / 2 - 0.5, L / 2 + dr - 1.5), (L / 2 - dr + 0.5, L / 2 - 0.5)])
    for ax in axs[2:].flatten():
        ax.plot(points[:, 0], points[:, 1], c='r', ls='-', lw=2, alpha=0.25, zorder=100)

    plt.show()


def compute_gap_over_region(L, deftype, m0_range, h_range, hdir, resolution=(25,25)):
    Lattice = DefectLattice(L, L, deftype, True, schottky_separation=5)
    parameters = tuple(product(np.linspace(m0_range[0], m0_range[1], resolution[0]), 
                               np.linspace(h_range[0], h_range[1], resolution[1])))

    hmap = {'x':0, 'y':1, 'z':2}
    def h_vec_map(h):
        arr = [0., 0., 0.]
        arr[hmap[hdir]] = h
        return arr

    def worker_function(i):
        m0, h = parameters[i]
        h_vec = h_vec_map(h)
        gap = compute_gap(Lattice, m0, h_vec, msub=-1)
        return [m0, h, gap]

    with tqdm_joblib(tqdm(total=len(parameters), desc=f"hi")) as progress_bar:
        data = np.array(Parallel(n_jobs=-1)(delayed(worker_function)(i) for i in range(len(parameters)))).T

    M0, H, GAP = data
    return [M0, H, np.flipud(GAP.reshape(resolution).T)]

if __name__ == "__main__":
    for h in [.5]: 
        if 0: compute_figures(20, ['vacancy'], h=h, overwrite=True, h_directions='x', m0_values=[.25])
        plt.show()
    if 0: 
        Lattice = DefectLattice(20, 20, 'substitution', True)
        find_defect_points(Lattice, 2.5, [0., 0., 0.], -2.5)
    if 1:
        method = 'substitution'
        hdir = 'x'
        M0, H, GAP = compute_gap_over_region(20, method, (0., 2.), (0., 2.), hdir, resolution=(51, 51))
    
        plt.imshow(GAP, extent=(np.min(M0), np.max(M0), np.min(H), np.max(H)))
        plt.xlabel('$m_0$')
        plt.ylabel(f'$h_{hdir}$', rotation=0)
        plt.title(f'Value of Smallest Magnitude Eigenvalue : {method}')
        plt.colorbar()
        plt.savefig('./NonHermitian/Plots/'+f"{method}_{hdir}.png")
        #plt.show()

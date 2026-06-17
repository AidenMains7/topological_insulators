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
                 schottky_separation:int=None, schottky_n_pairs:int=1, frenkel_x_disp:float = -1.5, frenkel_y_disp:float = -0.5): # type: ignore
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
def compute_hamiltonian(Lattice:DefectLattice, m0:float, h_vector:np.ndarray, t:float, t0:float, hsub:"np.ndarray|None" = None):
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    I, Sx, Sy, Cx_plus_Cy = Lattice.wannier_matrices
    hx, hy, hz = h_vector

    defect_indices = Lattice.defect_indices

    hx_matrix = 1.0j * hx * I
    hy_matrix = 1.0j * hy * I
    hz_matrix = 1.0j * hz * I

    if defect_indices is not None:
        if hsub is None and Lattice.defect_type in ['substitution', 'interstitial', 'frenkel_pair']: # type: ignore
            raise ValueError(f"`hsub` cannot be None when defect_indices are provided for 'substitution', 'interstitial', 'frenkel_pair'")
        for idx in defect_indices:
            if (idx >= 0) and (Lattice.defect_type != "schottky"):
                hx_matrix[idx, idx] = 1.0j * hsub[0] # type: ignore
                hy_matrix[idx, idx] = 1.0j * hsub[1] # type: ignore
                hz_matrix[idx, idx] = 1.0j * hsub[2] # type: ignore

    onsite_mass = m0 * I
    dx = t * Sx + hx_matrix
    dy = t * Sy + hy_matrix
    dz = (hz_matrix + onsite_mass) + t0 * Cx_plus_Cy

    hamiltonian = np.kron(dx, pauli_x) + np.kron(dy, pauli_y) + np.kron(dz, pauli_z)

    if Lattice.defect_type == "schottky":
        mask = np.full(hamiltonian.shape[0], True)
        for i, idx in enumerate(defect_indices): # type: ignore
            mask[2 * idx + i % 2] = False
        hamiltonian = hamiltonian[np.ix_(mask, mask)]
    return hamiltonian


def get_separated_points(z, k=6, threshold=0.75):
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

    knn_idxs = np.argsort(knn)
    real_idxs = np.argsort(real_weight)[-2:]
    imag_idxs = np.argsort(imag_weight)[-2:]

    idxs = np.concatenate((knn_idxs, real_idxs, imag_idxs))
    return knn_idxs


def compute_eigenvectors_eigenvalues(Lattice:DefectLattice, m0:float, 
                                     h0_vector:np.ndarray, hsub_vector:"np.ndarray|None" = None, 
                                     n_closest_to_zero:int = 2):

    if n_closest_to_zero is not None:
        assert (n_closest_to_zero <= len(Lattice.X) * 2), "Number of selected indices must be <= number of indices"
    
    hamiltonian = compute_hamiltonian(Lattice, m0, h0_vector, 1.0, 1.0, hsub_vector)
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True) # type: ignore
    sort_idxs = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sort_idxs]
    left_eigenvectors = left_eigenvectors[:, sort_idxs]
    right_eigenvectors = right_eigenvectors[:, sort_idxs]

    hcond = np.linalg.cond(hamiltonian)
    if hcond > 1e10:
        print(f'Hamiltonian condition number for (m0, h0_vector, hsub_vector) = ({m0}, {h0_vector}, {hsub_vector}): {hcond}')

    # Assuming particle-hole symmetry
    #close_to_zero_idxs = np.lexsort((np.abs(eigenvalues.real), np.abs(eigenvalues.imag)))[:n_closest_to_zero]
    abs_sorted_idxs = np.argsort(np.abs(eigenvalues), kind='stable')
    close_to_zero_idxs = abs_sorted_idxs[:n_closest_to_zero]

    if all([Lattice.defect_type == 'substitution',
            m0 == -1.0,
            h0_vector[0] == 0.5]):
        close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[-2:]
    if all([Lattice.defect_type == 'substitution',
            m0 == -1.0,
            h0_vector[2] == 0.5]):
        close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[-2:]
    if all([Lattice.defect_type == 'interstitial',
            m0 == -1.5,
            h0_vector[2] == 0.5]):
        close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[-2:]
    if all([Lattice.defect_type == 'interstitial',
            m0 == -1.5,
            h0_vector[2] == 1.5]):
        close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[:2]
    if all([Lattice.defect_type == 'interstitial',
            m0 == -1.0,
            h0_vector[2] in [0.5]]):
        close_to_zero_idxs = get_separated_points(eigenvalues)[-4:-2]
        print(close_to_zero_idxs)
    if all([Lattice.defect_type == 'interstitial',
            m0 == -1.0,
            h0_vector[2] in [1.5]]):
        close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[:2]
    if all([Lattice.defect_type == 'frenkel_pair',
            m0 == -1.,
            h0_vector[2] == 1.5]):
        close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[:2]

    eigval_sum = np.sum(eigenvalues[close_to_zero_idxs])
    if Lattice.defect_type != 'frenkel_pair':
        while (np.abs(eigval_sum) > 1e-6) and (n_closest_to_zero < len(eigenvalues)):
            n_closest_to_zero += 2
            close_to_zero_idxs = abs_sorted_idxs[:n_closest_to_zero]
            eigval_sum = np.sum(eigenvalues[close_to_zero_idxs])

    close_left = left_eigenvectors[:, close_to_zero_idxs] # Eigenstates of selected eigenvalues
    close_right = right_eigenvectors[:, close_to_zero_idxs]

    close_left = np.sum(np.abs(close_left) ** 2, axis = 1) # Modulus of the complex vectors
    close_right = np.sum(np.abs(close_right) ** 2, axis = 1)
    L = np.sum(np.abs(left_eigenvectors) ** 2, axis = 1)
    R = np.sum(np.abs(right_eigenvectors) ** 2, axis = 1)

    def sum_over_orbitals(arr:np.ndarray):
        return arr[0::2] + arr[1::2]

    def ensure_eigenvector_shape_for_schottky(eigenvector):
        mask = np.full(hamiltonian.shape[0] + len(Lattice.defect_indices), True) # type: ignore
        for i, idx in enumerate(Lattice.defect_indices): # type: ignore
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
# endregion
# =============================================================================
# =============================================================================
# region Plotting
def plot_on_lattice(fig:plt.Figure, ldos_ax:plt.Axes, Lattice:DefectLattice, color_array:np.ndarray, plot_type:str,  # type: ignore
                    cmap:str = 'cividis', title:"str|None" = None, tick_fontsize:int = 16, label_fontsize:int = 20, scatter_size:int=10,
                    rasterized:bool = True):
    
    lattice = Lattice.lattice
    X = Lattice.X
    Y = Lattice.Y

    if plot_type == 'trisurf':
        ax_pos = ldos_ax.get_position()
        ldos_ax.remove()
        ldos_ax = fig.add_axes(ax_pos, projection="3d") # type: ignore
        plot = ldos_ax.plot_trisurf(X, Y, color_array, cmap=cmap, linewidth=0.2, antialiased=False, rasterized=rasterized) # type: ignore
    elif plot_type == 'scatter':
        plot = ldos_ax.scatter(X, Y, c=color_array, cmap=cmap, s=scatter_size, marker='.', rasterized=rasterized)
    elif plot_type == 'imshow':
        Z = np.full(lattice.size, np.nan)
        filled_idxs = np.argwhere(lattice.flatten() >= 0).flatten()
        Z[filled_idxs] = color_array
        plot = ldos_ax.imshow(Z.reshape(lattice.shape), cmap=cmap, origin='lower', extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)), rasterized=rasterized) # type: ignore
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
    cbar.set_ticks(ticks) # type: ignore

    cbar.ax.yaxis.offsetText.set_fontsize(tick_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Ticks
    xticks = [0, np.max(X)]
    yticks = [0, np.max(Y)]

    ldos_ax.set_xticks(xticks)
    ldos_ax.set_yticks(yticks)
    ldos_ax.set_xticklabels([1, Lattice.Lx], fontsize=tick_fontsize) # type: ignore
    ldos_ax.set_yticklabels([1, Lattice.Ly], fontsize=tick_fontsize) # type: ignore

    ldos_ax.set_xlabel("$x$", fontsize=label_fontsize, labelpad=-15)
    ldos_ax.set_ylabel("$y$", rotation=0, fontsize=label_fontsize, labelpad=-10)

    ldos_ax.set_title(title, fontsize=16) # type: ignore
    return ldos_ax, cax


def plot_complex_spectrum(spectrum_ax:plt.Axes, eigenvalues:np.ndarray, scatter_kwargs = {}, highlighted_idxs:"int|None" = None,  # type: ignore
                          zoomGap:bool = False):
    eig_real, eig_imag = eigenvalues.real, eigenvalues.imag
    scat = spectrum_ax.scatter(eig_real, eig_imag, **scatter_kwargs, rasterized = False)
    #scat_real = spectrum_ax.scatter(np.arange(len(eig_real)), eig_real, c='blue', s=25, zorder=2, rasterized = False)
    #scat_imag = spectrum_ax.scatter(np.arange(len(eig_imag)), eig_imag, c='orange', s=25, zorder=2, rasterized = False)

    if isinstance(highlighted_idxs, (np.ndarray, list, tuple)):
        scat2 = spectrum_ax.scatter(eig_real[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=2, rasterized = False)
        #scat_real2 = spectrum_ax.scatter(np.arange(len(eig_real))[highlighted_idxs], eig_real[highlighted_idxs], c='red', s=25, zorder=3, rasterized = False)
        #scat_imag2 = spectrum_ax.scatter(np.arange(len(eig_imag))[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=3, rasterized = False)

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
            [0.7, 0.05, 0.25, 0.25], # type: ignore
            xlim = (min_real - dx, max_real + dx), ylim = (min_imag - dy, max_imag + dy),
            xticklabels = [], yticklabels = [])
        axins.scatter(eig_real, eig_imag, c='k', s=25, zorder=1)
        axins.scatter(eig_real[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=2)
        #axins.get_xaxis().set_visible(False)
        #axins.get_yaxis().set_visible(False)
        spectrum_ax.indicate_inset_zoom(axins, edgecolor='black')

    return spectrum_ax


def plot_spectrum_ldos(fig, axs, Lattice:DefectLattice, m0:float, h0_vector:np.ndarray, hsub_vector:"np.ndarray|None" = None, zoomGap:bool = False):
    eigvec_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h0_vector, hsub_vector) 
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


def plot_many_spectrum_lr(defect_type: str, L: int, m0_values: list[float], h_dir: str, h0_values: list[float], hsub_values: "list[float]|None" = None, ext: str = '.png', defect_radius:int = 1, out_filename:str = ''):
    assert defect_type in ['none', 'vacancy', 'schottky', 'substitution', 'interstitial', 'frenkel_pair'], "defect_type must be one of 'none', 'vacancy', 'schottky', 'substitution', 'interstitial', or 'frenkel_pair'"
    assert h_dir in 'xyz', "h_dir must be 'x', 'y', or 'z'"
    if hsub_values is not None:
        assert len(h0_values) == len(hsub_values) == len(m0_values), "h0_values, hsub_values, and m0_values must be of equal length"

    if defect_type in ['none', 'vacancy', 'schottky']:
        hsub_values = h0_values

    directory = "./NonHermitian/Plots/" + defect_type.capitalize() + "/"
    basename = f"{defect_type}_h{h_dir}"

     
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc(('xtick.major', 'ytick.major'), width=2.5) # type: ignore

    fig, axs = plt.subplots(len(m0_values), 5, figsize=(6 * 5, 6 * len(m0_values)))

    if len(m0_values) == 1:
        axs = np.array(axs).reshape(1, 5)

    for i, (m0, h0, hsub) in enumerate(zip(m0_values, h0_values, hsub_values)): # type: ignore
        Lattice = DefectLattice(L, L, defect_type, True, schottky_separation = L // 4, 
                                frenkel_x_disp = -3.5, frenkel_y_disp = -3.5, defect_radius = defect_radius)

        if h_dir == 'x':
            h_vector = np.array([h0, 0.0, 0.0])
            hsub_vector = np.array([hsub, 0.0, 0.0]) if hsub is not None else None
        elif h_dir == 'y':
            h_vector = np.array([0.0, h0, 0.0])
            hsub_vector = np.array([0.0, hsub, 0.0]) if hsub is not None else None
        elif h_dir == 'z':
            h_vector = np.array([0.0, 0.0, h0])
            hsub_vector = np.array([0.0, 0.0, hsub]) if hsub is not None else None

        fig, axs[i, :] = plot_spectrum_ldos(fig, axs[i, :], Lattice, m0, h_vector, hsub_vector)

        if Lattice.defect_type in ["vacancy", "schottky", "none"]:
            annotation_text = f'$m_0 = {m0}$\n$h_0={h0} \\hat{{{h_dir}}}$'
        elif Lattice.defect_type == "substitution":
            annotation_text = f'$m_0 = {m0}$\n$h_0={h0} \\hat{{{h_dir}}}$\n$h_0^{{\\rm sub}}={hsub} \\hat{{{h_dir}}}$'
        elif Lattice.defect_type in ["interstitial", "frenkel_pair"]:
            annotation_text = f'$m_0 = {m0}$\n$h_0={h0} \\hat{{{h_dir}}}$\n$h_0^{{\\rm int}}={hsub} \\hat{{{h_dir}}}$'

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

    if out_filename == '':
        plt.savefig(directory + basename + ext, bbox_inches='tight', pad_inches=0, dpi=96)
    else:
        plt.savefig(directory + out_filename + ext, bbox_inches='tight', pad_inches=0, dpi=96)
    #plt.show()

# endregion
# =============================================================================
# =============================================================================

def main():
    L = 20

    h_dir = 'x'
    m0_values = [-1., -1.]
    h0_values = [0.5, 1.5]
    hsub_values = [1.5, 0.5]
    plot_many_spectrum_lr('vacancy',  L, m0_values, h_dir, h0_values)
    plot_many_spectrum_lr('schottky', L, m0_values, h_dir, h0_values)
    plot_many_spectrum_lr('frenkel_pair', L, m0_values, h_dir, h0_values, hsub_values)

    m0_values = [-1.5, -1.5]
    h0_values = [0.5, 1.5]
    hsub_values = [1.5, 0.5]
    plot_many_spectrum_lr('substitution', L, m0_values, h_dir, h0_values, hsub_values, defect_radius = 1)
    plot_many_spectrum_lr('interstitial', L, m0_values, h_dir, h0_values, hsub_values, defect_radius = 1)
    
    h_dir = 'z'
    m0_values = [-1., -1.]
    h0_values = [0.5, 1.5]
    hsub_values = [1.5, 0.5]
    plot_many_spectrum_lr('vacancy',  L, m0_values, h_dir, h0_values)
    plot_many_spectrum_lr('schottky', L, m0_values, h_dir, h0_values)
    plot_many_spectrum_lr('frenkel_pair', L, m0_values, h_dir, h0_values, hsub_values)

    m0_values = [-1.5, -1.5]
    h0_values = [0.5, 1.5]
    hsub_values = [1.5, 0.5]
    plot_many_spectrum_lr('substitution', L, m0_values, h_dir, h0_values, hsub_values, defect_radius = 1)
    plot_many_spectrum_lr('interstitial', L, m0_values, h_dir, h0_values, hsub_values, defect_radius = 1)


def plot_spectrum_and_ldos():
    L = 20
    Lattice = DefectLattice(L, L, 'substitution', True, defect_radius = 1)

    H = compute_hamiltonian(Lattice, -1., np.array([0.5, 0.0, 0.0]), 1.0, 1.0, np.array([1.5, 0.0, 0.0]))
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(H, left=True, right=True, overwrite_a=True) # type: ignore
    sort_idxs = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sort_idxs]
    left_eigenvectors = left_eigenvectors[:, sort_idxs]
    right_eigenvectors = right_eigenvectors[:, sort_idxs]

    eigenvalue_dict = compute_eigenvectors_eigenvalues(Lattice, -1., 
                        np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0]), n_closest_to_zero = 2)
  
    eigenvalues = eigenvalue_dict['eigenvalues']
    selected_idxs = eigenvalue_dict['selected_idxs']
    selected_left_eigenvectors = eigenvalue_dict['selected_left_eigenvectors']
    selected_right_eigenvectors = eigenvalue_dict['selected_right_eigenvectors']
    L = eigenvalue_dict['L']
    R = eigenvalue_dict['R']
    n = np.arange(len(eigenvalues))
    
    # Spectrum & LDOS
    if 0:
        fig, axs = plt.subplots(1, 7, figsize=(6*7, 6))
        axs[0].scatter(n, eigenvalues.real, c='black', s=25, zorder=0)
        axs[0].scatter(n[selected_idxs], eigenvalues.real[selected_idxs], c='red', s=25, zorder=1)
        axs[1].scatter(n, eigenvalues.imag, c='black', s=25, zorder=0)
        axs[1].scatter(n[selected_idxs], eigenvalues.imag[selected_idxs], c='red', s=25, zorder=1)

        for ax in axs[:2]:
            ax.set_xlabel("$n$")
            ax.set_ylabel("$E_n$")
        
        axs[0].set_title("Real Part of Eigenvalues")
        axs[1].set_title("Imaginary Part of Eigenvalues")

        axs[2].scatter(eigenvalues.real, eigenvalues.imag, c='black', s=25, zorder=0)
        axs[2].scatter(eigenvalues.real[selected_idxs], eigenvalues.imag[selected_idxs], c='red', s=25, zorder=1)
        axs[2].set_xlabel("$\\Re(E)$")
        axs[2].set_ylabel("$\\Im(E)$")
        axs[2].set_title("Complex Eigenvalue Spectrum")

        axs[3].tripcolor(Lattice.X, Lattice.Y, selected_left_eigenvectors, shading='flat', cmap='cividis')
        axs[4].tripcolor(Lattice.X, Lattice.Y, selected_right_eigenvectors, shading='flat', cmap='cividis')
        axs[5].tripcolor(Lattice.X, Lattice.Y, L, shading='flat', cmap='cividis')
        axs[6].tripcolor(Lattice.X, Lattice.Y, R, shading='flat', cmap='cividis')
        plt.savefig('./NonHermitian/Plots/temp.png')


def plot_ipr(method, Ls: list[int], m0: float, h0: float, hsub: float, h_dir: str, ax):
    left_IPRs = []
    eigenvalues_list = []

    h_dir_mapping = {'x': 0, 'y': 1, 'z': 2}
    h0_vector = np.zeros(3)
    hsub_vector = np.zeros(3)
    h0_vector[h_dir_mapping[h_dir]] = h0
    hsub_vector[h_dir_mapping[h_dir]] = hsub

    for L in Ls:
        Lattice = DefectLattice(L, L, method, True, defect_radius = 1)

        H = compute_hamiltonian(Lattice, m0, h0_vector, 1.0, 1.0, hsub_vector)
        eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(H, left=True, right=True, overwrite_a=True) # type: ignore
        sort_idxs = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[sort_idxs]
        left_eigenvectors = left_eigenvectors[:, sort_idxs]

        left_eigvec_sq = np.abs(left_eigenvectors) ** 2
        left_eigvec_sq = left_eigvec_sq[len(eigenvalues) // 2:, :] + left_eigvec_sq[:len(eigenvalues) // 2, :]
        left_IPR = np.sum(left_eigvec_sq ** 2, axis=0)
        
        left_IPRs.append(left_IPR)
        eigenvalues_list.append(eigenvalues)
        print(f"Completed computation for L = {L}")

    for L, eigs, left_IPR in zip(Ls, eigenvalues_list, left_IPRs):
        t = np.abs(eigs) * np.sign(eigs.real)
        label = f'L={L}'
        ax.scatter(t, left_IPR,  s=25, alpha=0.25, label=label)

    ax.set_ylabel("IPR")
    ax.set_title("Left Eigenvector IPR")
    ax.set_xlabel("$|E|\\times {\\rm sign}(\\Re(E))$")
    ax.legend()

    return ax


def plot_ipr_figure(method, h_dir, Ls, m0_values, h0_values, hsub_values):
    fig, axs = plt.subplots(1, len(m0_values), figsize=(6 * len(m0_values), 6))
    axs = np.array(axs)
    
    all_annotations = [f"({letter})" for letter in 'abcdefghijklmnopqrstuvwxyz']

    for i, (m0, h0, hsub) in enumerate(zip(m0_values, h0_values, hsub_values)):
        print(f"Plotting for m0={m0}, h0={h0}, hsub={hsub}")
        axs[i] = plot_ipr(method, Ls, m0, h0, hsub, h_dir, axs[i])
        axs[i].annotate(
            all_annotations[i],
            xy = (0.025, 0.975),
            xycoords='axes fraction', 
            ha='left', 
            fontsize=24, 
            rotation=0,
            va='top',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
        )

    plt.tight_layout()
    plt.savefig(f'./NonHermitian/Plots/IPR/{method}_ipr_{h_dir}.png')


if __name__ == "__main__":
    #main()
    method = 'vacancy'
    Ls = [10, 20, 30, 40]

    m0_values = [-1.0, -1.0]
    h0_values = [0.5, 1.5]
    hsub_values = [1.5, 0.5]

    for h_dir in 'xz':
        plot_ipr_figure(method, h_dir, Ls, m0_values, h0_values, hsub_values)

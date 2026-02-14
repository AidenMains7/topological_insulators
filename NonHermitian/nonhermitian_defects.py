import numpy as np
import scipy.linalg as spla
from scipy.sparse import dok_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
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
    X *= 2
    Y *= 2

    large_lattice = np.full((2 * Ly - 1, 2 * Lx - 1), np.nan)
    large_lattice[Y, X] = np.arange(len(X))

    Y_pos = []
    X_pos = []
    for i in range(-interstitial_radius, interstitial_radius):
        for j in range(-interstitial_radius, interstitial_radius):
            if abs(i) + abs(j) < interstitial_radius:
                Y_pos.append(Ly - 1 + 2 * i)
                X_pos.append(Lx - 1 + 2 * j)
                large_lattice[Ly - 1 + 2 * i, Lx - 1 + 2 * j] = np.inf

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


class DefectLattice:
    def __init__(self, Lx:int, Ly:int, defect_type:str, pbc:bool, defect_radius:int=1,
                 schottky_separation:int=None, schottky_n_pairs:int=1, frenkel_x_disp:int = -1.5, frenkel_y_disp:int = -0.5):
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
                raise ValueError()

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

    # region DefectLattice properties
    @property
    def lattice(self):
        return self._lattice
    @property
    def defect_indices(self):
        return self._defect_indices
    @property
    def X(self):
        return self._X
    @property
    def Y(self):
        return self._Y
    @property
    def defect_type(self):
        return self._defect_type
    @property
    def pbc(self):
        return self._pbc
    @property
    def wannier_matrices(self):
        return self._wannier_matrices
    @property
    def dx(self):
        return self._dx
    @property
    def dy(self):
        return self._dy
    @property
    def Lx(self):
        return self._Lx
    @property
    def Ly(self):
        return self._Ly
    # endregion
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
    
    def compute_wannier_matrices_polar(self):
        dx = self.dx
        dy = self.dy
        theta = np.arctan2(dy, dx)
        dr = np.sqrt(dx ** 2 + dy ** 2)

        # Create masks for different types of hopping. 
        distance_mask = ((dr <= 1 + 1e-6) & (dr > 1e-6)) # Mask for distances close to 1
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


def get_n_separated_points(z, k=6, n=2):
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

    plt.scatter(np.arange(knn.size), knn, zorder=0)

    top = np.argwhere(knn > np.median(knn))
    q3 = np.median(knn[top])
    
    plt.axhline(q3, c='k', ls='--', zorder=2)
    plt.show()
    plt.scatter(z.real, z.imag, c=knn)
    plt.show()

    idx = np.argsort(knn)[-n:]

    return z[idx], idx


def get_factor_separated_points(z, k=6, factor=10.0):
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

    # Typical bulk spacing
    median_spacing = np.median(knn)

    # Outliers = unusually large local spacing
    mask = knn > factor * median_spacing

    return z[mask], np.where(mask)[0]


def compute_eigenvectors_eigenvalues(Lattice:DefectLattice, m0:float, 
                                     h_vector:np.ndarray, msub:float = None, 
                                     n_closest_to_zero:int=2):
    
    assert n_closest_to_zero <= len(Lattice.X) * 2, "Number of selected indices must be <= number of indices"
    
    hamiltonian = compute_hamiltonian(Lattice, m0, h_vector, 1.0, 1.0, msub)
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True)
    sort_idxs = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sort_idxs]
    left_eigenvectors = left_eigenvectors[:, sort_idxs]
    right_eigenvectors = right_eigenvectors[:, sort_idxs]

    close_to_zero_idxs = np.argsort(np.abs(eigenvalues))[:n_closest_to_zero]
    _, close_to_zero_idxs = get_n_separated_points(eigenvalues)
    
    close_left = left_eigenvectors[:, close_to_zero_idxs]
    close_right = right_eigenvectors[:, close_to_zero_idxs]

    close_left = np.sum(np.abs(close_left) ** 2, axis = 1)
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
        'close_to_zero_idxs' : close_to_zero_idxs,
        'close_to_zero_left_eigenvectors' : sum_over_orbitals(close_left),
        'close_to_zero_right_eigenvectors' : sum_over_orbitals(close_right)
        }
    return data_dictionary
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

# region Plotting
def plot_on_lattice(fig:plt.Figure, ldos_ax:plt.Axes, Lattice:DefectLattice, color_array:np.ndarray, plot_type:str, 
                    cmap:str = 'cividis', title:str = None, tick_fontsize:int = 16, label_fontsize:int = 20, scatter_size:int=10):
    
    lattice = Lattice.lattice
    Y, X = np.where(lattice >= 0)[:]

    if plot_type == 'trisurf':
        ax_pos = ldos_ax.get_position()
        ldos_ax.remove()
        ldos_ax = fig.add_axes(ax_pos, projection="3d")
        plot = ldos_ax.plot_trisurf(X, Y, color_array, cmap=cmap, linewidth=0.2, antialiased=False)

    elif plot_type == 'scatter':
        plot = ldos_ax.scatter(X, Y, c=color_array, cmap=cmap, s=scatter_size, marker='.')

    elif plot_type == 'imshow':
        Z = np.full(lattice.size, np.nan)
        filled_idxs = np.argwhere(lattice.flatten() >= 0).flatten()
        Z[filled_idxs] = color_array
        plot = ldos_ax.imshow(Z.reshape(lattice.shape), cmap=cmap, origin='lower', extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)))
    elif plot_type == "tripcolor":
        plot = ldos_ax.tripcolor(X, Y, color_array, cmap=cmap)
    elif plot_type == "tricontourf":
        plot = ldos_ax.tricontourf(X, Y, color_array, 10, cmap=cmap)
    else:
        raise ValueError("Plot type not provided correctly. It is:", plot_type)

    divider = make_axes_locatable(ldos_ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)

    cbar = fig.colorbar(plot, cax=cax)
    vmin, vmax = plot.get_clim()
    cbar.ax.yaxis.set_ticks([vmin, vmax])
    cbar.ax.yaxis.set_ticklabels([str(np.round(v, 1)) for v in [vmin, vmax]])
    cbar.ax.tick_params(labelsize=tick_fontsize)

    xticks = [0, np.max(X)]
    yticks = [0, np.max(Y)]

    ldos_ax.set_xticks(xticks)
    ldos_ax.set_yticks(yticks)
    ldos_ax.set_xticklabels([1, Lattice.Lx], fontsize=tick_fontsize)
    ldos_ax.set_yticklabels([1, Lattice.Ly], fontsize=tick_fontsize)

    ldos_ax.set_xlabel("$x$", fontsize=label_fontsize, labelpad=-15)
    ldos_ax.set_ylabel("$y$", rotation=0, fontsize=label_fontsize, labelpad=-10)
    #if plot_type != 'trisurf':
    #    ldos_ax.yaxis.tick_right()
    #    ldos_ax.yaxis.set_label_position("right")
    #    cbar.ax.yaxis.tick_left()


    ldos_ax.set_title(title, fontsize=16)
    return ldos_ax, cax


def plot_complex_spectrum(spectrum_ax:plt.Axes, eigenvalues:np.ndarray, defect_indices:list = None, scatter_kwargs = {}, highlighted_idxs:int = None, zoomGap:bool = False):
    eig_real, eig_imag = eigenvalues.real, eigenvalues.imag
    scat = spectrum_ax.scatter(eig_real, eig_imag, **scatter_kwargs)

    if isinstance(highlighted_idxs, (np.ndarray, list, tuple)):
        scat2 = spectrum_ax.scatter(eig_real[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=2)

    if defect_indices != None:
        defect_indices = np.array(defect_indices)
        defect_indices = defect_indices[defect_indices >= 0]

        #scat2 = spectrum_ax.scatter(eig_real[2 * defect_indices], eig_imag[2 * defect_indices], c='red', s=25, zorder=2)
        #scat3 = spectrum_ax.scatter(eig_real[2 * defect_indices + 1], eig_imag[2 * defect_indices + 1], c='red', s=25, zorder=2)

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
        dx = 1.5
        dy = 0.1 / (16 * 2)
        x1, x2, y1, y2 = -dx, dx, -dy, dy
        axins = spectrum_ax.inset_axes(
            [0.7, 0.05, 0.25, 0.25],
            xlim = (x1, x2), ylim = (y1, y2),
            xticklabels = [], yticklabels = [])
        axins.scatter(eig_real, eig_imag, c='k', s=25, zorder=1)
        axins.scatter(eig_real[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=2)
        axins.get_xaxis().set_visible(False)
        axins.get_yaxis().set_visible(False)
        spectrum_ax.indicate_inset_zoom(axins, edgecolor='black')

    return spectrum_ax


def plot_probe_single_value(Lattice:DefectLattice, m0:float, h_vector:list, msub:float = None, ext:str = '.png', overwrite:bool = False, doSave:bool = False):
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

    directory = "./NonHermitian/Plots/" + Lattice.defect_type.capitalize() + "/"
    basename = f"{Lattice.defect_type}_h{hdir}={h}_m0={m0}"
    if Lattice.defect_type not in ['none', 'vacancy', 'schottky']:
        basename += f"_msub={msub}"

    basename += f"_L={Lattice.Lx}"

    if os.path.exists(directory + basename + ext) and not overwrite:
        print("Image file already exists for: ", f"{Lattice.defect_type}, m0={m0}, msub={msub}")
        return
    
    fig, axs = plt.subplots(1, 5, figsize=(30, 6))
    eigvec_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h_vector, msub, 2)
    eigenvalues, L, R, close_idxs, close_L, close_R = eigvec_dict.values()

    if Lattice.defect_type in ["interstitial", "frenkel_pair"]:
        plot_type = "tripcolor"
    else:
        plot_type = "imshow"

    plot_complex_spectrum(axs[0], eigenvalues, Lattice.defect_indices, {'c':'k'}, highlighted_idxs = close_idxs)
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


    plt.savefig(directory + basename + ext)
    if doSave:
        np.savez(directory.replace("Plots", "Data") + basename + '.npz', **eigvec_dict)


def plot_info(fig, axs, Lattice:DefectLattice, m0:float, h_vector:list, msub:float = None, zoomGap:bool = False):
    eigvec_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h_vector, msub, 2)
    eigenvalues, L, R, close_idxs, close_L, close_R = eigvec_dict.values()

    if Lattice.defect_type in ["interstitial", "frenkel_pair"]:
        plot_type = "tripcolor"
    else:
        plot_type = "tripcolor"

    plot_complex_spectrum(axs[0], eigenvalues, Lattice.defect_indices, {'c':'k'}, close_idxs, zoomGap = zoomGap)
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

    directory = "./NonHermitian/Plots/" + Lattice.defect_type.capitalize() + "/"
    basename = f"{Lattice.defect_type}_h{hdir}={h}"
    if Lattice.defect_type == "frenkel_pair":
        basename += f"_fx={Lattice._fp_xdisp}_fy={Lattice._fp_ydisp}"
    basename += f"_L={Lattice.Lx}"
    fig, axs = plt.subplots(len(m0_values), 5, figsize=(6 * 5, 6 * len(m0_values)))

    if len(msub_values) != len(m0_values):
        msub_values = [None] * len(m0_values)

    for i, (m0, msub) in enumerate(zip(m0_values, msub_values)):
        if m0 == 2.5 and h_vector[0] != 0 and Lattice.defect_type == "substitution":
            zoomGap = True
        else:
            zoomGap = False

        fig, axs[i, :] = plot_info(fig, axs[i, :], Lattice, m0, h_vector, msub, zoomGap)

        
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
    
    plt.savefig(directory + basename + ext)


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


# endregion


if __name__ == "__main__":

    Lattice = DefectLattice(20, 20, "interstitial", True)
    _ = compute_eigenvectors_eigenvalues(Lattice, 1.0, [0., 0., 1.5], msub = -1.0)

    raise SystemExit

    L = 20
    twelve_m0 = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -2.5, -2.5, -2.5]
    twelve_msub = [-2.5, -1.0, 1.0, -2.5, -1.0, 2.5, -2.5, 1.0, 2.5, -1.0, 1.0, 2.5]
    fpx = 0
    fpy = 0

    for deftype in ['interstitial']:
        Lattice = DefectLattice(L, L, deftype, True, schottky_separation=7)
        for hv in [[0., 0.0, 0.25]]:
            if deftype in ['vacancy', 'schottky', 'none']:
                big_nvalues_probe(Lattice, [2.5, 1.0, -1.0, -2.5], hv, [])
            else:
                big_nvalues_probe(Lattice, twelve_m0, hv, twelve_msub)
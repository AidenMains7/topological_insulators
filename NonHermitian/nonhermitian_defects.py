import numpy as np
import scipy.linalg as spla
from scipy.sparse import dok_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
import os, h5py
from itertools import product

from cProfile import Profile
import pstats
import functools


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
    assert (Lx*Ly % 2 == 1), "Side lengths must be odd"
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
    assert (Lx * Ly % 2 == 1), "Side lengths must be odd"
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


def generate_frenkel_pair_lattice(Lx:int, Ly:int):
    assert (Lx * Ly % 2 == 1), "Side lengths must be odd"

    lattice, _ = generate_square_lattice(Lx, Ly)
    Y, X = np.where(lattice >= 0)
    X = X * 2
    Y = Y * 2

    large_lattice = np.full((2 * Ly - 1, 2 * Lx - 1), np.nan)
    large_lattice[Y, X] = np.arange(len(X))
    large_lattice[Ly - 1, Lx - 1] = -1 # Vacancy at center
    large_lattice[Ly - 2, Lx - 4] = np.inf # Interstitial site

    large_lattice[np.where(large_lattice >= 0)] = np.arange(len(np.where(large_lattice >= 0)[0].flatten()))
    defect_indices = [-1, int(large_lattice[Ly - 2, Lx - 4])]
    return large_lattice, defect_indices

# endregion

class DefectLattice:
    def __init__(self, Lx:int, Ly:int, defect_type:str, pbc:bool, defect_radius:int=1,
                 schottky_separation:int=None, schottky_n_pairs:int=1):
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
                self._lattice, self._defect_indices = generate_frenkel_pair_lattice(Lx, Ly)
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
        if msub == None and defect_type in ['substitution', 'interstitial', 'frenkel_pair']:
            raise ValueError(f"`mu` cannot be None when defect_indices are provided for 'substitution', 'interstitial', 'frenkel_pair'")
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


def compute_sum_normed_eigenvectors(hamiltonian:np.ndarray, defect_type:str, defect_indices:list):
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True)
    sort_idxs = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sort_idxs]
    left_eigenvectors = left_eigenvectors[:, sort_idxs]
    right_eigenvectors = right_eigenvectors[:, sort_idxs]

    L = np.sum(np.einsum('ij,ij->ij', left_eigenvectors.conj(), left_eigenvectors), axis=1).real
    R = np.sum(np.einsum('ij,ij->ij', right_eigenvectors.conj(), right_eigenvectors), axis=1).real
    L_over_R = L / R

    def sum_over_orbitals(arr:np.ndarray):
        return arr[0::2] + arr[1::2]

    # Properly handle Schottky Pair parity elimination
    if (defect_indices != None) and (defect_type == 'schottky'):
        mask = np.full(hamiltonian.shape[0] + len(defect_indices), True)
        for i, idx in enumerate(defect_indices):
            mask[2 * idx + i % 2] = False
        new = np.zeros(mask.shape, dtype=L.dtype)
        new[mask] = L_over_R
        L_over_R = new

    return eigenvalues, sum_over_orbitals(L_over_R)


def compute_eigenvectors_eigenvalues(Lattice:DefectLattice, m0:float, h_vector:np.ndarray, msub:float = None):
    hamiltonian = compute_hamiltonian(Lattice, m0, h_vector, 1.0, 1.0, msub)
    eigenvalues, L_over_R = compute_sum_normed_eigenvectors(hamiltonian, defect_type = Lattice.defect_type, defect_indices = Lattice.defect_indices)
    data_dictionary = {
        'eigenvalues': eigenvalues,
        'L_over_R': L_over_R
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
def plot_real_spectrum(spectrum_ax:plt.Axes, eigenvalues:np.ndarray, n_highlighted_sites:int=2):
    assert n_highlighted_sites % 2 == 0, "n_highlighted_sites must be an even natural number"

    eig_indices = np.arange(len(eigenvalues))
    scat_real = spectrum_ax.scatter(eig_indices, eigenvalues.real, c='black', s=25, zorder=1)

    center_idxs = np.array([len(eigenvalues) // 2 + i for i in range(-n_highlighted_sites // 2, n_highlighted_sites // 2)])
    scat2 = spectrum_ax.scatter(eig_indices[center_idxs], eigenvalues.real[center_idxs], c='red', s=25, zorder=2)

    xticks = [0, len(eigenvalues) / 2, len(eigenvalues)]
    spectrum_ax.set_xticks(xticks)
    spectrum_ax.set_xticklabels([str(int(tick + 1)) for tick in xticks], fontsize=12)
    spectrum_ax.set_xlabel("Eigenvalue Index ($n$)", fontsize=16)
    spectrum_ax.set_ylabel("Real Eigenvalue Energy $E_n$", fontsize=16)
    return spectrum_ax


def plot_on_lattice(fig:plt.Figure, ldos_ax:plt.Axes, lattice:np.ndarray, color_array:np.ndarray, plot_type:str, cmap:str = 'cividis', title:str = None):

    Y, X = np.where(lattice >= 0)[:]

    if plot_type == 'trisurf':
        ax_pos = ldos_ax.get_position()
        ldos_ax.remove()
        ldos_ax = fig.add_axes(ax_pos, projection="3d")
        plot = ldos_ax.plot_trisurf(X, Y, color_array, cmap=cmap, linewidth=0.2, antialiased=False)

    elif plot_type == 'scatter':
        plot = ldos_ax.scatter(X, Y, c=color_array, cmap=cmap, s=50, marker='.', edgecolors='k')

    elif plot_type == 'imshow':
        Z = np.full(lattice.size, np.nan)
        filled_idxs = np.argwhere(lattice.flatten() >= 0).flatten()
        Z[filled_idxs] = color_array
        plot = ldos_ax.imshow(Z.reshape(lattice.shape), cmap=cmap, origin='lower', extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)))
    else:
        raise ValueError()

    ldos_ax.set_aspect("equal")

    cax = inset_axes(
        ldos_ax, 
        width="100%",  # width as a percentage of parent
        height="100%",  # height as a percentage of parent
        bbox_to_anchor=(-0.15, 0.3/2, 0.1, 0.70),  # (x0, y0, width, height) in axes fraction (on the left)
        bbox_transform=ldos_ax.transAxes,
        borderpad=0
    )

    cbar = plt.colorbar(plot, ax=ldos_ax, cax=cax)
    cbar.ax.yaxis.tick_left()
    vmin, vmax = plot.get_clim()
    cbar.ax.yaxis.set_ticks([vmin, vmax])
    cbar.ax.yaxis.set_ticklabels([str(np.round(v, 3)) for v in [vmin, vmax]])

    xticks = [0, np.max(X), np.max(X)]
    yticks = [0, np.max(Y), np.max(Y)]

    ldos_ax.set_xticks(xticks)
    ldos_ax.set_yticks(yticks)
    ldos_ax.set_xticklabels([str(tick + 1) for tick in xticks], fontsize=12)
    ldos_ax.set_yticklabels([str(tick + 1) for tick in yticks], fontsize=12)

    ldos_ax.set_xlabel("$L_x$", fontsize=16, labelpad=-15)
    ldos_ax.set_ylabel("$L_y$", rotation=0, fontsize=16, labelpad=-10)
    if plot_type != 'trisurf':
        ldos_ax.yaxis.tick_right()
        ldos_ax.yaxis.set_label_position("right")


    ldos_ax.set_title(title, fontsize=16)
    return ldos_ax, cax


def plot_complex_spectrum(spectrum_ax:plt.Axes, eigenvalues:np.ndarray, defect_indices:list = None):
    eig_real, eig_imag = eigenvalues.real, eigenvalues.imag
    scat = spectrum_ax.scatter(eig_real, eig_imag, c='black', s=25, zorder=1)


    if defect_indices != None:
        defect_indices = np.array(defect_indices)
        defect_indices = defect_indices[defect_indices >= 0]

        scat2 = spectrum_ax.scatter(eig_real[2 * defect_indices], eig_imag[2 * defect_indices], c='red', s=25, zorder=2)
        scat3 = spectrum_ax.scatter(eig_real[2 * defect_indices + 1], eig_imag[2 * defect_indices + 1], c='red', s=25, zorder=2)

    xmax, ymax = np.round(np.max(eig_real), 1), np.round(np.max(eig_imag), 1)
    xticks = [-xmax, 0.0, xmax]
    yticks = [-ymax, 0.0, ymax]
    spectrum_ax.set_xticks(xticks)
    spectrum_ax.set_yticks(yticks)
    spectrum_ax.set_xticklabels(xticks, fontsize=12)
    spectrum_ax.set_yticklabels(yticks, fontsize=12)
    spectrum_ax.set_xlabel("$\\Re(E)$", fontsize=16)
    spectrum_ax.set_ylabel("$\\Im(E)$", fontsize=16, rotation=0)

    spectrum_ax.set_title(f"Eigenvalue spectra in the complex plane", fontsize=16)
    return spectrum_ax


def plot_nh_figure(fig:plt.Figure, eigval_ax:plt.Axes, data_dictionary:dict, Lattice:DefectLattice):
    eigenvalues, L_over_R = data_dictionary.values()
    lattice, defect_indices = Lattice.lattice, Lattice.defect_indices

    box = eigval_ax.get_position()
    zx = box.width / 15
    zy = box.width / 15
    n = 3
    eigvec_ax = fig.add_axes([box.x0 + box.width * (1 - 1/n) - zx, box.y0 + zy, box.width / n, box.height / n])

    eigval_ax = plot_complex_spectrum(eigval_ax, eigenvalues, defect_indices)
    eigvec_ax, colorbar_ax = plot_on_lattice(fig, eigvec_ax, lattice, L_over_R, "scatter" if Lattice.defect_type in ["interstital", "frenkel_pair"] else "imshow")
    return fig, eigval_ax, eigvec_ax, colorbar_ax


def plot_comparison_of_regimes(Lattice:DefectLattice, h_vector, m0_values:np.ndarray, msub_values:np.ndarray = [], resolution_scale:int = 6):
    # If msub is not applicable, using m0 as columns and only one row.
    # Otherwise, use m0 as rows and msub as columns

    m0_array = np.empty(1)
    msub_array = np.empty(1)

    if len(msub_values) <= 1:
        if msub_values == []:
            msub_values = [None]
        n_rows = 1
        n_cols = len(m0_values)
        m0_array = np.array(m0_values)[np.newaxis, :]
        msub_array = np.repeat(np.array(msub_values)[np.newaxis, :], n_cols, axis=1)
    else:
        n_rows = len(m0_values)
        n_cols = len(msub_values)
        m0_array = np.repeat(np.flipud(np.array(m0_values)[:, np.newaxis]), n_cols, axis=1)
        msub_array = np.repeat(np.array(msub_values)[np.newaxis, :], n_rows, axis=0)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(resolution_scale * n_cols, resolution_scale * n_rows - 2))
    if n_rows == 1:
        axs = axs.reshape(1, n_cols)
    for j in range(n_cols):
        for i in range(n_rows):
            m0 = m0_array[i, j]
            msub = msub_array[i, j]
            dd = compute_eigenvectors_eigenvalues(Lattice, m0, h_vector, msub)
            fig, axs[i, j], eigvec_ax, cbar_ax = plot_nh_figure(fig, axs[i, j], dd, Lattice)
            axs[i, j].set_title("")
            axs[i, j].annotate(
                f"$m_0={m0}$\n$m_0^{{\\text{{sub}}}}={msub}$",
                xy = (0.025, 0.95),
                xycoords = 'axes fraction',
                ha = 'left',
                va = 'top',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.0)
            )

            if i != n_rows - 1:
                axs[i, j].set_xlabel("")
            if j != 0:
                axs[i, j].set_ylabel("")
            if i + j == 3 and Lattice.defect_type in ["substitution", "interstitial", "frenkel_pair"]:
                axs[i, j].remove()
                eigvec_ax.remove()
                cbar_ax.remove()

    fig.suptitle(f"Non-Hermitian Skin Effect : {Lattice.defect_type.capitalize()} Defect : $\\vec{{h}}=({h_vector[0]}, {h_vector[1]}, {h_vector[2]})$", fontsize=20,)
# endregion

@profile
def main():
    Lattice = DefectLattice(31, 31, "none", False)
    plot_comparison_of_regimes(Lattice, [0.25, 0.0, 0.0], [-2.5, -1.0, 1.0, 2.5])
    #plt.savefig('temp.png')
    plt.savefig(f'{Lattice.defect_type}.png')


if __name__ == "__main__":
    main()
"""
Geometrically trivial defects in non-Hermitian Chern insulators
"""

import numpy as np
import scipy.linalg as spla
from scipy.sparse import dok_matrix
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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


class DefectLattice:
    def __init__(self, Lx:int, Ly:int, defect_type:str, pbc:bool, defect_radius:int=1,
                 schottky_separation:int=0, schottky_n_pairs:int=1, 
                 frenkel_x_disp:float = -1.5, frenkel_y_disp:float = -0.5,
                 break_c4:bool = False): 
        assert Lx % 2 == 0
        assert Ly % 2 == 0
        self._defect_type = defect_type
        self._pbc = pbc
        self._Lx = Lx
        self._Ly = Ly
        match defect_type:
            case 'none':
                self._lattice, self._defect_indices = self.generate_square_lattice(Lx, Ly)
            case 'vacancy':
                self._lattice, self._defect_indices = self.generate_vacancy_lattice(Lx, Ly, defect_radius)
            case 'schottky':
                self._lattice, self._defect_indices = self.generate_schottky_lattice(Lx, Ly, schottky_separation, schottky_n_pairs)
            case 'substitution':
                self._lattice, self._defect_indices = self.generate_substitution_lattice(Lx, Ly, defect_radius, break_c4)
            case 'interstitial':
                self._lattice, self._defect_indices = self.generate_interstitial_lattice(Lx, Ly, defect_radius, break_c4)
            case 'frenkel_pair':
                self._lattice, self._defect_indices = self.generate_frenkel_pair_lattice(Lx, Ly, frenkel_x_disp, frenkel_y_disp)
                self._fp_xdisp = frenkel_x_disp
                self._fp_ydisp = frenkel_y_disp
            case _:
                raise ValueError('Defect type not properly provided')

        self._defect_positions = np.array([[(x, y) for y, x in zip(*np.where(self.lattice == idx))] for idx in self.defect_indices])[:, 0, :].T

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

    # region Properties
    @property
    def lattice(self): return self._lattice
    @property
    def defect_indices(self): return self._defect_indices
    @property
    def defect_positions(self): return self._defect_positions
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
    # endregion


    def generate_square_lattice(self, Lx:int, Ly:int):
        return np.arange(Lx*Ly).reshape((Ly, Lx)), []


    def generate_vacancy_lattice(self, Lx:int, Ly:int, vacancy_radius:int=1):
        #assert (Lx*Ly % 2 == 1), "Side lengths must be odd"
        assert vacancy_radius > 0, "Defect radius must be positive definite"
        assert vacancy_radius <= (min(Lx, Ly) // 2 + 1), "Defect must fit inside the lattice."

        lattice, _ = self.generate_square_lattice(Lx, Ly)
        defect_indices = []
        vacancy_index = -1
        for i in range(-vacancy_radius, vacancy_radius):
            for j in range(-vacancy_radius, vacancy_radius):
                if abs(i) + abs(j) < vacancy_radius:
                    lattice[Ly // 2 + i, Lx // 2 + j] = vacancy_index
                    defect_indices.append(vacancy_index)
                    vacancy_index -= 1

        return lattice, defect_indices


    def generate_schottky_lattice(self, Lx:int, Ly:int, separation:int, n_pairs:int = 1):
        assert ((Lx + separation) % 2 == 1) and ((Ly + separation) % 2 == 1), "Separation must be odd for even side lengths, and even for odd side lengths"
        assert n_pairs in [1, 2], "Number of pairs must be either 1 or 2"
        assert (separation <= Lx) and (separation <= Ly), "Separation must be less than the side lengths"
        assert separation > 0, "Separation must be positive definite"
        if separation is None:
            separation = Lx % 2 + 1
        
        lattice, _ = self.generate_square_lattice(Lx, Ly)
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


    def generate_substitution_lattice(self, Lx:int, Ly:int, substitution_radius:int = 1, break_c4:bool = False):
        #assert (Lx * Ly % 2 == 1), "Side lengths must be odd"
        assert substitution_radius > 0, "Defect radius must be positive definite"
        assert substitution_radius <= (min(Lx, Ly) // 2 + 1), "Defect must fit inside the lattice."

        lattice, _ = self.generate_square_lattice(Lx, Ly)
        defect_indices = []
        for i in range(-substitution_radius, substitution_radius):
            for j in range(-substitution_radius, substitution_radius):
                if abs(i) + abs(j) < substitution_radius:
                    defect_indices.append(lattice[Ly // 2 + i, Lx // 2 + j])

        if break_c4:
            defect_indices.append(lattice[Ly // 2 + substitution_radius // 2, Lx // 2 + substitution_radius // 2])
            defect_indices.append(lattice[Ly // 2 - substitution_radius // 2, Lx // 2 - substitution_radius // 2])

        return lattice, np.unique(defect_indices)


    def generate_interstitial_lattice(self, Lx:int, Ly:int, interstitial_radius:int=1, break_c4:bool = False):
        assert (Lx % 2 == 0) and (Ly % 2 == 0), "Side lengths must be even"
        assert interstitial_radius > 0, "Defect radius must be positive definite"
        assert interstitial_radius <= (min(Lx, Ly) // 2 + 1), "Defect must fit inside the lattice."
        if break_c4:
            assert interstitial_radius % 2 == 0, f"If break_c4 is True then interstital_radius must be even. Values are `{interstitial_radius}` and `{break_c4}`"
        
        lattice, _ = self.generate_square_lattice(Lx, Ly)
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


        if break_c4:
            y1 = int(Ly * scale / 2 - (displacement * scale) + scale * interstitial_radius / 2)
            x1 = int(Lx * scale / 2 - (displacement * scale) + scale * interstitial_radius / 2)
            y2 = int(Ly * scale / 2 - (displacement * scale) - scale * interstitial_radius / 2)
            x2 = int(Lx * scale / 2 - (displacement * scale) - scale * interstitial_radius / 2)
            Y_pos.append(y1)
            Y_pos.append(y2)
            X_pos.append(x1)
            X_pos.append(x2)
            large_lattice[y1, x1] = np.inf
            large_lattice[y2, x2] = np.inf


        large_lattice[np.where(large_lattice >= 0)] = np.arange(len(np.where(large_lattice >= 0)[0].flatten()))
        defect_indices = list(large_lattice[Y_pos, X_pos].astype(int))
        return large_lattice, defect_indices


    def generate_frenkel_pair_lattice(self, Lx:int, Ly:int, x_disp:float, y_disp:float):
        #assert (Lx * Ly % 2 == 1), "Side lengths must be odd"
        assert ((x_disp % 1 == 0.5) and (y_disp % 1 == 0.5)), "Displacements must be odd half integer"
        assert (abs(x_disp) < Lx / 2) and (abs(y_disp) < Ly / 2), "Interstitial displacement must be within the lattice"

        # Convert x_disp, y_disp to the doubled lattice linear length
        x_disp = int(2 * x_disp)
        y_disp = int(2 * y_disp)


        lattice, _ = self.generate_square_lattice(Lx, Ly)
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


    def plot(self, ax:Axes|None = None) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6,6))

        ax.scatter(self.X, self.Y, c='k', s=100)
        ax.scatter(self.X[self.defect_indices], self.Y[self.defect_indices], c='r', s=100)
        return ax


def compute_hamiltonian(Lattice:DefectLattice, m0:float, h_vector:"np.ndarray|tuple", t:float, t0:float, hsub_vector:"np.ndarray|tuple|None" = None):
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
        if hsub_vector is None and Lattice.defect_type in ['substitution', 'interstitial', 'frenkel_pair']: 
            raise ValueError(f"`hsub` cannot be None when defect_indices are provided for 'substitution', 'interstitial', 'frenkel_pair'")
        for idx in defect_indices:
            if (idx >= 0) and (Lattice.defect_type != "schottky") and isinstance(hsub_vector, np.ndarray):
                hx_matrix[idx, idx] = 1.0j * hsub_vector[0] 
                hy_matrix[idx, idx] = 1.0j * hsub_vector[1] 
                hz_matrix[idx, idx] = 1.0j * hsub_vector[2] 

    onsite_mass = m0 * I
    dx = t * Sx + hx_matrix
    dy = t * Sy + hy_matrix
    dz = (hz_matrix + onsite_mass) + t0 * Cx_plus_Cy

    hamiltonian = np.kron(dx, pauli_x) + np.kron(dy, pauli_y) + np.kron(dz, pauli_z)

    if Lattice.defect_type == "schottky":
        mask = np.full(hamiltonian.shape[0], True)
        for i, idx in enumerate(defect_indices): 
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


def compute_ipr(eigenvectors:np.ndarray):
    eigvec_sq = np.abs(eigenvectors) ** 2
    eigvec_sq = eigvec_sq[eigenvectors.shape[0] // 2:, :] + eigvec_sq[:eigenvectors.shape[0] // 2, :]
    IPR = np.sum(eigvec_sq ** 2, axis=0)
    return IPR


def get_close_to_zero_idxs(eigenvectors, n_idxs):
    ipr = compute_ipr(eigenvectors)
    sort = np.argsort(ipr)
    idxs  = sort[-n_idxs:]
    return idxs


def compute_eigenvectors_eigenvalues(Lattice:DefectLattice, m0:float, 
                                     h0_vector:"np.ndarray|tuple", hsub_vector:"np.ndarray|tuple|None" = None, 
                                     n_closest_to_zero:int = 2) -> dict[str, np.ndarray]:
    if n_closest_to_zero is not None:
        assert (n_closest_to_zero <= len(Lattice.X) * 2), "Number of selected indices must be <= number of indices"
    
    hamiltonian = compute_hamiltonian(Lattice, m0, h0_vector, 1.0, 1.0, hsub_vector)
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True) 
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

    left_ipr = compute_ipr(left_eigenvectors)
    right_ipr = compute_ipr(right_eigenvectors)
    ipr = (left_ipr + right_ipr) / 2
    ipr_sorted_idxs = np.argsort(left_ipr)
    close_to_zero_idxs = ipr_sorted_idxs[-n_closest_to_zero:]

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
        "eigenvalues" : eigenvalues, 
        "L" : sum_over_orbitals(L), 
        "R" : sum_over_orbitals(R), 
        "selected_idxs" : close_to_zero_idxs,
        "selected_left_eigenvectors" : sum_over_orbitals(close_left),
        "selected_right_eigenvectors" : sum_over_orbitals(close_right),
        "left_eigenvectors" : left_eigenvectors,
        "right_eigenvectors" : right_eigenvectors,
        "left_ipr" : left_ipr,
        "right_ipr": right_ipr,
        "average_ipr": ipr,
        "left_eigenvectors": left_eigenvectors,
        "right_eigenvectors": right_eigenvectors
        }
    return data_dictionary



def main():
    pass


if __name__ == "__main__":
    main()




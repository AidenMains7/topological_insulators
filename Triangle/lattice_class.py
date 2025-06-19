import numpy as np
import scipy.linalg as spla

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from itertools import product
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
import os, h5py
from time import time


class Lattice:
    def __init__(self, pbc:bool, lattice_generating_function, distance_generating_function, hopping_generating_function, *args, **kwargs):
        self._pbc = pbc
        self._lattice = lattice_generating_function(*args, **kwargs)
        self._Y, self._X = np.where(self._lattice >= 0) if self._lattice is not None else (None, None)
        self._dx, self._dy = distance_generating_function(*args, **kwargs)
        self._NN, self._NNN = hopping_generating_function(*args, **kwargs)

        pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)
        self._pauli_matrices = [pauli1, pauli2, pauli3]

    def __str__(self):
        return f"Lattice with size {self.lattice.shape} and {self.number_of_sites} sites. PBC is {self.pbc}."
    
    @property
    def info(self):
        print(f"Lattice size: {self.lattice.shape}")
        print(f"Number of sites: {self.number_of_sites}")
        print(f"Periodic boundary conditions: {self.pbc}")
        
    @property
    def lattice(self):
        return self._lattice
    
    @property
    def X(self):
        return self._X
    
    @property 
    def Y(self):
        return self._Y
    
    @property
    def number_of_sites(self):
        return self.X.size
    
    @property
    def lattice_size(self):
        return self.lattice.size

    @property
    def dx(self):
        return self._dx
    
    @property
    def dy(self):
        return self._dy

    @property
    def NN(self):
        return self._NN

    @property
    def NNN(self):
        return self._NNN
    
    @property
    def pauli_matrices(self):
        return self._pauli_matrices

    @property
    def pbc(self):
        return self._pbc

    def generate_lattice(self, **kwargs):
        raise NotImplementedError("This method should be implemented in a child class.")
    
    def compute_distances(self):
        raise NotImplementedError("This method should be implemented in a child class.")
    
    def compute_NN_and_NNN(self):
        raise NotImplementedError("This method should be implemented in a child class.")
    
    def plot(self, ax:plt.Axes=None, title:str = None, plotNN:bool = True, plotNNN:bool = False, plotIdxs:bool = False, scaling:tuple = (1/2, 1/2/np.sqrt(3)), plotSites:bool = True, site_color:str = "black", NN_color:str = "black", NNN_color:str = "black", *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if scaling is not None:
            X = self.X * scaling[0]
            Y = self.Y * scaling[1]
        else:
            X = self.X
            Y = self.Y

        if plotSites:
            ax.scatter(X, Y, c=site_color, zorder=1, s=25)

        if plotNN: 
            if isinstance(self.NN, dict):
                NN_sum = np.sum(list(self.NN.values()), axis=0)
                i_idx, j_idx = np.where(NN_sum)
                valid_indices = (i_idx < len(X)) & (j_idx < len(X))
                i_idx, j_idx = i_idx[valid_indices], j_idx[valid_indices]
                ax.plot([X[i_idx], X[j_idx]], [Y[i_idx], Y[j_idx]], c = NN_color, alpha=1., ls='-', zorder=0)
        if plotNNN:
            if isinstance(self.NNN, dict):
                colors = ["blue", "orange", "red"]
                for arr, color in zip(self.NNN.values(), colors):
                    i_idx, j_idx = np.where(arr)
                    valid_indices = (i_idx < len(X)) & (j_idx < len(X))
                    i_idx, j_idx = i_idx[valid_indices], j_idx[valid_indices]
                    ax.plot([X[i_idx], X[j_idx]], [Y[i_idx], Y[j_idx]], c=NNN_color, alpha=1., ls='--', zorder=0)

        if plotIdxs:
            for y, x in zip(Y, X):
                ax.text(x, y, f"{self.lattice[y, x]}", color="red", fontsize=12, ha="center", va="center")

        title = title if title else f"Lattice with {self.number_of_sites} sites"
        ax.set_title(title)
        plt.tight_layout()

    def plot_distances(self, idx:int = None, cmap:str = "viridis", *args, **kwargs):
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))

        if idx is None:
            idx = len(self.X) // 2

        distances = [self.dx, self.dy, np.sqrt(self.dx**2 + self.dy**2)]
        labels = ["dx", "dy", "d"]
        for i, (distance, label) in enumerate(zip(distances, labels)):
            axs[i].set_title(label)
            axs[i].set_xlabel("X")
            axs[i].set_ylabel("Y")
            axs[i].scatter(self.X, self.Y, c=distance[idx], cmap=cmap, zorder=0, s=25)
            axs[i].scatter(self.X[idx], self.Y[idx], c='red', s=50, zorder=1)

        cbar = fig.colorbar(axs[i].collections[0], ax=axs[i], orientation='vertical')
        cbar.set_label("Distance to site {}".format(idx), rotation=270, labelpad=15)

        plt.tight_layout()
        plt.show()

    def compute_hamiltonian(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in a child class.")
    
    def compute_projector(self, hamiltonian, *args, **kwargs):
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2]
        highest_lower_band = lower_band[-1]

        D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
        D_herm_conj = np.einsum('i,ij->ij', D, eigenvectors.conj().T)

        projector = eigenvectors @ D_herm_conj
        return projector

    def compute_bott_index(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in a child class.")



class TriangularLattice(Lattice):
    def __init__(self, pbc, generation):
        self._generation = generation
        super().__init__(pbc, self.generate_lattice_tiled, self.compute_distances, self.compute_hopping, generation=generation)

    def __str__(self):
        return f"TriangularLattice (Generation {self.generation})"

    @property
    def generation(self):
        return self._generation
    
    def generate_lattice_not_tiled(self, *args, **kwargs):
        def recursive_lattice(_gen):
            if _gen == 0:
                return np.array([0.0, -np.sqrt(3)/2, np.sqrt(3)/2, 1.0, -0.5, -0.5]).reshape(2, 3)
            else:
                smaller = recursive_lattice(_gen - 1)
                x_range = np.max(smaller[0]) - np.min(smaller[0])
                y_range = np.max(smaller[1]) - np.min(smaller[1])
                lattice_points = np.zeros((2, smaller.shape[1]*4))

                shifts = {
                    "top": np.array([[0], [y_range]]),
                    "left": np.array([[-x_range/2], [0]]),
                    "right": np.array([[x_range/2], [0]]),
                }

                for i, shift in enumerate(shifts.values()):
                    lattice_points[:, i::4] = smaller + shift
                
                flipped_max_y = np.min((smaller+shifts["left"])[1])
                flipped = smaller.copy()
                flipped[1] *= -1
                flipped[1] -= np.min(flipped[1]) - flipped_max_y

                lattice_points[:, 3::4] = flipped
                return np.unique(lattice_points, axis=1)
            
        coordinates = recursive_lattice(self.generation)

        xmin, ymin = np.min(coordinates[0]), np.min(coordinates[1])
        coordinates[0] -= xmin
        coordinates[1] -= ymin
        coordinates[0] *= 2/np.sqrt(3)
        coordinates[1] *= 2
        coordinates = np.unique(np.round(coordinates).astype(int), axis=1)
        sorted_idxs = np.lexsort((coordinates[0], coordinates[1]))
        coordinates = coordinates[:, sorted_idxs]
        
        lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
        lattice[coordinates[1], coordinates[0]] = np.arange(coordinates.shape[1])
        return lattice

    def generate_lattice_tiled(self, doHex:bool = False, *args, **kwargs):
        lattice = self.generate_lattice_not_tiled()
        Y, X = np.where(lattice >= 0)[:]
        upright_coordinates = np.array([X, Y])
        flipped_coordinates = upright_coordinates.copy()

        flipped_coordinates[1] *= -1
        flipped_coordinates[1] -= np.min(flipped_coordinates[1])
        flipped_coordinates[0] -= np.min(flipped_coordinates[0])

        if doHex:
            upright_shifts = [np.array([0, 0]).T, np.array([-np.max(X), 0]).T, np.array([-np.max(X)//2, -np.max(Y)]).T]
            flipped_shifts = [np.array([-np.max(X)//2, 0]).T, np.array([0, -np.max(Y)]).T, np.array([-np.max(X), -np.max(Y)]).T]
        else:
            upright_shifts = [np.array([0, 0]).T]
            flipped_shifts = [np.array([-np.max(X)//2, 0]).T]

        hexagon_coordinates = []
        for shift in upright_shifts:
            hexagon_coordinates.append(upright_coordinates + shift.reshape(2, 1))
        for shift in flipped_shifts:
            hexagon_coordinates.append(flipped_coordinates + shift.reshape(2, 1))

        hexagon_coordinates = np.concatenate(hexagon_coordinates, axis=1)
        hexagon_coordinates = np.unique(hexagon_coordinates, axis=1)

        hexagon_coordinates[0] -= np.min(hexagon_coordinates[0])
        hexagon_coordinates[1] -= np.min(hexagon_coordinates[1])
        sorted_idxs = np.lexsort((hexagon_coordinates[0], hexagon_coordinates[1]))
        hexagon_coordinates = hexagon_coordinates[:, sorted_idxs]
        lattice = np.full((np.max(hexagon_coordinates[1])+1, np.max(hexagon_coordinates[0])+1), -1)
        lattice[hexagon_coordinates[1], hexagon_coordinates[0]] = np.arange(hexagon_coordinates.shape[1])
        return lattice

    def compute_distances(self, printPBC:bool = False, *args, **kwargs):
        dx = self.X - self.X[:, np.newaxis]
        dy = self.Y - self.Y[:, np.newaxis]
        if not self.pbc:
            return dx, dy
        else:
            x_range = np.max(self.X) - np.min(self.X)
            y_range = np.max(self.Y) - np.min(self.Y)

            parallelogram_displacements = {
                "center": np.array([0, 0]).T,
                "top_left": np.array([-x_range / 3 - 1, y_range + 3]).T,
                "left": np.array([-x_range * 2 / 3 - 2, 0]).T,
                "bottom_left": np.array([-x_range - 3, -y_range - 3]).T,
                "bottom": np.array([-x_range/3 - 1, -y_range - 3]).T,
                "bottom_right": np.array([x_range / 3 + 1, -y_range - 3]).T,
                "right": np.array([x_range* 2 / 3 + 2, 0]).T,
                "top_right": np.array([x_range + 3, y_range + 3]).T,
                "top": np.array([x_range / 3 + 1, y_range + 3]).T
            }
            hexagon_displacements = {
                "center": np.array([0, 0]).T,
                "top": np.array([1, y_range+3]).T,
                "top_left": np.array([-x_range * 3 / 4 - 1, y_range / 2 + 3]).T,
                "bottom_left": np.array([-x_range * 3 / 4 - 2, -y_range / 2]).T,
                "bottom": np.array([-1, -y_range - 3]).T,
                "bottom_right": np.array([x_range * 3 / 4 + 1, -y_range / 2 - 3]).T,
                "top_right": np.array([x_range * 3 / 4 + 2, y_range / 2]).T,
            }

            x_shifted = np.empty((dx.shape[0], dx.shape[1], len(hexagon_displacements)), dtype=dx.dtype)
            y_shifted = np.empty((dy.shape[0], dy.shape[1], len(hexagon_displacements)), dtype=dy.dtype)
            for i, shift in enumerate(hexagon_displacements.values()):
                x_shifted[:, :, i] = dx + shift[0]
                y_shifted[:, :, i] = dy + shift[1]

            if printPBC:
                for i, shift in enumerate(hexagon_displacements.values()):
                    x = self.X + shift[0]
                    y = self.Y + shift[1]
                    plt.scatter(x, y, zorder=0, alpha=0.5, label=list(hexagon_displacements.keys())[i])
                plt.legend(loc="upper left")
                plt.show()

            distances = x_shifted**2 + y_shifted**2
            minimal_hop = np.argmin(distances, axis=-1)
            i_idxs, j_idxs = np.indices(minimal_hop.shape)
            
            dx = x_shifted[i_idxs, j_idxs, minimal_hop]
            dy = y_shifted[i_idxs, j_idxs, minimal_hop]
        return dx, dy

    def compute_hopping(self, *args, **kwargs):
        # For NN
        # In real space:
        # b1: (1, 0)
        # b2: (1 / 2, sqrt(3) / 2)
        # b2_tilde: (1 / 2, -sqrt(3) / 2)
        # In integer:
        # b1: (2, 0)
        # b2: (1, 3)
        # b2_tilde: (1, -3)
        b1 = (self.dx == 2) & (self.dy == 0)
        b2 = (self.dx == 1) & (self.dy == 3)
        b2tilde = (self.dx == 1) & (self.dy == -3)

        # For NNN
        # In real space:
        # c1: (0,    sqrt(3))
        # c2: (3 / 2,  sqrt(3) / 2)
        # c3: (3 / 2, -sqrt(3) / 2)
        # In integer:
        # c1: (0, 6)
        # c2: (3, 3)
        # c3: (3, -3)
        c1 = (self.dx == 0) & (self.dy == 6)
        c2 = (self.dx == 3) & (self.dy == 3)
        c3 = (self.dx == 3) & (self.dy == -3)
        b1, b2, b2tilde, c1, c2, c3 = map(lambda x: x.astype(np.complex128), (b1, b2, b2tilde, c1, c2, c3))
        return {"b1": b1, "b2": b2, "b2tilde": b2tilde}, {"c1": c1, "c2": c2, "c3": c3}

    def compute_hamiltonian(self, M:float, B_tilde:float, B:float, t:float, A_tilde:float, NN:np.ndarray = None, NNN:np.ndarray = None, *args, **kwargs):
        if NN is None:
            NN = self.NN
        if NNN is None:
            NNN = self.NNN

        b1, b2, b2tilde = NN["b1"], NN["b2"], NN["b2tilde"]
        c1, c2, c3 = NNN["c1"], NNN["c2"], NNN["c3"]

        I = np.eye(b1.shape[0], dtype=np.complex128)

        d1 = (t / 2j) * (b1) + (t / 4j) * (b2 + b2tilde)
        d1 += d1.conj().T

        d2 = (-t * np.sqrt(3) / 4j) * (b2 - b2tilde)
        d2 += d2.conj().T

        d3 = (B) * (b1 + b2 + b2tilde)
        d3 += d3.conj().T
        d3 += (M - 4 * B) * (I)

        dtilde1 = (np.sqrt(3) * A_tilde / 8j) * (c2 + c3)
        dtilde1 += dtilde1.conj().T

        dtilde2 = (-A_tilde / 2j) * (c1) + (A_tilde / 4j) * (c2 - c3)
        dtilde2 += dtilde2.conj().T

        dtilde3 = (B_tilde) * (c1 + c2 + c3)
        dtilde3 += dtilde3.conj().T
        dtilde3 += (-6 * B_tilde) * (I)

        pauli_x, pauli_y, pauli_z = self.pauli_matrices
        H_from_NN = np.kron(d1, pauli_x) + np.kron(d2, pauli_y) + np.kron(d3, pauli_z)
        H_from_NNN = np.kron(dtilde1, pauli_x) + np.kron(dtilde2, pauli_y) + np.kron(dtilde3, pauli_z)

        return H_from_NN + H_from_NNN
    
    def compute_bott_index(self, projector:np.ndarray, X:np.ndarray = None, Y:np.ndarray = None, *args, **kwargs):
        def convert_from_discrete_to_triangular_basis(x, y):
            # First, convert to real space
            x = x.astype(float) / 2
            y = y.astype(float) / 3 * (np.sqrt(3) / 2)

            # Then, with basis vectors (a=1)
            # a1 = (1, 0)
            # a2 = (1/2, sqrt(3)/2)
            # We get:
            a1 = x - y / np.sqrt(3)
            a2 = y * 2 / np.sqrt(3)
            return a1, a2
    
        if X is None or Y is None:
            X = np.repeat(self.X, 2)
            Y = np.repeat(self.Y, 2)
        else:
            X = np.repeat(X, 2)
            Y = np.repeat(Y, 2)

        X, Y = convert_from_discrete_to_triangular_basis(X, Y)

        Lx = np.max(X) - np.min(X)
        Ly = np.max(Y) - np.min(Y)

        x_unitary = np.exp(1j * 2 * np.pi * X / Lx)
        y_unitary = np.exp(1j * 2 * np.pi * Y / Ly)

        x_unitary_proj = np.einsum('i,ij->ij', x_unitary, projector)
        y_unitary_proj = np.einsum('i,ij->ij', y_unitary, projector)
        x_unitary_dagger_proj = np.einsum('i,ij->ij', x_unitary.conj(), projector)
        y_unitary_dagger_proj = np.einsum('i,ij->ij', y_unitary.conj(), projector)

        I = np.eye(projector.shape[0], dtype=np.complex128)
        A = I - projector + projector @ x_unitary_proj @ y_unitary_proj @ x_unitary_dagger_proj @ y_unitary_dagger_proj

        bott_index = round(np.imag(np.sum(np.log(spla.eigvals(A)))) / (2 * np.pi))
        return bott_index



class SierpinskiTriangularLattice(TriangularLattice):
    def __init__(self, pbc, generation):
        self._ParentLattice = TriangularLattice(pbc, generation)
        super().__init__(pbc, generation)

        self._NN, self._NNN = self.compute_bond_elimination(self.lattice, self.NN, self.NNN)
        self._site_mask = ( (self.lattice + 1).astype(bool) )[self.ParentLattice.Y, self.ParentLattice.X]

    def __str__(self):
        return f"SierpinskiTriangleLattice (Generation {self.generation})" 

    @property
    def site_mask(self):
        return self._site_mask

    @property
    def ParentLattice(self):
        return self._ParentLattice

    def generate_lattice_not_tiled(self, *args, **kwargs):
        def recursive_fractal(_gen):
            if _gen == 0:
                return {"lattice_points": np.array([0.0, -np.sqrt(3)/2, np.sqrt(3)/2, 1.0, -0.5, -0.5]).reshape(2, 3), "triangular_hole_locations": None}
            else:
                fractal_dict = recursive_fractal(_gen - 1)
                smaller = fractal_dict["lattice_points"]

                x_range = np.max(smaller[0]) - np.min(smaller[0])
                y_range = np.max(smaller[1]) - np.min(smaller[1])
                fractal = np.zeros((2, smaller.shape[1]*3))

                shifts = {
                    "top": np.array([[0], [y_range]]),
                    "left": np.array([[-x_range/2], [0]]),
                    "right": np.array([[x_range/2], [0]]),
                }

                for i, shift in enumerate(shifts.values()):
                    fractal[:, i::3] = smaller + shift

                # Hole size, x location, y location
                location_of_hole = np.array([np.mean(fractal[0]), np.min(fractal[1]), _gen-1]).T

                if fractal_dict["triangular_hole_locations"] is not None:
                    smaller_triangular_hole_locations = fractal_dict["triangular_hole_locations"]
                    new_hole_locations = np.zeros((3, smaller_triangular_hole_locations.shape[1]*3))
                    for i, shift in enumerate(shifts.values()):
                        new_hole_locations[2, i::3] = smaller_triangular_hole_locations[2] # hole size
                        new_hole_locations[[0,1], i::3] = smaller_triangular_hole_locations[[0,1]] + shift # x pos
                    
                    all_hole_points = np.append(new_hole_locations, location_of_hole.reshape(3, 1), axis=1)
                else:
                    all_hole_points = location_of_hole.reshape(3, 1)

                return {"lattice_points": np.unique(fractal, axis=1), "triangular_hole_locations": np.unique(all_hole_points, axis=1)}
        fractal_dict = recursive_fractal(self.generation)
        coordinates = fractal_dict["lattice_points"]
        
        xmin, ymin = np.min(coordinates[0]), np.min(coordinates[1])
        coordinates[0] -= xmin
        coordinates[1] -= ymin
        coordinates[0] *= 2/np.sqrt(3)
        coordinates[1] *= 2
        coordinates = np.unique(np.round(coordinates).astype(int), axis=1)
        sorted_idxs = np.lexsort((coordinates[0], coordinates[1]))
        coordinates = coordinates[:, sorted_idxs]
        
        lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
        lattice[coordinates[1], coordinates[0]] = np.arange(coordinates.shape[1])

        if fractal_dict["triangular_hole_locations"] is None:
            return {"lattice": lattice, "triangular_hole_locations": None}
        
        hole_locations = fractal_dict["triangular_hole_locations"]
        hole_locations[0] = (hole_locations[0] - xmin) * 2/np.sqrt(3)
        hole_locations[1] = (hole_locations[1] - ymin) * 2

        self.hole_locations = np.round(hole_locations).astype(int)
        return lattice

    def generate_lattice_tiled(self, doHex:bool = False, *args, **kwargs):
        lattice = self.generate_lattice_not_tiled()
        Y, X = np.where(lattice >= 0)[:]
        upright_coordinates = np.array([X, Y])
        flipped_coordinates = upright_coordinates.copy()

        flipped_hole_locations = self.hole_locations.copy()
        flipped_hole_locations[1] *= -1
        flipped_hole_locations[1] -= np.min(flipped_hole_locations[1]) - 6
        flipped_hole_locations[0] -= np.min(flipped_hole_locations[0]) - np.min(self.hole_locations[0])
        flipped_hole_locations[2] *= -1

        flipped_coordinates[1] *= -1
        flipped_coordinates[1] -= np.min(flipped_coordinates[1])
        flipped_coordinates[0] -= np.min(flipped_coordinates[0])

        if doHex:
            upright_shifts = [np.array([0, 0]).T, np.array([-np.max(X), 0]).T, np.array([-np.max(X)//2, -np.max(Y)]).T]
            flipped_shifts = [np.array([-np.max(X)//2, 0]).T, np.array([0, -np.max(Y)]).T, np.array([-np.max(X), -np.max(Y)]).T]
        else:
            upright_shifts = [np.array([0, 0]).T]
            flipped_shifts = [np.array([-np.max(X)//2, 0]).T]

        hexagon_coordinates = []
        for shift in upright_shifts:
            hexagon_coordinates.append(upright_coordinates + shift.reshape(2, 1))
        for shift in flipped_shifts:
            hexagon_coordinates.append(flipped_coordinates + shift.reshape(2, 1))

        hexagon_coordinates = np.concatenate(hexagon_coordinates, axis=1)
        hexagon_coordinates = np.unique(hexagon_coordinates, axis=1)

        hexagon_hole_locations = []
        for shift in upright_shifts:
            shift = np.array([shift[0], shift[1], 0])
            hexagon_hole_locations.append(self.hole_locations + shift.reshape(3, 1))
        for shift in flipped_shifts:
            shift = np.array([shift[0], shift[1], 0])
            hexagon_hole_locations.append(flipped_hole_locations + shift.reshape(3, 1))
            
        hexagon_hole_locations = np.concatenate(hexagon_hole_locations, axis=1)
        hexagon_hole_locations = np.unique(hexagon_hole_locations, axis=1)
        hexagon_hole_locations[0] -= np.min(hexagon_coordinates[0])
        hexagon_hole_locations[1] -= np.min(hexagon_coordinates[1])
        self.hole_locations = hexagon_hole_locations
        
        hexagon_coordinates[0] -= np.min(hexagon_coordinates[0])
        hexagon_coordinates[1] -= np.min(hexagon_coordinates[1])
        sorted_idxs = np.lexsort((hexagon_coordinates[0], hexagon_coordinates[1]))
        hexagon_coordinates = hexagon_coordinates[:, sorted_idxs]
        lattice = np.full((np.max(hexagon_coordinates[1])+1, np.max(hexagon_coordinates[0])+1), -1)
        lattice[hexagon_coordinates[1], hexagon_coordinates[0]] = np.arange(hexagon_coordinates.shape[1])
        return lattice

    def compute_bond_elimination(self, lattice, NN, NNN, *args, **kwargs):
        def get_side_length(n):
            if n == 0:
                return 2
            elif n < 0:
                raise ValueError("n must be a non-negative integer")
            else:
                return 2*get_side_length(n-1) - 1
            
        hole_x, hole_y, hole_n = self.hole_locations
        remove_n0_holes_mask = (hole_n != 0)
        hole_x, hole_y, hole_n = hole_x[remove_n0_holes_mask], hole_y[remove_n0_holes_mask], hole_n[remove_n0_holes_mask]
        moving_vector_right = np.array([1, 3]).T
        moving_vector_left = np.array([-1, 3]).T
        hole_coordinates = np.vstack((hole_x, hole_y)).T

        NN_bonds_to_remove = []
        NNN_bonds_to_remove = []

        b1_vector = np.array([2, 0])
        b2_vector = np.array([1, 3])
        b2tilde_vector = np.array([1, -3])

        c1_vector = np.array([0, 6])
        c2_vector = np.array([3, 3])
        c3_vector = np.array([3, -3])

        for pos, n in zip(hole_coordinates, hole_n):
            # n is the relative generation size of the hole. Positive n means the hole is an inverted triangle, negative n means the hole is an upright triangle.
            b1_break_pos = pos + np.sign(n) * moving_vector_left
            b1_end_pos = b1_break_pos + np.sign(n) * b1_vector
            b2_break_pos = pos + (get_side_length(abs(n))-2) * np.sign(n) * moving_vector_left
            b2_end_pos = b2_break_pos + np.sign(n) * b2_vector
            b2tilde_break_pos = pos + (get_side_length(abs(n))-2) * np.sign(n) * moving_vector_right
            b2tilde_end_pos = b2tilde_break_pos + np.sign(n) * (-b2tilde_vector)

            b1_i = lattice[b1_break_pos[1], b1_break_pos[0]]
            b1_j = lattice[b1_end_pos[1], b1_end_pos[0]]
            b2_i = lattice[b2_break_pos[1], b2_break_pos[0]]
            b2_j = lattice[b2_end_pos[1], b2_end_pos[0]]
            b2tilde_i = lattice[b2tilde_break_pos[1], b2tilde_break_pos[0]]
            b2tilde_j = lattice[b2tilde_end_pos[1], b2tilde_end_pos[0]]

            if b1_i >= 0 and b1_j >= 0:
                NN_bonds_to_remove.append((b1_i, b1_j))
            if b2_i >= 0 and b2_j >= 0:
                NN_bonds_to_remove.append((b2_i, b2_j))
            if b2tilde_i >= 0 and b2tilde_j >= 0:
                NN_bonds_to_remove.append((b2tilde_i, b2tilde_j))

            c1_break_positions = [(pos + (get_side_length(abs(n))-displacement) * np.sign(n) * mv) for mv, displacement in tuple(product([moving_vector_left, moving_vector_right], [2, 3]))]
            c1_end_positions = [c1_break_pos + np.sign(n) * (c1_vector) for c1_break_pos in c1_break_positions]

            c2_mvs = [moving_vector_left, moving_vector_left, moving_vector_right, moving_vector_left, moving_vector_left]
            c2_displacements = [2, 4, get_side_length(abs(n))-1, 3, get_side_length(abs(n)) - 1]
            c2_shifts = [0, 0, 0, np.array([0, 6]).T, 0]
            c2_flip_factors = np.array([1, 1, -1, -1, 1])
            c2_parameters = [(mv, displacement, additional_shifts) for mv, displacement, additional_shifts in zip(c2_mvs, c2_displacements, c2_shifts)]
            c2_break_positions = [(pos + (get_side_length(abs(n)) - displacement) * np.sign(n) * mv + additional_shifts * np.sign(n)) for mv, displacement, additional_shifts in c2_parameters]
            c2_end_positions = [c2_break_pos + np.sign(n) * (c2_vector) * ff for c2_break_pos, ff in zip(c2_break_positions, c2_flip_factors)]

            c3_mvs = [moving_vector_right, moving_vector_right, moving_vector_left, moving_vector_right, moving_vector_right]
            c3_displacements = c2_displacements
            c3_shifts = c2_shifts
            c3_flip_factors = -c2_flip_factors
            c3_parameters = [(mv, displacement, additional_shifts) for mv, displacement, additional_shifts in zip(c3_mvs, c3_displacements, c3_shifts)]
            c3_break_positions = [(pos + (get_side_length(abs(n)) - displacement) * np.sign(n) * mv + additional_shifts * np.sign(n)) for mv, displacement, additional_shifts in c3_parameters]
            c3_end_positions = [c3_break_pos + np.sign(n) * (c3_vector) * ff for c3_break_pos, ff in zip(c3_break_positions, c3_flip_factors)]        #print("n =", n)

            def check_position_list(position_list):
                def _check_position(pos):
                    if 0 <= pos[0] < lattice.shape[1] and 0 <= pos[1] < lattice.shape[0]:
                        return True
                    return False
                return np.array([_check_position(pos) for pos in position_list])

            c2_break_positions = np.array(c2_break_positions)
            c2_end_positions = np.array(c2_end_positions)
            c2_mask = check_position_list(c2_break_positions) & check_position_list(c2_end_positions)
            c2_break_positions = c2_break_positions[c2_mask]
            c2_end_positions = c2_end_positions[c2_mask]

            c3_break_positions = np.array(c3_break_positions)
            c3_end_positions = np.array(c3_end_positions)
            c3_mask = check_position_list(c3_break_positions) & check_position_list(c3_end_positions)
            c3_break_positions = c3_break_positions[c3_mask]
            c3_end_positions = c3_end_positions[c3_mask]

            c1_is = [lattice[c1_break_pos[1], c1_break_pos[0]] for c1_break_pos in c1_break_positions]
            c1_js = [lattice[c1_end_pos[1], c1_end_pos[0]] for c1_end_pos in c1_end_positions]

            c2_is = [lattice[c2_break_pos[1], c2_break_pos[0]] for c2_break_pos in c2_break_positions]
            c2_js = [lattice[c2_end_pos[1], c2_end_pos[0]] for c2_end_pos in c2_end_positions]
            
            c3_is = [lattice[c3_break_pos[1], c3_break_pos[0]] for c3_break_pos in c3_break_positions]
            c3_js = [lattice[c3_end_pos[1], c3_end_pos[0]] for c3_end_pos in c3_end_positions]

            for c1_i, c1_j in zip(c1_is, c1_js):
                if c1_i >= 0 and c1_j >= 0:
                    NNN_bonds_to_remove.append((c1_i, c1_j))
            for c2_i, c2_j in zip(c2_is, c2_js):
                if c2_i >= 0 and c2_j >= 0:
                    NNN_bonds_to_remove.append((c2_i, c2_j))
            for c3_i, c3_j in zip(c3_is, c3_js):
                if c3_i >= 0 and c3_j >= 0:
                    NNN_bonds_to_remove.append((c3_i, c3_j))

        b1_mask, b2_mask, b2tilde_mask = tuple(np.copy(mask) for mask in NN.values())
        c1_mask, c2_mask, c3_mask = tuple(np.copy(mask) for mask in NNN.values())

        for i, j in NN_bonds_to_remove:
            b1_mask[i, j] = 0.0 + 0.0j
            b1_mask[j, i] = 0.0 + 0.0j
            b2_mask[i, j] = 0.0 + 0.0j
            b2_mask[j, i] = 0.0 + 0.0j
            b2tilde_mask[i, j] = 0.0 + 0.0j
            b2tilde_mask[j, i] = 0.0 + 0.0j
        for i, j in NNN_bonds_to_remove:
            c1_mask[i, j] = 0.0 + 0.0j
            c1_mask[j, i] = 0.0 + 0.0j
            c2_mask[i, j] = 0.0 + 0.0j
            c2_mask[j, i] = 0.0 + 0.0j
            c3_mask[i, j] = 0.0 + 0.0j
            c3_mask[j, i] = 0.0 + 0.0j

        return {"b1": b1_mask, "b2": b2_mask, "b2tilde": b2tilde_mask}, {"c1": c1_mask, "c2": c2_mask, "c3": c3_mask}

    def compute_hamiltonian(self, method:str, M:float, B_tilde:float, B:float, t:float, A_tilde:float, *args, **kwargs) -> np.ndarray:
        site_mask = np.repeat(self.site_mask, 2)
        if method == "triangular":
            return self.ParentLattice.compute_hamiltonian(M, B_tilde, B, t, A_tilde, *args, **kwargs)
        else:
            parent_NN_bondelim, parent_NNN_bondelim = self.compute_bond_elimination(self.lattice, self.ParentLattice.NN, self.ParentLattice.NNN)
            triangular_H_with_bond_elim = self.ParentLattice.compute_hamiltonian(M, B_tilde, B, t, A_tilde, parent_NN_bondelim, parent_NNN_bondelim, *args, **kwargs)
            if method == "site_elim":
                H_aa = triangular_H_with_bond_elim[np.ix_(site_mask, site_mask)]
                return H_aa
            elif method == "renorm":
                H_aa = triangular_H_with_bond_elim[np.ix_(site_mask, site_mask)]
                H_ab = triangular_H_with_bond_elim[np.ix_(site_mask, ~site_mask)]
                H_ba = triangular_H_with_bond_elim[np.ix_(~site_mask, site_mask)]
                H_bb = triangular_H_with_bond_elim[np.ix_(~site_mask, ~site_mask)]

                try:
                    H_renorm = H_aa - H_ab @ spla.solve(H_bb, H_ba, assume_a='her', overwrite_a=True, overwrite_b=True, check_finite=True)
                except np.linalg.LinAlgError as e:
                    print(f"Matrix inversion failed at (M = {M:.2f}, B_tilde = {B_tilde:.2f}):", e)
                    H_renorm = np.nan
                
            return H_renorm

    def compute_disorder(self, strength, system_size):
        disorder_array = np.random.uniform(-strength/2, strength/2, size=system_size)
        disorder_array -= np.mean(disorder_array)
        return np.diag(disorder_array).astype(np.complex128)

    def compute_phase_diagram(self, method:str, M_range:tuple = (-2.0, 8.0), B_tilde_range:tuple = (0.0, 0.5), resolution:tuple = None, B:float = 1.0, t:float = 1.0, A_tilde:float = 0.0, doOverwrite:bool = False, directory:str = None, *args, **kwargs):
        sizestr = f"{resolution[0]}x{resolution[1]}" if resolution is not None else "auto"
        fbasename = f"bott_index_g{self.generation}_{method}_{sizestr}_.h5"
        directory = "./Triangle/PhaseDiagrams/Bott/" if directory is None else directory

        if os.path.exists(directory + fbasename) and not doOverwrite:
            print(f"{directory + fbasename} already exists. Set doOverwrite=True to overwrite.")
            return directory + fbasename

        if resolution is None:
            M_values = np.arange(M_range[0], M_range[1], 0.1)
            B_tilde_values = np.arange(B_tilde_range[0], B_tilde_range[1], 0.05)
        else:
            print(M_range)
            M_values = np.linspace(M_range[0], M_range[1], resolution[0])
            B_tilde_values = np.linspace(B_tilde_range[0], B_tilde_range[1], resolution[1])

        parameter_values = tuple(product(M_values, B_tilde_values))

        def worker_function(parameters, *args, **kwargs):
            try:
                M, B_tilde = parameters
                hamiltonian = self.compute_hamiltonian(method=method, M=M, B_tilde=B_tilde, B=B, t=t, A_tilde=A_tilde, *args, **kwargs)
                if np.isnan(hamiltonian).any():
                    return [M, B_tilde, np.nan]
                projector = self.compute_projector(hamiltonian)
                if method != "triangular":
                    bott_index = self.compute_bott_index(projector)
                else:
                    bott_index = self.compute_bott_index(projector, X = self.ParentLattice.X, Y = self.ParentLattice.Y)
                return [M, B_tilde, bott_index]
            except Exception as e:
                print(f"Error at (M = {M:.2f}, B_tilde = {B_tilde:.2f}):", e)
                return [M, B_tilde, np.nan]
        
        data = generic_multiprocessing(worker_function, parameter_values, n_jobs=4, *args, **kwargs)
        M_data, B_tilde_data, bott_data = np.array(data).T
        data_dict = {
            "M_values": M_data,
            "B_tilde_values": B_tilde_data,
            "bott_values": bott_data.reshape(M_values.size, B_tilde_values.size).T,
        }
        return generic_save_to_h5(data_dict, fbasename, directory, doOverwrite, *args, **kwargs)

    def compute_disorder_averaging(self, method:str, M:float, B_tilde:float, strength:float, iterations:int = 100, B:float = 1.0, t:float = 1.0, A_tilde:float = 0.0, *args, **kwargs):
        def worker_function(*args, **kwargs):
            hamiltonian = self.compute_hamiltonian(method=method, M=M, B_tilde=B_tilde, B=B, t=t, A_tilde=A_tilde, *args, **kwargs)
            if any(np.isnan(hamiltonian)):
                return np.nan
            disorder = self.compute_disorder(strength, self.hamiltonian.shape[0])
            projector = self.compute_projector(hamiltonian + disorder)
            if method != "triangular":
                bott_index = self.compute_bott_index(projector)
            else:
                bott_index = self.compute_bott_index(projector, X=self.ParentLattice.X, Y=self.ParentLattice.Y)
            return bott_index

        data = generic_multiprocessing(worker_function, range(iterations), n_jobs=4, *args, **kwargs)
        data = np.array(data)[~np.isnan(data)]
        return np.mean(data)

    def compute_disorder_phase_diagram(self, method:str, strength:float, iterations:int, M_range:tuple = (-2.0, 8.0), B_tilde:tuple = (0.0, 0.5), resolution:tuple = None, B:float = 1.0, t:float = 1.0, A_tilde:float = 0.0, doOverwrite:bool = False, directory:str = None, *args, **kwargs):
        sizestr = f"{resolution[0]}x{resolution[1]}" if resolution is not None else "auto"
        fbasename = f"bott_index_disorder_w{strength:.3f}_i{iterations}_g{self.generation}_{method}_{sizestr}_.h5"
        directory = "./Triangle/PhaseDiagrams/Bott/" if directory is None else directory

        if os.path.exists(directory + fbasename) and not doOverwrite:
            print(f"{directory + fbasename} already exists. Set doOverwrite=True to overwrite.")
            return directory + fbasename

        if resolution is None:
            M_values = np.arange(M_range[0], M_range[1], 0.1)
            B_tilde_values = np.arange(B_tilde[0], B_tilde[1], 0.05)
        else:
            M_values = np.linspace(M_range[0], M_range[1], resolution[0])
            B_tilde_values = np.linspace(B_tilde[0], B_tilde[1], resolution[1])

        parameter_values = tuple(product(M_values, B_tilde_values))

        def worker_function(parameters, *args, **kwargs):
            M, B_tilde = parameters
            averaged_bott = self.compute_disorder_averaging(method, M, B_tilde, strength, iterations, B, t, A_tilde, doProgressBar=False *args, **kwargs)
            return [M, B_tilde, averaged_bott]
        
        data = generic_multiprocessing(worker_function, parameter_values, n_jobs=4, *args, **kwargs)
        M_data, B_tilde_data, bott_data = np.array(data).T
        data_dict = {
            "M_values": M_data,
            "B_tilde_values": B_tilde_data,
            "bott_values": bott_data.reshape(M_values.size, B_tilde_values.size).T,
        }
        return generic_save_to_h5(data_dict, fbasename, directory, doOverwrite, *args, **kwargs)

    def plot_site_removal_comparison(self, *args, **kwargs):
        tri_X, tri_Y = self.ParentLattice.X, self.ParentLattice.Y
        frac_X, frac_Y = self.X, self.Y

        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].scatter(tri_X, tri_Y, c='r', s=15)
        axs[0].scatter(frac_X, frac_Y, c='k', s=15)
        axs[1].scatter(frac_X, frac_Y, c='k', s=15)
        for ax in axs.flatten():
            ax.axis("off")
            ax.set_aspect("equal")

        fig.suptitle("Generation {}".format(self.generation), fontsize=16)
        fig.tight_layout()
        plt.show()

    def plot_site_NN_removal(self, *args, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(10,5))

        self.ParentLattice.plot(ax, NN_color="red", site_color="red")
        self.plot(ax)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_title("")
        plt.tight_layout()
        plt.show()



def generic_multiprocessing(func:callable, parameter_values:tuple, n_jobs:int = -1, progress_title:str = "Progress bar without a title.", doProgressBar:bool = True, *args, **kwargs):
    if doProgressBar:
        with tqdm_joblib(tqdm(total=len(parameter_values), desc=progress_title)) as progress_bar:
            data = Parallel(n_jobs=n_jobs)(delayed(func)(params, *args, **kwargs) for params in parameter_values)
    else:
        data = Parallel(n_jobs=n_jobs)(delayed(func)(params, *args, **kwargs) for params in parameter_values)
    return data


def generic_save_to_h5(data_dictionary:dict, filename:str, directory:str, *args, **kwargs):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
 
    with h5py.File(filepath, 'w') as f:
        for key, value in data_dictionary.items():
            f.create_dataset(name=key, data=value)
    return filepath
      

def plot_bottom_line(LatticeObj:Lattice, method:str, resolution:int = 51, doOverwrite:bool = False):
    M_values:np.ndarray = np.linspace(6., 7., resolution)

    fbasename = f"bottom_line_{method}.h5"
    fdirname = "./Triangle/PhaseDiagrams/Bott/"

    def bott_from_params(M):
        H = LatticeObj.compute_hamiltonian(method=method, M=M, B_tilde=0.0, B=1.0, t=1.0, A_tilde=0.0)
        P = LatticeObj.compute_projector(H)
        bott_index = LatticeObj.compute_bott_index(P, X=LatticeObj.X, Y=LatticeObj.Y)
        return [M, bott_index]

    if os.path.exists(fdirname + fbasename) and not doOverwrite:
        filepath = fdirname + fbasename
    else:
        data = generic_multiprocessing(bott_from_params, M_values, n_jobs=4, LatticeObj=LatticeObj, method=method, B_tilde=0.0)
        M_values, B_tilde_values, bott_values = np.array(data).T
        data_dict = {
            "M_values": M_values,
            "B_tilde_values": B_tilde_values,
            "bott_values": bott_values,
        }
        filepath = generic_save_to_h5(data_dict, fbasename, fdirname, doOverwrite=doOverwrite)
   
    with h5py.File(filepath, "r") as f:
        M_values = f["M_values"][:]
        B_tilde_values = f["B_tilde_values"][:]
        bott_values = f["bott_values"][:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(M_values, bott_values, c='k', zorder=3)
    ax.set_xlabel("M")
    ax.set_ylabel("Bott Index")
    ax.set_title(f"Bott Index : NN only : Method {method}")

    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels([-2, -1, 0, 1, 2])


    ax.set_xticks([-4, -2, 6, 7, 8])
    for xpos in [-4, -2, 6, 7, 8]:
        ax.axvline(x=xpos, color='black', alpha=0.5, linestyle='--', zorder=0, lw=1)

    plt.grid(alpha=0.5, zorder=0)
    plt.tight_layout()
    #plt.savefig("./Triangle/PhaseDiagrams/Bott/"+f"bottom_line_{method}.png")
    plt.show()


def plot_imshow(fig, ax, X_values, Y_values, Z_values, cmap='Spectral', plotColorbar=True, doDiscreteColormap=True):
    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]
    Z_values = np.where(Z_values == -0, 0, Z_values)

    not_nan_mask = ~np.isnan(Z_values)
    Z_range = [np.min(Z_values[not_nan_mask]), np.max(Z_values[not_nan_mask])]
    unique_values = np.arange(Z_range[0], Z_range[1] + 1, 1).astype(int)

    if doDiscreteColormap:
        if len(unique_values) < 25:
            cmap = plt.get_cmap(cmap)
            discrete_colors = cmap(np.linspace(0, 1, len(unique_values)))
            cmap = ListedColormap(discrete_colors)
            norm = BoundaryNorm(boundaries=np.append(unique_values, unique_values[-1] + 1), ncolors=len(unique_values))

    im = ax.imshow(Z_values, extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]], 
                   origin='lower', aspect='auto', cmap=cmap, interpolation='none', 
                   rasterized=True, norm=norm)
    
    if plotColorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_ticks(unique_values + 0.5)
        cbar.set_ticklabels([str(val) for val in unique_values], fontsize=16)

    return fig, ax


def plot_from_file(filename:str, Atilde:float):
    with h5py.File(filename, "r") as f:
        M_data = f["M_values"][:]
        B_tilde_data = f["B_tilde_values"][:]
        bott_data = f["bott_values"][:]

    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    
    plot_imshow(fig, ax, M_data, B_tilde_data, bott_data)

    linex = np.linspace(6.0, np.max(M_data), 500)
    ax.plot(linex, linex/8 - 0.75, c='k', lw=1, zorder=2)
    for xpos in [-2.0, 7.0]:
        ax.axvline(x=xpos, color='black', linewidth=1, zorder=2)
    ax.set_yticks([1/8, 1/4, 1/2])
    ax.set_yticklabels([r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$'], fontsize=16)
    xticks = [-2, 6, 7, 8]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=16)
    ax.set_ylabel(r"$\tilde{B}$", rotation=90, fontsize=16)
    ax.set_xlabel("M", fontsize=16)

    ax.set_xlim([np.min(M_data), np.max(M_data)])
    ax.set_ylim([np.min(B_tilde_data), np.max(B_tilde_data)])
    ax.set_title(filename.split("/")[-1].split(".")[0]+f"Atilde={Atilde}", fontsize=16)
    plt.tight_layout()
    return fig, ax



# -----------------------------------------------------------------------
if __name__ == "__main__":
    FracLat = SierpinskiTriangularLattice(True, 6)
    FracLat.ParentLattice.info

    t0 = time()
    fname = FracLat.compute_phase_diagram("renorm", resolution=(45, 9))
    print("\nTime taken:", time() - t0)

    fig, ax = plot_from_file(fname, Atilde=0.0)
    plt.savefig(fname.replace(".h5", ".png"))
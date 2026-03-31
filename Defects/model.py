import os
import inspect
from itertools import product
from typing import cast
from collections import Counter

import numpy as np

import scipy.linalg as spla
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D

from joblib import Parallel, delayed
from tqdm_joblib import tqdm, tqdm_joblib


def plot_spectrum_ax(spectrum_ax: Axes, eigenvalues: np.ndarray, scatter_label: str, ldos_idxs: np.ndarray):
    """Internal helper to render the 1D energy spectrum and highlight boundary states."""
    x_values = np.arange(len(eigenvalues))
    idxs_mask = np.isin(x_values, ldos_idxs)
    
    scat1 = spectrum_ax.scatter(x_values[~idxs_mask], eigenvalues[~idxs_mask], s=25, color='black', zorder=0)
    scat2 = spectrum_ax.scatter(x_values[idxs_mask], eigenvalues[idxs_mask], s=25, color='red', zorder=1)

    n_eigenvalues = len(eigenvalues)
    spectrum_ax.set_xticks([0, n_eigenvalues // 2, n_eigenvalues])
    spectrum_ax.set_xticklabels([str(i) for i in [1, n_eigenvalues // 2, n_eigenvalues]], fontsize=16)

    spectrum_ax.tick_params(axis='both', labelsize=20, width=2)
    epsilon = 0.25
    spectrum_ax.set_ylim(-3.0 - epsilon, 3.0 + epsilon)
    spectrum_ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])

    for spine in spectrum_ax.spines.values():
        spine.set_linewidth(2.0)
        
    spectrum_ax.annotate(
        scatter_label,
        xy=(0.95, 0.5),
        xycoords='axes fraction',
        ha='right',
        va='bottom',
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.0)
    )
    scat1.set_rasterized(True)
    scat2.set_rasterized(True)


def plot_ldos_ax(ax:Axes, fig: Figure, LDOS: np.ndarray, X: np.ndarray, Y: np.ndarray, lattice:np.ndarray, plot_type: str, doInterpolation:bool, interpolation_type:str):
    """Internal helper to project the LDOS data onto 2D meshes or 3D topological surfaces."""
    if plot_type == 'tri':
        box = ax.get_position()
        ldos_ax = fig.add_axes(rect = (box.x0, box.y0 + box.height * 0.55, box.width * 0.5, box.height * 0.5))
        ldos_ax.set_aspect('equal')
        plot = ldos_ax.tripcolor(X, Y, LDOS, cmap='jet') 
    
    elif plot_type == 'imshow':
        box = ax.get_position()
        ldos_ax = fig.add_axes(rect = (box.x0, box.y0 + box.height * 0.55, box.width * 0.5, box.height * 0.5))
        ldos_ax.set_aspect('equal')
        Z = np.full(lattice.size, np.nan)
        filled_idxs = np.argwhere(lattice.flatten() >= 0).flatten()
        Z[filled_idxs] = LDOS
        
        plot = ldos_ax.pcolormesh(Z.reshape(lattice.shape), cmap='cividis')

    elif plot_type == 'surface':
        if doInterpolation:
            grid_res = 101  
            xi = np.linspace(np.min(X), np.max(X), grid_res)
            yi = np.linspace(np.min(Y), np.max(Y), grid_res)
            XI, YI = np.meshgrid(xi, yi)
            points = np.column_stack((X, Y))

            match interpolation_type:
                case 'log':
                    eps = 1e-12
                    LDOS_log = np.log(LDOS + eps)
                    LDOS_log_interp = griddata(points, LDOS_log, (XI, YI), method='linear', fill_value=np.nan)
                    LDOS_interp = np.exp(LDOS_log_interp) - eps
                    LDOS_interp[np.isnan(LDOS_interp)] = 0.0
                case 'linear':
                    LDOS_interp = griddata(points, LDOS, (XI, YI), method='linear', fill_value=0)
                    LDOS_interp = gaussian_filter(LDOS_interp, sigma=1.0)
                case 'rbf':
                    rbf = Rbf(X, Y, LDOS, function='multiquadric', epsilon=0.1) 
                    LDOS_interp = rbf(XI, YI)
                    LDOS_interp = np.nan_to_num(LDOS_interp, nan=0.0)
                case 'kde':
                    xy = np.vstack([X, Y])
                    kde = gaussian_kde(xy, weights=LDOS, bw_method=0.5)
                    xi = np.linspace(np.min(X), np.max(X), 301)
                    yi = np.linspace(np.min(Y), np.max(Y), 301)
                    XI, YI = np.meshgrid(xi, yi)
                    LDOS_kde = kde(np.vstack([XI.ravel(), YI.ravel()])).reshape(XI.shape)
                    LDOS_kde *= (LDOS.max() / LDOS_kde.max())
                    LDOS_interp = LDOS_kde
                case _:
                    raise ValueError(f"Unknown interpolation method: {interpolation_type}")

            if np.max(LDOS_interp) > 0:
                LDOS_interp = LDOS_interp * (np.max(LDOS) / np.max(LDOS_interp))
            X, Y, LDOS = XI.ravel(), YI.ravel(), LDOS_interp.ravel()

        box = ax.get_position()
        ldos_ax = cast(Axes3D, fig.add_axes(rect = (box.x0, box.y0 + box.height * 0.55, box.width * 0.5, box.height * 0.5), projection='3d'))
        plot = ldos_ax.plot_trisurf(X, Y, LDOS, cmap='jet', linewidth=0.2, antialiased=False)

        ldos_ax.set_zticklabels([]) # type: ignore
        ldos_ax.set_zlabel("")
        ldos_ax.set_facecolor((1, 1, 1, 0))
        ldos_ax.grid(False)

    ldos_ax.set_xticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
    ldos_ax.set_yticks([np.min(Y), (np.max(Y) + np.min(Y)) // 2, np.max(Y)])
    ldos_ax.set_xticklabels([str(int(np.min(X) + 1)), "$L_x$", str(int(np.max(X) + 1))], fontsize=14)
    ldos_ax.set_yticklabels([str(int(np.min(Y) + 1)), "$L_y$", str(int(np.max(Y) + 1))], fontsize=14)
    plot.set_clim(vmin=0)
    
    cax = inset_axes(
        ax, 
        width="100%",  
        height="100%",  
        bbox_to_anchor=(0.8, 0.05, 0.1, 0.35),  
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    
    cbar = fig.colorbar(ldos_ax.collections[0], cax=cax, orientation='vertical')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    formatter.set_scientific(True)
    formatter.format = "%.1f"
    cbar.formatter = formatter
    cbar.update_ticks()
    
    vmin, vmax = plot.get_clim()
    cbar.ax.yaxis.set_ticks([vmin, (vmax + vmin) / 2, vmax])
    cbar.ax.yaxis.offsetText.set_position((0., 0.0))
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.offsetText.set_fontsize(14)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.set_ticks_position('left')
    
    plot.set_rasterized(True)
    return ldos_ax


class DefectSquareLattice:
    """
    A unified framework for generating and analyzing a tight-binding square lattice 
    subject to various topological and structural defects.
    
    This class supports the construction of pristine lattices as well as those containing 
    vacancies, substitutions, interstitials, Frenkel pairs, and Schottky defects. It 
    facilitates the calculation of real-space geometries, Wannier polar matrices, 
    Hamiltonians (with or without disorder), Local Density of States (LDOS), and 
    topological invariants (Bott index).
    """

    def __init__(self, Lx: int, Ly: int, defect_type: str, pbc: bool = True, 
                 frenkel_pair_index: int = 0, schottky_distance: int = 1, schottky_type: int = 0, 
                 doLargeDefect: bool = False, r0: float = 1.0, R: float = 1.0, 
                 doSquareDefect: bool = False, sqdWidth: int = 0,
                 *args, **kwargs):
        """
        Initializes the DefectSquareLattice with specified spatial and defect parameters.

        Args:
            Lx (int): The number of lattice sites along the x-direction.
            Ly (int): The number of lattice sites along the y-direction.
            defect_type (str): The classification of the defect to embed. Accepted 
                values are "none", "vacancy", "substitution", "interstitial", 
                "frenkel_pair", or "schottky".
            pbc (bool, optional): If True, enforces periodic boundary conditions 
                across the lattice edges. Defaults to True.
            frenkel_pair_index (int, optional): A categorical index (0 through 7) 
                determining the specific displacement vector for a Frenkel pair. Defaults to 0.
            schottky_distance (int, optional): The spatial separation between paired 
                Schottky defects, expressed as a multiplier of sqrt(2). Defaults to 1.
            schottky_type (int, optional): The topological configuration of the Schottky 
                defect (0, 1, or 2). Defaults to 0.
            doLargeDefect (bool, optional): If True, expands the central defect footprint 
                to include nearest neighbors. Useful via the `LargeDefectLattice` property. 
                Has no effect for "none", "schottky", or "frenkel_pair". Defaults to False.
            r0 (float, optional): Decay length parameter for Wannier amplitudes. Defaults to 1.0.
            R (float, optional): Cutoff radius for hopping terms. Defaults to 1.0.
            doSquareDefect (bool, optional): If True, shapes the localized defect as a 
                square patch. Defaults to False.
            sqdWidth (int, optional): The discrete width of the square defect if 
                `doSquareDefect` is enabled. Defaults to None.
                
        Raises:
            ValueError: If parity/sizing constraints for specific defect types are violated, 
                or if an unrecognized `defect_type` is provided.
        """
        # Set values within the class
        self._pbc = pbc
        self._Lx = Lx
        self._Ly = Ly
        self._defect_type = defect_type
        self._doLargeDefect = doLargeDefect
        self._frenkel_pair_index = frenkel_pair_index
        self._schottky_distance = schottky_distance
        self._schottky_type = schottky_type

        # Pauli matrices for Hamiltonian computation
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)     
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._pauli_matrices = [pauli_x, pauli_y, pauli_z]

        # Generate the pristine lattice and ensure proper sizing.
        self._pristine_lattice = self.compute_lattice() 
        if self.defect_type == "interstitial" and self.Lx % 2 == 1 and self.Ly % 2 == 1:
            raise ValueError("Both Lx and Ly must be even for interstitial defect.")
        elif self.defect_type == "schottky" and ((self.Lx + self.schottky_distance) % 2 != 1 or (self.Ly + self.schottky_distance) % 2 != 1):
            raise ValueError("Lx or Ly + schottky distance must be odd for schottky defect. They are {} and {}".format(self.Lx + self.schottky_distance, self.Ly + self.schottky_distance))
        elif self.defect_type in ["vacancy", "substitution", "frenkel_pair"] and self.Lx % 2 == 0 and self.Ly % 2 == 0:
            raise ValueError("Both Lx and Ly must be odd for vacancy, substitution, and frenkel_pair defects.")

        # Generate the defect lattice based on the defect type.
        match self.defect_type:
            case "none":
                self._lattice = self._pristine_lattice.copy()
                self._defect_indices = []
            case "vacancy":
                self._lattice = self.compute_vacancy_lattice(doSquareDefect)
                self._defect_indices = []
            case "substitution":
                self._lattice = self._pristine_lattice.copy()
                if self.doLargeDefect:
                    self._defect_indices = []
                    center_idx = [self.Ly // 2, self.Lx // 2]
                    # Create a large defect by marking the center and its immediate neighbors
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            if abs(i) + abs(j) == 2:
                                continue
                            self._defect_indices.append(self.lattice[center_idx[0] + i, center_idx[1] + j])
                elif doSquareDefect:
                    self._defect_indices = []
                    center_idx = [self.Ly // 2, self.Lx // 2]
                    # Create a large defect by marking the center and its immediate neighbors
                    width = sqdWidth
                    for i in np.arange(-width // 2 + 1, width // 2 + 1):
                        for j in np.arange(-width // 2 + 1, width // 2 + 1):
                            self._defect_indices.append(self.lattice[center_idx[0] + i, center_idx[1] + j])
                else:
                    self._defect_indices = [self.lattice[self.Ly // 2, self.Lx // 2]]
            case "interstitial":
                self._lattice = self.compute_interstitial_lattice()
                lattice_max = np.max(self.lattice)
                if self.doLargeDefect:
                    self._defect_indices = [lattice_max // 2, 
                                            lattice_max // 2 + 1, 
                                            lattice_max // 2 - 1,
                                            lattice_max // 2 + self.Ly + 2,
                                            lattice_max // 2 - self.Ly - 2]
                else:
                    self._defect_indices = [lattice_max // 2]
            case "frenkel_pair":
                if frenkel_pair_index not in range(8):
                    raise ValueError(f"Frenkel pair index must be between 0 and 7, got {frenkel_pair_index}.")
                self._lattice, self._defect_indices = self.compute_frenkel_pair_lattice(frenkel_pair_index)
            case "schottky":
                self._lattice, self._defect_indices = self.compute_schottky_lattice()
            case _:
                raise ValueError(f"Unknown defect type: {defect_type}")
            
        if self.defect_type == "schottky":
            rows_to_remove = []
            for i, defect_index in enumerate(self.defect_indices):
                if i % 2 == 0:
                    # Removal (vacancy) of down parity
                    rows_to_remove.append(defect_index * 2 + 1)
                else:
                    # Removal (vacancy) of up parity
                    rows_to_remove.append(defect_index * 2)
            mask = np.full(self.Lx * self.Ly * 2, True, dtype=bool)
            mask[rows_to_remove] = False
            self._mask = mask

        # Get the lattice coordinates. Adjust for the interstitial and frenkel pair defects 
        # (the lattices are generated with doubled coordinates).
        self._Y, self._X = np.where(self._lattice >= 0)[:]
        if self._defect_type in ["interstitial", "frenkel_pair"]:
            self._X = self._X.astype(float) / 2
            self._Y = self._Y.astype(float) / 2

        self._system_size = len(self.X)

        # Compute the distances and Wannier matrices
        self.compute_distances()
        self.compute_wannier_polar(r0=r0, R=R)

        # Create a large defect lattice if needed
        if not self._doLargeDefect:
            self.LargeDefectLattice = DefectSquareLattice(
                Lx, Ly, defect_type, pbc=pbc, doLargeDefect=True, 
                frenkel_pair_index=self._frenkel_pair_index, schottky_distance=self.schottky_distance
            )
        else:
            self.LargeDefectLattice = None

    # region Properties
    @property
    def Lx(self): return self._Lx
    @property
    def Ly(self): return self._Ly
    @property
    def pbc(self): return self._pbc 
    @property
    def defect_type(self): return self._defect_type
    @property
    def defect_indices(self): return self._defect_indices
    @property
    def X(self): return self._X
    @property
    def Y(self): return self._Y
    @property
    def dx(self): return self._dx
    @property
    def dy(self): return self._dy
    @property
    def lattice(self):
        if self.defect_type in ["interstitial", "frenkel_pair"]:
            print(f"Warning: Lattice coordinates must be halved for '{self.defect_type}' defects.", end=' ')
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                print("Called from line", frame.f_back.f_lineno)
        return self._lattice
    @property
    def pauli_matrices(self): return self._pauli_matrices
    @property
    def system_size(self): return self._system_size
    @property
    def Sx(self): return self._Sx
    @property
    def Sy(self): return self._Sy
    @property
    def Cx_plus_Cy(self): return self._Cx_plus_Cy
    @property
    def I(self): return self._I
    @property
    def doLargeDefect(self): return self._doLargeDefect
    @property
    def schottky_distance(self): return self._schottky_distance
    # endregion

    # region Geometry
    def compute_lattice(self) -> np.ndarray:
        """
        Constructs the baseline geometry for a pristine square lattice.

        Returns:
            np.ndarray: A 2D array of shape (Ly, Lx) mapping spatial coordinates to site indices.
        """
        return np.arange(self.Lx * self.Ly).reshape((self.Ly, self.Lx))

    def compute_vacancy_lattice(self, doSquare: bool = False, *args, **kwargs) -> np.ndarray:
        """
        Generates a modified lattice grid incorporating a central vacancy defect.

        If initialized with `doLargeDefect=True`, the vacancy is expanded into a 
        cross-like pattern involving nearest neighbors. If `doSquare=True`, a 5x5 
        square void is excised from the center.

        Args:
            doSquare (bool, optional): Flag to trigger a 5x5 square vacancy geometry. 
                Defaults to False.

        Returns:
            np.ndarray: The modified lattice array with vacant sites marked as -1 and 
                remaining sites consecutively re-indexed.
        """
        lattice = self._pristine_lattice.copy()

        x_center = self.Lx // 2
        y_center = self.Ly // 2

        lattice[y_center, x_center] = -1
        vacant_positions = []
        if self._doLargeDefect:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if abs(i) + abs(j) == 2 or (i == 0 and j == 0):
                        continue
                    lattice[y_center + i, x_center + j] = -1
                    vacant_positions.append((y_center + i, x_center + j))

        elif doSquare:
            for i in [-2, -1, 0, 1, 2]:
                for j in [-2, -1, 0, 1, 2]:
                    lattice[y_center + i, x_center + j] = -1
                    vacant_positions.append((y_center + i, x_center + j))

        self._vacant_positions = vacant_positions
        mask = lattice >= 0
        lattice[mask] = np.arange(np.count_nonzero(mask))
        return lattice

    def compute_interstitial_lattice(self) -> np.ndarray:
        """
        Generates a spatial grid mapping for a lattice containing an interstitial defect.

        The defect is positioned at the geometric center (mean coordinates). To account 
        for half-integer spacing, the underlying coordinate array dimensions are inherently 
        doubled prior to return. If `doLargeDefect` is True, nearest neighbors are also 
        flagged as interstitials.

        Returns:
            np.ndarray: An expanded lattice array mapping coordinate pairs to site indices.
        """
        Y, X = np.where(self._pristine_lattice >= 0)
        x_mean = np.round(np.mean(X), 1)
        y_mean = np.round(np.mean(Y), 1)
        coordinates = np.array([X, Y])

        if self._doLargeDefect:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if abs(i) + abs(j) != 2:
                        coordinates = np.concatenate((coordinates, np.array([[x_mean + i], [y_mean + j]])), axis=1)
        else:
            coordinates = np.concatenate((coordinates, np.array([[x_mean], [y_mean]])), axis=1)

        coordinates = np.unique(np.round(coordinates * 2).astype(int), axis=1)
        coordinates = coordinates[:, np.lexsort((coordinates[0], coordinates[1]))]

        interstitial_lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
        interstitial_lattice[coordinates[1], coordinates[0]] = np.arange(len(coordinates[0]))
        return interstitial_lattice

    def compute_frenkel_pair_lattice(self, displacement_index: int) -> tuple[np.ndarray, list[int]]:
        """
        Generates a lattice topology encompassing a Frenkel pair (vacancy + interstitial).

        A central vacancy is excised, and the displaced atom is inserted as an interstitial 
        at a relative coordinate dictated by the `displacement_index`.

        Args:
            displacement_index (int): Identifier (0-7) determining the direction and 
                magnitude of the atomic displacement.

        Returns:
            tuple[np.ndarray, list[int]]: A tuple comprising the expanded defect lattice 
                array and a list containing the newly assigned index of the interstitial site.
                
        Raises:
            ValueError: If `displacement_index` falls outside the valid range [0, 7].
        """
        if displacement_index < 0 or displacement_index > 7:
            raise ValueError("Displacement index must be between 0 and 7.")

        x_center = self.Lx // 2
        y_center = self.Ly // 2
        center = np.array([x_center, y_center]).reshape(2, 1)
        temporary_lattice = self._pristine_lattice.copy()
        temporary_lattice[center[1], center[0]] = -1
        Y, X = np.where(temporary_lattice >= 0)[:]
        coordinates = (np.array([X, Y]) * 2).astype(int)

        values = [-3, -1, 1, 3]
        displacements = np.array(list(product(values, repeat=2)))
        good_displacements = []
        for d in displacements:
            if np.abs(d[0]) == np.abs(d[1]):
                pass
            else:
                good_displacements.append(d.reshape(2,1))

        displacements = np.array(good_displacements)

        displacement_location = center * 2 + displacements[displacement_index]
        coordinates = np.concatenate((coordinates, displacement_location), axis=1)

        new_lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
        new_lattice[coordinates[1], coordinates[0]] = np.arange(len(coordinates[0]))

        new_y, new_x = np.where(new_lattice >= 0)[:]
        defect_index = np.argwhere(new_x%2).flatten()[0]
        return new_lattice, [defect_index]

    def compute_schottky_lattice(self) -> tuple[np.ndarray, list[int]]:
        """
        Generates the lattice framework for a system hosting Schottky defects.

        The exact coordinates of the paired orbital vacancies are determined by the 
        internal state variables `_schottky_distance` and `_schottky_type`.

        Returns:
            tuple[np.ndarray, list[int]]: A tuple containing the structural lattice array 
                and a list of 1D site indices corresponding to the defect locations.
        """
        lattice = self._pristine_lattice.copy()

        midpoint_right = self.Lx // 2 + self._schottky_distance // 2
        midpoint_left = self.Lx // 2 - self._schottky_distance // 2 - 1

        if self._schottky_type == 0:
            up_parity_idx =   lattice[midpoint_right, midpoint_right]
            down_parity_idx = lattice[midpoint_left, midpoint_left]
            defect_idxs = [up_parity_idx, down_parity_idx]
            vacant_positions = [(midpoint_right, midpoint_right), (midpoint_left, midpoint_left)]
        elif self._schottky_type == 1:
            up_parity_idx1 =   lattice[midpoint_right, midpoint_right]
            down_parity_idx1 = lattice[midpoint_left, midpoint_left]
            up_parity_idx2 =   lattice[midpoint_right - self._schottky_distance, midpoint_right]
            down_parity_idx2 = lattice[midpoint_left + self._schottky_distance, midpoint_left]
            defect_idxs = [up_parity_idx1, down_parity_idx1, up_parity_idx2, down_parity_idx2]
            vacant_positions = [(midpoint_right, midpoint_right), (midpoint_left, midpoint_left), (midpoint_right - self._schottky_distance, midpoint_right), (midpoint_left + self._schottky_distance, midpoint_left)]
        elif self._schottky_type == 2:
            up_parity_idx1 =   lattice[midpoint_right, midpoint_right]
            up_parity_idx2 =   lattice[midpoint_left, midpoint_left]
            down_parity_idx1 = lattice[midpoint_right - self._schottky_distance, midpoint_right]
            down_parity_idx2 = lattice[midpoint_left + self._schottky_distance, midpoint_left]
            defect_idxs = [up_parity_idx1, down_parity_idx1, up_parity_idx2, down_parity_idx2]
            vacant_positions = [(midpoint_right, midpoint_right), (midpoint_left, midpoint_left), (midpoint_right - self._schottky_distance, midpoint_right), (midpoint_left + self._schottky_distance, midpoint_left)]

        self._vacant_positions = vacant_positions
        return lattice, defect_idxs
    
    def compute_distances(self):
        """
        """
        dx = self.X - self.X[:, None]
        dy = self.Y - self.Y[:, None]

        if self.pbc:
            # Apply periodic boundary conditions
            multipliers = tuple(product([-1, 0, 1], repeat=2))
            shifts = [(i * self.Lx, j * self.Ly) for i, j in multipliers]

            x_shifted = np.empty((dx.shape[0], dx.shape[1], len(shifts)), dtype=dx.dtype)
            y_shifted = np.empty((dy.shape[0], dy.shape[1], len(shifts)), dtype=dy.dtype)
            for i, (dx_shift, dy_shift) in enumerate(shifts):
                x_shifted[:, :, i] = dx + dx_shift
                y_shifted[:, :, i] = dy + dy_shift

            distances = x_shifted**2 + y_shifted**2
            minimal_hop = np.argmin(distances, axis=-1)
            i_idxs, j_idxs = np.indices(minimal_hop.shape)

            dx = x_shifted[i_idxs, j_idxs, minimal_hop]
            dy = y_shifted[i_idxs, j_idxs, minimal_hop]
            
        self._dx, self._dy = dx, dy

    def compute_wannier_polar(self, r0: float = 1.0, R: float = 1.0, *args, **kwargs):
        """
        Args:
            r0 (float, optional): Decay constant for hopping amplitudes. Defaults to 1.0.
            R (float, optional): Radial cutoff threshold beyond which hopping elements 
                are truncated. Defaults to 1.0.
        """
        dx, dy = self.dx, self.dy
        theta = np.arctan2(dy, dx)  
        dr = np.sqrt(dx ** 2 + dy ** 2)

        distance_mask = ((dr <= R + 1e-6) & (dr > 1e-6))
        principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & distance_mask 
        diagonal_mask  = ((np.isclose(np.abs(dx), np.abs(dy), atol=1e-4)) & (dx != 0)) & distance_mask
        hopping_mask = principal_mask | diagonal_mask
    
        d_cos = np.where(hopping_mask, np.cos(theta), 0. + 0.j)
        d_sin = np.where(hopping_mask, np.sin(theta), 0. + 0.j)
        amplitude = np.where(hopping_mask, np.exp(1. - dr / r0), 0. + 0.j)

        Cx_plus_Cy = amplitude / 2
        Sx = 1j * d_cos * amplitude / 2
        Sy = 1j * d_sin * amplitude / 2

        self._Cx_plus_Cy = Cx_plus_Cy
        self._Sx = Sx
        self._Sy = Sy
        self._I = np.eye(Sx.shape[0], dtype=complex)
    # endregion

    # region Computation
    def compute_hamiltonian(self, M_background: float, M_substitution: float|None = None, t: float = 1.0, t0: float = 1.0, 
                            tau_x: np.ndarray|None = None, tau_y: np.ndarray|None = None, tau_z: np.ndarray|None = None,
                            potentialDisorder: bool = False, massDisorder: bool = False, hopDisorder: bool = False, disorder_strength: float = 0.0) -> np.ndarray:
        """
        Args:
            M_background (float): The base mass parameter characterizing the pristine lattice.
            M_substitution (float, optional): The distinct mass assigned to substitution 
                and interstitial defect sites. Required if handling these specific defects.
            t (float, optional): Primary nearest-neighbor hopping coefficient (d1, d2 terms). Defaults to 1.0.
            t0 (float, optional): Secondary hopping parameter scaling the d3 term. Defaults to 1.0.
            tau_x (np.ndarray, optional): Custom Pauli X matrix. Defaults to standard implementation.
            tau_y (np.ndarray, optional): Custom Pauli Y matrix. Defaults to standard implementation.
            tau_z (np.ndarray, optional): Custom Pauli Z matrix. Defaults to standard implementation.
            potentialDisorder (bool, optional): Activates uniform random disorder across the onsite potential matrix. Defaults to False.
            massDisorder (bool, optional): Activates uniform random disorder across the onsite mass matrix. Defaults to False.
            hopDisorder (bool, optional): Activates uniform random disorder across the hopping matrices. Defaults to False.
            disorder_strength (float, optional): Fractional intensity multiplier for the chosen disorder distribution.

        Returns:
            np.ndarray: The finalized multi-orbital Hamiltonian matrix. For Schottky defects, 
                removed degrees of freedom are properly excised from this matrix.
                
        Raises:
            ValueError: If `M_substitution` is missing for defect types that obligatorily require it.
        """
        if potentialDisorder:
            delta_u = disorder_strength
            potential_disorder = np.random.uniform(-delta_u / 2, delta_u / 2, self.I.shape[0])
            potential_disorder -= np.mean(potential_disorder)
        else:
            potential_disorder = np.zeros(self.I.shape[0])

        if massDisorder:
            delta_m = M_background * disorder_strength
            mass_disorder = np.random.uniform(-delta_m / 2, delta_m / 2, self.I.shape[0])
            mass_disorder -= np.mean(mass_disorder)
        else:
            mass_disorder = np.zeros(self.I.shape[0])

        if hopDisorder:
            delta_t = t * disorder_strength
            t_disorder = np.random.uniform(-delta_t / 2, delta_t / 2, self.I.shape)
            t_disorder -= np.mean(t_disorder)
        else:
            t_disorder = np.zeros(self.I.shape)

        if self.defect_type in ["substitution", "interstitial", "frenkel_pair"]:
            if M_substitution is None:
                raise ValueError("M_substitution must be provided for 'substitution', 'interstitial', or 'frenkel pair' defects.")
            onsite_mass = (M_background) * self.I + np.diag(mass_disorder)
            onsite_mass[self.defect_indices, self.defect_indices] = M_substitution
        else:
            onsite_mass = (M_background) * self.I + np.diag(mass_disorder)

        d1 = (t + t_disorder) * self.Sx
        d2 = (t + t_disorder) * self.Sy

        if M_substitution is None:
            M_substitution = 1.0
        d3 = onsite_mass + t0 * (self.Cx_plus_Cy) * np.sign(M_substitution)

        if tau_x is None:
            tau_x = self.pauli_matrices[0]
        if tau_y is None:
            tau_y = self.pauli_matrices[1]
        if tau_z is None:
            tau_z = self.pauli_matrices[2]

        hamiltonian = np.kron(d1, tau_x) + np.kron(d2, tau_y) + np.kron(d3, tau_z) + np.kron(np.diag(potential_disorder), np.eye(2)).astype(np.complex128)

        if self.defect_type == "schottky":
            hamiltonian = hamiltonian[np.ix_(self._mask, self._mask)]

        return hamiltonian

    def compute_projector(self, hamiltonian: np.ndarray) -> np.ndarray:
        """
        Args:
            hamiltonian (np.ndarray): The full system Hamiltonian matrix.

        Returns:
            np.ndarray: The mathematical projector matrix spanned by the lower band eigenstates.
        """
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2]
        highest_lower_band = lower_band[-1]

        D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
        D_herm_conj = np.einsum('i,ij->ij', D, eigenvectors.conj().T)
        projector = eigenvectors @ D_herm_conj
        return projector

    def compute_bott_index(self, projector: np.ndarray) -> float:
        """
        Args:
            projector (np.ndarray): The spectral projection operator for the lower band.

        Returns:
            float: The numeric value of the computed Bott Index topological invariant.
        """
        X = np.repeat(self.X, 2)
        Y = np.repeat(self.Y, 2)
        if self.defect_type == "schottky":
            X = X[self._mask]
            Y = Y[self._mask]
            
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
        bott_index = np.imag(np.sum(np.log(spla.eigvals(A)))) / (2 * np.pi)
        return bott_index

    def compute_LDOS(self, hamiltonian: np.ndarray, number_of_states: int = 2, *args, **kwargs) -> dict:
        """
        Args:
            hamiltonian (np.ndarray): The full system Hamiltonian matrix.
            number_of_states (int, optional): The inclusive number of boundary/gap eigenstates 
                (symmetric around E=0) to integrate for the LDOS profile. Defaults to 2.

        Returns:
            dict: A comprehensive dictionary payload mapping:
                - "LDOS": The normalized, parity-summed real-space distribution vector.
                - "eigenvalues": The ordered global energy spectrum.
                - "gap": The numeric energy gap isolating the upper and lower eigenbands.
                - "bandwidth": The full spectral width.
                - "ldos_idxs": The explicit integer indices marking the eigenstates utilized.
        """
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        number_of_states += number_of_states % 2
        mid_index = len(eigenvalues) // 2
        
        lower_idxs = np.arange(len(eigenvalues))[:mid_index][-number_of_states // 2:]
        upper_idxs = np.arange(len(eigenvalues))[mid_index:][:number_of_states // 2]
        selected_indices = np.concatenate((lower_idxs, upper_idxs))

        LDOS = np.sum(np.abs(eigenvectors[:, selected_indices]) ** 2, axis=1)
        if self.defect_type == "schottky":
            new_LDOS = np.zeros(self.Lx * self.Ly * 2, dtype=LDOS.dtype)
            new_LDOS[self._mask] = LDOS
            LDOS = new_LDOS
            
        LDOS = LDOS[0::2] + LDOS[1::2]
        LDOS = LDOS / np.sum(LDOS)
        gap = abs(np.max(eigenvalues[lower_idxs]) - np.min(eigenvalues[upper_idxs]))
        bandwidth = np.max(eigenvalues) - np.min(eigenvalues)

        data_dict = {
            "LDOS": LDOS,
            "eigenvalues": eigenvalues,
            "gap": gap,
            "bandwidth": bandwidth,
            "ldos_idxs": selected_indices
        }
        return data_dict

    def _compute_for_figure(self, m_background: float, m_substitution: float, number_of_states: int) -> tuple:
        """
        Args:
            m_background (float): Base background mass.
            m_substitution (float): Defect substitution mass.
            number_of_states (float): Eigenstate count incorporated for LDOS integration.

        Returns:
            tuple: An ordered collection spanning (LDOS array, eigenvalue array, gap, 
                Bott index, X coordinate map, Y coordinate map, LDOS spectral indices).
        """
        def _average_over_frenkel_pair():
            all_LDOS = []
            all_x = []
            all_y = []
            all_eigenvalues = []
            all_gap = []
            all_bott = []
            all_ldos_idxs = []
            for frenkel_pair_index in range(8):
                NewLattice = DefectSquareLattice(self.Lx, self.Ly, self.defect_type, pbc=self.pbc, frenkel_pair_index=frenkel_pair_index)
                hamiltonian = NewLattice.compute_hamiltonian(m_background, m_substitution)
                ldos_dict = NewLattice.compute_LDOS(hamiltonian, number_of_states = 2)
                this_LDOS, eigenvalues, gap, bandwidth, ldos_idxs = ldos_dict["LDOS"], ldos_dict["eigenvalues"], ldos_dict["gap"], ldos_dict["bandwidth"], ldos_dict["ldos_idxs"]
                projector = NewLattice.compute_projector(hamiltonian)
                bott_index = NewLattice.compute_bott_index(projector)
                
                all_LDOS.append(this_LDOS)
                all_x.append(NewLattice.X)
                all_y.append(NewLattice.Y)
                all_eigenvalues.append(eigenvalues)
                all_gap.append(gap)
                all_bott.append(bott_index)
                all_ldos_idxs.append(ldos_idxs)

            all_LDOS = np.concatenate(all_LDOS)
            all_x = np.concatenate(all_x, axis=0)
            all_y = np.concatenate(all_y, axis=0)
            all_gap = np.mean(all_gap)
            all_bott = np.mean(all_bott)

            coords = np.column_stack((all_x, all_y))
            unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)

            summed_LDOS = np.zeros(len(unique_coords), dtype=all_LDOS.dtype)
            np.add.at(summed_LDOS, inverse_indices, all_LDOS)

            summed_LDOS /= np.sum(summed_LDOS)
            X, Y = unique_coords[:, 0], unique_coords[:, 1]
            return summed_LDOS, all_eigenvalues[0], np.mean(all_gap), np.mean(all_bott), X, Y, all_ldos_idxs[0]
        
        if self.defect_type == "frenkel_pair":
            LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = _average_over_frenkel_pair()
        else:
            hamiltonian = self.compute_hamiltonian(m_background, m_substitution)
            projector = self.compute_projector(hamiltonian)
            bott_index = self.compute_bott_index(projector)
            ldos_dict = self.compute_LDOS(hamiltonian, number_of_states)
            LDOS, eigenvalues, gap, bandwidth, ldos_idxs = ldos_dict["LDOS"], ldos_dict["eigenvalues"], ldos_dict["gap"], ldos_dict["bandwidth"], ldos_dict["ldos_idxs"]
            X, Y = self.X, self.Y
            
        return LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs

    def _compute_for_figure_disorder(self, disorder_type: str, m_background: float, 
                                     m_substitution: float, number_of_states: int, 
                                     disorder_strength: float, n_iterations: int = 10, 
                                     n_jobs: int = -1, overwrite: bool = False,
                                     save_dir: str = "./Data/") -> str:
        """
        Conducts statistically robust parallel evaluations for lattices exhibiting uniform 
        random structural or mass disorder. Automates caching to a dynamically specified directory.

        Args:
            disorder_type (str): Domain of perturbation. Valid keys are 'onsite', 'mass', or 'hopping'.
            m_background (float): Base background mass.
            m_substitution (float): Defect substitution mass.
            number_of_states (float): Eigenstate count incorporated for LDOS integration.
            disorder_strength (float): The multiplicative scale governing disorder bounds.
            n_iterations (int, optional): The integer count of stochastic samples to generate 
                for averaging over the ensemble. Defaults to 10.
            n_jobs (int, optional): Defines the number of multi-threading backend jobs 
                to initialize via joblib. Defaults to -1 (using all cores).
            overwrite (bool, optional): If True, aggressively recreates local caches rather 
                than utilizing existing files. Defaults to False.
            save_dir (str, optional): The directory path where the resulting `.npz` file 
                will be saved. Defaults to "./Data/".

        Returns:
            str: Path indicating the physical location of the stored `.npz` archive 
                 packaging the averaged arrays and structural metadata.
        """
        assert disorder_type in ['onsite', 'mass', 'hopping']

        os.makedirs(save_dir, exist_ok=True)

        if self.defect_type == "frenkel_pair":
            fname = f"{disorder_type}_{self.defect_type}_Lx={self.Lx}_Ly={self.Ly}_mback={m_background}_msub={m_substitution}_fp={self._frenkel_pair_index}_w={disorder_strength}_n={n_iterations}.npz"
        else:
            fname = f"{disorder_type}_{self.defect_type}_Lx={self.Lx}_Ly={self.Ly}_mback={m_background}_msub={m_substitution}_w={disorder_strength}_n={n_iterations}.npz"
        
        file_path = os.path.join(save_dir, fname)
        
        if os.path.exists(file_path) and not overwrite:
            print(f"File already exists: {file_path}")
            return file_path

        def _worker(i):
            if disorder_type == 'onsite':
                hamiltonian_local = self.compute_hamiltonian(m_background, m_substitution, potentialDisorder=True, disorder_strength=disorder_strength)
            elif disorder_type == 'mass':
                hamiltonian_local = self.compute_hamiltonian(m_background, m_substitution, massDisorder=True, disorder_strength=disorder_strength)
            elif disorder_type == 'hopping':
                hamiltonian_local = self.compute_hamiltonian(m_background, m_substitution, hopDisorder=True, disorder_strength=disorder_strength)

            projector = self.compute_projector(hamiltonian_local)
            bott_index = self.compute_bott_index(projector)
            ldos_dict = self.compute_LDOS(hamiltonian_local, number_of_states)
            LDOS, eigenvalues, gap, bandwidth, ldos_idxs = ldos_dict["LDOS"], ldos_dict["eigenvalues"], ldos_dict["gap"], ldos_dict["bandwidth"], ldos_dict["ldos_idxs"]
            X, Y = self.X, self.Y
            return LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs
    
        with tqdm_joblib(tqdm(total=n_iterations, desc=f"{self.defect_type} : m_back={m_background}  : m_sub={m_substitution}")) as progress_bar:
            data = Parallel(n_jobs=n_jobs)(delayed(_worker)(i) for i in range(n_iterations))
        
        all_LDOS, all_eigenvalues, all_gap, all_bott_index, all_X, all_Y, all_ldos_idxs = zip(*data)
        
        LDOS = np.mean(all_LDOS, axis=0)
        eigenvalues = np.mean(all_eigenvalues, axis=0)
        gap = np.mean(all_gap)
        bott_index = np.mean(all_bott_index)
        X = all_X[0]
        Y = all_Y[0]
        
        data_dict = {
            "LDOS": LDOS, 
            "eigenvalues": eigenvalues, 
            "gap": gap, 
            "bott_index": bott_index, 
            "X": X, 
            "Y": Y, 
            "ldos_idxs": all_ldos_idxs[0], 
            "disorder_strength": disorder_strength, 
            "n_iterations": n_iterations
        }
        
        np.savez(file_path, **data_dict)
        return file_path
    # endregion


# region Plotting
    def plot_distances(self, idx: int, cmap: str = "inferno", doLargeDefectFigure: bool = False, *args, **kwargs):
        """
        Visualizes the spatial displacement (dx, dy, and absolute distance d) from a specified 
        reference site to all other sites within the lattice structure.

        Generates a tripartite figure pane showing horizontal, vertical, and radial distances.
        The reference site is explicitly highlighted with a red bounding circle.

        Args:
            idx (int, optional): The 1D index of the reference site. If None, defaults to the 
                geometric center of the lattice array.
            cmap (str, optional): Matplotlib colormap applied to the distance gradients. 
                Defaults to "inferno".
            doLargeDefectFigure (bool, optional): If True, extracts distances mapped over the 
                expanded `LargeDefectLattice` framework. Defaults to False.
        """
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        if idx is None:
            idx = len(self.X) // 2

        if doLargeDefectFigure and self.LargeDefectLattice is not None:
            dx, dy, d = self.LargeDefectLattice.dx, self.LargeDefectLattice.dy, np.sqrt(self.LargeDefectLattice.dx**2 + self.LargeDefectLattice.dy**2)
            X, Y = self.LargeDefectLattice.X, self.LargeDefectLattice.Y
        else:
            dx, dy, d = self.dx, self.dy, np.sqrt(self.dx**2 + self.dy**2)
            X, Y = self.X, self.Y

        distances = [dx, dy, d]
        labels = ["dx", "dy", "d"]
        
        for i, (distance, label) in enumerate(zip(distances, labels)):
            axs[i].set_title(label)
            axs[i].set_xlabel("X")
            axs[i].set_ylabel("Y")
            
            axs[i].scatter(X, Y, c=distance[idx], cmap=cmap, zorder=0, s=25)
            axs[i].scatter(self.X[idx], self.Y[idx], s=100, facecolors='none', edgecolors='red', zorder=1)
            axs[i].set_aspect('equal')
            
            cbar = fig.colorbar(axs[i].collections[0], ax=axs[i], orientation='vertical')
            cbar.set_label(f"Distance to site {idx}", rotation=270, labelpad=15)
            
        plt.tight_layout()
        plt.show()

    def plot_defect_idxs(self, ax: Axes|None = None) -> Axes:
        """
        Renders a 2D topological map of the lattice, explicitly highlighting pristine sites, 
        vacancies, and embedded defect locations with distinct hierarchical markers.

        Args:
            ax (plt.Axes, optional): A pre-existing Matplotlib Axes object to draw upon. 
                If None, a new 8x8 figure and axes are internally generated. Defaults to None.

        Returns:
            plt.Axes: The configured Matplotlib Axes object containing the rendered lattice map.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
        standard_site_size = 50
        large_site_size = 150
        
        ax.set_aspect('equal')
        ax.scatter(self.X, self.Y, s=standard_site_size, edgecolors='black', facecolors='black', linewidth=0.)
        
        for axes in [ax]:
            if len(self.defect_indices) == 0:
                for i, (x, y) in enumerate(self._vacant_positions):
                    if i == 2:
                        axes.scatter(x, y, s=large_site_size, edgecolors='red', facecolors='none', linewidth=1.5)
                    else:
                        axes.scatter(x, y, s=standard_site_size, edgecolors='red', facecolors='none', linewidth=1.5, alpha=1.0)
            
            elif len(self.defect_indices) in [2, 4]:
                for x, y in self._vacant_positions:
                    axes.scatter(x, y, s=large_site_size, edgecolors='none', facecolors='white')
                for i, defect_idx in enumerate(self.defect_indices):
                    c = "red" if i % 2 == 0 else "blue"
                    axes.scatter(self.X[defect_idx], self.Y[defect_idx], s=standard_site_size, facecolors='none', edgecolors=c, linewidth=1.5)
            
            elif len(self.defect_indices) == 5:
                for i, defect_idx in enumerate(self.defect_indices):
                    if (i == 0 and self.defect_type == "interstitial") or (i == 2 and self.defect_type == "substitution"):
                        axes.scatter(self.X[defect_idx], self.Y[defect_idx], s=large_site_size, color='red', edgecolors='black', linewidth=0.)
                    else:
                        axes.scatter(self.X[defect_idx], self.Y[defect_idx], s=standard_site_size, color='red')
            
            elif len(self.defect_indices) == 1:
                axes.scatter(self.X[self.defect_indices[0]], self.Y[self.defect_indices[0]], s=large_site_size, color='red', edgecolors='black', linewidth=0.)
                
        tick_vals = [np.min(self.X), (np.min(self.X) + np.max(self.X)) / 2, np.max(self.X)]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([str(int(label + 1)) for label in tick_vals], fontsize=16)
        ax.set_yticks(tick_vals)
        ax.set_yticklabels([str(int(label + 1)) for label in tick_vals], fontsize=16)
        
        ax.set_xlabel("$L_x$", fontsize=20)
        ax.set_ylabel("$L_y$", fontsize=20)
        
        return ax

    def plot_spectrum_ldos(self, m_background_values: list[float] = [2.5, 1.0, -1.0, -2.5], m_substitution_values: list[float|None] = [-2.5, -1.0, 1.0, 2.5],
                            doLargeDefectFigure: bool = False, doInterpolation: bool = True, plot_type: str = 'surface', interpolation_type:str = 'linear'):
            """
            Plots the global energy spectrum and the spatially resolved Local Density of States 
            (LDOS) for the defected lattice.

            Constructs a multi-panel figure for various combinations of background and substitution 
            mass parameters. Each subplot features both a 1D scatter projection of the eigenvalue 
            spectrum (highlighting mid-gap boundary states) and a 2D/3D plot of the corresponding 
            LDOS.

            Args:
                m_background_values (list[float], optional): A sequence of background mass parameters 
                    ($m_0$) to iterate over. Defaults to [2.5, 1.0, -1.0, -2.5].
                m_substitution_values (list[float], optional): A sequence of defect/substitution mass 
                    parameters to pair with the background masses. Defaults to [-2.5, -1.0, 1.0, 2.5].
                doLargeDefectFigure (bool, optional): If True, computes and renders the LDOS mapping 
                    over the expanded `LargeDefectLattice` framework. Defaults to False.
                doInterpolation (bool, optional): If True, smooths the discrete real-space LDOS array 
                    onto a finer continuous grid for aesthetic visualization. Defaults to True.
                plot_type (str, optional): The Matplotlib rendering style for the LDOS. Valid inputs 
                    include 'surface', 'imshow', or 'tri'. Defaults to 'surface'.
                interpolation_type (str, optional): The method used to interpolate data. Only used 
                    for the 'surface' plotting type.  Valid inputs include 'linear', 'log', 'rbf', 
                    or 'kde'.

            Returns:
                tuple[plt.Figure, np.ndarray]: The configured Matplotlib Figure and an array of the 
                    generated Axes objects.
            """
            if self.defect_type in ["none", "vacancy"]:
                m_substitution_values = [None] if doLargeDefectFigure is False else [None] * 2
                n_cols, n_rows = len(m_background_values), len(m_substitution_values)
            elif self.defect_type == "schottky":
                m_substitution_values = [None]
                n_cols, n_rows = len(m_background_values), len(m_substitution_values)
            elif Counter(m_background_values) == Counter(m_substitution_values):
                n_cols, n_rows = len(m_substitution_values) - 1, len(m_background_values)
            else:
                n_cols, n_rows = len(m_substitution_values), len(m_background_values)

            scale = 6
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(scale * n_cols, scale * n_rows))
            plt.subplots_adjust(wspace=0.4)

            if isinstance(axs, Axes):
                axs = np.array(axs).reshape(1, 1)
            elif n_rows == 1:
                axs = axs[np.newaxis, :]
            elif n_cols == 1:
                axs = axs[:, np.newaxis]

            for j, m_background in enumerate(m_background_values):
                good_m_sub_vals = np.array(m_substitution_values)[np.array(m_substitution_values) != m_background]
                for i, m_substitution in enumerate(good_m_sub_vals):

                    if i == 1 and doLargeDefectFigure and self.defect_type in ["vacancy"] and self.LargeDefectLattice is not None:
                        LDOS, eigenvalues, _, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution, 2)
                    elif doLargeDefectFigure and self.defect_type not in ["vacancy"] and self.LargeDefectLattice is not None:
                        LDOS, eigenvalues, _, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution, 2)
                    else:
                        LDOS, eigenvalues, _, bott_index, X, Y, ldos_idxs = self._compute_for_figure(m_background, m_substitution, 2)
                    
                    if self.defect_type in ['none', 'vacancy', 'schottky']:
                        ax = axs[i, j]
                    else:
                        ax = axs[j, i]

                    if self.defect_type in ["none", "vacancy", "schottky"]:
                        param_name = f"$m_0={m_background}$"
                    elif self.defect_type in ["substitution"]:
                        param_name = f"$m_0^{{\\rm back}}={m_background}$\n" + f"$m_0^{{\\rm sub}}={m_substitution}$"
                    else:
                        param_name = f"$m_0^{{\\rm back}}={m_background}$\n" + f"$m_0^{{\\rm int}}={m_substitution}$"

                    label = param_name + f"\nB $={bott_index:.1f}$"

                    plot_spectrum_ax(ax, eigenvalues, label, ldos_idxs)
                    plot_ldos_ax(ax, fig, LDOS, X, Y, self.lattice, plot_type, doInterpolation, interpolation_type)

            return fig, axs
    
    # endregion


def plot_disorder_figure(mback: float, msub: float, Lx:int, Ly:int, 
                         disorder_type: str, disorder_strength: float, n_iterations: int, 
                         doFP: bool = True, doInterpolation: bool = True, interpolation_type:str = 'linear'):
    """
    Args:
        mback (float): Base onsite background mass parameter.
        msub (float): Substituted atomic mass embedded at defect boundaries.
        disorder_type (str): Domain key classifying the applied perturbation 
            (e.g., 'onsite', 'mass', 'hopping').
        disorder_strength (float): Maximum absolute scale defining uniform noise breadth.
        n_iterations (int): Configuration count bounding the disorder ensemble average.
        doFP (bool, optional): If True, explicitly calculates and includes the 
            orientational average over the 8 distinct Frenkel pair geometries. Defaults to True.
        doInterpolation (bool, optional): Smoothes discrete lattice mappings using 
            Gaussian KDE projection prior to render. Defaults to True.

    Returns:
        str: Expected file prefix constructed from parameter invariants, used seamlessly 
             by external callers for saving `.png`/`.svg` plots.
    """

    def _average_over_frenkel_pair(Lx, Ly, mback, mint):
        """Internal helper resolving spatial isotropy globally across Frenkel pair arrangements."""
        all_LDOS = []
        all_x = []
        all_y = []
        all_eigenvalues = []
        all_bott = []
        for frenkel_pair_index in range(8):
            Lattice = DefectSquareLattice(Lx, Ly, 'frenkel_pair', True, frenkel_pair_index=frenkel_pair_index)
            fname = Lattice._compute_for_figure_disorder(disorder_type, mback, mint, 2, disorder_strength=disorder_strength, n_iterations=n_iterations)
            data = np.load(fname)
            all_LDOS.append(data["LDOS"])
            all_x.append(data["X"])
            all_y.append(data["Y"])
            all_eigenvalues.append(data["eigenvalues"])
            all_bott.append(data["bott_index"])
            W = data["disorder_strength"]
        
        all_LDOS = np.concatenate(all_LDOS, axis=0)
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        all_bott = np.mean(all_bott)

        coords = np.stack((all_x, all_y), axis=1)
        unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        summed_LDOS = np.zeros(len(unique_coords))
        np.add.at(summed_LDOS, inverse_indices, all_LDOS)
        summed_LDOS /= np.sum(summed_LDOS)
        LDOS = summed_LDOS
        eigenvalues = np.mean(all_eigenvalues, axis=0)
        X, Y = unique_coords[:, 0], unique_coords[:, 1]
        return {"LDOS": LDOS, "eigenvalues": eigenvalues, "X": X, "Y": Y, "disorder_strength": W, "bott_index": all_bott}


    fig, axs = plt.subplots(1, 5, figsize=(30, 6))

    for method in ["vacancy", "schottky", "substitution", "interstitial", "frenkel_pair"]:
        if method in ["schottky", "interstitial"]:
            this_Lx, this_Ly = Lx - 1, Ly - 1
        else:
            this_Lx, this_Ly = Lx, Ly

        Lattice = DefectSquareLattice(this_Lx, this_Ly, method, True)

        if method != "frenkel_pair":
            fname = Lattice._compute_for_figure_disorder(disorder_type, mback, msub, 2, disorder_strength=disorder_strength, n_iterations=n_iterations)
            data = np.load(fname)
        elif doFP:
            data = _average_over_frenkel_pair(this_Lx, this_Ly, mback, msub)
        else:
            data = None

        try:
            if data is None:
                raise Exception("'data' is None.")
            ax = axs[["vacancy", "schottky", "substitution", "interstitial", "frenkel_pair"].index(method)]

            ldos_idxs = np.arange(len(data["eigenvalues"]))
            ldos_idxs = ldos_idxs[(len(ldos_idxs) // 2 - 1):(len(ldos_idxs) // 2 + 1)]

            if method in ["vacancy", "schottky"]:
                label = f"$m_0={mback:.1f}$\n$W={data['disorder_strength']:.2f}$"
            elif method in ["substitution"]:
                label = f"$m_0^{{\\text{{back}}}}={mback:.1f}$\n$m_0^{{\\text{{sub}}}}={msub:.1f}$\n$W={data['disorder_strength']:.2f}$"
            else:
                label = f"$m_0^{{\\text{{back}}}}={mback:.1f}$\n$m_0^{{\\text{{int}}}}={msub:.1f}$\n$W={data['disorder_strength']:.2f}$"
            
            if method != "frenkel_pair":
                label += f"\nBI=${data['bott_index']:.2f}$"
                
            plot_spectrum_ax(ax, data["eigenvalues"], label, ldos_idxs)

            ax.set_title(method.capitalize(), fontsize=16)
            plot_ldos_ax(ax, fig, data["LDOS"], data["X"], data["Y"], Lattice.lattice, 'surface', doInterpolation, interpolation_type)
        except Exception as e:
            print("Exception: ", e)


def plot_bott_phase_diagram(m0_range: tuple, n: int, r0: float, R: float):
    """
    Args:
        m0_range (tuple): Defined parametric sequence spanning the (minimum, maximum) 
            background/onsite mass boundary logic.
        n (int): Explicit integer quantity of interpolated steps dividing `m0_range`.
        r0 (float): Mathematical decay length parameter constraining exponential hopping 
            amplitudes, expressed linearly in base units of the lattice constant `a`.
        R (float): Hard spherical cutoff threshold beyond which hopping interaction 
            integrals strictly evaluate to zero, expressed in units of `a`.
    """
    Lattice = DefectSquareLattice(20, 20, "none", True, r0=r0, R=R)

    values = np.linspace(m0_range[0], m0_range[1], n)
    
    def worker(m0):
        hamiltonian = Lattice.compute_hamiltonian(m0)
        proj = Lattice.compute_projector(hamiltonian)
        bott_index = Lattice.compute_bott_index(proj)
        return [m0, bott_index] 
    
    with tqdm_joblib(tqdm(total=len(values), desc='')) as progress_bar:
        data = np.array(Parallel(n_jobs=-1)(delayed(worker)(params) for params in values))

    m0, bott = data.T

    fig, ax = plt.subplots(1, 1, figsize=(6 ,6))
    ax.scatter(m0, bott)
    ax.set_xlabel('$m_0$')
    ax.set_ylabel('Bott Index')
    ax.set_title(f"Bott Index Phase Diagram for $r_0={r0}$ and $R={R}$")



def main():
    pass


if __name__ == "__main__":
    main()
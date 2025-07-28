import numpy as np
import scipy.linalg as spla
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from itertools import product
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import inspect
from joblib import Parallel, delayed
from tqdm_joblib import tqdm, tqdm_joblib


class DefectSquareLattice:
    """
    Class to generate a square lattice with various types of defects.
    """
    def __init__(self, side_length:int, defect_type:str, pbc:bool = True, frenkel_pair_index:int = 0, schottky_distance:int = 1, schottky_type:int = 0, doLargeDefect:bool = False, *args, **kwargs):
        """
        Initialize the defect square lattice.
        Parameters:
            side_length (int): Length of the sides of the square lattice.
            defect_type (str): Type of defect to introduce. Options are "none", "vacancy", "substitution", "interstitial", "frenkel_pair", or "schottky".
            pbc (bool): Whether to apply periodic boundary conditions.
            frenkel_pair_index (int): Index for the Frenkel pair defect, must be between 0 and 7.
            schottky_distance (int): Distance between Schottky defects (multiplication factor of sqrt(2))
            schottky_type (int): Type of Schottky defect, can be 0, 1, or 2. 
            doLargeDefect (bool): Whether to create a large defect. Keep at False, and use the self.LargeDefectLattice property to access the large defect lattice. This does nothing for "none", "schottky", and "frenkel_pair" defects.
        """

        # Set values within the class
        self._pbc = pbc
        self._side_length = side_length
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
        if self.defect_type == "interstitial" and self.side_length % 2 == 1:
            raise ValueError("Side length must be even for interstitial defect.")
        elif self.defect_type == "schottky" and (self.side_length + self.schottky_distance) % 2 != 1:
            raise ValueError("Side length + schottky distance must be odd for schottky defect. It is {}.".format(self.side_length + self.schottky_distance))
        elif self.defect_type not in ["interstitial", "schottky"] and self.side_length % 2 == 0:
            raise ValueError("Side length must be odd for non-interstitial defects.")

        # Generate the defect lattice based on the defect type.
        match self.defect_type:
            case "none":
                self._lattice = self._pristine_lattice.copy()
                self._defect_indices = None
            case "vacancy":
                self._lattice = self.compute_vacancy_lattice()
                self._defect_indices = None
            case "substitution":
                self._lattice = self._pristine_lattice.copy()
                if self.doLargeDefect:
                    self._defect_indices = []
                    center_idx = [self._side_length // 2, self._side_length // 2]
                    # Create a large defect by marking the center and its immediate neighbors
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            if abs(i) + abs(j) == 2:
                                continue
                            self._defect_indices.append(self.lattice[center_idx[0] + i, center_idx[1] + j])
                else:
                    self._defect_indices = [self.lattice[self._side_length // 2, self._side_length // 2]]
            case "interstitial":
                self._lattice = self.compute_interstitial_lattice()
                lattice_max = np.max(self.lattice)
                if self.doLargeDefect:
                    self._defect_indices = [lattice_max // 2, 
                                            lattice_max // 2 + 1, 
                                            lattice_max // 2 - 1,
                                            lattice_max // 2 + self.side_length + 2,
                                            lattice_max // 2 - self.side_length - 2]
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

        # Get the lattice coordinates. Adjust for the interstitial and frenkel pair defects (the lattices are generated with doubled coordinates).
        self._Y, self._X = np.where(self._lattice >= 0)[:]
        if self._defect_type in ["interstitial", "frenkel_pair"]:
            self._X = self._X.astype(float) / 2
            self._Y = self._Y.astype(float) / 2

        self._system_size = len(self.X)

        # Compute the distances and Wannier matrices
        self.compute_distances()
        self.compute_wannier_polar()

        # Create a large defect lattice if needed
        if not self._doLargeDefect:
            self.LargeDefectLattice = DefectSquareLattice(side_length, defect_type, pbc=pbc, doLargeDefect=True, frenkel_pair_index=self._frenkel_pair_index, schottky_distance=self.schottky_distance)

    # region Properties
    @property
    def side_length(self):
        return self._side_length
    @property
    def pbc(self):
        return self._pbc 
    @property
    def defect_type(self):
        return self._defect_type
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
    def dx(self):
        return self._dx
    @property
    def dy(self):
        return self._dy
    @property
    def lattice(self):
        if self.defect_type == "interstitial":
            print("Warning: Lattice coordinates must be halved for 'interstitial' defects.")
            print("Called from line:", inspect.currentframe().f_back.f_lineno)
        if self.defect_type == "frenkel_pair":
            print("Warning: Lattice coordinates must be halved for 'frenkel pair' defects.")
            print("Called from line:", inspect.currentframe().f_back.f_lineno)
        return self._lattice
    @property
    def pauli_matrices(self):
        return self._pauli_matrices
    @property
    def system_size(self):
        return self._system_size
    @property
    def Sx(self):
        return self._Sx
    @property
    def Sy(self):
        return self._Sy
    @property
    def Cx_plus_Cy(self):
        return self._Cx_plus_Cy
    @property
    def CxSy(self):
        return self._CxSy
    @property
    def SxCy(self):
        return self._SxCy
    @property
    def CxCy(self):
        return self._CxCy
    @property
    def I(self):
        return self._I
    @property
    def doLargeDefect(self):
        return self._doLargeDefect
    @property
    def schottky_distance(self):
        return self._schottky_distance

    # endregion

    # region Geometry
    def compute_lattice(self, *args, **kwargs):
        """        Generate a pristine square lattice with side length `side_length`."""
        return np.arange(self.side_length**2).reshape((self.side_length, self.side_length))

    def compute_vacancy_lattice(self, *args, **kwargs):
        """Generate a vacancy lattice with a single vacancy at the center. If `doLargeDefect` is True, it will create a large defect by marking the center and its immediate neighbors as vacancies."""
        lattice = self._pristine_lattice.copy()
    
        center_idx = self.side_length // 2
        lattice[center_idx, center_idx] = -1
        vacant_positions = []

        if self._doLargeDefect:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if abs(i) + abs(j) == 2:
                        continue
                    lattice[center_idx + i, center_idx + j] = -1
                    vacant_positions.append((center_idx + i, center_idx + j))

        self._vacant_positions = vacant_positions
        return lattice

    def compute_interstitial_lattice(self):
        """
        Generate an interstitial lattice by placing interstitial defects at the mean position of the pristine lattice.
        If `doLargeDefect` is True, it will create a large defect by marking the mean position and its immediate neighbors as interstitials.
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

    def compute_frenkel_pair_lattice(self, displacement_index:int):
        """
        Generate a Frenkel pair lattice with a defect at the center and an additional displacement based on the provided index.

        The vacancy is generated at the center, and in a square lattice, there are eight possible displacements. In general, we average over these (in later functions).
        Parameters:
            displacement_index (int): Index of the displacement to apply, must be between 0 and 7.
        """
        if displacement_index < 0 or displacement_index > 7:
            raise ValueError("Displacement index must be between 0 and 7.")
        
        center = np.array([self.side_length//2, self.side_length//2]).reshape(2, 1)
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

    def compute_schottky_lattice(self):
        """Generate a Schottky defect lattice with defects at specific positions based on the schottky_distance and schottky_type."""
        lattice = self._pristine_lattice.copy()

        midpoint_right = self.side_length // 2 + self._schottky_distance // 2
        midpoint_left = self.side_length // 2 - self._schottky_distance // 2 - 1

        if self._schottky_type == 0:
            # Generate a Schottky defect lattice with defects at the center. Here "O" is a typical site, "U" is an up parity site (down removed), and "D" is the reverse.
            # The defect section of the lattice then appears as: 
            #  O  | ... |  U
            # ... | ... | ...
            #  D  | ... |  O 
            up_parity_idx =   lattice[midpoint_right, midpoint_right]
            down_parity_idx = lattice[midpoint_left, midpoint_left]
            defect_idxs = [up_parity_idx, down_parity_idx]
            vacant_positions = [(midpoint_right, midpoint_right), (midpoint_left, midpoint_left)]
        elif self._schottky_type == 1:
            #  D  | ... |  U
            # ... | ... | ...
            #  D  | ... |  U
            up_parity_idx1 =   lattice[midpoint_right, midpoint_right]
            down_parity_idx1 = lattice[midpoint_left, midpoint_left]
            up_parity_idx2 =   lattice[midpoint_right - self._schottky_distance, midpoint_right]
            down_parity_idx2 = lattice[midpoint_left + self._schottky_distance, midpoint_left]
            defect_idxs = [up_parity_idx1, down_parity_idx1, up_parity_idx2, down_parity_idx2]
            vacant_positions = [(midpoint_right, midpoint_right), (midpoint_left, midpoint_left), (midpoint_right - self._schottky_distance, midpoint_right), (midpoint_left + self._schottky_distance, midpoint_left)]
        elif self._schottky_type == 2:
            #  D  | ... |  U
            # ... | ... | ...
            #  U  | ... |  D
            up_parity_idx1 =   lattice[midpoint_right, midpoint_right]
            up_parity_idx2 =   lattice[midpoint_left, midpoint_left]
            down_parity_idx1 = lattice[midpoint_right - self._schottky_distance, midpoint_right]
            down_parity_idx2 = lattice[midpoint_left + self._schottky_distance, midpoint_left]
            defect_idxs = [up_parity_idx1, down_parity_idx1, up_parity_idx2, down_parity_idx2]
            vacant_positions = [(midpoint_right, midpoint_right), (midpoint_left, midpoint_left), (midpoint_right - self._schottky_distance, midpoint_right), (midpoint_left + self._schottky_distance, midpoint_left)]

        self._vacant_positions = vacant_positions
        return lattice, defect_idxs
    
    def compute_distances(self, *args, **kwargs):
        """
        Compute the distances between all pairs of sites in the lattice.
        """
        # Displacement matrices. dx[i, j] is the x-displacement between site i and site j.
        dx = self.X - self.X[:, None]
        dy = self.Y - self.Y[:, None]

        if self.pbc:
            # Apply periodic boundary conditions
            multipliers = tuple(product([-1, 0, 1], repeat=2))
            shifts = [(i * self.side_length, j * self.side_length) for i, j in multipliers]

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
        self._dx, self._dy = dx, dy

    def compute_wannier_polar(self, *args, **kwargs):
        """Compute the Wannier polar matrices based on the displacements. While the construction is only necessary for defects containing an interstitial defect (interstitial, frenkek_pair), 
        it is computed for all defect types for consistency. Typical behavior is, of course, recovered in the case of no interstitial defect."""

        dx, dy = self.dx, self.dy
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

        self._Cx_plus_Cy = Cx_plus_Cy
        self._Sx = Sx
        self._Sy = Sy
        self._I = np.eye(Sx.shape[0], dtype=complex)

    # endregion

    # region Computation
    def compute_hamiltonian(self, M_background:float, M_substitution:float = None, t:float = 1.0, t0:float = 1.0):
        """Compute the Hamiltonian for the defect lattice based on the defect type and provided parameters.
        
        Parameters:
            M_background (float): Background mass for the Hamiltonian.
            M_substitution (float, optional): Mass for substitution defects. Required for "substitution", "interstitial", and "frenkel_pair" defects.
            t (float): Hopping parameter for d1, d2 terms.
            t0 (float): Hopping parameter for d3 term. 
        """

        if self.defect_type in ["substitution", "interstitial", "frenkel_pair"]:
            if M_substitution is None:
                raise ValueError("M_substitution must be provided for 'substitution', 'interstitial', or 'frenkel pair' defects.")
            onsite_mass = M_background * self.I
            onsite_mass[self.defect_indices, self.defect_indices] = M_substitution
        else:
            onsite_mass = M_background * self.I

        # Hopping vector terms
        d1 = t * self.Sx
        d2 = t * self.Sy

        if M_substitution is None: # If no substitution mass is provided, set at a positive value so np.sign() does not return NaN.
            M_substitution = 1.0
        d3 = onsite_mass + t0 * (self.Cx_plus_Cy) * np.sign(M_substitution) # Symmetrize (comment out and look at the LDOS for "interstitial", m_back = -1.0, m_sub = -2.5)
        
        # Construct the tight-binding Hamiltonian
        hamiltonian = np.kron(d1, self.pauli_matrices[0]) + np.kron(d2, self.pauli_matrices[1]) + np.kron(d3, self.pauli_matrices[2])

        if self.defect_type == "schottky":
            for i, defect_index in enumerate(self.defect_indices):
                if i % 2 == 0:
                    # Removal (vacancy) of down parity
                    hamiltonian[defect_index * 2 + 1, defect_index * 2 + 1] = 0.
                else:
                    # Removal (vacancy) of up parity
                    hamiltonian[defect_index * 2, defect_index * 2] = 0.
        return hamiltonian

    def compute_projector(self, hamiltonian:np.ndarray):
        """Compute the projector onto the lower band of the Hamiltonian."""
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2] # Lower band eigenvalues
        highest_lower_band = lower_band[-1] # Highest eigenvalue in the lower band

        D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j) # Projector diagonal matrix
        D_herm_conj = np.einsum('i,ij->ij', D, eigenvectors.conj().T)
        projector = eigenvectors @ D_herm_conj # Projector matrix
        return projector

    def compute_bott_index(self, projector:np.ndarray):
        """Compute the Bott index for the given projector."""

        # Repeated (two orbitals)
        X = np.repeat(self.X, 2)
        Y = np.repeat(self.Y, 2)
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
        bott_index = round(np.imag(np.sum(np.log(spla.eigvals(A)))) / (2 * np.pi))
        return bott_index

    def compute_LDOS(self, hamiltonian:np.ndarray, number_of_states:int = 2, *args, **kwargs):
        """Compute the Local Density of States (LDOS) for the given Hamiltonian.
        Parameters:
            hamiltonian (np.ndarray): The Hamiltonian matrix.
            number_of_states (int): Number of states to consider for the LDOS calculation. Defaults to 2.
        Returns:
            dict: A dictionary containing the LDOS, eigenvalues, gap, bandwidth, and indices of the states used for LDOS calculation.    
        """

        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        number_of_states += number_of_states % 2 # Ensure even number of states
        mid_index = len(eigenvalues) // 2
        lower_idxs = np.arange(len(eigenvalues))[:mid_index][-number_of_states // 2:]
        upper_idxs = np.arange(len(eigenvalues))[mid_index:][:number_of_states // 2]
        selected_indices = np.concatenate((lower_idxs, upper_idxs)) # Indices of the selected states to be used in LDOS

        LDOS = np.sum(np.abs(eigenvectors[:, selected_indices])**2, axis=1)
        LDOS = LDOS[0::2] + LDOS[1::2] # Sum the two parities
        LDOS = LDOS / np.sum(LDOS) # Normalize the LDOS
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

    def _compute_for_figure(self, m_background:float, m_substitution:float, number_of_states:float):
        """Helper function to compute the LDOS, eigenvalues, gap, and bott index for the defect lattice."""

        def _average_over_frenkel_pair():
            """Average over all Frenkel pair possibilities to compute the LDOS, eigenvalues, gap, and bott index."""
            all_LDOS = []
            all_x = []
            all_y = []
            all_eigenvalues = []
            all_gap = []
            all_bott = []
            for frenkel_pair_index in range(8):
                NewLattice = DefectSquareLattice(self.side_length, self.defect_type, pbc=self.pbc, frenkel_pair_index=frenkel_pair_index)
                hamiltonian = NewLattice.compute_hamiltonian(m_background, m_substitution)
                ldos_dict = NewLattice.compute_LDOS(hamiltonian, number_of_states=2)
                LDOS, eigenvalues, gap, bandwidth, ldos_idxs = ldos_dict["LDOS"], ldos_dict["eigenvalues"], ldos_dict["gap"], ldos_dict["bandwidth"], ldos_dict["ldos_idxs"]
                projector = NewLattice.compute_projector(hamiltonian)
                bott_index = NewLattice.compute_bott_index(projector)
                all_LDOS.append(LDOS)
                all_x.append(NewLattice.X)
                all_y.append(NewLattice.Y)
                all_eigenvalues.append(eigenvalues)
                all_gap.append(gap)
                all_bott.append(bott_index)
            all_LDOS = np.concatenate(all_LDOS, axis=0)
            all_x = np.concatenate(all_x, axis=0)
            all_y = np.concatenate(all_y, axis=0)
            all_gap = np.mean(all_gap)
            all_bott = np.mean(all_bott)
            coords = np.stack((all_x, all_y), axis=1)
            unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
            summed_LDOS = np.zeros(len(unique_coords))
            np.add.at(summed_LDOS, inverse_indices, all_LDOS)
            summed_LDOS /= np.sum(summed_LDOS)
            LDOS, gap = summed_LDOS, np.mean(all_gap)
            eigenvalues = np.mean(all_eigenvalues, axis=0)
            X, Y = unique_coords[:, 0], unique_coords[:, 1]
            bott_index = np.mean(all_bott)
            return LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs

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

    def _compute_for_figure_disorder(self, m_background:float, m_substitution:float, number_of_states:float, disorder_strength:float, n_iterations:int = 10, n_jobs:int = -1):
        """
        Compute the LDOS, eigenvalues, gap, and bott index for a disordered system. Uses parallel processing to average over multiple disorder realizations.
        Parameters:
            m_background (float): Background mass for the Hamiltonian.
            m_substitution (float): Mass for substitution defects.
            number_of_states (int): Number of states to consider for the LDOS calculation.
            disorder_strength (float): Strength of the disorder to be applied to the Hamiltonian.
            n_iterations (int): Number of disorder realizations to average over.
            n_jobs (int): Number of parallel jobs to use for computation.
        """
        hamiltonian = self.compute_hamiltonian(m_background, m_substitution)

        def _worker(hamiltonian, i):
            # Make a copy to avoid modifying the shared array in parallel jobs
            hamiltonian_local = hamiltonian.copy()
            disorder = np.random.uniform(-disorder_strength / 2, disorder_strength / 2, size=hamiltonian_local.shape[0])
            disorder -= np.mean(disorder)
            hamiltonian_local += np.diag(disorder)

            projector = self.compute_projector(hamiltonian_local)
            bott_index = self.compute_bott_index(projector)
            ldos_dict = self.compute_LDOS(hamiltonian_local, number_of_states)
            LDOS, eigenvalues, gap, bandwidth, ldos_idxs = ldos_dict["LDOS"], ldos_dict["eigenvalues"], ldos_dict["gap"], ldos_dict["bandwidth"], ldos_dict["ldos_idxs"]
            X, Y = self.X, self.Y
            return LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs
    
        with tqdm_joblib(tqdm(total=n_iterations, desc=f"{self.defect_type} : m_back={m_background}  : m_sub={m_substitution}")) as progress_bar:
            data = Parallel(n_jobs=-1)(delayed(_worker)(hamiltonian, i) for i in range(n_iterations))
        
        all_LDOS, all_eigenvalues, all_gap, all_bott_index, all_X, all_Y, all_ldos_idxs = zip(*data)
        LDOS = np.mean(all_LDOS, axis=0)
        eigenvalues = np.mean(all_eigenvalues, axis=0)
        gap = np.mean(all_gap)
        bott_index = np.mean(all_bott_index)
        X = all_X[0]
        Y = all_Y[0]
        return LDOS, eigenvalues, gap, bott_index, X, Y, all_ldos_idxs[0]

    # endregion

    # region Plotting
    def plot_distances(self, idx:int = None, cmap:str = "inferno", doLargeDefectFigure:bool = False, *args, **kwargs):
        """
        Plot the distances from a given site to all other sites in the lattice.
        """
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))

        if idx is None:
            idx = len(self.X) // 2

        if doLargeDefectFigure:
            dx, dy, d = self.LargeDefectLattice.dx, self.LargeDefectLattice.dy, np.sqrt(self.LargeDefectLattice.dx**2 + self.LargeDefectLattice.dy**2)
            X, Y = self.LargeDefectLattice.X, self.LargeDefectLattice.Y
        else:
            dx, dy, d = self.dx, self.dy, np.sqrt(self.dx**2 + self.dy**2)
            X, Y = self.X, self.Y

        distances = [dx, dy, d]
        labels = ["dx", "dy", "d"]
        for i, (distance, label) in enumerate(zip(distances, labels)):
            #distance = np.where(np.isclose(distance, 1/2), 1.0, 0.0)
            axs[i].set_title(label)
            axs[i].set_xlabel("X")
            axs[i].set_ylabel("Y")
            axs[i].scatter(X, Y, c=distance[idx], cmap=cmap, zorder=0, s=25)
            axs[i].scatter(self.X[idx], self.Y[idx], s=100, facecolors='none', edgecolors='red', zorder=1)
            axs[i].set_aspect('equal')

        cbar = fig.colorbar(axs[i].collections[0], ax=axs[i], orientation='vertical')
        cbar.set_label("Distance to site {}".format(idx), rotation=270, labelpad=15)
        plt.tight_layout()
        plt.show()

    def plot_defect_idxs(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        standard_site_size = 50
        large_site_size = 150

        ax.set_aspect('equal')
        ax.scatter(self.X, self.Y, s=standard_site_size, edgecolors='black', facecolors='black', linewidth=0.)

        for axes in [ax]:
            if self.defect_indices is None:
                for i, (x, y) in enumerate(self._vacant_positions):
                    if i == 2:
                        axes.scatter(x, y, s=large_site_size, edgecolors='red', facecolors='none', linewidth=1.5)
                    else:
                        axes.scatter(x, y, s=standard_site_size, edgecolors='red', facecolors='none', linewidth=1.5, alpha=1.0)
                    
            elif len(self.defect_indices) in [2, 4]:
                for x, y in self._vacant_positions:
                    axes.scatter(x, y, s=large_site_size, edgecolors='none', facecolors='white')
                for i, defect_idx in enumerate(self.defect_indices):
                    if i % 2 == 0:
                        c = "red"
                    elif i % 2 == 1:
                        c = "blue"
                    axes.scatter(self.X[defect_idx], self.Y[defect_idx], s=standard_site_size, facecolors = 'none', edgecolors = c, linewidth = 1.5)

            elif len(self.defect_indices) == 5:
                for defect_idx in self.defect_indices:
                    # Plot all defect indices, highlight the center one with a larger size
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
    
    def plot_spectrum_ldos(self, m_background_values:"list[float]" = [2.5, 1.0, -1.0, -2.5], 
                           m_substitution_values:"list[float]" = [2.5, 1.0, -1.0, -2.5], 
                           doLargeDefectFigure:bool = False, doDisorder:bool = False, 
                           n_iterations:int = 10, doInterpolation:bool = True):
        """
        Plot the spectrum and local density of states (LDOS) for the defect lattice.

        Parameters:
            m_background_values (list[float]): List of background mass values to consider.
            m_substitution_values (list[float]): List of substitution mass values to consider.
            doLargeDefectFigure (bool): If True, use the large defect lattice for plotting.
            doDisorder (bool): If True, compute the LDOS with disorder.
            n_iterations (int): Number of disorder realizations to average over if doDisorder is True.
            doInterpolation (bool): If True, interpolate the LDOS onto a finer grid for smoother visualization.

        This function creates a grid of subplots, where each subplot shows:
            - The energy spectrum of the Hamiltonian (with LDOS states highlighted)
            - The LDOS as a 3D surface plot or scatter plot (depending on settings)
        The function supports disorder averaging and interpolation for smoother visualization.

        For "none" and "vacancy" defect types, the substitution mass is set to None. 
        """
        def plot_spectrum_ax(spectrum_ax:plt.Axes, eigenvalues:np.ndarray, scatter_label:str, ldos_idxs:np.ndarray):
            # Plot the energy spectrum for the given eigenvalues
            # Highlight the states used for LDOS calculation in red
            x_values = np.arange(len(eigenvalues))
            idxs_mask = np.isin(x_values, ldos_idxs)
            # Plot all eigenvalues in black
            spectrum_ax.scatter(x_values[~idxs_mask], eigenvalues[~idxs_mask], s=25, color = 'black', zorder = 0)
            # Highlight selected LDOS states in red
            spectrum_ax.scatter(x_values[ idxs_mask], eigenvalues[ idxs_mask], s=25, color = 'red',   zorder = 1)
            spectrum_ax.set_xticks([])
            spectrum_ax.set_xlabel(r"$n$", fontsize=20)
            spectrum_ax.set_ylabel(r"$E_n$", fontsize=20)
            spectrum_ax.tick_params(axis='y', labelsize=20)
            # Annotate with gap, Bott index, and other info
            spectrum_ax.annotate(
                scatter_label,
                xy=(0.95, 0.025),
                xycoords='axes fraction',
                ha='right',
                va='bottom',
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.0)
            )

        def plot_ldos_ax(ax:plt.Axes, LDOS, X, Y, doScatter:bool = False):
            # Plot the LDOS for the lattice
            if doScatter:
                # If doScatter is True, plot LDOS as a scatter plot in an inset axis
                # Create an inset axis for the scatter plot
                inset_ax = inset_axes(
                    ax,
                    width="100%",  # width as a percentage of parent
                    height="100%",  # height as a percentage of parent
                    bbox_to_anchor=(0.1, 0.60, 0.375, 0.375),  # (x0, y0, width, height) in axes fraction
                    bbox_transform=ax.transAxes,
                    loc='upper left',
                    borderpad=0
                )
                # Create a colorbar axis for the inset
                cax = inset_axes(
                    inset_ax, 
                    width="5%",  # width as a percentage of parent
                    height="100%",  # height as a percentage of parent
                    bbox_to_anchor=(0.1, 0.3, 1, 0.6),  # (x0, y0, width, height) in axes fraction
                    bbox_transform=inset_ax.transAxes,
                    borderpad = 0.0
                )
                # Choose dot size based on lattice size
                if self.side_length <= 15:
                    dot_size = 25
                elif self.side_length <= 20:
                    dot_size = 10
                else:
                    dot_size = 5
                # Plot LDOS as colored dots
                scat = inset_ax.scatter(X, Y, c=LDOS, s=dot_size, cmap='jet')
                # Set axis ticks and labels for lattice coordinates
                inset_ax.set_xticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
                inset_ax.set_yticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
                inset_ax.set_xticklabels([str(int(np.min(X) + 1)), "$L_x$", str(int(np.max(X) + 1))], fontsize=14)
                inset_ax.set_yticklabels([str(int(np.min(X) + 1)), "$L_y$", str(int(np.max(X) + 1))], fontsize=14)
                inset_ax.set_aspect('equal')
                # Add colorbar for LDOS values
                cbar = fig.colorbar(inset_ax.collections[0], cax=cax)
                formatter = ticker.ScalarFormatter(useMathText = True)
                formatter.set_powerlimits((0,  0))
                formatter.set_scientific(True)
                formatter.format = "%.1f"
                cbar.formatter = formatter
                cbar.update_ticks()
                cbar.ax.yaxis.offsetText.set_position((10., 1.05))
                cbar.ax.yaxis.offsetText.set_fontsize(14)
                cbar.ax.tick_params(labelsize=14)
                # Set background and z-order for inset
                inset_ax.set_facecolor((1, 1, 1, 0.8))
                inset_ax.set_zorder(10)
                return inset_ax
            else:
                # If doScatter is False, plot LDOS as a 3D surface plot
                if doInterpolation:
                    # Interpolate LDOS onto a finer grid for smoother visualization
                    grid_res = 201  # resolution of the interpolation grid
                    xi = np.linspace(np.min(X), np.max(X), grid_res)
                    yi = np.linspace(np.min(Y), np.max(Y), grid_res)
                    XI, YI = np.meshgrid(xi, yi)
                    points = np.column_stack((X, Y))
                    LDOS_interp = griddata(points, LDOS, (XI, YI), method='linear', fill_value=0)
                    # Rescale interpolated LDOS to match original min/max
                    ldos_min, ldos_max = np.min(LDOS), np.max(LDOS)
                    interp_min, interp_max = np.min(LDOS_interp), np.max(LDOS_interp)
                    if interp_max > interp_min and ldos_max > ldos_min:
                        LDOS_interp = (LDOS_interp - interp_min) / (interp_max - interp_min)
                        LDOS_interp = LDOS_interp * (ldos_max - ldos_min) + ldos_min
                    X, Y, LDOS = XI.ravel(), YI.ravel(), LDOS_interp.ravel()
                # Create a new 3D axis for the surface plot
                box = ax.get_position()
                surf_ax = fig.add_axes([box.x0, box.y0 + box.height * 0.55, box.width * 0.5, box.height * 0.5], projection='3d')
                # Plot LDOS as a 3D surface
                surf = surf_ax.plot_trisurf(X, Y, LDOS, cmap='inferno', linewidth=0.2, antialiased=False)
                # Set axis ticks and labels for lattice coordinates
                surf_ax.set_xticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
                surf_ax.set_yticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
                surf_ax.set_xticklabels([str(int(np.min(X) + 1)), "$L_x$", str(int(np.max(X) + 1))], fontsize=14)
                surf_ax.set_yticklabels([str(int(np.min(X) + 1)), "$L_y$", str(int(np.max(X) + 1))], fontsize=14)
                surf.set_clim(vmin=0)
                # Hide z-axis labels and grid for cleaner look
                surf_ax.set_zticklabels([])
                surf_ax.set_zlabel("")
                surf_ax.set_facecolor((1, 1, 1, 0))
                surf_ax.grid(False)

                # Optionally remove pane color for full transparency
                #surf_ax.xaxis.set_pane_color((1, 1, 1, 0))
                #surf_ax.yaxis.set_pane_color((1, 1, 1, 0))
                #surf_ax.zaxis.set_pane_color((1, 1, 1, 0))

                # Add colorbar for LDOS values
                cax = inset_axes(
                    surf_ax, 
                    width="7.5%",  # width as a percentage of parent
                    height="100%",  # height as a percentage of parent
                    bbox_to_anchor=(0.1, 0.425, 1, 0.4),  # (x0, y0, width, height) in axes fraction
                    bbox_transform=surf_ax.transAxes,
                    borderpad = 0.0
                )
                cbar = fig.colorbar(surf_ax.collections[0], cax=cax)
                formatter = ticker.ScalarFormatter(useMathText = True)
                formatter.set_powerlimits((0,  0))
                formatter.set_scientific(True)
                formatter.format = "%.1f"
                cbar.formatter = formatter
                cbar.update_ticks()
                cbar.ax.yaxis.offsetText.set_position((5., 1.0))
                cbar.ax.yaxis.offsetText.set_fontsize(14)
                cbar.ax.tick_params(labelsize=14)
                return surf_ax
        
        if self.defect_type in ["none", "vacancy"]:
            m_substitution_values = [None] if doLargeDefectFigure is False else [None] * 2
            n_cols, n_rows = len(m_background_values), len(m_substitution_values)
        elif self.defect_type == "schottky":
            m_substitution_values = [None] * 3
            n_cols, n_rows = len(m_background_values), len(m_substitution_values)
        else:
            n_cols, n_rows = len(m_background_values), len(m_substitution_values) - 1

        scale = 6
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(scale * n_cols, scale * n_rows))
        plt.subplots_adjust(wspace = 0.4)

        if self.defect_type not in ["none", "vacancy", "schottky"]:
            plt.subplots_adjust(top=0.9)
            set_labels = [f"({lab})" for lab in "abcdefghijklmnopqrstuvwxyz"[:len(m_background_values)]]
            for i, m_background in enumerate(m_background_values):
                label_xpos = axs[0, i].get_position().x0 + axs[0, i].get_position().width / 2
                label_ypos = axs[0, i].get_position().y1 + 0.045
                if n_rows == 1:
                    fig.text(label_xpos, label_ypos, set_labels[i], fontsize=36, ha='center')
                else:
                    fig.text(label_xpos, label_ypos, set_labels[i], fontsize=36, ha='center')

        if n_rows == 1:
            axs = np.array([axs])

        for j, m_background in enumerate(m_background_values):
            good_m_sub_vals = np.array(m_substitution_values)[np.array(m_substitution_values) != m_background]
            for i, m_substitution in enumerate(good_m_sub_vals):
                if doDisorder:
                    None_Lattice = DefectSquareLattice(self.side_length, "none", pbc=self.pbc)
                    _, _, gap_none, _, _, _, _ = None_Lattice._compute_for_figure(m_background, m_substitution, 2)
                    disorder_strength = gap_none * 0.25

                if self.defect_type == "schottky":
                    Lat = DefectSquareLattice(self.side_length, "schottky", pbc=self.pbc, schottky_type=i, schottky_distance=self._schottky_distance)
                    if doDisorder:
                        d_LDOS, d_eigenvalues, d_gap, d_bott_index, d_X, d_Y, d_ldos_idxs = Lat._compute_for_figure(m_background, m_substitution, 2, disorder_strength, n_iterations)
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = Lat._compute_for_figure(m_background, m_substitution, 2)
                elif i == 1 and doLargeDefectFigure and self.defect_type in ["vacancy"]:
                    if doDisorder:
                        d_LDOS, d_eigenvalues, d_gap, d_bott_index, d_X, d_Y, d_ldos_idxs = self.LargeDefectLattice._compute_for_figure_disorder(m_background, m_substitution, 2, disorder_strength, n_iterations)
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution, 2)
                elif doLargeDefectFigure and self.defect_type not in ["vacancy"]:
                    if doDisorder:
                        d_LDOS, d_eigenvalues, d_gap, d_bott_index, d_X, d_Y, d_ldos_idxs = self.LargeDefectLattice._compute_for_figure_disorder(m_background, m_substitution, 2, disorder_strength, n_iterations)
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution, 2)
                else:
                    if doDisorder:
                        d_LDOS, d_eigenvalues, d_gap, d_bott_index, d_X, d_Y, d_ldos_idxs = self._compute_for_figure_disorder(m_background, m_substitution, 2, disorder_strength, n_iterations)
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self._compute_for_figure(m_background, m_substitution, 2)
                
                if doDisorder:
                    LDOS = d_LDOS
                    eigenvalues = d_eigenvalues
                    bott_index = d_bott_index
                    X = d_X
                    Y = d_Y
                    ldos_idxs = d_ldos_idxs
                    undisordered_gap = gap
                    disordered_gap = d_gap

                ax = axs[i, j]

                if self.defect_type in ["none", "vacancy", "schottky"]:
                    param_name = f"$m_0={m_background}$"
                elif self.defect_type in ["substitution"]:
                    param_name = f"$m_0^{{\\text{{sub}}}}={m_substitution}$"
                else:
                    param_name = f"$m_0^{{\\text{{int}}}}={m_substitution}$"

                if doDisorder:
                    gap_label = f"Gap = {disordered_gap:.2f}"
                    w_label = f"\n$W = {disorder_strength:.2f}$"
                    perc_label = f"\nGap$\\Delta={(disordered_gap - undisordered_gap) / undisordered_gap * 100:.2f}\\%$"
                    label = gap_label + "\n" + param_name + f"\n$BI = {bott_index}$" + w_label + perc_label
                else:
                    label = f"Gap = {gap:.2f}" + "\n" + param_name + f"\n$BI = {bott_index}$"

                plot_spectrum_ax(ax, eigenvalues, label, ldos_idxs)
                surf_ax = plot_ldos_ax(ax, LDOS, X, Y)
        return fig, axs
    
    # endregion

    # region Parallel Computation

    @staticmethod
    def generic_multiprocessing(func:callable, parameter_values:tuple, n_jobs:int = -1, progress_title:str = "Progress bar without a title.", doProgressBar:bool = True, *args, **kwargs):
        """
        Generic function to parallelize the execution of a function over a set of parameters.
        Parameters:
            func (callable): The function to be parallelized.
            parameter_values (tuple): A tuple of parameter values to be passed to the function.
            n_jobs (int): The number of jobs to run in parallel. -1 means using all available cores.
            progress_title (str): Title for the progress bar.
            doProgressBar (bool): Whether to show a progress bar.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.
        """
        if doProgressBar:
            with tqdm_joblib(tqdm(total=len(parameter_values), desc=progress_title)) as progress_bar:
                data = Parallel(n_jobs=n_jobs)(delayed(func)(params, *args, **kwargs) for params in parameter_values)
        else:
            data = Parallel(n_jobs=n_jobs)(delayed(func)(params, *args, **kwargs) for params in parameter_values)
        return data
    
    def _compute_bott_from_parameters(self, parameters:tuple):
        """
        Helper function to compute the Bott index from a tuple of parameters.
        Parameters:
            parameters (tuple): A tuple containing the background mass and substitution mass.
        """
        m_background, m_substitution = parameters
        if m_substitution is None:
            m_substitution = m_background
        hamiltonian = self.compute_hamiltonian(m_background, m_substitution)
        projector = self.compute_projector(hamiltonian)
        bott_index = self.compute_bott_index(projector)
        return [m_background, m_substitution, bott_index]

    def plot_bott_phase_diagram(self, m_background_values:"list[float]", m_substitution_values:"list[float] | None" = None, num_jobs:int = -1):
        """
        Plots the Bott phase diagram for the given background and substitution mass values.
        Parameters:
            m_background_values (list[float]): List of background mass values.
            m_substitution_values (list[float] | None): List of substitution mass values. If None, only the background values are used.
            num_jobs (int): Number of parallel jobs to run. -1 means using all available cores.
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
            ax (matplotlib.axes.Axes): The axes object containing the plot.
        """
        parameters = tuple(product(m_background_values, m_substitution_values)) if m_substitution_values is not None else tuple(product(m_background_values, [None]))
        data = self.generic_multiprocessing(
            func = self._compute_bott_from_parameters, 
            parameter_values = parameters, 
            n_jobs = num_jobs, 
            progress_title = "Computing Bott Index", 
            doProgressBar = True
        )
        data = np.array(data, dtype=float)
        data = data[np.lexsort((data[:, 0], data[:, 1]))]

        fig, ax = plt.subplots(figsize=(10, 6))
        mb_vals, ms_vals, b_vals = data[:, 0], data[:, 1], data[:, 2]

        if m_substitution_values is not None:
            for i in range(len(m_substitution_values)):
                ax.scatter(mb_vals[ms_vals == m_substitution_values[i]], b_vals[ms_vals == m_substitution_values[i]], 
                        s=25, label=f"$m_0^{{\\text{{sub}}}}={m_substitution_values[i]}$", color=f"C{i}", alpha=0.5)
        else:
            ax.scatter(mb_vals, b_vals, s=25, color='black')
        
        ax.set_xlabel(r"$m_0$", fontsize=20)
        ax.set_ylabel("Bott Index", fontsize=20)
        ax.set_yticks([-1, 0, 1])
        xticks = [-4, -2, 0, 2, 4]
        ax.set_xticks(xticks)

        phase_labels = ["Trivial", "$\\Gamma$ Phase", "$M$ Phase", "Trivial"]
        cmap = plt.get_cmap("viridis")

        for i in range(len(xticks) - 1):
            #ax.axvspan(xticks[i], xticks[i+1], color=colors[i], alpha=0.15)
            ax.text((xticks[i] + xticks[i+1]) / 2, 0.8, phase_labels[i], fontsize=20, ha='center', va='center', color='black')

        ax.tick_params(axis='both', labelsize=18)
        for tick in [-4, -2, 0, 2, 4]:
            ax.axvline(tick, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        return fig, ax
    # endregion


def generate_figures(lcm_or_ldos:str, defect_types: list = ["none", "vacancy", "schottky", "substitution", "interstitial", "frenkel_pair"], base_side_length: int = 24, schottky_distance: int = 3, 
                     directory:str = ".", doDisorder:bool = False, n_iterations:int = 10):
    """
    Generate figures for the specified defect types and save them to the given directory.
    Parameters:
        lcm_or_ldos (str): Specify whether to plot 'lcm' or 'ldos'. 'lcm' is the Local Chern marker, 'ldos' is the local density of states.
        defect_types (list): List of defect types to consider.
        base_side_length (int): The base side length of the square lattice.
        schottky_distance (int): The distance for the Schottky defect.
        directory (str): The directory where the figures will be saved.
        doDisorder (bool): Whether to include disorder in the calculations.
        n_iterations (int): The number of iterations for disorder calculations.
    """
    for defect_type in defect_types:
        side_length = base_side_length
        if defect_type not in ["interstitial", "schottky"]:
            side_length = base_side_length + 1

        for dLDF in [True, False]:
            if defect_type in ["none", "frenkel_pair", "schottky"] and dLDF:
                continue
            if defect_type == "vacancy" and not dLDF:
                continue
            Lattice = DefectSquareLattice(side_length, defect_type, schottky_distance = schottky_distance)

            if lcm_or_ldos == "lcm":
                Lattice.plot_lcm(doLargeDefectFigure=dLDF)
            elif lcm_or_ldos == "ldos":
                Lattice.plot_spectrum_ldos(dLDF, doDisorder, n_iterations)
            else:
                raise ValueError("lcm_or_ldos must be 'lcm' or 'ldos'")
            
            if defect_type == "none":
                title = "SL"
            else:
                title = defect_type
            
            if doDisorder:
                title += "_disorder"

            if dLDF and defect_type not in ["vacancy", "schottky", "none"]:
                title = "large_" + title
            
            fname = directory + f"{title}_" + lcm_or_ldos.upper() + ".png"
            plt.savefig(fname)
            print(f"Saved figure for {defect_type} with dLDF={dLDF} in {fname}")


if __name__ == "__main__":
    doPlotDistances = 0
    doPlotDefects   = 0
    plotLDOS = 1
    batchFigureGeneration = 0

    useLargeDefect = False
    Lattice = DefectSquareLattice(side_length = 25, defect_type = "vacancy", pbc = True)
    
    # Plot the distances between the provided site and other sites in the lattice
    if doPlotDistances:
        site_index = Lattice.system_size // 2 - 11
        Lattice.plot_distances(site_index, cmap='jet', doLargeDefectFigure = useLargeDefect)
        plt.show()

    # Plot the lattice with defects highlighted
    if doPlotDefects:
        if useLargeDefect:
            Lattice.LargeDefectLattice.plot_defect_idxs()
        else:
            Lattice.plot_defect_idxs()
        plt.show()

    # Plot the LDOS spectrum for the defect lattice
    if plotLDOS:
        # For "none", "schottky", "frenkel_pair", doLargeDefectFigure does nothing.
        # For "vacancy", it plots the large defect lattice as the second row and the small defect lattice as the first row.
        # For "substitution", "interstitial", all plots pertain to the large defect.

        # For "schottky", each row corresponds to a different schottky type as described in the  
        Lattice.plot_spectrum_ldos(doLargeDefectFigure=useLargeDefect, doDisorder=False)
        plt.show()
    
    if batchFigureGeneration:
        generate_figures("ldos", ["none", "vacancy", "schottky", "substitution", "interstitial", "frenkel_pair"], base_side_length=14)





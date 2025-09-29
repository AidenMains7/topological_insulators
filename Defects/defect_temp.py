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


class DefectLattice:
    def __init__(self, Lx:int, Ly:int, defect_type:str, pbc:bool = True, 
                 schottky_distance:int = 1, schottky_type:str = "I",
                 frenkel_pair_displacement_index:int = 0):
        self._Lx = Lx
        self._Ly = Ly
        self._defect_type = defect_type
        self._pbc = pbc
        self._schottky_distance = schottky_distance
        self._schottky_type = schottky_type
        self._frenkel_pair_displacement_index = frenkel_pair_displacement_index

        assert defect_type in ["none", "vacancy", "schottky", "substitution", "interstitial", "frenkel_pair", "large_vacancy", "large_substitution", "large_interstitial"], \
            f"Invalid defect type: {defect_type}."
        if defect_type == "none":
            pass
        elif defect_type in ["interstitial", "schottky", "large_interstitial"]:
            assert Lx % 2 == 0 and Ly % 2 == 0, "Lx and Ly must be even for interstitial and schottky defects."
        else:
            pass
            # assert Lx % 2 == 1 and Ly % 2 == 1, "Lx and Ly must be odd for vacancy, substitution, and frenkel pair defects."

        self._pristine_lattice = self.generate_pristine_lattice()
        self._vacancy_idxs = [] 
        self._defect_idxs =  []

        match defect_type:
            case "none":
                self._lattice = self._pristine_lattice
            case "vacancy":
                self._lattice = self.generate_vacancy_lattice()
            case "large_vacancy":
                self._lattice = self.generate_large_vacancy_lattice()
            case "schottky":
                self._lattice = self.generate_schottky_lattice()
            case "substitution":
                self._lattice = self.generate_substitution_lattice()
            case "large_substitution":
                self._lattice = self.generate_large_substitution_lattice()
            case "interstitial":
                self._lattice = self.generate_interstitial_lattice()
            case "large_interstitial":
                self._lattice = self.generate_large_interstitial_lattice()
            case "frenkel_pair":
                self._lattice = self.generate_frenkel_pair_lattice(self.frenkel_pair_displacement_index)
            case _:
                raise ValueError(f"Invalid defect type: {defect_type}.")

        self._Y, self._X = np.where(self.lattice >= 0)

        if defect_type in ["interstitial", "frenkel_pair", "large_interstitial"]:
            self._X = self._X / 2
            self._Y = self._Y / 2

        self._dx, self._dy = self.compute_distances()
        if defect_type in ["none", "vacancy", "schottky", "substitution"]:
            self._Sx, self._Sy, self._Cx_plus_Cy, self._I = self.compute_wannier_fourier()
        else:
            self._Sx, self._Sy, self._Cx_plus_Cy, self._I = self.compute_wannier_symmetry()

    # region Properties
    @property
    def Lx(self):
        return self._Lx
    @property
    def Ly(self):
        return self._Ly
    @property
    def defect_type(self):
        return self._defect_type
    @property
    def pbc(self):
        return self._pbc
    @property
    def lattice(self):
        if self.defect_type in ["interstitial", "large_interstitial"]:
            print("Warning: Lattice coordinates must be halved for 'interstitial' defects.")
            print("Called from line:", inspect.currentframe().f_back.f_lineno)
        if self.defect_type == "frenkel_pair":
            print("Warning: Lattice coordinates must be halved for 'frenkel pair' defects.")
            print("Called from line:", inspect.currentframe().f_back.f_lineno)
        return self._lattice
    @property
    def pristine_lattice(self):
        return self._pristine_lattice
    @property
    def vacancy_idxs(self):
        return self._vacancy_idxs
    @property
    def defect_idxs(self):
        return self._defect_idxs
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
    def Cx_plus_Cy(self):
        return self._Cx_plus_Cy
    @property
    def Sx(self):
        return self._Sx
    @property
    def Sy(self):
        return self._Sy
    @property
    def I(self):
        return self._I
    @property
    def schottky_distance(self):
        return self._schottky_distance
    @property
    def site_mask(self):
        if hasattr(self, '_site_mask'):
            return self._site_mask
        else:
            raise AttributeError("Site mask not initialized")
    @property
    def parity_site_mask(self):
        if hasattr(self, '_parity_site_mask'):
            return self._parity_site_mask
        else:
            raise AttributeError("Parity site mask not initialized")
    @property
    def frenkel_pair_displacement_index(self):
        return self._frenkel_pair_displacement_index
    # endregion

    # region Lattice Generation Methods
    def generate_pristine_lattice(self):
        return np.arange(self.Lx * self.Ly).reshape((self.Ly, self.Lx))

    def generate_vacancy_lattice(self):
        lattice = self.generate_pristine_lattice()
        x_center, y_center = self.Lx // 2, self.Ly // 2
        self._vacancy_idxs.append(lattice[y_center, x_center])

        site_mask = np.full(lattice.size, True)
        site_mask[self._vacancy_idxs] = False
        self._site_mask = site_mask
        self._parity_site_mask = np.repeat(site_mask, 2)
        return lattice
    
    def generate_large_vacancy_lattice(self):
        lattice = self.generate_pristine_lattice()
        x_center, y_center = self.Lx // 2, self.Ly // 2

        # Create a vacancy at the center of the lattice
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if abs(i) + abs(j) == 2:
                    continue
                self._vacancy_idxs.append(lattice[y_center + i, x_center + j])

        site_mask = np.full(lattice.size, True)
        site_mask[self._vacancy_idxs] = False
        self._site_mask = site_mask
        self._parity_site_mask = np.repeat(site_mask, 2)
        return lattice

    def generate_schottky_lattice(self):
        schottky_distance = self._schottky_distance
        schottky_type = self._schottky_type

        assert schottky_type in ["I", "II", "III"], "Invalid schottky type. Must be 'I', 'II', or 'III'."
        assert ((self.Lx + schottky_distance) % 2 == 1) and ((self.Ly + schottky_distance) % 2 == 1), "Side length + schottky distance must be odd for schottky defects."
       # assert (schottky_distance < self.Lx // 2) and (schottky_distance < self.Ly // 2), "Schottky distance must be less than half the side length."

        lattice = self._pristine_lattice.copy()
        x_midpoint_right = self.Lx // 2 + schottky_distance // 2
        x_midpoint_left = self.Lx // 2 - schottky_distance // 2 - 1

        y_midpoint_right = self.Ly // 2 + schottky_distance // 2
        y_midpoint_left = self.Ly // 2 - schottky_distance // 2 - 1

        if schottky_type == "I":
            # Generate a Schottky defect lattice with defects at the center. Here "O" is a typical site, "U" is an up parity site (down removed), and "D" is the reverse.
            # The defect section of the lattice then appears as: 
            #  O  | ... |  U
            # ... | ... | ...
            #  D  | ... |  O 
            up_parity_idx =   lattice[y_midpoint_right, x_midpoint_right]
            down_parity_idx = lattice[y_midpoint_left, x_midpoint_left]
            vacancy_idxs = (up_parity_idx, down_parity_idx)
            vacant_positions = [(x_midpoint_right, y_midpoint_right), (x_midpoint_left, y_midpoint_left)]
        elif schottky_type == "II":
            #  D2 | ... |  U1
            # ... | ... | ...
            #  D1 | ... |  U2
            up_parity_idx1 =   lattice[y_midpoint_right, x_midpoint_right]
            down_parity_idx1 = lattice[y_midpoint_left,  x_midpoint_left]
            up_parity_idx2 =   lattice[y_midpoint_right - schottky_distance, x_midpoint_right]
            down_parity_idx2 = lattice[y_midpoint_left + schottky_distance,  x_midpoint_left]
            vacancy_idxs = (up_parity_idx1, down_parity_idx1, up_parity_idx2, down_parity_idx2)
            vacant_positions = [(x_midpoint_right, y_midpoint_right), (x_midpoint_left, y_midpoint_left), (x_midpoint_right, y_midpoint_right - schottky_distance), (x_midpoint_left, y_midpoint_left + schottky_distance)]
        elif schottky_type == "III":
            #  D1 | ... |  U1
            # ... | ... | ...
            #  U2 | ... |  D2
            up_parity_idx1 =   lattice[y_midpoint_right, x_midpoint_right]
            up_parity_idx2 =   lattice[y_midpoint_left, x_midpoint_left]
            down_parity_idx1 = lattice[y_midpoint_right - schottky_distance, x_midpoint_right]
            down_parity_idx2 = lattice[y_midpoint_left + schottky_distance,  x_midpoint_left]
            vacancy_idxs = (up_parity_idx1, down_parity_idx1, up_parity_idx2, down_parity_idx2)
            vacant_positions = [(x_midpoint_right, y_midpoint_right), (x_midpoint_left, y_midpoint_left), (x_midpoint_right, y_midpoint_right - schottky_distance), (x_midpoint_left, y_midpoint_left + schottky_distance)]

        self._vacancy_idxs = vacancy_idxs
        parity_site_mask = np.repeat(np.full(lattice.size, True), 2)
        for i, idx in enumerate(vacancy_idxs):
            parity_site_mask[2 * idx + (i + 1) % 2] = False
        self._parity_site_mask = parity_site_mask
        return lattice

    def generate_substitution_lattice(self):
        lattice = self.generate_pristine_lattice()

        # Create a substitution at the center of the lattice
        x_center, y_center = self.Lx // 2, self.Ly // 2
        self._defect_idxs.append(lattice[y_center, x_center])
        return lattice
    
    def generate_large_substitution_lattice(self):
        lattice = self.generate_pristine_lattice()

        # Create five substitutions at the center of the lattice
        x_center, y_center = self.Lx // 2, self.Ly // 2
        
        n = 1
        for i in np.arange(-n, n+1):
            for j in np.arange(-n, n+1):
                if abs(i) + abs(j) >= n+1:
                    continue
                self._defect_idxs.append(lattice[y_center + i, x_center + j])
        return lattice

    def generate_interstitial_lattice(self):
        Y, X = np.where(self._pristine_lattice >= 0)
        x_center, y_center = np.mean(X), np.mean(Y)
        interstitial_position = np.array([[x_center], [y_center]])

        coordinates = np.array([X, Y])
        coordinates = np.concatenate((coordinates, interstitial_position), axis=1)
        coordinates = np.unique(np.round(coordinates * 2).astype(int), axis=1)
        coordinates = coordinates[:, np.lexsort((coordinates[0], coordinates[1]))]

        interstitial_lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
        interstitial_lattice[coordinates[1], coordinates[0]] = np.arange(len(coordinates[0]))
    
        Y, X = np.where(interstitial_lattice >= 0)
        int_idxs = np.arange(len(coordinates[0]))[(X % 2) == 1]
        self._defect_idxs = int_idxs
        return interstitial_lattice

    def generate_large_interstitial_lattice(self):
        Y, X = np.where(self._pristine_lattice >= 0)
        x_mean = np.round(np.mean(X), 1)
        y_mean = np.round(np.mean(Y), 1)

        coordinates = np.array([X, Y])
        for i in np.arange(-1, 2):
            for j in np.arange(-1, 2):
                if abs(i) + abs(j) >= 2:
                    continue
                coordinates = np.concatenate((coordinates, np.array([[x_mean + i], [y_mean + j]])), axis=1)        
        coordinates = np.unique(np.round(coordinates * 2).astype(int), axis=1)
        coordinates = coordinates[:, np.lexsort((coordinates[0], coordinates[1]))]

        interstitial_lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
        interstitial_lattice[coordinates[1], coordinates[0]] = np.arange(len(coordinates[0]))

        Y, X = np.where(interstitial_lattice >= 0)
        int_idxs = np.arange(len(coordinates[0]))[(X % 2) == 1]
        self._defect_idxs = int_idxs

        return interstitial_lattice

    def generate_frenkel_pair_lattice(self, displacement_index:int = 0):
        if displacement_index < 0 or displacement_index > 7:
            raise ValueError("Displacement index must be between 0 and 7.")

        x_center, y_center = self.Lx // 2, self.Ly // 2

        lattice = self._pristine_lattice.copy()

        Y, X = np.where(lattice >= 0)[:]
        coordinates = (np.array([X, Y]) * 2).astype(int)

        values = [-3, -1, 1, 3]
        displacements = np.array(list(product(values, repeat=2)))
        good_displacements = []
        for d in displacements:
            if np.abs(d[0]) != np.abs(d[1]):
                good_displacements.append(d.reshape(2,1))

        displacements = np.array(good_displacements)
        displacement_location = np.array([[x_center], [y_center]]) * 2 + displacements[displacement_index]
        coordinates = np.concatenate((coordinates, displacement_location), axis=1)

        new_lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
        new_lattice[coordinates[1], coordinates[0]] = np.arange(coordinates.size // 2)

        new_y, new_x = np.where(new_lattice >= 0)[:]
        self._defect_idxs = [np.argwhere(new_x % 2).flatten()[0]]
        self._fp_displacements = displacements / 2


        self.vacancy_idxs.append(new_lattice[y_center, x_center])
        site_mask = np.full(lattice.size+1, True)
        site_mask[self._vacancy_idxs] = False
        self._site_mask = site_mask
        self._parity_site_mask = np.repeat(site_mask, 2)
        return new_lattice
    # endregion

    # region Lattice Geometry Methods
    def compute_distances(self):
        """
        Compute the distances between all pairs of sites in the lattice.
        """
        # Displacement matrices. dx[i, j] is the x-displacement between site i and site j.
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
            minimal_hop = np.argmin(distances, axis = -1)
            i_idxs, j_idxs = np.indices(minimal_hop.shape)

            dx = x_shifted[i_idxs, j_idxs, minimal_hop]
            dy = y_shifted[i_idxs, j_idxs, minimal_hop]
        return dx, dy

    def compute_wannier_symmetry(self):
        dx, dy = self.dx, self.dy
        theta = np.arctan2(dy, dx)  
        dr = np.sqrt(dx ** 2 + dy ** 2)

        # Create masks for different types of hopping. 
        distance_mask = ((dr <= 1 + 1e-6) & (dr > 1e-6)) # Mask for distances close to 1
        principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & distance_mask 
        diagonal_mask  = ((np.isclose(np.abs(dx), np.abs(dy), atol=1e-4)) & (dx != 0)) & distance_mask
        hopping_mask = principal_mask | diagonal_mask
    
        # Compute the Wannier matrices based on the masks
        cos_dphi = np.where(hopping_mask, np.cos(theta), 0. + 0.j)
        sin_dphi = np.where(hopping_mask, np.sin(theta), 0. + 0.j)
        amplitude = np.where(hopping_mask, np.exp(1. - dr), 0. + 0.j)

        # Momentum space matrices constructed from the real-space displacements based on arxiv.org/abs/2407.13767
        Cx_plus_Cy = amplitude / 2 
        Sx = 1j * cos_dphi * amplitude / 2
        Sy = 1j * sin_dphi * amplitude / 2

        return Sx, Sy, Cx_plus_Cy, np.eye(Sx.shape[0], dtype=complex)

    def compute_wannier_fourier(self):
        dx, dy = self.dx, self.dy

        Sx = np.zeros(dx.shape, dtype=complex)
        Sy = np.zeros(dy.shape, dtype=complex)
        Cx = np.zeros(dx.shape, dtype=complex)
        Cy = np.zeros(dy.shape, dtype=complex)
        I = np.eye(dx.shape[0], dtype=complex)

        x_hop = (dx == 1) & (dy == 0)
        y_hop = (dx == 0) & (dy == 1)

        Sx[x_hop] = 1j / 2
        Sy[y_hop] = 1j / 2
        Cx[x_hop] = 1 / 2
        Cy[y_hop] = 1 / 2

        Cx_plus_Cy = Cx + Cy
        Sx += Sx.conj().T
        Sy += Sy.conj().T
        Cx_plus_Cy += Cx_plus_Cy.conj().T

        return Sx, Sy, Cx_plus_Cy, I

    # endregion

    # region Hamiltonian Methods
    def compute_hamiltonian(self, m_back:float, m_sub:float = None, t:float = 1.0, t0:float = 1.0, hamiltonian_method:str = "site_elimination"):
        """
        Methods are described in arxiv:2407.13767.
        For "none" and "substitution" defects, we use typical hamiltonian construction.
        For "vacancy" and "schottky" defects, we use renormalization, site elimination, or symmetry methods of construction.
        For "interstitial" and "frenkel_pair" defects, we use the symmetry method for hamiltonian construction. 
        
        """

        onsite_mass = self.I * m_back


        if self.defect_type in ["substitution", "interstitial", "frenkel_pair", "large_substitution", "large_interstitial"]:
            assert m_sub != None, "m_sub must be provided for substitution, interstitial, and frenkel pair defects."
            for idx in self.defect_idxs:
                onsite_mass[idx, idx] = m_sub

        d1 = t * self.Sx
        d2 = t * self.Sy
        d3 = onsite_mass + t0 * self.Cx_plus_Cy

        pauli_x = np.array([[0, 1], [1, 0]],    dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]],   dtype=complex)

        hamiltonian = np.kron(d1, pauli_x) + np.kron(d2, pauli_y) + np.kron(d3, pauli_z)

        if self.defect_type in ["vacancy", "schottky", "frenkel_pair", "large_vacancy"]:
            # Renormalization for vacancy defects

            H_aa = hamiltonian[np.ix_(self.parity_site_mask, self.parity_site_mask)]
            H_ab = hamiltonian[np.ix_(self.parity_site_mask, ~self.parity_site_mask)]
            H_ba = hamiltonian[np.ix_(~self.parity_site_mask, self.parity_site_mask)]
            H_bb = hamiltonian[np.ix_(~self.parity_site_mask, ~self.parity_site_mask)]

            if hamiltonian_method == "renormalization":
                hamiltonian = H_aa - H_ab @ spla.solve(H_bb, H_ba, assume_a='her', check_finite=False, overwrite_a=True)
            elif hamiltonian_method == "site_elimination":
                hamiltonian = H_aa
        return hamiltonian
        
    def compute_projector(self, hamiltonian:np.ndarray):
        """Compute the projector onto the lower band of the Hamiltonian."""
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2] # Lower band eigenvalues
        highest_lower_band = lower_band[-1] # Highest eigenvalue in the lower band

        d_operator = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j) # Projector diagonal matrix
        d_dagger = np.einsum('i,ij->ij', d_operator, eigenvectors.conj().T)
        projector = eigenvectors @ d_dagger
        return projector

    def compute_bott_index(self, projector:np.ndarray):
        """Compute the Bott index for the given projector."""

        # Repeated (two orbitals)
        X = np.repeat(self.X, 2)[self.parity_site_mask]
        Y = np.repeat(self.Y, 2)[self.parity_site_mask]

        x_unitary = np.exp(1j * 2 * np.pi * X / self.Lx) # unitary operator in the x-direction
        y_unitary = np.exp(1j * 2 * np.pi * Y / self.Ly)
        x_unitary_proj = np.einsum('i,ij->ij', x_unitary, projector) 
        y_unitary_proj = np.einsum('i,ij->ij', y_unitary, projector)
        x_unitary_dagger_proj = np.einsum('i,ij->ij', x_unitary.conj(), projector)  
        y_unitary_dagger_proj = np.einsum('i,ij->ij', y_unitary.conj(), projector)

        I = np.eye(projector.shape[0], dtype=np.complex128) 
        A = I - projector + projector @ x_unitary_proj @ y_unitary_proj @ x_unitary_dagger_proj @ y_unitary_dagger_proj # BI operator given in arxiv:2407.13767 [Eq. (5)]
        bott_index = round(np.imag(np.sum(np.log(spla.eigvals(A)))) / (2 * np.pi))
        return bott_index

    def compute_LDOS(self, hamiltonian:np.ndarray, number_of_states:int = 2) -> tuple:
        """
        Compute the Local Density of States (LDOS) for the given Hamiltonian.
        Parameters:
            hamiltonian (np.ndarray): The Hamiltonian matrix.
            number_of_states (int): Number of states to consider for the LDOS calculation. Defaults to 2.
        Returns:
            ldos (np.ndarray): the Local Density of States.
            eigenvalues (np.ndarray): the eigenvalues of the Hamiltonian.
            ap (float): the energy gap between the lower and upper bands.
            bandwidth (float): the bandwidth of the system.
            selected_indices (np.ndarray): indices of the selected states used in LDOS calculation.
        """

        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        number_of_states += number_of_states % 2 # Ensure even number of states
        mid_index = len(eigenvalues) // 2
        lower_idxs = np.arange(len(eigenvalues))[:mid_index][-number_of_states // 2:] 
        upper_idxs = np.arange(len(eigenvalues))[mid_index:][:number_of_states // 2]
        selected_indices = np.concatenate((lower_idxs, upper_idxs)) # Indices of the selected states to be used in LDOS

        LDOS = np.sum(np.abs(eigenvectors[:, selected_indices])**2, axis=1)

        # Adjust for missing parities
        if self.defect_type == "schottky":
            LDOS_expanded = np.zeros(self.Lx * self.Ly * 2, dtype=LDOS.dtype)
            LDOS_expanded[self.parity_site_mask] = LDOS
            LDOS = LDOS_expanded
        LDOS = LDOS[0::2] + LDOS[1::2] # Sum the two orbitals
        LDOS = LDOS / np.sum(LDOS) # Normalize the LDOS
        
        gap = abs(np.max(eigenvalues[lower_idxs]) - np.min(eigenvalues[upper_idxs]))
        width_of_selected_states = np.max(eigenvalues[selected_indices]) - np.min(eigenvalues[selected_indices])
        bandwidth = np.max(eigenvalues) - np.min(eigenvalues)

        return LDOS, eigenvalues, gap, bandwidth, selected_indices

    def average_LDOS_over_frenkel_pair(self, m_back:float, m_sub:float, number_of_states:int = 2):
        assert self.defect_type == "frenkel_pair", "This method is only applicable for Frenkel pair defects."
        
        FP_Lats = [DefectLattice(self.Lx, self.Ly, "frenkel_pair", pbc=self.pbc, frenkel_pair_displacement_index=i) for i in range(8)]
            
        def worker(i):
            FP_Lat = FP_Lats[i]
            LDOS, eigenvalues, gap, bandwidth, selected_indices = self.compute_LDOS(FP_Lat.compute_hamiltonian(m_back, m_sub), number_of_states)
            return LDOS, eigenvalues, gap, bandwidth, selected_indices, FP_Lat.X[self.site_mask], FP_Lat.Y[self.site_mask]

        # Compute LDOS for each displacement
        ldos_list = Parallel(n_jobs=-1, backend='threading')(
            delayed(worker)(i) for i in range(len(FP_Lats))
        )
        
        all_LDOS = np.concatenate([ldos[0] for ldos in ldos_list], axis=0)
        all_eigenvalues = np.array([ldos[1] for ldos in ldos_list])
        all_gaps = np.mean([ldos[2] for ldos in ldos_list])
        all_bandwidths = np.mean([ldos[3] for ldos in ldos_list])
        selected_indices = ldos_list[4][0]
        all_X = np.concatenate([ldos[5] for ldos in ldos_list], axis=0)
        all_Y = np.concatenate([ldos[6] for ldos in ldos_list], axis=0)


        coords = np.stack((all_X, all_Y), axis=1)
        unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        summed_LDOS = np.zeros(len(unique_coords))
        print(summed_LDOS.shape, inverse_indices.shape, all_LDOS.shape)
        np.add.at(summed_LDOS, inverse_indices, all_LDOS)
        summed_LDOS /= np.sum(summed_LDOS)
        LDOS = summed_LDOS
        eigenvalues = np.mean(all_eigenvalues, axis=0)
        X, Y = unique_coords[:, 0], unique_coords[:, 1]
        return LDOS, eigenvalues, all_gaps, all_bandwidths, selected_indices, X, Y


    # endregion

    # region Visualization Methods

    def plot_spectrum_LDOS(self):

        def plot_spectrum(spectrum_ax:plt.Axes, eigenvalues:np.ndarray, scatter_label:str, ldos_idxs:np.ndarray):
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
            return spectrum_ax

        def plot_ldos(ax:plt.Axes, LDOS:np.ndarray, X:np.ndarray, Y:np.ndarray, doInterpolation:bool = True):
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

        # -----------------------------------------------------------------------------------------------------------------------
        fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        m_back_values = [2.5, 1.0, -1.0, -2.5]

        for j, m_back in enumerate(m_back_values):
            good_m_sub_values = np.array(m_back_values)[np.array(m_back_values) != m_back]
            for i, m_sub in enumerate(good_m_sub_values):
                if self.defect_type == "schottky":
                    SubLat = DefectLattice(self.Lx, self.Ly, "schottky", pbc=self.pbc, schottky_distance=self.schottky_distance, schottky_type=["I", "II", "III"][i])
                    H = SubLat.compute_hamiltonian(m_back=m_back)
                    LDOS, eigenvalues, gap, bandwidth, selected_indices = SubLat.compute_LDOS(H, 2)
                    X, Y = SubLat.X, SubLat.Y
                    flabel = f"Gap={gap:.2f}\n$m_0={m_back}$"
                elif self.defect_type == "frenkel_pair":
                    LDOS, eigenvalues, gap, bandwidth, selected_indices, X, Y = self.average_LDOS_over_frenkel_pair(m_back=m_back, m_sub=m_sub)
                    flabel = f"Gap={gap:.2f}\n$m_0={m_back}$\n$m_{{sub}} = {m_sub}$"
                else:
                    H = self.compute_hamiltonian(m_back=m_back, m_sub=m_sub)
                    LDOS, eigenvalues, gap, bandwidth, selected_indices = self.compute_LDOS(H, 2)
                    X, Y = self.X, self.Y
                    flabel = f"Gap={gap:.2f}\n$m_{{sub}} = {m_sub}$\n$m_{{back}} = {m_back}$"


                plot_spectrum(axs[i, j], eigenvalues, flabel, selected_indices)
                plot_ldos(axs[i, j], LDOS, X, Y)

    def plot_defect_idxs(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        standard_site_size = 50
        large_site_size = 150

        ax.set_aspect('equal')
        ax.scatter(self.X, self.Y, s=standard_site_size, edgecolors='black', facecolors='black', linewidth=0.)

        for axes in [ax]: 
            if len(self.vacancy_idxs) in [2, 4]:
                for i, idx in enumerate(self.vacancy_idxs):
                    x, y = self.X[idx], self.Y[idx]
                    axes.scatter(x, y, s=large_site_size, edgecolors='none', facecolors='white', zorder=10)
                    if i % 2 == 0:
                        c = "red"
                    elif i % 2 == 1:
                        c = "blue"
                    axes.scatter(x, y, s=standard_site_size, facecolors = 'none', edgecolors = c, linewidth = 1.5, zorder=11)

            elif len(self.defect_idxs) == 5:
                for defect_idx in self.defect_idxs:
                    # Plot all defect indices, highlight the center one with a larger size
                    for i, defect_idx in enumerate(self.defect_idxs):
                        if (i == 0 and self.defect_type == "interstitial") or (i == 2 and self.defect_type == "substitution"):
                            axes.scatter(self.X[defect_idx], self.Y[defect_idx], s=large_site_size, color='red', edgecolors='black', linewidth=0.)
                        else:
                            axes.scatter(self.X[defect_idx], self.Y[defect_idx], s=standard_site_size, color='red')
            elif len(self.defect_idxs) == 1:
                axes.scatter(self.X[self.defect_idxs[0]], self.Y[self.defect_idxs[0]], s=large_site_size, color='red', edgecolors='black', linewidth=0.)


        for idx in range(len(self.X)):
            ax.text(self.X[idx], self.Y[idx], str(idx), color='white', fontsize=8, ha='center', va='center', zorder=20)

        tick_vals = [np.min(self.X), (np.min(self.X) + np.max(self.X)) / 2, np.max(self.X)]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([str(int(label + 1)) for label in tick_vals], fontsize=16)
        ax.set_yticks(tick_vals)
        ax.set_yticklabels([str(int(label + 1)) for label in tick_vals], fontsize=16)

        ax.set_xlabel("$L_x$", fontsize=20)
        ax.set_ylabel("$L_y$", fontsize=20)
        plt.show()
        return ax

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

    # endregion


def multiprocessing_gap_comparison(Lat1:DefectLattice, Lat2:DefectLattice, m_back_values, m_sub_values, n_jobs = -1):
    parameters = tuple(product(m_back_values, m_sub_values))
    def worker(i):
        m_back, m_sub = parameters[i]
        gap_pristine = Lat1.compute_LDOS(Lat1.compute_hamiltonian(m_back, m_sub), 2)[2]
        gap_defect = Lat2.compute_LDOS(Lat2.compute_hamiltonian(m_back, m_sub), 2)[2]
        return (gap_pristine, gap_defect)
    
    with tqdm_joblib(tqdm(total=len(parameters), desc="Computing gaps")) as progress_bar:
        gap_values = Parallel(n_jobs=n_jobs)(delayed(worker)(i) for i in range(len(parameters)))
    return gap_values


def gap_comparison(Lx, Ly, defect_type):
    Pristine = DefectLattice(Lx, Ly, "none", pbc=True)
    Defect = DefectLattice(Lx, Ly, defect_type, pbc=True)

    m_sub_range = (-4.0, 4.0)

    resolution = 201
    m_sub_values = np.linspace(m_sub_range[0], m_sub_range[1], resolution)

    m_back_values = [2.5, 1.0, -1.0, -2.5]

    gap_values = []
    x_vals = []

    for m_back in m_back_values:
        if m_back < -2.0:
            good_msub_values = m_sub_values[m_sub_values >= -2.0]
        elif m_back > 2.0:
            good_msub_values = m_sub_values[m_sub_values <= 2.0]
        elif m_back < 0.0 and m_back > -2.0:
            good_msub_values = m_sub_values[(m_sub_values <= -2.0) | (m_sub_values >= 0.0)]
        elif m_back > 0.0 and m_back < 2.0:
            good_msub_values = m_sub_values[(m_sub_values <= 0.0) | (m_sub_values >= 2.0)]

        gap_values.append(multiprocessing_gap_comparison(Pristine, Defect, [m_back], good_msub_values))
        x_vals.append(good_msub_values)

    fig, axs = plt.subplots(len(m_back_values), 1, figsize = (10, 4 * len(m_back_values)), sharex=True)
    if len(m_back_values) == 1:
        axs = np.array([axs])
    
    for i, m_back in enumerate(m_back_values):
        gap_set = gap_values[i]
        pristine_gaps, defect_gaps = zip(*gap_set)

        axs[i].scatter(x_vals[i], pristine_gaps, s=25, label="Pristine", color='blue', alpha=0.5)
        axs[i].scatter(x_vals[i], defect_gaps, s=25, label=f"Defect $m_0^{{\\text{{back}}}}={m_back}$", color='red', alpha=0.5)

        max_gap_index = np.argmax(defect_gaps)
        if m_back in [2.5, -2.5]:
            axs[i].annotate(
                f"Maximum Gap: {defect_gaps[max_gap_index]:1.4f}\noccurs at\n$m_0^{{\\text{{sub}}}}={x_vals[i][max_gap_index]:1.4f}$",
                xy=(0.95, 0.03),
                xycoords='axes fraction',
                ha='right',
                va='bottom',
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.0)
            )

        axs[i].set_xlabel(r"$m_0^{\text{sub}}$", fontsize=20)
        axs[i].set_ylabel("Gap", fontsize=20)
        axs[i].legend(loc='center left')

        xticks = [-4, -2.5, -2, -1.0, 0, 1.0, 2, 2.5, 4]
        axs[i].set_xticks(xticks)
        axs[i].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        for tick in xticks:
            axs[i].axvline(tick, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    

    fig.suptitle(f"Gap Comparison for {defect_type} Defect Type\n$L_x={Lx}, L_y={Ly}$", fontsize=24)
    plt.tight_layout()
    plt.savefig(f"gap_comparison_{defect_type}.png")


def gap_comparison_vac(Lx, Ly, defect_type):
    assert defect_type in ["vacancy", "large_vacancy", "schottky"], f"Invalid defect type: {defect_type}"
    Pristine = DefectLattice(14, 14, "none", pbc=True)
    Defect = DefectLattice(Lx, Ly, defect_type, pbc=True)
    resolution = 201

    m_range = (-4.0, 4.0)
    m_values = np.linspace(m_range[0], m_range[1], resolution)

    gap_values = multiprocessing_gap_comparison(Pristine, Defect, m_values, [None])

    fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)

    pristine_gaps, defect_gaps = zip(*gap_values)

    ax.scatter(m_values, pristine_gaps, s=25, label="Pristine", color='blue', alpha=0.5)
    ax.scatter(m_values, defect_gaps, s=25, label=f"Defect", color='red', alpha=0.5)

    ax.set_xlabel(r"$m_0$", fontsize=20)
    ax.set_ylabel("Gap", fontsize=20)
    ax.legend(loc='center left')

    xticks = [-4, -2, 0, 2, 4]
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    for tick in xticks:
        ax.axvline(tick, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    other_xticks = [-2.5, -1.0, 1.0, 2.5]
    ax.set_xticks(np.concatenate([xticks, other_xticks]))
    for tick in other_xticks:
        ax.axvline(tick, color='black', linestyle='--', linewidth=0.5, alpha=0.5)


    fig.suptitle(f"Gap Comparison for {defect_type} Defect Type\n$L_x={Lx}, L_y={Ly}$", fontsize=24)
    plt.tight_layout()
    plt.savefig(f"gap_comparison_{defect_type}.png")


def multiprocessing_schottky_gap():

    Lx = Ly = 30
    schottky_distances = np.arange(1, Lx, 2)
    m_back = -1.

    print(schottky_distances)

    def worker(i):
        schottky_distance = schottky_distances[i]

        SchottkyLattice = DefectLattice(Lx, Ly, "schottky", pbc = True, schottky_distance = schottky_distance, schottky_type = "I")
        schottky_ldos_values = SchottkyLattice.compute_LDOS(SchottkyLattice.compute_hamiltonian(m_back), 2)
        gap_schottky = schottky_ldos_values[2]

        DefectlessLattice = DefectLattice(Lx, Ly, "none", pbc = True)
        defectless_ldos_values = DefectlessLattice.compute_LDOS(DefectlessLattice.compute_hamiltonian(m_back), 2)
        gap_defectless = defectless_ldos_values[2]

        return [schottky_distance, gap_schottky, gap_defectless]

    with tqdm_joblib(tqdm(total=len(schottky_distances), desc="Computing Schottky gaps")) as progress_bar:
       results = Parallel(n_jobs=-1)(delayed(worker)(i) for i in range(len(schottky_distances)))
    return results


def plot_defect_figure():
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    defect_types = ["vacancy", "substitution", "interstitial", "frenkel_pair"]
    sl = 14

    for defect_type, ax in zip(defect_types, axs.flatten()):
        if defect_type != "interstitial":
            Lattice = DefectLattice(sl+1, defect_type, pbc=True)
        else:
            Lattice = DefectLattice(sl, defect_type, pbc=True)

        ax.scatter(Lattice.X, Lattice.Y, s=50, edgecolors='black', facecolors='black', linewidth=0.5)

        if defect_type in ["interstitial", "substitution"]:
            ax.scatter(Lattice.X[Lattice.defect_idxs], Lattice.Y[Lattice.defect_idxs], s=150, color='red', edgecolors='black', linewidth=0.5, zorder=3)
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if abs(i) + abs(j) == 2:
                        continue
                    ax.scatter(Lattice.X[Lattice.defect_idxs[0]] + i, Lattice.Y[Lattice.defect_idxs[0] ] + j, s=50, color='red', edgecolors='black', linewidth=0.5, zorder=2)
        if defect_type == "frenkel_pair":
            x_center, y_center = Lattice.Lx // 2, Lattice.Ly // 2
            ax.scatter(x_center, y_center, s=50, color='white', edgecolors='red', linewidth=2.0, zorder=3)
            for disp in Lattice._fp_displacements:
                ax.scatter(Lattice._vacancy_position[0] + disp[0], Lattice._vacancy_position[1] + disp[1], s=50, color='red', edgecolors='black', linewidth=0.5)
        if defect_type == "vacancy":
            x_center, y_center = Lattice.Lx // 2, Lattice.Ly // 2
            ax.scatter(x_center, y_center, s=150, color='white', edgecolors='red', linewidth=2.0, zorder=3)
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if abs(i) + abs(j) == 2:
                        continue
                    ax.scatter(x_center + i, y_center + j, s=50, color='white', edgecolors='red', linewidth=2.0, zorder=2)

        ax.set_xticks([0, (Lattice.Lx - 1) / 2, Lattice.Lx - 1])
        ax.set_xticklabels(['1', '$L_x$', str(Lattice.Lx)], fontsize=16)
        ax.set_yticks([0, (Lattice.Ly - 1) / 2, Lattice.Ly - 1])
        ax.set_yticklabels(['1', '$L_y$', str(Lattice.Ly)], fontsize=16)
    
    plt.tight_layout()
    plt.savefig("defect_lattices.png")
    plt.show()


if __name__ == "__main__":
    for dt in ["vacancy", "substitution", "interstitial", "schottky", "frenkel_pair"]:
        if dt in ["interstitial", "schottky"]:
            Lattice = DefectLattice(6, 6, dt)
        else:
            Lattice = DefectLattice(5, 5, dt)

        fig, ax = plt.subplots(figsize=(6, 6))
        X, Y = Lattice.X, Lattice.Y
        s0 = 100
        s1 = 400
        s2 = 500
        ax.scatter(X, Y, s=s1, edgecolors='red', facecolors='white', linewidth=1, linestyle='-')
        ax.scatter(X, Y, s=s0, edgecolors='blue', facecolors='white', linewidth=1, linestyle='-')

        if Lattice.defect_type == "vacancy":
            ax.scatter(X[Lattice.vacancy_idxs], Y[Lattice.vacancy_idxs], s=s2, edgecolors='white', facecolors='white', linewidth=2.0)

        elif Lattice.defect_type == "schottky":
            for i, idx in enumerate(Lattice.vacancy_idxs):
                ax.scatter(X[idx], Y[idx], s=s2, edgecolors='white', facecolors='white', linewidth=2.0)
                if i % 2 == 0:
                    ax.scatter(X[idx], Y[idx], s=s1, edgecolors='red', facecolors='white', linewidth=1, linestyle='-')
                else:
                    ax.scatter(X[idx], Y[idx], s=s0, edgecolors='blue', facecolors='white', linewidth=1, linestyle='-')

        elif Lattice.defect_type in ["substitution", "interstitial"]:
            for i, idx in enumerate(Lattice.defect_idxs):
                ax.scatter(X[idx], Y[idx], s=s2, edgecolors='white', facecolors='white', linewidth=2.0)
                ax.scatter(X[idx], Y[idx], s=s1, edgecolors='red', facecolors='white', linewidth=1, linestyle='--')
                ax.scatter(X[idx], Y[idx], s=s0, edgecolors='blue', facecolors='white', linewidth=1, linestyle='--')

        ax.set_title(f"{Lattice.defect_type.replace("_", " ").capitalize()} Defect", fontsize=24)
        ax.set_xticks([0, (Lattice.Lx - 1) / 2, Lattice.Lx - 1])
        ax.set_xticklabels(['$1$', '$L_x$', f"${str(Lattice.Lx)}$"], fontsize=24)
        ax.xaxis.set_tick_params(width=2)
        ax.set_yticks([0, (Lattice.Ly - 1) / 2, Lattice.Ly - 1])
        ax.set_yticklabels(['$1$', '$L_y$', f"${str(Lattice.Ly)}$"], fontsize=24)
        ax.yaxis.set_tick_params(width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.tight_layout()
        plt.savefig(f'fig1_{Lattice.defect_type}.png')


    def plot_gap_vs_lattice_size():
        def worker(sl):
            Lat = DefectLattice(sl, sl, "vacancy")
            LDOS, eigenvalues, gap, bandwidth, selected_indices = Lat.compute_LDOS(Lat.compute_hamiltonian(2.0, None), 2)
            return [sl, gap]
        
        sl_values = np.arange(11, 51, 1)[::-1]

        with tqdm_joblib(tqdm(total=len(sl_values), desc="Computing gaps for even and odd lattice sizes")) as progress_bar:
            results = Parallel(n_jobs=-1)(delayed(worker)(sl) for sl in sl_values)

        sls, gaps = zip(*results)
        plt.scatter(sls[::2], gaps[::2], s=25, color='blue', label='Even Lattice Size', alpha=0.5)
        plt.scatter(sls[1::2], gaps[1::2], s=25, color='red', label='Odd Lattice Size', alpha=0.5)
        plt.xlabel("Lattice Size (L)", fontsize=20)
        plt.ylabel("Energy Gap", fontsize=20)
        plt.legend()
        plt.show()


    if False:
        for defect_method in ["frenkel_pair"]:
            if defect_method in ["interstitial", "schottky"]:
                gap_comparison(14, 14, defect_method)
            else:
                gap_comparison(15, 15, defect_method)

    if False:
        data = multiprocessing_schottky_gap()
        schottky_distances, schottky_gaps, gap_defectless = zip(*data)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(schottky_distances, schottky_gaps, s=25, color='blue', alpha=0.5, label='Schottky Gap')
        ax.scatter(schottky_distances, gap_defectless, s=25, color='red', alpha=0.5, label='Defectless Gap')
        ax.set_xlabel("Schottky Distance (multiples of $\\sqrt{2}a$)", fontsize=20)
        ax.set_ylabel("Gap", fontsize=20)
        ax.set_title("Schottky Gap vs Distance", fontsize=24)
        ax.legend()
        ax.set_xticks(schottky_distances)
        ax.set_xticklabels([str(d) for d in schottky_distances], fontsize=16)
        plt.savefig("schottky_gap_vs_distance_I.png")
        plt.show()


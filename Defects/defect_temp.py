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
    def __init__(self, side_length:int, defect_type:str, pbc:bool = True):
        self._side_length = side_length
        self._defect_type = defect_type
        self._pbc = pbc

        assert defect_type in ["vacancy", "schottky", "substitution", "interstitial", "frenkel_pair"], \
            f"Invalid defect type: {defect_type}."
        if defect_type in ["interstitial", "schottky"]:
            assert side_length % 2 == 0, "Side length must be even for interstitial and schottky defects."
        else:
            assert side_length % 2 == 1, "Side length must be odd for vacancy, substitution, and frenkel pair defects."
        
        self._pristine_lattice = self.generate_pristine_lattice()
        self._vacancy_idxs = [] 
        self._defect_idxs =  []

        match defect_type:
            case "none":
                self._lattice = self._pristine_lattice
            case "vacancy":
                self._lattice = self.generate_vacancy_lattice()
            case "schottky":
                self._lattice = self.generate_schottky_lattice()
            case "substitution":
                self._lattice = self.generate_substitution_lattice()
            case "interstitial":
                self._lattice = self.generate_interstitial_lattice()

        self._Y, self._X = np.where(self.lattice >= 0)

        if defect_type == "interstitial":
            self._X = self._X / 2
            self._Y = self._Y / 2
        self._dx, self._dy = self.compute_distances()

        self._site_mask = np.full(len(self.X), True)
        if defect_type == "vacancy":
            self._site_mask[self.vacancy_idxs] = False

        if defect_type in ["none", "vacancy", "schottky", "substitution"]:
            self._Sx, self._Sy, self._Cx_plus_Cy, self._I = self.compute_wannier_fourier()
        else:
            self._Sx, self._Sy, self._Cx_plus_Cy, self._I = self.compute_wannier_symmetry()

    # region Properties
    @property
    def side_length(self):
        return self._side_length
    @property
    def defect_type(self):
        return self._defect_type
    @property
    def pbc(self):
        return self._pbc
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
    # endregion

    # region Lattice Generation Methods
    def generate_pristine_lattice(self):
        return np.arange(self.side_length**2).reshape((self.side_length, self.side_length))

    def generate_vacancy_lattice(self):
        lattice = self.generate_pristine_lattice()
        center_idx = self.side_length // 2
        self._vacancy_idxs.append(lattice[center_idx, center_idx])
        lattice[center_idx , center_idx] = -1
        return lattice
    
    def generate_large_vacancy_lattice(self):
        lattice = self.generate_pristine_lattice()
        center_idx = self.side_length // 2
        # Create a vacancy at the center of the lattice
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if abs(i) + abs(j) == 2:
                    continue
                self._vacancy_idxs.append(lattice[center_idx + i, center_idx + j])
                lattice[center_idx + i, center_idx + j] = -1
        return lattice

    def generate_schottky_lattice(self, schottky_type:str = "III", schottky_distance:int = 5):
        assert schottky_type in ["I", "II", "III"], "Invalid schottky type. Must be 'I', 'II', or 'III'."
        assert (self.side_length + schottky_distance) % 2 == 1, "Side length + schottky distance must be odd for schottky defects."
        assert schottky_distance < self.side_length // 2, "Schottky distance must be less than half the side length."

        lattice = self._pristine_lattice.copy()
        midpoint_right = self.side_length // 2 + schottky_distance // 2
        midpoint_left = self.side_length // 2 - schottky_distance // 2 - 1

        if schottky_type == "I":
            # Generate a Schottky defect lattice with defects at the center. Here "O" is a typical site, "U" is an up parity site (down removed), and "D" is the reverse.
            # The defect section of the lattice then appears as: 
            #  O  | ... |  U
            # ... | ... | ...
            #  D  | ... |  O 
            up_parity_idx =   lattice[midpoint_right, midpoint_right]
            down_parity_idx = lattice[midpoint_left, midpoint_left]
            defect_idxs = [up_parity_idx, down_parity_idx]
            vacant_positions = [(midpoint_right, midpoint_right), (midpoint_left, midpoint_left)]
        elif schottky_type == "II":
            #  D  | ... |  U
            # ... | ... | ...
            #  D  | ... |  U
            up_parity_idx1 =   lattice[midpoint_right, midpoint_right]
            down_parity_idx1 = lattice[midpoint_left, midpoint_left]
            up_parity_idx2 =   lattice[midpoint_right - schottky_distance, midpoint_right]
            down_parity_idx2 = lattice[midpoint_left + schottky_distance, midpoint_left]
            defect_idxs = [up_parity_idx1, down_parity_idx1, up_parity_idx2, down_parity_idx2]
            vacant_positions = [(midpoint_right, midpoint_right), (midpoint_left, midpoint_left), (midpoint_right - schottky_distance, midpoint_right), (midpoint_left + schottky_distance, midpoint_left)]
        elif schottky_type == "III":
            #  D  | ... |  U
            # ... | ... | ...
            #  U  | ... |  D
            up_parity_idx1 =   lattice[midpoint_right, midpoint_right]
            up_parity_idx2 =   lattice[midpoint_left, midpoint_left]
            down_parity_idx1 = lattice[midpoint_right - schottky_distance, midpoint_right]
            down_parity_idx2 = lattice[midpoint_left + schottky_distance, midpoint_left]
            defect_idxs = [up_parity_idx1, down_parity_idx1, up_parity_idx2, down_parity_idx2]
            vacant_positions = [(midpoint_right, midpoint_right), (midpoint_left, midpoint_left), (midpoint_right - schottky_distance, midpoint_right), (midpoint_left + schottky_distance, midpoint_left)]

        self._vacancy_idxs = defect_idxs
        return lattice

    def generate_substitution_lattice(self):
        lattice = self.generate_pristine_lattice()

        # Create a substitution at the center of the lattice
        center_idx = self.side_length // 2
        self._defect_idxs.append(lattice[center_idx, center_idx])
        return lattice
    
    def generate_large_substitution_lattice(self):
        lattice = self.generate_pristine_lattice()

        # Create five substitutions at the center of the lattice
        center_idx = self.side_length // 2
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if abs(i) + abs(j) == 2:
                    continue
                self._defect_idxs.append(lattice[center_idx + i, center_idx + j])
        return lattice

    def generate_interstitial_lattice(self):
        Y, X = np.where(self._pristine_lattice >= 0)
        x_mean = np.round(np.mean(X), 1)
        y_mean = np.round(np.mean(Y), 1)

        coordinates = np.array([X, Y])
        coordinates = np.concatenate((coordinates, np.array([[x_mean], [y_mean]])), axis=1)
        coordinates = np.unique(np.round(coordinates * 2).astype(int), axis=1)
        coordinates = coordinates[:, np.lexsort((coordinates[0], coordinates[1]))]

        interstitial_lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
        interstitial_lattice[coordinates[1], coordinates[0]] = np.arange(len(coordinates[0]))

        self._defect_idxs = [np.max(interstitial_lattice) // 2]
        return interstitial_lattice

    def generate_frenkel_pair_lattice(self):
        pass
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

    # region Hamiltoninan Methods
    def compute_hamiltonian(self, m_back:float, m_sub:float = None, t:float = 1.0, t0:float = 1.0, hamiltonian_method:str = "renormalization"):
        """
        Methods are described in arxiv:2407.13767.
        For "none" and "substitution" defects, we use typical hamiltonian construction.
        For "vacancy" and "schottky" defects, we use renormalization, site elimination, or symmetry methods of construction.
        For "interstitial" and "frenkel_pair" defects, we use the symmetry method for hamiltonian construction. 
        
        """
        if self.defect_type in ["substitution", "interstitial", "frenkel_pair"]:
            assert m_sub != None, "m_sub must be provided for substitution, interstitial, and frenkel pair defects."
        
        onsite_mass = self.I * m_back

        if self.defect_type in ["substitution", "interstitial", "frenkel_pair"]:
            for idx in self.defect_idxs:
                onsite_mass[idx, idx] = m_sub

        d1 = t * self.Sx
        d2 = t * self.Sy
        d3 = onsite_mass + t0 * self.Cx_plus_Cy

        pauli_x = np.array([[0, 1], [1, 0]],    dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]],   dtype=complex)

        hamiltonian = np.kron(d1, pauli_x) + np.kron(d2, pauli_y) + np.kron(d3, pauli_z)

        if self.defect_type in ["vacancy", "schottky"]:
            # Renormalization for vacancy defects
            site_mask = np.repeat(self._site_mask, 2)

            if self.defect_type == "schottky":
                for i, idx in enumerate(self.vacancy_idxs):
                    if i % 2 == 0:
                        site_mask[2 * idx + 1] = False
                    else:
                        site_mask[2 * idx] = False

            H_aa = hamiltonian[np.ix_(site_mask, site_mask)]
            H_ab = hamiltonian[np.ix_(site_mask, ~site_mask)]
            H_ba = hamiltonian[np.ix_(~site_mask, site_mask)]
            H_bb = hamiltonian[np.ix_(~site_mask, ~site_mask)]

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
        X = np.repeat(self.X[self._site_mask], 2)
        Y = np.repeat(self.Y[self._site_mask], 2)
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

    def compute_LDOS(self, hamiltonian:np.ndarray, number_of_states:int = 2):
        """Compute the Local Density of States (LDOS) for the given Hamiltonian.
        Parameters:
            hamiltonian (np.ndarray): The Hamiltonian matrix.
            number_of_states (int): Number of states to consider for the LDOS calculation. Defaults to 2.
        Returns:
            LDOS (np.ndarray): The Local Density of States.
            eigenvalues (np.ndarray): The eigenvalues of the Hamiltonian.
            gap (float): The energy gap between the lower and upper bands.
            bandwidth (float): The bandwidth of the system.
            selected_indices (np.ndarray): Indices of the selected states used in LDOS calculation.
        """
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        number_of_states += number_of_states % 2 # Ensure even number of states
        mid_index = len(eigenvalues) // 2
        lower_idxs = np.arange(len(eigenvalues))[:mid_index][-number_of_states // 2:] 
        upper_idxs = np.arange(len(eigenvalues))[mid_index:][:number_of_states // 2]
        selected_indices = np.concatenate((lower_idxs, upper_idxs)) # Indices of the selected states to be used in LDOS

        LDOS = np.sum(np.abs(eigenvectors[:, selected_indices])**2, axis=1)
        LDOS = LDOS[0::2] + LDOS[1::2] # Sum the two orbitals
        LDOS = LDOS / np.sum(LDOS) # Normalize the LDOS
        gap = abs(np.max(eigenvalues[lower_idxs]) - np.min(eigenvalues[upper_idxs]))
        bandwidth = np.max(eigenvalues) - np.min(eigenvalues)

        return LDOS, eigenvalues, gap, bandwidth, selected_indices

    # endregion

    # region Visualization Methods

    def figure_spectrum_LDOS(self):

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

        fig, axs = plt.subplots(3, 4, figsize=(20, 15))

        m_back_values = [2.5, 1.0, -1.0, -2.5]

        for j, m_back in enumerate(m_back_values):
            good_m_sub_values = np.array(m_back_values)[np.array(m_back_values) != m_back]
            for i, m_sub in enumerate(good_m_sub_values):
                LDOS, eigenvalues, gap, bandwidth, selected_indices = self.compute_LDOS(self.compute_hamiltonian(m_back=m_back, m_sub=m_sub), 2)

                plot_spectrum(axs[i, j], eigenvalues, f"gap={gap:.2f}\n$m_{{sub}} = {m_sub}$\n$m_{{back}} = {m_back}$", selected_indices)
        
        plt.show()




    # endregion

if __name__ == "__main__":
    # Example usage
    Lattice = DefectLattice(side_length=14, defect_type="interstitial", pbc=True)
    Lattice.figure_spectrum_LDOS()
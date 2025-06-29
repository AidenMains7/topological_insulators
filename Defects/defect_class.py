import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import matplotlib.ticker as ticker
import scipy.linalg as spla
from matplotlib.widgets import Slider
import inspect
from joblib import Parallel, delayed
from tqdm_joblib import tqdm, tqdm_joblib



class DefectSquareLattice:
    def __init__(self, side_length:int, defect_type:str, pbc:bool = True, frenkel_pair_index:int = 0, doLargeDefect:bool = False, *args, **kwargs):
        self._pbc = pbc
        self._side_length = side_length
        self._defect_type = defect_type
        self._doLargeDefect = doLargeDefect
        self._frenkel_pair_index = frenkel_pair_index

        pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)
        self._pauli_matrices = [pauli1, pauli2, pauli3]

        self._pristine_lattice = self.generate_lattice() 
        match self.defect_type:
            case "none":
                self._lattice = self._pristine_lattice.copy()
                self._defect_indices = None
            case "vacancy":
                self._lattice = self.generate_vacancy_lattice()
                self._defect_indices = None
            case "substitution":
                self._lattice = self._pristine_lattice.copy()
                if self.doLargeDefect:
                    self._defect_indices = []
                    center_idx = [self._side_length // 2, self._side_length // 2]
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            if abs(i) + abs(j) == 2:
                                continue
                            self._defect_indices.append(self.lattice[center_idx[0] + i, center_idx[1] + j])
                else:
                    self._defect_indices = [self.lattice[self._side_length // 2, self._side_length // 2]]
            case "interstitial":
                self._lattice = self.generate_interstitial_lattice()
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
            case _:
                raise ValueError(f"Unknown defect type: {defect_type}")

        self._Y, self._X = np.where(self._lattice >= 0)[:]
        if self._defect_type in ["interstitial", "frenkel_pair"]:
            self._X = self._X.astype(float) / 2
            self._Y = self._Y.astype(float) / 2
            pass

        self._system_size = len(self.X)
        self.compute_distances()
        self.compute_wannier_polar()

        if not doLargeDefect:
            self.LargeDefectLattice = DefectSquareLattice(side_length, defect_type, pbc=pbc, doLargeDefect=True, frenkel_pair_index=self._frenkel_pair_index)

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

    # endregion

    # region Geometry
    def generate_lattice(self, *args, **kwargs):
        return np.arange(self.side_length**2).reshape((self.side_length, self.side_length))

    def generate_vacancy_lattice(self, *args, **kwargs):
        if self.side_length % 2 == 0:
            raise ValueError("Side length must be odd for a single vacancy in the center.")
        lattice = self._pristine_lattice.copy()
    
        center_idx = self.side_length // 2
        lattice[center_idx, center_idx] = -1

        if self._doLargeDefect:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if abs(i) + abs(j) == 2:
                        continue
                    lattice[center_idx + i, center_idx + j] = -1

        return lattice

    def generate_interstitial_lattice(self):
        if self.side_length % 2 == 1:
            raise ValueError("Side length must be even for a single interstitial in the center.")
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
        if self.side_length % 2 == 0:
            raise ValueError("Side length must be odd for a single vacancy in the center.")
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

    def compute_distances(self, *args, **kwargs):
        dx = self.X - self.X[:, None]
        dy = self.Y - self.Y[:, None]
        if self.pbc:
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
        dx, dy = self.dx, self.dy

        theta = np.arctan2(dy, dx)  
        dr = np.sqrt(dx ** 2 + dy ** 2)

        principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & ((dr <= 1 + 1e-6) & (dr > 1e-6))
        diagonal_mask  = ((np.isclose(np.abs(dx), np.abs(dy), atol=1e-4)) & (dx != 0)) & ((dr <= 1/np.sqrt(2) + 1e-6) & (dr > 1e-6))
        hopping_mask = principal_mask | diagonal_mask
    
        d_cos = np.where(hopping_mask, np.cos(theta), 0. + 0.j)
        d_sin = np.where(hopping_mask, np.sin(theta), 0. + 0.j)

        amplitude = np.where(hopping_mask, np.exp(1. - dr), 0. + 0.j)

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

        if self.defect_type in ["vacancy", "none"]:
            onsite_mass = M_background * self.I
        elif self.defect_type in ["substitution", "interstitial", "frenkel_pair"]:
            if M_substitution is None:
                raise ValueError("M_substitution must be provided for 'substitution', 'interstitial', or 'frenkel pair' defects.")
            onsite_mass = M_background * self.I
            onsite_mass[self.defect_indices, self.defect_indices] = M_substitution


        #fig, ax = plt.subplots()
        #ax.scatter(self.X, self.Y, facecolor='none', edgecolor='black', label='Sites')
        #ax.set_aspect('equal', adjustable='box')
        #ax.scatter(self.X[self.defect_indices], self.Y[self.defect_indices], c='red', label='Defect')
        #y, x = np.where(np.isin(self.lattice, self.defect_indices))
        #ax.scatter(x/2, y/2, c='blue', alpha=0.5)
        #plt.show()
        #scat = ax.scatter(self.X, self.Y, c=onsite_mass.diagonal().astype(float).real)
        #cbar = plt.colorbar(scat, ax=ax)
        #cbar.set_ticks(np.sort(np.unique(onsite_mass.diagonal().real)))
        #ax.set_title(f"m_back={M_background}, m_sub={M_substitution}")
        #plt.show()
        if M_substitution is None:
            M_substitution = 1.
        d1 = t * self.Sx * np.sign(M_substitution)
        d2 = t * self.Sy * np.sign(M_substitution)
        d3 = onsite_mass + t0 * (self.Cx_plus_Cy) * np.sign(M_substitution)

        hamiltonian = np.kron(d1, self.pauli_matrices[0]) + np.kron(d2, self.pauli_matrices[1]) + np.kron(d3, self.pauli_matrices[2])
        return hamiltonian

    def compute_projector(self, hamiltonian):
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2]
        highest_lower_band = lower_band[-1]

        D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
        D_herm_conj = np.einsum('i,ij->ij', D, eigenvectors.conj().T)

        projector = eigenvectors @ D_herm_conj
        return projector

    def compute_bott_index(self, projector:np.ndarray):
        X = np.repeat(self.X, 2)
        Y = np.repeat(self.Y, 2)
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

    def compute_local_chern_operator(self, hamiltonian, *args, **kwargs):
        projector = self.compute_projector(hamiltonian)
        X = np.diag(np.repeat(self.X, 2))
        Y = np.diag(np.repeat(self.Y, 2))

        Q = np.eye(projector.shape[0], dtype=np.complex128) - projector
        C_L = -4 * np.pi * np.imag(projector @ X @ Q @ Y @ projector)
        return C_L

    def compute_LDOS(self, hamiltonian:np.ndarray, number_of_states:int = 2, *args, **kwargs):
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        number_of_states += number_of_states % 2
        mid_index = len(eigenvalues) // 2
        lower_idxs = np.arange(len(eigenvalues))[:mid_index][-number_of_states // 2:]
        upper_idxs = np.arange(len(eigenvalues))[mid_index:][:number_of_states // 2]
        selected_indices = np.concatenate((lower_idxs, upper_idxs))

        LDOS = np.sum(np.abs(eigenvectors[:, selected_indices])**2, axis=1)
        LDOS = LDOS[0::2] + LDOS[1::2]
        LDOS = LDOS / np.sum(LDOS)
        gap = abs(np.max(eigenvalues[lower_idxs]) - np.min(eigenvalues[upper_idxs]))
        bandwidth = np.max(eigenvalues) - np.min(eigenvalues)

        LDOS -= np.min(LDOS)
        if np.max(LDOS) > 0:
            LDOS /= np.max(LDOS)

        data_dict = {
            "LDOS": LDOS,
            "eigenvalues": eigenvalues,
            "gap": gap,
            "bandwidth": bandwidth,
            "ldos_idxs": selected_indices
        }
        return data_dict

    def _compute_for_figure(self, m_background:float, m_substitution:float, number_of_states:float):
        def _average_over_frenkel_pair():
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

    # endregion

    # region Plotting
    def plot_spectrum_ldos(self, m_background_values:"list[float]" = [2.5, 1.0, -1.0, -2.5], 
                             m_substitution_values:"list[float] | None" = None, doLargeDefectFigure:bool = False, number_of_states:int = 2): 
        def plot_ldos_ax(ldos_ax:plt.Axes, LDOS, X, Y):
            # Dynamically set marker size based on number of points and axes size
            bbox = ldos_ax.get_window_extent().transformed(ldos_ax.figure.dpi_scale_trans.inverted())
            width, height = bbox.width * ldos_ax.figure.dpi, bbox.height * ldos_ax.figure.dpi
            area = width * height
            N = len(X)
            # Heuristic: marker area is a fraction of axes area divided by number of points
            marker_area = max(area / (N * 10), 0)
            scat = ldos_ax.scatter(X, Y, c=LDOS, s=marker_area, cmap='jet')
            ldos_ax.set_xticks([np.min(X), (np.max(X) + np.min(X)) / 2, np.max(X)])
            ldos_ax.set_yticks([np.min(X), (np.max(X) + np.min(X)) / 2, np.max(X)])
            tick_labels = [np.min(X) + 1, (np.max(X) + np.min(X)) // 2 + 1, np.max(X) + 1]
            ldos_ax.set_xticklabels([str(int(label)) for label in tick_labels], fontsize=20)
            ldos_ax.set_yticklabels([str(int(label)) for label in tick_labels], fontsize=20)
            ldos_ax.set_xlabel(r"$x$", fontsize=20)
            ldos_ax.set_ylabel(r"$y$", fontsize=20)
            ldos_ax.set_aspect('equal')
            return ldos_ax
        
        def plot_spectrum_ax(spectrum_ax:plt.Axes, eigenvalues:np.ndarray, scatter_label:str, ldos_idxs:np.ndarray):
            x_values = np.arange(len(eigenvalues))
            idxs_mask = np.isin(x_values, ldos_idxs)
            spectrum_ax.scatter(x_values[~idxs_mask], eigenvalues[~idxs_mask], s=25, color = 'black', zorder = 0)
            spectrum_ax.scatter(x_values[ idxs_mask], eigenvalues[ idxs_mask], s=25, color = 'red',   zorder = 1)
            spectrum_ax.set_xticks([])
            spectrum_ax.set_xlabel(r"$n$", fontsize=20)
            spectrum_ax.set_ylabel(r"$E_n$", fontsize=20)
            spectrum_ax.tick_params(axis='y', labelsize=20)
            spectrum_ax.annotate(
                scatter_label,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                ha='left',
                va='top',
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.0)
            )

        if m_substitution_values is None:
            m_substitution_values = np.array(m_background_values).copy()

        # Get shape of the figure based on the defect type
        if self.defect_type in ["none", "vacancy"]:
            m_substitution_values = [None] if doLargeDefectFigure is False else [None] * 2
            n_cols, n_rows = 2 * len(m_background_values), len(m_substitution_values)
        else:
            n_cols, n_rows = 2 * len(m_background_values), len(m_substitution_values) - 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

        if n_rows == 1:
            axs = np.array([axs])

        for i, m_background in enumerate(m_background_values):
            spectrum_axs = axs[:, 0 + 2 * i].flatten()
            ldos_axs = axs[:, 1 + 2 * i].flatten()
            
            good_m_sub_vals = np.array(m_substitution_values)[np.array(m_substitution_values) != m_background]

            for j, (spectrum_ax, ldos_ax, m_substitution) in enumerate(zip(spectrum_axs, ldos_axs, good_m_sub_vals)):
                if m_substitution == m_background:
                    continue
                
                if j == 1 and doLargeDefectFigure and self.defect_type in ["none", "vacancy"]:
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution, number_of_states)
                elif doLargeDefectFigure and self.defect_type not in ["none", "vacancy"]:
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution, number_of_states)
                else:
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self._compute_for_figure(m_background, m_substitution, number_of_states)
                
                LDOS -= np.min(LDOS)
                if np.max(LDOS) > 0:
                    LDOS /= np.max(LDOS)
                
                plot_ldos_ax(ldos_ax, LDOS, X, Y)
                if self.defect_type in ["none", "vacancy"]:
                    param_name = r"$m_0=$"+f"{m_background}"
                elif self.defect_type in ["substitution"]:
                    param_name = f"$m_0^{{\\text{{sub}}}}=$"+f"{m_substitution}"
                else:
                    param_name = f"$m_0^{{\\text{{int}}}}=$"+f"{m_substitution}"

                plot_spectrum_ax(spectrum_ax, eigenvalues, f"Gap = {gap:.2f}\nBott Index = {bott_index}\n"+param_name, ldos_idxs)

                if False:
                    if self.defect_type not in ["none", "vacancy"]:
                        spectrum_ax.annotate(f"$m_0^{{sub}}$ = {m_substitution}", xy=(-0.25, 0.5), xycoords='axes fraction', ha='center', fontsize=12, rotation=90, va='center')
                    else:
                        spectrum_ax.annotate(f"$m_0^{{back}}$ = {m_background}", xy=(-0.25, 0.5), xycoords='axes fraction', ha='center', fontsize=12, rotation=90, va='center')

                cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.ax.yaxis.set_ticks([0.0, 1.0])
                cbar.ax.tick_params(labelsize=20)
                cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.f'))

        
        if n_rows == 1:
            plt.subplots_adjust(top=0.8)
        else:
            plt.subplots_adjust(top=0.9)
        set_labels = [f"({lab})" for lab in "abcdefghijklmnopqrstuvwxyz"[:len(m_background_values)]]
        for i, m_background in enumerate(m_background_values):
            if n_rows == 1:
                fig.text((2*i+1)/(2 * len(m_background_values)), 0.85, set_labels[i], fontsize=36, ha='center')
            else:
                fig.text((2*i+1)/(2 * len(m_background_values)), 0.95, set_labels[i], fontsize=36, ha='center')
        plt.tight_layout()
        return fig, axs

    def plot_distances(self, idx:int = None, cmap:str = "inferno", doLargeDefectFigure:bool = False, *args, **kwargs):
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
        print(np.sort(d[idx]))
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

    def plot_defect_idxs(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')

        ax.scatter(self.X, self.Y, s=25, edgecolors='black', facecolors='none')
        for defect_index in self.defect_indices:
            defect_x = self.X[defect_index]
            defect_y = self.Y[defect_index]
            ax.scatter(defect_x, defect_y, color='red', s=30)
        plt.show()

    def plot_wannier(self, idx:int = None):
        if  idx is None:
            idx = len(self.X) // 2
        
        arrays = [self.Sx.imag, self.Sy.imag, self.Cx_plus_Cy.real]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for ax, array, label in zip(axs, arrays, ["Sx.imag", "Sy.imag", "Cx_plus_Cy.real"]):
            ax.set_title(label)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.scatter(self.X, self.Y, c=array[idx], cmap='viridis', s=25)
            ax.scatter(self.X[idx], self.Y[idx], s=100, facecolors='none', edgecolors='red')
            ax.set_aspect('equal')
        # Normalize colorbar to all plots
        vmin = min(np.min(np.real(array[idx])) for array in arrays)
        vmax = max(np.max(np.real(array[idx])) for array in arrays)
        for ax, array in zip(axs, arrays):
            sc = ax.collections[0]
            sc.set_clim(vmin, vmax)
            cbar = fig.colorbar(sc, ax=ax)
            # Set colorbar ticks to unique values in all three plots
            all_vals = np.concatenate([np.real(array[idx]).flatten() for array in arrays])
            unique_ticks = np.unique(all_vals)
            cbar.set_ticks(unique_ticks)
            cbar.set_label("Value", rotation=270, labelpad=15)
        plt.tight_layout()
        plt.show()
        
    def plot_lcm(self, m_background_values:"list[float]" = [2.5, 1.0, -1.0, -2.5], 
                             m_substitution_values:"list[float] | None" = None, doLargeDefectFigure:bool = False):
        # Get shape of the figure based on the defect type
        if m_substitution_values is None:
            m_substitution_values = np.array(m_background_values).copy()
        if self.defect_type in ["none", "vacancy"]:
            m_substitution_values = [None] if doLargeDefectFigure is False else [None] * 2
            n_cols, n_rows = len(m_background_values), len(m_substitution_values)
        else:
            n_cols, n_rows = len(m_background_values), len(m_substitution_values) - 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

        if n_rows == 1:
            axs = np.array([axs])

        for i, m_background in enumerate(m_background_values):
            good_m_sub_vals = np.array(m_substitution_values)[np.array(m_substitution_values) != m_background]
            for j, m_substitution in enumerate(good_m_sub_vals):
                if m_substitution == m_background:
                    continue

                if (j == 1 and doLargeDefectFigure and self.defect_type in ["none", "vacancy"]) or (doLargeDefectFigure and self.defect_type not in ["none", "vacancy"]):
                    H = self.LargeDefectLattice.compute_hamiltonian(m_background, m_substitution)
                    diagonal_values = np.diag(self.LargeDefectLattice.compute_local_chern_operator(H))
                    X, Y = self.LargeDefectLattice.X, self.LargeDefectLattice.Y
                else:
                    H = self.compute_hamiltonian(m_background, m_substitution)
                    diagonal_values = np.diag(self.compute_local_chern_operator(H))
                    X, Y = self.X, self.Y


                # Remove n% of the width from each side of the lattice for X, Y, and the colormap
                width = X.max() - X.min()
                height = Y.max() - Y.min()
                edge_width = 0.1
                x_min = X.min() + edge_width * width
                x_max = X.max() - edge_width * width
                y_min = Y.min() + edge_width * height
                y_max = Y.max() - edge_width * height

                mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
                X_bulk = X[mask]
                Y_bulk = Y[mask]
                diagonal_values = diagonal_values[::2][mask] + diagonal_values[1::2][mask]

                scat = axs[j, i].scatter(X_bulk, Y_bulk, s=50, c=diagonal_values, cmap='jet', edgecolors='black', linewidths=0.5)
                axs[j, i].set_aspect('equal')

                x_ticks = [X_bulk.min(), (X_bulk.min() + X_bulk.max()) / 2, X_bulk.max()]
                y_ticks = [Y_bulk.min(), (Y_bulk.min() + Y_bulk.max()) / 2, Y_bulk.max()]

                axs[j, i].set_xticks(x_ticks, minor=False)
                axs[j, i].set_xticklabels([str(int(label + 1)) for label in x_ticks], fontsize=16)
                axs[j, i].set_yticks(y_ticks, minor=False)
                axs[j, i].set_yticklabels([str(int(label + 1)) for label in y_ticks], fontsize=16)

                axs[j, i].set_xlabel(r"$X$", fontsize=20)
                axs[j, i].set_ylabel(r"$Y$", fontsize=20)

                if self.defect_type in ["none", "vacancy"]:
                    axmassname = ""
                elif self.defect_type in ["substitution"]:
                    axmassname = fr"$m_0^{{\text{{sub}}}}={m_substitution}$"
                else:
                    axmassname = fr"$m_0^{{\text{{int}}}}={m_substitution}$"
                axs[j, i].set_title(axmassname, fontsize=20)

                cbar = plt.colorbar(scat, ax=axs[j, i], orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_ticks(np.linspace(np.min(diagonal_values), np.max(diagonal_values), 5))
                cbar.ax.tick_params(labelsize=16)
                cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))

        plt.tight_layout()
        if n_rows == 1:
            plt.subplots_adjust(top=0.9)
        else:
            plt.subplots_adjust(top=0.9)
        set_labels = [f"({lab})" for lab in "abcdefghijklmnopqrstuvwxyz"[:len(m_background_values)]]
        for i, m_background in enumerate(m_background_values):
            label_xpos = axs[0, i].get_position().x0 + axs[0, i].get_position().width / 2
            label_ypos = axs[0, i].get_position().y1 + 0.07
            if n_rows == 1:
                fig.text(label_xpos, label_ypos, set_labels[i], fontsize=36, ha='center')
            else:
                fig.text(label_xpos, label_ypos, set_labels[i], fontsize=36, ha='center')

        return fig, axs

    def plot_bott_phase_diagram(self):
        pass
    # endregion

    # region Parallel Computation

    @staticmethod
    def generic_multiprocessing(func:callable, parameter_values:tuple, n_jobs:int = -1, progress_title:str = "Progress bar without a title.", doProgressBar:bool = True, *args, **kwargs):
        if doProgressBar:
            with tqdm_joblib(tqdm(total=len(parameter_values), desc=progress_title)) as progress_bar:
                data = Parallel(n_jobs=n_jobs)(delayed(func)(params, *args, **kwargs) for params in parameter_values)
        else:
            data = Parallel(n_jobs=n_jobs)(delayed(func)(params, *args, **kwargs) for params in parameter_values)
        return data
    
    def _compute_bott_from_parameters(self, parameters:tuple):
        m_background, m_substitution = parameters
        if m_substitution is None:
            m_substitution = m_background
        hamiltonian = self.compute_hamiltonian(m_background, m_substitution)
        projector = self.compute_projector(hamiltonian)
        bott_index = self.compute_bott_index(projector)
        return [m_background, m_substitution, bott_index]

    def plot_bott_phase_diagram(self, m_background_values:"list[float]", m_substitution_values:"list[float] | None" = None, num_jobs:int = -1):
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
        
        ax.set_xlabel(r"$m_0^{\text{back}}$", fontsize=18)
        ax.set_ylabel("Bott Index", fontsize=18)
        ax.set_yticks([-1, 0, 1])
        ax.set_xticks([-4, -2, 0, 2, 4])
        for tick in [-4, -2, 0, 2, 4]:
            ax.axvline(tick, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax.legend()
        plt.show()
            
    # endregion


def generate_ldos_figures(defect_types: list = ["none", "vacancy", "substitution", "interstitial", "frenkel_pair"], base_side_length: int = 24, directory:str = "./Defects/temp/LDOS/"):
    for defect_type in defect_types:
        side_length = base_side_length
        if defect_type != "interstitial":
            side_length = base_side_length + 1

        for dLDF in [False, True]:
            if defect_type == "none" and dLDF:
                continue
            if defect_type == "vacancy" and not dLDF:
                continue
            Lattice = DefectSquareLattice(side_length, defect_type)

            Lattice.plot_spectrum_ldos(doLargeDefectFigure=dLDF, number_of_states=2)

            if defect_type == "none":
                title = "SL"
            else:
                title = defect_type
            
            if dLDF and defect_type != "vacancy":
                plt.savefig(directory + f"large_{title}_LDOS.png")
            else:
                plt.savefig(directory + f"{title}_LDOS.png")

def probe_lattice_instance(defect_type:str = "interstitial", base_side_length:int = 16):
    side_length = base_side_length if defect_type == "interstitial" else base_side_length + 1
    Lattice = DefectSquareLattice(side_length, defect_type, pbc=True)

    plotHamiltonians = 0
    plotEigvals = 0
    plotEigvecs = 0
    plotCoupling = 0
    plotEigvalRange = 1

    if plotEigvecs:
        eigvecs1 = spla.eigh(H1, overwrite_a=True)[1]
        eigvecs2 = spla.eigh(H2, overwrite_a=True)[1]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        X, Y = Lattice.X, Lattice.Y
        axs[0].scatter(X, Y, c=np.real(eigvecs1[:, eigvecs1.shape[0] // 2])[0::2], cmap='inferno', s=50)
        axs[1].scatter(X, Y, c=np.real(eigvecs2[:, eigvecs2.shape[0] // 2])[1::2], cmap='inferno', s=50)

        axs[0].set_aspect('equal')
        axs[1].set_aspect('equal')
        plt.show()

    if plotHamiltonians:
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs[0, 0].imshow(np.real(H1), cmap='Spectral')
        axs[0, 0].set_title("H1 Real Part")
        axs[0, 1].imshow(np.imag(H1), cmap='Spectral')
        axs[0, 1].set_title("H1 Imaginary Part")

        axs[1, 0].imshow(np.real(H2), cmap='Spectral')
        axs[1, 0].set_title("H2 Real Part")
        axs[1, 1].imshow(np.imag(H2), cmap='Spectral')
        axs[1, 1].set_title("H2 Imaginary Part")

        for ax in axs.flat:
            ax.label_outer()
            ax.set_xticks([])
            ax.set_yticks([])
            for index in Lattice.defect_indices:
                ax.hlines(y=index-0.5, xmin=0, xmax=H1.shape[0], color='black', linestyle='--', linewidth=1.0)
                ax.hlines(y=index+0.5, xmin=0, xmax=H1.shape[0], color='black', linestyle='--', linewidth=1.0)
        plt.tight_layout()
        plt.show()

    if plotEigvalRange:
        m_back = 1.0
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        m_sub_values = np.linspace(-4.0, 4.0, 25)
        for i, m_sub in enumerate(m_sub_values):
            H1, osm1 = Lattice.compute_hamiltonian(m_back, m_sub)
            H2, osm2 = Lattice.compute_hamiltonian(-m_back, -m_sub)
            eigvals1, _ = spla.eigh(H1, overwrite_a=True)
            eigvals2, _ = spla.eigh(H2, overwrite_a=True)
            # Only get the center fifth of eigenvalues
            center = eigvals1.size // 2
            number = 10
            eigvals1 = eigvals1[center - number : center + number]
            eigvals2 = eigvals2[center - number : center + number]
            # Stack eigenvalues as rows for imshow
            if i == 0:
                eigval_matrix1 = eigvals1[np.newaxis, :]
                eigval_matrix2 = eigvals2[np.newaxis, :]
            else:
                eigval_matrix1 = np.vstack([eigval_matrix1, eigvals1[np.newaxis, :]])
                eigval_matrix2 = np.vstack([eigval_matrix2, eigvals2[np.newaxis, :]])

            if i == 0:
                osm_matrix1 = osm1[np.newaxis, :]
                osm_matrix2 = osm2[np.newaxis, :]
            else:
                osm_matrix1 = np.vstack([osm_matrix1, osm1[np.newaxis, :]])
                osm_matrix2 = np.vstack([osm_matrix2, osm2[np.newaxis, :]])
        # Show the eigenvalue matrix as an image: each row is the spectrum for a given m_sub
        im1 = axs[0].imshow(eigval_matrix1, aspect='auto', cmap='viridis', origin='lower',
                           extent=[0, eigval_matrix1.shape[1], -2.5, 2.5])
        im2 = axs[1].imshow(eigval_matrix2, aspect='auto', cmap='viridis', origin='lower',
                           extent=[0, eigval_matrix2.shape[1], -2.5, 2.5])
        im3 = axs[2].imshow(eigval_matrix1 - eigval_matrix2, aspect='auto', cmap='viridis', origin='lower',
                           extent=[0, eigval_matrix1.shape[1], -2.5, 2.5])

        for ax in axs.flatten()[[0, 1, 2]]:
            ax.set_xlabel('Eigenvalue Index')
            ax.set_ylabel('m_sub Value Index')
        axs[0].set_title(f'Eigenvalues vs m_sub $(m_{{back}}={m_back})$\n(each row is spectrum)')
        axs[1].set_title(f'Eigenvalues vs -m_sub $(m_{{back}}={m_back})$\n(each row is spectrum)')
        axs[2].set_title('Difference of Eigenvalues (H1 - H2)')
        plt.colorbar(im1, ax=axs[0], orientation='vertical', label='Eigenvalue')
        plt.colorbar(im2, ax=axs[1], orientation='vertical', label='Eigenvalue')
        plt.colorbar(im3, ax=axs[2], orientation='vertical', label='Eigenvalue Difference (H1 - H2)')

        plt.tight_layout()
        
        fig2, axs2 = plt.subplots(2, 3, figsize=(10, 5))
        arrs = [H1, H2, H1 - H2]
        titles = ["H1", "H2", "(H1 - H2)"]
        ims = []
        for i, (arr, title) in enumerate(zip(arrs, titles)):
            im0 = axs2[0, i].imshow(np.real(arr), cmap='Spectral')
            axs2[0, i].set_title(f"{title} Real Part")
            im00 = axs2[1, i].imshow(np.imag(arr), cmap='Spectral')
            axs2[1, i].set_title(f"{title} Imaginary Part")
            ims.append(im0)
            ims.append(im00)
            for ax in axs2[:, i]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        for im, ax in zip(ims, axs2.flatten()):
            cbar = fig2.colorbar(im, ax=ax, orientation='vertical')
            cbar.set_label("Value", rotation=270, labelpad=15)
            im.set_clim(-1, 1)

        plt.tight_layout()
        plt.show()

    if plotEigvals:
        eig1, eigvec1 = spla.eigh(H1, overwrite_a=True)
        eig2, eigvec2 = spla.eigh(H2, overwrite_a=True)
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        axs[0].scatter(np.arange(len(eig1)), eig1, label='H1 Eigenvalues', color='black', s=25)
        axs[1].scatter(np.arange(len(eig2)), eig2, label='H2 Eigenvalues', color='black', s=25)
        axs[2].scatter(np.arange(len(eig1)), eig1 - eig2, label='H1 - H2 Eigenvalues Difference', color='black', s=25)
        titles = ["Eigenvalues of H1", "Eigenvalues of H2", "Difference of Eigenvalues (H1 - H2)"]
        for ax, title in zip(axs, titles):

            ax.set_xlabel('State Index')
            ax.set_ylabel('Eigenvalue')
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

    if plotCoupling:
        for index in Lattice.LargeDefectLattice.defect_indices:
            fig, axs = plt.subplots(1, 2, figsize=(8, 8))
            axs[0].set_aspect('equal')
            axs[1].set_aspect('equal')
            #ax.scatter(Lattice.LargeDefectLattice.X, Lattice.LargeDefectLattice.Y, s=25, edgecolors='black', facecolors='none')
            # Prepare the data for imshow
            x = Lattice.LargeDefectLattice.X.astype(int)
            y = Lattice.LargeDefectLattice.Y.astype(int)
            values1 = abs((H1[index][::2] + H1[index][1::2]).real)
            values1 = abs((H2[index][::2] + H2[index][1::2]).real)
            # Create a 2D grid for imshow for both Hamiltonians
            grid_shape = (y.max() + 1, x.max() + 1)
            value_grid1 = np.full(grid_shape, np.nan)
            value_grid2 = np.full(grid_shape, np.nan)
            values1 = ((H1[index][::2] + H1[index][1::2]).real)
            values2 = ((H2[index][::2] + H2[index][1::2]).real)
            value_grid1[y, x] = values1
            value_grid2[y, x] = values2

            im1 = axs[0].imshow(value_grid1, origin='lower', cmap='viridis',
                        extent=[x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5])
            im2 = axs[1].imshow(value_grid2, origin='lower', cmap='viridis',
                        extent=[x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5])

            axs[0].scatter(x[index], y[index], c='red', s=50)
            axs[1].scatter(x[index], y[index], c='red', s=50)

            cbar1 = plt.colorbar(im1, ax=axs[0], orientation='vertical')
            cbar2 = plt.colorbar(im2, ax=axs[1], orientation='vertical')
            cbar1.set_label("Value (H1)", rotation=270, labelpad=15)
            cbar2.set_label("Value (H2)", rotation=270, labelpad=15)
            # Normalize colorbars to range -1 to 1
            im1.set_clim(-1, 1)
            im2.set_clim(-1, 1)
            cbar1.set_ticks([-1, 0, 1])
            cbar2.set_ticks([-1, 0, 1])

            for ax in axs:
                ax.set_xticks(np.arange(x.min() - 0.5, x.max() + 1, 1.0), minor=False)
                ax.set_yticks(np.arange(y.min() - 0.5, y.max() + 1, 1.0), minor=False)
                ax.grid(which='major', color='black', linestyle='--', linewidth=1.0)
        plt.show()

def generate_lcm_figures(defect_types: list = ["none", "vacancy", "substitution", "interstitial", "frenkel_pair"], base_side_length: int = 24, directory:str = "./Defects/temp/LDOS/"):
    for defect_type in defect_types:
        side_length = base_side_length
        if defect_type != "interstitial":
            side_length = base_side_length + 1

        for dLDF in [True, False]:
            if defect_type == "none" and dLDF:
                continue
            if defect_type == "vacancy" and not dLDF:
                continue
            Lattice = DefectSquareLattice(side_length, defect_type, pbc=True)
            Lattice.plot_lcm(doLargeDefectFigure=dLDF)

            if defect_type == "none":
                title = "SL"
            else:
                title = defect_type
            if dLDF and defect_type != "vacancy":
                plt.savefig(directory + f"large_{title}_LCM.png")
            else:
                plt.savefig(directory + f"{title}_LCM.png")


if __name__ == "__main__":
    pass
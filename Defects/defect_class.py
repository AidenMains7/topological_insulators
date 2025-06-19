import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import matplotlib.ticker as ticker
import scipy.linalg as spla
from scipy.sparse import dok_matrix
import matplotlib.lines as mlines

class DefectSquareLattice:
    def __init__(self, side_length:int, defect_type:str, pbc:bool = True, frenkel_pair_index:int = None, doLargeDefect:bool = False, *args, **kwargs):
        self._pbc = pbc
        self._side_length = side_length
        self._defect_type = defect_type
        self._doLargeDefect = doLargeDefect
        self._frenkel_pair_index = frenkel_pair_index if frenkel_pair_index is not None else None

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
                if self.doLargeDefect:
                    self._defect_indices = [np.max(self.lattice) // 2, 
                                            np.max(self.lattice) // 2 + 1, 
                                            np.max(self.lattice) // 2 - 1,
                                            np.max(self.lattice) // 2 + self.side_length + 2,
                                            np.max(self.lattice) // 2 - self.side_length - 2,]
                else:
                    self._defect_indices = [np.max(self.lattice) // 2]
            case "frenkel_pair":
                if frenkel_pair_index is None:
                    raise ValueError("frenkel_pair_index must be provided for 'frenkel_pair' defect type.")
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

    def generate_interstitial_lattice(self, *args, **kwargs):
        if self.side_length % 2 == 1:
            raise ValueError("Side length must be even for a single interstitial in the center.")
        Y, X = np.where(self._pristine_lattice >= 0)
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        coordinates = np.array([X, Y])

        if self._doLargeDefect:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if abs(i) + abs(j) == 2:
                        continue
                    coordinates = np.concatenate((coordinates, np.array([[x_mean + i], [y_mean + j]])), axis=1)
        else:
            coordinates = np.concatenate((coordinates, np.array([[x_mean], [y_mean]])), axis=1)

        coordinates = (coordinates * 2).astype(int)

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
        distance_mask = (dr <= 1.0) & (dr > 0.0)

        principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & (distance_mask)
        diagonal_mask  = ((np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0))) & (distance_mask)
        all_mask = principal_mask | diagonal_mask

        d_cos = np.where(all_mask, np.cos(theta), 0. + 0.j)
        d_sin = np.where(all_mask, np.sin(theta), 0. + 0.j)

        amplitude = np.where(distance_mask, np.exp(1 - dr), 0. + 0.j)

        Cx_plus_Cy = amplitude / 2
        Sx = 1j * d_cos * amplitude / 2
        Sy = 1j * d_sin * amplitude / 2

        if False and self.defect_index is not None:
            plt.scatter(self.X, self.Y, c=amplitude[self.defect_index, :].astype(float))
            plt.colorbar()
            plt.show()

        self._Cx_plus_Cy = Cx_plus_Cy
        self._Sx = Sx
        self._Sy = Sy
        self._I = np.eye(Sx.shape[0], dtype=complex)

    def compute_hamiltonian(self, M_background:float, M_substitution:float = None, t:float = 1.0, t0:float = 1.0, *args, **kwargs):

        if self.defect_type in ["vacancy", "none"]:
            onsite_mass = M_background * self.I
        elif self.defect_type in ["substitution", "interstitial", "frenkel_pair"]:
            if M_substitution is None:
                raise ValueError("M_substitution must be provided for 'substitution', 'interstitial', or 'frenkel pair' defects.")
            onsite_mass = M_background * self.I
            for defect_index in self.defect_indices:
                onsite_mass[defect_index, defect_index] = M_substitution

        d1 = t * self.Sx 
        d2 = t * self.Sy
        d3 = onsite_mass + t0 * (self.Cx_plus_Cy)
        hamiltonian = np.kron(d1, self.pauli_matrices[0]) + np.kron(d2, self.pauli_matrices[1]) + np.kron(d3, self.pauli_matrices[2])
        return hamiltonian

    def compute_projector(self, hamiltonian, *args, **kwargs):
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2]
        highest_lower_band = lower_band[-1]

        D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
        D_herm_conj = np.einsum('i,ij->ij', D, eigenvectors.conj().T)

        projector = eigenvectors @ D_herm_conj
        return projector

    def compute_bott_index(self, projector:np.ndarray, *args, **kwargs):
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

    def compute_LDOS(self, hamiltonian:np.ndarray, number_of_states:int = 2, plotSpectrum:bool = False, *args, **kwargs):
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        number_of_states += number_of_states % 2
        mid_index = len(eigenvalues) // 2
        lower_idxs = np.arange(len(eigenvalues))[:mid_index][-number_of_states // 2:]
        upper_idxs = np.arange(len(eigenvalues))[mid_index:][:number_of_states // 2]
        selected_indices = np.concatenate((lower_idxs, upper_idxs))

        if plotSpectrum:
            fig, ax = plt.subplots()
            if False:
                ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label='Spectrum')
            else:
                ax.plot(eigenvalues, 'o-', label="Spectrum")
            ax.scatter(selected_indices, eigenvalues[selected_indices], color='red', label='Selected States', zorder=3)
            ax.set_title('Energy Spectrum with Selected States Highlighted')
            ax.set_xlabel('State Index')
            ax.set_ylabel('Energy')
            ax.legend()
            plt.show()

        LDOS = np.sum(np.abs(eigenvectors[:, selected_indices])**2, axis=1)
        LDOS = LDOS[0::2] + LDOS[1::2]
        LDOS = LDOS / np.sum(LDOS)
        gap = abs(np.min(eigenvalues[lower_idxs]) - np.max(eigenvalues[upper_idxs]))
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

    def _compute_for_figure(self, m_background:float, m_substitution:float):
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
            ldos_dict = self.compute_LDOS(hamiltonian, number_of_states=2)
            LDOS, eigenvalues, gap, bandwidth, ldos_idxs = ldos_dict["LDOS"], ldos_dict["eigenvalues"], ldos_dict["gap"], ldos_dict["bandwidth"], ldos_dict["ldos_idxs"]
            X, Y = self.X, self.Y
        return LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs

    def plot_spectrum_ldos(self, m_background_values:"list[float]" = [2.5, 1.0, -1.0, -2.5], 
                             m_substitution_values:"list[float] | None" = None, doLargeDefectFigure:bool = False):
        
        def plot_ldos_ax(ldos_ax:plt.Axes, LDOS, X, Y):
            ldos_ax.scatter(X, Y, c=LDOS, s= 25, cmap='inferno')
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
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution)
                elif doLargeDefectFigure and self.defect_type not in ["none", "vacancy"]:
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution)
                else:
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self._compute_for_figure(m_background, m_substitution)
                
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

        plt.tight_layout()
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
        return fig, axs

    def plot_distances(self, idx:int = None, cmap:str = "inferno", doLargeDefectFigure:bool = False, *args, **kwargs):
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))

        if idx is None:
            idx = len(self.X) // 2

        if doLargeDefectFigure:
            dx, dy, d = self.LargeDefectLattice.dx, self.LargeDefectLattice.dy, np.sqrt(self.LargeDefectLattice.dx**2 + self.LargeDefectLattice.dy**2)
            X, Y = self.LargeDefectLattice.X, self.LargeDefectLattice.Y
            lattice = self.LargeDefectLattice.lattice
        else:
            dx, dy, d = self.dx, self.dy, np.sqrt(self.dx**2 + self.dy**2)
            X, Y = self.X, self.Y
            lattice = self.lattice

        distances = [dx, dy, d]
        labels = ["dx", "dy", "d"]
        for i, (distance, label) in enumerate(zip(distances, labels)):
            axs[i].set_title(label)
            axs[i].set_xlabel("X")
            axs[i].set_ylabel("Y")
            axs[i].scatter(X, Y, c=distance[idx], cmap=cmap, zorder=0, s=25)
            axs[i].scatter(X[idx], Y[idx], c='red', s=50, zorder=1)
            axs[i].set_aspect('equal')

        cbar = fig.colorbar(axs[i].collections[0], ax=axs[i], orientation='vertical')
        cbar.set_label("Distance to site {}".format(idx), rotation=270, labelpad=15)
        print(lattice)
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

def BI_PD():
    d_type = "none"
    sys_size = 15
    Lattice = DefectSquareLattice(sys_size, d_type, frenkel_pair_index=None)
    bott_vals = []
    for M in np.linspace(-4., 4., 51):
        H = Lattice.compute_hamiltonian(M, None)
        projector = Lattice.compute_projector(H)
        bott_index = Lattice.compute_bott_index(projector)
        bott_vals.append(bott_index)

    fig, ax = plt.subplots()
    ax.scatter(np.linspace(-4., 4., 51), bott_vals, s=25, color='black')
    ax.set_xlabel(r"$m_0$", fontsize=18)
    ax.set_ylabel("Bott Index", fontsize=18)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([-4, -2, 0, 2, 4])
    for tick in [-4, -2, 0, 2, 4]:
        ax.axvline(tick, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig("BI_PD.png")
    plt.show()

def main():
    for method in ["none", "vacancy"]:
        if method in ["none", "vacancy", "substitution", "frenkel_pair"]:
            side_length = 25
        else:
            side_length = 24
        for flag in [True, False]:
            if flag and method in ["none", "frenkel_pair"]:
                continue
            if not flag and method in ["vacancy"]:
                continue

            SDLattice = DefectSquareLattice(side_length, method, frenkel_pair_index=0)
            SDLattice.plot_spectrum_ldos(doLargeDefectFigure=flag)
            if flag:
                plt.savefig(f"large_{method}_LDOS.png")
            else:
                plt.savefig(f"{method}_LDOS.png")

if __name__ == "__main__":
    BI_PD()
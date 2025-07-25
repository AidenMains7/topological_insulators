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
from matplotlib.patches import Rectangle


class DefectSquareLattice:
    def __init__(self, side_length:int, defect_type:str, pbc:bool = True, frenkel_pair_index:int = 0, doLargeDefect:bool = False, schottky_distance:int = None):
        self._pbc = pbc
        self._side_length = side_length
        self._defect_type = defect_type
        self._doLargeDefect = doLargeDefect
        self._frenkel_pair_index = frenkel_pair_index
        self._schottky_distance = schottky_distance

        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._pauli_matrices = [pauli_x, pauli_y, pauli_z]

        self._pristine_lattice = self.generate_lattice() 
        if self.defect_type == "interstitial" and self.side_length % 2 == 1:
            raise ValueError("Side length must be even for interstitial defect.")
        elif self.defect_type == "schottky" and (self.side_length + self.schottky_distance) % 2 != 1:
            raise ValueError("Side length + schottky distance must be odd for schottky defect. It is {}.".format(self.side_length + self.schottky_distance))
        elif self.defect_type not in ["interstitial", "schottky"] and self.side_length % 2 == 0:
            raise ValueError("Side length must be odd for non-interstitial defects.")
        
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
            case "schottky":
                self._lattice, self._defect_indices = self.generate_schottky_lattice()

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
    def generate_lattice(self, *args, **kwargs):
        return np.arange(self.side_length**2).reshape((self.side_length, self.side_length))

    def generate_vacancy_lattice(self, *args, **kwargs):
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

    def generate_interstitial_lattice(self):
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

    def generate_schottky_lattice(self):
        lattice = self._pristine_lattice.copy()
        up_pos = self.side_length // 2 + self.schottky_distance // 2
        down_pos = self.side_length // 2 - self.schottky_distance // 2
        if self.side_length % 2 == 0:
            down_pos -= 1
        up_parity_idx = lattice[up_pos, up_pos]
        down_parity_idx = lattice[down_pos, down_pos]

        self._vacant_positions = [(up_pos, up_pos), (down_pos, down_pos)]
        return lattice, [up_parity_idx, down_parity_idx]

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

        distance_mask = ((dr <= 1 + 1e-6) & (dr > 1e-6))
        principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & distance_mask
        diagonal_mask  = ((np.isclose(np.abs(dx), np.abs(dy), atol=1e-4)) & (dx != 0)) & distance_mask
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

        if self.defect_type in ["substitution", "interstitial", "frenkel_pair", "schottky"]:
            if M_substitution is None:
                raise ValueError("M_substitution must be provided for 'substitution', 'interstitial', or 'frenkel pair' defects.")
            onsite_mass = M_background * self.I
            onsite_mass[self.defect_indices, self.defect_indices] = M_substitution
        else:
            onsite_mass = M_background * self.I

        if M_substitution is None:
            M_substitution = 1.

        # Maintain symmetry for interstitial and frenkel_pair defects
        if self.defect_type not in ["interstitial", "frenkel_pair"]:
            d1 = t * self.Sx
            d2 = t * self.Sy
            d3 = onsite_mass + t0 * (self.Cx_plus_Cy)
        else:
            d1 = t * self.Sx
            d2 = t * self.Sy
            d3 = onsite_mass + t0 * (self.Cx_plus_Cy) #* np.sign(M_substitution)

        hamiltonian = np.kron(d1, self.pauli_matrices[0]) + np.kron(d2, self.pauli_matrices[1]) + np.kron(d3, self.pauli_matrices[2])


        if False:
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            for i, (ax, data, title, pauli) in enumerate(zip(axs, [d1, d2, d3, hamiltonian], [f"real + imag\nnp.kron(d{i+1}, pauli{i+1})" for i in range(3)] + ["Hamiltonian"], self.pauli_matrices + [None])):
                ax.set_title(title)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")


                index = 8
                if True and i < 3:
                    data = np.kron(data, pauli)[index] + np.kron(data, pauli)[index + 1]
                    data = data[::2] + data[1::2]
                    colors = data

                    print(np.sum(data))
                elif i == 3:
                    data = hamiltonian[index] + hamiltonian[index + 1]
                    colors = data[::2] + data[1::2]
                    colors = colors.real
                else:
                    colors = data[index] + data[index + 1]

                colors = colors.real + colors.imag

                ax.scatter(self.X, self.Y, c=colors, cmap='jet', edgecolors='black', linewidths=0.5)
                ax.set_aspect('equal')
                cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical').set_label("Amplitude")
                ax.collections[0].set_clim(-1, 1)
            plt.tight_layout()
            plt.show()


        if self.defect_type == "schottky":
            hamiltonian[self.defect_indices[0] * 2 + 1, self.defect_indices[0] * 2 + 1] = 0
            hamiltonian[self.defect_indices[1] * 2, self.defect_indices[1] * 2] = 0
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

    def _compute_for_figure_disorder(self, m_background:float, m_substitution:float, number_of_states:float, disorder_strength:float, n_iterations:int = 10):
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
                    
            elif len(self.defect_indices) == 2:
                for x, y in self._vacant_positions:
                    axes.scatter(x, y, s=large_site_size, edgecolors='none', facecolors='white')
                axes.scatter(self.X[self.defect_indices[0]], self.Y[self.defect_indices[0]], s=standard_site_size, facecolors = 'none', edgecolors = 'red', linewidth = 1.5)
                axes.scatter(self.X[self.defect_indices[1]], self.Y[self.defect_indices[1]], s=standard_site_size, facecolors = 'none', edgecolors = 'blue', linewidth = 1.5)
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
        #plt.savefig(f"{self.defect_type}_lattice.png")

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

    def plot_spectrum_ldos(self, doLargeDefectFigure:bool = False, doDisorder:bool = False, n_iterations:int = 10, doInterpolation:bool = True):
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
                xy=(0.95, 0.025),
                xycoords='axes fraction',
                ha='right',
                va='bottom',
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.0)
            )

        def plot_ldos_ax(ax:plt.Axes, LDOS, X, Y, doScatter:bool = False):
            if doScatter:


                inset_ax = inset_axes(
                    ax,
                    width="100%",  # width as a percentage of parent
                    height="100%",  # height as a percentage of parent
                    bbox_to_anchor=(0.1, 0.60, 0.375, 0.375),  # (x0, y0, width, height) in axes fraction
                    bbox_transform=ax.transAxes,
                    loc='upper left',
                    borderpad=0
                )
                cax = inset_axes(
                    inset_ax, 
                    width="5%",  # width as a percentage of parent
                    height="100%",  # height as a percentage of parent
                    bbox_to_anchor=(0.1, 0.3, 1, 0.6),  # (x0, y0, width, height) in axes fraction
                    bbox_transform=inset_ax.transAxes,
                    borderpad = 0.0
                )

                if self.side_length <= 15:
                    dot_size = 25
                elif self.side_length <= 20:
                    dot_size = 10
                else:
                    dot_size = 5

                scat = inset_ax.scatter(X, Y, c=LDOS, s=dot_size, cmap='jet')
                inset_ax.set_xticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
                inset_ax.set_yticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
                inset_ax.set_xticklabels([str(int(np.min(X) + 1)), "$L_x$", str(int(np.max(X) + 1))], fontsize=14)
                inset_ax.set_yticklabels([str(int(np.min(X) + 1)), "$L_y$", str(int(np.max(X) + 1))], fontsize=14)
                inset_ax.set_aspect('equal')

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
                
                inset_ax.set_facecolor((1, 1, 1, 0.8))
                inset_ax.set_zorder(10)
                return inset_ax
            else:

                if doInterpolation:
                    # Interpolate LDOS onto a finer grid for smoother visualization
                        grid_res = 201  # resolution of the interpolation grid
                        xi = np.linspace(np.min(X), np.max(X), grid_res)
                        yi = np.linspace(np.min(Y), np.max(Y), grid_res)
                        XI, YI = np.meshgrid(xi, yi)

                        points = np.column_stack((X, Y))
                        LDOS_interp = griddata(points, LDOS, (XI, YI), method='linear', fill_value=0)
                        # Ensure LDOS_interp has the same min/max as the original LDOS
                        ldos_min, ldos_max = np.min(LDOS), np.max(LDOS)
                        interp_min, interp_max = np.min(LDOS_interp), np.max(LDOS_interp)
                        # Rescale only if the interpolated range is different and not constant
                        if interp_max > interp_min and ldos_max > ldos_min:
                            LDOS_interp = (LDOS_interp - interp_min) / (interp_max - interp_min)
                            LDOS_interp = LDOS_interp * (ldos_max - ldos_min) + ldos_min
                        X, Y, LDOS = XI.ravel(), YI.ravel(), LDOS_interp.ravel()

                # Do 3d plot
                box = ax.get_position()
                surf_ax = fig.add_axes([box.x0, box.y0 + box.height * 0.55, box.width * 0.5, box.height * 0.5], projection='3d')
                surf = surf_ax.plot_trisurf(X, Y, LDOS, cmap='inferno', linewidth=0.2, antialiased=False)
                surf_ax.set_xticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
                surf_ax.set_yticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
                surf_ax.set_xticklabels([str(int(np.min(X) + 1)), "$L_x$", str(int(np.max(X) + 1))], fontsize=14)
                surf_ax.set_yticklabels([str(int(np.min(X) + 1)), "$L_y$", str(int(np.max(X) + 1))], fontsize=14)
                surf.set_clim(vmin=0)

                surf_ax.set_zticklabels([])
                surf_ax.set_zlabel("")
                surf_ax.set_facecolor((1, 1, 1, 0))
                surf_ax.grid(False)
                # Remove the color of the pane (make it fully transparent)
                #surf_ax.xaxis.set_pane_color((1, 1, 1, 0))
                #surf_ax.yaxis.set_pane_color((1, 1, 1, 0))
                #surf_ax.zaxis.set_pane_color((1, 1, 1, 0))

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
        

        m_background_values = [2.5, 1.0, -1.0, -2.5]
        m_substitution_values = [2.5, 1.0, -1.0, -2.5]

        if self.defect_type in ["none", "vacancy"]:
            m_substitution_values = [None] if doLargeDefectFigure is False else [None] * 2
            n_cols, n_rows = len(m_background_values), len(m_substitution_values)
        else:
            n_cols, n_rows = len(m_background_values), len(m_substitution_values) - 1

        scale = 6
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(scale * n_cols, scale * n_rows))
        plt.subplots_adjust(wspace = 0.4)

        if self.defect_type not in ["none", "vacancy"]:
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
    

                if i == 1 and doLargeDefectFigure and self.defect_type in ["vacancy"]:
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

                if self.defect_type in ["none", "vacancy"]:
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
    
    def compare_interpolation(self, doInterpolation:bool = True, doGaussianBlur:bool = False):
        def plot_ldos_ax(ax:plt.Axes, LDOS, X, Y, doInterpolation:bool, doGaussianBlur:bool):
            if doInterpolation and not doGaussianBlur:
                # Interpolate LDOS onto a finer grid for smoother visualization
                    grid_res = self.side_length * 3 + (self.side_length + 1) % 2
                    xi = np.linspace(np.min(X), np.max(X), grid_res)
                    yi = np.linspace(np.min(Y), np.max(Y), grid_res)
                    XI, YI = np.meshgrid(xi, yi)

                    points = np.column_stack((X, Y))
                    LDOS_interp = griddata(points, LDOS, (XI, YI), method='linear', fill_value=0)

                    ldos_min, ldos_max = np.min(LDOS), np.max(LDOS)
                    interp_min, interp_max = np.min(LDOS_interp), np.max(LDOS_interp)
                    LDOS_interp *= ldos_max / interp_max

                    X, Y, LDOS = XI.ravel(), YI.ravel(), LDOS_interp.ravel()

            if doGaussianBlur and not doInterpolation:
                sigma = 1.0
                LDOS_blurred = gaussian_filter(LDOS, sigma=sigma)
                LDOS_blurred *= np.max(LDOS) / np.max(LDOS_blurred)
                LDOS = LDOS_blurred

            surf = ax.plot_trisurf(X, Y, LDOS, cmap='inferno', linewidth=0.2, antialiased=False)
            ax.set_xticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
            ax.set_yticks([np.min(X), (np.max(X) + np.min(X)) // 2, np.max(X)])
            ax.set_xticklabels([str(int(np.min(X) + 1)), "$L_x$", str(int(np.max(X) + 1))], fontsize=14)
            ax.set_yticklabels([str(int(np.min(X) + 1)), "$L_y$", str(int(np.max(X) + 1))], fontsize=14)
            surf.set_clim(vmin=0)
            #ax.view_init(elev=90, azim=-90)

            ax.set_zticklabels([])
            ax.set_zlabel("")
            ax.set_facecolor((1, 1, 1, 0))
            ax.grid(False)
            # Remove the color of the pane (make it fully transparent)
            #surf_ax.xaxis.set_pane_color((1, 1, 1, 0))
            #surf_ax.yaxis.set_pane_color((1, 1, 1, 0))
            #surf_ax.zaxis.set_pane_color((1, 1, 1, 0))

            cax = inset_axes(
                ax, 
                width="7.5%",  # width as a percentage of parent
                height="100%",  # height as a percentage of parent
                bbox_to_anchor=(0.1, 0.425, 1, 0.4),  # (x0, y0, width, height) in axes fraction
                bbox_transform=ax.transAxes,
                borderpad = 0.0
            )
            cbar = fig.colorbar(ax.collections[0], cax=cax)
            formatter = ticker.ScalarFormatter(useMathText = True)
            formatter.set_powerlimits((0,  0))
            formatter.set_scientific(True)
            formatter.format = "%.1f"
            cbar.formatter = formatter
            cbar.update_ticks()

            cbar.ax.yaxis.offsetText.set_position((2.0, 1.0))
            cbar.ax.yaxis.offsetText.set_fontsize(14)
            cbar.ax.tick_params(labelsize=14)

            return ax
    
        if self.defect_type not in ["interstitial", "substitution"]:
            raise ValueError
        
        n_rows, n_cols = 3, 8
        scale = 8
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale), subplot_kw={'projection': '3d'})


        for i, m_background in enumerate([2.5, 1.0, -1.0, -2.5]):
            good_m_sub_vals = [2.5, 1.0, -1.0, -2.5]
            good_m_sub_vals = np.array(good_m_sub_vals)[np.array(good_m_sub_vals) != m_background]
            for j, m_substitution in enumerate(good_m_sub_vals):
                if (j == 1 and self.defect_type in ["vacancy"]) or (self.defect_type not in ["vacancy"]):
                    hamiltonian = self.LargeDefectLattice.compute_hamiltonian(m_background, m_substitution)
                    LDOS = self.LargeDefectLattice.compute_LDOS(hamiltonian)["LDOS"]
                    X, Y = self.LargeDefectLattice.X, self.LargeDefectLattice.Y
                else:
                    hamiltonian = self.compute_hamiltonian(m_background, m_substitution)
                    LDOS = self.compute_LDOS(hamiltonian)["LDOS"]
                    X, Y = self.X, self.Y
                
                regular_ax = axs[j, 2 * i]
                interp_ax = axs[j, 2 * i + 1]
                plot_ldos_ax(regular_ax, LDOS, X, Y, doInterpolation=False, doGaussianBlur=False)
                plot_ldos_ax(interp_ax, LDOS, X, Y, doInterpolation, doGaussianBlur)

                title_param = "interpolation" if doInterpolation else "Gaussian Blur"
                regular_ax.set_title("" + f"$m_0^{{\\text{{back}}}}={m_background}$\n$m_0^{{\\text{{sub}}}}={m_substitution}$\nWithout {title_param}", fontsize=16)
                interp_ax.set_title("" + f"$m_0^{{\\text{{back  }}}}={m_background}$\n$m_0^{{\\text{{sub}}}}={m_substitution}$\nWith {title_param}", fontsize=16)

        plt.subplots_adjust(wspace=.4, hspace=.4)
        
        title_param = "interpolation" if doInterpolation else "Gaussian Blur"
        fig.suptitle(f"Comparison of LDOS with and without {title_param}\n{self.defect_type.capitalize()}", fontsize=20)

        for i in range(4):
            if i != 3:
                pos0 = axs[0, 2 * i + 1].get_position()
                pos1 = axs[0, 2 * i + 2].get_position()
                x_pos = pos0.x1 + (pos1.x0 - pos0.x1) / 2
                fig.lines.append(plt.Line2D([x_pos, x_pos], [0, pos0.y1], color='black', linestyle='-', linewidth=2, transform=fig.transFigure, zorder=10))


        plt.savefig("temp2.png")
        
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
        
        ax.set_xlabel(r"$m_0$", fontsize=20)
        ax.set_ylabel("Bott Index", fontsize=20)
        ax.set_yticks([-1, 0, 1])
        xticks = [-4, -2, 0, 2, 4]
        ax.set_xticks(xticks)

        phase_labels = ["Trivial", "$\\Gamma$ Phase", "$M$ Phase", "Trivial"]
        cmap = plt.get_cmap("viridis")
        colors = ['white', cmap(0.5), cmap(0.0), 'white']

        for i in range(len(xticks) - 1):
            #ax.axvspan(xticks[i], xticks[i+1], color=colors[i], alpha=0.15)
            ax.text((xticks[i] + xticks[i+1]) / 2, 0.8, phase_labels[i], fontsize=20, ha='center', va='center', color='black')

        ax.tick_params(axis='both', labelsize=18)
        for tick in [-4, -2, 0, 2, 4]:
            ax.axvline(tick, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        return fig, ax
    # endregion


def generate_figures(lcm_or_ldos:str, defect_types: list = ["none", "vacancy", "substitution", "interstitial", "frenkel_pair"], base_side_length: int = 24, schottky_distance: int = 3, 
                     directory:str = "./Defects/Plots/LDOS/", doDisorder:bool = False, n_iterations:int = 10):
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

            if dLDF and defect_type != "vacancy":
                plt.savefig(directory + f"large_{title}_" + lcm_or_ldos.upper() + ".png")
            else:
                plt.savefig(directory + f"{title}_" + lcm_or_ldos.upper() + ".png")

            print(f"Saved figure for {defect_type} with dLDF={dLDF} in {directory + f'{title}_{lcm_or_ldos.upper()}.png'}")


def compare_gap(side_length, defect_type, doLargeDefect:bool = False):
    PristineLattice = DefectSquareLattice(side_length + side_length % 2 - 1, "none", pbc=True)
    DefectLattice = DefectSquareLattice(side_length, defect_type, pbc=True)

    M_back_values = np.concatenate((np.linspace(-4.0, 4.0, 51), [-4.0, -2.0, 0.0, 2.0, 4.0]))
    M_back_values = np.unique(np.sort(M_back_values))
    M_sub_values = [-2.5, -1.0, 1.0, 2.5] if defect_type not in ["vacancy"] else [None]
    parameters = tuple(product(M_back_values, M_sub_values))

    def worker(params):
        M_back, M_sub = params
        _, _, gap_pristine, _, _, _, _ = PristineLattice._compute_for_figure(M_back, M_sub, 2)
        if not doLargeDefect:
            _, _, gap_defect, _, _, _, _ = DefectLattice._compute_for_figure(M_back, M_sub, 2)
        else:
            _, _, gap_defect, _, _, _, _ = DefectLattice.LargeDefectLattice._compute_for_figure(M_back, M_sub, 2)
        return [M_back, M_sub, gap_pristine, gap_defect]
    
    with tqdm_joblib(tqdm(total=len(list(parameters)), desc="Computing gaps")) as progress_bar:
        data = Parallel(n_jobs=-1)(delayed(worker)(params) for params in parameters)
    
    return data


def plot_gap_comparison(side_length, defect_type, doLargeDefectFigure:bool = False):
    data = compare_gap(side_length, defect_type, doLargeDefect=doLargeDefectFigure)

    mback_vals, msub_vals, gap_pristine, gap_defect = np.array(data).T

    if defect_type != "vacancy":
        n_msub = len(np.unique(msub_vals))
    else:
        n_msub = 1

    fig, axs = plt.subplots(n_msub, 1, figsize=(10, 4 * n_msub), sharex=True)
    if n_msub == 1:
        axs = np.array([axs])
        unique_msub = [None]
    else:
        unique_msub = np.unique(msub_vals)

    for i, m_sub in enumerate(unique_msub):
        if m_sub is None:
            mask = np.arange(len(msub_vals))  # No substitution, use all values
        else:
            mask = msub_vals == m_sub

        axs[i].scatter(mback_vals[mask], gap_pristine[mask], s=25, label="Pristine", color='blue', alpha=0.5)
        if m_sub is not None:
            axs[i].scatter(mback_vals[mask], gap_defect[mask], s=25, label=f"Defect $m_0^{{\\text{{sub}}}}={m_sub}$", color='red', alpha=0.5)
        else:
            axs[i].scatter(mback_vals[mask], gap_defect[mask], s=25, label="Defect", color='red', alpha=0.5)

        axs[i].set_xlabel(r"$m_0$", fontsize=20)
        axs[i].set_ylabel("Gap", fontsize=20)
        axs[i].legend()

        xticks = [-4, -2, 0, 2, 4]
        axs[i].set_xticks(xticks)
        axs[i].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        for tick in xticks:
            axs[i].axvline(tick, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    if doLargeDefectFigure:
        fig.suptitle(f"Comparison of Gap for large {defect_type} defect\nversus pristine bulk gap", fontsize=20)
    else:
        fig.suptitle(f"Comparison of Gap for {defect_type} defect\nversus pristine bulk gap", fontsize=20)
    plt.tight_layout()

    if doLargeDefectFigure:
        plt.savefig(f"gap_comparison_large_{defect_type}.png")
    else:
        plt.savefig(f"gap_comparison_{defect_type}.png")
    plt.close()


def defect_lattices_plot():
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    labels = ["("+"abcdefghijklmnopqrstuvwxyz"[i]+")" for i in range(len(axs.flatten()))]

    for i, (ax, defect_type) in enumerate(zip(axs.flatten(), ["vacancy", "schottky", "substitution", "interstitial"])):
        if defect_type in ["vacancy", "substitution"]:
            sl = 15
        else:
            sl = 14
        if defect_type == "schottky":
            Lattice = DefectSquareLattice(sl, defect_type, pbc=True, schottky_distance=1)
        else:
            Lattice = DefectSquareLattice(sl, defect_type, pbc=True).LargeDefectLattice
        ax = Lattice.plot_defect_idxs(ax=ax)


    plt.tight_layout()
    plt.savefig("defect_lattices.png")



if __name__ == "__main__":
    generate_figures("ldos", ["interstitial"], 14)



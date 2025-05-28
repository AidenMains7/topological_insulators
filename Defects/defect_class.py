import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import matplotlib.ticker as ticker
import scipy.linalg as spla

class DefectSquareLattice:
    def __init__(self, side_length:int, defect_type:str, pbc:bool = True, frenkel_pair_index:int = None, *args, **kwargs):
        self._pbc = pbc
        self._side_length = side_length
        self._defect_type = defect_type

        pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)
        self._pauli_matrices = [pauli1, pauli2, pauli3]

        self._pristine_lattice = self.generate_lattice() 
        match self.defect_type:
            case "none":
                self._lattice = self._pristine_lattice.copy()
                self._defect_index = None
            case "vacancy":
                self._lattice = self.generate_vacancy_lattice()
                self._defect_index = None
            case "substitution":
                self._lattice = self._pristine_lattice.copy()
                self._defect_index = self.lattice[self._side_length//2, self._side_length//2]
            case "interstitial":
                self._lattice = self.generate_interstitial_lattice()
                self._defect_index = np.max(self.lattice) // 2
            case "frenkel_pair":
                if frenkel_pair_index is None:
                    raise ValueError("frenkel_pair_index must be provided for 'frenkel_pair' defect type.")
                self._lattice, self._defect_index = self.compute_frenkel_pair_lattice(frenkel_pair_index)
            case _:
                raise ValueError(f"Unknown defect type: {defect_type}")

        self._Y, self._X = np.where(self._lattice >= 0)[:]
        if self._defect_type in ["interstitial", "frenkel_pair"]:
            self._X = self._X.astype(float) / 2
            self._Y = self._Y.astype(float) / 2
        self._dx, self._dy = self.compute_distances()

        self.Cx_plus_Cy, self.Sx, self.Sy, self.I = self.compute_wannier_polar()

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
    def defect_index(self):
        return self._defect_index
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

    def generate_lattice(self, *args, **kwargs):
        return np.arange(self.side_length**2).reshape((self.side_length, self.side_length))

    def generate_vacancy_lattice(self, *args, **kwargs):
        if self.side_length % 2 == 0:
            raise ValueError("Side length must be odd for a single vacancy in the center.")
        lattice = self._pristine_lattice.copy()
        lattice[self.side_length//2, self.side_length//2] = -1
        return lattice

    def generate_interstitial_lattice(self, *args, **kwargs):
        if self.side_length % 2 == 1:
            raise ValueError("Side length must be even for a single interstitial in the center.")
        Y, X = np.where(self._pristine_lattice >= 0)
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        coordinates = np.array([X, Y])
        coordinates = (np.concatenate((coordinates, np.array([[x_mean], [y_mean]])), axis=1) * 2).astype(int)
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
        return new_lattice, defect_index

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

        return dx, dy

    def compute_wannier_polar(self, *args, **kwargs):
        dx, dy = self.dx, self.dy

        theta = np.arctan2(dy, dx) % (2 * np.pi)
        distance_mask = np.maximum(np.abs(dx), np.abs(dy)) <= 1

        principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & (distance_mask)
        diagonal_mask  = ((np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0))) & (distance_mask)
        all_mask = principal_mask | diagonal_mask

        d_r = np.where(all_mask, np.sqrt(dx**2 + dy**2), 0.0 + 0.0j)
        F_p = np.where(principal_mask, np.exp(1  - d_r), 0. + 0.j)
        d_cos = np.where(all_mask, np.cos(theta), 0. + 0.j)
        d_sin = np.where(all_mask, np.sin(theta), 0. + 0.j)

        Cx_plus_Cy = F_p / 2
        Sx = 1j * d_cos * F_p / 2
        Sy = 1j * d_sin * F_p / 2

        return Cx_plus_Cy, Sx, Sy, np.eye(Sx.shape[0], dtype=complex)

    def compute_hamiltonian(self, M_background:float, M_substitution:float = None, t:float = 1.0, t0:float = 1.0, *args, **kwargs):
        if self.defect_type in ["vacancy", "none"]:
            onsite_mass = M_background * self.I
        elif self.defect_type in ["substitution", "interstitial", "frenkel_pair"]:
            if M_substitution is None:
                raise ValueError("M_substitution must be provided for 'substitution', 'interstitial', or 'frenkel pair' defects.")
            onsite_mass = M_background * self.I
            onsite_mass[self.defect_index, self.defect_index] = M_substitution

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

    def compute_LDOS(self, hamiltonian:np.ndarray, number_of_states:int = 2, plotSpectrum:bool = False, returnEigenvalues:bool = False, returnGap:bool = False, *args, **kwargs):
        eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
        number_of_states += number_of_states % 2
        mid_index = len(eigenvalues) // 2
        lower_states = eigenvalues[:mid_index][-number_of_states // 2:]
        upper_states = eigenvalues[mid_index:][:number_of_states // 2]
        selected_states = np.concatenate((lower_states, upper_states))
        selected_indices = np.concatenate((
            np.where(eigenvalues == lower_states[:, None])[1],
            np.where(eigenvalues == upper_states[:, None])[1]
        ))

        if plotSpectrum:
            fig, ax = plt.subplots()
            if False:
                ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label='Spectrum')
            else:
                ax.plot(eigenvalues, 'o-', label="Spectrum")
            ax.scatter(selected_indices, selected_states, color='red', label='Selected States', zorder=3)
            ax.set_title('Energy Spectrum with Selected States Highlighted')
            ax.set_xlabel('State Index')
            ax.set_ylabel('Energy')
            ax.legend()
            plt.show()

        LDOS = np.sum(np.abs(eigenvectors[:, selected_indices])**2, axis=1)
        LDOS = LDOS[0::2] + LDOS[1::2]
        LDOS = LDOS / np.sum(LDOS)
        gap = np.min(lower_states) - np.max(upper_states)


        if returnGap and returnEigenvalues:
            return LDOS, eigenvalues, gap
        if returnGap:
            return LDOS, gap
        if returnEigenvalues:
            return LDOS, eigenvalues
        return LDOS


def format_ldos_ax(ldos_ax, title):
    ldos_ax.spines['top'].set_visible(True)
    ldos_ax.spines['right'].set_visible(True)
    ldos_ax.spines['bottom'].set_visible(True)
    ldos_ax.spines['left'].set_visible(True)
    ldos_ax.set_xticks([])
    ldos_ax.set_yticks([])
    ldos_ax.set_aspect('equal')
    ldos_ax.set_title(title)
    return ldos_ax


def make_big_plot(plot_defect_type:str, side_length:int, clipHighestLowest:bool = False):


    # -------------------------------------
    if plot_defect_type == "substitution":
        Lattice = DefectSquareLattice(side_length, plot_defect_type)

        M_back_vals = [-2.5, -1.0, 1.0, 2.5]
        M_sub_vals = [-2.5, -1.0, 1.0, 2.5]
        n_rows, n_cols = 3, 2 * len(M_back_vals)
        scale = 4
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(scale*n_cols, scale*n_rows))  

        for i, M_background in enumerate(M_back_vals):
            spectrum_axs = axs[:, 0+2*i].flatten()
            ldos_axs = axs[:, 1+2*i].flatten()
            M_sub_vals_removed = M_sub_vals.copy()
            M_sub_vals_removed.remove(M_background)
            for spectrum_ax, ldos_ax, M_substitution in zip(spectrum_axs, ldos_axs, M_sub_vals_removed):
                hamiltonian = Lattice.compute_hamiltonian(M_background, M_substitution)
                LDOS, eigenvalues, gap = Lattice.compute_LDOS(hamiltonian, number_of_states=2, returnEigenvalues=True, returnGap=True)
                ldos_ax.scatter(Lattice.X, Lattice.Y, c=LDOS, s=50, cmap='inferno')
                ldos_ax = format_ldos_ax(ldos_ax, rf"$m_0^{{sub}}$ = {M_substitution}")

                spectrum_ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label="Gap = {:.2e}".format(gap))
                spectrum_ax.set_xlabel("Eigenvalue Index")
                spectrum_ax.set_ylabel("Energy")
                spectrum_ax.set_title(f"Spectrum")
                spectrum_ax.legend()
                #spectrum_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

                if ldos_ax.collections:  # Check if collections exist
                    norm = plt.Normalize(vmin=max(1e-10, min(coll.get_array().min() for coll in ldos_ax.collections)),
                                            vmax=max(coll.get_array().max() for coll in ldos_ax.collections))
                    for coll in ldos_ax.collections:
                        coll.set_norm(norm)
                    
                cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label("LDOS")
                cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

                fig.text((2*i+1)/8, 0.95, rf"$m_0^{{back}}$ = {M_background}", fontsize=20, ha='center')
          
        formatter = ticker.FormatStrFormatter('%.1e')
        for ax in axs.flatten():
            for im in ax.collections:
                if hasattr(im, 'colorbar') and im.colorbar is not None:
                    im.colorbar.ax.yaxis.set_major_formatter(formatter)


        fig.suptitle(rf"Substitution", fontsize=30)
        direc="./Defects/Plots/LDOS/"
        plt.tight_layout()
        plt.savefig(direc+f"LDOS_{Lattice.defect_type}_{side_length}_all.png")

    # -------------------------------------
    elif plot_defect_type == "interstitial":
        Lattice = DefectSquareLattice(side_length, plot_defect_type)

        M_back_vals = [-2.5, -1.0, 1.0, 2.5]
        M_sub_vals = [-2.5, -1.0, 1.0, 2.5]
        n_rows, n_cols = len(M_sub_vals), 2 * len(M_back_vals)
        scale = 4
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(scale*n_cols, scale*n_rows))   

        for i, M_background in enumerate(M_back_vals):  
            spectrum_axs = axs[:, 0+2*i].flatten()
            ldos_axs = axs[:, 1+2*i].flatten()
            for spectrum_ax, ldos_ax, M_substitution in zip(spectrum_axs, ldos_axs, M_sub_vals):
                hamiltonian = Lattice.compute_hamiltonian(M_background, M_substitution)
                LDOS, eigenvalues, gap = Lattice.compute_LDOS(hamiltonian, number_of_states=2, returnEigenvalues=True, returnGap=True)
                if clipHighestLowest:
                    LDOS = np.clip(LDOS, np.sort(LDOS)[1], np.sort(LDOS)[-2])
                ldos_ax.scatter(Lattice.X, Lattice.Y, c=LDOS, s=50, cmap='inferno')
                ldos_ax = format_ldos_ax(ldos_ax, rf"$m_0^{{sub}}$ = {M_substitution}")
    
                spectrum_ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label="Gap = {:.2e}".format(gap))
                spectrum_ax.set_xlabel("Eigenvalue Index")
                spectrum_ax.set_ylabel("Energy")
                spectrum_ax.set_title(f"Spectrum")
                spectrum_ax.legend()
                #spectrum_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

                if ldos_ax.collections:  # Check if collections exist
                    norm = plt.Normalize(vmin=max(1e-10, min(coll.get_array().min() for coll in ldos_ax.collections)),
                                            vmax=max(coll.get_array().max() for coll in ldos_ax.collections))
                    for coll in ldos_ax.collections:
                        coll.set_norm(norm)
                    
                cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label("LDOS")
                cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))
            fig.text((2*i+1)/8, 0.95, rf"$m_0^{{back}}$ = {M_background}", fontsize=20, ha='center')


        formatter = ticker.FormatStrFormatter('%.1e')
        for ax in axs.flatten():
            for im in ax.collections:
                if hasattr(im, 'colorbar') and im.colorbar is not None:
                    im.colorbar.ax.yaxis.set_major_formatter(formatter)

        if not clipHighestLowest:
            fig.suptitle("Interstitial", fontsize=30)
        else:
            fig.suptitle("Interstitial (clipped)", fontsize=30)

        plt.tight_layout()

        direc="./Defects/Plots/LDOS/"
        if not clipHighestLowest:
            plt.savefig(direc+f"LDOS_{Lattice.defect_type}_{side_length}_all.png")  
        else:
            plt.savefig(direc+f"LDOS_clip_{Lattice.defect_type}_{side_length}_all.png")

    # -------------------------------------
    elif plot_defect_type in ["none", "vacancy"]:
        Lattice = DefectSquareLattice(side_length, plot_defect_type)

        M_back_vals = [-2.5, -1.0, 1.0, 2.5]
        fig, axs = plt.subplots(4, 2, figsize=(8, 12))
        spectrum_axs = axs[:, 0].flatten()
        ldos_axs = axs[:, 1].flatten()
        for spectrum_ax, ldos_ax, M_background in zip(spectrum_axs, ldos_axs, M_back_vals):
            hamiltonian = Lattice.compute_hamiltonian(M_background, None)
            LDOS, eigenvalues, gap = Lattice.compute_LDOS(hamiltonian, number_of_states=2, returnEigenvalues=True, returnGap=True)
            ldos_ax.scatter(Lattice.X, Lattice.Y, c=LDOS, s=10x``, cmap='inferno')
            ldos_ax.spines['top'].set_visible(True)
            ldos_ax.spines['right'].set_visible(True)
            ldos_ax.spines['bottom'].set_visible(True)
            ldos_ax.spines['left'].set_visible(True)
            ldos_ax.set_xticks([])
            ldos_ax.set_yticks([])
            ldos_ax.set_aspect('equal')
            ldos_ax.set_title(rf"$m_0^{{back}}$ = {M_background}")

            spectrum_ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label="Gap = {:.2e}".format(gap))
            spectrum_ax.set_xlabel("Eigenvalue Index")
            spectrum_ax.set_ylabel("Energy")
            spectrum_ax.set_title(f"Spectrum")
            spectrum_ax.legend()
            #spectrum_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

            if ldos_ax.collections:  # Check if collections exist
                norm = plt.Normalize(vmin=max(1e-10, min(coll.get_array().min() for coll in ldos_ax.collections)),
                                        vmax=max(coll.get_array().max() for coll in ldos_ax.collections))
                for coll in ldos_ax.collections:
                    coll.set_norm(norm)
                
            cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label("LDOS")
            cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

        formatter = ticker.FormatStrFormatter('%.1e')
        for ax in axs.flatten():
            for im in ax.collections:
                if hasattr(im, 'colorbar') and im.colorbar is not None:
                    im.colorbar.ax.yaxis.set_major_formatter(formatter)

        fig.suptitle(f"{Lattice.defect_type.capitalize()}", fontsize=20)
        direc="./Defects/Plots/LDOS/"
        plt.tight_layout()
        plt.show()
        #plt.savefig(direc+f"LDOS_{Lattice.defect_type}.png")

    # -------------------------------------
    elif plot_defect_type == "frenkel_pair":
        M_back_vals = [-2.5, -1.0, 1.0, 2.5]
        M_sub_vals = [-2.5, -1.0, 1.0, 2.5]
        n_rows, n_cols = 4, 8
        scale = 4
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(scale*n_cols, scale*n_rows))   
        for i, M_background in enumerate(M_back_vals):  
            spectrum_axs = axs[:, 0+2*i].flatten()
            ldos_axs = axs[:, 1+2*i].flatten()
            for spectrum_ax, ldos_ax, M_substitution in zip(spectrum_axs, ldos_axs, M_sub_vals):
                all_LDOS = []
                all_x = []
                all_y = []
                all_eigenvalues = []
                all_gap = []
                for frenkel_pair_index in range(8):
                    Lattice = DefectSquareLattice(side_length, plot_defect_type, frenkel_pair_index=frenkel_pair_index)
                    hamiltonian = Lattice.compute_hamiltonian(M_background, M_substitution)
                    LDOS, eigenvalues, gap = Lattice.compute_LDOS(hamiltonian, number_of_states=2, returnEigenvalues=True, returnGap=True)
                    all_LDOS.append(LDOS)
                    all_x.append(Lattice.X)
                    all_y.append(Lattice.Y)
                    all_eigenvalues.append(eigenvalues)
                    all_gap.append(gap)
                
                all_x = np.concatenate(all_x)
                all_y = np.concatenate(all_y)
                all_LDOS = np.concatenate(all_LDOS)
                # Find unique (X, Y) pairs and sum LDOS for each unique site
                coords = np.stack((all_x, all_y), axis=1)
                unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
                summed_LDOS = np.zeros(len(unique_coords))
                np.add.at(summed_LDOS, inverse_indices, all_LDOS)
                summed_LDOS /= 8.0

                if clipHighestLowest:
                    summed_LDOS = np.clip(summed_LDOS, np.sort(summed_LDOS)[1], np.sort(summed_LDOS)[-2])
                ldos_ax.scatter(unique_coords[:, 0], unique_coords[:, 1], c=summed_LDOS, s=50, cmap='inferno')
                ldos_ax = format_ldos_ax(ldos_ax, rf"$m_0^{{sub}}$ = {M_substitution}")
    
                spectrum_ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label="Gap = {:.2e}".format(gap))
                spectrum_ax.set_xlabel("Eigenvalue Index")
                spectrum_ax.set_ylabel("Energy")
                spectrum_ax.set_title(f"Spectrum")
                #spectrum_ax.legend()
                #spectrum_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

                if ldos_ax.collections:  # Check if collections exist
                    norm = plt.Normalize(vmin=max(1e-10, min(coll.get_array().min() for coll in ldos_ax.collections)),
                                            vmax=max(coll.get_array().max() for coll in ldos_ax.collections))
                    for coll in ldos_ax.collections:
                        coll.set_norm(norm)
                    
                cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label("LDOS")
                cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

            fig.text((2*i+1)/8, 0.95, rf"$m_0^{{back}}$ = {M_background}", fontsize=20, ha='center')



        formatter = ticker.FormatStrFormatter('%.1e')
        for ax in axs.flatten():
            for im in ax.collections:
                if hasattr(im, 'colorbar') and im.colorbar is not None:
                    im.colorbar.ax.yaxis.set_major_formatter(formatter)

        if not clipHighestLowest:
            fig.suptitle("Frenkel Pair", fontsize=30)
        else:
            fig.suptitle("Frenkel Pair (clipped)", fontsize=30)
        plt.tight_layout()
        
        direc="./Defects/Plots/LDOS/"
        if not clipHighestLowest:
            plt.savefig(direc+f"LDOS_{Lattice.defect_type}_{side_length}_all.png")  
            pass
        else:
            plt.savefig(direc+f"LDOS_clip_{Lattice.defect_type}_{side_length}_all.png")
            pass


def probe_point():
    Lattice = DefectSquareLattice(14, "interstitial")
    dx, dy = Lattice.dx, Lattice.dy

    theta = np.arctan2(dy, dx) % (2 * np.pi)
    distance_mask = np.maximum(np.abs(dx), np.abs(dy)) <= 1

    principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & (distance_mask)
    diagonal_mask  = ((np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0))) & (distance_mask)
    all_mask = principal_mask | diagonal_mask

    d_r = np.where(all_mask, np.sqrt(dx**2 + dy**2), 0.0 + 0.0j)
    F_p = np.where(principal_mask, np.exp(1  - d_r), 0. + 0.j)
    d_cos = np.where(all_mask, np.cos(theta), 0. + 0.j)
    d_sin = np.where(all_mask, np.sin(theta), 0. + 0.j)

    X, Y = Lattice.X, Lattice.Y

    plt.scatter(X, Y, c=diagonal_mask[np.max(Lattice.lattice)//2+7].real, s=50, cmap='inferno')
    plt.colorbar()
    plt.show()



if __name__ == "__main__":
    make_big_plot("vacancy", 51)



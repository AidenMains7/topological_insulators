import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as spla

class Lattice:
    def __init__(self, *args, **kwargs):
        pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)
        self._pauli_matrices = [pauli1, pauli2, pauli3]
        
        self._lattice = self.generate_lattice(*args, **kwargs)
        self._Y, self._X = np.where(self.lattice >= 0)
        self._X = self._X - np.mean(self._X)
        self._Y = self._Y - np.mean(self._Y)
        self._R, self._Theta = self.convert_cartesian_to_polar()

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
    def R(self):
        return self._R
    @property
    def Theta(self):
        return self._Theta
    @property
    def dx(self):
        return self._dx
    @property
    def dy(self):
        return self._dy
    @property
    def dr(self):
        return self._dr
    @property
    def pauli_matrices(self):
        return self._pauli_matrices

    def generate_lattice(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def convert_cartesian_to_polar(self):
        r = np.sqrt(self.X**2 + self.Y**2)
        theta = np.arctan2(self.Y, self.X) % (2 * np.pi)
        return r, theta
    
    def plot_polar(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(self.Theta, self.R, s=50, alpha=0.5)
        ax.set_title('Polar Plot of Lattice Points')
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
        cbar.set_label('Theta (radians)')
        
        plt.show()

    def plot_cartesian(self):
        fig, ax = plt.subplots()
        ax.scatter(self.X, self.Y, s=50, alpha=0.5)
        ax.set_title('Cartesian Plot of Lattice Points')
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
        cbar.set_label('X and Y Coordinates')

        for x, y, label in zip(self.X, self.Y, self.lattice.flatten()):
            ax.text(x, y, str(label), fontsize=8, ha='center', va='center')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')
        plt.show()

    def plot_distances(self, index):
        fig, ax = plt.subplots()

        smallest_values = np.sort(self.dr[index])[:8]
        print(smallest_values)
        ax.scatter(self.X, self.Y, c=self.dr[index], cmap='viridis', s=50)
        ax.set_title('Distance from Point {}'.format(index))
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
        cbar.set_label('Distance (dr)')
        ax.scatter(self.X[index], self.Y[index], color='red', s=100, label='Selected Point')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')
        plt.show()

    def compute_distances(self):
        dx = self.X[:, np.newaxis] - self.X[np.newaxis, :]
        dy = self.Y[:, np.newaxis] - self.Y[np.newaxis, :]
        return dx, dy, np.sqrt(dx**2 + dy**2)
    
    def compute_hopping(self):
        dx, dy, dr = self.dx, self.dy, self.dr
        phi = np.arctan2(dy, dx) % (2 * np.pi)
        distance_mask = np.maximum(np.abs(dx), np.abs(dy)) <= 1

        restricted_dr = np.where(distance_mask, dr, 0.0 + 0.0j)
        d_cos = np.where(distance_mask, np.cos(phi), 0.0 + 0.0j)
        d_sin = np.where(distance_mask, np.sin(phi), 0.0 + 0.0j)
        F_p = np.where(distance_mask, np.exp(1 - restricted_dr), 0.0 + 0.0j)

        Cx_plus_Cy = F_p / 2
        Sx = 1j * d_cos * F_p / 2
        Sy = 1j * d_sin * F_p / 2

        return Cx_plus_Cy, Sx, Sy, np.eye(Sx.shape[0], dtype=complex)

    def compute_hamiltonian(self, t:float, t0:float, M:float):
        Cx_plus_Cy, Sx, Sy, I = self.compute_hopping()
        d1 = t * Sx
        d2 = t * Sy
        d3 = M * I + t0 * Cx_plus_Cy
        hamiltonian = np.kron(d1, self.pauli_matrices[0]) + \
                      np.kron(d2, self.pauli_matrices[1]) + \
                      np.kron(d3, self.pauli_matrices[2])
        return hamiltonian

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

    def plot_spectrum_ldos(self, M:float):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        hamiltonian = self.compute_hamiltonian(t=1.0, t0=1.0, M=M)
        ldos, eigenvalues, gap = self.compute_LDOS(hamiltonian, number_of_states=2, plotSpectrum=False, returnEigenvalues=True, returnGap=True)
        axs[0].scatter(np.arange(len(eigenvalues)), eigenvalues, label=f"Gap = {gap:.2f}")
        axs[0].legend()
        axs[0].set_title('Spectrum of the Hamiltonian')
        axs[0].set_xlabel('Index')
        axs[0].set_ylabel('Energy')
        axs[1].scatter(self.X, self.Y, c=ldos, cmap='viridis', s=50)
        axs[1].set_title('Local Density of States (LDOS)')
        cbar = plt.colorbar(axs[1].collections[0], ax=axs[1], orientation='vertical')
        cbar.set_label('LDOS')
        axs[1].set_xlabel('X Coordinate')
        axs[1].set_ylabel('Y Coordinate')
        axs[1].axis('equal')
        plt.tight_layout()
        plt.show()




class SquareDisclination(Lattice):
    def __init__(self, type_of_disclination:str, initial_angle:float, final_angle:float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type_of_disclination == "add":
            self._R, self._Theta, self._X, self._Y = self.add_lattice_section(initial_angle, final_angle)
        elif type_of_disclination == "remove":
            self._R, self._Theta, self._X, self._Y = self.remove_lattice_section(initial_angle, final_angle)

        self._R, self._Theta, self._X, self._Y = self.remove_repeated_sites()
        self._R, self._Theta, self._X, self._Y = self.sort_lattice_sites()
        self._dx, self._dy, self._dr = self.compute_distances()
    
    def generate_lattice(self, side_length:int):
        return np.arange(side_length**2).reshape((side_length, side_length))
    
    def remove_lattice_section(self, init_angle:float, final_angle:float):
        mask = (self.Theta > init_angle) & (self.Theta < final_angle)
        R = self.R[~mask]
        Theta = self.Theta[~mask]

        Theta = (Theta - final_angle) % (2 * np.pi)
        Theta *= 1/np.max(Theta) * 2 * np.pi

        X, Y = R * np.cos(Theta), R * np.sin(Theta)

        return R, Theta, X, Y
    
    def add_lattice_section(self, init_angle:float, final_angle:float):
        mask = (self.Theta >= init_angle) & (self.Theta <= final_angle)

        percent_of_total = (final_angle - init_angle) / (2 * np.pi)
        repeat_factor = 1 + 1 / percent_of_total
        R = self.R[mask]
        Theta = self.Theta[mask]

        Theta = (Theta - np.min(Theta)) / np.max(Theta) * (2 * np.pi / (repeat_factor))
        Theta = np.concatenate([Theta + np.max(Theta) * i for i in range(int(repeat_factor))])
        R = np.concatenate([R for _ in range(int(repeat_factor))])

        Theta = Theta % (2 * np.pi)

        X, Y = R * np.cos(Theta), R * np.sin(Theta)
        return R, Theta, X, Y
    
    def remove_repeated_sites(self):
        stack_coords = np.column_stack((self.X, self.Y))
        unique_indices = np.unique(np.round(stack_coords, 5), axis=0, return_index=True)[1]


        unique_X = self.X[unique_indices]
        unique_Y = self.Y[unique_indices]
        unique_R = self.R[unique_indices]
        unique_Theta = self.Theta[unique_indices]
        return unique_R, unique_Theta, unique_X, unique_Y
    
    def sort_lattice_sites(self):
        
        sorted_indices = np.lexsort((self.Y, self.X))
        sorted_indices = np.argsort(self._R)
        return self.R[sorted_indices], self.Theta[sorted_indices], self.X[sorted_indices], self.Y[sorted_indices]



if __name__ == "__main__":
    side_length = 17
    Disclination = SquareDisclination(side_length=side_length, type_of_disclination="remove", initial_angle = 0., final_angle = np.pi/2)
    Disclination.plot_spectrum_ldos(M=0.0)


    if False:
        M_vals = np.linspace(-10., 10., 11)
        bott_vals = []
        for M in M_vals:
            hamiltonian = Disclination.compute_hamiltonian(t=1.0, t0=1.0, M=M)
            projector = Disclination.compute_projector(hamiltonian=hamiltonian)
            bott_index = Disclination.compute_bott_index(projector=projector)
            bott_vals.append(bott_index)

        plt.scatter(M_vals, bott_vals)
        plt.xlabel('M')
        plt.ylabel('Bott Index')
        plt.show()
        
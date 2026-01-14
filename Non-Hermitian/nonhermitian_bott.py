import numpy as np
import scipy.linalg as spla
from itertools import product
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm_joblib import tqdm, tqdm_joblib
from matplotlib.colors import ListedColormap, BoundaryNorm
import h5py,os


def compute_lattice(Lx, Ly):
    return np.arange(Lx*Ly).reshape(Ly, Lx)

def compute_distances(lattice, pbc):
    """
    Compute the distances between all pairs of sites in the lattice.
    """
    # Displacement matrices. dx[i, j] is the x-displacement between site i and site j.
    Y, X = np.where(lattice >= 0)
    Ly, Lx = lattice.shape
    dx = X - X[:, None]
    dy = Y - Y[:, None]

    if pbc:
        # Apply periodic boundary conditions
        multipliers = tuple(product([-1, 0, 1], repeat=2))
        shifts = [(i * Lx, j * Ly) for i, j in multipliers]

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

def compute_wannier(dx, dy):
    hop_xp = ((dx == 1) & (dy == 0))
    hop_yp = ((dx == 0) & (dy == 1))
    NN_pos = hop_xp | hop_yp
    Cx_plus_Cy = NN_pos * 1/2
    Sx = hop_xp * 1j/2
    Sy = hop_yp * 1j/2
    Cx_plus_Cy += Cx_plus_Cy.conj().T
    Sx += Sx.conj().T
    Sy += Sy.conj().T
    return {"Cx_plus_Cy":Cx_plus_Cy, "Sx":Sx, "Sy":Sy, "I":np.eye(dx.shape[0])}

def compute_wannier_polar(dx, dy):
    """Compute the Wannier polar matrices based on the displacements. While the construction is only necessary for defects containing an interstitial defect (interstitial, frenkek_pair), 
    it is computed for all defect types for consistency. Typical behavior is, of course, recovered in the case of no interstitial defect."""
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
    return {"Cx_plus_Cy":Cx_plus_Cy, "Sx":Sx, "Sy":Sy, "I":np.eye(dx.shape[0])}


def compute_hamiltonian(wannier_matrices:dict, m0:float, h_vector:np.ndarray, t:float = 1.0, t0:float = 1.0):
    # Pauli matrices for Hamiltonian computation
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)     
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    Cx_plus_Cy, Sx, Sy, I = wannier_matrices["Cx_plus_Cy"], wannier_matrices["Sx"], wannier_matrices["Sy"], wannier_matrices["I"]
    
    # Hopping vector terms
    d1 = t * Sx
    d2 = t * Sy
    d3 = m0 * I + t0 * Cx_plus_Cy

    hamiltonian = np.kron(d1, pauli_x) + np.kron(d2, pauli_y) + np.kron(d3, pauli_z)

    # Non hermitian coupling
    hx, hy, hz = h_vector
    H_NH_x = np.kron(1j * hx * I, pauli_x)
    H_NH_y = np.kron(1j * hy * I, pauli_y)
    H_NH_z = np.kron(1j * hz * I, pauli_z)
    H_NH = H_NH_x + H_NH_y + H_NH_z 

    return hamiltonian + H_NH

def compute_projector(hamiltonian:np.ndarray):
    """Compute the projector onto the lower band of the Hamiltonian."""
    eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
    lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2] # Lower band eigenvalues
    highest_lower_band = lower_band[-1] # Highest eigenvalue in the lower band

    D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j) # Projector diagonal matrix
    D_herm_conj = np.einsum('i,ij->ij', D, eigenvectors.conj().T)
    projector = eigenvectors @ D_herm_conj # Projector matrix
    return projector

def compute_bott_index(lattice:np.ndarray, projector:np.ndarray):
    """Compute the Bott index for the given projector."""
    Y, X = np.where(lattice >= 0)
    # Repeated (two orbitals)
    X = np.repeat(X, 2)
    Y = np.repeat(Y, 2)
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
    bott_index = np.imag(np.sum(np.log(spla.eigvals(A)))) / (2 * np.pi)
    return bott_index


def plot_phase_diagram(fig, ax, 
                       X_values, Y_values, Z_values, 
                       labels:list=None, title:str=None, 
                       X_ticks=None, Y_ticks=None, X_tick_labels=None, Y_tick_labels=None,
                       cmap='Spectral', plotColorbar=True, doDiscreteColormap=True):
    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]
    Z_values = np.where(Z_values == -0, 0, Z_values)

    not_nan_mask = ~np.isnan(Z_values)
    unique_values = np.sort(np.unique(Z_values[not_nan_mask]).astype(int))
    #unique_values = np.arange(-3, 4, 1)

    if doDiscreteColormap:
        if len(unique_values) < 25:
            cmap = plt.get_cmap(cmap)
            discrete_colors = cmap(np.linspace(0, 1, len(unique_values)))
            cmap = ListedColormap(discrete_colors)
            norm = BoundaryNorm(boundaries=np.append(unique_values, unique_values[-1] + 1), ncolors=len(unique_values))

    im = ax.imshow(Z_values, extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]], 
                   origin='lower', aspect='auto', cmap=cmap, interpolation='none', 
                   rasterized=True, norm=norm)
    
    if title is not None:
        ax.set_title(title)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1], rotation=0)

    if X_ticks is not None:
        ax.set_xticks(X_ticks)
    if Y_ticks is not None:
        ax.set_yticks(Y_ticks)
    if X_tick_labels is not None:
        ax.set_xticklabels(X_tick_labels)
    if Y_tick_labels is not None:
        ax.set_yticklabels(Y_tick_labels)

    if plotColorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_ticks(unique_values+0.5)
        cbar.set_ticklabels([str(val) for val in unique_values], fontsize=16)

    return fig, ax


if __name__ == "__main__":

    resolution = (51, 51)
    m0_values = np.linspace(-2.0, 2.0, resolution[0])
    h_values = np.linspace(-1.0, 1.0, resolution[1])
    h_type = 'x'
    filename = f"NH_bott_{h_type}_{resolution[0]}x{resolution[1]}.h5"

    if not os.path.exists(filename):
        parameters = tuple(product(m0_values, h_values))
        lattice = compute_lattice(15, 15)
        dx, dy = compute_distances(lattice, False)
        wannier_matrices = compute_wannier(dx, dy)
        def worker(i):
            m0, h = parameters[i]

            match h_type:
                case 'x':
                    h_vector = [h, 0., 0.]
                case 'y':
                    h_vector = [0., h, 0.]
                case 'z':
                    h_vector = [0., 0., h]
            hamiltonian = compute_hamiltonian(wannier_matrices, m0, h_vector)
            projector = compute_projector(hamiltonian)
            bott_index = compute_bott_index(lattice, projector)
            return [m0, h, bott_index]

        with tqdm_joblib(tqdm(total=len(parameters), desc="Computing m0 vs h_i phase diagram for the Bott Index")) as progress_bar:
            data = np.array(Parallel(n_jobs=-1)(delayed(worker)(i) for i in range(len(parameters)))).T
        m0 = data[0]
        h = data[1]
        bott_index = np.round(data[2].reshape(resolution).T, 3)
        with h5py.File(filename, 'w') as f:
            f.create_dataset(name = 'm0', data = m0)
            f.create_dataset(name = 'h', data = h)
            f.create_dataset(name = 'bott_index', data = bott_index)

    with h5py.File(filename, 'r') as f:
        m0 = f['m0'][:]
        h = f['h'][:]
        bott_index = f['bott_index'][:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_phase_diagram(fig, ax, m0, h, bott_index)

    x_line = np.linspace(-1.0, 1.0)
    ax.plot(x_line, x_line, ls='--', c='k')
    ax.plot(x_line, -x_line, ls='--', c='k')
    x_line2 = np.linspace(1.0, 2.0)
    ax.plot(x_line2, x_line2 - 2, ls='--', c='k')
    ax.plot(x_line2, 2 - x_line2, ls='--', c='k')
    x_line3 = np.linspace(-2.0, -1.0)
    ax.plot(x_line3, x_line3 + 2, ls='--', c='k')
    ax.plot(x_line3, - 2 - x_line3, ls='--', c='k')
    plt.show()
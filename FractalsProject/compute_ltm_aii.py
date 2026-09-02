import numpy as np
import scipy.linalg as spla
from scipy.sparse import coo_array, csr_array
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

# Pauli matrices
s0 = np.eye(2, dtype=np.complex128)
sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

t0, tx, ty, tz = s0, sx, sy, sz

# Gamma matrices defined in Eq. (37): Spin (sigma) x Orbital (tau)
G1 = np.kron(sx, tx)
G2 = np.kron(sy, tx)
G3 = np.kron(sz, tx)
G4 = np.kron(s0, ty)
G5 = np.kron(s0, tz)


def get_building_blocks(M, M1, M2, A0, B0, mu=0.0):
    """Computes the 4x4 onsite matrix and directional hopping matrices Tx, Ty, Tz."""
    M_tilde = M + 2 * M1 + 4 * M2

    H_onsite = M_tilde * G5 - mu * np.kron(s0, t0)
    Tx = -M2 * G5 + 0.5j * A0 * G2
    Ty = -M2 * G5 - 0.5j * A0 * G1
    Tz = -M1 * G5 - 0.5j * B0 * G4

    return H_onsite, Tx, Ty, Tz


def compute_k_space_hamiltonian(kx, ky, kz, M, M1, M2, A0, B0, mu=0.0):
    """Computes the 4x4 Bloch Hamiltonian H(k) at momentum (kx, ky, kz)."""
    H_onsite, Tx, Ty, Tz = get_building_blocks(M, M1, M2, A0, B0, mu)

    H_k = (
        H_onsite
        + Tx * np.exp(1j * kx)
        + Tx.conj().T * np.exp(-1j * kx)
        + Ty * np.exp(1j * ky)
        + Ty.conj().T * np.exp(-1j * ky)
        + Tz * np.exp(1j * kz)
        + Tz.conj().T * np.exp(-1j * kz)
    )
    return H_k


def build_lattice_hamiltonian(Lx, Ly, Lz, M, M1=1.0, M2=1.0, A0=2.0, B0=2.0, mu=0.0, pbc=False):
    """Builds the 3D real-space sparse lattice Hamiltonian.

    Parameters:
        Lx, Ly, Lz : int - Lattice dimensions.
        M, M1, M2  : float - Mass parameters.
        A0, B0     : float - Spin-orbit / orbital mixing strengths.
        mu         : float - Chemical potential.
        pbc        : bool - Periodic Boundary Conditions (True/False).

    Returns:
        H_sparse : scipy.sparse.csr_array of shape (4*N, 4*N)
    """
    H_onsite, Tx, Ty, Tz = get_building_blocks(M, M1, M2, A0, B0, mu)
    N_sites = Lx * Ly * Lz

    def site_idx(x, y, z):
        return x + Lx * (y + Ly * z)

    row_ind, col_ind, data = [], [], []

    def add_block(r_site, c_site, matrix):
        for r in range(4):
            for c in range(4):
                val = matrix[r, c]
                if val != 0:
                    row_ind.append(4 * r_site + r)
                    col_ind.append(4 * c_site + c)
                    data.append(val)

    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                i = site_idx(x, y, z)

                # Onsite
                add_block(i, i, H_onsite)

                # +x Hopping
                if x + 1 < Lx or pbc:
                    j = site_idx((x + 1) % Lx, y, z)
                    add_block(i, j, Tx)
                    add_block(j, i, Tx.conj().T)

                # +y Hopping
                if y + 1 < Ly or pbc:
                    j = site_idx(x, (y + 1) % Ly, z)
                    add_block(i, j, Ty)
                    add_block(j, i, Ty.conj().T)

                # +z Hopping
                if z + 1 < Lz or pbc:
                    j = site_idx(x, y, (z + 1) % Lz)
                    add_block(i, j, Tz)
                    add_block(j, i, Tz.conj().T)

    return coo_array(
        (data, (row_ind, col_ind)),
        shape=(4 * N_sites, 4 * N_sites),
        dtype=np.complex128,
    ).tocsr()


def compute_topological_marker(l, eigenvalues, eigenvectors, fermi_energy:float=0.0):
    filled_idxs = np.argwhere(eigenvalues < fermi_energy).flatten()
    empty_idxs = np.argwhere(eigenvalues > fermi_energy).flatten()
    if np.sum(eigenvalues == fermi_energy) > 0:
        raise ValueError("Fermi energy coincides with an eigenvalue.")

    # Vectorized computation of projection operators P and Q
    V_filled = eigenvectors[:, filled_idxs]
    P = V_filled @ V_filled.conj().T

    V_empty = eigenvectors[:, empty_idxs]
    Q = V_empty @ V_empty.conj().T

    X, Y, Z = np.where(l > -100)
    n_dof_per_site = eigenvectors.shape[0] // len(X)
    X = np.diag(np.repeat(X, n_dof_per_site))
    Y = np.diag(np.repeat(Y, n_dof_per_site))
    Z = np.diag(np.repeat(Z, n_dof_per_site))

    N_D = -8 * np.pi * 1.0j

    sigma1 = np.array([[0.0, 1.0], [1.0, 0.0]]).astype(np.complex128)
    sigma3 = np.array([[1.0, 0.0], [0.0, -1.0]]).astype(np.complex128)
    G3 = np.kron(sigma3, sigma1)
    W = np.kron(np.eye(eigenvalues.shape[0] // 4), G3)

    A = Q @ X @ P @ Y @ Q @ Z @ P
    B = P @ X @ Q @ Y @ P @ Z @ Q
    C = N_D * W @ (A + B)
    return C


def plot_3d_voxels(voxels, colors, cmap='viridis', edgecolors='k', alpha=0.8):
    """
    Plots a 3D voxel grid where voxels and colors share the same 3D spatial shape.

    Parameters:
        voxels (np.ndarray): 3D array (X, Y, Z). Non-zero or True values indicate filled voxels.
        colors (np.ndarray): Array with shape matching `voxels` (X, Y, Z). Contains color strings,
                             RGBA values, or scalar numerical values to map via `cmap`.
        cmap (str or Colormap): Matplotlib colormap used if `colors` contains numerical data.
        edgecolors (str): Line color for voxel edges.
        alpha (float): Opacity of the voxel faces.

    Returns:
        fig, ax: Matplotlib Figure and Axes3D objects.
    """
    filled = np.asarray(voxels, dtype=bool)
    colors = np.asarray(colors)

    if filled.shape != colors.shape[:3]:
        raise ValueError(f"Shape mismatch: voxels {filled.shape} vs colors {colors.shape[:3]}")

    # Convert scalar numeric color arrays to RGBA via the colormap
    if np.issubdtype(colors.dtype, np.number) and colors.ndim == 3:
        # Normalize strictly over the filled voxel regions
        vmin = np.nanmin(colors[filled]) if np.any(filled) else 0
        vmax = np.nanmax(colors[filled]) if np.any(filled) else 1
        norm = Normalize(vmin=vmin, vmax=vmax)
        color_mapper = plt.get_cmap(cmap)
        facecolors = color_mapper(norm(colors))
    else:
        facecolors = colors

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels
    ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors, alpha=alpha)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return fig, ax







Lx = Ly = 11
Lz = 11
l = np.full((Lx, Ly, Lz), 1, dtype=int)
H = build_lattice_hamiltonian(Lx, Ly, Lz, -2.0, pbc=False).toarray()
eigenvalues, eigenvectors = spla.eigh(H)

C = compute_topological_marker(l, eigenvalues, eigenvectors)
C = np.diag(C).reshape(-1, 4).sum(axis=1).real


X, Y, Z = np.where(l == 1)
x_center = np.mean(X)
y_center = np.mean(Y)
z_center = np.mean(Z)

r = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2 + (Z - z_center) ** 2)

plt.scatter(r, C,  alpha=0.5)
plt.show()


idxs = [eigenvalues.size // 2, eigenvalues.size // 2 - 1]
ldos = np.sum(np.abs(eigenvectors[:, idxs]) ** 2, axis=1)
ldos = ldos.reshape(-1, 4).sum(axis=1).reshape(l.shape)
fig, vax = plot_3d_voxels(l == 1, ldos)
plt.show()
import numpy as np
import scipy.linalg as spla
from scipy.sparse import csr_array, kron as sp_kron, eye as sp_eye
import os, h5py
from tqdm_joblib import tqdm_joblib, tqdm
from joblib import Parallel, delayed

from project_tools import lattice, model
from hypercubic import solve

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def compute_topological_marker(l, method, eigenvalues, eigenvectors, fermi_energy:float=0.0, symmetrize=True):
    filled_idxs = np.argwhere(eigenvalues < fermi_energy).flatten()
    empty_idxs = np.argwhere(eigenvalues > fermi_energy).flatten()
    if np.sum(eigenvalues == fermi_energy) > 0:
        raise ValueError("Fermi energy coincides with an eigenvalue.")

    # Vectorized computation of projection operators P and Q
    V_filled = eigenvectors[:, filled_idxs]
    P = V_filled @ V_filled.conj().T

    V_empty = eigenvectors[:, empty_idxs]
    Q = V_empty @ V_empty.conj().T

    if method in ['site_elim', 'renorm']: 
        X, Y, Z = np.where(l > 0)
    else:
        X, Y, Z = np.where(l >= 0)
    n_dof_per_site = eigenvectors.shape[0] // len(X)
    X_op = np.diag(np.repeat(X, n_dof_per_site))
    Y_op = np.diag(np.repeat(Y, n_dof_per_site))
    Z_op = np.diag(np.repeat(Z, n_dof_per_site))
    pos_ops = {"x": X_op, "y": Y_op, "z": Z_op}

    N_D = 8 * np.pi * 1.0j

    sigma0 = np.array([[1., 0.], [0., 1.]]).astype(np.complex128)
    sigma2 = np.array([[0., -1.j], [1.j, 0.]]).astype(np.complex128)

    G5 = -np.kron(sigma2, sigma0)

    W = np.kron(np.eye(eigenvalues.shape[0] // 4), G5)


    def eval_term(x1, x2, x3):
        A = Q @ x1 @ P @ x2 @ Q @ x3 @ P
        B = P @ x1 @ Q @ x2 @ P @ x3 @ Q
        return A + B

    if symmetrize:
        permutations = [
            (("x", "y", "z"), +1.0),
            (("y", "z", "x"), +1.0),
            (("z", "x", "y"), +1.0),
            (("y", "x", "z"), -1.0),
            (("x", "z", "y"), -1.0),
            (("z", "y", "x"), -1.0),
        ]
    else:
        permutations = [
            (("x", "y", "z"), +1.0)
        ]
    term_sum = np.zeros(eigenvectors.shape, dtype=np.complex128)
    for (p1, p2, p3), sgn in permutations:
        term_sum += (
            sgn
            * eval_term(pos_ops[p1], pos_ops[p2], pos_ops[p3]) 
            / 6.0
        )

    C = N_D * W @ term_sum
    return C


def compute_wrapper(method, M, n=None, L=None, pasted=False, save_data=True, directory="./data/local_marker/sponge/"):
    if method == 'cube':
        l = np.ones((L, L, L), dtype=int) # type: ignore
    else:
        l = lattice.build_lattice("sponge", n=n, block_scale=1, pasted=pasted)
    params = {"M": M, "M_alt": M, "M_prime": 0.01, "disorder_seed": 0, "disorder_strength": 0.0, "t": 1., "B": 1., "g": 0, "gauge": "N"}

    size_tag = f"_L={l.shape[0]}" if method == 'cube' else f"_n={n}_L={l.shape[0]}"
    filename = f"{method}_M={params["M"]:.3f}" + size_tag + ".h5"

    if os.path.exists(directory + filename):
        with h5py.File(directory + filename, "r") as f:
            C:np.ndarray = f["C"][()] # type: ignore
            eigenvalues:np.ndarray = f["eigenvalues"][()] # type: ignore
            m_read = f["M"][()] # type: ignore
            assert np.isclose(params["M"], m_read) # type: ignore
        return C, eigenvalues, l # type: ignore

    if method == 'cube' and L == None:
        raise ValueError()
    if method != 'cube' and n == None:
        raise ValueError()
    
    if method == 'cube':
        m = model.build_model_arbitrary(L, 3)
    else:
        m = model.build_model("sponge", n=n, block_scale=1, pasted=pasted, hole_treatment=method)

    if method == 'renorm':
        res = solve.schur_solve(m, "sector", 0, params=params, hermitian=True, return_LDOS=True)
    else:
        res = solve.solve_model(m, apply_vacancies=True if method in ['site_elim'] else False, hermitian=True, return_LDOS=True, params=params)

    eigenvalues = res['eigenvalues']
    eigenvectors = res['eigenvectors']
    C = np.real(np.diag(compute_topological_marker(l, method, eigenvalues, eigenvectors)).reshape(-1, 4).sum(axis=1))

    if save_data:
        with h5py.File(directory + filename, "w") as f:
            f.create_dataset(name="C", data=C)
            f.create_dataset(name="eigenvalues", data=eigenvalues)
            f.create_dataset(name="M", data=params["M"])

    return C, eigenvalues, l


def plot_lcm(method, l, C, plot_type='radial'):
    if method in ['site_elim', 'renorm']: 
        mask = l > 0
    else:
        mask = l >= 0
    X, Y, Z = np.asarray(np.where(mask), dtype=float)
    X -= np.mean(X)
    Y -= np.mean(Y)
    Z -= np.mean(Z)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    n = round(np.log(l.shape[0])/np.log(3))

    if (0.0 < M <= 4.0) or (8.0 < M <= 12.0):
        y = 1.0
    elif 4.0 < M <= 8.0:
        y = -2.0
    else:
        y = 0.0

    plt.axhline(y, c='k', ls='--', alpha=0.5, zorder=-10)
    if method == 'cube':
        plt.title(f"L={l.shape[0]} : M={M:.2f}")
    else:
        plt.title(f"{method} : n={n} : M={M:.2f}")
    plt.ylim(-3.0, 2.0)

    if plot_type == 'radial': 
        plt.scatter(r, C.flatten(), alpha=0.5)
        plt.xlabel('Distance from origin $\\vec r$'); plt.ylabel("$C(\\vec r)$")
    elif plot_type == 'body_diagonal':
        pos = []
        cs = []
        C_box = np.full(l.shape, np.nan)
        C_box[mask] = C
        for i in range(l.shape[0]):
            pos.append(i)
            cs.append(C_box[i, i, i])
        plt.scatter(pos, cs)
        plt.xlabel('Position along body diagonal'); plt.ylabel("$C(\\vec r)$")

    if method == 'cube':
        plt.savefig(f"./figures/3D/{method}_L={l.shape[0]}_M={M:.2f}.png")
    else:
        plt.savefig(f"./figures/3D/{method}_n={n}_M={M:.2f}.png")
    plt.close()


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
        if vmin == vmax:
            vmax = vmin + 1
        norm = Normalize(vmin=vmin, vmax=vmax)
        color_mapper = plt.get_cmap(cmap)
        facecolors = color_mapper(norm(colors))
    else:
        facecolors = colors

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels
    v = ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors, alpha=alpha)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if np.issubdtype(colors.dtype, np.number) and colors.ndim == 3:
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=color_mapper),
            ax=ax,
            label='Value',
        )
    return fig, ax


if __name__ == "__main__":
    method = 'site_elim'
    for M in [2.0, 6.0, 10.0, -2.0]:
        C, eigenvalues, l = compute_wrapper(method, M, L=6, n=2)
        #plot_lcm(method, l, C, 'body_diagonal')
        C_box = np.full(l.shape, np.nan)
        C_box[l == 1] = C
        plot_3d_voxels(l == 1, C_box)
        plt.show()






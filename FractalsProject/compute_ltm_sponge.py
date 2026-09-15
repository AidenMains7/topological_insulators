import numpy as np
import scipy.sparse as sp
import os, h5py
import threadpoolctl

from project_tools import lattice, model
from hypercubic import solve

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

DEFAULT_DATA_SAVE_DIRECTORY = "./data/local_marker/sponge/"

def compute_topological_marker(
    l: np.ndarray,
    method: str,
    eigenvalues: np.ndarray,
    eigenvectors,
    fermi_energy: float = 0.0,
    symmetrize: bool = True,
):
    filled_idxs = np.argwhere(eigenvalues < fermi_energy).flatten()
    empty_idxs = np.argwhere(eigenvalues > fermi_energy).flatten()

    if np.any(np.isclose(eigenvalues, fermi_energy)):
        raise ValueError("Fermi energy coincides with an eigenvalue.")

    # Compute projection operators P and Q
    # Works for both numpy dense ndarrays and scipy sparse matrices
    V_filled = eigenvectors[:, filled_idxs]
    P = V_filled @ V_filled.conj().T

    V_empty = eigenvectors[:, empty_idxs]
    Q = V_empty @ V_empty.conj().T

    if method in ["site_elim", "renorm"]:
        X, Y, Z = np.where(l > 0)
    else:
        X, Y, Z = np.where(l >= 0)

    n_total = eigenvectors.shape[0]
    n_dof_per_site = n_total // len(X)

    # Position operators as sparse diagonal matrices (CSR format)
    X_vec = np.repeat(X, n_dof_per_site).astype(np.complex128)
    Y_vec = np.repeat(Y, n_dof_per_site).astype(np.complex128)
    Z_vec = np.repeat(Z, n_dof_per_site).astype(np.complex128)

    X_op = sp.diags(X_vec, format="csr")
    Y_op = sp.diags(Y_vec, format="csr")
    Z_op = sp.diags(Z_vec, format="csr")
    pos_ops = {"x": X_op, "y": Y_op, "z": Z_op}

    N_D = 8.0 * np.pi * 1.0j

    # Sparse construction of Chiral/Symmetry Operator W
    sigma0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    sigma2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    G5 = -np.kron(sigma2, sigma0)

    W = sp.kron(
        sp.eye(n_total // 4, format="csr"),
        sp.csr_matrix(G5),
        format="csr",
    )

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
        norm_factor = 6.0
    else:
        permutations = [(("x", "y", "z"), +1.0)]
        norm_factor = 1.0

    # Initialize term_sum matching input matrix structure
    if sp.issparse(P):
        term_sum = sp.csr_matrix((n_total, n_total), dtype=np.complex128)
    else:
        term_sum = np.zeros((n_total, n_total), dtype=np.complex128)

    for (p1, p2, p3), sgn in permutations:
        term_sum = term_sum + (
            (sgn / norm_factor)
            * eval_term(pos_ops[p1], pos_ops[p2], pos_ops[p3])
        )

    C = N_D * (W @ term_sum)
    return C


def compute_wrapper(method, M, n=None, L=None, b=1, pasted=False, save_data=True,
                     directory=DEFAULT_DATA_SAVE_DIRECTORY, M_alt=None,
                     n_threads=None, driver="evr"):
    """
    n_threads : int or None
        Caps how many BLAS threads the dense diagonalization is allowed to
        use (via threadpoolctl), for this call only -- doesn't touch global
        env vars or affect anything outside this `with` block. None leaves
        whatever's already configured (env vars / library defaults) alone.
    driver : {"evr", "evd", "ev", "evx"}
        LAPACK driver passed through to scipy.linalg.eigh. "evd"
        (divide-and-conquer) does more total work than the default "evr" but
        in a much more parallelizable structure, so it's the one most likely
        to actually benefit from n_threads > 1 -- but which wins depends on
        matrix size and core count, so benchmark both on the target machine
        rather than assuming; "evr" was faster single-threaded in testing here.
    """
    if method == 'cube':
        l = np.ones((L, L, L), dtype=int) # type: ignore
    else:
        l = lattice.build_lattice("sponge", n=n, block_scale=b, pasted=pasted)
    params = {"M": M, "M_alt": M_alt, "M_prime": 0.01, "disorder_seed": 0, "disorder_strength": 0.0, "t": 1., "B": 1., "g": 0, "gauge": "N"}

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
        m = model.build_model("sponge", n=n, block_scale=b, pasted=pasted, hole_treatment=method)

    solver_kwargs = {"driver": driver}
    with threadpoolctl.threadpool_limits(limits=n_threads, user_api="blas"):
        if method == 'renorm':
            res = solve.schur_solve(m, "sector", 0, params=params, hermitian=True,
                                     return_LDOS=True, solver_kwargs=solver_kwargs)
        else:
            res = solve.solve_model(m, apply_vacancies=True if method in ['site_elim'] else False,
                                     hermitian=True, return_LDOS=True, params=params,
                                     solver_kwargs=solver_kwargs)

    eigenvalues = res['eigenvalues']
    eigenvectors = res['eigenvectors']
    C = np.real(np.diag(compute_topological_marker(l, method, eigenvalues, eigenvectors)).reshape(-1, 4).sum(axis=1))

    if save_data:
        with h5py.File(directory + filename, "w") as f:
            f.create_dataset(name="C", data=C)
            f.create_dataset(name="eigenvalues", data=eigenvalues)
            f.create_dataset(name="M", data=params["M"])

    return C, eigenvalues, l


def plot_lcm(method, M, M_alt, l, n, b, C, plot_type='radial'):
    if method in ['site_elim', 'renorm']: 
        mask = l > 0
    else:
        mask = l >= 0
    X, Y, Z = np.asarray(np.where(mask), dtype=float)
    X -= np.mean(X)
    Y -= np.mean(Y)
    Z -= np.mean(Z)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    n = round(np.log(l.shape[0] / b / (pasted + 1))/np.log(3))

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
        plt.title(f"{method} : n={n} : L={l.shape[0]} : M={M:.2f} : M_alt={M_alt:.2f}")
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
        plt.savefig(f"./figures/3D/{method}_L={l.shape[0]}_M={M:.2f}_M_alt={M_alt:.2f}.png")
    else:
        plt.savefig(f"./figures/3D/{method}_n={n}_b={b}_p={pasted}_M={M:.2f}_M_alt={M_alt:.2f}.png")
    plt.close()


def plot_3d_voxels(voxels, colors, cmap='viridis', edgecolors='k', alpha=0.8,
                   ax=None, plot_diagonal_half=False):
    """
    Plots a 3D voxel grid where voxels and colors share the same 3D spatial shape.

    Parameters:
        voxels (np.ndarray): 3D array (X, Y, Z). Non-zero or True values indicate filled voxels.
        colors (np.ndarray): Array with shape matching `voxels` (X, Y, Z). Contains color strings,
                             RGBA values, or scalar numerical values to map via `cmap`.
        cmap (str or Colormap): Matplotlib colormap used if `colors` contains numerical data.
        edgecolors (str): Line color for voxel edges.
        alpha (float): Opacity of the voxel faces.
        plot_diagonal_half (bool): If True, show only the half-space X >= Y,
                       leaving the diagonal cross section visible.

    Returns:
        fig, ax: Matplotlib Figure and Axes3D objects.
    """
    filled = np.asarray(voxels, dtype=bool)
    colors = np.asarray(colors)

    if filled.shape != colors.shape[:3]:
        raise ValueError(f"Shape mismatch: voxels {filled.shape} vs colors {colors.shape[:3]}")

    if plot_diagonal_half:
        x, y, z = np.indices(filled.shape)
        diagonal_half = (x >= y) & (y >= z)
        filled = filled & diagonal_half

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

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
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
    return v


if __name__ == "__main__":
    from time import time
    method = 'substituted'; n=1; b=2; pasted=True

    M_alt = 2.0; M = -0.8
    t0 = time()
    C, eigenvalues, l = compute_wrapper(method, M, L=None, n=n, b=b, pasted=pasted, M_alt=M_alt, n_threads=6)
    print(f"{time()-t0:.2f}s")
    plot_lcm(method, M, M_alt, l, n, b, C, 'body_diagonal')
    #C_box = np.full(l.shape, np.nan)
    #C_box[l == 1] = C
    #plot_3d_voxels(l == 1, C_box)
    #plt.show()
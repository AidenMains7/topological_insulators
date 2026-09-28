import numpy as np
import scipy.sparse as sp
import os, h5py

from itertools import product
import threadpoolctl
from tqdm_joblib import tqdm, tqdm_joblib
from joblib import Parallel, delayed

from project_tools import lattice, model
from hypercubic import solve
from compute_ltm_cantor import compute_ldos

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import rcParams

DEFAULT_DATA_SAVE_DIRECTORY = "./data/local_marker/sponge/"
rcParams['axes.linewidth'] = 2.5
rcParams['xtick.major.width'] = 2.5
rcParams['ytick.major.width'] = 2.5

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
        l = np.ones((L * b, L * b, L * b), dtype=int) # type: ignore
    else:
        l = lattice.build_lattice("sponge", n=n, block_scale=b, pasted=pasted)
    params = {"M": M, "M_alt": M_alt, "M_prime": 0.01, "disorder_seed": 0, "disorder_strength": 0.0, "t": 1., "B": 1., "g": 0, "gauge": "N"}

    size_tag = f"_L={l.shape[0]}" if method == 'cube' else f"_n={n}_L={l.shape[0]}"
    filename = f"{method}_M={params['M']:.3f}_Malt={params['M_alt']}:.3f" + size_tag + ".h5"
    print(filename)
    print(os.path.exists(directory + filename))

    if os.path.exists(directory + filename):
        with h5py.File(directory + filename, "r") as f:
            C:np.ndarray = f["C"][()] # type: ignore
            eigenvalues:np.ndarray = f["eigenvalues"][()] # type: ignore
            ldos = f["LDOS"][()] # type: ignore
            m_read = f["M"][()] # type: ignore
            assert np.isclose(params["M"], m_read) # type: ignore
        return C, eigenvalues, ldos, l # type: ignore

    if method == 'cube' and L == None:
        raise ValueError()
    if method != 'cube' and n == None:
        raise ValueError()
    
    if method == 'cube':
        m = model.build_model_arbitrary(L, 3, b=b)
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
    ldos = compute_ldos(eigenvalues, eigenvectors)

    if save_data:
        with h5py.File(directory + filename, "w") as f:
            f.create_dataset(name="C", data=C)
            f.create_dataset(name="eigenvalues", data=eigenvalues)
            f.create_dataset(name="LDOS", data=ldos)
            f.create_dataset(name="M", data=params["M"])

    return C, eigenvalues, ldos, l


def plot_lcm(ax, method, M, M_alt, l, n, b, C, plot_type='radial', pasted=False, **plot_kwargs):
    ax_init_none = False
    if ax == None:
        fig, ax = plt.subplots(1, 1)
        ax_init_none = True

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

    ax.axhline(y, c='k', ls='--', alpha=0.5, zorder=-10)
    if method == 'cube':
        ax.set_title(f"L={l.shape[0]} : M={M:.2f}")
    else:
        ax.set_title(f"{method} : n={n} : L={l.shape[0]} : M={M:.2f} : M_alt={M_alt:.2f}")
    ax.set_ylim(-3.0, 2.0)

    if plot_type == 'radial': 
        ax.scatter(r, C.flatten(), alpha=0.5, **plot_kwargs)
        ax.set_xlabel('Distance from origin $\\vec r$'); ax.set_ylabel("$C(\\vec r)$")
    elif plot_type == 'body_diagonal':
        pos = []
        cs = []
        C_box = np.full(l.shape, np.nan)
        C_box[mask] = C
        for i in range(l.shape[0]):
            pos.append(i)
            cs.append(C_box[i, i, i])
        ax.scatter(pos, cs, **plot_kwargs)
        ax.set_xlabel('Position along body diagonal'); ax.set_ylabel("$C(\\vec r)$")

        xticks = np.linspace(0, np.max(pos), 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(round(t + 1, 2)) for t in xticks])

        yticks = [-2.0, -1.0, 0.0, 1.0]
        ax.set_yticks(yticks)

    if ax_init_none == True:
        if method == 'cube':
            plt.savefig(f"./figures/3D/{method}_L={l.shape[0]}_M={M:.2f}_M_alt={M_alt:.2f}.svg")
        else:
            plt.savefig(f"./figures/3D/{method}_n={n}_b={b}_p={pasted}_M={M:.2f}_M_alt={M_alt:.2f}.svg")
        plt.close()


def plot_3d_voxels(voxels, colors, cmap='viridis', edgecolors='k', alpha=0.8,
                   ax=None, plot_diagonal_half=False, **plot_kwargs):
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
    v = ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors, alpha=alpha, **plot_kwargs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #if np.issubdtype(colors.dtype, np.number) and colors.ndim == 3:
    #    fig.colorbar(
    #        plt.cm.ScalarMappable(norm=norm, cmap=color_mapper),
    #        ax=ax,
    #        label='Value',
    #    )
    return v, ax


def plot_3d_path_visualization():
    def _make_arrays(lattice_array):
        L_half = lattice_array.shape[0] // 2
        octant_idxs = np.arange(lattice_array.shape[0])[lattice_array.shape[0]//2:]
        octant_mask = np.full(lattice_array.shape, False)
        octant_mask[np.ix_(octant_idxs, octant_idxs, octant_idxs)] = True
        octant = lattice_array[octant_mask].reshape([L_half]*3)
        
        color_array = (~octant.astype(bool)).astype(int)
        # Get the path along y=z=0 from x=0 to x=L_half
        color_array[0:L_half-1, 0, 0] = 2
        # Then along x=L_half z=0 from y=0 to y=L_half
        color_array[L_half-1, 0:L_half-1, 0] = 2
        # Then along x=y=L_half from z=0 to z=L_half
        color_array[L_half-1, L_half-1, 0:L_half-1] = 2
        # Then from diag (L_half, L_half, L_half) to (0, 0, 0) with holes filled with np.nan
        diag_idx = np.arange(L_half-1, -1, -1)
        color_array[diag_idx, diag_idx, diag_idx] = 2

        # Mask to show only a portion of the octant
        positions = np.indices(octant.shape)
        mask1 = positions[0] >= positions[1]
        mask2 = positions[1] >= positions[2]
        mask = ~(mask1 & mask2)
        octant[mask] = False
        return octant, color_array

    l = lattice.build_lattice('sponge', n=1, block_scale=4, pasted=True)
    l_cube = np.ones(l.shape)

    o1, c1 = _make_arrays(l_cube)
    o2, c2 = _make_arrays(l)

    from matplotlib.colors import ListedColormap, LightSource
    black = np.array([0., 0., 0., 1.])
    red = np.array([1., 0., 0., 1.])
    cmap_arr = np.array([black, red])
    new_cmap = ListedColormap(cmap_arr)

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    ls = LightSource(azdeg=315, altdeg=45)

    plot_3d_voxels(o1, c1, new_cmap, 'w', 1.0, axs[0], shade=True, lightsource=ls)
    plot_3d_voxels(o2, c2, new_cmap, 'w', 1.0, axs[1], shade=True, lightsource=ls)
    plot_3d_voxels(o1, c1, new_cmap, None, 0.1, axs[1], shade=True, lightsource=ls)

    for ax in axs:
        ax.set_axis_off()
        ax.view_init(elev=30, azim=-45, roll=0)

    plt.savefig("./figures/3D/graphic.svg")
    plt.show()


def get_3d_ldos_path(l, ldos):
    mask = (l == 1)
    box = np.full(l.shape, np.nan)
    box[mask] = ldos[::2] + ldos[1::2]

    mask2 = np.full(l.shape, False)
    idxs = np.arange(l.shape[0])[l.shape[0]//2:]
    mask2[np.ix_(idxs, idxs, idxs)] = True
    L_half = int(l.shape[0] / 2)

    box = box[mask2].reshape(L_half, L_half, L_half)
    # Get the path along y=z=0 from x=0 to x=L_half
    leg1 = box[0:L_half-1, 0, 0]
    # Then along x=L_half z=0 from y=0 to y=L_half
    leg2 = box[L_half-1, 0:L_half-1, 0]
    # Then along x=y=L_half from z=0 to z=L_half
    leg3 = box[L_half-1, L_half-1, 0:L_half-1]
    # Then from diag (L_half, L_half, L_half) to (0, 0, 0) with holes filled with np.nan
    diag_idx = np.arange(L_half-1, -1, -1)
    leg4 = box[diag_idx, diag_idx, diag_idx]

    path = np.concatenate([leg1, leg2, leg3, leg4], dtype=float).flatten()
    t = np.arange(path.size)
    return t, path


if __name__ == "__main__":
    plot_3d_path_visualization()
    raise SystemExit
    methods = ['cube', 'site_elim', 'renorm']
    Ms = [-2.0, 2.0, 6.0]
    n = 1; b = 4

    fig, axs = plt.subplots(len(methods), 2, figsize=(12, 8), sharex=False, sharey=False)
    #fig2, axs2 = plt.subplots(len(methods), len(Ms), figsize=(20, 20), sharex=True, sharey=True, subplot_kw={'projection':'3d'})
    for i in range(len(methods)):
        for j in range(len(Ms)):
            method = methods[i]
            M = Ms[j]
            pasted=False if method == 'cube' else True
            C, eigenvalues, ldos, l = compute_wrapper(method, M, L=24, n=n, pasted=pasted, b=1 if method == 'cube' else b)
            plot_lcm(axs[i, 0], method, M, M, l, n, b, C,'body_diagonal', pasted=pasted, label=f"M={M}")

            t, path = get_3d_ldos_path(l, ldos)
            L_half = int(l.shape[0] / 2)

            axs[i, 1].scatter(t, path)
            axs[i, 1].set_xticks(np.arange(0, path.size, L_half-1))
            axs[i, 1].set_xticklabels([str(int(ti+1)) for ti in axs[i, 1].get_xticks()])

            axs[i, 0].legend()
    plt.tight_layout()
    plt.savefig(f"./figures/3D/.ldos.svg")
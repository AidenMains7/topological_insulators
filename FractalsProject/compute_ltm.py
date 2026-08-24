import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
import os, h5py
from tqdm_joblib import tqdm_joblib, tqdm
from joblib import Parallel, delayed

from project_tools import lattice, model
from hypercubic import solve


def compute_eigvals(H):
    eigvals, eigvectors = spla.eigh(H, overwrite_a=True)
    sort_idxs = np.argsort(eigvals)
    eigvals = eigvals[sort_idxs]
    eigvectors = eigvectors[sort_idxs, :]
    return eigvals, eigvectors


def compute_topological_marker(eigenvalues, eigenvectors):
    # C_1D = N_D W [QxP + PxQ]

    n = len(eigenvalues)
    empty_idxs =  np.arange(n)[:n // 2]
    filled_idxs = np.arange(n)[n // 2:]

    P = np.zeros(eigenvectors.shape, dtype=eigenvectors.dtype)
    Q = np.zeros(eigenvectors.shape, dtype=eigenvectors.dtype)

    for i in filled_idxs:
        vec = eigenvectors[:, i]
        P += np.outer(vec, vec.conj())

    for i in empty_idxs:
        vec = eigenvectors[:, i]
        Q += np.outer(vec, vec.conj())

    pauli_z = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]])
    W = np.kron(np.eye(eigenvectors.shape[0] // 2), pauli_z)

    X = np.diag(np.repeat(np.arange(eigenvectors.shape[0] // 2, dtype=eigenvectors.dtype), 2))

    N_D = 1.
    alpha = Q @ X @ P
    C = (N_D * W @ (alpha + alpha.conj().T)).real

    return C, eigenvalues, eigenvectors


def compute_wrapper(m, l, M, method, M_alt=None):
    params = dict(M=M, M_alt=M_alt if M_alt != None else M, t=1.0, B=1.0, disorder_strength=0.0, disorder_seed=0)
    if method == 'renorm':
        result = solve.schur_solve(m, "sector", 0, params=params, hermitian=True)
        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]
    else:
        H = m.assemble(True, **params).toarray()    
        eigenvalues, eigenvectors = compute_eigvals(H)

    C, _, _ = compute_topological_marker(eigenvalues, eigenvectors)
    return C


def compute_phase(h_method:str, n:int, b:int, M_values:np.ndarray, directory:str = "./"):
    l = lattice.build_lattice("cantor", n, block_scale=b)
    m = model.build_model("cantor", n, hole_treatment=h_method, block_scale=b)
    filename = directory + h_method + f"_n={n}_b={b}.h5"

    if os.path.exists(filename):
        with h5py.File(filename, "r") as f:
            Ms:np.ndarray = f["Ms"][()] # type: ignore
            means:np.ndarray = f["means"][()] # type: ignore
            return Ms, means

    def _worker(M):
        C = compute_wrapper(m, l, M, h_method)
        return [M, np.mean(np.diag(C))]

    with tqdm_joblib(tqdm(total=len(M_values))) as progress_bar:
        data = np.array(Parallel(n_jobs=-2)(delayed(_worker)(M) for M in M_values)).T
    Ms, means = data

    with h5py.File(filename, "w") as f:
        f.create_dataset(name="Ms", data=Ms)
        f.create_dataset(name="means", data=means)
    return Ms, means


def plot_local_topological_marker(n, b, M, method, ax=None, M_alt=None):
    l = lattice.build_lattice("cantor", n, block_scale=b)
    m = model.build_model("cantor", n, hole_treatment=method, block_scale=b)
    C = compute_wrapper(m, l, M, method, M_alt)
    c_diag = np.diag(C)
    c_diag = c_diag[::2] + c_diag[1::2]

    t = np.arange(l.size)
    y = np.full(t.shape, np.nan)
    site_mask = l.astype(bool)
    y[site_mask] = c_diag

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    extent = (np.min(t), np.max(t), max(-3.0, np.min(y)), min(3.0, np.max(y)))
    ax.imshow(l[np.newaxis], aspect='auto', cmap='Greys', alpha=0.25, zorder=-1, extent=extent)
    ax.plot(t, y)
    ax.set_ylim(extent[2], extent[3])
    ax.figure.suptitle(f"{method}\nn={n}, L={l.size}")
    ax.axhline(-1.0)

if __name__ == "__main__":

    if 1: 
        Ms = np.arange(-1, 5, 1)
        #Ms = [-0.5, -0.1, -0.05, 0.05, 0.1, 0.5]
        #Ms = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        fig, axs = plt.subplots(2, 3)
        for ax, m in zip(axs.flatten(), Ms):
            plot_local_topological_marker(3, 27, m, "site_elim", ax)
            ax.annotate(
                f"$M={m:.2f}$",
                xy=(0.05, 0.95),
                ha="left",
                va="top",
                xycoords='axes fraction'
            )

        
        plt.tight_layout()
        plt.savefig('renorm_0.png')
        plt.show()


    #plot_local_topological_marker(3, 9, -0.1, "renorm")
    #plt.show()
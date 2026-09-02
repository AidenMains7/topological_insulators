import numpy as np
import scipy.linalg as spla
import os, h5py
from tqdm_joblib import tqdm_joblib, tqdm
from joblib import Parallel, delayed

from project_tools import lattice, model
from hypercubic import solve

import matplotlib.pyplot as plt
import scipy.sparse as spsp

def compute_wannier_matrices(L:int):
    Sx = spsp.dok_matrix((L, L), dtype=np.complex128)
    Cx = spsp.dok_matrix((L, L), dtype=Sx.dtype)

    for i in range(L - 1):
        Sx[i, i + 1] = 1j / 2
        Cx[i, i + 1] = 1.

    Sx += Sx.conj().T
    Cx += Cx.conj().T

    I = spsp.identity(L, dtype=Sx.dtype, format="dok")
    return I, Sx, Cx


def compute_hamiltonian(l, M, B, t, M_alt=None, *args, **kwargs):
    I, Sx, Cx = compute_wannier_matrices(l.size)

    if M_alt is None:
        M_alt = M

    M_term = I.todense().copy()
    site_mask = l.astype(bool)
    M_term[np.ix_(site_mask, site_mask)] *= M
    M_term[np.ix_(~site_mask, ~site_mask)] *= M_alt

    d1 = t * Sx
    d2 = M_term + (-2 * B) * I + 2 * B * Cx

    if spsp.issparse(d1):
        d1 = d1.todense()
    if spsp.issparse(d2):
        d2 = d2.todense()

    pauli_x = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]])
    pauli_y = np.array([[0.0 + 0.0j, -1j], [1j, 0.0 + 0.0j]])
    return np.kron(d1, pauli_x) + np.kron(d2, pauli_y)

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

    return C


def compute_wrapper(n, b, M, method, M_alt=None, overwrite:bool = False, directory="./data/local_marker/") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    VALID_METHODS = ['renorm', 'renorm_alt', 'substituted', 'site_elim', 'substituted_alt', 'site_elim_alt']
    assert method in VALID_METHODS, f"method must be in {VALID_METHODS}"

    l = lattice.build_lattice("cantor", n, block_scale=b)
    params = dict(M=M, M_alt=M_alt if M_alt != None else M, t=1.0, B=1.0, disorder_strength=0.0, disorder_seed=0)

    filename = f"{method}_M={params['M']:.3f}_M_alt={params['M_alt']:.3f}_n={n}_L={l.size}.h5"
    cache_path = directory + filename
    if os.path.exists(cache_path) and not overwrite:
        with h5py.File(cache_path, "r") as f:
            C:np.ndarray = f["C"][()] # type: ignore
            m_read = f["M"][()] # type: ignore
            m_alt_read = f["M_alt"][()] # type: ignore
            eigenvalues = f["eigenvalues"][()] # type: ignore
            eigenvectors = f["eigenvectors"][()]             # type: ignore
            assert np.isclose(params["M"], m_read) and np.isclose(params["M_alt"], m_alt_read) # type: ignore
        return C, eigenvalues, eigenvectors # type: ignore

    if method in ["renorm", "site_elim", "substituted"]:
        m = model.build_model("cantor", n, hole_treatment=method, block_scale=b)

    if method == 'renorm':
        result = solve.schur_solve(m, "sector", 0, params=params, hermitian=True)
        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]
    elif method == 'site_elim':
        H = m.assemble(True, format='csr', **params).toarray()
    elif method == 'substituted':
        H = m.assemble(False, format='csr', **params).toarray() 
    else: 
        H = compute_hamiltonian(l, **params)
        #H = m.assemble(False, format='csr', **params).toarray()  
        if method == "substituted_alt":
            pass
        else:
            mask = np.repeat(l.astype(bool), 2)
            W = np.random.random(H.shape[0]) * 2 - 1
            W *= 1e-6
            W -= np.mean(W)
            H += np.diag(W)
            H_aa = H[np.ix_(mask, mask)]
            H_ab = H[np.ix_(mask, ~mask)]
            H_ba = H[np.ix_(~mask, mask)]
            H_bb = H[np.ix_(~mask, ~mask)]
            if method == "site_elim_alt":
                H = H_aa
            elif method == 'renorm_alt':
                H = H_aa + H_ab @ spla.solve(H_bb, H_ba)
    if method != "renorm":
        eigenvalues, eigenvectors = compute_eigvals(H)
    C = compute_topological_marker(eigenvalues, eigenvectors)

    if overwrite:
        with h5py.File(directory + filename, "w") as f:
            f.create_dataset(name="C", data=C)
            f.create_dataset(name="eigenvalues", data=eigenvalues)
            f.create_dataset(name="eigenvectors", data=eigenvectors)
            f.create_dataset(name="M", data=params["M"])
            f.create_dataset(name="M_alt", data=params["M_alt"])
    return C, eigenvalues, eigenvectors


def compute_phase(h_method:str, n:int, b:int, M_values:np.ndarray, directory:str = "./"):
    l = lattice.build_lattice("cantor", n, block_scale=b)
    filename = directory + h_method + f"_n={n}_b={b}.h5"

    if os.path.exists(filename):
        with h5py.File(filename, "r") as f:
            Ms:np.ndarray = f["Ms"][()] # type: ignore
            means:np.ndarray = f["means"][()] # type: ignore
            return Ms, means

    def _worker(M):
        C, _, _ = compute_wrapper(n, b, M, h_method)
        return [M, np.mean(np.diag(C))]

    with tqdm_joblib(tqdm(total=len(M_values))) as progress_bar:
        data = np.array(Parallel(n_jobs=-2)(delayed(_worker)(M) for M in M_values)).T
    Ms, means = data

    with h5py.File(filename, "w") as f:
        f.create_dataset(name="Ms", data=Ms)
        f.create_dataset(name="means", data=means)
    return Ms, means


def compute_ldos(n, b, M, method, M_alt=None, overwrite=True, tol=1e-10):
    _, eigenvalues, eigenvectors = compute_wrapper(n, b, M, method, M_alt, overwrite)

    # 1. Asymmetry failsafe: Sort by distance to E=0 instead of strict signs.
    # Prevents crashes if the spectrum is entirely positive/negative or shifted.
    abs_energies = np.abs(eigenvalues)
    min_E = np.min(abs_energies)

    # 2. Degeneracy failsafe: Capture ALL states within numerical tolerance 
    # of the closest energy level (e.g., multiple edge modes or flat bands).
    idxs = np.where(abs_energies <= min_E + tol)[0]

    # Fallback: If the gap is clean, non-degenerate, and shifted, ensure 
    # we still capture at least the two closest states (HOMO/LUMO equivalent).
    if len(idxs) < 2 and len(eigenvalues) >= 2:
        idxs = np.argsort(abs_energies)[:2]

    # Slice COLUMNS (eigenstates), and sum across the entire degenerate subspace
    ldos = np.sum(np.abs(eigenvectors[:, idxs]) ** 2, axis=1)


    
    #mask = np.abs(eigenvalues) < 1000
    #plt.scatter(np.arange(len(eigenvalues))[mask], eigenvalues[mask])
    #plt.scatter(np.arange(len(eigenvalues))[idxs], eigenvalues[idxs], c='red')
    #plt.show()
    
    
    # Trace over the internal degrees of freedom (e.g., spin/sublattice)
    ldos = ldos[::2] + ldos[1::2]
    return ldos


if __name__ == "__main__":
    pass
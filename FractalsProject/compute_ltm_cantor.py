import numpy as np
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

    return C


def compute_wrapper(m, n, b, M, method, M_alt=None, save_data:bool = False, directory="./data/local_marker/cantor/") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = dict(M=M, M_alt=M_alt if M_alt != None else M, t=1.0, B=1.0, disorder_strength=0.0, disorder_seed=0)
    l = lattice.build_lattice("cantor", n, block_scale=b)
    filename = f"{method}_M={params["M"]:.3f}_M_alt={params["M_alt"]:.3f}_n={n}_L={l.size}.h5"
    if os.path.exists(directory + filename):
        with h5py.File(directory + filename, "r") as f:
            C:np.ndarray = f["C"][()] # type: ignore
            m_read = f["M"][()] # type: ignore
            m_alt_read = f["M_alt"][()] # type: ignore
            eigenvalues = f["eigenvalues"][()] # type: ignore
            ldos = f["LDOS"][()] # type: ignore
            assert np.isclose(params["M"], m_read) and np.isclose(params["M_alt"], m_alt_read) # type: ignore
        return C, eigenvalues, ldos # type: ignore

    if method == 'renorm':
        result = solve.schur_solve(m, "sector", 0, params=params, hermitian=True)
        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]
    else:
        H = m.assemble(True, **params).toarray()    
        eigenvalues, eigenvectors = compute_eigvals(H)

    C = compute_topological_marker(eigenvalues, eigenvectors)
    ldos = compute_ldos(eigenvalues, eigenvectors)

    if save_data:
        with h5py.File(directory + filename, "w") as f:
            f.create_dataset(name="C", data=C)
            f.create_dataset(name="eigenvalues", data=eigenvalues)
            f.create_dataset(name="LDOS", data=ldos)
            f.create_dataset(name="M", data=params["M"])
            f.create_dataset(name="M_alt", data=params["M_alt"])
    return C, eigenvalues, ldos


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
        C, _, _ = compute_wrapper(m, n, b, M, h_method)
        return [M, np.mean(np.diag(C))]

    with tqdm_joblib(tqdm(total=len(M_values))) as progress_bar:
        data = np.array(Parallel(n_jobs=-2)(delayed(_worker)(M) for M in M_values)).T
    Ms, means = data

    with h5py.File(filename, "w") as f:
        f.create_dataset(name="Ms", data=Ms)
        f.create_dataset(name="means", data=means)
    return Ms, means


def compute_ldos(eigenvalues, eigenvectors, tol=1e-10):
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

    # Trace over the internal degrees of freedom (e.g., spin/sublattice)
    ldos = ldos[::2] + ldos[1::2]
    return ldos

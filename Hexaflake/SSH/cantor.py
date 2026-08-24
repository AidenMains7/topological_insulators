import numpy as np
from scipy import linalg as spla
from matplotlib import pyplot as plt
from itertools import product

from joblib import Parallel, delayed
from tqdm_joblib import tqdm, tqdm_joblib
import h5py, os

import dan_lattice as dl
from ssh import compute_topological_marker


def compute_wannier_matrices(lattice):
    X = np.arange(lattice.size)

    dx = X[np.newaxis, :] - X[:, np.newaxis]

    xp_mask = np.isclose(dx, 1.0).astype(bool)

    Sx = np.full(dx.shape, 0., dtype=np.complex128)
    Cx = np.full(dx.shape, 0., dtype=np.complex128)

    Sx[xp_mask] = 1j / 2
    Cx[xp_mask] = 1 / 2

    Sx += Sx.conj().T
    Cx += Cx.conj().T
    return np.eye(dx.shape[0]), Sx, Cx


def compute_hamiltonian(lattice, method:str, M:float, M_ALT:float|None = None, B:float = 1.0, t:float = 1.0):
    assert method in ["renorm", "site_elim", "sub"]

    pauli_x = np.array([[0. + 0.j, 1. + 0.j], [1. + 0.j, 0. + 0.j]])
    pauli_y = np.array([[0. + 0.j, 0. - 1.j], [0. + 1.j, 0. + 0.j]])

    I, Sx, Cx = compute_wannier_matrices(lattice)

    M_term = M * I
    hole_idxs = np.argwhere(lattice == 0).flatten()

    if method == "sub":
        M_term[hole_idxs, hole_idxs] = M_ALT

    term1 = t * Sx
    term2 = M_term - 2 * B * (I - Cx)

    H = np.kron(term1, pauli_x) + np.kron(term2, pauli_y)

    hole_mask = np.full(lattice.size, False)
    hole_mask[hole_idxs] = True

    H_aa = H[np.ix_(~hole_mask, ~hole_mask)]
    H_bb = H[np.ix_(hole_mask, hole_mask)]
    H_ab = H[np.ix_(~hole_mask, hole_mask)]
    H_ba = H[np.ix_(hole_mask, ~hole_mask)]

    if method == "site_elim":
        return H_aa
    elif method == "renorm":
        try:
            X = spla.solve(H_bb, H_ba, assume_a='her', check_finite=True, overwrite_a=True, overwrite_b=True)
            return H_aa - H_ab @ X
        except Exception as e:
            print(f"Exception in computing the hamiltonian: {e}")
            return np.nan
    elif method == "sub":
        return H
    else:
        raise ValueError


def compute_wrapper(n:int, b:int, method:str, M:float, M_ALT:float|None = None, B:float = 1.0, t:float = 1.0):
    lattice = dl.build_lattice("cantor", n, block_scale=b)
    H = compute_hamiltonian(lattice, method, M, M_ALT, B, t)
    C, eigenvalues, eigenvectors = compute_topological_marker(H)
    return C, eigenvalues, eigenvectors, lattice


def compute_phase(h_method:str, n:int, b:int, M_values:np.ndarray):
    lattice = dl.build_lattice("cantor", n, block_scale=b)
    filename = "./Hexaflake/SSH/Data/" + h_method + f"_n={n}_b={b}.h5"

    if os.path.exists(filename):
        with h5py.File(filename, "r") as f:
            Ms:np.ndarray = f["Ms"][()] # type: ignore
            means:np.ndarray = f["means"][()] # type: ignore
            return Ms, means

    def _worker(M):
        H = compute_hamiltonian(lattice, h_method, M)
        C, _, _ = compute_topological_marker(H)
        y = np.diag(C)[::2] + np.diag(C)[1::2]
        return [M, np.mean(y)]

    with tqdm_joblib(tqdm(total=len(M_values))) as progress_bar:
        data = np.array(Parallel(n_jobs=-2)(delayed(_worker)(m) for m in M_values)).T
    Ms, means = data

    with h5py.File(filename, "w") as f:
        f.create_dataset(name="Ms", data=Ms)
        f.create_dataset(name="means", data=means)
    return Ms, means


def compute_phase_substitution(n:int, b:int, M_values:np.ndarray, M_ALT_values:np.ndarray):
    lattice = dl.build_lattice("cantor", n, block_scale=b)

    filename = "./Hexaflake/SSH/Data/" + f"sub_n={n}_b={b}.h5"
    if os.path.exists(filename):
        with h5py.File(filename, "r") as f:
            Ms:np.ndarray = np.array(f["Ms"][()]) # type: ignore
            means:np.ndarray = np.array(f["means"][()]) # type: ignore
            M_ALTs:np.ndarray = np.array(f["M_ALTs"][()]) # type: ignore
        return Ms, M_ALTs, means.reshape(len(np.unique(M_ALTs)), len(np.unique(Ms))) # type: ignore

    jobs = tuple(product(M_values, M_ALT_values))

    def _worker(M, M_ALT):
        H = compute_hamiltonian(lattice, "sub", M, M_ALT)
        C, _, _ = compute_topological_marker(H)
        y = np.diag(C)[::2] + np.diag(C)[1::2]
        return [M, M_ALT, np.mean(y)]

    with tqdm_joblib(tqdm(total=len(jobs))) as progress_bar:
        data = np.array(Parallel(n_jobs=-2)(delayed(_worker)(*job) for job in jobs)).T

    Ms, M_ALTs, means = data

    with h5py.File(filename, "w") as f:
        f.create_dataset(name="Ms", data=Ms)
        f.create_dataset(name="means", data=means)
        f.create_dataset(name="M_ALTs", data=M_ALTs)

    return Ms, M_ALTs, means.reshape(len(np.unique(M_ALTs)), len(np.unique(Ms)))
    

if __name__ == "__main__":
    C, eigenvalues, eigenvectors, lattice = compute_wrapper(3, 27, "sub", 1.0, M_ALT=1.)
    t = np.arange(C.shape[0])
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(lattice[:, np.newaxis].T, aspect='auto')
    axs[1].plot(t, np.diag(C))
    axs[2].scatter(np.arange(eigenvalues.size), eigenvalues)
    plt.show()

    if 0:
        n = 3
        b = 3
        l = dl.build_lattice("cantor", n, block_scale=b)
        L = l.size

        Ms = np.linspace(-1.0, 5.0, 101)
        #_, renorm_means = compute_phase("renorm", n, b, Ms)
        _, site_elim_means = compute_phase("site_elim", n, b, Ms)
        #plt.scatter(Ms, renorm_means, label='renorm')
        plt.scatter(Ms, site_elim_means, label='site_elim')
        plt.xlabel("M")
        plt.ylabel('mean(C(r))')
        plt.xticks([-1.0, 2.0, 5.0])
        #plt.yticks([0., 1.0])
        plt.legend()
        plt.show()

    if 0:
        _, _, means = compute_phase_substitution(n, b, np.linspace(-1.0, 5.0, 25), np.linspace(-1.0, 5.0, 25))
        plt.imshow(means, origin='lower', extent=(-1.0, 5.0, -1.0, 5.0), vmin=0., vmax=1.)
        plt.xlabel("M")
        plt.ylabel("M_ALT")
        plt.colorbar(label='mean(C(r))')
        plt.title(f"n={n}, L={L}")
        plt.savefig(f"./Hexaflake/SSH/Figures/sub_n={n}_L={L}.png")
        plt.savefig(f"./Hexaflake/SSH/Figures/sub_n={n}_L={L}.svg")

    if 0:
        M, M_ALT = 1.0, 1.0
        n = 3
        lattice = dl.build_lattice("cantor", n, block_scale=9)

        H = compute_hamiltonian(lattice, M, M_ALT)

        C, eigvals, eigvecs = compute_topological_marker(H)

        y = np.diag(C)
        y = y[::2] + y[1::2]
        plt.plot(np.arange(len(y)), y)
        plt.ylim(-0.25, 1.25)
        plt.title(f"SUBSTITUTED : n={n}, L={lattice.size}\nM={M} : M_ALT={M_ALT}")
        plt.show()



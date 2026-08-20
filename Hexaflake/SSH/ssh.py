import numpy as np
from scipy import linalg as spla
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec


def compute_hamiltonian(L:int, dt:float, t:float=1.0):
    H = np.full((L * 2, L * 2), 0.0)

    # Designate even indices as sublattice A and odd indices as sublattice B
    # Open boundary conditions

    # L unit cells, each containing one sublattice A and one sublattice B
    for i in range(L):
        H[2 * i + 0, 2 * i + 1] = t + dt # A_i to B_i

        if i < L - 1:
            H[2 * (i + 1) + 0, 2 * i + 1] = t - dt # A_i+1 to B_i

    H += H.conj().T
    return H


def compute_eigvals(H):
    eigvals, eigvectors = spla.eigh(H, overwrite_a=True)
    sort_idxs = np.argsort(eigvals)
    eigvals = eigvals[sort_idxs]
    eigvectors = eigvectors[sort_idxs, :]
    return eigvals, eigvectors


def compute_topological_marker(hamiltonian):
    # C_1D = N_D W [QxP + PxQ]

    eigenvalues, eigenvectors = compute_eigvals(hamiltonian)

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


def make_figure_layout():
    fig = plt.figure()

    gs = gridspec.GridSpec(1, 2, figure=fig)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1])

    ax1 = fig.add_subplot(gs[0, 0], label='local')
    ax2 = fig.add_subplot(gs1[0, 0], label='nonlocal1')
    ax3 = fig.add_subplot(gs1[1, 0], label='nonlocal2')
    return fig

if __name__ == "__main__":
    L = 50
    Ms = [-0.5, -0.2, 0.2, 0.5]
    colors = ['r', 'g', 'b', 'orange']

    fig, axs = plt.subplots(1, 1)
    axs = [axs]

    for m, c in zip(Ms, colors):
        C, eigvals, eigvectors = compute_topological_marker(L, m, 1.)
        t = np.arange(L)
        y = np.diag(C)
        y = y[::2] + y[1::2]
        axs[0].plot(t, y, label=f"$\\delta t = {m:+.1f}$", c=c)

    axs[0].legend()
    axs[0].set_ylim(-0.5, 1.1)
    axs[0].set_xticks(np.arange(0, L+1, 5))
    axs[0].set_ylabel("$C(\\mathbf {r})$", fontsize=16)
    axs[0].set_xlabel("$\\mathbf {r}$", fontsize=16)

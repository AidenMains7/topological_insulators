import numpy as np
import scipy.sparse as sps
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt


def wannier_matrices(L_x, L_y, x_pbc, y_pbc):
    system_size = L_x*L_y
    lattice = np.arange(system_size).reshape((L_y, L_x))

    cx = sps.dok_matrix((system_size, system_size), dtype=np.complex128)
    sx = sps.dok_matrix((system_size, system_size), dtype=np.complex128)
    cy = sps.dok_matrix((system_size, system_size), dtype=np.complex128)
    sy = sps.dok_matrix((system_size, system_size), dtype=np.complex128)

    for y in range(L_y):
        for x in range(L_x):
            i = lattice[y, x]

            x_pos = (x + 1) % L_x
            j_x = lattice[y, x_pos]
            x_hop = x_pbc or x_pos != 0

            y_pos = (y + 1) % L_y
            j_y = lattice[y_pos, x]
            y_hop = y_pbc or y_pos != 0

            if x_hop:
                cx[i, j_x] = 1/2
                sx[i, j_x] = 1j/2

            if y_hop:
                cy[i, j_y] = 1/2
                sy[i, j_y] = 1j/2

    cx += cx.conj().T
    sx += sx.conj().T
    cy += cy.conj().T
    sy += sy.conj().T

    return cx.toarray(), sx.toarray(), cy.toarray(), sy.toarray()


def main():
    mats = wannier_matrices(12, 15, False, False)
    fig, axs = plt.subplots(1, 4)

    for i in range(4):
        vals = eigvalsh(mats[i], overwrite_a=True)
        axs[i].scatter(np.arange(vals.size), vals)

    plt.show()


if __name__ == '__main__':
    main()



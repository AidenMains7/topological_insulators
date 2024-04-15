import numpy as np
import scipy.sparse as sps
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt


def wannier_matrices(L_x, x_pbc):
    """
    Function for constructing Wannier representation matrices for cosine and sine hopping in a 1D crystal lattice.

    :param L_x: Number of sites in the chain
    :param x_pbc: Set 'True' for periodic boundary conditions, 'False' for open boundary conditions
    :return: Cosine and sine matrices cx, sx
    """

    # constructing the lattice from the system dimensions
    # it is trivial in this case since it's a 1D chain
    system_size = L_x
    lattice = np.arange(system_size)

    # matrices are initialized as scipy sparse dok matrices
    # this prevents excessive memory from building up during the loop when system size is very large
    cx = sps.dok_matrix((system_size, system_size), dtype=np.complex128)
    sx = sps.dok_matrix((system_size, system_size), dtype=np.complex128)

    for x in range(L_x):
        i = lattice[x]  # site index in terms of spatial coordinates (trivial in 1D)

        x_pos = (x + 1) % L_x  # coordinate of the nearest neighbor in the positive x direction
        j_x = lattice[x_pos]  # index of the nearest neighbor in the positive x direction
        x_hop = x_pbc or x_pos != 0  # condition for if hopping is allowed in the positive x direction

        if x_hop:
            # only positive direction hopping is accounted for in the loop
            cx[i, j_x] = 1/2
            sx[i, j_x] = 1j/2

    # negative direction hopping is automatically taken care of by adding
    # the Hermitian adjoint of each matrix to itself
    cx += cx.conj().T
    sx += sx.conj().T

    return cx.toarray(), sx.toarray()  # matrices are converted to numpy arrays before returning


# for computationally demanding tasks, data should be computed and saved first before later being imported and plotted
# but small systems such as this one can be computed quickly on the fly then immediately plotted
def compute_and_plot_data(L_x):
    cx_obc, sx_obc = wannier_matrices(L_x, x_pbc=False)
    cx_pbc, sx_pbc = wannier_matrices(L_x, x_pbc=True)

    # 'eigvalsh' from scipy.linalg is used instead of numpy.linalg because it allows the parameter 'overwrite_a'
    # this can be set to 'True' which allows the code to overwrite the matrix when doing computations
    # this saves memory and reduces computation time, but should only be used if we no longer
    # need the matrix after we compute the eigenvalues
    cx_obc_eigvals = eigvalsh(cx_obc, overwrite_a=True)
    sx_obc_eigvals = eigvalsh(sx_obc, overwrite_a=True)
    cx_pbc_eigvals = eigvalsh(cx_pbc, overwrite_a=True)
    sx_pbc_eigvals = eigvalsh(sx_pbc, overwrite_a=True)

    indices = np.arange(cx_obc_eigvals.size)

    # initializing a 2 by 2 grid of subplots
    fig, axs = plt.subplots(2, 2)

    # a 'raw string' is made using 'r' to render Latex code when using Matplotlib
    axs[0, 0].scatter(indices, cx_obc_eigvals)
    axs[0, 0].set_title(r'$C_x$ OBC')

    axs[0, 1].scatter(indices, sx_obc_eigvals)
    axs[0, 1].set_title(r'$S_x$ OBC')

    axs[1, 0].scatter(indices, cx_pbc_eigvals)
    axs[1, 0].set_title(r'$C_x$ PBC')

    axs[1, 1].scatter(indices, sx_pbc_eigvals)
    axs[1, 1].set_title(r'$S_x$ PBC')

    # adding x and y axes labels to each subplot
    for row in axs:
        for ax in row:
            ax.set_ylabel(r'$E_n$', rotation=0)
            ax.set_xlabel('n')

    plt.tight_layout()

    plt.show()


# main function is typically used as a wrapper/ used to call the custom functions defined above
def main():
    compute_and_plot_data(100)


# the construct below ensures that the 'main' function is only executed if this script is directly run
# if this script is instead being called by another module, 'main' will not be executed
if __name__ == '__main__':
    main()



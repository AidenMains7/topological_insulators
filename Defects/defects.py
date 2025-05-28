import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from scipy.linalg import eigh, eigvals
from scipy.sparse import dok_matrix



def compute_square_lattice(side_length:int):
    return np.arange(side_length**2).reshape(side_length, side_length)


def generate_vacancy_lattice(side_length:int):
    if side_length % 2 == 0:
        raise ValueError("Side length must be odd for a single vacancy in the center.")
    lattice = compute_square_lattice(side_length)
    lattice[side_length//2, side_length//2] = -1
    return lattice


def compute_interstitial_lattice(side_length:int):
    if side_length % 2 == 1:
        raise ValueError("Side length must be even for a single interstitial in the center.")
    lattice = compute_square_lattice(side_length)
    y_pos, x_pos = np.where(lattice >= 0)
    x_mean = np.mean(x_pos)
    y_mean = np.mean(y_pos)
    coordinates = np.array([x_pos, y_pos])
    coordinates = (np.concatenate((coordinates, np.array([[x_mean], [y_mean]])), axis=1) * 2).astype(int)

    new_lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
    new_lattice[coordinates[1], coordinates[0]] = np.arange(len(coordinates[0]))

    return new_lattice


def compute_frenkel_pair_lattice(side_length:int, displacement_index:int):
    if side_length % 2 == 0:
        raise ValueError("Side length must be odd for a single vacancy in the center.")
    if displacement_index < 0 or displacement_index > 7:
        raise ValueError("Displacement index must be between 0 and 7.")
    
    lattice = compute_square_lattice(side_length)
    center = np.array([side_length//2, side_length//2]).reshape(2, 1)
    lattice[center[1], center[0]] = -1
    
    y_pos, x_pos = np.where(lattice >= 0)
    coordinates = (np.array([x_pos, y_pos]) * 2).astype(int)


    values = [-3, -1, 1, 3]
    displacements = np.array(list(product(values, repeat=2)))
    good_displacements = []
    for d in displacements:
        if np.abs(d[0]) == np.abs(d[1]):
            pass
        else:
            good_displacements.append(d.reshape(2,1))

    displacements = np.array(good_displacements)

    coordinates = np.concatenate((coordinates, center * 2 + displacements[displacement_index]), axis=1)

    new_lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
    new_lattice[coordinates[1], coordinates[0]] = np.arange(len(coordinates[0]))
    return new_lattice


def compute_distances(lattice, pbc):
    y, x = np.where(lattice >= 0)[:]

    side_length = lattice.shape[0]
    
    dx = x - x[:, None]
    dy = y - y[:, None]
    if pbc:
        multipliers = tuple(product([-1, 0, 1], repeat=2))
        shifts = [(i * side_length, j * side_length) for i, j in multipliers]

        x_shifted = np.empty((dx.shape[0], dx.shape[1], len(shifts)), dtype=dx.dtype)
        y_shifted = np.empty((dy.shape[0], dy.shape[1], len(shifts)), dtype=dy.dtype)
        for i, (dx_shift, dy_shift) in enumerate(shifts):
            x_shifted[:, :, i] = dx + dx_shift
            y_shifted[:, :, i] = dy + dy_shift

        distances = x_shifted**2 + y_shifted**2
        minimal_hop = np.argmin(distances, axis = -1)
        i_idxs, j_idxs = np.indices(minimal_hop.shape)

        dx = x_shifted[i_idxs, j_idxs, minimal_hop]
        dy = y_shifted[i_idxs, j_idxs, minimal_hop]

    return dx, dy

def compute_wannier_matrices(lattice, pbc):
    dx, dy = compute_distances(lattice, pbc)
    xp_mask = (dx == 1) & (dy == 0)
    yp_mask = (dx == 0) & (dy == 1)

    xpyp_mask = (dx == 1) & (dy == 1)
    xnyp_mask = (dx == -1) & (dy == 1)

    Cx =   dok_matrix((side_length**2, side_length**2), dtype=complex)
    Sx =   dok_matrix((side_length**2, side_length**2), dtype=complex)
    Cy =   dok_matrix((side_length**2, side_length**2), dtype=complex)
    Sy =   dok_matrix((side_length**2, side_length**2), dtype=complex)
    CxCy = dok_matrix((side_length**2, side_length**2), dtype=complex)
    CySx = dok_matrix((side_length**2, side_length**2), dtype=complex)
    CxSy = dok_matrix((side_length**2, side_length**2), dtype=complex)
    I =    np.eye(side_length**2, dtype=complex)

    Sx[xp_mask] = 1j / 2
    Cx[xp_mask] = 1 / 2
    Cy[yp_mask] = 1 / 2
    Sy[yp_mask] = 1j / 2

    CxCy[xpyp_mask] = 1 / 4
    CxCy[xnyp_mask] = 1 / 4

    CxSy[xpyp_mask] = 1j / 4
    CxSy[xnyp_mask] = 1j / 4

    CySx[xpyp_mask] = 1j / 4
    CySx[xnyp_mask] = -1j / 4

    wannier_dict = {
        'Cx_plus_Cy': Cx + Cy,
        'Sx': Sx,
        'Sy': Sy,
    }

    wannier_dict = {k: (v + v.conj().T).toarray() for k, v in wannier_dict.items()}
    wannier_dict["I"] = I
    return wannier_dict


def compute_wannier_polar(defect_type, lattice, pbc):
    dx, dy = compute_distances(lattice, pbc)

    if defect_type in ["interstitial", "frenkel_pair"]:
        dx = dx.astype(float) / 2
        dy = dy.astype(float) / 2

    theta = np.arctan2(dy, dx)

    mask_dist = np.maximum(np.abs(dx), np.abs(dy)) <= 1

    principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & (mask_dist)
    diagonal_mask  = ((np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0))) & (mask_dist)
    all_mask = principal_mask | diagonal_mask


    d_r = np.where(all_mask, np.sqrt(dx**2 + dy**2), 0.0 + 0.0j)
    F_p = np.where(principal_mask, np.exp(1  - d_r), 0. + 0.j)
    d_cos = np.where(all_mask, np.cos(theta), 0. + 0.j)
    d_sin = np.where(all_mask, np.sin(theta), 0. + 0.j)

    Cx_plus_Cy = F_p / 2
    Sx = 1j * d_cos * F_p / 2
    Sy = 1j * d_sin * F_p / 2

    wannier_dict = {
        "Cx_plus_Cy": Cx_plus_Cy,
        "Sx": Sx,
        "Sy": Sy,
        "I": np.eye(Sx.shape[0], dtype=complex),
    }

    return wannier_dict
    


def compute_hamiltonian(defect_type, wannier_matrices, M_background, t0=1., t1=1., M_substitution = None, substitution_index = None):

    pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)

    Cx_plus_Cy, Sx, Sy, I = wannier_matrices.values()

    if defect_type in ["vacancy", "square"]:
        onsite_mass = M_background * I
    elif defect_type in ["substitution", "interstitial", "frenkel_pair"]:
        onsite_mass = M_background * I
        onsite_mass[substitution_index, substitution_index] = M_substitution

    d1 = t1 * Sx
    d2 = t1 * Sy
    d3 = onsite_mass + t0 * (Cx_plus_Cy)

    hamiltonian = np.kron(d1, pauli1) + np.kron(d2, pauli2) + np.kron(d3, pauli3)
    return hamiltonian


def compute_projector(hamiltonian):
    eigenvalues, eigenvectors = eigh(hamiltonian, overwrite_a=True)
    lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2]
    highest_lower_band = lower_band[-1]

    D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
    D_dagger = np.einsum('i,ij->ij', D, eigenvectors.conj().T)

    projector = eigenvectors @ D_dagger
    return projector


def compute_bott_index(projector, lattice):
    Y, X = np.where(lattice >= 0)[:]

    X = np.repeat(X, 2)
    Y = np.repeat(Y, 2)

    Ux = np.exp(1j * 2 * np.pi * X / lattice.shape[0])
    Uy = np.exp(1j * 2 * np.pi * Y / lattice.shape[1])

    UxP = np.einsum('i,ij->ij', Ux, projector)
    UyP = np.einsum('i,ij->ij', Uy, projector)
    Ux_daggerP = np.einsum('i,ij->ij', Ux.conj(), projector)
    Uy_daggerP = np.einsum('i,ij->ij', Uy.conj(), projector)

    A = np.eye(projector.shape[0], dtype=np.complex128) - projector + projector @ UxP @ UyP @ Ux_daggerP @ Uy_daggerP
    bott_index = round(np.imag(np.sum(np.log(eigvals(A)))) / (2 * np.pi))

    return bott_index


def compute_LDOS(lattice, hamiltonian):
    number_of_states = 2
    y, x = np.where(lattice >= 0)[:]
    x, y = np.repeat(x, 2), np.repeat(y , 2) 

    eigenvalues, eigenvectors = eigh(hamiltonian, overwrite_a=True)
    eigenvalue_idxs = np.arange(eigenvalues.size)
    all_positive_idxs = eigenvalues > 0

    positive_lowest =  eigenvalue_idxs[all_positive_idxs][np.argsort(eigenvalues[all_positive_idxs])][:number_of_states // 2]
    negative_highest = eigenvalue_idxs[~all_positive_idxs][np.argsort(eigenvalues[~all_positive_idxs])[::-1]][:number_of_states // 2]

    LDOS_idxs = np.concatenate((negative_highest, positive_lowest))
    other_idxs = np.delete(eigenvalue_idxs, LDOS_idxs)

    LDOS = np.sum(np.abs(eigenvectors[:, LDOS_idxs])**2, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))   
    ax.scatter(x, y, c=LDOS, cmap='viridis', s=10)
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
    plt.show()



def temp(defect_type:str, side_length:int):
    if defect_type == "square":
        lattice = compute_square_lattice(side_length)
        wannier = compute_wannier_polar(defect_type, lattice, True)
        substitution_index = 0
    elif defect_type == "vacancy":
        lattice = compute_vacancy_lattice(side_length)
        wannier = compute_wannier_polar(defect_type, lattice, True)
        substitution_index = 0
    if defect_type == "substitution":
        lattice = compute_square_lattice(side_length)
        wannier = compute_wannier_polar(defect_type, lattice, True)
        substitution_index = side_length // 2
    elif defect_type == "interstitial":
        lattice = compute_interstitial_lattice(side_length)
        wannier = compute_wannier_polar(defect_type, lattice, True)
        substitution_index = np.max(lattice)
    elif defect_type == "frenkel_pair":
        lattice = compute_frenkel_pair_lattice(side_length, 0)
        wannier = compute_wannier_polar(defect_type, lattice, True)
        substitution_index = np.max(lattice)

    fig, axs = plt.subplots(2,2, figsize=(10, 10))
    


    for ax, M_sub in zip(axs.flatten(), [-2.5, -1., 1., 2.5]):
        data = []
        for M_background in np.arange(-4., 4.25, 0.25):
            hamiltonian = compute_hamiltonian(defect_type, wannier, M_background, 1., 1., M_sub, substitution_index)
            projector = compute_projector(hamiltonian)
            bott = compute_bott_index(projector, lattice)
            data.append((M_background, bott))
            compute_LDOS(lattice, hamiltonian)
        data = np.array(data).T
        ax.scatter(data[0], data[1])
        ax.set_xlabel("M")
        ax.set_ylabel("Bott Index")
        ax.set_title(f"M_substitution = {M_sub}")
    fig.suptitle(f"{defect_type.capitalize()} Lattice Bott Index")
    plt.show()

if __name__ == "__main__":

    if True:
        side_length = 15
        lattice = compute_square_lattice(side_length)
        wannier = compute_wannier_polar("square", lattice, True)
        data = []
        for M in np.linspace(-4., 4., 51):
    
            hamiltonian = compute_hamiltonian("square", wannier, M, 1., 1.)
            projector = compute_projector(hamiltonian)
            bott = compute_bott_index(projector, lattice)
            data.append((M, bott))
        
        data = np.array(data).T
        plt.scatter(data[0], data[1])
        for x in [-4., -2., 0., 2., 4.]:
            plt.axvline(x=x, color='k', linestyle='--', alpha=0.5)
        plt.xlabel("M")
        plt.ylabel("Bott Index")
        plt.title("Square Lattice Bott Index from Tight Binding Model")
        plt.show()

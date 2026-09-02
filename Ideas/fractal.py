import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt
from fractions import Fraction


def make_cantor_triangle(n:int, scale:float=2.):
    if n == 0:
        return np.array([[0, 0]]).reshape(2, -1)

    smaller = make_cantor_triangle(n-1, scale) / scale
    angles = [np.pi * i for i in [1/2, 7/6, 11/6]]
    displacements = [(np.cos(a), np.sin(a)) for a in angles]
    displacements = np.array(displacements) / np.sqrt(3)

    top = smaller.copy()
    left = smaller.copy()
    right = smaller.copy()

    top[0] -= np.mean(top[0])
    top[1] -= np.max(top[1])
    left[0] -= np.min(left[0])
    left[1] -= np.min(left[1])
    right[0] -= np.max(right[0])
    right[1] -= np.min(right[1])

    fracs = [top, left, right]
    this = np.concatenate([arr + np.array(d).reshape((2, -1)) for d, arr in zip(displacements, fracs)], axis=1)
    this = np.unique(np.round(this, 8), axis=1)
    return this


def find_gcd_decmials(values):
    fracs = [Fraction(str(v)).limit_denominator() for v in values]
    numerators = [f.numerator for f in fracs]
    denominators = [f.denominator for f in fracs]
    common_denominator = np.lcm.reduce(denominators)
    numerators = [n * common_denominator // d for n, d in zip(numerators, denominators)]
    gcd_numerator = np.gcd.reduce(numerators)
    assert gcd_numerator > 0
    assert common_denominator > 0
    return gcd_numerator, common_denominator


def make_lattices(n:int, scale:float):
    # Generate a triangular lattice of points that fills the bounding box of the Cantor triangle.
    # The lattice is built on the same skewed basis as the self-similar triangle construction.

    fractal = make_cantor_triangle(n, scale)

    x_min = fractal[0].min()
    x_max = fractal[0].max()
    y_min = fractal[1].min()
    y_max = fractal[1].max()

    bottom_row = fractal[:, fractal[1] == y_min]
    spacings = np.diff(bottom_row[0])

    if np.isclose(scale - int(scale), 0.): 
        a = (1.0 / scale) ** (n - 1) 
    else: 
        numerator, denominator = find_gcd_decmials(spacings)
        a = numerator / denominator

    a1 = [1.0 * a, 0.0]
    a2 = [0.5 * a, np.sqrt(3)/2 * a]


    n_side = np.round(1/a + 1, 0).astype(int)

    points = []
    for j in range(int(n_side)):
        for i in range(n_side - j):
            point = i * np.array(a1) + j * np.array(a2) + np.array([x_min, y_min])
            points.append(point)

    data = {
        "triangular_coordinates": (np.array(points).T - np.array([[x_min], [y_min]])) / spacings[0],
        "fractal_coordinates": (fractal - np.array([[x_min], [y_min]])) / spacings[0],
        "fractal_dimension": np.log(3)/np.log(scale),
    }
    return data


def compute_connection(coordinates, max_distance=np.inf):
    X = coordinates[0]
    Y = coordinates[1]

    dx = X[:, np.newaxis] - X[np.newaxis, :]
    dy = Y[:, np.newaxis] - Y[np.newaxis, :]
    dr = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx) + np.pi) % (2 * np.pi)

    b1_mask = np.isclose(theta, 0.)
    np.fill_diagonal(b1_mask, False)
    b2_mask = np.isclose(theta, np.pi / 3)
    b3_mask = np.isclose(theta, 2 * np.pi / 3)

    c1_mask = np.isclose(theta, np.pi / 2)
    c2_mask = np.isclose(theta, np.pi / 6)
    c3_mask = np.isclose(theta, 5 * np.pi / 6)
    
    distance_mask = dr <= max_distance   

    masks = [b1_mask, b2_mask, b3_mask, c1_mask, c2_mask, c3_mask]
    return [m & distance_mask for m in masks], dr


def compute_hamiltonian(hopping_masks, dr, a, M, B, B_tilde, t1, t2):
    b1, b2, b3, c1, c2, c3 = [m.astype(np.complex128) * np.exp(1 - dr / a) for m in hopping_masks]

    I = np.eye(b1.shape[0], dtype=np.complex128)
    amplitude = np.exp(1 - dr / a)
    
    d1 = (t1 / 2j) * b1 + (t1 / 4j) * (b2 + b3)
    d1 += d1.conj().T

    d2 = (-t1 * np.sqrt(3) / 4j) * (b2 - b3)
    d2 += d2.conj().T

    d3 = B * (b1 + b2 + b3)
    d3 += d3.conj().T
    d3 += (M - 4 * B) * I

    dtilde1 = (np.sqrt(3) * t2 / 8j) * (c2 + c3)
    dtilde1 += dtilde1.conj().T

    dtilde2 = (-t2 / 2j) * (c1) + (t2 / 4j) * (c2 - c3)
    dtilde2 += dtilde2.conj().T

    dtilde3 = (B_tilde) * (c1 + c2 + c3)
    dtilde3 += dtilde3.conj().T
    dtilde3 += (-6 * B_tilde) * I

    pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)

    H_d = np.kron(d1, pauli1) + np.kron(d2, pauli2) + np.kron(d3, pauli3)
    H_tilde = np.kron(dtilde1, pauli1) + np.kron(dtilde2, pauli2) + np.kron(dtilde3, pauli3)
    return H_d + H_tilde


def compute_ldos(eigenvalues, eigenvectors, n_idxs = 2):
    eigenvalues = np.sort(eigenvalues)
    negative_idxs = np.argwhere(eigenvalues < 0).flatten()
    positive_idxs = np.argwhere(eigenvalues > 0).flatten()

    idxs = np.concatenate((negative_idxs[-n_idxs // 2:], positive_idxs[:n_idxs // 2]))
    ldos = np.sum(np.abs(eigenvectors[:, idxs])**2, axis=1)
    return ldos[::2] + ldos[1::2]



if __name__ == "__main__":

    data = make_lattices(n=4, scale=2)
    x1, y1 = data["triangular_coordinates"]
    x2, y2 = data["fractal_coordinates"]


    m = 0.
    H = compute_hamiltonian(*compute_connection(data["triangular_coordinates"], 1.), a=1., M=m, B=1.0, B_tilde=0.0, t1=1, t2=0.0)

    eigenvalues, eigenvectors = spla.eigh(H)

    ldos = compute_ldos(eigenvalues, eigenvectors, n_idxs=2)

    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(np.arange(len(eigenvalues)), eigenvalues, s=30, alpha=0.5)
    scat = axs[1].scatter(x1, y1, c=ldos, s=30, cmap="plasma", alpha=1.)
    plt.colorbar(scat, ax=axs[1])
    plt.show()


    if 0:
        t = np.linspace(2., 10., 1001)
        t = np.concatenate((t, [np.pi]))
        values = [make_lattices(n=2, scale=s) for s in t]
        ns = np.array([v[0] for v in values])
        ds = np.array([v[1] for v in values])
        fs = ns / ds
        #plt.scatter(t, ns, label="numerator")

        fig, axs = plt.subplots(1, 2)
        axs[0].scatter(t, ds, label="denominator", alpha=0.5)
        axs[1].scatter(t, 1/fs, label="n_in_row", alpha=0.5)
        for ax in axs:
            ax.set_yscale('log')
            ax.legend()
        plt.show()

    if 0:
        n = 2; s = 3.1

        x, y = make_cantor_triangle(n, scale=s)
        x2, y2 = make_parent_triangular_lattice(n, scale=s)
        #plt.scatter(lattice[0], lattice[1], color='gray', alpha=0.5, s=50)
        plt.scatter(x, y, s=30, color='red', alpha=0.5, zorder=1)
        plt.scatter(x2, y2, s=30, color='b', alpha=0.5, zorder=0)
        plt.show()
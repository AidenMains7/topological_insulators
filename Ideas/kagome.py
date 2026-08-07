import numpy as np
from matplotlib import pyplot as plt




def construct_kagome_lattice(nx, ny):
    triangle_angles = [0., np.pi, 3 * np.pi / 2]
    triangle_pos = np.array([[np.cos(a), np.sin(a)] for a in triangle_angles]).T
    t1 = triangle_pos - np.array([np.min(triangle_pos[0]), np.max(triangle_pos[1])])[:, np.newaxis]
    t2 = -t1

    unit_cell_pos = np.concatenate((t1, t2), axis=1)
    unit_cell_pos = np.unique(unit_cell_pos, axis=1)


    tile_v1 = np.array([np.max(unit_cell_pos[0] - np.min(unit_cell_pos[0])), 0])[:, np.newaxis]
    tile_v2 = np.array([2., 2.])[:, np.newaxis]

    print(tile_v1.shape)

    lattice_positions = []
    for i in range(nx):
        for j in range(ny):
            lattice_positions.append(unit_cell_pos + i * tile_v1 + j * tile_v2)


    lattice_positions = np.concatenate(lattice_positions, axis=1)
    lattice_positions = np.unique(lattice_positions, axis=1)

    lattice_positions[0] -= np.min(lattice_positions[0])
    lattice_positions[1] -= np.min(lattice_positions[1])
    
    lattice_positions = np.round(lattice_positions).astype(int)

    X = lattice_positions[0]
    Y = lattice_positions[1]

    dx = X[:, np.newaxis] - X[np.newaxis, :]
    dy = Y[:, np.newaxis] - Y[np.newaxis, :]

    NN = ((np.abs(dx) == 2) & (np.abs(dy) == 0)) | ((np.abs(dx) == 1) & (np.abs(dy) == 1)) 



    plt.scatter(lattice_positions[0], lattice_positions[1])

    print(np.indices(NN.shape).shape)

    for i in range(NN.shape[0]):
        for j in range(NN.shape[1]):
            if NN[i, j] == True:
                plt.plot([X[i], X[j]], [Y[i], Y[j]], c='k', zorder=-10)
    plt.axis('equal')
    plt.show()



def compute_kagome_hamiltonian(k:np.ndarray, t:float):
    def _r(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    a1 = np.array([1, 1 / np.sqrt(3)]).flatten()
    a2 = (_r(np.pi / 3) @ a1).flatten()

    l = 0.3 * t
    t_tilde = t * (1. + 1.j * l)

    k = k.T
    arr1 = np.dot(k, a1)
    arr2 = np.dot(k, a2)
    arr3 = np.dot(k, (a1 + a2))

    zero_arr = np.zeros(len(k))
    h1 = np.array([
        [zero_arr, t_tilde * np.cos(arr2 / 2), np.conj(t_tilde) * np.cos(arr1 / 2)],
        [zero_arr, zero_arr,                   t_tilde * np.cos(arr3 / 2)],
        [zero_arr, zero_arr,                   zero_arr]
    ])

    h1 = np.rollaxis(h1, -1, 0)
    h_diag_part = t * l / (3 * np.sqrt(3)) * (np.cos(arr1) + np.cos(arr2) + np.cos(arr3))
    hamiltonian = h1 + np.swapaxes(h1.conj(), -1, -2) - h_diag_part[:, np.newaxis, np.newaxis] * np.eye(3, dtype=np.complex128)[np.newaxis, ...]
    return hamiltonian


def compute_energies(k, t):
    h = compute_kagome_hamiltonian(k, t)
    eigenvalues = np.linalg.eigvalsh(h)
    return eigenvalues

def plot_kagome_bands():
    # The FBZ of the kagome lattice is the hexagon centered around the origin with a vertex at (4pi/3, 0)

    gamma = [0., 0.]
    K = [4 * np.pi / 3, 0.]
    M = [np.pi, np.pi / np.sqrt(3)]

    path_points = np.array([K, gamma, M])
    labels = np.array(['$K$', '$\\Gamma$', '$M$'])

    res = 101

    paths = []
    for i in range(path_points.shape[0]):
        x1 = path_points[i, 0]
        y1 = path_points[i, 1]
        x2 = path_points[(i + 1) % len(path_points), 0]
        y2 = path_points[(i + 1) % len(path_points), 1]
        paths.append([np.linspace(x1, x2, res, endpoint=False), np.linspace(y1, y2, res, endpoint=False)])

    momentums = np.concatenate(paths, axis=1)

    energies = compute_energies(momentums, 1.)


    fig, ax = plt.subplots(1, 1)
    for i in range(energies.shape[1]):
        ax.plot(np.arange(len(energies)), energies[:, i])
        ax.axvline((i + 1) * res, ls='--', c='k', zorder=-10, alpha=0.5, lw=1)
    
    ax.set_xticks([(i) * res for i in range(len(path_points))])
    ax.set_xticklabels(labels)

    plt.show()



def main():
    construct_kagome_lattice(4, 4)
    #plot_kagome_bands()


if __name__ == "__main__":
    main()
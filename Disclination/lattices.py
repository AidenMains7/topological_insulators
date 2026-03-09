import numpy as np
from matplotlib import pyplot as plt


def generate_dodecahedron_lattice():
    angles = np.arange(12) * (np.pi / 6) + np.pi / 12

    x_unit = np.cos(angles)
    y_unit = np.sin(angles)

    r = ((x_unit[1] - x_unit[0])**2 + (y_unit[1] - y_unit[0])**2) ** (-1/2)
    
    x_unit *= r
    y_unit *= r
    
    width = np.max(x_unit) - np.min(x_unit)

    v1 = (width + 1) * np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])
    v2 = np.array([width + 1, 0])

    xs = []
    ys = []
    for m1 in range(10):
        for m2 in range(10):
            xs.append(x_unit + m1 * v1[0] + m2 * v2[0])
            ys.append(y_unit + m1 * v1[1] + m2 * v2[1])

        

    X = np.array(xs).flatten()
    Y = np.array(ys).flatten()

    
    sort_idxs = np.argsort(Y)
    X = X[sort_idxs]
    Y = Y[sort_idxs]

    dx = X[:, np.newaxis] - X[np.newaxis, :]
    dy = Y[:, np.newaxis] - Y[np.newaxis, :]
    dr = np.sqrt(dx ** 2 + dy ** 2)
    distance_mask = (dr <= 1 + 1e-6) & (dr > 0)

    plt.scatter(X, Y, zorder=1, c='k')

    i_idx, j_idx = np.where(distance_mask)
    plt.plot([X[i_idx], X[j_idx]], [Y[i_idx], Y[j_idx]], c='k', ls='-', zorder=0)


    plt.axis('equal')
    plt.show()
    








if __name__ == "__main__":
    generate_dodecahedron_lattice()
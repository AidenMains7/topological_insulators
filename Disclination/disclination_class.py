import numpy as np
from matplotlib import pyplot as plt

def generate_square_lattice(side_length:int):
    return np.arange(side_length**2).reshape((side_length, side_length))

def get_coordinates(lattice):
    Y, X = np.where(lattice >= 0)
    X = X - np.mean(X)
    Y = Y - np.mean(Y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X) % (2 * np.pi)
    return r, theta, X, Y

def cut_angle(lattice, min_angle, max_angle):
    r, theta, X, Y = get_coordinates(lattice)
    mask = (theta > min_angle) & (theta < max_angle)
    return r[~mask], theta[~mask], X[~mask], Y[~mask]

def mend_disclination(lattice, min_angle, max_angle):
    r, theta, X, Y = cut_angle(lattice, min_angle, max_angle)
    theta -= max_angle
    theta = theta % (2 * np.pi)  # Normalize angles to [0, 2π]
    theta = theta / np.max(theta) * 2 * np.pi  # Normalize to [0, 2π]
    return r, theta, X, Y

if __name__ == "__main__":
    lattice = generate_square_lattice(15)
    alpha, beta = 0., np.pi/2
    r, theta, X, Y = mend_disclination(lattice, alpha, beta)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    sc = ax.scatter(theta, r, c=theta, cmap='Spectral', s=100, alpha=0.75)
    plt.colorbar(sc, label='Angle (radians)')
    plt.show()



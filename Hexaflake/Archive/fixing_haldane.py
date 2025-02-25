import numpy as np
from scipy.linalg import eigvals, eigvalsh, eigh, eig
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from itertools import product

import sys
sys.path.append(".")
from Carpet.plotting import plot_imshow


def generate_coordinates(generation:int, hexagon_lattice_size:"int|None"=None):
    if generation < 0:
        raise ValueError("Fractal generation must be >= 0.")

    # Create the unit cell
    unit_cell = np.array([[np.cos(a), np.sin(a)] for a in [np.pi*(i)/3 for i in [1, 2, 3, 4, 5, 6]]]).T

    def fractal_iteration(_gen:int):
        if _gen == 0:
            return unit_cell
        else:
            smaller = fractal_iteration(_gen-1)
            points = (3**_gen)*np.append(unit_cell, np.array([[0], [0]]), axis=1)
            new = np.empty((2, 1))
            for i in range(7):
                new = np.append(new, points[:, i][:, np.newaxis]+smaller, axis=1)
            new = new[:, 1:]
            return np.unique(new, axis=1)
    
    def honeycomb_lattice(side_length: int) -> np.ndarray:
        """
        Generate a 2D honeycomb lattice with the specified side length.

        Parameters:
        side_length (int): Number of hexagon tiles on each side.

        Returns:
        np.ndarray: A 2xN array representing the coordinates of the lattice.
        """

        def _row_of_n_hexagons(n: int) -> np.ndarray:
            """Generate a row of `n` hexagons."""
            row = unit_cell.copy()
            for i in range(1, n):
                row = np.append(row, unit_cell + np.array([[3 * i], [0]]), axis=1)
            row[0] -= np.mean(row[0])
            row[1] -= np.mean(row[1])
            return row

        # Initialize the lattice
        hexagon_lattice = np.empty((2, 0))
        offset = np.sqrt(3) * 3 / 2

        # Generate the upper half of the honeycomb
        for i, num_hexagons in enumerate(range(side_length, 2 * side_length)):
            row = _row_of_n_hexagons(num_hexagons)
            row += np.array([[0], [i * offset]])
            hexagon_lattice = np.append(hexagon_lattice, row, axis=1)

        # Reflect the upper half to generate the lower half
        lower_half = np.array([[0], [2 * np.max(hexagon_lattice[1]) - np.sqrt(3)]]) - hexagon_lattice
        hexagon_lattice = np.append(hexagon_lattice, lower_half, axis=1)

        # Center the lattice around the origin
        hexagon_lattice[0] -= np.mean(hexagon_lattice[0])
        hexagon_lattice[1] -= np.mean(hexagon_lattice[1])

        # Ensure unique points
        hexagon_lattice = np.unique(hexagon_lattice, axis=1)
        return hexagon_lattice

    # Return final points values
    hexaflake = fractal_iteration(generation)
    if hexagon_lattice_size is None or hexagon_lattice_size < 1:
        return (hexaflake, honeycomb_lattice((3**generation - 1)//2 + 1))
    else:
        return (hexaflake, honeycomb_lattice(hexagon_lattice_size))

# SMALL FUNCTINOS
def plot_hexaflake_honeycomb(gen, sl=None):
    hexaflake, honeycomb = generate_coordinates(gen, sl)
    plt.scatter(honeycomb[0], honeycomb[1], alpha=0.5, c='purple')
    plt.scatter(hexaflake[0], hexaflake[1], alpha=0.5, c='orange')
    plt.show()

def scale_to_integer(coordinates:np.ndarray):
    c = coordinates.copy()
    c[0] = c[0]*2.0
    c[1] = c[1]*2.0/np.sqrt(3)
    c[0] -= np.min(c[0])
    c[1] -= np.min(c[1])
    return np.round(c).astype(int)

def scale_to_irrational(coordinates:np.ndarray):
    c = coordinates.copy()
    c[0] = c[0]/2.0
    c[1] = c[1]/(2.0/np.sqrt(3))
    return c
    
def coords_to_lattice(coordinates:np.ndarray):
    """
    Generates a lattice with -1 at empty points and an index at filled points

    Input should be as real-space coordinates, not scaled
    """
    c = scale_to_integer(coordinates)    

    lx, ly = np.max(c[0])+1, np.max(c[1])+1

    lattice = np.full((ly, lx), -1, dtype=int)
    lattice[c[1], c[0]] = np.arange(c.shape[1])
    return lattice

def lattice_to_coords(lattice:np.ndarray):
    coords = np.where(lattice >= 0)
    return np.vstack((coords[1], coords[0]))

def plot_lattice_imshow(generation, sl=None):
    arrs = generate_coordinates(generation, sl)
    lats = [coords_to_lattice(arr) for arr in arrs]

    fig, ax = plt.subplots(1, len(lats))
    for i in range(len(lats)):
        lat = lats[i]
        y, x = lat.shape
        Z = (lat+1).astype(bool).astype(int)
        X, Y = np.arange(x), np.arange(y)
        fig, ax[i], cbar = plot_imshow(fig, ax[i], X, Y, Z)
    plt.show()

# BIG FUNCTION
def sublattices(generation):
    """
    
    """
    hexaflake, honeycomb = generate_coordinates(generation)

    honeycomb_mask = (coords_to_lattice(honeycomb)+1).astype(bool)
    hexaflake_mask = (coords_to_lattice(hexaflake)+1).astype(bool)

    vacancies_mask = honeycomb_mask & (~hexaflake_mask)

    ly, lx = honeycomb_mask.shape

    triangular_A_mask = np.full((ly, lx), False, dtype=bool)
    for y in range(ly):
        row = np.where(honeycomb_mask[y])[0]
        if (y+1)%3 == 1:
            start = 1
        elif (y+1)%3 == 2:
            start = 0
        else:
            start = 1
        sites = row[start::2]
        triangular_A_mask[y, sites] = True
    triangular_B_mask = honeycomb_mask & (~triangular_A_mask)

    n_sites = np.sum(honeycomb_mask)

    discrete_coordinates = np.unique(scale_to_integer(honeycomb), axis=1)
    triangular_A = triangular_A_mask[discrete_coordinates[1], discrete_coordinates[0]]
    triangular_B = triangular_B_mask[discrete_coordinates[1], discrete_coordinates[0]]
    hexaflake = hexaflake_mask[discrete_coordinates[1], discrete_coordinates[0]]
    vacancies = vacancies_mask[discrete_coordinates[1], discrete_coordinates[0]]

    arrays_dict = {
        'triangular_A': triangular_A,
        'triangular_B': triangular_B,
        'hexaflake': hexaflake,
        'vacancies': vacancies,
        'discrete_honeycomb_coordinates': discrete_coordinates
    }
    return arrays_dict

def displacement_array(coordinates:np.ndarray, pbc:bool):
    x, y = coordinates

    # dx[i,j] = x of ith point - x of jth point
    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y


    if pbc:
        x_max, y_max = (np.max(arr) for arr in [x, y])

        n=2
        edge_idxs = {
            'top': np.argwhere(y >= y_max-n+1),
            'bottom': np.argwhere(y <= n-1),
            'bottom_right': np.argwhere(y-x <= -(x_max*3/4)+n+1),
            'bottom_left': np.argwhere(y+x <= y_max/2+n+1),
            'top_right': np.argwhere(y+x >= y_max/2+x_max-n-1),
            'top_left': np.argwhere(y-x >= y_max/2-n-1)
        }
        pbc_shifts = {
            'bottom': (-3, -y_max-1),
            'top': (3, y_max+1),
            'top_left': (-x_max*3/4, y_max/2+2),
            'top_right': (x_max*3/4+3, y_max/2-1),
            'bottom_left': (-x_max*3/4-3, -y_max/2+1),
            'bottom_right': (x_max*3/4, -y_max/2-2)
        }
        x_pbc_border = np.concatenate([x[edge_pts]+shift[0] for edge_pts, shift in zip(edge_idxs.values(), pbc_shifts.values())]).flatten()
        y_pbc_border = np.concatenate([y[edge_pts]+shift[1] for edge_pts, shift in zip(edge_idxs.values(), pbc_shifts.values())]).flatten()
        unique_pbc_border, unique_border_idxs = np.unique(np.vstack((x_pbc_border, y_pbc_border)), return_index=True, axis=1)

        pbc_orig_idxs = np.concatenate([edge_pts for edge_pts in edge_idxs.values()]).flatten()

        if False:
            plt.scatter(x_pbc_border, y_pbc_border, alpha=0.5, color='green')
            plt.scatter(x_pbc_orig, y_pbc_orig, alpha=0.25, color='red')
            plt.scatter(x, y, alpha=0.125, color='blue')
            plt.show()


        x_pbc_border, y_pbc_border = unique_pbc_border
        pbc_x = np.concatenate((x, x_pbc_border))
        pbc_y = np.concatenate((y, y_pbc_border))
        pbc_dx = pbc_x[:, np.newaxis] - pbc_x
        pbc_dy = pbc_y[:, np.newaxis] - pbc_y

        arr = dx.copy()
        obc_d_desired = dx.copy()[np.ix_(np.arange(x.size), pbc_orig_idxs[unique_border_idxs])]
        pbc_d_desired = pbc_dx[np.ix_(np.arange(x.size), np.arange(x.size, pbc_x.size))]
        xidxs, yidxs = np.argwhere(obc_d_desired > np.abs(pbc_d_desired)).T
        arr[xidxs, yidxs] = pbc_d_desired[xidxs, yidxs]

        print(f"{x[0]}, {y[0]}")
        print()
        pbc_dx = arr
        if True:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            fig,     ax, cbar = plot_imshow(fig, ax, np.arange(pbc_dx.shape[0]), np.arange(pbc_dx.shape[1]), np.abs(np.flipud(pbc_dx)), doDiscreteCmap=True)
            ax.set_title("Without repeats in border")
            #plt.savefig("no_repeat_border.png")
            plt.show()

        x_to_pbc = pbc_dx[np.ix_(np.arange(x.size, pbc_x.size), np.arange(x.size))]
        y_to_pbc = pbc_dy[np.ix_(np.arange(y.size, pbc_y.size), np.arange(y.size))]

    return dx, dy


if __name__ == "__main__":
    arr_dict = sublattices(1)
    coords = arr_dict['discrete_honeycomb_coordinates']
    dx, dy = displacement_array(coords, True)

    print(dx)

    tri_A = arr_dict['triangular_A']
    tri_B = arr_dict['triangular_B']
    hexaflake = arr_dict['hexaflake']
    vacancies = arr_dict['vacancies']
    n_sites = tri_A.size*2

    reordered_idxs = np.concatenate(np.arange(n_sites)[hexaflake],  np.arange(n_sites)[vacancies])

    dx, dy = dx[np.ix_(reordered_idxs, reordered_idxs)], dx[np.ix_(reordered_idxs, reordered_idxs)]

    A_to_A = tri_A[:, np.newaxis] & tri_A[np.newaxis, :]
    B_to_B = tri_B[:, np.newaxis] & tri_B[np.newaxis, :]


    w1 = (() & ()) | (() & ()) | (() & ())

    fig, ax = plt.subplots(1,2)
    ax[0].set_title('dx')
    ax[1].set_title('dy')

    arrs = [dx, dy]
    for i in range(2):
        Z = arrs[i]
        X = np.arange(Z.shape[1])
        Y = np.arange(Z.shape[0])
        fig, ax[i], cbar = plot_imshow(fig, ax[i], X, Y, Z)


    plt.show()

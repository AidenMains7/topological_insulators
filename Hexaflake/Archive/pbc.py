import numpy as np
from scipy.linalg import eigvals, eigvalsh, eigh, eig
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from itertools import product

import sys
sys.path.append(".")
from Carpet.plotting import plot_imshow

from fixing_haldane import sublattices

def displacement_array(coordinates:np.ndarray, pbc:bool):
    x, y = coordinates
    if not pbc:
        dx = x - x[:, np.newaxis]
        dy = y - y[:, np.newaxis]
        return dx, dy
    else:
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
        #pbc_edge_idxs = np.concatenate([edge_pts for edge_pts in edge_idxs.values()]).flatten()

        x

        return dx, dy


def compute_hopping_array(coordinates_discrete, PBC):
    """
    Computes the hopping arrays (delta_x and delta_y) between lattice sites,
    accounting for periodic boundary conditions (PBC) if specified.

    Parameters:
    - coordinates_discrete (np.ndarray): Array of discrete x and y coordinates of lattice sites.
    - PBC (bool): Flag indicating whether to apply periodic boundary conditions.

    Returns:
    - delta_x (np.ndarray): Array of x-coordinate differences between sites.
    - delta_y (np.ndarray): Array of y-coordinate differences between sites.
    """
    # Extract x and y coordinates
    x, y = coordinates_discrete.T

    if not PBC:
        # Compute differences in x and y coordinates without PBC
        delta_x = x[:, np.newaxis] - x
        delta_y = y[:, np.newaxis] - y
    else:
        # Compute differences with PBC using predefined shifts
        a = np.max(y) + 1
        b = (a + 3) // 2
        c = (a - 3) // 2
        d = 2 * a - b
        e = 2 * a - c

        # Define shift vectors for PBC
        shifts = np.array([
            [0, 0],
            [-3, a],
            [3, -a],
            [d, b],
            [-d, -b],
            [-e, c],
            [e, -c]
        ])

        delta_x_stack = []
        delta_y_stack = []

        # Apply shifts and compute coordinate differences
        for shift in shifts:
            shifted_delta_x = x[:, np.newaxis] - (x + shift[0])
            shifted_delta_y = y[:, np.newaxis] - (y + shift[1])

            delta_x_stack.append(shifted_delta_x)
            delta_y_stack.append(shifted_delta_y)

        # Convert lists to arrays
        delta_x_stack = np.array(delta_x_stack)
        delta_y_stack = np.array(delta_y_stack)

        def get_smallest(stack):
            # Find indices of minimal distance (closest images)
            idx_array = np.argmin(stack**2, axis=0)

            # Create indices for selecting minimal differences
            i_indices, j_indices = np.indices(idx_array.shape)

            # Select minimal coordinate differences
            return stack[idx_array, i_indices, j_indices]
         
        delta_x = get_smallest(delta_x_stack)
        delta_y = get_smallest(delta_y_stack)

    # Convert differences to integers
    return delta_x.astype(np.int64), delta_y.astype(np.int64)



if __name__ == "__main__":
    arr_dict = sublattices(3)
    coords = arr_dict['discrete_honeycomb_coordinates']
    x, y = coords
    #dx, dy = displacement_array(coords, True)

    def pli(arr):
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        fig, ax, cbar = plot_imshow(fig, ax, np.arange(arr.shape[0]), np.arange(arr.shape[1]), arr)
        plt.show()


    dxd, dyd = compute_hopping_array(coords.T, True)
    dxd_ob, dyd_ob = compute_hopping_array(coords.T, False)

    desired_index = 0
    distances = dxd[0, :]

    plt.scatter(x[desired_index], y[desired_index], c='k')
    plt.scatter(x, y, c=distances)
    plt.colorbar()
    #plt.scatter(x[[40, 8]], y[[40, 8]])
    plt.show()
    
    #print(coords[:, [24, 32, 33, 34]])
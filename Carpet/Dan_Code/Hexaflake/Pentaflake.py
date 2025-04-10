import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial import ConvexHull
import time

def lattice_int_arrays(iterations):
    """
    Generates integer arrays representing the coordinates of a pristine or fractal lattice
    based on the number of iterations.

    Parameters:
        iterations (int): The number of iterations for generating the lattice.

    Returns:
        tuple: A tuple containing:
            - pristine_coord_arr (ndarray): Integer array for the pristine pentagon-shaped lattice.
            - fractal_coord_arr (ndarray): Integer array for the pentaflake (fractal) lattice.
    """
    # Initial coordinates of a regular hexagon
    initial_x = np.cos((np.pi / 3) * np.arange(6))
    initial_y = np.sin((np.pi / 3) * np.arange(6))

    def vertices(n):
        """
        Recursively generates the vertices of the fractal lattice.

        Parameters:
            n (int): The current iteration level.

        Returns:
            tuple: Arrays of x and y coordinates.
        """
        if n == 0:
            return initial_x, initial_y

        new_x, new_y = [], []
        previous_x, previous_y = vertices(n - 1)

        # Scaling factor for the current iteration
        new_scale = 3 ** n
        # Centers for the new hexagons added in this iteration
        new_x_centers, new_y_centers = [0], [0]
        new_x_centers += list(new_scale * np.cos((np.pi / 3) * np.arange(6)))
        new_y_centers += list(new_scale * np.sin((np.pi / 3) * np.arange(6)))

        # Generate new vertices by translating previous vertices to new centers
        for new_x_center, new_y_center in zip(new_x_centers, new_y_centers):
            new_x += list(previous_x + new_x_center)
            new_y += list(previous_y + new_y_center)

        return np.array(new_x), np.array(new_y)

    # Generate vertices for the given number of iterations
    x, y = vertices(iterations)

    # Convert floating point coordinates to integer grid coordinates
    x_ints = np.round(2 * (x - np.min(x))).astype(int)
    y_ints = np.round((y - np.min(y)) / (np.sqrt(3) / 2)).astype(int)

    # Create an integer array to represent the fractal lattice
    fractal_coord_arr = np.zeros(
        (np.max(y_ints) + 1, np.max(x_ints) + 1), dtype=int
    )
    y_unique_ints = np.unique(y_ints)
    for y_int in y_unique_ints:
        these_x_ints = np.unique(x_ints[y_ints == y_int])
        fractal_coord_arr[y_int, these_x_ints] = 1

    # Template for a single hexagon layer
    temp = np.array(
        [
            [0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
        ]
    )

    # Calculate dimensions for the pristine lattice array
    layers = (3 ** iterations - 1) // 2 + 1
    x_width = 2 * (3 ** (iterations + 1)) - 1
    piece1 = np.zeros((3 * layers, 2 * (3 ** (iterations + 1)) - 1), dtype=int)

    start = 0
    num_pieces = 3 ** iterations
    for j in range(layers):
        section = np.zeros((3, x_width + 1), dtype=int)
        for i in range(num_pieces):
            section[:, start + 6 * i : start + 6 * (i + 1)] = temp
        piece1[3 * j : 3 * (j + 1), :] = section[:, :-1]
        start += 3
        num_pieces -= 1

    # Reflect the upper half to get the lower half of the lattice
    piece2 = np.flipud(piece1[3:, :])

    # Combine both halves to create the full pristine lattice
    pristine_coord_arr = np.vstack((piece2, piece1))

    return pristine_coord_arr, fractal_coord_arr


def int_array_to_coords(bool_array):
    """
    Converts a boolean array representing lattice points into x and y coordinates.

    Parameters:
        bool_array (ndarray): A 2D boolean array where True indicates the presence of an atom.

    Returns:
        tuple: Arrays of x and y coordinates.
    """
    # Get indices where the boolean array is True
    y_ints, x_ints = np.where(bool_array)

    # Convert integer grid coordinates back to real space coordinates
    x = x_ints / 2
    y = (np.sqrt(3) / 2) * y_ints
    x = x - (1 / 2) * (np.max(x) - np.min(x))  # Center the x coordinates
    y = y - (1 / 2) * (np.max(y) - np.min(y))  # Center the y coordinates

    return x, y


def find_matching_row_indices(array1, array2):
    """
    Finds indices in array2 that match rows in array1.

    Parameters:
        array1 (ndarray): Array of points to find.
        array2 (ndarray): Array of points to search in.

    Returns:
        ndarray: Indices in array2 that match array1.
    """
    # Check for matches between each row of array1 and rows in array2
    matches = np.all(array1[:, np.newaxis] == array2, axis=2)
    # Get indices of matching rows
    matching_indices = np.where(matches.any(axis=1), matches.argmax(axis=1), -1)
    return matching_indices


def sort_by_angle(coord_pairs):
    """
    Sorts coordinate pairs based on their angle from the origin.

    Parameters:
        coord_pairs (ndarray): Array of [y, x] coordinate pairs.

    Returns:
        ndarray: Indices that would sort the array by angle.
    """
    # Shift coordinates to center them around the origin
    coord_pairs_shift = np.copy(coord_pairs).astype(np.float64)
    coord_pairs_shift[:, 0] -= (1 / 2) * (
        np.max(coord_pairs[:, 0]) - np.min(coord_pairs[:, 0])
    )
    coord_pairs_shift[:, 1] -= (1 / 2) * (
        np.max(coord_pairs[:, 1]) - np.min(coord_pairs[:, 1])
    )
    # Calculate angles using arctangent
    angles = np.arctan2(*tuple(coord_pairs_shift.T))
    # Return indices that would sort the array by angle
    return np.argsort(angles)


def line_segment(x0, y0, x1, y1):
    """
    Generates the integer points along a line segment between two points.

    Parameters:
        x0, y0 (int): Starting point coordinates.
        x1, y1 (int): Ending point coordinates.

    Returns:
        list: List of points along the line segment.
    """
    if x0 == x1:
        # Vertical line
        x_range = np.repeat(x0, np.abs(y1 - y0) + 1)
        y_range = np.arange(y0, y1 + 1)
    elif x0 < x1:
        # Line with positive x direction
        x_range = np.arange(x0, x1 + 1)
        y_range = ((y1 - y0) / (x1 - x0)) * (x_range - x0) + y0
    else:
        # Line with negative x direction
        x_range = np.arange(x1, x0 + 1)
        y_range = ((y0 - y1) / (x0 - x1)) * (x_range - x1) + y1

    # Return combined x and y coordinates
    return list(np.column_stack((y_range, x_range)))


def in_hull(point, hull):
    """
    Checks if a point is inside a given convex hull.

    Parameters:
        point (tuple): The (y, x) coordinates of the point to check.
        hull (tuple): The convex hull defined by unique y values and corresponding x ranges.

    Returns:
        bool: True if the point is inside the hull, False otherwise.
    """
    y, x = point
    y_hull_unique, x_hull_ranges = hull
    idx = np.where(y_hull_unique == y)[0]
    if idx.size == 0:
        return False
    x_min, x_max = x_hull_ranges[idx[0]][0], x_hull_ranges[idx[0]][1]
    # Check if x is within the x-range for the given y
    if x_min <= x <= x_max:
        return True
    return False


def hull_indices(y_vals, x_vals, hulls):
    """
    Assigns each point to a hull index if it lies within any of the given hulls.

    Parameters:
        y_vals (ndarray): Array of y coordinates of points.
        x_vals (ndarray): Array of x coordinates of points.
        hulls (tuple): Tuple of hulls to check against.

    Returns:
        ndarray: Array of hull indices corresponding to each point.
    """
    points = np.column_stack((y_vals, x_vals))

    def hull_index(point_):
        # Check which hull the point belongs to
        for i, hull in enumerate(hulls):
            if in_hull(point_, hull):
                return i
        return np.nan  # Return NaN if point is not in any hull

    hull_idxs = []
    for point in points:
        hull_idxs.append(hull_index(point))

    return np.array(hull_idxs)


def d_vecs_pre(bool_array, pbc):
    """
    Precomputes the components of the nearest-neighbor hopping vectors for the Hamiltonian.

    Parameters:
        bool_array (ndarray): 2D boolean array representing the lattice.
        pbc (bool): If True, applies periodic boundary conditions.

    Returns:
        tuple: Three arrays representing components of the hopping vectors.
    """
    # Get the positions of the lattice points
    points = np.column_stack(np.where(bool_array))

    if pbc:
        # Apply periodic boundary conditions

        # Shift to center
        x0_shift = (np.max(points[:, 1]) - np.min(points[:, 1])) // 2
        y0_shift = (np.max(points[:, 0]) - np.min(points[:, 0])) // 2

        # Get the convex hull of the points
        hull_inds = ConvexHull(points).vertices
        hull_coords = points[hull_inds]
        sorts = sort_by_angle(hull_coords)
        sorted_ints = hull_inds[sorts]
        vertices = points[sorted_ints]

        # Generate edge points
        edge_points = []
        for i in range(len(vertices)):
            y0_, x0_ = vertices[i]
            y1_, x1_ = vertices[(i + 1) % len(vertices)]
            edge_points += line_segment(x0_, y0_, x1_, y1_)

        # Extract unique edge points and shift to center
        all_y_edge, all_x_edge = tuple(
            np.unique(np.array(edge_points).astype(int), axis=0).T
        )
        all_y_edge -= y0_shift
        all_x_edge -= x0_shift

        # Build x ranges for each unique y in edge
        y_edge_unique = np.unique(all_y_edge)
        x_edge_ranges = []
        for y_edge in y_edge_unique:
            y_idxs = np.where(all_y_edge == y_edge)[0]
            x_vals = all_x_edge[y_idxs]
            x_edge_ranges.append([np.min(x_vals), np.max(x_vals)])
        x_edge_ranges = np.array(x_edge_ranges)

        # Define shifts for periodic images
        shifts = np.array(
            [
                [0, 0],
                [39, 15],
                [-3, 27],
                [-42, 12],
                [-39, -15],
                [3, -27],
                [42, -12],
            ]
        )

        # Adjust shifts for matching edges
        shifts[1] += np.array([1, -1])
        shifts[6] += np.array([-1, -1])
        shifts[5] += np.array([-2, 0])
        shifts[4] += np.array([-1, 1])
        shifts[3] += np.array([1, 1])
        shifts[2] += np.array([2, 0])

        shifts *= 3  # Scale shifts appropriately
        shifts = shifts[:, ::-1]  # Swap columns to match x, y ordering

        # Create hulls with shifted coordinates
        hulls = tuple(
            [
                (y_edge_unique + shift[0], x_edge_ranges + shift[1])
                for shift in shifts
            ]
        )

        # Shift original points to center
        original_y, original_x = points[:, 0] - y0_shift, points[:, 1] - x0_shift

        all_x, all_y = [], []
        for shift in shifts:
            all_x += list(original_x + shift[1])
            all_y += list(original_y + shift[0])

        all_x, all_y = np.array(all_x), np.array(all_y)
        # Scale coordinates to real space
        all_x_scaled = all_x / 2
        all_y_scaled = all_y * np.sqrt(3) / 2

        # Compute radial distances from center
        all_r_from_center = np.sqrt(all_x_scaled ** 2 + all_y_scaled ** 2)
        range_lim = (
            1.5
            * ((np.max(all_y_edge) - np.min(all_y_edge)) / 2)
            * np.sqrt(3)
            / 2
        )

        # Keep only points within the valid range
        within_valid_range = np.where(all_r_from_center < range_lim)[0]
        all_x, all_y = all_x[within_valid_range], all_y[within_valid_range]

        # Compute differences between all pairs of points
        dx = all_x.reshape(all_x.size, 1) - all_x
        dy = all_y.reshape(all_y.size, 1) - all_y

        # Assign hull indices to points
        hull_idxs = hull_indices(all_y, all_x, hulls)

        # Initialize shift arrays
        x_shifts = np.repeat(0, hull_idxs.size)
        y_shifts = np.repeat(0, hull_idxs.size)

        # Apply shifts based on hull indices
        for i in range(7):
            these_idxs = np.where(hull_idxs == i)[0]
            x_shifts[these_idxs] += shifts[i, 1]
            y_shifts[these_idxs] += shifts[i, 0]

        # Correct for shifts to get original positions
        all_x += x0_shift - x_shifts
        all_y += y0_shift - y_shifts

    else:
        # No periodic boundary conditions
        all_y, all_x = np.where(bool_array)
        dx = all_x.reshape(all_x.size, 1) - all_x
        dy = all_y.reshape(all_y.size, 1) - all_y

    system_size = np.sum(bool_array)

    # Initialize matrices for hopping vectors
    d1_nn = np.zeros((system_size, system_size), dtype=np.complex128)
    d2_nn = np.zeros((system_size, system_size), dtype=np.complex128)
    d3_nn = np.zeros((system_size, system_size), dtype=np.complex128)

    # Identify nearest neighbors for different directions
    # Direction 1 for sublattice A
    nn1A_rows, nn1A_cols = np.where((dx == -2) & (dy == 0))
    # Direction 2 for sublattice A
    nn2A_rows, nn2A_cols = np.where((dx == 1) & (dy == 1))
    # Direction 3 for sublattice A
    nn3A_rows, nn3A_cols = np.where((dx == 1) & (dy == -1))

    # Extract coordinates for the identified pairs
    nn1A_x0, nn1A_y0, nn1A_xf, nn1A_yf = (
        all_x[nn1A_rows],
        all_y[nn1A_rows],
        all_x[nn1A_cols],
        all_y[nn1A_cols],
    )
    nn2A_x0, nn2A_y0, nn2A_xf, nn2A_yf = (
        all_x[nn2A_rows],
        all_y[nn2A_rows],
        all_x[nn2A_cols],
        all_y[nn2A_cols],
    )
    nn3A_x0, nn3A_y0, nn3A_xf, nn3A_yf = (
        all_x[nn3A_rows],
        all_y[nn3A_rows],
        all_x[nn3A_cols],
        all_y[nn3A_cols],
    )

    # Create point pairs for matching
    nn1A_p0, nn1A_pf = np.column_stack((nn1A_y0, nn1A_x0)), np.column_stack(
        (nn1A_yf, nn1A_xf)
    )
    nn2A_p0, nn2A_pf = np.column_stack((nn2A_y0, nn2A_x0)), np.column_stack(
        (nn2A_yf, nn2A_xf)
    )
    nn3A_p0, nn3A_pf = np.column_stack((nn3A_y0, nn3A_x0)), np.column_stack(
        (nn3A_yf, nn3A_xf)
    )

    # Find indices in the original point array
    nn1A_idxs0 = find_matching_row_indices(nn1A_p0, points)
    nn1A_idxsf = find_matching_row_indices(nn1A_pf, points)
    nn2A_idxs0 = find_matching_row_indices(nn2A_p0, points)
    nn2A_idxsf = find_matching_row_indices(nn2A_pf, points)
    nn3A_idxs0 = find_matching_row_indices(nn3A_p0, points)
    nn3A_idxsf = find_matching_row_indices(nn3A_pf, points)

    # Initialize masks for the hopping terms
    nn1A_mask = np.full((system_size, system_size), False)
    nn2A_mask = np.full((system_size, system_size), False)
    nn3A_mask = np.full((system_size, system_size), False)

    # Set True for nearest neighbor pairs
    nn1A_mask[nn1A_idxs0, nn1A_idxsf] = True
    nn2A_mask[nn2A_idxs0, nn2A_idxsf] = True
    nn3A_mask[nn3A_idxs0, nn3A_idxsf] = True

    # Repeat the same for sublattice B
    nn1B_rows, nn1B_cols = np.where((dx == 2) & (dy == 0))
    nn2B_rows, nn2B_cols = np.where((dx == -1) & (dy == -1))
    nn3B_rows, nn3B_cols = np.where((dx == -1) & (dy == 1))

    nn1B_x0, nn1B_y0, nn1B_xf, nn1B_yf = (
        all_x[nn1B_rows],
        all_y[nn1B_rows],
        all_x[nn1B_cols],
        all_y[nn1B_cols],
    )
    nn2B_x0, nn2B_y0, nn2B_xf, nn2B_yf = (
        all_x[nn2B_rows],
        all_y[nn2B_rows],
        all_x[nn2B_cols],
        all_y[nn2B_cols],
    )
    nn3B_x0, nn3B_y0, nn3B_xf, nn3B_yf = (
        all_x[nn3B_rows],
        all_y[nn3B_rows],
        all_x[nn3B_cols],
        all_y[nn3B_cols],
    )

    nn1B_p0, nn1B_pf = np.column_stack((nn1B_y0, nn1B_x0)), np.column_stack(
        (nn1B_yf, nn1B_xf)
    )
    nn2B_p0, nn2B_pf = np.column_stack((nn2B_y0, nn2B_x0)), np.column_stack(
        (nn2B_yf, nn2B_xf)
    )
    nn3B_p0, nn3B_pf = np.column_stack((nn3B_y0, nn3B_x0)), np.column_stack(
        (nn3B_yf, nn3B_xf)
    )

    nn1B_idxs0 = find_matching_row_indices(nn1B_p0, points)
    nn1B_idxsf = find_matching_row_indices(nn1B_pf, points)
    nn2B_idxs0 = find_matching_row_indices(nn2B_p0, points)
    nn2B_idxsf = find_matching_row_indices(nn2B_pf, points)
    nn3B_idxs0 = find_matching_row_indices(nn3B_p0, points)
    nn3B_idxsf = find_matching_row_indices(nn3B_pf, points)

    nn1B_mask = np.full((system_size, system_size), False)
    nn2B_mask = np.full((system_size, system_size), False)
    nn3B_mask = np.full((system_size, system_size), False)

    nn1B_mask[nn1B_idxs0, nn1B_idxsf] = True
    nn2B_mask[nn2B_idxs0, nn2B_idxsf] = True
    nn3B_mask[nn3B_idxs0, nn3B_idxsf] = True

    ###############################################
    # Assign hopping vectors based on masks
    d1_nn[nn1A_mask] = 1j / 2
    d1_nn[nn2A_mask] = -1j / 4
    d1_nn[nn3A_mask] = -1j / 4

    d1_nn[nn1B_mask] = -1j / 2
    d1_nn[nn2B_mask] = 1j / 4
    d1_nn[nn3B_mask] = 1j / 4
    ###############################################
    d2_nn[nn2A_mask] = 1j * np.sqrt(3) / 4
    d2_nn[nn3A_mask] = -1j * np.sqrt(3) / 4

    d2_nn[nn2B_mask] = -1j * np.sqrt(3) / 4
    d2_nn[nn3B_mask] = 1j * np.sqrt(3) / 4
    ###############################################
    d3_nn[nn1A_mask] = 1.0 + 0.0j
    d3_nn[nn2A_mask] = 1.0 + 0.0j
    d3_nn[nn3A_mask] = 1.0 + 0.0j

    d3_nn[nn1B_mask] = 1.0 + 0.0j
    d3_nn[nn2B_mask] = 1.0 + 0.0j
    d3_nn[nn3B_mask] = 1.0 + 0.0j
    ###############################################

    return d1_nn, d2_nn, d3_nn


def hamiltonian_nn(iterations, M, t=1.0, B=1.0, pbc=False, fractal=False):
    """
    Constructs the nearest-neighbor tight-binding Hamiltonian for the lattice.

    Parameters:
        iterations (int): Number of iterations for lattice generation.
        M (float): Mass term in the Hamiltonian.
        t (float, optional): Hopping parameter. Default is 1.
        B (float, optional): Coefficient for the sigma_z term. Default is 1.
        pbc (bool, optional): If True, applies periodic boundary conditions. Default is False.
        fractal (bool, optional): If True, uses the fractal lattice. Default is False.

    Returns:
        tuple: The Hamiltonian matrix and the coordinates of the lattice points.
    """
    # Generate the integer arrays representing the lattice
    bool_arrays = lattice_int_arrays(iterations)
    # Choose between pristine or fractal lattice
    bool_array = bool_arrays[0] if not fractal else bool_arrays[1]
    # Precompute the hopping vectors
    d_vec_pre = d_vecs_pre(bool_array, pbc)

    # Pauli matrices
    tau1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    tau2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    tau3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    d1_pre, d2_pre, d3_pre = d_vec_pre[:]

    unity = np.eye(d1_pre.shape[0], dtype=np.complex128)

    # Build the mass term
    d3 = (M - 4 * B) * unity + B * d3_pre

    # Construct the Hamiltonian
    h = (
        np.kron(t * d1_pre, tau1)
        + np.kron(t * d2_pre, tau2)
        + np.kron(d3, tau3)
    )

    # Get the real-space coordinates of the lattice points
    coords = int_array_to_coords(bool_array)

    return h, coords


def plot_data(coords, eigenvalues, eigenvectors, num_zero_modes):
    """
    Processes and plots the eigenvalues and eigenvectors.

    Parameters:
        coords (tuple): Coordinates of the lattice points.
        eigenvalues (ndarray): Array of eigenvalues.
        eigenvectors (ndarray): Array of eigenvectors.
        num_zero_modes (int): Number of zero modes to consider.

    Returns:
        tuple: Processed indices and data for plotting.
    """
    # Find indices of the eigenvalues closest to zero
    zero_idxs = np.argsort(np.abs(eigenvalues))[:num_zero_modes]
    zero_idxs = zero_idxs[np.argsort(eigenvalues[zero_idxs])]
    other_idxs = np.delete(np.arange(eigenvalues.size), zero_idxs)

    # Get the corresponding eigenvectors
    zero_vecs = eigenvectors[:, zero_idxs]

    # Compute amplitudes squared (probability densities)
    amps = (np.abs(zero_vecs) ** 2).sum(axis=1)

    # Compute local density of states
    ldos_vals = amps[::2] + amps[1::2]
    sorts = np.argsort(ldos_vals)
    ldos_vals = ldos_vals[sorts]
    x, y = coords[0][sorts], coords[1][sorts]

    return other_idxs, zero_idxs, x, y, ldos_vals


def main():
    """
    Main function to generate the Hamiltonian, compute eigenvalues and eigenvectors,
    and plot the results.
    """
    iterations = 3
    M = -1.0
    num_close_to_zero = 2

    # Generate the Hamiltonian and coordinates
    h, coords = hamiltonian_nn(iterations, M=M, pbc=False)

    print(f"System size: {coords[0].size}")

    t0 = time.time()
    # Compute eigenvalues and eigenvectors
    vals, vecs = eigh(h, overwrite_a=False)
    print(f"Hamiltonian solved in {round(time.time()-t0)} seconds")

    # Process data for plotting
    other_idxs, zero_idxs, x, y, ldos_vals = plot_data(
        coords, vals, vecs, num_close_to_zero
    )

    # Plot eigenvalues and local density of states
    fig, axs = plt.subplots(1, 2)

    axs[0].scatter(other_idxs, vals[other_idxs], c="k")
    axs[0].scatter(zero_idxs, vals[zero_idxs], c="b")
    axs[0].set_title("Eigenvalues")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("Eigenvalue")

    cax = axs[1].scatter(x, y, c=ldos_vals, cmap="binary", s=15)
    axs[1].set_aspect("equal")
    axs[1].set_title("Local Density of States")
    fig.colorbar(cax, ax=axs[1])

    plt.show()


if __name__ == "__main__":
    main()


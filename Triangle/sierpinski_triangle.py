import numpy as np
from matplotlib import pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy as sp
import os
from joblib import Parallel, delayed
from itertools import product
from tqdm_joblib import tqdm_joblib, tqdm
import h5py
from matplotlib import colormaps 
from scipy.integrate import dblquad
from scipy.spatial import ConvexHull
import cProfile, pstats
from mpl_toolkits.mplot3d import Axes3D

# returns lattice
def compute_triangular_lattice(generation):
    def recursive_lattice(_gen):
        if _gen == 0:
            return np.array([0.0, -np.sqrt(3)/2, np.sqrt(3)/2, 1.0, -0.5, -0.5]).reshape(2, 3)
        else:
            smaller = recursive_lattice(_gen - 1)
            x_range = np.max(smaller[0]) - np.min(smaller[0])
            y_range = np.max(smaller[1]) - np.min(smaller[1])
            lattice_points = np.zeros((2, smaller.shape[1]*4))

            shifts = {
                "top": np.array([[0], [y_range]]),
                "left": np.array([[-x_range/2], [0]]),
                "right": np.array([[x_range/2], [0]]),
            }

            for i, shift in enumerate(shifts.values()):
                lattice_points[:, i::4] = smaller + shift
            
            flipped_max_y = np.min((smaller+shifts["left"])[1])
            flipped = smaller.copy()
            flipped[1] *= -1
            flipped[1] -= np.min(flipped[1]) - flipped_max_y

            lattice_points[:, 3::4] = flipped
            return np.unique(lattice_points, axis=1)
        
    coordinates = recursive_lattice(generation)
    xmin, ymin = np.min(coordinates[0]), np.min(coordinates[1])
    coordinates[0] -= xmin
    coordinates[1] -= ymin
    coordinates[0] *= 2/np.sqrt(3)
    coordinates[1] *= 2
    coordinates = np.unique(np.round(coordinates).astype(int), axis=1)
    sorted_idxs = np.lexsort((coordinates[0], coordinates[1]))
    coordinates = coordinates[:, sorted_idxs]
    
    lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
    lattice[coordinates[1], coordinates[0]] = np.arange(coordinates.shape[1])
    return lattice

# returns lattice, triangular_hole_locations as integer
def compute_sierpinski_triangle(generation):
    def recursive_fractal(_gen):
        if _gen == 0:
            return {"lattice_points": np.array([0.0, -np.sqrt(3)/2, np.sqrt(3)/2, 1.0, -0.5, -0.5]).reshape(2, 3), "triangular_hole_locations": None}
        else:
            fractal_dict = recursive_fractal(_gen - 1)
            smaller = fractal_dict["lattice_points"]

            x_range = np.max(smaller[0]) - np.min(smaller[0])
            y_range = np.max(smaller[1]) - np.min(smaller[1])
            fractal = np.zeros((2, smaller.shape[1]*3))

            shifts = {
                "top": np.array([[0], [y_range]]),
                "left": np.array([[-x_range/2], [0]]),
                "right": np.array([[x_range/2], [0]]),
            }

            for i, shift in enumerate(shifts.values()):
                fractal[:, i::3] = smaller + shift

            # Hole size, x location, y location
            location_of_hole = np.array([np.mean(fractal[0]), np.min(fractal[1]), _gen-1]).T

            if fractal_dict["triangular_hole_locations"] is not None:
                smaller_triangular_hole_locations = fractal_dict["triangular_hole_locations"]
                new_hole_locations = np.zeros((3, smaller_triangular_hole_locations.shape[1]*3))
                for i, shift in enumerate(shifts.values()):
                    new_hole_locations[2, i::3] = smaller_triangular_hole_locations[2] # hole size
                    new_hole_locations[[0,1], i::3] = smaller_triangular_hole_locations[[0,1]] + shift # x pos
                
                all_hole_points = np.append(new_hole_locations, location_of_hole.reshape(3, 1), axis=1)
            else:
                all_hole_points = location_of_hole.reshape(3, 1)

            return {"lattice_points": np.unique(fractal, axis=1), "triangular_hole_locations": np.unique(all_hole_points, axis=1)}
    fractal_dict = recursive_fractal(generation)
    coordinates = fractal_dict["lattice_points"]
    
    xmin, ymin = np.min(coordinates[0]), np.min(coordinates[1])
    coordinates[0] -= xmin
    coordinates[1] -= ymin
    coordinates[0] *= 2/np.sqrt(3)
    coordinates[1] *= 2
    coordinates = np.unique(np.round(coordinates).astype(int), axis=1)
    sorted_idxs = np.lexsort((coordinates[0], coordinates[1]))
    coordinates = coordinates[:, sorted_idxs]
    
    lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
    lattice[coordinates[1], coordinates[0]] = np.arange(coordinates.shape[1])


    if fractal_dict["triangular_hole_locations"] is None:
        return {"lattice": lattice, "triangular_hole_locations": None}
    
    hole_locations = fractal_dict["triangular_hole_locations"]
    hole_locations[0] -= xmin
    hole_locations[1] -= ymin
    hole_locations[0] *= 2/np.sqrt(3)
    hole_locations[1] *= 2
    hole_locations = np.round(hole_locations).astype(int)
    return {"lattice": lattice, "triangular_hole_locations": hole_locations}

# not good yet
def tile_triangle_to_implement_pbc(generation, doFractal=True):
    if doFractal:
        fractal_dict = compute_sierpinski_triangle(generation)
        lattice, triangular_hole_locations = fractal_dict["lattice"], fractal_dict["triangular_hole_locations"]
    else:
        lattice = compute_triangular_lattice(generation)
        triangular_hole_locations = None
    
    y, x = np.where(lattice >= 0)
    coordinates = np.vstack((x, y))

    arrays = [coordinates.copy()]
    if doFractal:
        arrays.append(triangular_hole_locations.copy())
        arrays[1][2] *= -1 # filp hole size as to imply that the hole is an upright triangle now
    
    for arr in arrays:
        arr[1] = np.max(coordinates[1])-arr[1] # flip upside down, adjust y position
        arr[0] += np.max(coordinates[0])//2 # shift x position to the right
  
    tiled_coords = np.hstack((coordinates, arrays[0]))
    
    y_min, x_min = np.min(tiled_coords[1]), np.min(tiled_coords[0])
    tiled_coords[0] -= x_min
    tiled_coords[1] -= y_min
    tiled_coords = np.unique(tiled_coords, axis=1)
    tiled_coords = tiled_coords[:, np.lexsort((tiled_coords[0], tiled_coords[1]))]

    lattice = np.full((np.max(tiled_coords[1])+1, np.max(tiled_coords[0])+1), -1)
    lattice[tiled_coords[1], tiled_coords[0]] = np.arange(tiled_coords.shape[1])

    if doFractal:
        tiled_hole_locations = np.hstack((triangular_hole_locations, arrays[1]))
        tiled_hole_locations[0] -= x_min
        tiled_hole_locations[1] -= y_min
        tiled_hole_locations = np.unique(tiled_hole_locations, axis=1)
    else:
        tiled_hole_locations = None

    return {"lattice" : lattice, "triangular_hole_locations" : tiled_hole_locations}

# works with pbc (check with tiled)
def calcluate_dx_dy(lattice, pbc):
    y, x = np.where(lattice >= 0)
    if not pbc:
        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
    else:
        x_range = np.max(x)
        y_range = np.max(y)

        displacements = {
            "center": np.array([0, 0]).T,
            "top_left": np.array([-x_range/3, y_range]).T,
            "left": np.array([-x_range*2/3, 0]).T,
            "bottom_left": np.array([-x_range, -y_range]).T,
            "bottom": np.array([-x_range/3, -y_range]).T,
            "bottom_right": np.array([x_range/3, -y_range]).T,
            "right": np.array([x_range*2/3, 0]).T,
            "top_right": np.array([x_range, y_range]).T,
            "top": np.array([x_range/3, y_range]).T
        }

        all_x_shifts = np.empty((1, x.shape[0], 9))
        all_y_shifts = np.empty((1, x.shape[0], 9))
        for i, shift in enumerate(displacements.values()):
            all_x_shifts[:, :, i] = x + shift[0]
            all_y_shifts[:, :, i] = y + shift[1]

        all_dx = x[:, np.newaxis, np.newaxis] - all_x_shifts
        all_dy = y[:, np.newaxis, np.newaxis] - all_y_shifts

        distances = all_dx**2 + all_dy**2
        minimal_distances = np.argmin(distances, axis=-1)
        i_idxs, j_idxs = np.indices(minimal_distances.shape)
        dx = x[:, np.newaxis, np.newaxis] - all_x_shifts
        dy = y[:, np.newaxis, np.newaxis] - all_y_shifts
        dx = dx[i_idxs, j_idxs, minimal_distances]
        dy = dy[i_idxs, j_idxs, minimal_distances]

    print(x.shape)
    print(dx.shape)


    return dx, dy


def plot_relative_distances(generation, PBC, rel_origin=(0., 0.), abs_dist=True, doFractal=False):

    lattice_dict = tile_triangle_to_implement_pbc(generation, doFractal)
    lattice = lattice_dict["lattice"]

    delta_x_discrete, delta_y_discrete = calcluate_dx_dy(lattice, PBC)

    y, x = np.where(lattice >= 0)

    x_origin_rel, y_origin_rel = rel_origin
    x_origin = (x_origin_rel / 2) * (x.max() - x.min())
    y_origin = (y_origin_rel / 2) * (y.max() - y.min())
    origin_site = np.argmin((x - x_origin)**2 + (y - y_origin)**2)

    x_true_origin, y_true_origin = x[origin_site], y[origin_site]

    x_dist = (1 / 2) * delta_x_discrete[origin_site]
    y_dist = (np.sqrt(3) / 2) * delta_y_discrete[origin_site]
    euclidean_dist = np.sqrt(x_dist**2 + y_dist**2)
    r_span = euclidean_dist.max()

    if abs_dist:
        x_dist_plot = np.abs(x_dist)
        y_dist_plot = np.abs(y_dist)
        norm_xy = plt.Normalize(0, r_span)
    else:
        x_dist_plot = x_dist
        y_dist_plot = y_dist
        norm_xy = plt.Normalize(-r_span, r_span)

    norm_r = plt.Normalize(0, r_span)

    other_sites = np.delete(np.arange(x.size), origin_site)

    num_cbar_ticks = 7
    num_axes_ticks = 5
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    xy_cmap = 'inferno' if abs_dist else 'jet'
    xy_origin_color = 'lime' if abs_dist else 'magenta'

    for i in range(3):
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y', rotation=0)
        axs[i].xaxis.set_label_coords(0.5, -0.1)
        axs[i].yaxis.set_label_coords(-0.133, 0.483)

    scatter1 = axs[0].scatter(
        x[other_sites],
        y[other_sites],
        c=x_dist_plot[other_sites],
        cmap=xy_cmap,
        s=20,
        norm=norm_xy
    )
    axs[0].scatter(x_true_origin, y_true_origin, c=xy_origin_color, s=100, marker='*')
    dx_title = '|Δx|' if abs_dist else 'Δx'
    axs[0].set_title(dx_title)
    axs[0].set_aspect('equal')
    axs[0].set_xticks(np.linspace(x.min(), x.max(), num_axes_ticks))
    axs[0].set_yticks(np.linspace(y.min(), y.max(), num_axes_ticks))
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

    divider1 = make_axes_locatable(axs[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cb1 = fig.colorbar(scatter1, cax=cax1)
    cb1.set_ticks(np.linspace(norm_xy.vmin, norm_xy.vmax, num_cbar_ticks))
    cb1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

    scatter2 = axs[1].scatter(
        x[other_sites],
        y[other_sites],
        c=y_dist_plot[other_sites],
        cmap=xy_cmap,
        s=20,
        norm=norm_xy
    )
    axs[1].scatter(x_true_origin, y_true_origin, c=xy_origin_color, s=100, marker='*')
    dy_title = '|Δy|' if abs_dist else 'Δy'
    axs[1].set_title(dy_title)
    axs[1].set_aspect('equal')
    axs[1].set_xticks(np.linspace(x.min(), x.max(), num_axes_ticks))
    axs[1].set_yticks(np.linspace(y.min(), y.max(), num_axes_ticks))
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

    divider2 = make_axes_locatable(axs[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cb2 = fig.colorbar(scatter2, cax=cax2)
    cb2.set_ticks(np.linspace(norm_xy.vmin, norm_xy.vmax, num_cbar_ticks))
    cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

    scatter3 = axs[2].scatter(
        x[other_sites],
        y[other_sites],
        c=euclidean_dist[other_sites],
        cmap='inferno',
        s=20,
        norm=norm_r
    )
    axs[2].scatter(x_true_origin, y_true_origin, c='lime', s=100, marker='*')
    axs[2].set_title('Δr')
    axs[2].set_aspect('equal')
    axs[2].set_xticks(np.linspace(x.min(), x.max(), num_axes_ticks))
    axs[2].set_yticks(np.linspace(y.min(), y.max(), num_axes_ticks))
    axs[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

    divider3 = make_axes_locatable(axs[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cb3 = fig.colorbar(scatter3, cax=cax3)
    cb3.set_ticks(np.linspace(norm_r.vmin, norm_r.vmax, num_cbar_ticks))
    cb3.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

    plt.tight_layout()
    plt.show()
 
# check with tiled
def calculate_hopping(dx, dy):
    # Must be in DISCRETE coordinates
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    NN = ((abs_dx == 2) & (abs_dy == 0)) | ((abs_dx == 1) & (abs_dy == 3))
    #NNN = ((abs_dy == 6) & ((abs_dx == 0) | (abs_dx == 2))) | ((abs_dy == 3) & (abs_dx == 3)) | ((abs_dy == 0) & (abs_dx == 4))
    
    # Three directions for NN:
    # b1: (1, 0)
    # b2: (1 / 2, sqrt(3) / 2)
    # b2_tilde: (1 / 2, -sqrt(3) / 2)

    # In integer:
    # b1: (2, 0)
    # b2: (1, 3)
    # b2_tilde: (1, -3)

    NN_b1 = (dx == 2) & (dy == 0)
    NN_b2 = (dx == 1) & (dy == 3)
    NN_b2_tilde = (dx == 1) & (dy == -3)

    # Three diretions for NNN: 
    # c1: (0,    sqrt(3))
    # c2: (3 / 2,  sqrt(3) / 2)
    # c3: (3 / 2, -sqrt(3) / 2)

    # In integer:
    # c1: (0, 6)
    # c2: (3, 3)
    # c3: (3, -3)

    NNN_c1 = (dx == 0) & (dy == 6)
    NNN_c2 = (dx == 3) & (dy == 3)
    NNN_c3 = (dx == 3) & (dy == -3)

    hopping = {
        "NN_b1": NN_b1,
        "NN_b2": NN_b2,
        "NN_b2_tilde": NN_b2_tilde,
        "NNN_c1": NNN_c1,
        "NNN_c2": NNN_c2,
        "NNN_c3": NNN_c3,
        "b_hopping_vectors": np.array([[1., 0.], [0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]]),
        "c_hopping_vectors": np.array([[0., np.sqrt(3)], [3/2, np.sqrt(3)/2], [3/2, -np.sqrt(3)/2]])
    }

    return {key: hopping[key].astype(np.complex128) for key in hopping.keys()}


def compute_hamiltonian(lattice, pbc, M, B_tilde, A_tilde=1.0, B=1.0, t1=1.0):
    dx, dy = calcluate_dx_dy(lattice, pbc)
    NN_arrays = calculate_hopping(dx, dy)
    NN_b1, NN_b2, NN_b2_tilde, NNN_c1, NNN_c2, NNN_c3, b_vecs, c_vecs = [NN_arrays[key] for key in NN_arrays.keys()]

    I = np.eye(NN_b1.shape[0], dtype=np.complex128)

    d1 = t1/(2j) * NN_b1 + t1/(4j) * (NN_b2 + NN_b2_tilde)
    d2 = (-t1 * np.sqrt(3)/(4j)) * (NN_b2 - NN_b2_tilde)
    d3 = (M - 4*B) * I + B * (NN_b1 + NN_b2 + NN_b2_tilde)

    d1 += d1.conj().T
    d2 += d2.conj().T
    d3 += d3.conj().T

    d1_tilde = np.sqrt(3)*A_tilde/(8j) * (NNN_c2 + NNN_c3)
    d2_tilde = -A_tilde/(2j) * NNN_c1 + A_tilde/(4j) * (NNN_c2 - NNN_c3)
    d3_tilde = -6*B_tilde * I + B_tilde * (NNN_c1 + NNN_c2 + NNN_c3)

    d1_tilde += d1_tilde.conj().T
    d2_tilde += d2_tilde.conj().T
    d3_tilde += d3_tilde.conj().T

    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    pauli = [pauli_x, pauli_y, pauli_z]

    H_NN = np.kron(d1, pauli[0]) + np.kron(d2, pauli[1]) + np.kron(d3, pauli[2])
    H_NNN = np.kron(d1_tilde, pauli[0]) + np.kron(d2_tilde, pauli[1]) + np.kron(d3_tilde, pauli[2])
    
    return {"H": H_NN + H_NNN, "d": np.array([d1, d2, d3]), "d_tilde": np.array([d1_tilde, d2_tilde, d3_tilde])}


def plot_bonds(lattice, NN, ax, plotIndicesNames=False):

    y_discrete, x_discrete = np.where(lattice >= 0)
    yidx, xidx = np.argwhere(lattice >= 0).T

    i, j = np.where(NN)
    upper_triangle = j > i
    i, j = i[upper_triangle], j[upper_triangle]

    ax.plot([x_discrete[i], x_discrete[j]], [y_discrete[i], y_discrete[j]], c=(0., 0., 0., 1.), zorder=1)

    if plotIndicesNames:
        for yidx, xidx in zip(yidx, xidx):
            plt.text(xidx, yidx, str(lattice[yidx, xidx]), fontsize=8, ha='right', c='r')
            ax.scatter(xidx, yidx, alpha=0.5, c='k')


def remove_fractal_bonds(triangular_hole_locations, lattice):
    def get_side_length(n):
        if n == 0:
            return 2
        elif n < 0:
            raise ValueError("n must be a non-negative integer")
        else:
            return 2*get_side_length(n-1) - 1
    dx, dy = calcluate_dx_dy(lattice, False)
    NN, NNN = calculate_hopping(dx, dy)

    hole_x, hole_y, hole_n = triangular_hole_locations
    remove_n0_holes_mask = (hole_n != 0)
    hole_x, hole_y, hole_n = hole_x[remove_n0_holes_mask], hole_y[remove_n0_holes_mask], hole_n[remove_n0_holes_mask]
    moving_vector_right = np.array([1, 3]).T
    moving_vector_left = np.array([-1, 3]).T
    hole_pos = np.vstack((hole_x, hole_y))

    for pos, n in zip(hole_pos.T, hole_n):
        # n is the relative generation size of the hole. Positive n means the hole is an inverted triangle, negative n means the hole is an upright triangle.
        horizontal_break_pos = pos + np.sign(n) * moving_vector_right
        h_break_end = horizontal_break_pos + np.sign(n) * np.array([-2, 0])

        right_vertical_break_pos = pos + (get_side_length(abs(n))-2) * np.sign(n) * moving_vector_right
        rv_break_end = right_vertical_break_pos + np.sign(n) * np.array([-1, 3])

        left_vertical_break_pos = pos + (get_side_length(abs(n))-2) * np.sign(n) * moving_vector_left
        lv_break_end = left_vertical_break_pos + np.sign(n) * np.array([1, 3])

        
        hor_i = lattice[horizontal_break_pos[1], horizontal_break_pos[0]]
        hor_j = lattice[h_break_end[1], h_break_end[0]]

        right_ver_i = lattice[right_vertical_break_pos[1], right_vertical_break_pos[0]]
        right_ver_j = lattice[rv_break_end[1], rv_break_end[0]]

        left_ver_i = lattice[left_vertical_break_pos[1], left_vertical_break_pos[0]]
        left_ver_j = lattice[lv_break_end[1], lv_break_end[0]]

        if hor_i >= 0 and hor_j >= 0:
            NN[hor_i, hor_j] = False
            NN[hor_j, hor_i] = False
        if right_ver_i >= 0 and right_ver_j >= 0:
            NN[right_ver_i, right_ver_j] = False
            NN[right_ver_j, right_ver_i] = False
        if left_ver_i >= 0 and left_ver_j >= 0:
            
            NN[left_ver_i, left_ver_j] = False
            NN[left_ver_j, left_ver_i] = False
    return NN


def plot_bonds_for_comparison():
    fractal_dict = tile_triangle_to_implement_pbc(5)
    triangular_dict = tile_triangle_to_implement_pbc(5, False)

    fractal_lattice = fractal_dict["lattice"]
    triangular_lattice = triangular_dict["lattice"]
    hole_locations = fractal_dict["triangular_hole_locations"]

    fractal_dx, fractal_dy = calcluate_dx_dy(fractal_lattice, False)
    fractal_NN = remove_fractal_bonds(hole_locations, fractal_lattice)

    triangular_dx, triangular_dy = calcluate_dx_dy(triangular_lattice, False)
    triangular_NN, triangular_NNN = calculate_hopping(triangular_dx, triangular_dy)

    fig, ax = plt.subplots(1, 2, figsize=(10,10))
    plot_bonds(fractal_lattice, fractal_NN, ax[0], plotIndicesNames=False)
    plot_bonds(triangular_lattice, triangular_NN, ax[1], plotIndicesNames=False)
    ax[0].set_title("Fractal Lattice")
    ax[1].set_title("Triangular Lattice")
    for ax in ax.flatten():
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_spectrum_and_LDOS(H, x, y, num_states=2, cmap='inferno'):
    eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)\

    num_ticks_E, decimals_E = 5, 2
    num_ticks_n, decimals_n = 5, 0
    num_ticks_x, decimals_x = 5, 2
    num_ticks_y, decimals_y = 5, 2
    num_ticks_LDOS, decimals_LDOS = 5, 2

    num_states += 1 - num_states % 2

    all_idxs = np.arange(eigenvalues.size)
    positive = eigenvalues > 0

    positive_idxs = all_idxs[positive][np.argsort(eigenvalues[positive])][:num_states // 2]
    negative_idxs = all_idxs[~positive][np.argsort(eigenvalues[~positive])[::-1]][:num_states // 2]

    LDOS_idxs = np.concatenate((negative_idxs, positive_idxs))
    other_idxs = np.delete(all_idxs, LDOS_idxs)

    LDOS = np.sum(np.abs(eigenvectors[:, LDOS_idxs]) ** 2, axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    spectrum_aspect = (
                              (eigenvalues.size - 1) / (eigenvalues.max() - eigenvalues.min())
                      ) * (
                              (y.max() - y.min()) / (x.max() - x.min())
                      )

    axs[0].scatter(other_idxs, eigenvalues[other_idxs], c='black', s=20)
    axs[0].scatter(LDOS_idxs[:x.size], eigenvalues[LDOS_idxs[:x.size]], c='red', s=30)
    axs[0].set_xlabel('n')
    axs[0].set_ylabel('E', rotation=0)
    axs[0].set_title('Spectrum')
    axs[0].set_aspect(spectrum_aspect)

    n_ticks = np.linspace(0, eigenvalues.size - 1, num_ticks_n)
    axs[0].set_xticks(n_ticks)
    axs[0].set_xticklabels([f'{tick:.{decimals_n}f}' for tick in n_ticks])

    E_ticks = np.linspace(eigenvalues.min(), eigenvalues.max(), num_ticks_E)
    axs[0].set_yticks(E_ticks)
    axs[0].set_yticklabels([f'{tick:.{decimals_E}f}' for tick in E_ticks])


    LDOS_scatter = axs[1].scatter(x, y, c=LDOS[:x.size], cmap=cmap, s=7.5)
    axs[1].set_aspect('equal')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y', rotation=0)
    axs[1].set_title('LDOS')

    x_ticks = np.linspace(x.min(), x.max(), num_ticks_x)
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels([f'{tick:.{decimals_x}f}' for tick in x_ticks])

    y_ticks = np.linspace(y.min(), y.max(), num_ticks_y)
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels([f'{tick:.{decimals_y}f}' for tick in y_ticks])

    bbox = axs[1].get_position()
    cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
    cbar = fig.colorbar(LDOS_scatter, cax=cbar_ax)
    LDOS_ticks = np.linspace(LDOS.min(), LDOS.max(), num_ticks_LDOS)
    cbar.set_ticks(LDOS_ticks)
    cbar.set_ticklabels([f'{tick:.{decimals_LDOS}e}' for tick in LDOS_ticks])

    plt.show()


def convert_from_discrete_to_triangular_basis(x, y):
    x = x.astype(float)/2
    y = y.astype(float)/(2*np.sqrt(3))
    a1 = x + y/np.sqrt(3)
    a2 = 2*y/np.sqrt(3)
    return a1, a2


def compute_bott_index(H:np.ndarray, lattice:np.ndarray) -> float:
    eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)
    highest_lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2][-1]

    D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
    D_dagger = np.einsum('i,ij->ij', D, eigenvectors.conj().T)
    P = eigenvectors @ D_dagger


    Y, X = np.where(lattice >= 0)[:]
    X, Y = np.repeat(X, 2), np.repeat(Y, 2)

    a1, a2 = convert_from_discrete_to_triangular_basis(x, y)
    a1, a2 = x, y

    a2, a1 = np.repeat(a2, 2), np.repeat(a1, 2)
    a1_factor = np.max(a1) - np.min(a1)
    a2_factor = np.max(a2) - np.min(a2)

    Ux = np.exp(1j*2*np.pi*a1/a1_factor)
    Uy = np.exp(1j*2*np.pi*a2/a2_factor)


    print(Ux.shape, P.shape)

    UxP = np.einsum('i,ij->ij', Ux, P)
    UyP = np.einsum('i,ij->ij', Uy, P)
    Ux_daggerP = np.einsum('i,ij->ij', Ux.conj(), P)
    Uy_daggerP = np.einsum('i,ij->ij', Uy.conj(), P)

    I = np.eye(P.shape[0], dtype=np.complex128)
    Ux = np.diag(Ux)
    Uy = np.diag(Uy)
    Vx = I - P + P @ Ux @ P
    Vy = I - P + P @ Uy @ P

    A2 = Vx @ Vy @ Vx.conj().T @ Vy.conj().T
    bott2 = np.imag(np.sum(np.log(sp.linalg.eigvals(A2)))) / (2 * np.pi)

    A = np.eye(P.shape[0], dtype=np.complex128) - P + P.dot(UxP).dot(UyP).dot(Ux_daggerP).dot(Uy_daggerP)
    bott = (np.imag(np.sum(np.log(sp.linalg.eigvals(A)))) / (2 * np.pi))
    return bott, bott2



def compute_bott_index_2(H, lattice):

    eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)

    lower_band = np.argsort(eigenvalues)[:eigenvalues.size // 2]
    V = eigenvectors[:, lower_band]

    y, x = np.where(lattice >= 0)[:]

    a1, a2 = convert_from_discrete_to_triangular_basis(x, y)

    a1, a2 = np.repeat(a1, 2), np.repeat(a2, 2)

    U1 = np.exp(1j * 2 * np.pi * a1)
    U2 = np.exp(1j * 2 * np.pi * a2)

    U1_proj = np.diag((V.conj().T @ (V @ U1)))
    U2_proj = np.diag((V.conj().T @ (V @ U2)))


    A = U2_proj @ U1_proj @ U2_proj.conj().T @ U1_proj.conj().T

    eigenvaluesA = sp.linalg.eigvals(A, overwrite_a=True)
    trace_logA = np.sum(np.log(eigenvaluesA))

    bott = round(np.imag(trace_logA) / (2 * np.pi))

    return bott


def compute_chern_number(M, B_tilde, t1=1.0, B=1.0, A_tilde=1.0):
    sqrt3 = np.sqrt(3)

    def compute_d_vector(kx, ky):
        d1 = t1 * (np.sin(kx) + np.sin(kx / 2) * np.cos(sqrt3 / 2 * ky))
        d2 = t1 * (-sqrt3 * np.cos(kx / 2) * np.sin(sqrt3 / 2 * ky))
        d3 = M - 2*B*(2 - np.cos(kx) - 2 * np.cos(kx / 2) * np.cos(sqrt3 * ky / 2))
        return np.array([d1, d2, d3])

    def compute_d_tilde_vector(kx, ky):
        d1 =  A_tilde * np.sin(3 * kx / 2) * np.cos(sqrt3 / 2 * ky) * (sqrt3 / 2)
        d2 = -A_tilde * (np.sin(sqrt3 * ky) - np.cos(3 * kx / 2) * np.sin(sqrt3 / 2 * ky))
        d3 = -2*B_tilde * (3 - np.cos(sqrt3 * ky) - 2 * np.cos(3 / 2 * kx) * np.cos(sqrt3 / 2 * ky))
        return np.array([d1, d2, d3])
    
    def compute_d_exp(kx, ky):
        b1 = [1, 0]
        b2 = [1/2, sqrt3/2]
        b2tilde = [1/2, -sqrt3/2]
        k = [kx, ky]
        d1 = t1/(2j) * np.exp(1j * np.dot(k, b1)) + t1/(4j) * (np.exp(1j * np.dot(k, b2)) + np.exp(1j * np.dot(k, b2tilde)))
        d2 = -t1 * sqrt3 / (4j) * (np.exp(1j * np.dot(k, b2)) - np.exp(-1j * np.dot(k, b2tilde)))
        d3 = (M - 4*B) + B * (np.exp(1j * np.dot(k, b1)) + np.exp(1j * np.dot(k, b2)) + np.exp(1j * np.dot(k, b2tilde)))
        d1 += d1.conj().T
        d2 += d2.conj().T
        d3 += d3.conj().T
        return np.array([d1, d2, d3])

    def compute_dtilde_exp(kx, ky):
        c1 = [0, sqrt3]
        c2 = [3/2, sqrt3/2]
        c3 = [3/2, -sqrt3/2]
        k = [kx, ky]
        d1 = sqrt3*A_tilde/(8j) * (np.exp(1j*np.dot(k, c2)) + np.exp(1j*np.dot(k, c3)))
        d2 = -A_tilde/(2j) * np.exp(1j*np.dot(k, c1)) + A_tilde/(4j) * (np.exp(1j * np.dot(k, c2)) - np.exp(1j * np.dot(k, c3)))
        d3 = -6*B_tilde + B_tilde * (np.exp(1j*np.dot(k, c1)) + np.exp(1j*np.dot(k, c2)) + np.exp(1j*np.dot(k, c3)))
        d1 += d1.conj().T
        d2 += d2.conj().T
        d3 += d3.conj().T
        return np.array([d1, d2, d3])

    
    def compute_hopping_vector(kx, ky, useExp=False):
        if not useExp:
            return (compute_d_vector(kx, ky) + compute_d_tilde_vector(kx, ky)).T
        else:  
            return (compute_d_exp(kx, ky) + compute_dtilde_exp(kx, ky)).T


    def compute_d_hat(kx, ky):
        d_vector = compute_hopping_vector(kx, ky)
        norm_values = []
        for i in range(d_vector.shape[0]):
            norm_values.append(np.linalg.norm(d_vector[i]))
        norm_values = np.repeat(norm_values, 3).reshape(d_vector.shape)
        return d_vector/norm_values
    
    def compute_finite_difference(f, kx, ky, dk=1e-3):
        df_kx = (f(kx + dk, ky) - f(kx - dk, ky)) / (2 * dk)
        df_ky = (f(kx, ky + dk) - f(kx, ky - dk)) / (2 * dk)
        return np.array([df_kx, df_ky])
    
    def vectorized_cross(v1, v2):
        result = [None]*v1.shape[0]
        for i in range(v1.shape[0]):
            result[i] = np.cross(v1[i], v2[i])
        return np.array(result)
    

    def compute_berry_curvature(kx, ky):
        d_hat = compute_d_hat(kx, ky)
        d_hat_dx, d_hat_dy = compute_finite_difference(compute_d_hat, kx, ky)
        if len(d_hat_dx) > 3:
            cross_products = vectorized_cross(d_hat_dx, d_hat_dy)
        else:
            cross_products = np.cross(d_hat_dx, d_hat_dy)
        bc = np.dot(d_hat.T, cross_products)/(4*np.pi)
        return bc
    
    def scipy_integration():

        def tl_bound(ky):
            return ky/sqrt3 - 4*np.pi/3
        def tr_bound(ky):
            return -ky/sqrt3 - 4*np.pi/3

        try:
            chern_top, _ = dblquad(
                compute_berry_curvature, 0, 2*np.pi/sqrt3,
                tl_bound, tr_bound
            )
        except Exception as e:
            print(f"Error at {M}, {B_tilde}: {e}")
            chern_top = np.nan

        return chern_top*2
    
    def monte_carlo_integration(func:object, n_samples=100000):
        x_range = np.array([-4*np.pi/3, 4*np.pi/3])
        y_range = np.array([-2*np.pi/sqrt3, 2*np.pi/sqrt3])
        
        x_samples = np.random.uniform(x_range[0], x_range[1], n_samples)
        y_samples = np.random.uniform(y_range[0], y_range[1], n_samples)

        # Filter the samples that are inside the BZ:
        hexagon_vertices = 4*np.pi/3 * np.array([[[np.cos(a)], [np.sin(a)]] for a in np.arange(0, 2*np.pi, np.pi/3)])[:, :, 0]
    
        def linear_eq(x, point1, point2):
            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
            return slope * (x - point1[0]) + point1[1]
        
        def inside_polygon(x, y):
            def top_left_eq(x):
                return linear_eq(x, hexagon_vertices[2], hexagon_vertices[3])
            def top_right_eq(x):
                return linear_eq(x, hexagon_vertices[0], hexagon_vertices[1])
            def bottom_left_eq(x):
                return linear_eq(x, hexagon_vertices[3], hexagon_vertices[4])
            def bottom_right_eq(x):
                return linear_eq(x, hexagon_vertices[5], hexagon_vertices[0])
            
            def in_top_portion_of_hexagon(x, y):
                proper_y = (y >= 0) & (y <= 2*np.pi/sqrt3)
                within_sides = (y <= top_left_eq(x)) & (y <= top_right_eq(x))
                return proper_y & within_sides
            def in_bottom_portion_of_hexagon(x, y):
                proper_y = (y >= -2*np.pi/sqrt3) & (y < 0)
                within_sides = (y >= bottom_left_eq(x)) & (y >= bottom_right_eq(x))
                return proper_y & within_sides
            
            within_hexagon = in_top_portion_of_hexagon(x, y) | in_bottom_portion_of_hexagon(x, y)
            return within_hexagon
        
        within_hexagon_mask = inside_polygon(x_samples, y_samples)
        x_inside, y_inside = x_samples[within_hexagon_mask], y_samples[within_hexagon_mask]
        if False:
            plt.scatter(x_inside, y_inside, c='red', s=1)
            plt.scatter(x_samples[~within_hexagon_mask], y_samples[~within_hexagon_mask], c='blue', s=1)
            plt.show()

        n_success = x_inside.size
        f_values = func(x_inside, y_inside)

        BZ_volume = 3 * sqrt3 / 2 * (4 * np.pi / 3) ** 2
        sample_volume = ((x_range[1] - x_range[0]) * (y_range[1] - y_range[0]))


        p_success = n_success / n_samples
        p_expected = BZ_volume / sample_volume
        #print(f"Percentage of success: {p_success:.2%}")
        #print(f"Expected percentage of success: {p_expected:.2%}")
        #print(f"Ratio: {p_success/p_expected:.2f}")

        return BZ_volume/n_samples * np.sum(f_values)

    return monte_carlo_integration(compute_berry_curvature)


def compute_phase_bott(method, generation, dimensions=(50,50), M_range=(-2.0, 8.0), B_tilde_range=(0.0, 0.5), t1=1.0, t2=1.0, n_jobs=-2, show_progress=True, directory='', fileOverwrite=False):
    
    M_values = np.linspace(M_range[0], M_range[1], dimensions[1])
    B_tilde_values = np.linspace(B_tilde_range[0], B_tilde_range[1], dimensions[0])

    triangular_dict = tile_triangle_to_implement_pbc(5, False)
    triangular_lattice = triangular_dict["lattice"]



    out_filename = directory+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]}).h5"
    if os.path.exists(out_filename) and fileOverwrite == False:
        return out_filename

    def worker_function(parameters):
        B_tilde, M = parameters
        try: 
            hamiltonian_dict = compute_hamiltonian(triangular_lattice, True, M, B_tilde, t1=t1)
            H = hamiltonian_dict["H"]
            bott = compute_bott_index_2(H, triangular_lattice)
            return [M, B_tilde, bott]
        except Exception as e:
            print(f"Error: {e}")
            return [M, B_tilde, np.nan]

        
    param_values = tuple(product(B_tilde_values, M_values))

    if show_progress:
        with tqdm_joblib(tqdm(total=len(param_values), desc=f"Computing undisordered phase diagram ({method})")) as progress_bar:
            M_data, B_tilde_data, bi_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(params) for params in param_values)).T
    else:
        M_data, B_tilde_data, bi_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(params) for params in param_values)).T
    
    data = {'B_tilde': B_tilde_data,
            'M': M_data,
            'bott_index': bi_data.reshape(dimensions)}
    
    with h5py.File(out_filename, 'w') as f:
        for k, v in zip(data.keys(), data.values()):
            f.create_dataset(name=k, data=v)

    return out_filename


def compute_phase_chern(M_values, B_tilde_values, n_jobs=-2, show_progress=True, directory='', fileOverwrite=False):
    out_filename = directory+f"chern.h5"

    parameters = tuple(product(M_values, B_tilde_values))

    if os.path.exists(out_filename) and fileOverwrite == False:
        return out_filename

    def worker_function(params):
        M, B_tilde = params
        try: 
            chern = round(compute_chern_number(M, B_tilde), 2)
            return [M, B_tilde, chern]
        except Exception as e:
            print(f"Error: {e}")
            return [M, B_tilde, np.nan]

    if show_progress:
        with tqdm_joblib(tqdm(total=len(parameters), desc=f"Computing undisordered phase diagram for Chern number.")) as progress_bar:
            M_data, B_tilde_data, chern_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(params) for params in parameters)).T
    else:
        M_data, B_tilde_data, chern_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(params) for params in parameters)).T
    
    data = {"M": M_data,
            "B_tilde": B_tilde_data,
            "Chern": chern_data.reshape(len(M_values), len(B_tilde_values))}
    
    with h5py.File(out_filename, 'w') as f:
        for k, v in zip(data.keys(), data.values()):
            f.create_dataset(name=k, data=v)

    return out_filename


def plot_phase_diagram(fig, ax, 
                       X_values, Y_values, Z_values, 
                       labels:list=None, title:str=None, 
                       X_ticks=None, Y_ticks=None, X_tick_labels=None, Y_tick_labels=None,
                       cbar_ticks=None, cbar_tick_labels=None,
                       cmap='Spectral', norm=None,
                       plotColorbar=True):

    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]
    Z_range = [np.floor(np.nanmin(Z_values)), np.ceil(np.nanmax(Z_values))]

    im = ax.imshow(Z_values, extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]], 
                   origin='lower', aspect='auto', cmap=cmap, interpolation='none', 
                   rasterized=True, norm=norm)

    if title is not None:
        ax.set_title(title)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1], rotation=0)

    if X_ticks is not None:
        ax.set_xticks(X_ticks)
    if Y_ticks is not None:
        ax.set_yticks(Y_ticks)
    if X_tick_labels is not None:
        ax.set_xticklabels(X_tick_labels)
    if Y_tick_labels is not None:
        ax.set_yticklabels(Y_tick_labels)

    if plotColorbar:
        cbar = fig.colorbar(im, ax=ax)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        if cbar_tick_labels is not None:
            cbar.set_ticklabels(cbar_tick_labels)

    return fig, ax


def path_over_brillouin_zone(path:list, generation=5, pbc=True, M_values=[0.0], B_tilde=0.0, B=1.0, A_tilde=1.0, t1=1.0, includeDTilde = True):

    r = 4*np.pi/3
    angles = [np.pi/3*i for i in range(6)]
    angles.append(angles[0])
    hexagon_corners = np.array([[r*np.cos(angle), r*np.sin(angle)] for angle in angles])


    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].set_aspect('equal')
    axs[0].plot(hexagon_corners[:, 0], hexagon_corners[:, 1], c='black')

    spectral_cmap = colormaps["Spectral"]

    high_symmetry_points = {
        r"$K_+$": [4*np.pi/3, 0],
        r"$K_-$": [-4*np.pi/3, 0],
        r"$M_0$": [0, 2*np.pi/np.sqrt(3)],
        r"$M_1$": [np.pi, np.pi/np.sqrt(3)],
        r"$M_{-1}$": [-np.pi, np.pi/np.sqrt(3)],
        r"$\Gamma$": [0, 0]
    }

    for point, coords in high_symmetry_points.items():
        if "M" in point:
            color = 'red'
        elif "K" in point:
            color = 'black'
        elif "Gamma" in point:
            color = 'orange'
        axs[0].scatter(*coords, zorder=3, c=color)
        offset = 0.5  # Offset to place the text label outside the hexagon
        if "Gamma" not in point:
            axs[0].text(coords[0] * (1 + offset / r), coords[1] * (1 + offset / r), f" {point}", 
                fontsize=10, ha='center', va='center', color=color, fontweight='bold', clip_on=True)
        else:
            axs[0].text(2*offset/r, 2*offset/r, f" {point}", 
                fontsize=10, ha='center', va='center', color=color, fontweight='bold', clip_on=True)

        # Extend plot bounds by offset in each direction
        x_min, x_max = hexagon_corners[:, 0].min(), hexagon_corners[:, 0].max()
        y_min, y_max = hexagon_corners[:, 1].min(), hexagon_corners[:, 1].max()
        offset *= 2
        axs[0].set_xlim(x_min - offset, x_max + offset)
        axs[0].set_ylim(y_min - offset, y_max + offset)


    arrow_length = r/4
    arrow_label_spacing = 0.25
    
    # Plot vertical arrow labeled ky
    axs[0].arrow(0, 0, 0, arrow_length, head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=2, length_includes_head=True)
    axs[0].text(arrow_label_spacing, arrow_length + arrow_label_spacing, r"$k_y$", fontsize=10, ha='center', va='center', color='black', fontweight='bold')

    # Plot horizontal arrow labeled kx
    axs[0].arrow(0, 0, arrow_length, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=2, length_includes_head=True)
    axs[0].text(arrow_length + arrow_label_spacing, arrow_label_spacing, r"$k_x$", fontsize=10, ha='center', va='center', color='black', fontweight='bold')

    hexagon = plt.Polygon(hexagon_corners, closed=True, color=spectral_cmap(0.5), alpha=0.5, zorder=0)
    axs[0].add_patch(hexagon)

    axs[0].set_title("Brillouin Zone of Triangular Lattice")

    # Plot the path over the Brillouin zone
    reformatted_names = []
    for p in path:
        if p in high_symmetry_points.keys():
            reformatted_names.append(p)
        else:
            if ("M1" in p):
                reformatted_names.append(r"$M_1$")
            elif ("M-1" in p):
                reformatted_names.append(r"$M_{-1}$")
            elif ("M0" in p):
                reformatted_names.append(r"$M_0$")
            elif ("K-" in p):
                reformatted_names.append(r"$K_-$")
            elif ("K+" in p):
                reformatted_names.append(r"$K_+$")
            elif ("G" in p):
                reformatted_names.append(r"$\Gamma$")    

    path_values = np.array([high_symmetry_points[point] for point in reformatted_names]).T
    #axs[0].plot(path_values[0], path_values[1], c='blue', zorder=3, ls=':')

    triangular_dict = tile_triangle_to_implement_pbc(generation, False)
    lattice = triangular_dict["lattice"]
    
    dx, dy = calcluate_dx_dy(lattice, pbc)
    hopping_dict = calculate_hopping(dx, dy)
    b_vectors = hopping_dict["b_hopping_vectors"]
    c_vectors = hopping_dict["c_hopping_vectors"]

    def compute_d_vector(kx, ky, M):
        d1 = t1 * (np.sin(kx) + 2*np.sin(kx/2) * np.cos(np.pi/3) * np.cos(np.sqrt(3)/2*ky))
        d2 = t1 * (-2*np.cos(kx/2) * np.sin(np.pi/3) * np.sin(np.sqrt(3)/2*ky))
        d3 = M - 2*B*(2 - np.cos(kx) - 2 * np.cos(kx/2) * np.cos(np.sqrt(3)*ky/2))
        return np.array([d1, d2, d3])
    
    def compute_d_tilde_vector(kx, ky):
        d1tilde = A_tilde * np.sin(kx*3/2)*np.sqrt(3)/2*np.cos(np.sqrt(3)*ky/2)
        d2tilde = -A_tilde * (np.sin(np.sqrt(3)*ky) - 2*np.cos(3*kx/2)*np.sin(np.pi/6)*np.sin(np.sqrt(3)*ky/2))
        d3tilde = -2*B_tilde* (3 - np.cos(np.sqrt(3)* ky) - 2*np.cos(3*ky/2) * np.cos(np.sqrt(3)*ky/2))
        return np.sqrt(d1tilde.real**2 + d2tilde.real**2 + d3tilde.real**2)

    def total_energy_of_momenta(kx, ky, M):

        if includeDTilde:
            d = compute_d_vector(kx, ky, M) + compute_d_tilde_vector(kx, ky)
        else:
            d = compute_d_vector(kx, ky, M)
        return np.linalg.norm(d)

    x_paths_array = []
    y_paths_array = []
    num_points_per_line = 200
    for i in range(path_values.shape[1] - 1):
        k1 = path_values[:, i]
        k2 = path_values[:, i + 1]

        num_arrows = 5  # Number of arrowheads along the arrow
        for j in range(num_arrows):
            fraction = j / num_arrows
            arrow_x = k1[0] + fraction * (k2[0] - k1[0])
            arrow_y = k1[1] + fraction * (k2[1] - k1[1])
            if fraction < 1.0:  # Ensure arrows do not go past the destination point
                axs[0].arrow(arrow_x, arrow_y, (k2[0] - k1[0]) / num_arrows, (k2[1] - k1[1]) / num_arrows, 
                     ls=':', head_width=0.1, head_length=0.1, fc='blue', ec='blue', zorder=3, 
                     length_includes_head=True, alpha=0.5)

        x_path = np.linspace(k1[0], k2[0], num_points_per_line)
        y_path = np.linspace(k1[1], k2[1], num_points_per_line)
        x_paths_array.append(x_path)
        y_paths_array.append(y_path)
        
    x = np.concatenate(x_paths_array)
    y = np.concatenate(y_paths_array)
    for M in M_values:
        energies = np.array([total_energy_of_momenta(k[0], k[1], M) for k in zip(x, y)])
        axs[1].plot(energies, zorder=3, label=f"M={M}")


    axs2_xticks = [i*num_points_per_line for i in range(len(path))]
    axs[1].set_xticks(axs2_xticks, labels=reformatted_names)
    #axs[1].set_title(f"Energy Spectrum along Path: M = {M}, B_tilde = {B_tilde}")
    axs[1].set_title(f"Energy along Path: B_tilde = {B_tilde}")

    for i in range(len(path)):
        axs[1].axvline(x=i*num_points_per_line, color='black', linestyle='--', linewidth=0.8, zorder=1)
    axs[1].axhline(y=0.0, color='black', linestyle='--', linewidth=0.8, zorder=1)
    axs[1].legend()
    plt.tight_layout()
    plt.show()



def plot_band(M, B_tilde, t1=1.0, B=1.0, A_tilde=1.0):
    n_samples = 5000
    sqrt3 = np.sqrt(3)
    x_range = np.array([-4*np.pi/3, 4*np.pi/3])
    y_range = np.array([-2*np.pi/sqrt3, 2*np.pi/sqrt3])
    
    x_samples = np.random.uniform(x_range[0], x_range[1], n_samples)
    y_samples = np.random.uniform(y_range[0], y_range[1], n_samples)

    # Filter the samples that are inside the BZ:
    hexagon_vertices = 4*np.pi/3 * np.array([[[np.cos(a)], [np.sin(a)]] for a in np.arange(0, 2*np.pi, np.pi/3)])[:, :, 0]

    def linear_eq(x, point1, point2):
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        return slope * (x - point1[0]) + point1[1]
    
    def inside_polygon(x, y):
        def top_left_eq(x):
            return linear_eq(x, hexagon_vertices[2], hexagon_vertices[3])
        def top_right_eq(x):
            return linear_eq(x, hexagon_vertices[0], hexagon_vertices[1])
        def bottom_left_eq(x):
            return linear_eq(x, hexagon_vertices[3], hexagon_vertices[4])
        def bottom_right_eq(x):
            return linear_eq(x, hexagon_vertices[5], hexagon_vertices[0])
        
        def in_top_portion_of_hexagon(x, y):
            proper_y = (y >= 0) & (y <= 2*np.pi/sqrt3)
            within_sides = (y <= top_left_eq(x)) & (y <= top_right_eq(x))
            return proper_y & within_sides
        def in_bottom_portion_of_hexagon(x, y):
            proper_y = (y >= -2*np.pi/sqrt3) & (y < 0)
            within_sides = (y >= bottom_left_eq(x)) & (y >= bottom_right_eq(x))
            return proper_y & within_sides
        
        within_hexagon = in_top_portion_of_hexagon(x, y) | in_bottom_portion_of_hexagon(x, y)
        return within_hexagon
    
    within_hexagon_mask = inside_polygon(x_samples, y_samples)
    x_inside = x_samples[within_hexagon_mask]
    y_inside = y_samples[within_hexagon_mask]

    def compute_d_vector(kx, ky):
        d1 = t1 * (np.sin(kx) + np.sin(kx / 2) * np.cos(sqrt3 / 2 * ky))
        d2 = t1 * (-sqrt3 * np.cos(kx / 2) * np.sin(sqrt3 / 2 * ky))
        d3 = M - 2*B*(2 - np.cos(kx) - 2 * np.cos(kx / 2) * np.cos(sqrt3 * ky / 2))
        return np.array([d1, d2, d3])

    def compute_d_tilde_vector(kx, ky):
        d1 =  A_tilde * np.sin(3 * kx / 2) * np.cos(sqrt3 / 2 * ky) * (sqrt3 / 2)
        d2 = -A_tilde * (np.sin(sqrt3 * ky) - np.cos(3 * kx / 2) * np.sin(sqrt3 / 2 * ky))
        d3 = -2*B_tilde * (3 - np.cos(sqrt3 * ky) - 2 * np.cos(3 / 2 * kx) * np.cos(sqrt3 / 2 * ky))
        return np.array([d1, d2, d3])

    def compute_hopping_vector(kx, ky, useExp=False):
        if not useExp:
            return (compute_d_vector(kx, ky) + compute_d_tilde_vector(kx, ky)).T

    def compute_d_hat(kx, ky):
        d_vector = compute_hopping_vector(kx, ky)
        norm_values = []
        for i in range(d_vector.shape[0]):
            norm_values.append(np.linalg.norm(d_vector[i]))
        norm_values = np.repeat(norm_values, 3).reshape(d_vector.shape)
        return d_vector/norm_values
    
    def compute_finite_difference(f, kx, ky, dk=1e-3):
        df_kx = (f(kx + dk, ky) - f(kx - dk, ky)) / (2 * dk)
        df_ky = (f(kx, ky + dk) - f(kx, ky - dk)) / (2 * dk)
        return np.array([df_kx, df_ky])
    
    def vectorized_cross(v1, v2):
        result = [None]*v1.shape[0]
        for i in range(v1.shape[0]):
            result[i] = np.cross(v1[i], v2[i])
        return np.array(result)
    
    def compute_berry_curvature(kx, ky):
        d_hat = compute_d_hat(kx, ky)
        d_hat_dx, d_hat_dy = compute_finite_difference(compute_d_hat, kx, ky)
        if len(d_hat_dx) > 3:
            cross_products = vectorized_cross(d_hat_dx, d_hat_dy)
        else:
            cross_products = np.cross(d_hat_dx, d_hat_dy)
        bc = np.dot(d_hat.T, cross_products)/(4*np.pi)
        return bc

    d_vector = compute_hopping_vector(x_inside, y_inside)
    d1_2 = np.power(d_vector[:, 0], 2)
    d2_2 = np.power(d_vector[:, 1], 2)
    d3 = d_vector[:, 2]

    eig1 = d3 + np.sqrt(d1_2 + d2_2)
    eig2 = d3 - np.sqrt(d1_2 + d2_2)

    dmag = np.sqrt(d1_2 + d2_2 + d3**2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #ax.scatter(x_inside, y_inside, eig1, c='green', s=5)
    #ax.scatter(x_inside, y_inside, eig2, c='blue', s=5)
    ax.scatter(x_inside, y_inside, dmag, c='red', s=5)

    ax.set_xlabel('kx')
    ax.set_ylabel('ky')

    plt.show()


def plot_band_2(M, B_tilde, t1=1.0, B=1.0, A_tilde=1.0):
    triangular_dict = tile_triangle_to_implement_pbc(4, False)
    triangular_lattice = triangular_dict["lattice"]
    h_dict = compute_hamiltonian(triangular_lattice, True, M, B_tilde)
    H = h_dict["H"]



if __name__ == "__main__":
    #plot_relative_distances(6, True, rel_origin=(.5,.5), doFractal=False)

    def find_btilde(M):
        return M/8 - 3/4

    #plot_band(8.0, 0.5)
    #path_over_brillouin_zone(["G", "M0", "K+", "G"], M_values = [-2.0, 6.0, 7.0], B_tilde = 0.0, pbc=True, includeDTilde=False)
    #path_over_brillouin_zone(["G", "M0", "K+", "G"], M_values = [-2.0, 6.0, 7.0], B_tilde = 0.5, pbc=True, includeDTilde=True)
    


    # Profile single chern computation
    if False:
        profiler = cProfile.Profile()
        profiler.enable()

        c = -compute_chern_number(6.5, 1/16)
        print(c)

        profiler.disable()
        stats = pstats.Stats(profiler) 	
        stats = stats.sort_stats('cumtime')
        #stats.print_stats(10)

    # Chern Phase Diagram
    if True:
        #fname = compute_phase_chern(np.linspace(-2.0, 8.0, 16), np.linspace(0.0, 1.0, 16), n_jobs=4, directory='./Triangle/PhaseDiagrams/', fileOverwrite=False)
        fname = './Triangle/PhaseDiagrams/chern.h5'
        with h5py.File(fname) as f:
            M = f['M'][:]
            chern = f['chern'][:]
            B_tilde = f["B_tilde"][:]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_phase_diagram(fig, ax, M, B_tilde, chern, labels=['M', 'B_tilde'], title='Chern Number Phase Diagram', cmap='Spectral', plotColorbar=True)
        plt.show()

        

    # Plot triangular coordinates as scatter plot
    if False:   
        triangular_dict = tile_triangle_to_implement_pbc(5, False)
        triangular_lattice = triangular_dict["lattice"]
        y, x = np.where(triangular_lattice >= 0)
        x = x.astype(float)/2
        y = y.astype(float)/(2*np.sqrt(3))

        a1 = x + y/np.sqrt(3)
        a2 = 2*y/np.sqrt(3)

        def convert_from_discrete_to_triangular_basis(x, y):
            x = x.astype(float)/2
            y = y.astype(float)/(2*np.sqrt(3))
            a1 = x + y/np.sqrt(3)
            a2 = 2*y/np.sqrt(3)
            return a1, a2

        plt.scatter(x, y)
        plt.scatter(a1, a2)
        plt.show()

    # Parallel phase diagram computation
    if False:
        fname = compute_phase_bott('triangular', 3, (5, 5), n_jobs=4, directory='./Triangle/PhaseDiagrams/')
        with h5py.File(fname) as f:
            print(f.keys())
            M = f['M'][:]
            B_tilde = f['B_tilde'][:]
            bi = f['bott_index'][:]


        fname = './Triangle/PhaseDiagrams/triangular_g4_(7_by_7).h5'
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_phase_diagram(fig, ax, M, B_tilde, bi, labels=['M', 'B_tilde'], title='Triangular Lattice Bott Index', cmap='Spectral', plotColorbar=True)
        plt.show()
              
    # Single phase computation; plot spectrum 
    if False:
        generation = 4
        triangular_dict = tile_triangle_to_implement_pbc(generation, False)
        triangular_lattice = triangular_dict["lattice"]
        y, x = np.where(triangular_lattice >= 0)[:]

        M, B_tilde = 6.0, 0.125
        hamiltonian_dict = compute_hamiltonian(triangular_lattice, True, M, B_tilde)
        H = hamiltonian_dict["H"]
        plot_spectrum_and_LDOS(H, x, y)
        bott = compute_bott_index(H, triangular_lattice)
        print(bott)


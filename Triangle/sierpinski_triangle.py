import numpy as np
from matplotlib import pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    NNN = ((abs_dy == 6) & ((abs_dx == 0) | (abs_dx == 2))) | ((abs_dy == 3) & (abs_dx == 3)) | ((abs_dy == 0) & (abs_dx == 4))
    return NN, NNN


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


def wrapper(generation, pbc, plotBonds=False):
    triangular_lattice = compute_triangular_lattice(generation)
    fractal_dict = compute_sierpinski_triangle(generation)
    fractal_lattice, triangular_hole_locations = fractal_dict["lattice"], fractal_dict["triangular_hole_locations"]

    frac_dx, frac_dy = calcluate_dx_dy(fractal_lattice, pbc)
    frac_NN = remove_fractal_bonds(triangular_hole_locations, fractal_lattice)

    tri_dx, tri_dy = calcluate_dx_dy(triangular_lattice, pbc)
    tri_NN, tri_NNN = calculate_hopping(tri_dx, tri_dy)

    if plotBonds and pbc == False:
        fig, axs = plt.subplots(1, 2, figsize=(10,10))
        plot_bonds(fractal_lattice, frac_NN, axs[1])
        plot_bonds(triangular_lattice, tri_NN, axs[0])
        axs[0].set_title("Triangular Lattice")
        axs[1].set_title("Fractal Lattice")

        for ax in axs.flatten():
            ax.set_aspect('equal')
        plt.show()


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

if __name__ == "__main__":
    plot_relative_distances(6, True, rel_origin=(.5,.5), doFractal=True)
    if False:
        fractal_dict = tile_triangle_to_implement_pbc(5)
        triangular_dict = tile_triangle_to_implement_pbc(5, False)

        fractal_lattice = fractal_dict["lattice"]
        triangular_lattice = triangular_dict["lattice"]
        hole_locations = fractal_dict["triangular_hole_locations"]

        fractal_dx, fractal_dy = calcluate_dx_dy(fractal_lattice, True)
        fractal_NN, fractal_NNN = calculate_hopping(fractal_dx, fractal_dy)
        fractal_NN = remove_fractal_bonds(hole_locations, fractal_lattice)

        triangular_dx, triangular_dy = calcluate_dx_dy(triangular_lattice, True)
        triangular_NN, triangular_NNN = calculate_hopping(triangular_dx, triangular_dy)


        index = np.random.randint(0, fractal_NN.shape[0])

        fig, ax = plt.subplots(figsize=(10, 10))
        y, x = np.where(fractal_lattice >= 0)
        scatter = ax.scatter(x, y, c=fractal_NNN[index], cmap='viridis', s=100)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='NNN Value')
        plt.title('Fractal Lattice with NNN Values')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

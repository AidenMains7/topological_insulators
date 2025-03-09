import numpy as np
from matplotlib import pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable




def sierpinski_triangle(generation):
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
    
    xmean, ymean = np.mean(coordinates[0]), np.mean(coordinates[1])
    coordinates[0] -= xmean
    coordinates[1] -= ymean
    if fractal_dict["triangular_hole_locations"] is None:
        return {"lattice_points": coordinates, "triangular_hole_locations": None}
    hole_locations = fractal_dict["triangular_hole_locations"]
    hole_locations[0] -= xmean
    hole_locations[1] -= ymean
    return {"lattice_points": coordinates, "triangular_hole_locations": hole_locations}


def tiled_sierpinski(generation):
    fractal = sierpinski_triangle(generation)
    x_max = np.max(fractal[0])
    flipped_fractal = fractal.copy()
    flipped_fractal[1] = np.max(fractal[1])-flipped_fractal[1]
    flipped_fractal[1] -= np.min(flipped_fractal[1]) - np.min(fractal[1])
    flipped_fractal[0] += x_max

    tiled = np.hstack((fractal, flipped_fractal))
    tiled[1] -= np.min(tiled[1])
    tiled[0] -= np.min(tiled[0])
    tiled = np.unique(tiled, axis=1)
    sorted_indices = np.lexsort((tiled[1], tiled[0]))
    tiled = tiled[:, sorted_indices]
    return tiled

def coordinates_to_lattice(coordinates):
    new_coords = coordinates.copy()
    new_coords[0] *= 2/np.sqrt(3)
    new_coords[1] *= 2
    new_coords[0] -= np.min(new_coords[0])
    new_coords[1] -= np.min(new_coords[1])
    new_coords = np.round(new_coords).astype(int)
    lattice = np.full((np.max(new_coords[1])-np.min(new_coords[1])+1, np.max(new_coords[0])-np.min(new_coords[0])+1), -1)
    lattice[new_coords[1], new_coords[0]] = np.arange(new_coords.shape[1])
    return lattice

def calcluate_dx_dy(coordinates, pbc):
    if not pbc:
        x, y = coordinates
        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
    else:
        x, y = coordinates
        if np.min(x) < 0:
            x -= np.min(x)
        if np.min(y) < 0:
            y -= np.min(y)

        x_range = np.max(x)
        y_range = np.max(y)

        displacements = {
            "center": np.array([0, 0]).T,
            "top_left": np.array([-x_range/3, y_range]).T,
            "left": np.array([-x_range*2/3, 0]).T,
            "bottom_left": np.array([-x_range*3/3, -y_range]).T,
            "bottom": np.array([-x_range/3, -y_range]).T,
            "bottom_right": np.array([x_range/3, -y_range]).T,
            "right": np.array([x_range*2/3, 0]).T,
            "top_right": np.array([x_range*3/3, y_range]).T,
            "top": np.array([x_range/3, y_range]).T
        }

        all_x_shifts = np.empty((1, coordinates.shape[1], 9))
        all_y_shifts = np.empty((1, coordinates.shape[1], 9))
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

def plot_relative_distances(n, PBC, rel_origin=(0., 0.), abs_dist=True, fractal=False):

    coords = tiled_sierpinski(n)
    x, y = coords
    delta_x_discrete, delta_y_discrete = calcluate_dx_dy(coords, PBC)


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
 

def to_discrete_lattice_coords(coords):
    coords[0] *= 2/np.sqrt(3)
    coords[1] *= 2
    coords[0] -= np.min(coords[0])
    coords[1] -= np.min(coords[1])
    coords = np.round(coords).astype(int)
    return coords

def calculate_hopping(dx, dy):
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    NN = ((abs_dx == 2) & (abs_dy == 0)) | ((abs_dx == 1) & (abs_dy == 3))
    return NN

def plot_bonds(generation, tiled=False):
    if tiled:
        coords = to_discrete_lattice_coords(tiled_sierpinski(generation))
    else:
        coords = to_discrete_lattice_coords(sierpinski_triangle(generation))
    x, y = coords
    dx, dy = calcluate_dx_dy(coords, False)
    NN = calculate_hopping(dx, dy)
    
    i, j = np.where(NN)
    upper_triangle = j > i
    i, j = i[upper_triangle], j[upper_triangle]

    plt.plot([x[i], x[j]], [y[i], y[j]], c=(0., 0., 0., 1.), zorder=1)
    plt.scatter(x, y)
    plt.show()

def get_triangle_from_bottom_vertex(x, y, n, doDiscrete=False):
    v1 = np.zeros((2, x.size))
    v1[0] = -(2**n)*np.sqrt(3)/2 + x
    v1[1] = (2**n)*3/2 + y
    v2 = np.zeros((2, x.size))
    v2[0] = (2**n)*np.sqrt(3)/2 + x
    v2[1] = (2**n)*3/2 + y
    v3 = np.zeros((2, x.size))
    v3[0] = x
    v3[1] = y
    if doDiscrete:
        v1 *= 2/np.sqrt(3)
        v2 *= 2/np.sqrt(3)
        v3 *= 2/np.sqrt(3)
    
    vals = np.empty((2, x.size, 3))
    for i, v in enumerate([v1, v2, v3]):
        vals[:, :, i] = v

    return vals
        
    


if __name__ == "__main__":
    #plot_bonds(3)
    if True:
        triangle_dict = sierpinski_triangle(4)
        hole_locations = triangle_dict["triangular_hole_locations"]
        coords = triangle_dict["lattice_points"]
        x_frac, y_frac = coords
        x, y, n = hole_locations
        print(x)
        tri = get_triangle_from_bottom_vertex(x, y, n, False)
        print(tri.shape)
        print(tri)
        plt.scatter(tri[0], tri[1], alpha=0.2, c='r')

        plt.scatter(x_frac, y_frac, alpha=0.1, c='k')
        for i in range(int(n.max()) + 1):
            mask = (n == i)
            plt.scatter(x[mask], y[mask], label=f'n={i}')
        plt.legend()
        plt.show()

    if False:
        for i in range(10):
            if i == 0:
                pass
            triangle_dict = sierpinski_triangle(i)
            coords = triangle_dict["lattice_points"]
            x_range = np.max(coords[0]) - np.min(coords[0])
            print(f"Generation {i}: x range = {x_range*2/np.sqrt(3):.0f}")



if False:
    coords = to_discrete_lattice_coords(sierpinski_triangle(4))
    x, y = coords
    dx, dy = calcluate_dx_dy(coords, False)
    NN = calculate_hopping(dx, dy)

    i, j = np.where(NN)
    upper_triangle = j > i
    i, j = i[upper_triangle], j[upper_triangle]
    plt.plot([x[i], x[j]], [y[i], y[j]], c=(0., 0., 0., 1.), zorder=1)

    plt.scatter(x, y)

    plt.show()

    


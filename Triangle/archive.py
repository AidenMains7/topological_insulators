import numpy as np
from matplotlib import pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ----------------

def find_triangle_lines(coordinates):
    x, y = coordinates
    x_min, x_max, y_min = np.min(x), np.max(x), np.min(y)

    idx3 = np.argwhere(coordinates[1] == y_min)[0]
    idx1 = np.argwhere(coordinates[0] == x_min)[0]
    idx2 = np.argwhere(coordinates[0] == x_max)[0]

    v1 = coordinates[:, idx1].flatten()
    v2 = coordinates[:, idx2].flatten()
    v3 = coordinates[:, idx3].flatten()

    slopes = {
        "left": (v1[1] - v3[1]) / (v1[0] - v3[0]),
        "right": (v2[1] - v3[1]) / (v2[0] - v3[0]),
        "top": (v2[1] - v1[1]) / (v2[0] - v1[0])
    }

    lines = {
        "left": slopes["left"] * (x - v3[0]) + v3[1],
        "right": slopes["right"] * (x - v3[0]) + v3[1],
        "top": slopes["top"] * (x - v1[0]) + v1[1]
    }

    below_top = np.where(y < lines["top"], True, False)
    above_left = np.where(y > lines["left"], True, False)
    above_right = np.where(y > lines["right"], True, False)
    return below_top & above_left & above_right


def find_triangle_lines2(coordinates):
            if fractal_dict["hole_boundary_points"] is not None:
                hole_boundary_points = fractal_dict["hole_boundary_points"]
                shifted_hole_boundary_points = np.zeros((2, hole_boundary_points.shape[1]*3))
                for i, shift in enumerate(shifts.values()):
                    shifted_hole_boundary_points[:, i::3] = hole_boundary_points + shift

            larger_hole_points = smaller
            larger_hole_points[1] *= -1
            larger_hole_points[1] -= np.min(larger_hole_points[1]) - np.min(fractal[1])
            in_boundary_idxs = find_triangle_lines(larger_hole_points)

            all_hole_points = np.append(larger_hole_points[:, ~in_boundary_idxs], shifted_hole_boundary_points, axis=1) if fractal_dict["hole_boundary_points"] is not None else larger_hole_points


def finite_difference_2D(func, kx, ky, M, B_tilde, B, t1, t2, tol=1e-4, *args, **kwargs):
    d_func_x = (func(kx + tol, ky, M, B_tilde, B, t1, t2) - func(kx - tol, ky, M, B_tilde, B, t1, t2)) / (2 * tol)
    d_func_y = (func(kx, ky + tol, M, B_tilde, B, t1, t2) - func(kx, ky - tol, M, B_tilde, B, t1, t2)) / (2 * tol)
    return d_func_x, d_func_y

# ---------------------------
#   triangle_geometry.py

def highlight_NNN(generation):
    lattice_hopping_dict = fractal_wrapper(generation, False)
    fractal_lattice = lattice_hopping_dict["fractal_lattice"]
    b1, b2, b2_tilde, c1, c2, c3 = lattice_hopping_dict["fractal_hopping_masks"].values()
    triangular_lattice = lattice_hopping_dict["triangular_lattice"]

    y, x = np.where(triangular_lattice >= 0)[:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    arrs = [c1, c2, c3]
    colors = ['b', 'orange','r']
    linestyles = ['--','-.', ':']
    for i, arr in enumerate(arrs):
        i_idx, j_idx = np.where(arr)
        valid_indices = (i_idx < len(x)) & (j_idx < len(x))  # Ensure indices are valid
        i_idx, j_idx = i_idx[valid_indices], j_idx[valid_indices]
        ax.plot([x[i_idx], x[j_idx]], [y[i_idx], y[j_idx]], c = colors[i], ls=linestyles[i], zorder=2)

    for arr in [b1, b2, b2_tilde]:
        i_idx, j_idx = np.where(arr)
        ax.plot([x[i_idx], x[j_idx]], [y[i_idx], y[j_idx]], c = 'k', zorder=1)


    if False:
        yidx, xidx = np.argwhere(fractal_lattice >= 0).T
        for yidx, xidx in zip(yidx, xidx):
            ax.text(xidx, yidx, str(fractal_lattice[yidx, xidx]), fontsize=12, ha='center', va='top', c='r')

    ax.scatter(x, y, c='k', zorder=0, s=4)
    plt.show()


def profile():
    with cProfile.Profile() as pr:
        generation = 6
        lattice_hopping_dict = fractal_wrapper(generation, False)
        fractal_lattice = lattice_hopping_dict["fractal_lattice"]
        b1, b2, b2_tilde, c1, c2, c3 = lattice_hopping_dict["fractal_hopping_masks"].values()

    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime').print_stats(10)


def average_speed(func, n_trials):
    times = []
    for _ in range(n_trials):
        start = time()
        func()
        times.append(time() - start)
    return np.mean(times), np.std(times)


def compare_complex_casting():
    arr = np.random.randint(0, 2, (1000, 1000)).astype(bool)

    def multiply():
        arr2 = arr * 1. + 1.j
        return arr2
    
    def convert_then_multiply():
        arr2 = arr.astype(np.complex128) * 1. + 1.j
        return arr2

    m1, s1 = average_speed(multiply, 100)
    m2, s2 = average_speed(convert_then_multiply, 100)

    print(f"multiply: {m1:.4f} ± {s1:.4f}")
    print(f"convert_then_multiply: {m2:.4f} ± {s2:.4f}")

    print((arr * 1. + 1.j).dtype)

# --------------------------
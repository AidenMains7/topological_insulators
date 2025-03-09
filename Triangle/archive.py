import numpy as np
from matplotlib import pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable





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

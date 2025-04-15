import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
from itertools import product
import os, h5py
from matplotlib.colors import ListedColormap, BoundaryNorm


def compute_square_d_vector(kx, ky, M, B_tilde, B, t1, t2):
    """
    kx, ky may be numpy arrays or floats.
    
    If kx and ky are numpy arrays, then the function returns a 2D array of shape (3, len(kx)).
    """
    d1 = t1 * np.sin(kx) + t2 * np.cos(kx) * np.sin(ky)
    d2 = t1 * np.sin(ky) + t2 * np.cos(ky) * np.sin(kx)
    d3 = M - 4 * B - 4 * B_tilde + 2 * B * (np.cos(kx) + np.cos(ky)) + 4 * B_tilde * np.cos(kx) * np.cos(ky)

    return np.array([d1, d2, d3])


def compute_triangular_d_vector(kx, ky, M, B_tilde, B, t1, A_tilde):
    sqrt3 = np.sqrt(3)

    #print(f"A_tilde = {A_tilde}")

    d1 = t1 * (np.sin(kx) + np.sin(kx / 2) * np.cos(np.pi/3) * np.cos(sqrt3 / 2 * ky))
    d2 = t1 * (-sqrt3 * np.cos(kx / 2) * np.sin(np.pi/3) * np.sin(sqrt3 / 2 * ky))
    d3 = M - 2 * B * (2 - np.cos(kx) - 2 * np.cos(kx / 2) * np.cos(sqrt3 * ky / 2))

    dtilde1 =  A_tilde * np.sin(3 * kx / 2) * np.cos(np.pi/6) * np.cos(sqrt3 / 2 * ky)
    dtilde2 = -A_tilde * (np.sin(sqrt3 * ky) - 2 * np.cos(3 * kx / 2) * np.sin(np.pi/6) * np.sin(sqrt3 / 2 * ky))
    dtilde3 = -2 * B_tilde * (3 - np.cos(sqrt3 * ky) - 2 * np.cos(3 / 2 * kx) * np.cos(sqrt3 / 2 * ky))
    
    d_vector = np.array([d1, d2, d3])
    dtilde_vector = np.array([dtilde1, dtilde2, dtilde3])
    return d_vector + dtilde_vector


def compute_unit_vector(vector):
    if vector.ndim == 1:
        vector = vector.reshape((3, 1))
    
    norms = np.linalg.norm(vector, axis=0)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    unit_vector = vector / norms
    unit_vector = np.clip(unit_vector, -1e2, 1e2)  # Ensure unit_vector is not too large
    return unit_vector
    

def compute_d_vector(kx, ky, M, B_tilde, B, t1, t2, A_tilde, doTriangular=False):
    if doTriangular:
        return compute_triangular_d_vector(kx, ky, M, B_tilde, B, t1, A_tilde)
    else:
        return compute_square_d_vector(kx, ky, M, B_tilde, B, t1, t2)


def compute_d_hat_and_derivatives(kx, ky, M, B_tilde, B, t1, t2, A_tilde, doTriangular=False):
    dkx = 1e-5

    d = compute_d_vector(kx, ky, M, B_tilde, B, t1, t2, A_tilde, doTriangular)
    d_norm = np.linalg.norm(d, axis=0, keepdims=True)
    d_dkx = (compute_d_vector(kx + dkx, ky, M, B_tilde, B, t1, t2, A_tilde, doTriangular) - 
            compute_d_vector(kx - dkx, ky, M, B_tilde, B, t1, t2, A_tilde, doTriangular)) / (2 * dkx)
    d_dky = (compute_d_vector(kx, ky + dkx, M, B_tilde, B, t1, t2, A_tilde, doTriangular) - 
            compute_d_vector(kx, ky - dkx, M, B_tilde, B, t1, t2, A_tilde, doTriangular)) / (2 * dkx)

    #epsilon = 1e-6
    #d_dkx = np.where(np.abs(d_dkx) < epsilon, 0, d_dkx)  # Avoid numerical issues
    #d_dky = np.where(np.abs(d_dky) < epsilon, 0, d_dky)  # Avoid numerical issues
    
    # Quotient Rule
    d_norm_safe = np.where(d_norm == 0, 1, d_norm)  # Avoid division by zero by replacing zero norms with 1
    #d_norm_safe = np.clip(d_norm, 1e-10, None)  # Clip values to avoid excessively large norms
    
    d_hat = d / d_norm_safe
    d_hat_dkx = (d_dkx / d_norm_safe) - (d_hat * np.einsum("ij,ij->j", d, d_dkx) / (d_norm_safe**2))
    d_hat_dky = (d_dky / d_norm_safe) - (d_hat * np.einsum("ij,ij->j", d, d_dky) / (d_norm_safe**2))

    return d_hat, d_hat_dkx, d_hat_dky


def compute_berry_curvature(kx, ky, M, B_tilde, B, t1, t2, A_tilde, doTriangular=False):
    d_hat, d_hat_dx, d_hat_dy = compute_d_hat_and_derivatives(kx, ky, M, B_tilde, B, t1, t2, A_tilde, doTriangular)

    dx_cross_dy = np.cross(d_hat_dx, d_hat_dy, axis=0)

    berry_curvature = np.einsum("ij,ij->j", d_hat, dx_cross_dy)
    return berry_curvature / (4 * np.pi)


def compute_square_brillouin_zone(resolution=(100, 100)):
    kx_limits = (-np.pi, np.pi)
    ky_limits = (-np.pi, np.pi)

    kx = np.linspace(kx_limits[0], kx_limits[1], resolution[0])
    ky = np.linspace(ky_limits[0], ky_limits[1], resolution[1])

    kx, ky = np.meshgrid(kx, ky)
    return kx.flatten(), ky.flatten()


def compute_triangular_brillouin_zone(resolution=(501, 501), doRotate=False):
    R = 4 * np.pi / 3

    angles = [i * np.pi / 3 for i in range(6)]
    hexagon_vertices = R * np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
    
    kx, ky = np.meshgrid(np.linspace(-R, R, resolution[0]), np.linspace(-R * np.sqrt(3) / 2, R * np.sqrt(3) / 2, resolution[1]))
    kx, ky = kx.flatten(), ky.flatten()
    # Check if points are inside the hexagon
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
            proper_y = (y >= 0) & (y <= 2 * np.pi / np.sqrt(3))
            within_sides = (y <= top_left_eq(x)) & (y <= top_right_eq(x))
            return proper_y & within_sides
        def in_bottom_portion_of_hexagon(x, y):
            proper_y = (y >= -2 * np.pi / np.sqrt(3)) & (y < 0)
            within_sides = (y >= bottom_left_eq(x)) & (y >= bottom_right_eq(x))
            return proper_y & within_sides
        
        within_hexagon = in_top_portion_of_hexagon(x, y) | in_bottom_portion_of_hexagon(x, y)
        return within_hexagon
    
    within_hexagon_mask = inside_polygon(kx, ky)
    kx, ky = kx[within_hexagon_mask], ky[within_hexagon_mask]


    if doRotate:
        theta = np.pi / 6
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coordinates = np.vstack((kx, ky))
        kx, ky = np.einsum("ij,jk->ik", rotation_matrix, coordinates).reshape(2, -1)

    return kx, ky


def compute_chern_number(M, B_tilde, B, t1, t2, A_tilde, doTriangular=False, resolution = (501, 501)):

    if doTriangular:
        kx, ky = compute_triangular_brillouin_zone(resolution)
    else:
        kx, ky = compute_square_brillouin_zone(resolution)


    berry_curvature = compute_berry_curvature(kx, ky, M, B_tilde, B, t1, t2, A_tilde, doTriangular)

    bc_max, bc_min = np.max(berry_curvature), np.min(berry_curvature)
    bc_mean, bc_std = np.mean(berry_curvature), np.std(berry_curvature)

    dkx = (np.max(kx) - np.min(kx)) / resolution[0]
    dky = (np.max(ky) - np.min(ky)) / resolution[1]

    sum_kx = np.sum(berry_curvature, axis=0) * dkx
    sum_total = -np.sum(sum_kx) * dky


    if np.abs(sum_total) > 1e10:
        print(f"Warning: Chern number is too large for M = {M:.3f}, B_tilde = {B_tilde:.3f}: {sum_total:.3e}.")
        #print(np.max(np.abs(berry_curvature)))
        return None, bc_min, bc_max, bc_mean, bc_std
    elif np.isnan(sum_total):
        return None, bc_min, bc_max, bc_mean, bc_std
    else:
        return round(sum_total), bc_min, bc_max, bc_mean, bc_std


def compute_chern_phase_diagram(M_range, B_tilde_range, B, t1, t2, A_tilde, output_file=None, directory='', overwrite=False, doTriangular=False, resolution=(25, 25)):
    M_values = np.linspace(M_range[0], M_range[1], resolution[0])
    B_tilde_values = np.linspace(B_tilde_range[0], B_tilde_range[1], resolution[1])
    parameter_values = tuple(product(M_values, B_tilde_values))

    if output_file is None:
        root_fname = 'square' if not doTriangular else 'triangular'
        output_file = os.path.join(directory, root_fname+f"_chern_phase_diagram_{resolution[0]}x{resolution[1]}.h5")
    
    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Use overwrite=True to overwrite.")
        return output_file

    def compute_single(params):
        M, B_tilde = params
        chern, bc_min, bc_max, bc_mean, bc_std = compute_chern_number(M, B_tilde, B, t1, t2, A_tilde, doTriangular)
        return [M, B_tilde, chern, bc_min, bc_max, bc_mean, bc_std]

    with tqdm_joblib(tqdm(total=len(parameter_values), desc=f"Computing phase diagram for Chern number.")) as progress_bar:
        M_data, B_tilde_data, chern_data, bcmin, bcmax, bcmean, bcstd = np.array(Parallel(n_jobs=4)(delayed(compute_single)(params) for params in parameter_values), dtype=float).T

    with h5py.File(output_file, "w") as f:
        f.create_dataset(name = "M", data=M_data)
        f.create_dataset(name = "B_tilde", data=B_tilde_data)
        f.create_dataset(name =  "chern", data=chern_data.reshape(resolution).T)
        f.create_dataset(name = "bc_min", data=bcmin.reshape(resolution).T)
        f.create_dataset(name = "bc_max", data=bcmax.reshape(resolution).T)
        f.create_dataset(name = "bc_mean", data=bcmean.reshape(resolution).T)
        f.create_dataset(name = "bc_std", data=bcstd.reshape(resolution).T)

    return output_file


def plot_phase_diagram(fig, ax, 
                       X_values, Y_values, Z_values, 
                       labels:list=None, title:str=None, 
                       X_ticks=None, Y_ticks=None, X_tick_labels=None, Y_tick_labels=None,
                       cmap='Spectral', plotColorbar=True, doDiscreteColormap=True):
    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]
    Z_values = np.where(Z_values == -0, 0, Z_values)

    not_nan_mask = ~np.isnan(Z_values)
    unique_values = np.sort(np.unique(Z_values[not_nan_mask]).astype(int))
    unique_values = np.arange(-3, 4, 1)

    if doDiscreteColormap:
        if len(unique_values) < 25:
            cmap = plt.get_cmap(cmap)
            discrete_colors = cmap(np.linspace(0, 1, len(unique_values)))
            cmap = ListedColormap(discrete_colors)
            norm = BoundaryNorm(boundaries=np.append(unique_values, unique_values[-1] + 1), ncolors=len(unique_values))

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
        cbar.set_ticks(unique_values+0.5)
        cbar.set_ticklabels([str(val) for val in unique_values], fontsize=16)

    return fig, ax



# -----------------

def fig2_partc():
    M_values = np.linspace(-2.0, 12.0, 51)
    chern_numbers = [compute_chern_number(M, 0.0, 1.0, 1.0, 0.0) for M in M_values]
    plt.axhline(0, color='black', linestyle='--')
    plt.scatter(M_values, chern_numbers, marker='o', zorder=3)
    plt.xlabel("M")
    plt.title("Square Lattice: NN-only")
    plt.ylabel("Chern Number")
    plt.savefig("./Triangle/PhaseDiagrams/"+"square_bottom_line.png")
    plt.show()


def fig2_partd():
    B_tilde_vals = np.linspace(0.7, 1.1, 51)
    chern_numbers = [compute_chern_number(10.0, B_tilde, 1.0, 1.0, 1.0) for B_tilde in B_tilde_vals]
    plt.scatter(B_tilde_vals, chern_numbers, marker='o')
    plt.xlabel("B_tilde")
    plt.title("Square Lattice: NNN, M = 10")
    plt.ylabel("Chern Number")
    plt.axhline(0, color='black', linestyle='--')
    plt.show()


def triangular_bottom_line():
    M_values = np.linspace(-2.0, 8.0, 51)
    chern_numbers = [compute_chern_number(M, 0.0, 1.0, 1.0, np.nan, 1.0, True)[0] for M in M_values]
    for xpos in [-2, 6, 7]:
        plt.axvline(xpos, color='black', linestyle='--')
    plt.scatter(M_values, chern_numbers, marker='o', zorder=3)
    plt.xlabel("M")
    plt.title("Triangular Lattice: NN-only")
    plt.ylabel("Chern Number")
    plt.savefig("./Triangle/PhaseDiagrams/"+"triangular_bottom_line.png")
    plt.show()


def plot_phase_diagram_example():
    directory = './Triangle/PhaseDiagrams/'
    output_file = compute_chern_phase_diagram((-2.0, 8.0), (0.0, 0.5), resolution=(15, 15), B=1.0, t1=1.0, t2=np.nan, A_tilde=1.0, directory=directory, overwrite=True, doTriangular=True)
    with h5py.File(output_file, "r") as f:
        M_data = f["M"][:]
        B_tilde_data = f["B_tilde"][:]
        chern_data = f["chern"][:]
        try:
            bc_min_data = f["bc_min"][:]
            bc_max_data = f["bc_max"][:]
            bc_mean_data = f["bc_mean"][:]
            bc_std_data = f["bc_std"][:]
        except:
            pass

    #chern_data = np.clip(chern_data, -2, 1)  # Clip values to avoid excessively large chern numbers

    if True:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
        labels = ['min', 'max', 'mean', 'std']
        for ax, Z in zip(axs.flatten(), [bc_min_data, bc_max_data, bc_mean_data, bc_std_data]):
            ax.plot_trisurf(M_data, B_tilde_data, Z.T.flatten(), cmap='viridis', edgecolor='none')
            ax.set_xlabel("M")
            ax.set_ylabel("B_tilde")
            ax.set_title(f"Berry Curvature {labels.pop(0)}")
        plt.show()



    fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plot_phase_diagram(fig, ax, M_data, B_tilde_data, chern_data,
                                 labels=["M", "B_tilde"],
                                 title="Triangular Lattice : Chern Number",
                                 cmap='Spectral',
                                 doDiscreteColormap=True)
    
    linex = np.linspace(6.0, np.max(M_data), 500)
    ax.plot(linex, linex/8 - 0.75, ls='--', c='k', lw=1, zorder=2)
    #linex2 = np.linspace(3.5, np.max(M_data), 500)
    #ax.plot(linex2, linex2/8 - 7/16, ls='--', c='k', lw=1, zorder=2)
    for xpos in [-2.0, 7.0]:
        ax.axvline(x=xpos, color='black', linestyle='--', linewidth=1, zorder=2)
    ax.set_yticks([1/8, 1/4, 1/2])
    ax.set_yticklabels([r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$'])
    ax.set_xticks([-2, 3.5, 6, 7, 8])
    ax.set_ylabel(r"$\tilde{B}$", rotation=90)

    ax.set_xlim([np.min(M_data), np.max(M_data)])
    ax.set_ylim([np.min(B_tilde_data), np.max(B_tilde_data)])
    ax.set_title(f"Chern # : A_tilde=0.0")

    #plt.savefig(output_file[:-2]+"png")
    plt.show()


def plot_bz():
    kx, ky = compute_triangular_brillouin_zone()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    plt.scatter(kx, ky, s=1, color='black', alpha=0.5)
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title(f"Triangular Brillouin Zone\n{len(kx):.2e} points" + r"$\approx$" + f"{np.sqrt(len(kx)):.0f} x {np.sqrt(len(kx)):.0f} grid")

    ax.set_xticks([-4 * np.pi / 3, -2 * np.pi / 3, 0, 2 * np.pi / 3, 4 * np.pi / 3])
    ax.set_xticklabels([r"$-\frac{4\pi}{3}$", r"$-\frac{2\pi}{3}$", r"$0$", r"$\frac{2\pi}{3}$", r"$\frac{4\pi}{3}$"])
    ax.set_yticks([-2 * np.pi / np.sqrt(3), 0, 2 * np.pi / np.sqrt(3)])
    ax.set_yticklabels([r"$-\frac{2\pi}{\sqrt{3}}$", r"$0$", r"$\frac{2\pi}{\sqrt{3}}$"])

    ax.axvline(x=-4 * np.pi / 3, color='red', linestyle='--', linewidth=1, alpha=0.25)
    ax.axvline(x=4 * np.pi / 3, color='red', linestyle='--', linewidth=1, alpha=0.25)
    ax.axhline(y=-2 * np.pi / np.sqrt(3), color='red', linestyle='--', linewidth=1, alpha=0.25)
    ax.axhline(y=2 * np.pi / np.sqrt(3), color='red', linestyle='--', linewidth=1, alpha=0.25)

    plt.show()



if __name__ == "__main__":
    triangular_bottom_line()
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from scipy.linalg import eigh
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
from itertools import product
import os
import h5py



def compute_square_lattice(side_length):
    return np.arange(side_length**2).reshape((side_length, side_length))


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

    d1 = t1 * (np.sin(kx) + np.sin(kx / 2) * np.cos(sqrt3 / 2 * ky))
    d2 = t1 * (-sqrt3 * np.cos(kx / 2) * np.sin(sqrt3 / 2 * ky))
    d3 = M - 2*B*(2 - np.cos(kx) - 2 * np.cos(kx / 2) * np.cos(sqrt3 * ky / 2))

    dtilde1 =  A_tilde * np.sin(3 * kx / 2) * np.cos(sqrt3 / 2 * ky) * (sqrt3 / 2)
    dtilde2 = -A_tilde * (np.sin(sqrt3 * ky) - np.cos(3 * kx / 2) * np.sin(sqrt3 / 2 * ky))
    dtilde3 = -2*B_tilde * (3 - np.cos(sqrt3 * ky) - 2 * np.cos(3 / 2 * kx) * np.cos(sqrt3 / 2 * ky))
    
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
    

def compute_d_vector(kx, ky, M, B_tilde, B=1.0, t1=1.0, t2=1.0, A_tilde=1.0, doTriangular=False):
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

    epsilon = 1e-10
    d_dkx = np.where(np.abs(d_dkx) < epsilon, 0, d_dkx)  # Avoid numerical issues
    d_dky = np.where(np.abs(d_dky) < epsilon, 0, d_dky)  # Avoid numerical issues
    
    # Quotient Rule
    d_norm_safe = np.where(d_norm == 0, 1, d_norm)  # Avoid division by zero by replacing zero norms with 1
    d_norm_safe = np.clip(d_norm, -1e4, 1e4)  # Clip values to avoid excessively large norms
    
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
    kx_limits = (0.0, np.pi)
    ky_limits = (0.0, np.pi)

    kx = np.linspace(kx_limits[0], kx_limits[1], resolution[0])
    ky = np.linspace(ky_limits[0], ky_limits[1], resolution[1])

    kx, ky = np.meshgrid(kx, ky)
    return kx.flatten(), ky.flatten()


def compute_triangular_brillouin_zone(resolution=(200, 200)):
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

    return kx, ky


def compute_chern_number(M, B_tilde, B=1.0, t1=1.0, t2=1.0, A_tilde=1.0, doTriangular=False):
    N = 200
    resolution = (N, N)

    if doTriangular:
        kx, ky = compute_triangular_brillouin_zone(resolution)
    else:
        kx, ky = compute_square_brillouin_zone(resolution)


    berry_curvature = compute_berry_curvature(kx, ky, M, B_tilde, B, t1, t2, A_tilde, doTriangular)

    dkx = (np.max(kx) - np.min(kx)) / resolution[0]
    dky = (np.max(ky) - np.min(ky)) / resolution[1]

    sum_kx = np.sum(berry_curvature, axis=0) * dkx
    sum_total = np.sum(sum_kx) * dky

    if not doTriangular:
        chern = np.round(sum_total * 4)
    else:
        chern = np.round(sum_total)

    if np.abs(chern) > 1e10:
        print(f"Warning: Chern number is too large for M = {M:.3f}, B_tilde = {B_tilde:.3f}: {chern:.3e}.")
        #print(np.max(np.abs(berry_curvature)))
        return None
    elif np.isnan(chern):
        return None
    else:
        return - int(chern)


def compute_chern_phase_diagram(M_range, B_tilde_range, resolution=(25, 25), B=1.0, t1=1.0, t2=0.0, A_tilde=1.0, output_file=None, directory='', overwrite=False, doTriangular=False):
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
        chern = compute_chern_number(M, B_tilde, B, t1, t2, A_tilde, doTriangular)
        return [M, B_tilde, chern]

    with tqdm_joblib(tqdm(total=len(parameter_values), desc=f"Computing phase diagram for Chern number.")) as progress_bar:
        M_data, B_tilde_data, chern_data = np.array(Parallel(n_jobs=4)(delayed(compute_single)(params) for params in parameter_values), dtype=float).T

    with h5py.File(output_file, "w") as f:
        f.create_dataset(name = "M", data=M_data)
        f.create_dataset(name = "B_tilde", data=B_tilde_data)
        f.create_dataset(name =  "chern", data=chern_data.reshape(resolution).T)
    return output_file


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





def compute_bott_index(H):
    eigenvalues, eigenvectors = eigh(H)
    lower_band_eigvals = np.sort(eigenvalues)[:eigenvalues // 2]
    highest_lower_band = np.max(lower_band_eigvals)

    D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
    P_part = np.einsum('i,ij->ij', D, eigenvectors.conj().T)
    P = eigenvectors @ P_part

    fig, axs = plt.sublpots(1, 2, figsize=(12, 6))
    axs[0].imshow(np.abs(P), cmap='viridis', aspect='auto')
    axs[1].imshow(eigenvalues, cmap='viridis', aspect='auto')
    plt.show()

    return P



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
    M_values = np.linspace(-2.0, 12.0, 51)
    chern_numbers = [compute_chern_number(M, 0.0, 1.0, 1.0, 0.0, 0.0, True) for M in M_values]
    plt.axhline(0, color='black', linestyle='--')
    plt.scatter(M_values, chern_numbers, marker='o', zorder=3)
    plt.xlabel("M")
    plt.title("Triangular Lattice: NN-only")
    plt.ylabel("Chern Number")
    plt.savefig("./Triangle/PhaseDiagrams/"+"triangular_bottom_line.png")
    plt.show()


def plot_phase_diagram_example():
    directory = './Triangle/PhaseDiagrams/'
    output_file = compute_chern_phase_diagram((-2.0, 12.0), (0.0, 2.0), resolution=(100, 100), B=1.0, t1=1.0, t2=0.0, A_tilde=0.0, directory=directory, overwrite=False, doTriangular=True)
    with h5py.File(output_file, "r") as f:
        M_data = f["M"][:]
        B_tilde_data = f["B_tilde"][:]
        chern_data = f["chern"][:]

    chern_data = np.clip(chern_data, -2, 3)  # Clip values to avoid excessively large chern numbers

    fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plot_phase_diagram(fig, ax, M_data, B_tilde_data, chern_data,
                                 labels=["M", "B_tilde"],
                                 title="Triangular Lattice : Chern Number",
                                 cmap='Spectral',
                                 cbar_ticks=np.arange(-2, 1, 1),
                                 cbar_tick_labels=[str(i) for i in np.arange(-2, 1, 1)])
    plt.savefig(output_file[:-2]+"png")
    plt.show()







if __name__ == "__main__":
    plot_phase_diagram_example()
import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
from itertools import product
import os, h5py
from matplotlib.colors import ListedColormap, BoundaryNorm


def compute_d_vector(kx, ky, m0, h_vector, t, t0):
    # Method 1: 
    d1 = t * np.sin(kx) + 1j * h_vector[0]
    d2 = t * np.sin(ky) + 1j * h_vector[1]
    d3 =  m0 + t0 * (np.cos(kx) + np.cos(ky)) + 1j *  h_vector[2]
    return np.array([d1, d2, d3])


def compute_unit_vector(vector):
    if vector.ndim == 1:
        vector = vector.reshape((3, 1))
    
    norms = np.linalg.norm(vector, axis=0)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    unit_vector = vector / norms
    #   unit_vector = np.clip(unit_vector, -1e2, 1e2)  # Ensure unit_vector is not too large
    return unit_vector
    

def compute_vector_derivative(vector_function:callable, kx, ky, direction:str, dk=1e-5, **kwargs):
    if direction == 'x':
        partial_vector = vector_function(kx + dk, ky, **kwargs) - vector_function(kx - dk, ky, **kwargs)
    elif direction == 'y':
        partial_vector = vector_function(kx, ky + dk, **kwargs) - vector_function(kx, ky - dk, **kwargs)
    return partial_vector / (2 * dk)


def compute_d_hat_and_derivatives(kx, ky, m0, h_vector, t, t0):
    d = compute_d_vector(kx, ky, m0, h_vector, t, t0)
    d_norm = np.linalg.norm(d, axis=0, keepdims=True)
    d_dkx = compute_vector_derivative(compute_d_vector, kx, ky, direction='x', m0=m0, h_vector=h_vector, t=t, t0=t0)
    d_dky = compute_vector_derivative(compute_d_vector, kx, ky, direction='y', m0=m0, h_vector=h_vector, t=t, t0=t0)

    d_norm_safe = np.where(d_norm == 0, 1, d_norm)
    d_hat = d / d_norm_safe

    #d_hat_dkx = (d_dkx / d_norm_safe) - (d_hat * np.einsum("ij,ij->j", d, d_dkx) / (d_norm_safe**2))
    #d_hat_dky = (d_dky / d_norm_safe) - (d_hat * np.einsum("ij,ij->j", d, d_dky) / (d_norm_safe**2))

    d_hat_dkx = (d_dkx / d_norm_safe) - (d_hat / 2 / (d_norm_safe**2)) * (np.einsum("ij,ij->j", np.conj(d_dkx), d) + np.einsum("ij,ij->j", np.conj(d), d_dkx))
    d_hat_dky = (d_dky / d_norm_safe) - (d_hat / 2 / (d_norm_safe**2)) * (np.einsum("ij,ij->j", np.conj(d_dky), d) + np.einsum("ij,ij->j", np.conj(d), d_dky))

    return d_hat, d_hat_dkx, d_hat_dky


def compute_berry_curvature(kx, ky, m0, h_vector, t, t0):
    d_hat, d_hat_dx, d_hat_dy = compute_d_hat_and_derivatives(kx, ky, m0, h_vector, t, t0)
    dx_cross_dy = np.cross(d_hat_dx, d_hat_dy, axis=0)

    berry_curvature = np.einsum("ij,ij->j", np.conj(d_hat), dx_cross_dy)
    return berry_curvature


def compute_chern_number(m0, h_vector, t, t0, resolution = (201, 201)):
    # Square First Brillouin Zone
    kx = np.linspace(-np.pi, np.pi, resolution[0], endpoint=False)
    ky = np.linspace(-np.pi, np.pi, resolution[1], endpoint=False)
    kx, ky = np.meshgrid(kx, ky)
    kx, ky = kx.flatten(), ky.flatten()

    berry_curvature = compute_berry_curvature(kx, ky, m0, h_vector, t, t0)
    dkx = 2 * np.pi / resolution[0]
    dky = 2 * np.pi / resolution[1]

    sum_kx = np.sum(berry_curvature, axis=0) * dkx
    sum_total = np.sum(sum_kx) * dky
 
    chern = np.real(sum_total / (4 * np.pi))

    if np.abs(chern) > 1e2:
        print(f"Warning: Chern number is too large for m0 = {m0:.3f}, h_vector = {h_vector} : {chern:.3e}.")
        return None
    elif np.isnan(chern):
        return None

    return chern


def FHS_chern(m0, h_vector, t, t0, n=0):
    qx = qy = 1.0
    Lx = Ly = 31
    dkx = 2 * np.pi / qx / Lx
    dky = 2 * np.pi / qy / Ly
    def compute_hamiltonian(kx, ky):
        paulix = np.array([[0, 1], [1, 0]])
        pauliy = np.array([[0, -1j], [1j, 0]])
        pauliz = np.array([[1, 0], [0, -1]])
        d1, d2, d3 = compute_d_vector(kx, ky, m0, h_vector, t, t0)
        return paulix * d1 + pauliy * d2 + pauliz * d3
    
    def compute_left_right_eigenvectors(kx, ky):
        hamiltonian = compute_hamiltonian(kx, ky)
        eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True)
        return left_eigenvectors[:, n], right_eigenvectors[:, n]

    def U_mu(mu, kx, ky):
        """
        U(1) link variable from the wavefunctions of the nth Bloch band
        
        :param mu: (str) either 'x' or 'y'
        :param k_l: (ndarray) 2xN array s.t. the first column is k_x values and the second is k_y values.
        """
        # N_\mu = |<n(k_l)|n(k_l + \hat{\mu})>|
        # U_\mu(k_l) = <n(k_l)|n(k_l + \hat{\mu})> / N_\mu
        kx2, ky2 = (kx + dkx, ky) if mu == 'x' else (kx, ky + dky)
        lvec1, rvec1 = compute_left_right_eigenvectors(kx, ky)
        lvec2, rvec2 = compute_left_right_eigenvectors(kx2, ky2)

        dot_left = np.dot(lvec1, lvec2)
        dot_right = np.dot(rvec1, rvec2)
        return dot_left/spla.norm(dot_left), dot_right/spla.norm(dot_right)


    def compute_field_strength(kx, ky, left_or_right='left'):
        if left_or_right == 'left':
            idx = 0
        else:
            idx = 1
        term1 = U_mu('x', kx,       ky      )[idx]
        term2 = U_mu('y', kx + dkx, ky      )[idx]
        term3 = U_mu('x', kx,       ky + dky)[idx].conj()
        term4 = U_mu('y', kx,       ky      )[idx].conj()
        return np.log(term1 * term2 * term3 * term4)
    

    # Square Lattice FBZ
    kxs, kys = np.linspace(-np.pi, np.pi, Lx, endpoint=False), np.linspace(-np.pi, np.pi, Ly, endpoint=False)
    kxs, kys = np.meshgrid(kxs, kys)
    kxs, kys = kxs.flatten(), kys.flatten()
    
    chern = np.round(sum(compute_field_strength(kx, ky) for kx, ky in zip(kxs, kys)) / (2 * np.pi * 1j), 3)
    print(chern)


def FHS_chern_fast(m0, h_vector, t, t0, n_band=0, Lx=31, Ly=31):
    # 1. Setup Grid
    k_vec = np.linspace(-np.pi, np.pi, Lx, endpoint=False)
    kx_grid, ky_grid = np.meshgrid(k_vec, k_vec)
    
    d = compute_d_vector(kx_grid, ky_grid, m0, h_vector, t, t0) 
    
    H = np.zeros((Lx, Ly, 2, 2), dtype=complex)

    # tau \cdot (\vec{d_NH})
    H[:, :, 0, 0] = d[2]
    H[:, :, 1, 1] = -d[2]
    H[:, :, 0, 1] = d[0] - 1j * d[1]
    H[:, :, 1, 0] = d[0] + 1j * d[1]

    H_flat = H.reshape(-1, 2, 2)
    L_vecs = np.zeros((Lx * Ly, 2), dtype=complex)
    R_vecs = np.zeros((Lx * Ly, 2), dtype=complex)
    
    for i in range(len(H_flat)):
        w, vl, vr = spla.eig(H_flat[i], left=True, right=True)
        idx = np.lexsort((np.imag(w), np.real(w)))
        
        # Extract the desired band
        chosen_idx = idx[n_band]
        L_vecs[i] = vl[:, chosen_idx]
        R_vecs[i] = vr[:, chosen_idx]

    # Reshape back to grid
    L_grid = L_vecs.reshape(Lx, Ly, 2)
    R_grid = R_vecs.reshape(Lx, Ly, 2)

    # 4. Compute Link Variables (Biorthogonal)
    # Roll to get neighbors (k+x, k+y)
    R_plus_x = np.roll(R_grid, -1, axis=1)
    R_plus_y = np.roll(R_grid, -1, axis=0)
    
    # Inner products <L(k)|R(k+mu)>
    ov_x = np.einsum('ijk,ijk->ij', np.conj(L_grid), R_plus_x)
    ov_y = np.einsum('ijk,ijk->ij', np.conj(L_grid), R_plus_y)
    
    # Check for Exceptional Points (where overlap is zero)
    # If <L|R> ~ 0, the gap is closed or bands merged -> Chern undefined
    min_overlap = np.min(np.abs(ov_x))
    if min_overlap < 1e-6:
        print(f"Warning: Possible Exceptional Point detected (Overlap ~ {min_overlap:.2e})")

    # Normalize (keep phase only)
    U_x = ov_x / (np.abs(ov_x) + 1e-12)
    U_y = ov_y / (np.abs(ov_y) + 1e-12)

    # 5. Field Strength F = ln( U_x * U_y(x) * U_x(y)* * U_y* )
    U_x_py = np.roll(U_x, -1, axis=0)
    U_y_px = np.roll(U_y, -1, axis=1)
    
    F_plaq = np.log(U_x * U_y_px * np.conj(U_x_py) * np.conj(U_y))
    
    # 6. Sum flux
    total_flux = np.sum(F_plaq)
    chern = np.real(total_flux / (2j * np.pi))
    
    return chern
    




def compute_chern_phase_diagram(m0_range, h_range, h_type,
                                output_file=None, directory='', overwrite=False, resolution=(25, 25)):
    m0_values = np.linspace(m0_range[0], m0_range[1], resolution[0])
    h_values = np.linspace(h_range[0], h_range[1], resolution[1])
    parameter_values = tuple(product(m0_values, h_values))

    if output_file is None:
        root_fname = 'square'
        output_file = os.path.join(directory, root_fname+f"_chern_phase_diagram_{resolution[0]}x{resolution[1]}.h5")
    
    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Use overwrite=True to overwrite.")
        return output_file

    def compute_single(params):
        m0, h = params
        match h_type:
            case 'x':
                h_vector = [h, 0.0, 0.0]
            case 'y':
                h_vector = [0.0, h, 0.0]
            case 'z':
                h_vector = [0.0, 0.0, h]
        chern = FHS_chern_fast(m0, h_vector, 1.0, 1.0)
        #chern2 = compute_chern_number2(m0, h, 1.0, n=n)
        return [m0, h, chern] #+ [chern2]

    with tqdm_joblib(tqdm(total=len(parameter_values), desc=f"Computing phase diagram for Chern number.")) as progress_bar:
        m0_data, h_data, chern_data = np.array(Parallel(n_jobs=-2)(delayed(compute_single)(params) for params in parameter_values), dtype=float).T

    with h5py.File(output_file, "w") as f:
        f.create_dataset(name = "m0", data=m0_data)
        f.create_dataset(name = "h", data=h_data)
        f.create_dataset(name =  "chern", data=chern_data.reshape(resolution).T)
        #f.create_dataset(name =  "chern2", data=chern2_data.reshape(resolution).T)
    return output_file


def plot_phase_diagram(fig, ax, 
                       X_values, Y_values, Z_values, 
                       labels:list=None, title:str=None, 
                       X_ticks=None, Y_ticks=None, X_tick_labels=None, Y_tick_labels=None,
                       cmap='Spectral', plotColorbar=True, doDiscreteColormap=True):
    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]
    Z_values = np.where(Z_values == -0, 0, Z_values)
    Z_rounded = np.round(Z_values)

    not_nan_mask = ~np.isnan(Z_values)
    unique_values = np.sort(np.unique(Z_rounded[not_nan_mask]).astype(int))

    if doDiscreteColormap:
        if len(unique_values) < 25:
            original_cmap = plt.get_cmap(cmap)
            discrete_colors = original_cmap(np.linspace(0, 1, len(unique_values)))
            new_cmap = ListedColormap(discrete_colors)
            norm = BoundaryNorm(boundaries=np.append(unique_values, unique_values[-1] + 1), ncolors=len(unique_values))
    else:
        new_cmap = cmap
        norm = None

    im = ax.imshow(Z_rounded, extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]], 
                   origin='lower', aspect='auto', cmap=new_cmap, interpolation='none', 
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

    if plotColorbar and doDiscreteColormap:
        cbar = fig.colorbar(im, ax=ax, ticks=unique_values)
        #cbar.set_ticks(unique_values+0.5)
        cbar.set_ticklabels([str(val) for val in unique_values])
    else:
        fig.colorbar(im, ax=ax)

    return fig, ax


if __name__ == "__main__":
    #print(compute_chern_number(1.0, [0,0,0], 1.0, 1.0))
    FHS_chern_fast(1.05, [0, 0.0, 1.0], 1.0, 1.0)
    h_dir = 'x'
    resolution = (51, 51)
    file = compute_chern_phase_diagram((-2.0, 2.0), (-1.0, 1.0), h_dir, overwrite=True, output_file = f"chern_pd_{h_dir}_{resolution[0]}x{resolution[1]}.h5", resolution=resolution)

    with h5py.File(file, "r") as f:
        m0 = f["m0"][:]
        h = f["h"][:]
        chern = f["chern"][:]
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    plot_phase_diagram(fig, ax, m0, h, chern, doDiscreteColormap=True)
    ax.axhline(0.0, -2.0, 2.0, c='k', ls='--')
    ax.axvline(0.0, -1.0, 1.0, c='k', ls='--')
    plt.show()

import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
from itertools import product
import os, h5py
from matplotlib.colors import ListedColormap, BoundaryNorm


def compute_d_vector(kx, ky, m0, h_vector, t, t0, a):
    d1 = t * np.sin(kx * a) + 1j * h_vector[0]
    d2 = t * np.sin(ky * a) + 1j * h_vector[1]
    d3 = m0 + t0 * np.cos(kx * a) + np.cos(ky * a) + 1j * h_vector[2]
    return [d1, d2, d3]

def compute_hamiltonian(kx, ky, m0, h_vector, t, t0, a):
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    d_vector = compute_d_vector(kx, ky, m0, h_vector, t, t0, a)
    return pauli_x * d_vector[0] + pauli_y * d_vector[1] + pauli_z * d_vector[2]

def compute_fhs_chern_number(m0, h_vector, Nx = 25, Ny = 25):
    kx_values = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    ky_values = np.linspace(-np.pi, np.pi, Ny, endpoint=False)
    dkx = 2 * np.pi / Nx
    dky = 2 * np.pi / Ny

    def compute_eigenvectors(kx_, ky_, band_index):
            hamiltonian = compute_hamiltonian(kx_, ky_, m0, h_vector, 1.0, 1.0, 1.0)
            eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True)
            
            # --- FIX: Sort eigenvalues and vectors by the real part of eigenvalues ---
            idx = np.argsort(eigenvalues.real)
            eigenvalues = eigenvalues[idx]
            left_eigenvectors = left_eigenvectors[:, idx]
            right_eigenvectors = right_eigenvectors[:, idx]
            # -----------------------------------------------------------------------

            return left_eigenvectors[:, band_index], right_eigenvectors[:, band_index]
    
    def compute_u1_link_variable(kx, ky, direction:str, band_index):
        left_eigenvector, _ = compute_eigenvectors(kx, ky, band_index)
        if direction == 'none':
            pass
        elif direction == 'x':
            _, shifted_right_eigenvector = compute_eigenvectors(kx + dkx, ky, band_index)
        elif direction == 'y':
            _, shifted_right_eigenvector = compute_eigenvectors(kx, ky + dky, band_index)

        product = np.dot(left_eigenvector.conj(), shifted_right_eigenvector)
        return product / np.abs(product)
    
    def compute_field_strength(kx_, ky_, band_index):
        term1 = compute_u1_link_variable(kx_, ky_, 'x', band_index)
        term2 = compute_u1_link_variable(kx_ + dkx, ky_, 'y', band_index)
        term3 = compute_u1_link_variable(kx_, ky_ + dky, 'x', band_index).conj()
        term4 = compute_u1_link_variable(kx_, ky_, 'y', band_index).conj()
        return np.log(term1 * term2 * term3 * term4)
    
    fs_values = []
    for kx in kx_values:
        for ky in ky_values:
            fs_values.append(compute_field_strength(kx, ky, 0))

    return (sum(fs_values) / (2 * np.pi * 1j)).real


def compute_chern_phase_diagram(m0_range, h_range, h_type,
                                output_file=None, directory='', overwrite=False, resolution=(25, 25)):
    m0_values = np.linspace(m0_range[0], m0_range[1], resolution[0])
    h_values = np.linspace(h_range[0], h_range[1], resolution[1])
    parameter_values = tuple(product(m0_values, h_values))

    if output_file is None:
        root_fname = 'square'
        output_file = os.path.join(directory, root_fname+f"_chern_phase_diagram_{resolution[0]}x{resolution[1]}.h5")
    else:
        output_file = os.path.join(directory, output_file)

    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Use overwrite=True to overwrite.")
        return output_file

    def compute_single(i):
        m0, h = parameter_values[i]
        match h_type:
            case 'x':
                h_vector = [h, 0.0, 0.0]
            case 'y':
                h_vector = [0.0, h, 0.0]
            case 'z':
                h_vector = [0.0, 0.0, h]
        chern = compute_fhs_chern_number(m0, h_vector)
        #chern2 = compute_chern_number2(m0, h, 1.0, n=n)
        return [m0, h, chern] #+ [chern2]

    with tqdm_joblib(tqdm(total=len(parameter_values), desc=f"Computing phase diagram for Chern number.")) as progress_bar:
        m0_data, h_data, chern_data = np.array(Parallel(n_jobs=-2)(delayed(compute_single)(i) for i in range(len(parameter_values))), dtype=float).T

    with h5py.File(output_file, "w") as f:
        f.create_dataset(name = "m0", data=m0_data)
        f.create_dataset(name = "h", data=h_data)
        f.create_dataset(name =  "chern", data=chern_data.reshape(resolution).T)
    return output_file



if __name__ == "__main__":
    output_file = compute_chern_phase_diagram((-4.0, 4.0), (-1.0, 1.0), 'x', overwrite=True, output_file='temp.h5', directory='./Non-Hermitian/Data/')
    
    with h5py.File(output_file, 'r') as f:
        m0 = f["m0"][:]
        h = f["h"][:]
        chern = f["chern"][:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    from nonhermitian_chern import plot_phase_diagram

    plot_phase_diagram(fig, ax, m0, h, chern)
    plt.show()
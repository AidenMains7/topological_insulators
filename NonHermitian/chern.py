import numpy as np
import scipy.linalg as spla

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
import os, h5py
from itertools import product

from cProfile import Profile
import pstats


# region Non-Hermitian d-vector
def compute_d_vector(kx:float, ky:float, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float):
    d1 = t * np.sin(kx * a) + 1.0j * h_vector[0]
    d2 = t * np.sin(ky * a) + 1.0j * h_vector[1]
    d3 = m0 + t0 * np.cos(kx * a) + t0 * np.cos(ky * a) + 1.0j * h_vector[2]
    vector = np.array((d1, d2, d3))
    return vector


def compute_d_vector_conj(kx:float, ky:float, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float):
    d1 = t * np.sin(kx * a) + 1.0j * h_vector[0]
    d2 = t * np.sin(ky * a) + 1.0j * h_vector[1]
    d3 = m0 + t0 * np.cos(kx * a) + t0 * np.cos(ky * a) + 1.0j * h_vector[2]
    return np.array((d1, d2, d3)).conj()

# endregion
# region Chern Number computation

def compute_normalized_vector(vector_generating_function:callable, kx:float, ky:float, vector_kwargs:dict=None, returnNorm:bool=False):
    vector = vector_generating_function(kx, ky, **vector_kwargs)
    norm = spla.norm(vector, axis=0)
    norm = np.where(norm == 0, 1, norm)
    if returnNorm:
        return vector / norm, norm
    return vector / norm


def compute_vector_finite_derivative(vector_generating_function:callable, kx:float, ky:float, direction:str, vector_kwargs:dict=None, dk:float=1e-5):
    if direction == 'x':
        d_v_dk = vector_generating_function(kx + dk, ky, **vector_kwargs) - vector_generating_function(kx - dk, ky, **vector_kwargs)
    elif direction == 'y':
        d_v_dk = vector_generating_function(kx, ky + dk, **vector_kwargs) - vector_generating_function(kx, ky - dk, **vector_kwargs)
    else:
        raise ValueError("Direction must be either 'x' or 'y'")
    return d_v_dk / (2 * dk)


def compute_berry_curvature(vector_generating_function:callable, kx:float, ky:float, vector_kwargs:dict):
    def conjugate_vector_generating_function(kx, ky, **vector_kwargs):
        return np.conj(vector_generating_function(kx, ky, **vector_kwargs))
    v = vector_generating_function(kx, ky, **vector_kwargs)
    vconj = np.conj(v)
    v_hat, norm = compute_normalized_vector(vector_generating_function, kx, ky, vector_kwargs, returnNorm=True)
    d_v_dkx = compute_vector_finite_derivative(vector_generating_function, kx, ky, 'x', vector_kwargs)
    d_v_dky = compute_vector_finite_derivative(vector_generating_function, kx, ky, 'y', vector_kwargs)

    d_vconj_dkx = compute_vector_finite_derivative(conjugate_vector_generating_function, kx, ky, 'x', vector_kwargs)
    d_vconj_dky = compute_vector_finite_derivative(conjugate_vector_generating_function, kx, ky, 'y', vector_kwargs)

    d_vhat_dkx = d_v_dkx / norm - v_hat / (2 * norm ** 2) * (np.dot(d_vconj_dkx, v) + np.dot(vconj, d_v_dkx))
    d_vhat_dky = d_v_dky / norm - v_hat / (2 * norm ** 2) * (np.dot(d_vconj_dky, v) + np.dot(vconj, d_v_dky))

    cross_product = np.cross(d_vhat_dkx, d_vhat_dky)
    berry_curvature = np.dot(v_hat, cross_product) / 2
    return berry_curvature


def compute_chern_number(vector_generating_function:callable, vector_kwargs:dict, brillouin_zone_resolution:int=101,
                         returnGapData:bool = True):
    kx_values = ky_values = np.linspace(-np.pi, np.pi, brillouin_zone_resolution, endpoint=False)
    kx_values, ky_values = np.meshgrid(kx_values, ky_values)
    kx_values, ky_values = kx_values.flatten(), ky_values.flatten()

    if returnGapData:
        min_real_gap = min_imag_gap = min_mag_gap = float('inf')
    berry_curvatures = []
    for kx, ky in zip(kx_values, ky_values):

        if returnGapData:
            v = vector_generating_function(kx, ky, **vector_kwargs)
            gap = np.sqrt(np.dot(v,v))
            real_gap = gap.real
            imag_gap = gap.imag
            mag_gap = spla.norm(v)
            if real_gap < min_real_gap:
                min_real_gap = real_gap
            if imag_gap < min_imag_gap:
                min_imag_gap = imag_gap
            if mag_gap < min_mag_gap:
                min_mag_gap = mag_gap

        bc = compute_berry_curvature(vector_generating_function, kx, ky, vector_kwargs)
        berry_curvatures.append(bc)

    dkx = 2 * np.pi / brillouin_zone_resolution
    chern_number = np.real(np.sum(berry_curvatures) * dkx * dkx / (2 * np.pi))
    if returnGapData:
        return chern_number, min_real_gap, min_imag_gap, min_mag_gap
    return chern_number


def plot_energy_in_complex_plane(m0, h_vector, t, t0):
    def compute_energies(kx, ky, m0, h_vector, t, t0):
        term1 = (t * np.sin(kx)) ** 2 - h_vector[0] ** 2 + 2 * 1.0j * h_vector[0] * (t * np.sin(kx))
        term2 = (t * np.sin(ky)) ** 2 - h_vector[1] ** 2 + 2 * 1.0j * h_vector[1] * (t * np.sin(ky))
        term3 = (m0 + t0 * (np.cos(kx) + np.cos(ky))) ** 2 - h_vector[2] ** 2 + 2 * 1.0j * h_vector[2] * (m0 + t0 * (np.cos(kx) + np.cos(ky)))
        return np.sqrt(term1 + term2 + term3)

    kx_values = ky_values = np.linspace(-np.pi, np.pi, 101, endpoint=False)
    kx_values, ky_values = np.meshgrid(kx_values, ky_values)
    energies = compute_energies(kx_values, ky_values, m0, h_vector, t, t0)
    energies = np.concatenate((energies, -energies))

    real_part = energies.flatten().real
    imaginary_part = energies.flatten().imag

    plt.scatter(real_part, imaginary_part)
    plt.show()



# endregion    
# region General Functions

def compute_phase_diagram_parallel(worker_function:callable, parameter_values, filename:str, 
                                   overwrite:bool=False, dataset_labels:list[str]=None):
    filename_base = filename.split('.')[0]
    filename = filename_base + '.h5'


    if os.path.exists(filename) and not overwrite:
        print(f"Filename '{filename}' already exists.")
        return filename

    with tqdm_joblib(tqdm(total=len(parameter_values))) as progress_bar:
        computed_data = Parallel(n_jobs=-1)(delayed(worker_function)(*params) for params in parameter_values)
    computed_data = np.array(computed_data).T

    if (dataset_labels == None) or (len(dataset_labels) != computed_data.shape[0]):
        dataset_labels = [f"dataset_{i}" for i in range(len(computed_data.shape[0]))]

    with h5py.File(filename, 'w') as f:
        for label, dataset in zip(dataset_labels, computed_data):
            f.create_dataset(name=label, data=dataset)

    return filename


def get_data_from_h5_file(filename:str, dataset_labels:list[str]=None):
    with h5py.File(filename, 'r') as f:
        if dataset_labels != None:
            data = []
            good_labels = []
            for label in dataset_labels:
                try:
                    data.append(f[label][:])
                    good_labels.append(label)
                except:
                    print(f"Label '{label}' not in file '{filename}'")
        else:
            data = [d[:] for d in f.values()]
            good_labels = list(f.keys())
    return data, good_labels


def plot_phase_diagram(fig, ax, 
                       X_values, Y_values, Z_values, 
                       labels:list=None, title:str=None, 
                       X_ticks=None, Y_ticks=None, X_tick_labels=None, Y_tick_labels=None,
                       cmap='Spectral', plotColorbar=True, doDiscreteColormap=True):
    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]
    Z_values = np.where(Z_values == -0, 0, Z_values)

    if doDiscreteColormap:
        not_nan_mask = ~np.isnan(Z_values)
        unique_values = np.sort(np.unique(Z_values[not_nan_mask]).astype(int))
        cmap = plt.get_cmap(cmap)
        discrete_colors = cmap(np.linspace(0, 1, len(unique_values)))
        cmap = ListedColormap(discrete_colors)
        norm = BoundaryNorm(boundaries=np.append(unique_values, unique_values[-1] + 1), ncolors=len(unique_values))
    else:
        cmap = plt.get_cmap(cmap)
        norm = None

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
        if doDiscreteColormap:
            cbar.set_ticks(unique_values+0.5)
            cbar.set_ticklabels([str(val) for val in unique_values], fontsize=16)

    return fig, ax


def profile_wrapper(func:callable, *args, **kwargs):
    profiler = Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    return result
    

# endregion
# region Chern Phase Diagram Stuff
def plot_chern_and_gaps(filename:str, labels:list[str]):
    data, labels = get_data_from_h5_file(filename, labels)
    m0, h, chern, min_real_gap, min_imag_gap, min_mag_gap = data

    n_unique_m0 = np.unique(m0).size
    n_unique_h = np.unique(h).size
    
    chern = chern.reshape(n_unique_m0, n_unique_h).T
    min_real_gap, min_imag_gap, min_mag_gap = min_real_gap.reshape(n_unique_m0, n_unique_h).T, min_imag_gap.reshape(n_unique_m0, n_unique_h).T, min_mag_gap.reshape(n_unique_m0, n_unique_h).T

    min_imag_gap = np.abs(min_imag_gap)

    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    fig, axes[0, 0] = plot_phase_diagram(fig, axes[0, 0], m0, h, chern, doDiscreteColormap=False)
    fig, axes[0, 1] = plot_phase_diagram(fig, axes[0, 1], m0, h, min_mag_gap, doDiscreteColormap=False, cmap='RdPu')
    fig, axes[1, 0] = plot_phase_diagram(fig, axes[1, 0], m0, h, min_real_gap, doDiscreteColormap=False, cmap='RdPu')
    fig, axes[1, 1] = plot_phase_diagram(fig, axes[1, 1], m0, h, min_imag_gap, doDiscreteColormap=False, cmap='RdPu')
    fig.suptitle(filename)
    axes[0, 0].set_title("Chern Number")
    axes[0, 1].set_title("Minimum Gap $\\sqrt{\\mathbf{d}^\\dagger \\cdot \\mathbf{d}}$")
    axes[1, 0].set_title("Minimum Gap $\\Re \\sqrt{\\mathbf{d} \\cdot \\mathbf{d}}$")
    axes[1, 1].set_title("Minimum Gap $\\Im \\sqrt{\\mathbf{d} \\cdot \\mathbf{d}}$")

    def plot_line(ax, xrange, yrange, slope, intercept):
        t = np.linspace(xrange[0], xrange[1], 101)
        ax.plot(t, slope * t + intercept, ls='--', c='k', lw=1)
        ax.set_ylim(yrange)
        ax.set_xlim(xrange)

    def plot_quadratic(ax, xrange, yrange):
        t = np.linspace(-2.0, 2.0, 101)
        ax.plot(t, np.sqrt(2 * np.abs(t) - t**2), ls='--', c='k', lw=1)
        ax.plot(t, -np.sqrt(2 * np.abs(t) - t**2), ls='--', c='k', lw=1)
        ax.set_ylim(yrange)
        ax.set_xlim(xrange)

    if   filename.find('_hx') + 1:
        h_dir = 'hx'
    elif filename.find('_hy') + 1:
        h_dir = 'hy'
    elif filename.find('_hz') + 1:
        h_dir = 'hz'
    else:
        h_dir = 'NAN'

    for ax in axes.flatten():
        ax.set_xlabel("$m_0$", fontsize=12)
        ax.set_ylabel(f"$h_{h_dir}$", rotation=0, fontsize=12)
        if h_dir in ['hx', 'hy']:
            slopes = [1, -1]
            intercepts = [2, 0, -2]
            values = tuple(product(slopes, intercepts))
            for s, i in values:
                plot_line(ax, (-3.0, 3.0), (-3.0, 3.0), s, i)
        else:
            plot_quadratic(ax, (0.0, 2.0), (0.0, 1.0))
    plt.tight_layout()
    plt.savefig(filename.replace(".h5", ".svg"))


def compute_chern_phase_diagram():
    def worker(m0:float, h:float, h_dir:str, t:float=1.0, t0:float=1.0, a:float=1.0):
        match h_dir:
            case 'x':
                h_vector = [h, 0.0, 0.0]
            case 'y':
                h_vector = [0.0, h, 0.0]
            case 'z':
                h_vector = [0.0, 0.0, h]
            case _:
                raise ValueError(f"h_dir must be in ['x', 'y', 'z']. It is {h_dir}")
        vector_kwargs = {
            'm0': m0,
            'h_vector': h_vector,
            't': t,
            't0': t0,
            'a': a
        }
        chern_number, min_real_gap, min_imag_gap, min_mag_gap = compute_chern_number(compute_d_vector, vector_kwargs)
        return [m0, h, chern_number, min_real_gap, min_imag_gap, min_mag_gap]

    m0_values = np.linspace(0.0, 2.0, 51)
    h_values = np.linspace(0.0, 1.0, 51)
    h_dir_values = ['x'] 
    parameter_values = tuple(product(m0_values, h_values, h_dir_values))

    labels = ['m0', 'h', 'chern', 'min_real_gap', 'min_imag_gap', 'min_mag_gap']
    directory = "NonHermitian/Data/"
    filename = directory+f"chern_h{h_dir_values[0]}.h5"
    filename = compute_phase_diagram_parallel(worker, parameter_values, filename = filename, 
                                              overwrite=False, dataset_labels=labels)

    plot_chern_and_gaps(filename, labels)
    #plt.show()

# endregion

def find_zeros_of_energy_hz():
    def f(m0, hz, kx, ky):
        m0 = m0[:, np.newaxis, np.newaxis, np.newaxis]
        hz = hz[np.newaxis, :, np.newaxis, np.newaxis]
        kx = kx[np.newaxis, np.newaxis, :, np.newaxis]
        ky = ky[np.newaxis, np.newaxis, np.newaxis, :]
        real = m0**2 - hz**2 + 2 + 2 * np.cos(kx) * np.cos(ky) + 2 * m0 * (np.cos(kx) + np.cos(ky))
        imag = 2 * hz * (m0 + np.cos(kx) + np.cos(ky))
        return real, imag
    
    N = 11
    Nk = 15
    m0 = np.linspace(0.0, 2.0, 25)
    hz = np.linspace(0.0, 1.0, 25)
    kx = ky =  np.linspace(0.0, np.pi, Nk)

    labels = ['m0', 'hz', 'kx', 'ky']
    values = (m0, hz, kx, ky)
    
    real, imag = f(m0, hz, kx, ky)
    real_zero = np.isclose(real, 0.0)
    imag_zero = np.isclose(imag, 0.0)
    both_zero = real_zero & imag_zero

    
    zero_idxs = np.argwhere(both_zero)
    print(zero_idxs)
    for zero_idx in zero_idxs:
        for i, idx in enumerate(zero_idx):
            print(labels[i], values[i][idx])
        print('-'*10)



if __name__ == "__main__":
    vector_kwargs = {
        'm0': 1.0,
        'h_vector': [0.0, 0.0, 0.0],
        't': 1.0,
        't0': 1.0,
        'a': 1.0
    }
    #chern_number = compute_chern_number(compute_d_vector, vector_kwargs, returnGapData=False)
    #print(f"Chern number: {chern_number}")

    compute_chern_phase_diagram()


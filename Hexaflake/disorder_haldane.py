import numpy as np
from matplotlib import pyplot as plt
from MaybeActualFinalHaldane2 import  compute_hamiltonian, compute_eigen_data, compute_geometric_data, compute_bott_index
from itertools import product
from tqdm_joblib import tqdm_joblib, tqdm
from joblib import Parallel, delayed
import scipy as sp
from testing_filesave import iterative_filename
from multiprocessing import Lock, Manager
import h5py, os, glob
from time import time



def plot_phase_diagram(fig, ax, phi_vals, M_vals, bott_vals, cmap='viridis', titleparams=None, doBoundary=True, doCbar=True, doLabels=True):
    phi_range = [np.min(phi_vals), np.max(phi_vals)]
    M_range = [np.min(M_vals), np.max(M_vals)]

    im = ax.imshow(bott_vals, extent=[phi_range[0], phi_range[1], M_range[0], M_range[1]], origin='lower', aspect='auto', cmap=cmap, interpolation='none', rasterized=True)

    if doBoundary:
        boundary_vals = np.linspace(-np.pi, np.pi, 500)
        ax.plot(boundary_vals, np.sin(boundary_vals)*np.sqrt(3)*3, c='k', ls='--', alpha=0.5)
        ax.plot(boundary_vals, -np.sin(boundary_vals)*np.sqrt(3)*3, c='k', ls='--', alpha=0.5)

    ax.set_title(str(titleparams))
    if doLabels:
        ax.set_xlabel('Phi')
        ax.set_ylabel('M', rotation=0)

    ax.set_xticks(np.arange(-np.pi, np.pi + np.pi/2, np.pi/2))
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    y_min, y_max = np.min(M_vals), np.max(M_vals)
    ax.set_yticks([-3*np.sqrt(3), 0, 3*np.sqrt(3)])
    ax.set_yticklabels([r'$-3\sqrt{3}$', r'$0$', r'$3\sqrt{3}$'])

    if doCbar:
        cbar = fig.colorbar(im, ax=ax)
        bott_min, bott_max = np.nanmin(bott_vals), np.nanmax(bott_vals)
        cbar_ticks = np.linspace(min(bott_min, -1), max(bott_max, 1), 5)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{tick:.{1}f}' for tick in cbar_ticks])

    return fig, ax


def compute_disorder_array(strength, system_size, df):
    disorder_array = np.random.uniform(-strength/2, strength/2, size=system_size)
    delta = np.sum(disorder_array)/system_size
    disorder_array -= delta
    disorder_array = np.repeat(disorder_array, df)
    return np.diag(disorder_array).astype(np.complex128)


def bott_from_hamiltonian(H, method, geometry_data):
    x, y = geometry_data['x'], geometry_data['y']
    eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)
    if method in ['site_elim', 'renorm']:
        hexaflake = geometry_data['hexaflake']
        x, y = x[hexaflake], y[hexaflake]
    return compute_bott_index({'x':x, 'y':y, 'eigenvalues':eigenvalues, 'eigenvectors':eigenvectors})


def compute_phase(method, generation, dimensions=(50,50), M_range=(-5.5,5.5), phi_range=(-np.pi, np.pi), t1=1.0, t2=1.0, n_jobs=-2, show_progress=True, directory='', fileOverwrite=False):
    
    M_values = np.linspace(M_range[0], M_range[1], dimensions[1])
    phi_values = np.linspace(phi_range[0], phi_range[1], dimensions[0])
    geometry_data = compute_geometric_data(generation, True)

    out_filename = directory+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]}).h5"
    if os.path.exists(out_filename) and fileOverwrite == False:
        return out_filename

    def worker_function(parameters):
        phi, M = parameters
        try:
            H = compute_hamiltonian(method, M, phi, t1, t2, geometry_data)
            bott = bott_from_hamiltonian(H, method, geometry_data)
            return [phi, M, bott]
        
        except Exception as e:
            print(f"Error for phi,M=({phi},{M}) : {e}")
            return [phi, M, np.nan]
        
    param_values = tuple(product(phi_values, M_values))

    if show_progress:
        with tqdm_joblib(tqdm(total=len(param_values), desc=f"Computing undisordered phase diagram ({method})")) as progress_bar:
            phi_data, M_data, bi_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(params) for params in param_values)).T
    else:
        phi_data, M_data, bi_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(params) for params in param_values)).T
    
    data = {'phi': phi_data,
            'M': M_data,
            'bott_index': bi_data.reshape(dimensions)}
    
    with h5py.File(out_filename, 'w') as f:
        for k, v in zip(data.keys(), data.values()):
            f.create_dataset(name=k, data=v)

    return out_filename


def compute_disorder_iterations(phi, M, method, strength, t1, t2, geometry_data, iterations=100, n_jobs=-2, show_progress=False):

    def worker_function(i):
        clean_H = compute_hamiltonian(method, M, phi, t1, t2, geometry_data)
        disorder_arr = compute_disorder_array(strength, clean_H.shape[0], 1)
        disorder_H = clean_H + disorder_arr
        bott = bott_from_hamiltonian(disorder_H, method, geometry_data)
        return bott
    
    if show_progress:
        with tqdm_joblib(tqdm(total=iterations, desc="Computing disorder iterations")) as progress_bar:
            iter_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(i) for i in range(iterations)))
    else:
        iter_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(i) for i in range(iterations)))

    return np.average(iter_data[~np.isnan(iter_data)])
    
    
def compute_disorder(in_filename, method, generation, strength, iterations=100, t1=1.0, t2=1.0, n_jobs=-2, intermittent_saving=True, show_progress = True, directory='', fileOverwrite=False):
    geometry_data = compute_geometric_data(generation, True)

    with h5py.File(in_filename, 'r') as f:
        phi_vals = f['phi'][:]
        M_vals = f['M'][:]
        bott_index_vals = f['bott_index'][:]

    manager = Manager()
    lock = manager.Lock()
    out_filename = in_filename.replace('.h5', f'_w{strength}.h5')
    if os.path.exists(out_filename) and fileOverwrite == False:
        return out_filename
    
    def worker_function(index):
        phi, M = phi_vals[index], M_vals[index]
        avg_bott = compute_disorder_iterations(phi, M, method, strength, t1=t1, t2=t2, geometry_data=geometry_data, iterations=iterations, n_jobs=n_jobs)
        if intermittent_saving:
            with lock:
                with h5py.File(out_filename, 'a') as f:
                    f['disorder_flat'][index] = avg_bott
                    f['computed_idxs'][index] = True
        return avg_bott

    disorder_bott_arr = np.zeros(phi_vals.shape)

    if not os.path.exists(out_filename):
        with h5py.File(out_filename, 'a') as f:
            f.create_dataset(name='disorder_flat', data=disorder_bott_arr)
            f.create_dataset(name='computed_idxs', data=disorder_bott_arr.astype(bool))
            f.create_dataset(name='disorder', data=np.zeros(bott_index_vals.shape))

    with h5py.File(out_filename, 'r') as f:
        wasComputed = f['computed_idxs'][:].flatten()
    nonzero_indices = bott_index_vals.astype(bool).flatten()
    compute_these = np.argwhere(nonzero_indices & ~wasComputed).flatten()


    if not np.any(compute_these):
        print(f"All disorder values already computed for {method}, W = {strength}.")
        return out_filename

    if show_progress:
        with tqdm_joblib(tqdm(total=len(compute_these), desc=f"Computing disorder values ({method}): W = {strength}")) as progress_bar:
            disorder_averages = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(i) for i in compute_these)).T
    else:
        disorder_averages = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(i) for i in compute_these)).T

    with h5py.File(out_filename, 'a') as f:
        if not intermittent_saving:
            disorder_bott_arr[compute_these] = disorder_averages
            f['disorder'][:] = disorder_bott_arr.reshape(bott_index_vals.shape)
        else:
            saved_disorder = f['disorder_flat'][:]
            if np.any(saved_disorder[compute_these]-disorder_averages):
                raise ValueError("Disorder values do not match between saved and computed values."
                                    +f"Saved: {saved_disorder[compute_these]}, Computed: {disorder_averages}")
            f['disorder'][:] = saved_disorder.reshape(bott_index_vals.shape)
    return out_filename

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def compute_and_save_phase_diagram(method, generation, disorder_strength, dimensions, iterations, n_jobs, directory):
    clean_file = compute_phase(method, generation, n_jobs=n_jobs, dimensions=dimensions, directory=directory)
    disorder_file = compute_disorder(clean_file, method, generation, disorder_strength, iterations=iterations, n_jobs=n_jobs, directory=directory, intermittent_saving=True, show_progress=True)
    return clean_file, disorder_file


def plot_comparison(clean_file, disorder_vals, method, generation, disorder_strength):
    with h5py.File(clean_file, 'r') as f:
        clean_dict = {k: v[:] for k, v in zip(f.keys(), f.values())}
    
    with h5py.File(disorder_vals, 'r') as f:
        disorder_vals = f['disorder'][:].T

    phi_vals, M_vals, bott_index_vals = clean_dict['phi'], clean_dict['M'], clean_dict['bott_index'].T

    for Z, tp, fp in zip([bott_index_vals, disorder_vals], ['Clean', f'Disordered (W={disorder_strength})'], ['clean', 'disorder']):
        fig, axs = plt.subplots(1, 1)
        fig, axs = plot_phase_diagram(fig, axs, phi_vals, M_vals, Z, titleparams=f'{tp}, {method}, g{generation}')
        if tp[0] == 'C':
            plt.savefig(f"{method}_g{generation}_{fp}.png")
        else:
            plt.savefig(f"{method}_g{generation}_w{disorder_strength}_{fp}.png")


def make_large_figure(directory='Haldane_Disorder_Data/Res2500_Avg100/', dimensions=(50,50)):
    files = glob.glob(os.path.join(directory, '*.h5'))
    disorder_strengths = [0.0]
    for file in files:
        filename = os.path.basename(file)
        if '_w' in filename:
            try:
                w_value = float(filename.split('_w')[1].split('.h5')[0])
                disorder_strengths.append(w_value)
            except ValueError:
                continue
    disorder_strengths = np.sort(np.unique(np.array(disorder_strengths)))


    fig, axs = plt.subplots(3, len(disorder_strengths), figsize=(35,10), sharex=True, sharey=True)
    methods = ['hexagon', 'renorm', 'site_elim']
    generation = 2
    cmap = 'viridis'
    bool_dict = {"doLabels": False, "doBoundary": False, "doCbar": False}

    for i, method in enumerate(methods):
        for j, w in enumerate(disorder_strengths):
            try:
                clean_file = directory + f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]}).h5"
                disorder_file = clean_file.replace('.h5', f'_w{w}.h5')
                with h5py.File(clean_file, 'r') as f:
                    clean_dict = {k: v[:] for k, v in zip(f.keys(), f.values())}

                phi_vals, M_vals, bott_index_vals = clean_dict['phi'], clean_dict['M'], clean_dict['bott_index'].T
                if j == 0:
                    fig, axs[i, j] = plot_phase_diagram(fig, axs[i,j], phi_vals, M_vals, bott_index_vals, titleparams=f"Undisordered", cmap=cmap, **bool_dict)
                else:
                    if os.path.exists(disorder_file):
                        with h5py.File(disorder_file, 'r') as f:
                            disorder_vals = f['disorder'][:].T  
                        fig, axs[i, j] = plot_phase_diagram(fig, axs[i,j], phi_vals, M_vals, disorder_vals, titleparams=f"W = {w}", cmap=cmap, **bool_dict)
                    else:
                        print(f"Disorder file {disorder_file} not found.")
            except Exception as e:
                print(f"Exception: {e}")

    method_labels = ["Honeycomb", "Renormalization", "Site Elimination"]
    for i, method in enumerate(method_labels):
        axs[i, 0].set_ylabel('M', fontsize=12, rotation=0)
        axs[i, 0].annotate(method, xy=(-0.3, 0.5), xytext=(-axs[i, 0].yaxis.labelpad - 5, 0),
                   xycoords=axs[i, 0].yaxis.label, textcoords='offset points',
                   size='large', ha='center', va='center', rotation=90)
        
    fig.suptitle('Bott Index for Various Methods and Disorders', fontsize=16)
    for ax in axs[-1, :]:
        ax.set_xlabel(r'$\phi$', fontsize=12)

    for ax in axs.flatten():
        t = np.linspace(-np.pi, np.pi, 1000)
        #ax.plot(t, np.sin(t)*np.sqrt(3)*3, c='k', ls=(0, (5, 1)), alpha=0.25)
        #ax.plot(t, -np.sin(t)*np.sqrt(3)*3, c='k', ls=(0, (5, 1)), alpha=0.25)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    pos1 = axs[0,0].get_position()
    pos2 = axs[-1,-1].get_position()
    cbar_ax = fig.add_axes([0.9, pos2.y0, 0.02, pos1.y1 - pos2.y0])
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Bott Index', fontsize=12)
    plt.show()
    for ax in axs.flatten():
        ax.set_aspect('equal', adjustable='box')

def compute_many_phase_diagrams():
    for method in ['renorm', 'site_elim', 'hexagon']:
        for W in (np.arange(10.0)+1.0):
            t0 = time()
            compute_and_save_phase_diagram(method, 2, W, (50, 50), 100, 4, 'Haldane_Disorder_Data/Res2500_Avg100/')
            print(f"Time for {method}, W={W}: {time()-t0:.2f} seconds.")

def probe_files(directory):
    for f in glob.glob(directory+'*.h5'):
        with h5py.File(f, 'r') as file:
            print(f)
            for k in file.keys():
                print(f"{k}: {file[k][:]}")
            print('\n')

def plot_files():
    directory = 'Haldane_Disorder_Data/Res2500_Avg100/'
    val_list = []
    for f in glob.glob(directory+'*.h5'):
        with h5py.File(f, 'r') as file:
            if 'disorder' in file.keys():
                val_list.append(file['disorder'][:])
    for v in val_list:
        plt.plot(range(len(v)), v)

        plt.show()

if __name__ == "__main__":
    compute_many_phase_diagrams()
    #make_large_figure(dimensions=(25,25))
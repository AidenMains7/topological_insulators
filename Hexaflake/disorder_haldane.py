import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import h5py, os, glob
from itertools import product
from tqdm_joblib import tqdm_joblib, tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager
from time import time
from fractions import Fraction
from MaybeActualFinalHaldane2 import compute_bott_index, compute_geometric_data, compute_hamiltonian, compute_disorder_array
from matplotlib.colors import ListedColormap, BoundaryNorm
import traceback



def compute_bott_from_hamiltonian(H, method, geometry_data):
    x, y = geometry_data['x'], geometry_data['y']
    eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)
    if method in ['site_elim', 'renorm']:
        hexaflake = geometry_data['hexaflake']
        x, y = x[hexaflake], y[hexaflake]
    return compute_bott_index({'x':x, 'y':y, 'eigenvalues':eigenvalues, 'eigenvectors':eigenvectors, 'S':geometry_data['x'].size})

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
# region Parallel Computation

def compute_phase(method, generation, dimensions=(50,50), M_range=(-5.5,5.5), phi_range=(-np.pi, np.pi), t1=1.0, t2=1.0, 
                  n_jobs=-2, show_progress=True, directory='', fileOverwrite=False,
                  M_values=None, phi_values=None, outfname:"str|None"=None):
    if M_values is None:
        M_values = np.linspace(M_range[0], M_range[1], dimensions[1])
    if phi_values is None:
        phi_values = np.linspace(phi_range[0], phi_range[1], dimensions[0])
    geometry_data = compute_geometric_data(generation, True)

    out_filename = directory+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]}).h5" if outfname is None else outfname
    if os.path.exists(out_filename) and fileOverwrite == False:
        return out_filename

    def worker_function(parameters):
        phi, M = parameters
        try:
            H = compute_hamiltonian(method, M, phi, t1, t2, geometry_data)
            bott = compute_bott_from_hamiltonian(H, method, geometry_data)
            return [phi, M, bott]
        
        except Exception as e:
            print(f"Error for phi,M=({phi},{M}) : {e}")
            return [phi, M, np.nan]
        
    param_values = tuple(product(phi_values, M_values))
    print(param_values[0])
    if show_progress:
        with tqdm_joblib(tqdm(total=len(param_values), desc=f"Computing undisordered phase diagram ({method})")) as progress_bar:
            phi_data, M_data, bi_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(params) for params in param_values)).T
    else:
        phi_data, M_data, bi_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(params) for params in param_values)).T
    
    data = {'phi': phi_data,
            'M': M_data,
            'bott_index': bi_data}
    
    with h5py.File(out_filename, 'w') as f:
        for k, v in zip(data.keys(), data.values()):
            f.create_dataset(name=k, data=v)

    return out_filename


def compute_disorder_iterations(phi, M, method, strength, t1, t2, geometry_data, iterations=100, n_jobs=-2, show_progress=False):
    def worker_function(i):
        H = compute_hamiltonian(method, M, phi, t1, t2, geometry_data, strength, True if method == 'renorm1' else False)
        bott = compute_bott_from_hamiltonian(H, method, geometry_data)
        return bott
    
    if show_progress:
        with tqdm_joblib(tqdm(total=iterations, desc="Computing disorder iterations")) as progress_bar:
            iter_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(i) for i in range(iterations)))
    else:
        iter_data = np.array(Parallel(n_jobs=n_jobs)(delayed(worker_function)(i) for i in range(iterations)))

    # Return the full array of all iterations instead of just the average
    return iter_data


def compute_disorder(in_filename, method, generation, strength, iterations=100, t1=1.0, t2=1.0, n_jobs=-2, show_progress=True, fileOverwrite=False, doHalf:bool = True):
    geometry_data = compute_geometric_data(generation, True)

    with h5py.File(in_filename, 'r') as f:
        phi_vals = f['phi'][:] # type: ignore
        M_vals = f['M'][:] # type: ignore
        bott_index_vals = f['bott_index'][:] # type: ignore

    out_filename = in_filename.replace('.h5', f'_w{strength}.h5')
    if method in ['renorm1', 'renorm2']:
        out_filename = out_filename.replace('renorm', method)

    if os.path.exists(out_filename) and fileOverwrite == False:
        return out_filename
    
    def worker_function(index):
        phi, M = phi_vals[index], M_vals[index] # type: ignore
        all_botts = compute_disorder_iterations(phi, M, method, strength, t1=t1, t2=t2, geometry_data=geometry_data, iterations=iterations, n_jobs=1)
        return phi, M, all_botts

    # Use a flat mask so the half-selection only depends on phi sign.
    nonzero_indices = bott_index_vals.astype(bool).ravel() # type: ignore
    phi_flat = phi_vals.ravel() # type: ignore
    if doHalf:
        compute_mask = np.logical_and(nonzero_indices, phi_flat >= np.pi / 2)
    else:
        compute_mask = nonzero_indices

    compute_these = np.flatnonzero(compute_mask)
    print(len(compute_these))

    if not np.any(compute_these):
        print(f"All disorder values already computed for {method}, W = {strength}.")
        return out_filename

    # Split the required computations into chunks (e.g., 10 separate batch files)
    num_chunks = 10
    chunks = np.array_split(compute_these, min(num_chunks, len(compute_these)))
    chunk_files = []

    for chunk_idx, chunk in enumerate(chunks):
        if len(chunk) == 0: continue
        
        if show_progress:
            print(f"Computing chunk {chunk_idx + 1}/{len(chunks)} for W = {strength}")
            with tqdm_joblib(tqdm(total=len(chunk), desc=f"Chunk {chunk_idx + 1}")) as progress_bar:
                results = Parallel(n_jobs=n_jobs)(delayed(worker_function)(i) for i in chunk)
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(worker_function)(i) for i in chunk)

        # Unpack results
        phis_chunk = np.array([res[0] for res in results]) # type: ignore
        Ms_chunk = np.array([res[1] for res in results]) # type: ignore
        botts_chunk = np.array([res[2] for res in results]) # Shape will be (len(chunk), iterations) # type: ignore

        # Save to temporary chunk file
        chunk_filename = out_filename.replace('.h5', f'_temp_chunk_{chunk_idx}.h5')
        with h5py.File(chunk_filename, 'w') as f:
            f.create_dataset('phi', data=phis_chunk)
            f.create_dataset('M', data=Ms_chunk)
            f.create_dataset('disorder_all', data=botts_chunk)
        
        chunk_files.append(chunk_filename)

    # Compile all chunk files into the final h5 file
    all_phis, all_Ms, all_botts = [], [], []

    for c_file in chunk_files:
        try:
            with h5py.File(c_file, 'r') as f:
                all_phis.append(f['phi'][:]) # type: ignore
                all_Ms.append(f['M'][:]) # type: ignore
                all_botts.append(f['disorder_all'][:]) # type: ignore
            os.remove(c_file) # Clean up intermediate files
        except Exception as e:
            print(f"Error compiling file {c_file}: {e}")

    # Concatenate the lists of arrays
    final_phis = np.concatenate(all_phis)
    final_Ms = np.concatenate(all_Ms)
    final_botts_all = np.concatenate(all_botts, axis=0)
    
    # Calculate the average ignoring NaNs so your existing plotting scripts still work
    final_botts_avg = np.nanmean(final_botts_all, axis=1)

    with h5py.File(out_filename, 'w') as f:
        f.create_dataset(name='phi', data=final_phis)
        f.create_dataset(name='M', data=final_Ms)
        f.create_dataset(name='disorder_all', data=final_botts_all) # New: 2D array of all iterations
        f.create_dataset(name='disorder', data=final_botts_avg)     # Legacy: 1D array of averages

    return out_filename

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
# region Plotting Functions

def plot_phase_diagram(fig, ax, 
                       X_values, Y_values, Z_values, 
                       labels:list=[], title:str="", 
                       X_ticks=None, Y_ticks=None, X_tick_labels=None, Y_tick_labels=None,
                       cbar_ticks=None, cbar_tick_labels=None,
                       cmap='Spectral', norm=None,
                       plotColorbar=True,
                       plotFull:bool = False):

    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]

    if np.ndim(Z_values) != 2 or Z_values.shape != (len(np.unique(Y_values)), len(np.unique(X_values))):
        Z_values = Z_values.reshape(len(np.unique(Y_values)), len(np.unique(X_values))).T

    im = ax.imshow(Z_values, extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]], 
                   origin='lower', aspect='auto', cmap=cmap, interpolation='none', 
                   rasterized=True, norm=norm)
    if plotFull:
        im2 = ax.imshow(np.flipud(Z_values), extent=[X_range[0], X_range[1], -Y_range[1], Y_range[0]], 
                    origin='lower', aspect='auto', cmap=cmap, interpolation='none', 
                    rasterized=True, norm=norm)
        im3 = ax.imshow(-Z_values, extent=[-X_range[1], X_range[0], Y_range[0], Y_range[1]], 
                    origin='lower', aspect='auto', cmap=cmap, interpolation='none', 
                    rasterized=True, norm=norm)
        im4 = ax.imshow(np.flipud(-Z_values), extent=[-X_range[1], X_range[0], -Y_range[1], Y_range[0]], 
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


def get_all_files_matching_criteria(files, contains_all:"list|None"=None, contains_any:"list|None"=None, does_not_contain:"list|None"=None):
    if contains_all is not None:
        files = [file for file in files if all([c in file for c in contains_all])]
    if contains_any is not None:
        files = [file for file in files if any([c in file for c in contains_any])]
    if does_not_contain is not None:
        files = [file for file in files if all([c not in file for c in does_not_contain])]
    return files


def get_disorder_strength_from_files(files):
    # A static method that only works for the current naming convention.
    disorder_strengths = []
    for file in files:
        filename = os.path.basename(file)
        if '_w' in filename:
            try:
                disorder_strength = float(filename.split('_w')[1].split('.h5')[0])
                disorder_strengths.append(disorder_strength)
            except ValueError:
                continue
    return np.sort(np.unique(disorder_strengths))


def global_bounds(arrays:list, returnAbsBounds=True):
    # Get maximum and minimum values from list of arrays
    global_min, global_max = 0.0, 0.0
    for arr in arrays:
        global_min = min(global_min, np.nanmin(arr))
        global_max = max(global_max, np.nanmax(arr))
    abs_max = max(np.abs(global_min), np.abs(global_max))
    if returnAbsBounds:
        return -abs_max, abs_max
    else:
        return global_min, global_max


def extract_data_from_h5_file(filename):
    try:
        with h5py.File(filename, 'r') as f:
            data = {k: v[:] for k, v in zip(f.keys(), f.values())}
        return data
    except Exception as e:
        print(f"Error extracting data from file: {e}")
        return None


def add_colorbar_to_figure(fig, axs, norm, cmap, cbar_label=None):
    plt.tight_layout(rect=(0, 0, 0.9, 1))
    axs_flattened = axs.flatten()
    pos1 = axs_flattened[0].get_position()
    pos2 = axs_flattened[-1].get_position()
    cbar_ax = fig.add_axes([0.9, pos2.y0, 0.02, pos1.y1 - pos2.y0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=16)
    
    norm_min, norm_max = norm.vmin, norm.vmax
    cbar.ax.yaxis.set_ticks(np.linspace(norm_min, norm_max, num=5))
    cbar.ax.tick_params(labelsize=14)

    return cbar
    

def pi_tick_labels(value):
    value /= np.pi
    fractional_value = Fraction(value).limit_denominator(10)
    if np.isclose(fractional_value.numerator, 0):
        return 0
    sign = "-" if fractional_value.numerator < 0 else ""
    if abs(fractional_value.numerator) == 1:
        numerator = "$\\pi$"
    else:
        numerator = f"{abs(fractional_value.numerator)}$\\pi$"
    if fractional_value.denominator == 1:
        return sign + numerator
    else:
        return sign + f"$\\frac{{{numerator.replace('$', ''	)}}}{{{fractional_value.denominator}}}$"

# endregion
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------


def make_large_figure(generation:int, dimensions:tuple, methods:list, disorder_strengths=None, 
                      directory=".", cmap="cividis", 
                      plotUndisordered=True, plotSineBoundary=True, 
                      row_labels=None, column_labels=None, title:str="", image_filename=None, plotFull=False):
    
    if type(methods) is str:
        methods = [methods]
    if any([m in methods for m in ['hexagon', 'site_elim', 'renorm1', 'renorm2']]) == False:

        raise ValueError("Invalid method. Options are ['hexagon', 'site_elim', 'renorm1', 'renorm2']")
    
    and_contain_list = [f'g{generation}', f'({dimensions[0]}_by_{dimensions[1]})']
    or_contain_list = methods

    files = glob.glob(os.path.join(directory, f'*.h5'))
    files = get_all_files_matching_criteria(files, contains_all=and_contain_list, contains_any=or_contain_list, does_not_contain=['_array'])
    
    if disorder_strengths is None:
        disorder_strengths = get_disorder_strength_from_files(files)

    n_cols = len(disorder_strengths) + 1 if plotUndisordered else len(disorder_strengths)
    fig, axs = plt.subplots(len(methods), n_cols, figsize=(30, 20), sharex=True, sharey=True)
    if len(methods) == 1:
        axs = axs.reshape(1, len(axs))
    elif n_cols == 1:
        axs = axs.reshape(len(methods), 1)

    clean_files = [file for file in files if 'w' not in file]
    disorder_files = [file for file in files if 'w' in file]

    clean_data = [extract_data_from_h5_file(file) for file in clean_files]
    disorder_data = [extract_data_from_h5_file(file) for file in disorder_files]

    clean_bott_data = [data['bott_index'].T for data in clean_data] # type: ignore
    disorder_bott_data = [data['disorder'] for data in disorder_data]	 # type: ignore
    
    if plotFull:
        X_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        X_tick_labels = ['$-1$', '$\\frac{1}{2}$', '$0$', '$\\frac{1}{2}$', '$1$']
        Y_ticks = [-3*np.sqrt(3), 0, 3*np.sqrt(3)]
        Y_tick_labels = ["$-3 \\sqrt{3}$", "0", "$3 \\sqrt{3}$"]
    else:
        X_ticks = [0., np.pi / 2, np.pi]
        X_tick_labels = ['0', '$\\frac{1}{2}$', '1']
        Y_ticks = [0., 3*np.sqrt(3)]
        Y_tick_labels = ['0', '$3 \\sqrt{3}$']
    tick_dict = {'X_ticks': X_ticks, 'X_tick_labels': X_tick_labels, 'Y_ticks': Y_ticks, 'Y_tick_labels': Y_tick_labels}

    global_min, global_max = global_bounds(clean_bott_data+disorder_bott_data)
    if plotFull:
        norm = plt.Normalize(vmin=min(global_min, -1.0), vmax=max(global_max, 1.0)) # type: ignore
    else:
        norm = plt.Normalize(vmin=-1.0, vmax=0.0) # type: ignore
    
    clean_files_array = np.empty((len(methods)), dtype=object)
    files_array = np.empty((len(methods), n_cols), dtype=object)
    for i, method in enumerate(methods):
        clean_files_array[i] = directory+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]}).h5"
        for j, disorder_strength in enumerate(disorder_strengths):
            files_array[i, j] = directory+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]})_w{disorder_strength}.h5"

    for i in range(len(methods)):
        clean_file = clean_files_array[i]
        try:
            loop_clean_data = extract_data_from_h5_file(clean_file)
            phi_values, M_values, bott_values = loop_clean_data['phi'], loop_clean_data['M'], loop_clean_data['bott_index'].T # type: ignore
        except Exception as e:
            print(f"Exception: {e}")
        for j in range(n_cols):
            if plotUndisordered:
                disorder_file = files_array[i, j - 1]
            else:
                disorder_file = files_array[i, j]
            try:
                if plotUndisordered and j == 0:
                    fig, axs[i, j] = plot_phase_diagram(fig, axs[i, j], phi_values, M_values, bott_values, cmap=cmap, norm=norm, **tick_dict, plotColorbar=False, plotFull = plotFull)
                else:
                    loop_disorder_data = extract_data_from_h5_file(disorder_file)
                    if loop_disorder_data is not None:
                        try:
                            disorder_values = loop_disorder_data['disorder'].T
                        except (KeyError, TypeError, AttributeError) as e0:
                            print(f"Missing or invalid 'disorder' data for axes[{i}, {j}] ({disorder_file}): {e0}")
                            continue

                        x_phi = phi_values
                        y_M = M_values

                        if ('phi' in loop_disorder_data) and ('M' in loop_disorder_data):
                            try:
                                x_phi = loop_disorder_data['phi']
                                y_M = loop_disorder_data['M']
                            except (TypeError, KeyError):
                                x_phi = phi_values
                                y_M = M_values

                        fig, axs[i, j] = plot_phase_diagram(
                            fig,
                            axs[i, j],
                            x_phi,
                            y_M,
                            disorder_values,
                            cmap=cmap,
                            norm=norm,
                            **tick_dict,
                            plotColorbar=False,
                            plotFull=plotFull,
                        )
            except Exception as e1:
                print(type(e1))
                print(f"Error plotting in axes[{i}, {j}]: {e1}")

    if row_labels is None:
        row_labels = methods
    for i, row_label in enumerate(row_labels):
        axs[i, 0].set_ylabel('M', fontsize=12, rotation=0)
        axs[i, 0].annotate(row_label, xy=(-0.3, 0.5), xytext=(-axs[i, 0].yaxis.labelpad - 5, 0),
                   xycoords=axs[i, 0].yaxis.label, textcoords='offset points',
                   size=16, ha='center', va='center', rotation=90)
        
    if column_labels is None:
        column_labels = [f"W = {strength}" for strength in disorder_strengths]
        if plotUndisordered:
            column_labels = ["Undisordered"] + column_labels
    for j, column_label in enumerate(column_labels):
        axs[-1, j].set_xlabel('$\\phi / \\pi$', fontsize=12)
        axs[0, j].set_title(column_label, fontsize=12)
        
    fig.suptitle(title, fontsize=16)

    if plotSineBoundary:
        for ax in axs.flatten():
            xmin, xmax = ax.get_xlim()
            t = np.linspace(xmin, xmax, 1000)
            ax.plot(t, np.sin(t)*np.sqrt(3)*3, c='k', ls=(0, (5, 1)), alpha=1., zorder=3)
            if ax.get_ylim()[0] < 0:
                ax.plot(t, -np.sin(t)*np.sqrt(3)*3, c='k', ls=(0, (5, 1)), alpha=1., zorder=3)

    for ax in axs.flatten():
        ax.tick_params(axis='both', labelsize=20)
        ax.set_title(ax.get_title(), fontsize=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax.set_aspect('equal')

    add_colorbar_to_figure(fig, axs, norm, cmap, "Bott Index")
    if image_filename is not None:
        plt.savefig(image_filename, bbox_inches='tight', transparent=False)


def compute_many_phase_diagrams(generation, disorder_strengths, methods, dimensions=(50,50), iterations=100, n_jobs=6, directory=".", doHalf:bool = True):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for disorder_strength in disorder_strengths:
        for method in methods:
            # Fix: Handle 'renorm' cleanly alongside 'renorm1' and 'renorm2' so clean_file is never None
            if method in ['renorm1', 'renorm2', 'renorm']:
                clean_file = compute_phase('renorm', generation, n_jobs=n_jobs, dimensions=dimensions, directory=directory, M_range=(0., 5.5), phi_range=(0., np.pi))
            else:
                clean_file = compute_phase(method, generation, n_jobs=n_jobs, dimensions=dimensions, directory=directory, M_range=(0., 5.5), phi_range=(0., np.pi))
            
            disorder_file = compute_disorder(clean_file, method, generation, disorder_strength, iterations=iterations, n_jobs=n_jobs, show_progress=True, doHalf=doHalf)


def compare_generations():
    generations = [2, 3, 4]
    method = 'site_elim'	
    resolution = (25, 25)
    directory = "./Hexaflake/Data/"
    files = [directory + f"{method}_g{gen}_({resolution[0]}_by_{resolution[1]}).h5" for gen in generations]

    fig, axs = plt.subplots(1, len(generations), figsize=(4 * len(generations), 4))

    file_data = [extract_data_from_h5_file(file) for file in files]
    M_data = [data["M"] for data in file_data] # type: ignore
    phi_data = [data["phi"] for data in file_data] # type: ignore
    bi_data = [np.round(data["bott_index"].flatten(), 3) for data in file_data] # type: ignore

    cmap = 'viridis'
    unique_values = np.array((-1, 0))
    cmap = plt.get_cmap(cmap)
    discrete_colors = cmap(np.linspace(0, 1, len(unique_values)))
    cmap = ListedColormap(discrete_colors)
    norm = BoundaryNorm(boundaries=np.append(unique_values, unique_values[-1] + 1), ncolors=len(unique_values))

    scatters = []
    for i in range(len(generations)):
        scat = axs[i].scatter(phi_data[i], M_data[i], c=bi_data[i], norm=norm, cmap=cmap)
        scatters.append(scat)
    for ax in axs.flatten():
        ax.set_xlabel("$\\phi / \\pi$", fontsize=16)
    axs[0].set_ylabel("M", fontsize=16, rotation=0)


    total_number = resolution[0]*resolution[1]
    percentages = [np.sum(-bid*100/total_number) for bid in bi_data]
    for i in range(len(generations)):
        axs[i].set_title(f"Generation {generations[i]}: Percent Nontrivial = {percentages[i]:.2f}")


    cbar = fig.colorbar(scatters[0], ax=axs[-1])
    cbar.set_ticks(unique_values+0.5) # type: ignore
    cbar.set_ticklabels([str(val) for val in unique_values], fontsize=16)
    cbar.set_label("Bott Index", fontsize=16)


    plt.tight_layout()
    plt.show()
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def get_info(generation):
    geometry_data = compute_geometric_data(generation, True)
    x = geometry_data['x']
    hex = geometry_data['hexaflake']
    print("Pristine  N sites:", x.size)
    print("Hexaflake N sites:", np.sum(hex))


def gen4_points():
    with h5py.File("./Hexaflake/Data/site_elim_g4_(25_by_25).h5", 'r') as f:
        phi_vals = f['phi'][:] # type: ignore
        M_vals = f['M'][:] # type: ignore
        bott_index_vals = f['bott_index'][:] # type: ignore

    idxs = [(0, 4), (0, 5), (1, 6), (1, 7), (2, 9), (2, 10), (3, 11), (3, 12), (4, 13), 
         (5, 15), (6, 16), (7, 17), (8, 18), (6, 13), (7, 14), (8, 14), (9, 14),
         (6, 14), (7, 15), (8, 16), (9, 17), (10, 17)]

    phi_unique = np.unique(phi_vals) # type: ignore
    M_unique = np.unique(M_vals) # type: ignore

    parameters = []
    for i, j in idxs:
        parameters.append((phi_unique[i], M_unique[j]))

    compute_phase('site_elim', 4, dimensions=(25, 25), directory="./Hexaflake/Data/", 
               M_values = M_unique, phi_values = phi_unique, n_jobs=-4,
               outfname = 'site_elim_g4_selected_points.h5')
    

def main():
    compute_these_disorder_strengths = [1.0]
    plot_these_disorder_strengths = [1.0, 5., 7.5, 8., 10., 12.5]

    methods = ['site_elim']
    plot_methods = ['hexagon', 'renorm1', 'renorm2', 'site_elim']
    titles = ['Pristine', 'Renormalization 1', 'Renormalization 2', 'Site Elimination']
    res = (25, 25)
    generation = 4
    compute_many_phase_diagrams(generation, compute_these_disorder_strengths, methods, res, 
                                iterations=20, n_jobs=28, directory="./Hexaflake/Data/", doHalf=True)
    make_large_figure(generation, res, plot_methods, 
                   disorder_strengths=plot_these_disorder_strengths,
                   directory="./Hexaflake/Data/",
                   cmap="jet", 
                   plotUndisordered=True, plotSineBoundary=False, plotFull=False,
                   row_labels=titles,
                   title="", 
                    image_filename=f"./Hexaflake/Figures/PhaseDiagram_{plot_methods[0]}_g{generation}.png",)





if __name__ == "__main__":	
    main()
    #with h5py.File('./Hexaflake/Data/site_elim_g2_(25_by_25)_w8.0.h5', 'r') as f:
    #   M = f['M'][:] # type: ignore
    #   disorder = f["disorder"][:] # type: ignore
    #   disorder_all = f["disorder_all"][:] # type: ignore
    #   phi = f["phi"][:] # type: ignore
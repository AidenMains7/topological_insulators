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
from HaldaneModel import compute_geometric_data, compute_hamiltonian, compute_bott_from_hamiltonian
from matplotlib.colors import ListedColormap, BoundaryNorm
import traceback




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

    directory = directory.split("New/")[0] + f"/Generation {generation}/"

    if method in ['renorm1', 'renorm2']:
        out_filename = directory+f"renorm_g{generation}_({dimensions[0]}_by_{dimensions[1]}).h5" if outfname is None else outfname
    else:
        out_filename = directory+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]}).h5" if outfname is None else outfname
    print(out_filename)
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


def compute_disorder(in_filename, method, generation, strength, iterations=100, t1=1.0, t2=1.0, n_jobs=-2, show_progress=True, fileOverwrite=False, doHalf:bool = True, num_chunks:int = 10):
    """Compute disorder samples by flattening all jobs and running them in chunks.

    This preserves the same output layout as compute_disorder, but avoids the
    nested Parallel call pattern by scheduling each (phi, M, disorder_index)
    sample as a single job.
    """
    geometry_data = compute_geometric_data(generation, True)

    with h5py.File(in_filename, 'r') as f:
        phi_vals = f['phi'][:] # type: ignore
        M_vals = f['M'][:] # type: ignore
        bott_index_vals = f['bott_index'][:] # type: ignore

    if method in ['renorm1', 'renorm2']:
        in_filename = in_filename.replace('renorm', method)
    out_filename = in_filename.replace('.h5', f'_w{strength}.h5')

    if os.path.exists(out_filename) and fileOverwrite == False:
        print(out_filename)
        return out_filename

    bott_index_vals = np.round(np.asarray(bott_index_vals), 8)
    nonzero_indices = bott_index_vals.astype(bool).ravel() # type: ignore
    phi_flat = phi_vals.ravel() # type: ignore
    M_flat = M_vals.ravel() # type: ignore
    if doHalf:
        compute_mask = np.logical_and(nonzero_indices, phi_flat >= np.pi / 2)
    else:
        compute_mask = nonzero_indices

    compute_these = np.flatnonzero(compute_mask)
    print(len(compute_these))

    if not np.any(compute_these):
        print(f"All disorder values already computed for {method}, W = {strength}.")
        return out_filename

    jobs = [(index, disorder_index) for index in compute_these for disorder_index in range(iterations)]
    chunk_count = min(num_chunks, len(jobs))
    job_chunk_size = int(np.ceil(len(jobs) / chunk_count))
    job_chunks = [jobs[i:i + job_chunk_size] for i in range(0, len(jobs), job_chunk_size)]

    def worker_function(job):
        index, disorder_index = job
        phi, M = phi_flat[index], M_flat[index] # type: ignore
        H = compute_hamiltonian(
            method,
            M,
            phi,
            t1,
            t2,
            geometry_data,
            strength,
            True if method == 'renorm1' else False,
        )
        bott = compute_bott_from_hamiltonian(H, method, geometry_data)
        return index, disorder_index, phi, M, bott

    chunk_files = []
    for chunk_idx, chunk in enumerate(job_chunks):
        chunk_filename = out_filename.replace('.h5', f'_temp_chunk_{chunk_idx}.h5')
        if os.path.exists(chunk_filename) and not fileOverwrite:
            print(f"Chunk already computed for W = {strength}, skipping {chunk_filename}")
            chunk_files.append(chunk_filename)
            continue
        if os.path.exists(chunk_filename) and fileOverwrite:
            os.remove(chunk_filename)

        if show_progress:
            print(f"Computing job chunk {chunk_idx + 1}/{len(job_chunks)} for W = {strength}")
            with tqdm_joblib(tqdm(total=len(chunk), desc=f"Job chunk {chunk_idx + 1}")) as progress_bar:
                results = Parallel(n_jobs=n_jobs)(delayed(worker_function)(job) for job in chunk)
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(worker_function)(job) for job in chunk)

        phis_chunk = np.empty(len(chunk), dtype=float)
        Ms_chunk = np.empty(len(chunk), dtype=float)
        botts_chunk = np.full((len(chunk), iterations), np.nan, dtype=float)

        for chunk_row, (job, result) in enumerate(zip(chunk, results)):
            index, disorder_index = job
            _, _, phi, M, bott = result # type: ignore
            phis_chunk[chunk_row] = phi
            Ms_chunk[chunk_row] = M
            botts_chunk[chunk_row, disorder_index] = bott

        with h5py.File(chunk_filename, 'w') as f:
            f.create_dataset(name='phi', data=phis_chunk)
            f.create_dataset(name='M', data=Ms_chunk)
            f.create_dataset(name='disorder_all', data=botts_chunk)

        chunk_files.append(chunk_filename)

    all_phis = []
    all_Ms = []
    all_botts_all = []

    for c_file in chunk_files:
        try:
            with h5py.File(c_file, 'r') as f:
                all_phis.append(f['phi'][:]) # type: ignore
                all_Ms.append(f['M'][:]) # type: ignore
                all_botts_all.append(f['disorder_all'][:]) # type: ignore
            os.remove(c_file)
        except Exception as e:
            print(f"Error compiling file {c_file}: {e}")

    final_phis = np.concatenate(all_phis)
    final_Ms = np.concatenate(all_Ms)
    final_botts_all = np.concatenate(all_botts_all, axis=0)

    def _safe_row_mean(row):
        finite = np.isfinite(row)
        if not np.any(finite):
            return 0.0
        return float(np.mean(row[finite]))

    def _disorder_key(phi, M, precision=12):
        return (round(float(phi), precision), round(float(M), precision))

    def _build_disorder_rows(source_phi_vals, source_M_vals, source_botts_all, iterations_local, fill_value=0.0):
        source_phi_vals = np.asarray(source_phi_vals).ravel()
        source_M_vals = np.asarray(source_M_vals).ravel()
        source_botts_all = np.asarray(source_botts_all, dtype=float)

        if source_botts_all.ndim == 1:
            source_botts_all = source_botts_all[:, None]

        grouped_rows = {}
        for idx, (phi, M) in enumerate(zip(source_phi_vals, source_M_vals)):
            key = _disorder_key(phi, M)
            row = np.asarray(source_botts_all[idx], dtype=float).ravel()
            if key not in grouped_rows:
                grouped_rows[key] = np.full(iterations_local, fill_value, dtype=float)
            if row.size == 0:
                continue

            finite_indices = np.flatnonzero(np.isfinite(row))
            if finite_indices.size == 0:
                continue

            disorder_index = int(finite_indices[0])
            if disorder_index < iterations_local:
                grouped_rows[key][disorder_index] = float(row[disorder_index])

        return grouped_rows

    def _reindex_disorder_data(target_phi_vals, target_M_vals, source_phi_vals, source_M_vals, source_botts_all, fill_value=0.0):
        target_phi_vals = np.asarray(target_phi_vals).ravel()
        target_M_vals = np.asarray(target_M_vals).ravel()
        source_phi_vals = np.asarray(source_phi_vals).ravel()
        source_M_vals = np.asarray(source_M_vals).ravel()
        source_botts_all = np.asarray(source_botts_all)

        if source_botts_all.ndim == 1:
            source_botts_all = source_botts_all[:, None]

        iterations_local = source_botts_all.shape[1] if source_botts_all.ndim > 1 else 1
        full_botts_all = np.full((target_phi_vals.size, iterations_local), fill_value, dtype=float)
        full_botts_avg = np.full(target_phi_vals.size, fill_value, dtype=float)

        source_lookup = _build_disorder_rows(
            source_phi_vals,
            source_M_vals,
            source_botts_all,
            iterations_local,
            fill_value=fill_value,
        )

        for idx, (phi, M) in enumerate(zip(target_phi_vals, target_M_vals)):
            row = source_lookup.get(_disorder_key(phi, M))
            if row is None:
                continue

            full_botts_all[idx] = row
            full_botts_avg[idx] = _safe_row_mean(row)

        return full_botts_all, full_botts_avg

    full_phis, full_Ms = np.asarray(phi_vals).ravel(), np.asarray(M_vals).ravel()
    if doHalf:
        mirrored_phis = np.pi - final_phis
        mirrored_Ms = final_Ms.copy()
        mirrored_botts_all = final_botts_all.copy()

        mirrored_mask = np.logical_and(mirrored_phis > 0, mirrored_phis < np.pi)
        mirrored_mask &= ~np.isclose(mirrored_phis, final_phis)

        final_phis = np.concatenate([final_phis, mirrored_phis[mirrored_mask]])
        final_Ms = np.concatenate([final_Ms, mirrored_Ms[mirrored_mask]])
        final_botts_all = np.concatenate([final_botts_all, mirrored_botts_all[mirrored_mask]], axis=0)

        sort_idx = np.lexsort((final_Ms, final_phis))
        final_phis = final_phis[sort_idx]
        final_Ms = final_Ms[sort_idx]
        final_botts_all = final_botts_all[sort_idx]

    full_botts_all, final_botts_avg = _reindex_disorder_data(
        full_phis,
        full_Ms,
        final_phis,
        final_Ms,
        final_botts_all,
        fill_value=0.0,
    )

    with h5py.File(out_filename, 'w') as f:
        f.create_dataset(name='phi', data=full_phis)
        f.create_dataset(name='M', data=full_Ms)
        f.create_dataset(name='disorder_all', data=final_botts_all)
        f.create_dataset(name='disorder', data=final_botts_avg)

    return out_filename


def repair_disorder_file(disorder_filename, clean_filename=None, fileOverwrite=True, fill_value=0.0):
    """Rewrite a disorder file so phi/M match the clean input grid.

    Any grid points that are missing from the disorder file are filled with
    `fill_value` in both `disorder` and `disorder_all`.
    """
    if clean_filename is None:
        disorder_basename = os.path.basename(disorder_filename)
        if '_w' in disorder_basename:
            clean_filename = os.path.join(
                os.path.dirname(disorder_filename),
                disorder_basename.split('_w', 1)[0] + '.h5',
            )
        else:
            raise ValueError("Could not infer the matching clean filename.")

    if not os.path.exists(disorder_filename):
        raise FileNotFoundError(disorder_filename)
    if not os.path.exists(clean_filename):
        raise FileNotFoundError(clean_filename)

    with h5py.File(clean_filename, 'r') as f:
        clean_phi = f['phi'][:]  # type: ignore
        clean_M = f['M'][:]  # type: ignore

    with h5py.File(disorder_filename, 'r') as f:
        source_phi = f['phi'][:] if 'phi' in f else np.array([])  # type: ignore
        source_M = f['M'][:] if 'M' in f else np.array([])  # type: ignore
        if 'disorder_all' in f:
            source_disorder_all = f['disorder_all'][:]  # type: ignore
        elif 'disorder' in f:
            source_disorder_all = f['disorder'][:]  # type: ignore
        else:
            source_disorder_all = np.array([])

    def _safe_row_mean(row):
        finite = np.isfinite(row)
        if not np.any(finite):
            return fill_value
        return float(np.mean(row[finite]))

    def _disorder_key(phi, M, precision=12):
        return (round(float(phi), precision), round(float(M), precision))

    def _build_disorder_rows(source_phi_vals, source_M_vals, source_botts_all, iterations_local, fill_value=0.0):
        source_phi_vals = np.asarray(source_phi_vals).ravel()
        source_M_vals = np.asarray(source_M_vals).ravel()
        source_botts_all = np.asarray(source_botts_all, dtype=float)

        if source_botts_all.ndim == 1:
            source_botts_all = source_botts_all[:, None]

        grouped_rows = {}
        for idx, (phi, M) in enumerate(zip(source_phi_vals, source_M_vals)):
            key = _disorder_key(phi, M)
            row = np.asarray(source_botts_all[idx], dtype=float).ravel()
            if key not in grouped_rows:
                grouped_rows[key] = np.full(iterations_local, fill_value, dtype=float)
            finite_indices = np.flatnonzero(np.isfinite(row))
            if finite_indices.size == 0:
                continue
            disorder_index = int(finite_indices[0])
            if disorder_index < iterations_local:
                grouped_rows[key][disorder_index] = float(row[disorder_index])

        return grouped_rows

    def _reindex_disorder_data(target_phi_vals, target_M_vals, source_phi_vals, source_M_vals, source_botts_all):
        target_phi_vals = np.asarray(target_phi_vals).ravel()
        target_M_vals = np.asarray(target_M_vals).ravel()
        source_phi_vals = np.asarray(source_phi_vals).ravel()
        source_M_vals = np.asarray(source_M_vals).ravel()
        source_botts_all = np.asarray(source_botts_all)

        if source_botts_all.ndim == 1:
            source_botts_all = source_botts_all[:, None]

        iterations_local = source_botts_all.shape[1] if source_botts_all.ndim > 1 else 1
        full_botts_all = np.full((target_phi_vals.size, iterations_local), fill_value, dtype=float)
        full_botts_avg = np.full(target_phi_vals.size, fill_value, dtype=float)

        source_lookup = _build_disorder_rows(
            source_phi_vals,
            source_M_vals,
            source_botts_all,
            iterations_local,
            fill_value=fill_value,
        )

        for idx, (phi, M) in enumerate(zip(target_phi_vals, target_M_vals)):
            row = source_lookup.get(_disorder_key(phi, M))
            if row is None:
                continue

            full_botts_all[idx] = row
            full_botts_avg[idx] = _safe_row_mean(row)

        return full_botts_all, full_botts_avg

    full_botts_all, full_disorder = _reindex_disorder_data(
        clean_phi,
        clean_M,
        source_phi,
        source_M,
        source_disorder_all,
    )

    if (not fileOverwrite) and os.path.exists(disorder_filename):
        return disorder_filename

    with h5py.File(disorder_filename, 'w') as f:
        f.create_dataset(name='phi', data=np.asarray(clean_phi).ravel())
        f.create_dataset(name='M', data=np.asarray(clean_M).ravel())
        f.create_dataset(name='disorder_all', data=full_botts_all)
        f.create_dataset(name='disorder', data=full_disorder)

    return disorder_filename


def repair_disorder_files(directory='.', recursive=True, fileOverwrite=True):
    """Repair all disorder files in a directory tree."""
    pattern = os.path.join(directory, '**', '*_w*.h5') if recursive else os.path.join(directory, '*_w*.h5')
    repaired_files = []
    for disorder_file in glob.glob(pattern, recursive=recursive):
        try:
            repaired_files.append(repair_disorder_file(disorder_file, fileOverwrite=fileOverwrite))
        except Exception as e:
            print(f"Error repairing {disorder_file}: {e}")
    return repaired_files


# endregion
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
    extent = [X_range[0], X_range[1], Y_range[0], Y_range[1]]
    extent = [0., np.pi, 0., 5.5]

    try:
        if np.ndim(Z_values) != 2 or Z_values.shape != (len(np.unique(Y_values)), len(np.unique(X_values))):
            l = np.sqrt(Z_values.size)
            if np.isclose(int(l) - l, 0.):
                Z_values = Z_values.reshape(int(l), int(l)).T
            else: 
                Z_values = Z_values.reshape(len(np.unique(Y_values)), len(np.unique(X_values))).T
    except:
        pass

    im = ax.imshow(Z_values, extent=extent, 
                   origin='lower', aspect='auto', cmap=cmap, interpolation='none', 
                   rasterized=True, norm=norm)
    #im = ax.scatter(X_values, Y_values, c=Z_values, cmap=cmap, rasterized=True, norm=norm)
    
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


def extract_data_from_h5_file(filename:str):
    try:
        with h5py.File(filename, 'r') as f:
            data = {k: v[:] for k, v in zip(f.keys(), f.values())}
        return data
    
    except Exception as e:
        print(f"Error extracting data from file: {e}")
        return {}


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


def make_large_figure(generation:int, dimensions:tuple, methods:list, disorder_strengths=None, 
                      directory=".", cmap="cividis", 
                      plotUndisordered=True, plotSineBoundary=True, 
                      row_labels=None, column_labels=None, title:str="", image_filename=None, plotFull=False,
                      data_fname = None):
    
    if type(methods) is str:
        methods = [methods]
    if any([m in methods for m in ['hexagon', 'site_elim', 'renorm1', 'renorm2']]) == False:
        raise ValueError("Invalid method. Options are ['hexagon', 'site_elim', 'renorm1', 'renorm2']")
    
    and_contain_list = [f'g{generation}', f'({dimensions[0]}_by_{dimensions[1]})']
    or_contain_list = methods.copy()
    if 'renorm1' in methods or 'renorm2' in methods:
        or_contain_list.append('renorm')

    blacklisted_subdirs = {'misc', 'new misc'}
    files = [
        file for file in glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True)
        if not any(f"{os.sep}{subdir}{os.sep}" in file for subdir in blacklisted_subdirs)
    ]
    files = get_all_files_matching_criteria(files, contains_all=and_contain_list, contains_any=or_contain_list, does_not_contain=['_temp_chunk'])

    if data_fname != None:
        files = [data_fname]
    
    if disorder_strengths is None:
        disorder_strengths = get_disorder_strength_from_files(files)

    n_cols = len(disorder_strengths) + 1 if plotUndisordered else len(disorder_strengths)
    print(methods)
    fig, axs = plt.subplots(len(methods), n_cols, figsize=(n_cols * 4, len(methods) * 4), sharex=True, sharey=True)
    if len(methods) == 1:
        axs = axs.reshape(1, len(axs))
    elif n_cols == 1:
        axs = axs.reshape(len(methods), 1)

    clean_files = [file for file in files if 'w' not in file]
    disorder_files = [file for file in files if 'w' in file]

    clean_data = [extract_data_from_h5_file(file) for file in clean_files]
    disorder_data = [extract_data_from_h5_file(file) for file in disorder_files]

    print(disorder_data[0].keys())

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
        clean_files_array[i] = directory+f"/Generation {generation}/"+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]}).h5"
        for j, disorder_strength in enumerate(disorder_strengths):
            files_array[i, j] = directory+f"/Generation {generation}/"+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]})_w{disorder_strength}.h5"

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
                            if j == 3 and i == 1:
                                pass
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
        #ax.set_aspect('equal')

        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels([])
        ax.set_ylim([0, 3 * np.sqrt(3)])
        ax.set_yticks([0, 3 * np.sqrt(3) / 2, 3 * np.sqrt(3)])
        ax.tick_params(width=3., length = 5.0)
        for spine in ax.spines.values():
            spine.set_linewidth(3.)

    for ax in axs[-1, :]:
        ax.set_xticklabels(["0", "$\\pi/2$", "$\\pi$"])

    for ax in axs[:, 0]:
        ax.set_yticklabels(["0", "$3 \\sqrt{3} / 2$", "$3 \\sqrt{3}$"])

    cbar = add_colorbar_to_figure(fig, axs, norm, cmap, "Bott Index")
    cbar.set_ticks([-1.0, -0.5, 0.])
    cbar.ax.tick_params(width=3., length=5.0, labelsize=20)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(3.)
    
    if image_filename is not None:
        plt.savefig(image_filename, bbox_inches='tight', transparent=False)


# endregion
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
 

def compute_many_phase_diagrams(generation, disorder_strengths, methods, dimensions=(50,50), iterations=100, n_jobs=6, directory=".", doHalf:bool = True, clean_file_override:str = '', num_chunks:int = 10):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for disorder_strength in disorder_strengths:
        for method in methods:
            if clean_file_override == '':
                clean_file = compute_phase(method, generation, n_jobs=n_jobs, dimensions=dimensions, directory=directory, M_range=(0., 5.5), phi_range=(0., np.pi))
            else:
                clean_file = clean_file_override
            disorder_file = compute_disorder(clean_file, method, generation, disorder_strength, iterations=iterations, n_jobs=n_jobs, show_progress=True, doHalf=doHalf, num_chunks=num_chunks)


#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
def main(generation, iterations, compute_methods, compute_these_disorder_strengths, do_compute=True, do_plot=True): 
    plot_methods = ['hexagon', 'renorm1', 'renorm2', 'site_elim']
    titles = ["Pristine", "Renormalization 1", "Renormalization 2", "Site Elimination"]
    res = (25, 25)
    if do_compute:
        compute_many_phase_diagrams(generation, compute_these_disorder_strengths, compute_methods, res, 
                                    iterations=iterations, n_jobs=-2, directory="./Hexaflake/Data/New/", doHalf=True)
    if do_plot:
        make_large_figure(generation, res, plot_methods, 
                        disorder_strengths = compute_these_disorder_strengths,
                        directory="./Hexaflake/Data/",
                        cmap="jet", 
                        plotUndisordered=True, plotSineBoundary=False, plotFull=False,
                        row_labels=titles,
                        title="", 
                        image_filename=f"./Hexaflake/Figures/generation{generation}_disorder.svg",
                        )


if __name__ == "__main__":    
    #main(3, 25, [], [1., 5., 7.5, 10., 12.5], False)
    #main(3, 50, ["renorm1"], [1.0])
    #main(3, 25, ["hexagon"], [7.5, 10.0, 12.5])
#
    #def plot_dirty_file_contents(fname):
    #    try:
    #        with h5py.File(fname, 'r') as f:
    #            p = f["phi"][()] # type: ignore
    #            m = f["M"][()] # type: ignore
    #            d = f["disorder"][()] # type: ignore
    #    except KeyError:
    #        with h5py.File(fname, 'r') as f:
    #            d = f["disorder"][()].flatten() # type: ignore
    #        p, m = None, None
    #    return p, m, d
#
    #def plot_clean_file_contents(fname):
    #    with h5py.File(fname, 'r') as f:
    #        phi = f["phi"][()] # type: ignore
    #        m = f["M"][()] # type: ignore
    #        bott = f["bott_index"][()] # type: ignore
    #    plt.scatter(phi, m, c=bott) # type: ignore
    #    plt.show()

    #for w in [7.5, 10.0, 12.5]:
    #    repair_disorder_file(f'./Hexaflake/Data/Generation 3/hexagon_g3_(25_by_25)_w{w}.h5')

    #with h5py.File('./Hexaflake/Data/Generation 2/renorm1_g2_(25_by_25)_w1.1.h5', 'r') as f:
    #    p = f['phi'][()]
    #    m = f['M'][()]
    #    d = f["disorder"][()]
    #    dall = f["disorder_all"][()]
    #print(d.shape)
    #plt.imshow(dall)
    #plt.show()

    for w in [7.5, 10.0, 12.5]:
        file1 = f"./Hexaflake/Data/Generation 3/hexagon_g3_(25_by_25)_w{w}.h5"
        file2 = file1.replace(".h5", "_i25.h5")

        with h5py.File(file1, 'r') as f:
            phi = f["phi"][()] # type: ignore
            M = f["M"][()] # type: ignore
            d1 = f["disorder"][()] # type: ignore
            d_all = f["disorder_all"][()] # type: ignore
        with h5py.File(file2, 'r') as f:
            compute_idxs = f["computed_idxs"][()] # type: ignore
            d2 = f["disorder"][()].flatten() # type: ignore

        with h5py.File(file1.replace(".h5", "_i50.h5"), "w") as f:
            f.create_dataset(name="phi", data=phi)
            f.create_dataset(name="M", data=M)
            f.create_dataset(name="disorder", data=(d1 + d2)/2)
            f.create_dataset(name="disorder_all", data=d_all)

        

        
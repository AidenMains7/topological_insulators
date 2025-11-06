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
from MaybeActualFinalHaldane2 import compute_bott_index, compute_geometric_data, compute_hamiltonian


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


def compute_disorder_array(strength, system_size, degrees_of_freedom=1):
	"""
	Generate a disorder array for the Hamiltonian.

	Parameters:
	strength (float): The strength of the disorder.
	system_size (int): The size of the system.
	degrees_of_freedom (int): Degrees of freedom

	Returns:
	np.ndarray: A diagonal matrix representing the disorder.
	"""
	disorder_array = np.random.uniform(-strength/2, strength/2, size=system_size)
	delta = np.sum(disorder_array)/system_size
	disorder_array -= delta
	disorder_array = np.repeat(disorder_array, degrees_of_freedom)
	return np.diag(disorder_array).astype(np.complex128)


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
			bott = compute_bott_from_hamiltonian(H, method, geometry_data)
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
		bott = compute_bott_from_hamiltonian(disorder_H, method, geometry_data)
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


def get_all_files_matching_criteria(files, contains_all:list=None, contains_any:list=None):
	if contains_all is not None:
		files = [file for file in files if all([c in file for c in contains_all])]
	if contains_any is not None:
		files = [file for file in files if any([c in file for c in contains_any])]
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
	plt.tight_layout(rect=[0, 0, 0.9, 1])
	axs_flattened = axs.flatten()
	pos1 = axs_flattened[0].get_position()
	pos2 = axs_flattened[-1].get_position()
	cbar_ax = fig.add_axes([0.9, pos2.y0, 0.02, pos1.y1 - pos2.y0])
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cbar = fig.colorbar(sm, cax=cbar_ax)
	if cbar_label is not None:
		cbar.set_label(cbar_label, fontsize=12)
	
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


#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------


def make_large_figure(generation:int, dimensions:tuple, methods:list, disorder_strengths=None, 
					  directory=".", cmap="Spectral", 
					  plotUndisordered=True, plotSineBoundary=True, 
					  row_labels=None, column_labels=None, title=None, image_filename=None):
	
	if type(methods) is str:
		methods = [methods]
	if any([m in methods for m in ['hexagon', 'site_elim', 'renorm']]) == False:
		raise ValueError("Invalid method. Options are ['hexagon', 'site_elim', 'renorm']")
	
	and_contain_list = [f'g{generation}', f'({dimensions[0]}_by_{dimensions[1]})']
	or_contain_list = methods

	files = glob.glob(os.path.join(directory, f'*.h5'))
	files = get_all_files_matching_criteria(files, contains_all=and_contain_list, contains_any=or_contain_list)
	
	if disorder_strengths is None:
		disorder_strengths = get_disorder_strength_from_files(files)
	if plotUndisordered:
		disorder_strengths = [np.nan] + disorder_strengths

	fig, axs = plt.subplots(len(methods), len(disorder_strengths), figsize=(15,10), sharex=True, sharey=True)

	clean_files = [file for file in files if 'w' not in file]
	disorder_files = [file for file in files if 'w' in file]

	clean_data = [extract_data_from_h5_file(file) for file in clean_files]
	disorder_data = [extract_data_from_h5_file(file) for file in disorder_files]

	clean_bott_data = [data['bott_index'].T for data in clean_data]
	disorder_bott_data = [data['disorder'] for data in disorder_data]	

	
	X_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
	X_tick_labels = [pi_tick_labels(value) for value in X_ticks]
	Y_ticks = [-3*np.sqrt(3), -2*np.sqrt(3), -np.sqrt(3), 0, np.sqrt(3), 2*np.sqrt(3), 3*np.sqrt(3)]
	Y_tick_labels = [f"{i:.2f}" for i in Y_ticks]
	tick_dict = {'X_ticks': X_ticks, 'X_tick_labels': X_tick_labels, 'Y_ticks': Y_ticks, 'Y_tick_labels': Y_tick_labels}

	global_min, global_max = global_bounds(clean_bott_data+disorder_bott_data)
	norm = plt.Normalize(vmin=global_min, vmax=global_max)
	
	files_array = np.empty((len(methods), len(disorder_strengths)), dtype=object)
	for i, method in enumerate(methods):
		for j, disorder_strength in enumerate(disorder_strengths):
			if np.isnan(disorder_strength):
				files_array[i, j] = directory+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]}).h5"
			else:
				files_array[i, j] = directory+f"{method}_g{generation}_({dimensions[0]}_by_{dimensions[1]})_w{disorder_strength}.h5"

	for i, method in enumerate(methods):
		file = files_array[i, 0]
		try:
			loop_clean_data = extract_data_from_h5_file(file)
			phi_values, M_values, bott_values = loop_clean_data['phi'], loop_clean_data['M'], loop_clean_data['bott_index'].T
		except Exception as e:
			print(f"Exception: {e}")
		for j, disorder_strength in enumerate(disorder_strengths):
			file = files_array[i, j]
			try:
				if np.isnan(disorder_strength):
					# plot clean, equivalent to (if plotUndisordered and j == 0)
					fig, axs[i, j] = plot_phase_diagram(fig, axs[i, j], phi_values, M_values, bott_values, cmap=cmap, norm=norm, **tick_dict, plotColorbar=False)
				else:
					loop_disorder_data = extract_data_from_h5_file(file)
					if loop_disorder_data is not None:
						fig, axs[i, j] = plot_phase_diagram(fig, axs[i, j], phi_values, M_values, loop_disorder_data['disorder'].T, cmap=cmap, norm=norm, **tick_dict, plotColorbar=False)
			except Exception as e:
				print(f"Error plotting in axes[{i}, {j}]: {e}")

	if row_labels is None:
		row_labels = methods
	for i, row_label in enumerate(row_labels):
		axs[i, 0].set_ylabel('M', fontsize=12, rotation=0)
		axs[i, 0].annotate(row_label, xy=(-0.3, 0.5), xytext=(-axs[i, 0].yaxis.labelpad - 5, 0),
				   xycoords=axs[i, 0].yaxis.label, textcoords='offset points',
				   size='large', ha='center', va='center', rotation=90)
		
	if column_labels is None:
		column_labels = [f"W = {strength}" for strength in disorder_strengths]
		if plotUndisordered:
			column_labels[0] = "Undisordered" 
	for j, column_label in enumerate(column_labels):
		axs[-1, j].set_xlabel(r'$\phi$', fontsize=12)
		axs[0, j].set_title(column_label, fontsize=12)
		
	fig.suptitle(title, fontsize=16)

	if plotSineBoundary:
		for ax in axs.flatten():
			t = np.linspace(-np.pi, np.pi, 1000)
			ax.plot(t, np.sin(t)*np.sqrt(3)*3, c='k', ls=(0, (5, 1)), alpha=0.25)
			ax.plot(t, -np.sin(t)*np.sqrt(3)*3, c='k', ls=(0, (5, 1)), alpha=0.25)

	for ax in axs.flatten():
		ax.set_xlim([0., np.pi])
		ax.set_ylim([0., 3*np.sqrt(3)])


	add_colorbar_to_figure(fig, axs, norm, cmap, "Bott Index")
	if image_filename is not None:
		plt.savefig(image_filename)
	plt.show()


def compute_many_phase_diagrams(generation, disorder_strengthsm, methods, dimensions=(50,50), iterations=100, n_jobs=6, directory="."):
	if not os.path.exists(directory):
		os.makedirs(directory)

	for disorder_strength in disorder_strengths:
		for method in methods:
			clean_file = compute_phase(method, generation, n_jobs=n_jobs, dimensions=dimensions, directory=directory,
							  M_range=(0., 5.5), phi_range=(0., np.pi))
			disorder_file = compute_disorder(clean_file, method, generation, disorder_strength, iterations=iterations, n_jobs=n_jobs, directory=directory, intermittent_saving=True, show_progress=True)


#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------


if __name__ == "__main__":
	disorder_strengths = [5.0, 4.0, 3.0, 2.0, 1.0]
	methods = ['site_elim']
	titles = ['Pristine', 'Renormalization', 'Site Elimination']
	compute_many_phase_diagrams(3, disorder_strengths, methods, (25, 25), iterations=100, n_jobs=4, directory="./Hexaflake/Data/")
	make_large_figure(3, (25, 25), methods, 
				   disorder_strengths=disorder_strengths,
				   directory="./Hexaflake/Data/", 
				   cmap="Spectral", plotUndisordered=True, plotSineBoundary=True,
				   row_labels=titles,
				   title="Bott Index Phase Diagram Varying With Disorder", image_filename="./Hexaflake/Figures/PhaseDiagram.png")
	
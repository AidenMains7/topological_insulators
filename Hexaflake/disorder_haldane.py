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


def compute_hexagon(n):
	"""
	Construct a boolean 2D array that represents a large hexagonal honeycomb lattice composed of.
	The size of the array is determined by the parameter n, which influences the dimensions.

	Args:
		n (int): An integer controlling the size of the resulting hexagon.

	Returns:
		np.ndarray: A 2D boolean array where True indicates the presence of
			a lattice site in the hexagon, and False indicates empty space.
	"""

	# Create end_piece array which forms the left and right edges of the lattice.
	end_piece = np.full((3 ** (n + 1), 3 * ((3 ** n - 1) // 2)), False, dtype=bool)
	for i in range((3 ** n - 1) // 2):
		start = (3 ** (n + 1) - 1) // 2 - 3 * i
		indices_1 = start + 2 * np.arange(3 * i + 1)
		indices_2 = start - 1 + 2 * np.arange(3 * i + 2)
		end_piece[indices_1, 3 * i] = True
		end_piece[indices_2, 3 * i + 1] = True

	# Create a repeated column of unit cell width to fill the middle section of the lattice.
	column = np.full((3 ** (n + 1), 6), False, dtype=bool)
	column[1::2, (0, 4)] = 1
	column[::2, (1, 3)] = 1

	# Repeat column to fill horizontal space, minus a strip at the end.
	middle = np.tile(column, (1, (3 ** n + 1) // 2))[:, :-1]

	# Combine the end pieces and the middle portion horizontally.
	hexagon_array = np.hstack((end_piece, middle, np.fliplr(end_piece)))

	return hexagon_array


def compute_hexaflake(n):
	"""
	Construct a boolean 2D array that represents a hexaflake pattern of order n.
	The hexaflake is created by recursively appending smaller hexagons around
	an initial hexagon shape, scaled by factors of 3.

	Args:
		n (int): The iteration order of the hexaflake. Higher values produce
			more fractal detail.

	Returns:
		np.ndarray: A 2D boolean array marking the presence of sites in the
			hexaflake (True) and empty space (False).
	"""

	# Directions in which to replicate the smaller hexagons.
	directions = np.array([[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]])
	# Scale factors determine how far to offset for each recursion level.
	scale_factors = 3 ** np.arange(1, n + 1)

	# Start with the 6 directions and build up by adding scaled copies.
	discrete_coordinates = directions.copy()
	for scale in scale_factors:
		offsets = scale * directions
		new_coordinates = []
		for offset in [[0, 0]] + offsets.tolist():
			new_coordinates.extend(discrete_coordinates + offset)
		discrete_coordinates = np.array(new_coordinates)

	x_discrete, y_discrete = discrete_coordinates.T

	# Shift coordinates so there are no negative indices.
	x_discrete += 3 ** (n + 1) - 1
	y_discrete += (3 ** (n + 1) - 1) // 2

	# Create the array for the hexaflake pattern.
	hexaflake_array = np.full(
		(3 ** (n + 1), 2 * 3 ** (n + 1) - 1),
		False,
		dtype=bool
	)
	hexaflake_array[y_discrete, x_discrete] = True

	return hexaflake_array


def compute_dx_and_dy_discrete(x_discrete, y_discrete, PBC):
	"""
	Compute the discrete differences in x and y coordinates between all pairs of points.
	If periodic boundary conditions (PBC) are enabled, the code accounts for wrapping
	by shifting the lattice in possible directions and selecting the minimal distance.

	Args:
		x_discrete (np.ndarray): 1D array of x-coordinates of points.
		y_discrete (np.ndarray): 1D array of y-coordinates of points.
		PBC (bool): Flag indicating whether to apply periodic boundary conditions.

	Returns:
		tuple:
			- (np.ndarray) delta_x_discrete: 2D array of differences in x-coordinates.
			- (np.ndarray) delta_y_discrete: 2D array of differences in y-coordinates.
	"""

	if not PBC:
		# Simple difference if no periodic boundary conditions.
		delta_x_discrete = x_discrete[np.newaxis, :] - x_discrete[:, np.newaxis]
		delta_y_discrete = y_discrete[np.newaxis, :] - y_discrete[:, np.newaxis]
	else:
		# Attempt all relevant shifts in the 2D lattice so that the distance is minimized.
		a = round(np.sqrt(2 * x_discrete.size - 3))
		b = (a + 3) // 2
		c = (a - 3) // 2
		d = 2 * a - b
		e = 2 * a - c

		# Potential shifts for wrapping.
		shifts = np.array([
			[0, 0],
			[-3, a],
			[3, -a],
			[d, b],
			[-d, -b],
			[-e, c],
			[e, -c]
		])

		# Shift coordinates according to each shift vector.
		shifted_x = (
				shifts[:, 0][:, np.newaxis, np.newaxis] +
				x_discrete[np.newaxis, :, np.newaxis]
		)
		shifted_y = (
				shifts[:, 1][:, np.newaxis, np.newaxis] +
				y_discrete[np.newaxis, :, np.newaxis]
		)

		# Calculate all candidate distances.
		distances = ((1 / 2) * (x_discrete[np.newaxis, np.newaxis, :] - shifted_x)) ** 2 + \
					((np.sqrt(3) / 2) * (y_discrete[np.newaxis, np.newaxis, :] - shifted_y)) ** 2

		# Identify the shift index yielding the minimal distance for each pair (i,j).
		idx_array = np.argmin(distances, axis=0)

		i_indices, j_indices = np.indices(idx_array.shape)

		# Retrieve the corresponding differences that gave the minimum distance.
		delta_x_discrete = x_discrete[np.newaxis, np.newaxis, :] - shifted_x
		delta_y_discrete = y_discrete[np.newaxis, np.newaxis, :] - shifted_y
		delta_x_discrete = delta_x_discrete[idx_array, i_indices, j_indices]
		delta_y_discrete = delta_y_discrete[idx_array, i_indices, j_indices]

	return delta_x_discrete.astype(np.int64), delta_y_discrete.astype(np.int64)


def compute_hopping_arrays(delta_x_discrete, delta_y_discrete):

	NN = ((np.abs(delta_x_discrete) ==  2) & (delta_y_discrete == 0)) | \
		 ((np.abs(delta_x_discrete) == 1) & (np.abs(delta_y_discrete) == 1))

	NNN = ((delta_x_discrete ==  0) & (np.abs(delta_y_discrete) == 2)) | \
		 ((np.abs(delta_x_discrete) == 3) & (np.abs(delta_y_discrete) == 1))

	CCW_directions = np.array([[1, -1, 1, 0], [1, 1, -1, 1], [-1, 0, -1, -1], [-1, 1, -1, 0], [-1, -1, 1, -1], [1, 0, 1, 1]], dtype=np.int8)

	i, j = np.where(NNN)

	k = np.argmax(NN[i] & NN[j], axis=1)

	x_i_to_k, y_i_to_k = delta_x_discrete[i, k], delta_y_discrete[i, k]
	x_k_to_j, y_k_to_j = delta_x_discrete[k, j], delta_y_discrete[k, j]

	NNN_directions = np.sign(np.array([x_i_to_k, y_i_to_k, x_k_to_j, y_k_to_j]).T).astype(np.int8)

	CCW = np.any(np.all(CCW_directions[None, :, :] == NNN_directions[:, None, :], axis=2), axis=1)

	NNN_CCW = np.full_like(NNN, False)
	NNN_CCW[i, j] = CCW

	return NN, NNN_CCW


def compute_geometric_data(n, PBC, return_dx_dy=False, print_info=False):

	hexagon_array = compute_hexagon(n)
	y_discrete, x_discrete = np.where(hexagon_array)

	sublattice_array = np.zeros_like(hexagon_array)
	sublattice_array[:, ::3] = hexagon_array[:, ::3] #one of the sublattices
	sublattice = ~sublattice_array[y_discrete, x_discrete] #opposite of former

	stagger_sublattices = np.empty(sublattice.size, dtype=np.int64)
	stagger_sublattices[::2] = np.sort(np.where(sublattice)[0])
	stagger_sublattices[1::2] = np.sort(np.where(~sublattice)[0])
	x_discrete, y_discrete = x_discrete[stagger_sublattices], y_discrete[stagger_sublattices]


	hexaflake_array = compute_hexaflake(n)
	hexaflake = hexaflake_array[y_discrete, x_discrete]

	if print_info:
		print(f"Generation: {n}")
		print(f"Honeycomb # filled sites: {y_discrete.size}")
		print(f"Hexaflake # filled sites: {np.where(hexaflake)[0].size}")


	delta_x_discrete, delta_y_discrete = compute_dx_and_dy_discrete(x_discrete, y_discrete, PBC)

	NN, NNN_CCW = compute_hopping_arrays(delta_x_discrete, delta_y_discrete)

	x = (1 / 2) * (x_discrete - 3 ** (n + 1) + 1)
	y = (np.sqrt(3) / 4) * (2 * y_discrete - 3 ** (n + 1) + 1)

	geometric_data = {
		'x': x,
		'y': y,
		'hexaflake': hexaflake,
		'NN': NN,
		'NNN_CCW': NNN_CCW,
		'x_discrete': x_discrete,
		'y_discrete': y_discrete
	}

	if return_dx_dy:
		geometric_data['delta_x_discrete'] = delta_x_discrete
		geometric_data['delta_y_discrete'] = delta_y_discrete

	return geometric_data


def compute_hamiltonian(method, M, phi, t1, t2, geometric_data):

	valid_methods = ['hexagon', 'site_elim', 'renorm']
	if method not in valid_methods:
		raise ValueError(f"Invalid method '{method}'. Options are {valid_methods}.")

	NN = geometric_data['NN']
	NNN_CCW = geometric_data['NNN_CCW']
	hexaflake = geometric_data['hexaflake']

	H = np.zeros(NN.shape, dtype=np.complex128)
	np.fill_diagonal(H, M*((-1)**(np.arange(H.shape[0]))))
	H[NN] = -t1
	H[NNN_CCW] = -t2 * np.sin(phi)*1j
	H[NNN_CCW.T] = t2 * np.sin(phi)*1j

	if method == 'renorm':
		H_aa = H[np.ix_(hexaflake, hexaflake)]
		H_bb = H[np.ix_(~hexaflake, ~hexaflake)]
		H_ab = H[np.ix_(hexaflake, ~hexaflake)]
		H_ba = H[np.ix_(~hexaflake, hexaflake)]

		H = H_aa - H_ab @ sp.linalg.solve(H_bb,H_ba,assume_a='her',check_finite=False,overwrite_a=True,overwrite_b=True)

	elif method == 'site_elim':
		H = H[np.ix_(hexaflake, hexaflake)]

	return H


def triangular_basis(x, y):

	a1 = (np.sqrt(3) / 2) * x - 0.5 * y
	a2 = (np.sqrt(3) / 2) * x + 0.5 * y

	return a1, a2


def compute_bott_index(eigen_data):

	eigenvalues, eigenvectors, x, y = [eigen_data[key] for key in 'eigenvalues, eigenvectors, x, y'.split(', ')]
	lower_band = np.argsort(eigenvalues)[:eigenvalues.size // 2]
	V = eigenvectors[:, lower_band]

	N = round((np.sqrt(2 * x.size - 3) - 3) / 2 + 2)
	L = np.sqrt(3) * N

	a1, a2 = triangular_basis(x, y)

	U1 = np.exp(1j * 2 * np.pi * a1 / L)[:, np.newaxis]
	U2 = np.exp(1j * 2 * np.pi * a2 / L)[:, np.newaxis]

	U1_proj = V.conj().T @ (V * U1)
	U2_proj = V.conj().T @ (V * U2)

	A = U2_proj @ U1_proj @ U2_proj.conj().T @ U1_proj.conj().T

	eigenvaluesA = sp.linalg.eigvals(A, overwrite_a=True)
	trace_logA = np.sum(np.log(eigenvaluesA))

	bott = round(np.imag(trace_logA) / (2 * np.pi))

	return bott


def compute_bott_from_hamiltonian(H, method, geometry_data):
	x, y = geometry_data['x'], geometry_data['y']
	eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)
	if method in ['site_elim', 'renorm']:
		hexaflake = geometry_data['hexaflake']
		x, y = x[hexaflake], y[hexaflake]
	return compute_bott_index({'x':x, 'y':y, 'eigenvalues':eigenvalues, 'eigenvectors':eigenvectors})

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
		numerator = "$\pi$"
	else:
		numerator = f"{abs(fractional_value.numerator)}$\pi$"
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
	print(X_tick_labels)
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

	add_colorbar_to_figure(fig, axs, norm, cmap, "Bott Index")
	if image_filename is not None:
		plt.savefig(image_filename)
	plt.show()


def compute_many_phase_diagrams(generation=2, dimensions=(50,50), iterations=100, n_jobs=-2, directory="."):
	if not os.path.exists(directory):
		os.makedirs(directory)

	for disorder_strength in [11.0]:
		for method in ['renorm', 'site_elim', 'hexagon']:
			clean_file = compute_phase(method, generation, n_jobs=n_jobs, dimensions=dimensions, directory=directory)
			disorder_file = compute_disorder(clean_file, method, generation, disorder_strength, iterations=iterations, n_jobs=n_jobs, directory=directory, intermittent_saving=True, show_progress=True)


#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------


if __name__ == "__main__":
	#compute_many_phase_diagrams()
	make_large_figure(2, (50,50), ['hexagon', 'renorm', 'site_elim'], 
				   disorder_strengths=[1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
				   directory="Haldane_Disorder_Data/Res2500_Avg100/", 
				   cmap="Spectral", plotUndisordered=True, plotSineBoundary=True,
				   row_labels=['Hexagon', 'Renormalization', 'Site Elimination'],
				   title="Bott Index Phase Diagram Varying With Disorder", image_filename="Haldane_Disorder_Data/Res2500_Avg100/PhaseDiagram.png")
	
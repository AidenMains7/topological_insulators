import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def sort_hexaflake_by_subflake(coords, generation):
    def order_by_subflake(coords, generation): 
            centers = (3**generation)*np.array([np.array(([2*np.cos(np.pi/3*a)], [2/np.sqrt(3)*np.sin(np.pi/3*a)])) for a in range(6)])[..., 0].T
            if generation >= 1:
                centers = np.append(centers, [[0],[0]], axis=1)[:, [1,2,0,6,3,5,4]]
            else:
                 centers = centers[:, [1, 2, 0, 3, 5, 4]]
            centers[0] += np.mean(coords[0])
            centers[1] += np.mean(coords[1])

            dx, dy = np.power(coords[..., np.newaxis] - centers[:, np.newaxis, :], 2)
            difference = dx + dy
            which_subflake = np.argmin(difference, axis=-1) # Assign each point to a subflake

            return np.argsort(which_subflake)

    def sort_by_x_then_y(coords):
        idxs = np.lexsort((-coords[0], -coords[1]), axis=0)
        return idxs
    
    reordered = order_by_subflake(coords, generation)
    subflake_idxs = [reordered[np.arange(len(reordered)//7*i,len(reordered)//7*(i+1))] for i in range(7)]
    
    sorted_idxs = np.concatenate([sfidx[sort_by_x_then_y(coords[:, sfidx])] for sfidx in subflake_idxs])
    return sorted_idxs


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


def compute_geometric_data(n, PBC, return_dx_dy=False):

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

	doSortBySubflake = False
	if doSortBySubflake:
		x_discrete, y_discrete = geometric_data['x_discrete'], geometric_data['y_discrete']
		hx, hy = x_discrete[hexaflake], y_discrete[hexaflake]
		n = int(np.log(len(hx)/6)/np.log(7))
		sorted_idxs = sort_hexaflake_by_subflake(np.vstack((hx[None, :], hy[None, :])), n)


	if method == 'renorm':
		H_aa = H[np.ix_(hexaflake, hexaflake)]
		H_bb = H[np.ix_(~hexaflake, ~hexaflake)]
		H_ab = H[np.ix_(hexaflake, ~hexaflake)]
		H_ba = H[np.ix_(~hexaflake, hexaflake)]

		if doSortBySubflake:
			H_aa = H_aa[np.ix_(sorted_idxs, sorted_idxs)]
			H_ab = H_ab[np.ix_(sorted_idxs, np.arange(H_ab.shape[1]))]
			H_ba = H_ba[np.ix_(np.arange(H_ba.shape[0]), sorted_idxs)]

		H = H_aa - H_ab @ sp.linalg.solve(H_bb,H_ba,assume_a='her',check_finite=False,overwrite_a=True,overwrite_b=True)

	elif method == 'site_elim':
		H = H[np.ix_(hexaflake, hexaflake)]
		if doSortBySubflake:
			H = H[np.ix_(sorted_idxs, sorted_idxs)]


	return H


def compute_sparse_hamiltonian(method, M, phi, t1, t2, geometric_data):
	NN = geometric_data['NN']
	NNN_CCW = geometric_data['NNN_CCW']

	if method == 'site_elim':
		hexaflake = geometric_data['hexaflake']
		NN = NN[np.ix_(hexaflake, hexaflake)]
		NNN_CCW = NNN_CCW[np.ix_(hexaflake, hexaflake)]

	i_diagonal = np.arange(NN.shape[0])
	data_diagonal = np.empty(NN.shape[0], dtype=np.complex128)
	data_diagonal[::2] = M
	data_diagonal[1::2] = -M

	i_nn, j_nn = np.where(NN)
	data_nn = np.full(i_nn.size, -t1, dtype=np.complex128)

	i_nnn_ccw, j_nnn_ccw = np.where(NNN_CCW)
	data_nnn_ccw = np.full(i_nnn_ccw.size, -t2*np.sin(phi)*1j)

	i_all = np.concatenate([i_diagonal, i_nn, i_nnn_ccw, j_nnn_ccw])
	j_all = np.concatenate([i_diagonal, j_nn, j_nnn_ccw, i_nnn_ccw])
	data_all = np.concatenate([data_diagonal, data_nn, data_nnn_ccw, -data_nnn_ccw])

	H = coo_matrix((data_all, (i_all, j_all)), shape=NN.shape).tocsr()

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


def compute_local_chern_markers(eigen_data):

	eigenvalues, eigenvectors, x, y = [eigen_data[key] for key in 'eigenvalues, eigenvectors, x, y'.split(', ')]
	lower_band = np.argsort(eigenvalues)[:eigenvalues.size // 2]
	V = eigenvectors[:, lower_band]

	a1, a2 = triangular_basis(x, y)

	P = V.dot(V.conj().T)

	a1P = np.einsum('i,ij->ij', a1, P)
	a2P = np.einsum('i,ij->ij', a2, P)

	C = 4 * np.pi * np.imag(np.diag(P.dot(a1P).dot(a2P)))

	return C


def compute_eigen_data(method, M, phi, t1, t2, geometric_data):

	x, y = geometric_data['x'], geometric_data['y']

	H = compute_hamiltonian(method, M, phi, t1, t2, geometric_data)

	dx = (1/2)*(H[::2, :][:, 1::2] + H[::2, :][:, 1::2].conj().T)
	dy = (1j/2)*(H[::2, :][:, 1::2] - H[::2, :][:, 1::2].conj().T)
	dz = H[::2, :][:, ::2]

	eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)

	if method in ['site_elim', 'renorm']:
		hexaflake = geometric_data['hexaflake']
		x, y = x[hexaflake], y[hexaflake]

	eigen_data = {
		'x': x,
		'y': y,
		'eigenvalues': eigenvalues,
		'eigenvectors': eigenvectors
	}

	return eigen_data


def compute_band_data(
		method='hexagon', sparse=False, PBC=True, resolution=100,
		M_resolution=None, phi_resolution=None,
		M_range=(-5.5, 5.5), phi_range=(-np.pi, np.pi),
		n=3, t1=1., t2=1., k=12, n_jobs=-8):

	valid_methods = ['hexagon', 'site_elim', 'renorm']
	if method not in valid_methods:
		raise ValueError(f"Invalid method '{method}'. Options are {valid_methods}.")

	if method == 'renorm' and sparse:
		raise ValueError('Sparse method is not applicable for method of renormalization.')

	M_resolution = resolution if M_resolution is None else M_resolution
	phi_resolution = resolution if phi_resolution is None else phi_resolution

	M_resolution += 1 - M_resolution % 2
	phi_resolution += 1 - phi_resolution % 2

	geometric_data = compute_geometric_data(n, PBC)

	def worker(param_idxs):
		M_idx, phi_idx = param_idxs
		M_val = (M_range[1] - M_range[0]) * (M_idx / (M_resolution - 1)) + M_range[0]
		phi_val = (phi_range[1] - phi_range[0]) * (phi_idx / (phi_resolution - 1)) + phi_range[0]
		try:
			if sparse:
				H = compute_sparse_hamiltonian(method, M_val, phi_val, t1, t2, geometric_data)
				eigenvalues_small = eigsh(H, k=k, which='LM', sigma=0. + 0.j, return_eigenvectors=False)
				eigenvalues_large = eigsh(H, k=k, which='LM', return_eigenvectors=False)
				eigenvalues = np.concatenate((eigenvalues_small, eigenvalues_large))

			else:
				H = compute_hamiltonian(method, M_val, phi_val, t1, t2, geometric_data)
				eigenvalues = sp.linalg.eigvalsh(H, overwrite_a=True)

			positive = eigenvalues > 0
			band_gap = eigenvalues[positive].min() - eigenvalues[~positive].max()
			band_width = eigenvalues.max() - eigenvalues.min()
			return [M_idx, phi_idx, band_gap, band_width]

		except Exception as e:
			print(f'An error occurred for (M, phi)=({M_val:.3f}, {phi_val:.3f}): {e}')
			return [M_idx, phi_idx, np.nan, np.nan]

	idxs_sets = tuple(product(range(M_resolution), range(phi_resolution)))

	with tqdm_joblib(tqdm(total=len(idxs_sets), desc='Computing band gap diagram...')):
		M_data, phi_data, band_gap_data, band_width_data = np.array(
			Parallel(n_jobs=n_jobs)(delayed(worker)(idx_set) for idx_set in idxs_sets)
		).T

	band_gap_array = np.empty((M_resolution, phi_resolution))
	band_gap_array[M_data.astype(np.int32), phi_data.astype(np.int32)] = band_gap_data

	band_width_array = np.empty((M_resolution, phi_resolution))
	band_width_array[M_data.astype(np.int32), phi_data.astype(np.int32)] = band_width_data

	band_data = dict(
		band_gap_array=band_gap_array,
		band_width_array=band_width_array,
		M_range=np.asarray(M_range),
		phi_range=np.asarray(phi_range),
		t1=t1,
		t2=t2
	)

	return band_data


def compute_phase_diagram(
        method='hexagon', resolution=100,
        M_resolution=None, phi_resolution=None,
        M_range=(-5.5, 5.5), phi_range=(-np.pi, np.pi),
        n=3, t1=1., t2=1.,  n_jobs=-8):

	valid_methods = ['hexagon', 'site_elim', 'renorm']
	if method not in valid_methods:
		raise ValueError(f"Invalid method '{method}'. Options are {valid_methods}.")

	M_resolution = resolution if M_resolution is None else M_resolution
	phi_resolution = resolution if phi_resolution is None else phi_resolution

	M_resolution += 1 - M_resolution % 2
	phi_resolution += 1 - phi_resolution % 2

	geometric_data = compute_geometric_data(n, True)

	def worker(param_idxs):
		M_idx, phi_idx = param_idxs
		M_val = (M_range[1] - M_range[0]) * (M_idx / (M_resolution - 1)) + M_range[0]
		phi_val = (phi_range[1] - phi_range[0]) * (phi_idx / (phi_resolution - 1)) + phi_range[0]

		try:
			eigen_data = compute_eigen_data(method, M_val, phi_val, t1, t2, geometric_data)
			bott = compute_bott_index(eigen_data)
			return [M_idx, phi_idx, bott]

		except Exception as e:
			print(f'An error occurred for (M, phi)=({M_val:.3f}, {phi_val:.3f}): {e}')
			return [M_idx, phi_idx, np.nan]

	idxs_sets = tuple(product(range(M_resolution), range(phi_resolution)))

	with tqdm_joblib(tqdm(total=len(idxs_sets), desc='Computing phase diagram...')):
		M_data, phi_data, bott_index_data = np.array(
			Parallel(n_jobs=n_jobs)(delayed(worker)(idx_set) for idx_set in idxs_sets)
		).T

	bott_index_array = np.empty((M_resolution, phi_resolution))
	bott_index_array[M_data.astype(np.int32), phi_data.astype(np.int32)] = bott_index_data

	phase_data = dict(
		bott_index_array=bott_index_array,
		M_range=np.asarray(M_range),
		phi_range=np.asarray(phi_range),
		t1=t1,
		t2=t2
	)

	return phase_data


def plot_spectrum_and_LDOS(eigen_data, num_states=2, cmap='inferno'):

	num_ticks_E, decimals_E = 5, 2
	num_ticks_n, decimals_n = 5, 0
	num_ticks_x, decimals_x = 5, 2
	num_ticks_y, decimals_y = 5, 2
	num_ticks_LDOS, decimals_LDOS = 5, 2

	num_states += 1 - num_states % 2

	x, y, eigenvalues, eigenvectors = [eigen_data[key] for key in 'x, y, eigenvalues, eigenvectors'.split(', ')]

	all_idxs = np.arange(eigenvalues.size)
	positive = eigenvalues > 0

	positive_idxs = all_idxs[positive][np.argsort(eigenvalues[positive])][:num_states // 2]
	negative_idxs = all_idxs[~positive][np.argsort(eigenvalues[~positive])[::-1]][:num_states // 2]

	LDOS_idxs = np.concatenate((negative_idxs, positive_idxs))
	other_idxs = np.delete(all_idxs, LDOS_idxs)

	LDOS = np.sum(np.abs(eigenvectors[:, LDOS_idxs]) ** 2, axis=1)

	fig, axs = plt.subplots(1, 2, figsize=(12, 6))

	spectrum_aspect = (
							  (eigenvalues.size - 1) / (eigenvalues.max() - eigenvalues.min())
					  ) * (
							  (y.max() - y.min()) / (x.max() - x.min())
					  )

	axs[0].scatter(other_idxs, eigenvalues[other_idxs], c='black', s=20)
	axs[0].scatter(LDOS_idxs, eigenvalues[LDOS_idxs], c='red', s=30)
	axs[0].set_xlabel('n')
	axs[0].set_ylabel('E', rotation=0)
	axs[0].set_title('Spectrum')
	axs[0].set_aspect(spectrum_aspect)

	n_ticks = np.linspace(0, eigenvalues.size - 1, num_ticks_n)
	axs[0].set_xticks(n_ticks)
	axs[0].set_xticklabels([f'{tick:.{decimals_n}f}' for tick in n_ticks])

	E_ticks = np.linspace(eigenvalues.min(), eigenvalues.max(), num_ticks_E)
	axs[0].set_yticks(E_ticks)
	axs[0].set_yticklabels([f'{tick:.{decimals_E}f}' for tick in E_ticks])

	LDOS_scatter = axs[1].scatter(x, y, c=LDOS, cmap=cmap, s=7.5)
	axs[1].set_aspect('equal')
	axs[1].set_xlabel('x')
	axs[1].set_ylabel('y', rotation=0)
	axs[1].set_title('LDOS')

	x_ticks = np.linspace(x.min(), x.max(), num_ticks_x)
	axs[1].set_xticks(x_ticks)
	axs[1].set_xticklabels([f'{tick:.{decimals_x}f}' for tick in x_ticks])

	y_ticks = np.linspace(y.min(), y.max(), num_ticks_y)
	axs[1].set_yticks(y_ticks)
	axs[1].set_yticklabels([f'{tick:.{decimals_y}f}' for tick in y_ticks])

	bbox = axs[1].get_position()
	cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
	cbar = fig.colorbar(LDOS_scatter, cax=cbar_ax)
	LDOS_ticks = np.linspace(LDOS.min(), LDOS.max(), num_ticks_LDOS)
	cbar.set_ticks(LDOS_ticks)
	cbar.set_ticklabels([f'{tick:.{decimals_LDOS}e}' for tick in LDOS_ticks])

	plt.show()


def plot_band_gap_and_width(band_data, cmap='inferno'):

	band_gap_array = band_data['band_gap_array']
	band_width_array = band_data['band_width_array']
	phi_range = band_data['phi_range']
	M_range = band_data['M_range']

	num_ticks_phi, decimals_phi = 5, 2
	num_ticks_M, decimals_M = 5, 2
	num_ticks_cb, decimals_cb = 5, 2

	fig, axs = plt.subplots(1, 2, figsize=(14, 6))

	im1 = axs[0].imshow(
		band_gap_array,
		extent=[phi_range[0], phi_range[1], M_range[0], M_range[1]],
		origin='lower',
		aspect='auto',
		cmap=cmap
	)
	axs[0].set_title('Band Gap')
	axs[0].set_xlabel('Phi')
	axs[0].set_ylabel('M', rotation=0)
	cbar1 = fig.colorbar(im1, ax=axs[0])

	phi_ticks = np.linspace(phi_range[0], phi_range[1], num_ticks_phi)
	axs[0].set_xticks(phi_ticks)
	axs[0].set_xticklabels([f'{tick:.{decimals_M}f}' for tick in phi_ticks])

	M_ticks = np.linspace(M_range[0], M_range[1], num_ticks_M)
	axs[0].set_yticks(M_ticks)
	axs[0].set_yticklabels([f'{tick:.{decimals_phi}f}' for tick in M_ticks])

	band_gap_min, band_gap_max = np.nanmin(band_gap_array), np.nanmax(band_gap_array)
	cbar1_ticks = np.linspace(band_gap_min, band_gap_max, num_ticks_cb)
	cbar1.set_ticks(cbar1_ticks)
	cbar1.set_ticklabels([f'{tick:.{decimals_cb}f}' for tick in cbar1_ticks])

	im2 = axs[1].imshow(
		band_width_array,
		extent=[phi_range[0], phi_range[1], M_range[0], M_range[1]],
		origin='lower',
		aspect='auto',
		cmap=cmap
	)
	axs[1].set_title('Band Width')
	axs[1].set_xlabel('Phi')
	axs[1].set_ylabel('M', rotation=0)
	cbar2 = fig.colorbar(im2, ax=axs[1])

	axs[1].set_xticks(phi_ticks)
	axs[1].set_xticklabels([f'{tick:.{decimals_M}f}' for tick in phi_ticks])
	axs[1].set_yticks(M_ticks)
	axs[1].set_yticklabels([f'{tick:.{decimals_phi}f}' for tick in M_ticks])

	band_width_min, band_width_max = np.nanmin(band_width_array), np.nanmax(band_width_array)
	cbar2_ticks = np.linspace(band_width_min, band_width_max, num_ticks_cb)
	cbar2.set_ticks(cbar2_ticks)
	cbar2.set_ticklabels([f'{tick:.{decimals_cb}f}' for tick in cbar2_ticks])

	plt.tight_layout()
	plt.show()


def plot_phase_diagram(phase_data, cmap='viridis', titleparams=None, outputfile='temp.png'):

	bott_index_array = phase_data['bott_index_array']
	phi_range = phase_data['phi_range']
	M_range = phase_data['M_range']

	num_ticks_phi, decimals_phi = 5, 2
	num_ticks_M, decimals_M = 5, 2
	num_ticks_cb, decimals_cb = 5, 2

	fig, ax = plt.subplots(figsize=(7, 6))
	im = ax.imshow(
		bott_index_array,
		extent=[phi_range[0], phi_range[1], M_range[0], M_range[1]],
		origin='lower',
		aspect='auto',
		cmap=cmap
	)

	temp = np.linspace(-np.pi, np.pi, 500)
	ax.plot(temp, np.sin(temp)*np.sqrt(3)*3, c='k', ls='--')
	ax.plot(temp, -np.sin(temp)*np.sqrt(3)*3, c='k', ls='--')

	ax.set_title(f'Phase Diagram (Bott Index){'\n'+titleparams}')
	ax.set_xlabel('Phi')
	ax.set_ylabel('M', rotation=0)
	cbar = fig.colorbar(im, ax=ax)

	phi_ticks = np.linspace(phi_range[0], phi_range[1], num_ticks_phi)
	ax.set_xticks(phi_ticks)
	ax.set_xticklabels([f'{tick:.{decimals_M}f}' for tick in phi_ticks])

	M_ticks = np.linspace(M_range[0], M_range[1], num_ticks_M)
	ax.set_yticks(M_ticks)
	ax.set_yticklabels([f'{tick:.{decimals_phi}f}' for tick in M_ticks])

	bott_min, bott_max = np.nanmin(bott_index_array), np.nanmax(bott_index_array)
	cbar_ticks = np.linspace(bott_min, bott_max, num_ticks_cb)
	cbar.set_ticks(cbar_ticks)
	cbar.set_ticklabels([f'{tick:.{decimals_cb}f}' for tick in cbar_ticks])

	plt.tight_layout()
	plt.savefig(outputfile)
	plt.show()


def plot_lattice_sites(n, fractal=False, change_basis=None):

	geometric_data = compute_geometric_data(n, False)
	x, y, NN, hexaflake = [
		geometric_data[key] for key in ['x', 'y', 'NN', 'hexaflake']
	]

	if fractal:
		NN = NN[np.ix_(hexaflake, hexaflake)]
		x = x[hexaflake]
		y = y[hexaflake]

	i, j = np.where(NN)
	upper_triangle = j > i
	i, j = i[upper_triangle], j[upper_triangle]

	point_colors = np.empty((x.size, 4))
	point_colors[::2, :] = np.array([1., 0., 0., 1.])
	point_colors[1::2, :] = np.array([0., 0.75, 1., 1.])

	line_color = (0., 0., 0., 1.)
	background_color = (0.75, 0.75, 0.75, 1.)

	fig, ax = plt.subplots()
	ax.set_facecolor(background_color)

	if change_basis:
		x, y = change_basis(x, y)

	ax.plot([x[i], x[j]], [y[i], y[j]], c=line_color, linewidth=2, zorder=1)
	ax.scatter(x, y, c=point_colors, s=3 ** (5 - n), zorder=2)

	ax.set_xlabel('x')
	ax.set_ylabel('y', rotation=0)

	num_ticks_xy, decimals_xy = 5, 2
	x_ticks = np.linspace(x.min(), x.max(), num_ticks_xy)
	y_ticks = np.linspace(y.min(), y.max(), num_ticks_xy)
	ax.set_xticks(x_ticks)
	ax.set_yticks(y_ticks)
	ax.set_xticklabels([f'{tick:.{decimals_xy}f}' for tick in x_ticks])
	ax.set_yticklabels([f'{tick:.{decimals_xy}f}' for tick in y_ticks])

	plt.axis('equal')
	plt.tight_layout()
	plt.show()


def plot_relative_distances(n, PBC, rel_origin=(0., 0.), abs_dist=True, fractal=False):

	geometric_data = compute_geometric_data(n, PBC, return_dx_dy=True)

	x, y, delta_x_discrete, delta_y_discrete = [
		geometric_data[key] for key in ['x', 'y', 'delta_x_discrete', 'delta_y_discrete']
	]

	if fractal:
		hexaflake = geometric_data['hexaflake']
		x, y = x[hexaflake], y[hexaflake]
		delta_x_discrete = delta_x_discrete[np.ix_(hexaflake, hexaflake)]
		delta_y_discrete = delta_y_discrete[np.ix_(hexaflake, hexaflake)]

	x_origin_rel, y_origin_rel = rel_origin
	x_origin = (x_origin_rel / 2) * (x.max() - x.min())
	y_origin = (y_origin_rel / 2) * (y.max() - y.min())
	origin_site = np.argmin((x - x_origin)**2 + (y - y_origin)**2)

	x_true_origin, y_true_origin = x[origin_site], y[origin_site]

	x_dist = (1 / 2) * delta_x_discrete[origin_site]
	y_dist = (np.sqrt(3) / 2) * delta_y_discrete[origin_site]
	euclidean_dist = np.sqrt(x_dist**2 + y_dist**2)
	r_span = euclidean_dist.max()

	if abs_dist:
		x_dist_plot = np.abs(x_dist)
		y_dist_plot = np.abs(y_dist)
		norm_xy = plt.Normalize(0, r_span)
	else:
		x_dist_plot = x_dist
		y_dist_plot = y_dist
		norm_xy = plt.Normalize(-r_span, r_span)

	norm_r = plt.Normalize(0, r_span)

	other_sites = np.delete(np.arange(x.size), origin_site)

	num_cbar_ticks = 7
	num_axes_ticks = 5
	fig, axs = plt.subplots(1, 3, figsize=(16, 4))

	xy_cmap = 'inferno' if abs_dist else 'jet'
	xy_origin_color = 'lime' if abs_dist else 'magenta'

	for i in range(3):
		axs[i].set_xlabel('x')
		axs[i].set_ylabel('y', rotation=0)
		axs[i].xaxis.set_label_coords(0.5, -0.1)
		axs[i].yaxis.set_label_coords(-0.133, 0.483)

	scatter1 = axs[0].scatter(
		x[other_sites],
		y[other_sites],
		c=x_dist_plot[other_sites],
		cmap=xy_cmap,
		s=20,
		norm=norm_xy
	)
	axs[0].scatter(x_true_origin, y_true_origin, c=xy_origin_color, s=100, marker='*')
	dx_title = '|Δx|' if abs_dist else 'Δx'
	axs[0].set_title(dx_title)
	axs[0].set_aspect('equal')
	axs[0].set_xticks(np.linspace(x.min(), x.max(), num_axes_ticks))
	axs[0].set_yticks(np.linspace(y.min(), y.max(), num_axes_ticks))
	axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

	divider1 = make_axes_locatable(axs[0])
	cax1 = divider1.append_axes("right", size="5%", pad=0.05)
	cb1 = fig.colorbar(scatter1, cax=cax1)
	cb1.set_ticks(np.linspace(norm_xy.vmin, norm_xy.vmax, num_cbar_ticks))
	cb1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

	scatter2 = axs[1].scatter(
		x[other_sites],
		y[other_sites],
		c=y_dist_plot[other_sites],
		cmap=xy_cmap,
		s=20,
		norm=norm_xy
	)
	axs[1].scatter(x_true_origin, y_true_origin, c=xy_origin_color, s=100, marker='*')
	dy_title = '|Δy|' if abs_dist else 'Δy'
	axs[1].set_title(dy_title)
	axs[1].set_aspect('equal')
	axs[1].set_xticks(np.linspace(x.min(), x.max(), num_axes_ticks))
	axs[1].set_yticks(np.linspace(y.min(), y.max(), num_axes_ticks))
	axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

	divider2 = make_axes_locatable(axs[1])
	cax2 = divider2.append_axes("right", size="5%", pad=0.05)
	cb2 = fig.colorbar(scatter2, cax=cax2)
	cb2.set_ticks(np.linspace(norm_xy.vmin, norm_xy.vmax, num_cbar_ticks))
	cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

	scatter3 = axs[2].scatter(
		x[other_sites],
		y[other_sites],
		c=euclidean_dist[other_sites],
		cmap='inferno',
		s=20,
		norm=norm_r
	)
	axs[2].scatter(x_true_origin, y_true_origin, c='lime', s=100, marker='*')
	axs[2].set_title('Δr')
	axs[2].set_aspect('equal')
	axs[2].set_xticks(np.linspace(x.min(), x.max(), num_axes_ticks))
	axs[2].set_yticks(np.linspace(y.min(), y.max(), num_axes_ticks))
	axs[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

	divider3 = make_axes_locatable(axs[2])
	cax3 = divider3.append_axes("right", size="5%", pad=0.05)
	cb3 = fig.colorbar(scatter3, cax=cax3)
	cb3.set_ticks(np.linspace(norm_r.vmin, norm_r.vmax, num_cbar_ticks))
	cb3.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

	plt.tight_layout()
	plt.show()


def plot_tiled_lattice(n, fractal=False, change_basis=None, s=10):

	colors = ((0, 0, 0),
			  (1, 0, 0),
			  (0, 1, 0),
			  (0, 0, 1),
			  (1, 1, 0),
			  (0, 1, 1),
			  (1, 0, 1))
	plot_background_color = (0.5, 0.5, 0.5)

	y_discrete, x_discrete = np.where(compute_hexagon(n))

	a = round(np.sqrt(2 * x_discrete.size - 3))
	b = (a + 3) // 2
	c = (a - 3) // 2
	d = 2 * a - b
	e = 2 * a - c

	shifts = np.array([
		[0, 0],
		[-3, a],
		[3, -a],
		[d, b],
		[-d, -b],
		[-e, c],
		[e, -c]
	])

	all_x_original = []
	all_y_original = []
	all_c = []

	for i, (sx, sy) in enumerate(shifts):
		x_shifted = x_discrete + sx
		y_shifted = y_discrete + sy

		all_x_original += list(x_shifted)
		all_y_original += list(y_shifted)
		all_c += [colors[i]] * x_shifted.size

	all_x_original, all_y_original, all_c = [
		np.array(this_list) for this_list in
		[all_x_original, all_y_original, all_c]
	]

	all_x_original = (1 / 2) * (all_x_original - 3 ** (n + 1) + 1)
	all_y_original = (np.sqrt(3) / 4) * (2 * all_y_original - 3 ** (n + 1) + 1)

	if fractal:
		hexaflake = compute_hexaflake(n)[y_discrete, x_discrete]
		hexaflake = np.tile(hexaflake, 7)
		all_x_original, all_y_original, all_c = [
			arr[hexaflake] for arr in [all_x_original, all_y_original, all_c]
		]

	num_ticks_xy, decimals_xy = 5, 2

	if change_basis:
		all_x_transformed, all_y_transformed = change_basis(all_x_original, all_y_original)

		fig, axes = plt.subplots(1, 2, figsize=(16, 8))

		axes[0].scatter(all_x_original, all_y_original, c=all_c, cmap='jet', s=s)
		axes[0].set_facecolor(plot_background_color)
		axes[0].set_title("Original Basis")
		axes[0].set_xlabel('x')
		axes[0].set_ylabel('y', rotation=0)

		x_min_orig, x_max_orig = all_x_original.min(), all_x_original.max()
		y_min_orig, y_max_orig = all_y_original.min(), all_y_original.max()
		x_ticks_orig = np.linspace(x_min_orig, x_max_orig, num_ticks_xy)
		y_ticks_orig = np.linspace(y_min_orig, y_max_orig, num_ticks_xy)
		axes[0].set_xticks(x_ticks_orig)
		axes[0].set_yticks(y_ticks_orig)
		axes[0].set_xticklabels([f'{tick:.{decimals_xy}f}' for tick in x_ticks_orig])
		axes[0].set_yticklabels([f'{tick:.{decimals_xy}f}' for tick in y_ticks_orig])

		axes[0].set_aspect((x_max_orig - x_min_orig)/(y_max_orig - y_min_orig))

		axes[1].scatter(all_x_transformed, all_y_transformed, c=all_c, cmap='jet', s=s)
		axes[1].set_facecolor(plot_background_color)
		axes[1].set_title("Transformed Basis")
		axes[1].set_xlabel('x')
		axes[1].set_ylabel('y', rotation=0)

		x_min_trans, x_max_trans = all_x_transformed.min(), all_x_transformed.max()
		y_min_trans, y_max_trans = all_y_transformed.min(), all_y_transformed.max()
		x_ticks_trans = np.linspace(x_min_trans, x_max_trans, num_ticks_xy)
		y_ticks_trans = np.linspace(y_min_trans, y_max_trans, num_ticks_xy)
		axes[1].set_xticks(x_ticks_trans)
		axes[1].set_yticks(y_ticks_trans)
		axes[1].set_xticklabels([f'{tick:.{decimals_xy}f}' for tick in x_ticks_trans])
		axes[1].set_yticklabels([f'{tick:.{decimals_xy}f}' for tick in y_ticks_trans])

		axes[1].set_aspect((x_max_trans - x_min_trans)/(y_max_trans - y_min_trans))

		plt.tight_layout()

	else:
		fig, ax = plt.subplots(figsize=(10, 10))
		ax.scatter(all_x_original, all_y_original, c=all_c, cmap='jet', s=s)
		ax.set_facecolor(plot_background_color)
		ax.set_title("Tiled Lattice")
		ax.set_xlabel('x')
		ax.set_ylabel('y', rotation=0)

		x_min, x_max = all_x_original.min(), all_x_original.max()
		y_min, y_max = all_y_original.min(), all_y_original.max()
		x_ticks = np.linspace(x_min, x_max, num_ticks_xy)
		y_ticks = np.linspace(y_min, y_max, num_ticks_xy)
		ax.set_xticks(x_ticks)
		ax.set_yticks(y_ticks)
		ax.set_xticklabels([f'{tick:.{decimals_xy}f}' for tick in x_ticks])
		ax.set_yticklabels([f'{tick:.{decimals_xy}f}' for tick in y_ticks])

		ax.set_aspect((x_max - x_min)/(y_max - y_min))

		plt.tight_layout()

	plt.show()


def main():

	compute_eigen = 1
	plot_eigen = 1
	compute_band = 0
	plot_band = 0
	compute_phase = 0
	plot_phase = 0
	plot_lattice = 0
	plot_distances = 0
	plot_tiled = 0

	if compute_eigen:
		n = 3
		PBC = True
		method = 'renorm'
		M_rel, phi = 0, np.pi / 2
		t1, t2 = 1., 1.

		M = M_rel * 3 * np.sqrt(3) * t2 * np.sin(phi)
		geometric_data = compute_geometric_data(n, PBC)
		eigen_data = compute_eigen_data(method, M, phi, t1, t2, geometric_data)
		np.savez('eigen_data.npz', **eigen_data)

	if plot_eigen:
		eigen_data = np.load('eigen_data.npz')
		plot_spectrum_and_LDOS(eigen_data)

	if compute_band:
		band_data = compute_band_data(method='hexagon',n=2, resolution=11, PBC=True, sparse=False)
		np.savez('band_data.npz', **band_data)

	if plot_band:
		data = np.load('band_data.npz')
		plot_band_gap_and_width(data)

	if compute_phase:
		phase_data = compute_phase_diagram(method='site_elim', n=2, resolution=25, n_jobs=4)
		np.savez('test.npz', **phase_data)

	if plot_phase:
		data = np.load('test.npz')
		plot_phase_diagram(data,titleparams='', outputfile='usleess.png')

	if plot_lattice:
		plot_lattice_sites(n=3, fractal=True, change_basis=None)

	if plot_distances:
		plot_relative_distances(n=2, PBC=True, rel_origin=(-1/4, 2/3), abs_dist=True, fractal=True)

	if plot_tiled:
		plot_tiled_lattice(n=2, fractal=False, change_basis=triangular_basis, s=8)


def hamiltonian_imshow(method, n, doBott, plotHamiltonian, plotHexaflake):	
	def get_subflake(coords, generation, doPlot=False):
		centers = (3**generation)*np.array([np.array(([2*np.cos(np.pi/3*a)], [2/np.sqrt(3)*np.sin(np.pi/3*a)])) for a in range(6)])[..., 0].T
		centers = np.append([[0],[0]],centers, axis=1) 
		centers[0] += np.mean(coords[0])
		centers[1] += np.mean(coords[1])

		dx, dy = np.power(coords[..., np.newaxis] - centers[:, np.newaxis, :], 2)
		difference = dx + dy
		which_subflake = np.argmin(difference, axis=-1)

		if doPlot:
			plt.scatter(coords[0], coords[1], c=which_subflake)
			plt.scatter(centers[0], centers[1], c='k', alpha=0.25)
			plt.show()

		return np.argsort(which_subflake)


	geo_data = compute_geometric_data(n, True)

	if doBott:
		eigen_data = compute_eigen_data(method, 0.0, -np.pi/2, 1.0, 1.0, geo_data)
		bott = compute_bott_index(eigen_data)
		print(f"Bott Index :: {bott}")

	if plotHamiltonian:
		H = compute_hamiltonian(method, 0.0, np.pi/2, 1.0, 1.0, geo_data)
		fig, axs = plt.subplots(1,2)
		ims_real = axs[0].imshow(H.real)
		axs[0].set_title("Real part")
		ims_imag = axs[1].imshow(H.imag)
		axs[1].set_title("Imaginary part")
		plt.show()

	if plotHexaflake:
		hexaflake = geo_data['hexaflake']
		x_discrete, y_discrete = geo_data['x_discrete'], geo_data['y_discrete']
		x, y = x_discrete[hexaflake], y_discrete[hexaflake]
		
		def plot_tiered_color(n_tiers, x, y):
			unit = np.linspace(0, 1, n_tiers)
			values = np.sort(np.repeat(unit, len(x)//n_tiers))[:len(x)]
			plt.scatter(x,y, c=plt.get_cmap('viridis')(values))
			plt.show()

		def plot_color(x, y):
			unit = np.linspace(0,1,x.size)
			plt.scatter(x,y, c=plt.get_cmap('viridis')(unit))
			plt.show()


		coords = np.empty((2, x.size))
		coords[0] = x
		coords[1] = y

		sorted_idxs = sort_hexaflake_by_subflake(coords, n)
		plot_color(coords[0, sorted_idxs], coords[1, sorted_idxs])

		


if __name__ == '__main__':
	main()
	#hamiltonian_imshow('site_elim', 3, False, False, True)
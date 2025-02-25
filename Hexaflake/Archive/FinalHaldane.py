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


def compute_hopping_arrays(delta_x_discrete, delta_y_discrete, sublattice):
	"""
	Based on discrete coordinate differences and sublattice membership (A or B),
	identify which pairs of sites are nearest neighbors (NN), and which pairs
	are next-nearest neighbors (NNN) with clockwise (CW) or counterclockwise (CCW)
	hopping phases.

	Args:
		delta_x_discrete (np.ndarray): 2D array of discrete x differences.
		delta_y_discrete (np.ndarray): 2D array of discrete y differences.
		sublattice (np.ndarray): 1D boolean indicating whether each site is sublattice A (True) or B (False).

	Returns:
		tuple:
			- (np.ndarray) NN: 2D boolean, True where i,j are nearest neighbors.
			- (np.ndarray) NNN_CCW: 2D boolean, True where i,j have next-nearest neighbor with CCW phase.
			- (np.ndarray) NNN_CW: 2D boolean, True where i,j have next-nearest neighbor with CW phase.
	"""

	# Sublattice membership checks.
	A_to_A = sublattice[:, None] & sublattice[None, :]
	B_to_B = ~sublattice[:, None] & ~sublattice[None, :]

	# Identify "wind" directions for phases. (wind_1, wind_2 partition the six directions).
	wind_1 = ((delta_x_discrete == 0) & (delta_y_discrete < 0)) | \
			 ((delta_x_discrete > 0) & (delta_y_discrete > 0)) | \
			 ((delta_x_discrete < 0) & (delta_y_discrete > 0))

	wind_2 = ((delta_x_discrete == 0) & (delta_y_discrete > 0)) | \
			 ((delta_x_discrete < 0) & (delta_y_discrete < 0)) | \
			 ((delta_x_discrete > 0) & (delta_y_discrete < 0))

	# Counterclockwise / clockwise arrays.
	CCW = (A_to_A & wind_1) | (B_to_B & wind_2)
	CW = (A_to_A & wind_2) | (B_to_B & wind_1)

	# Nearest neighbors (NN): steps of 2 in x with 0 in y, or steps of 1 in x with 1 in y.
	NN = ((np.abs(delta_x_discrete) == 2) & (delta_y_discrete == 0)) | \
		 ((np.abs(delta_x_discrete) == 1) & (np.abs(delta_y_discrete) == 1))

	# Next-nearest neighbors (NNN): steps of 0 in x with 2 in y, or 3 in x with 1 in y.
	NNN = ((delta_x_discrete == 0) & (np.abs(delta_y_discrete) == 2)) | \
		  ((np.abs(delta_x_discrete) == 3) & (np.abs(delta_y_discrete) == 1))

	# Subset NNN with CCW or CW phases.
	NNN_CCW = NNN & CCW
	NNN_CW = NNN & CW

	return NN, NNN_CCW, NNN_CW


def compute_geometric_data(n, PBC):
	"""
	Compute various geometrical arrays and differences needed for constructing
	the Hamiltonian. This includes the hexagon lattice array, hexaflake lattice array,
	sublattice assignments, and discrete differences in coordinates.

	Args:
		n (int): Size parameter for the lattice and hexaflake.
		PBC (bool): Flag indicating whether periodic boundary conditions are used.

	Returns:
		dict: A dictionary containing:
			- x, y (np.ndarray): Real-space coordinates of each site.
			- (np.ndarray) delta_x_discrete, delta_y_discrete: 2D arrays of differences in x-coordinates and y-coordinates.
			- hexaflake (np.ndarray): Boolean sub-array indicating sites in the hexaflake lattice.
			- sublattice (np.ndarray): Boolean array indicating sublattice assignments (A/B).
			- NN, NNN_CCW, NNN_CW (np.ndarray): Hopping arrays for nearest neighbors,
			  next-nearest neighbors with CCW, and next-nearest neighbors with CW, respectively.
	"""

	# Compute the background hexagon lattice and the fractal (hexaflake).
	hexagon_array = compute_hexagon(n)
	hexaflake_array = compute_hexaflake(n)

	# Sublattice array: 'A' sites are assigned True in regular horizontal intervals.
	sublattice_array = np.zeros_like(hexagon_array)
	sublattice_array[:, ::3] = hexagon_array[:, ::3]

	# Extract coordinates (y, x) in the hexagon array that are True.
	y_discrete, x_discrete = np.where(hexagon_array)

	# Convert discrete coordinates to real-space coordinates x, y.
	x = (1 / 2) * (x_discrete - 3 ** (n + 1) + 1)
	y = (np.sqrt(3) / 4) * (2 * y_discrete - 3 ** (n + 1) + 1)

	# Identify sublattice membership for each site in the lattice.
	sublattice = sublattice_array[y_discrete, x_discrete]

	# Identify which of those sites belong to the fractal (hexaflake).
	hexaflake = hexaflake_array[y_discrete, x_discrete]

	# Compute the discrete differences in x and y, respecting PBC if set.
	delta_x_discrete, delta_y_discrete = compute_dx_and_dy_discrete(
		x_discrete, y_discrete, PBC
	)

	# Determine the neighbor relationships.
	NN, NNN_CCW, NNN_CW = compute_hopping_arrays(
		delta_x_discrete, delta_y_discrete, sublattice
	)

	geometric_data = {
		'x': x,
		'y': y,
		'delta_x_discrete': delta_x_discrete,
		'delta_y_discrete': delta_y_discrete,
		'hexaflake': hexaflake,
		'sublattice': sublattice,
		'NN': NN,
		'NNN_CCW': NNN_CCW,
		'NNN_CW': NNN_CW
	}

	return geometric_data


def compute_hamiltonian(method, M, phi, t1, t2, geometric_data):
	"""
	Construct the Hamiltonian matrix (either the full hexagon or a reduced
	site-eliminated/renormalized version for the fractal).

	Args:
		method (str): Method used to construct the Hamiltonian. One of:
			- 'hexagon': Use the full lattice.
			- 'site_elim': Eliminate sites not in the fractal (hexaflake).
			- 'renorm': Use a renormalized Hamiltonian approach.
		M (float): Sublattice potential term (on-site energy difference).
		phi (float): Phase factor for complex next-nearest neighbor hopping.
		t1 (float): Nearest-neighbor hopping amplitude.
		t2 (float): Next-nearest neighbor hopping amplitude.
		geometric_data (dict): Contains sublattice info, neighbor relationships, etc.

	Returns:
		np.ndarray: A complex 2D Hamiltonian matrix.
	"""

	valid_methods = ['hexagon', 'site_elim', 'renorm']
	if method not in valid_methods:
		raise ValueError(f"Invalid method '{method}'. Options are {valid_methods}.")

	# Extract geometric data arrays.
	sublattice = geometric_data['sublattice']
	NN = geometric_data['NN']
	NNN_CCW = geometric_data['NNN_CCW']
	NNN_CW = geometric_data['NNN_CW']
	hexaflake = geometric_data['hexaflake']

	# Initialize the Hamiltonian with zeros.
	H = np.zeros(NN.shape, dtype=np.complex128)

	# Place sublattice potential M on A sites, -M on B sites.
	np.fill_diagonal(H, np.where(sublattice, M, -M))

	# Populate Hamiltonian with hopping terms.
	H[NN] = -t1
	H[NNN_CCW] = t2 * np.exp(1j * phi)
	H[NNN_CW] = t2 * np.exp(-1j * phi)

	# Methods that reduce the Hamiltonian (site elimination or renormalization).
	if method == 'renorm':
		# Partition the Hamiltonian into fractal (A) vs. non-fractal (B) blocks.
		H_aa = H[np.ix_(hexaflake, hexaflake)]
		H_bb = H[np.ix_(~hexaflake, ~hexaflake)]
		H_ab = H[np.ix_(hexaflake, ~hexaflake)]
		H_ba = H[np.ix_(~hexaflake, hexaflake)]

		# Renormalize by integrating out the non-fractal sites.
		H = H_aa - H_ab @ sp.linalg.solve(
			H_bb,
			H_ba,
			assume_a='her',
			check_finite=False,
			overwrite_a=True,
			overwrite_b=True
		)

	elif method == 'site_elim':
		# Simply keep the fractal sub-block.
		H = H[np.ix_(hexaflake, hexaflake)]

	return H


def precompute_sparse_hamiltonian_data(geometric_data, fractal):
	"""
	Precompute arrays needed to construct a sparse Hamiltonian matrix
	(indices of sublattice sites, neighbor pairs, etc.), optionally
	restricting to fractal (hexaflake) sites.

	Args:
		geometric_data (dict): Contains sublattice, neighbor relationships, etc.
		fractal (bool): Whether to restrict data to fractal sites only.

	Returns:
		tuple: Precomputed data suitable for assembling a sparse Hamiltonian.
	"""

	# Extract relevant arrays.
	sublattice = geometric_data['sublattice']
	NN = geometric_data['NN']
	NNN_CCW = geometric_data['NNN_CCW']
	NNN_CW = geometric_data['NNN_CW']

	# Restrict to fractal sites if requested.
	if fractal:
		hexaflake = geometric_data['hexaflake']
		sublattice = sublattice[hexaflake]
		NN = NN[np.ix_(hexaflake, hexaflake)]
		NNN_CCW = NNN_CCW[np.ix_(hexaflake, hexaflake)]
		NNN_CW = NNN_CW[np.ix_(hexaflake, hexaflake)]

	# Separate out sublattice A vs. B indices.
	sublatticeA_idxs = np.arange(sublattice.size)[sublattice]
	sublatticeB_idxs = np.arange(sublattice.size)[~sublattice]

	# Find matrix positions where neighbor relationships are True.
	row_nn, col_nn = np.where(NN)
	row_nnn_ccw, col_nnn_ccw = np.where(NNN_CCW)
	row_nnn_cw, col_nnn_cw = np.where(NNN_CW)

	return (sublatticeA_idxs, sublatticeB_idxs,
			row_nn, col_nn, row_nnn_ccw, col_nnn_ccw, row_nnn_cw, col_nnn_cw)


def compute_sparse_hamiltonian(M, phi, t1, t2, precomputed_sparse_data):
	"""
	Construct the sparse Hamiltonian matrix in COO (and then CSR) format
	using precomputed neighbor indices and sublattice data.

	Args:
		M (float): Sublattice potential term (A = M, B = -M).
		phi (float): Phase for complex next-nearest neighbor hopping.
		t1 (float): Nearest-neighbor hopping amplitude.
		t2 (float): Next-nearest neighbor hopping amplitude.
		precomputed_sparse_data (tuple): Outputs from `precompute_sparse_hamiltonian_data`.

	Returns:
		scipy.sparse.csr_matrix: The Hamiltonian in CSR sparse format.
	"""

	(sublatticeA_idxs, sublatticeB_idxs,
	 row_nn, col_nn,
	 row_nnn_ccw, col_nnn_ccw,
	 row_nnn_cw, col_nnn_cw) = precomputed_sparse_data

	# Determine total dimension.
	system_size = sublatticeA_idxs.size + sublatticeB_idxs.size

	# On-site potential for sublattices A and B.
	data_sublatticeA = np.full(sublatticeA_idxs.size, M, dtype=np.complex128)
	data_sublatticeB = np.full(sublatticeB_idxs.size, -M, dtype=np.complex128)

	# Hopping data.
	data_nn = np.full(row_nn.size, -t1, dtype=np.complex128)
	data_nnn_ccw = np.full(row_nnn_ccw.size, t2 * np.exp(1j * phi), dtype=np.complex128)
	data_nnn_cw = np.full(row_nnn_cw.size, t2 * np.exp(-1j * phi), dtype=np.complex128)

	# Concatenate row, col, and data for all contributions.
	all_rows = np.concatenate([sublatticeA_idxs, sublatticeB_idxs, row_nn, row_nnn_ccw, row_nnn_cw])
	all_cols = np.concatenate([sublatticeA_idxs, sublatticeB_idxs, col_nn, col_nnn_ccw, col_nnn_cw])
	all_data = np.concatenate([data_sublatticeA, data_sublatticeB, data_nn, data_nnn_ccw, data_nnn_cw])

	# Build the sparse Hamiltonian (COO -> CSR).
	H = coo_matrix((all_data, (all_rows, all_cols)), shape=(system_size, system_size)).tocsr()

	return H


def compute_bott_index(eigenvectors, x, y):
	"""
	Compute the Bott index from the given eigenvectors and site coordinates.

	The function constructs complex phase factors (unitary operators) in
	two directions (U1, U2), projects them onto the occupied states
	(subspace spanned by the columns of `V`), and uses them to form a
	closed loop operator `A`. The Bott index is extracted from the
	eigenvalues of this operator.

	Args:
		eigenvectors (np.ndarray): Eigenvectors sorted such that the first
			`num_sites` columns span the lower band.
		x (np.ndarray): x-coordinates of lattice sites.
		y (np.ndarray): y-coordinates of lattice sites.

	Returns:
		int: The Bott index, an integer (rounded) topological invariant.
	"""
	# Number of lattice sites.
	num_sites = x.size

	# V is the subspace spanned by the lower band (first num_sites columns).
	V = eigenvectors[:, :num_sites]

	# Estimate the linear dimension N of the system (assuming a specific geometry).
	N = round((np.sqrt(2 * num_sites - 3) - 3) / 2 + 2)
	# Characteristic length scale (L) in the lattice geometry.
	L = np.sqrt(3) * N

	# Compute new coordinates q1, q2, which will be used to construct phase factors.
	q1 = (np.sqrt(3) / 2) * x - (1 / 2) * y
	q2 = (np.sqrt(3) / 2) * x + (1 / 2) * y

	# Define unitary phase factors (exponential of i*k*r).
	U1 = np.exp(1j * 2 * np.pi * q1 / L)[:, np.newaxis]
	U2 = np.exp(1j * 2 * np.pi * q2 / L)[:, np.newaxis]

	# Project these operators into the occupied subspace.
	U1_proj = V.conj().T @ (V * U1)
	U2_proj = V.conj().T @ (V * U2)

	# Construct the closed loop operator A.
	A = U1_proj @ U2_proj @ U1_proj.conj().T @ U2_proj.conj().T

	# Compute eigenvalues of A, then take the sum of the logarithms.
	eigenvaluesA = sp.linalg.eigvals(A, overwrite_a=True)
	trace_logA = np.sum(np.log(eigenvaluesA))

	# Bott index is the integer part (rounded) of the phase over 2π.
	bott = round(np.imag(trace_logA) / (2 * np.pi))

	return bott


def compute_eigen_data(n, PBC, method, M, phi, t1, t2):
	"""
	Compute eigenvalues and eigenvectors of the Hamiltonian for a given set
	of parameters.

	Args:
		n (int): Lattice size/fractal iteration parameter.
		PBC (bool): Periodic boundary conditions flag.
		method (str): Method to construct Hamiltonian ('hexagon', 'site_elim', 'renorm').
		M (float): On-site energy for sublattice A sites.
		phi (float): Complex phase for next-nearest neighbor hopping.
		t1 (float): Nearest-neighbor hopping amplitude.
		t2 (float): Next-nearest neighbor hopping amplitude.

	Returns:
		dict: A dictionary containing:
			- x, y (np.ndarray): Real-space coordinates of the retained sites.
			- eigenvalues (np.ndarray): Hamiltonian eigenvalues.
			- eigenvectors (np.ndarray): Hamiltonian eigenvectors.
	"""

	# Get geometric data (coordinates, sublattice, neighbors).
	geometric_data = compute_geometric_data(n, PBC)
	x, y = geometric_data['x'], geometric_data['y']

	# Construct Hamiltonian matrix.
	H = compute_hamiltonian(method, M, phi, t1, t2, geometric_data)

	# Diagonalize (standard Hermitian eigenvalue problem).
	eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)

	# If using site elimination or renorm, restrict x,y to fractal subset.
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
		method='hexagon', sparse=False, PBC=True, resolution=10,
		M_resolution=None, phi_resolution=None,
		M_range=(-5.5, 5.5), phi_range=(-np.pi, np.pi),
		n=3, t1=1., t2=1., k=20, n_jobs=4):
	"""
	Compute band gap and band width data over ranges of M and phi.

	Args:
		method (str): Lattice method ('hexagon', 'site_elim', or 'renorm').
		sparse (bool): If True, construct a sparse Hamiltonian and use `eigsh`.
		PBC (bool): Periodic boundary conditions flag.
		resolution (int): Default resolution for scanning M and phi if not set.
		M_resolution (int, optional): Number of M points. Defaults to `resolution`.
		phi_resolution (int, optional): Number of phi points. Defaults to `resolution`.
		M_range (tuple): Range (min, max) of M values.
		phi_range (tuple): Range (min, max) of phi values.
		n (int): Lattice size / fractal iteration order.
		t1 (float): Nearest-neighbor hopping amplitude.
		t2 (float): Next-nearest neighbor hopping amplitude.
		k (int): Number of eigenvalues to compute if using sparse routine.
		n_jobs (int): Number of parallel jobs for `joblib.Parallel`. Negative
			indicates using all but the specified number of cores.

	Returns:
		dict: A dictionary containing:
			- band_gap_array (np.ndarray): 2D array of band gaps.
			- band_width_array (np.ndarray): 2D array of band widths.
			- M_range (np.ndarray): The range of M values used.
			- phi_range (np.ndarray): The range of phi values used.
			- t1, t2 (float): Hopping amplitudes used.
	"""

	valid_methods = ['hexagon', 'site_elim', 'renorm']
	if method not in valid_methods:
		raise ValueError(f"Invalid method '{method}'. Options are {valid_methods}.")

	if method == 'renorm' and sparse:
		raise ValueError('Sparse method is not applicable for method of renormalization.')

	# Determine resolution for M and phi if not specified.
	M_resolution = resolution if M_resolution is None else M_resolution
	phi_resolution = resolution if phi_resolution is None else phi_resolution

	# Ensure odd resolution for symmetric scanning.
	M_resolution += 1 - M_resolution % 2
	phi_resolution += 1 - phi_resolution % 2

	# Precompute geometric data and partial data for sparse Hamiltonian.
	geometric_data = compute_geometric_data(n, PBC)
	fractal = False if method == 'hexagon' else True
	precomputed_sparse_data = precompute_sparse_hamiltonian_data(geometric_data, fractal)

	def worker(param_idxs):
		# Each worker in parallel computes the band gap and width for a single (M, phi).
		M_idx, phi_idx = param_idxs
		M_val = (M_range[1] - M_range[0]) * (M_idx / (M_resolution - 1)) + M_range[0]
		phi_val = (phi_range[1] - phi_range[0]) * (phi_idx / (phi_resolution - 1)) + phi_range[0]
		try:
			if sparse:
				# Compute sparse Hamiltonian and partial eigenvalues (k).
				H = compute_sparse_hamiltonian(M_val, phi_val, t1, t2, precomputed_sparse_data)
				eigenvalues = eigsh(H, k=k, which='LM', return_eigenvectors=False)
			else:
				# Compute full Hamiltonian.
				H = compute_hamiltonian(method, M_val, phi_val, t1, t2, geometric_data)
				eigenvalues = sp.linalg.eigvalsh(H, overwrite_a=True)

			# Band gap: difference between the smallest positive eigenvalue and largest negative eigenvalue.
			positive = eigenvalues > 0
			band_gap = eigenvalues[positive].min() - eigenvalues[~positive].max()
			# Band width: overall range of eigenvalues.
			band_width = eigenvalues.max() - eigenvalues.min()
			return [M_idx, phi_idx, band_gap, band_width]

		except Exception as e:
			print(f'An error occurred for (M, phi)=({M_val:.3f}, {phi_val:.3f}): {e}')
			return [M_idx, phi_idx, np.nan, np.nan]

	# Generate all combinations of M and phi indices.
	idxs_sets = tuple(product(range(M_resolution), range(phi_resolution)))

	# Parallel loop over all (M, phi) combinations.
	with tqdm_joblib(tqdm(total=len(idxs_sets), desc='Computing band gap diagram...')):
		M_data, phi_data, band_gap_data, band_width_data = np.array(
			Parallel(n_jobs=n_jobs)(delayed(worker)(idx_set) for idx_set in idxs_sets)
		).T

	# Reshape band gap and band width arrays in 2D.
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
        method='hexagon', resolution=10,
        M_resolution=None, phi_resolution=None,
        M_range=(-5.5, 5.5), phi_range=(-np.pi, np.pi),
        n=3, t1=1., t2=1., n_jobs=4):
	"""
	Compute a phase diagram by evaluating the Bott index over a grid of M and phi values.
	Periodic boundary conditions are assumed internally to ensure a well-defined topology.

	Args:
		method (str): One of 'hexagon', 'site_elim', or 'renorm'. Determines
			how the Hamiltonian is constructed (full lattice, site elimination,
			or renormalization).
		resolution (int): Default number of points to sample for M and phi if
			M_resolution or phi_resolution are not specified.
		M_resolution (int, optional): Number of M points. Defaults to `resolution`.
		phi_resolution (int, optional): Number of phi points. Defaults to `resolution`.
		M_range (tuple): Range (min, max) of M values to scan.
		phi_range (tuple): Range (min, max) of phi values to scan.
		n (int): System size / fractal iteration order.
		t1 (float): Nearest-neighbor hopping amplitude.
		t2 (float): Next-nearest neighbor hopping amplitude.
		n_jobs (int): Number of parallel jobs for `joblib.Parallel`. Negative
			value indicates using all but the specified number of cores.

	Returns:
		dict: A dictionary containing:
			- bott_index_array (np.ndarray): 2D array (size M_resolution x phi_resolution)
			  of Bott indices.
			- M_range (np.ndarray): The range of M values used for scanning.
			- phi_range (np.ndarray): The range of phi values used for scanning.
			- t1, t2 (float): Hopping amplitudes used.
	"""
	valid_methods = ['hexagon', 'site_elim', 'renorm']
	if method not in valid_methods:
		raise ValueError(f"Invalid method '{method}'. Options are {valid_methods}.")

	# Determine resolution for M and phi if not specified.
	M_resolution = resolution if M_resolution is None else M_resolution
	phi_resolution = resolution if phi_resolution is None else phi_resolution

	# Ensure odd resolution for symmetric scanning.
	M_resolution += 1 - M_resolution % 2
	phi_resolution += 1 - phi_resolution % 2

	# Precompute geometric data with PBC=True for topological phases.
	geometric_data = compute_geometric_data(n, True)
	x, y = geometric_data['x'], geometric_data['y']

	def worker(param_idxs):
		# Each worker in parallel computes the Bott index for a single (M, phi).
		M_idx, phi_idx = param_idxs
		M_val = (M_range[1] - M_range[0]) * (M_idx / (M_resolution - 1)) + M_range[0]
		phi_val = (phi_range[1] - phi_range[0]) * (phi_idx / (phi_resolution - 1)) + phi_range[0]
		try:
			H = compute_hamiltonian(method, M_val, phi_val, t1, t2, geometric_data)
			eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)
			bott = compute_bott_index(eigenvectors, x, y)
			return [M_idx, phi_idx, bott]

		except Exception as e:
			print(f'An error occurred for (M, phi)=({M_val:.3f}, {phi_val:.3f}): {e}')
			return [M_idx, phi_idx, np.nan]

	# Generate all combinations of M and phi indices.
	idxs_sets = tuple(product(range(M_resolution), range(phi_resolution)))

	# Parallel loop over all (M, phi) combinations.
	with tqdm_joblib(tqdm(total=len(idxs_sets), desc='Computing phase diagram...')):
		M_data, phi_data, bott_index_data = np.array(
			Parallel(n_jobs=n_jobs)(delayed(worker)(idx_set) for idx_set in idxs_sets)
		).T

	# Reshape Bott index array in 2D.
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
	"""
	Plot the full spectrum of eigenvalues, highlighting selected states,
	and plot the local density of states (LDOS) of those states in real space.

	Args:
		eigen_data (dict): Contains 'x', 'y', 'eigenvalues', and 'eigenvectors'.
		num_states (int): Number of states (eigenvalues) near zero to highlight (even number).
		cmap (str): Matplotlib colormap for the LDOS scatter plot.
	"""

	# Number of ticks and decimals for those ticks to display on each axis.
	num_ticks_E, decimals_E = 5, 2
	num_ticks_n, decimals_n = 5, 0
	num_ticks_x, decimals_x = 5, 2
	num_ticks_y, decimals_y = 5, 2
	num_ticks_LDOS, decimals_LDOS = 5, 2

	# Ensure num_states is even.
	num_states += 1 - num_states % 2

	# Extract data from eigen_data.
	x, y, eigenvalues, eigenvectors = [
		eigen_data[key] for key in ['x', 'y', 'eigenvalues', 'eigenvectors']
	]

	all_idxs = np.arange(eigenvalues.size)
	positive = eigenvalues > 0

	# Choose the top few negative eigenstates and the bottom few positive eigenstates.
	positive_idxs = all_idxs[positive][np.argsort(eigenvalues[positive])][:num_states // 2]
	negative_idxs = all_idxs[~positive][np.argsort(eigenvalues[~positive])[::-1]][:num_states // 2]

	# Indices of selected states to highlight for the LDOS. These are plotted in red in the spectrum.
	LDOS_idxs = np.concatenate((negative_idxs, positive_idxs))
	# Other indices to plot in the spectrum (in black).
	other_idxs = np.delete(all_idxs, LDOS_idxs)

	# Compute local density of states by summing squared wavefunction components for the selected states.
	LDOS = np.sum(np.abs(eigenvectors[:, LDOS_idxs]) ** 2, axis=1)

	fig, axs = plt.subplots(1, 2, figsize=(12, 6))

	# Aspect ratio for the spectrum plot to make it visually balanced.
	spectrum_aspect = (
							  (eigenvalues.size - 1) / (eigenvalues.max() - eigenvalues.min())
					  ) * (
							  (y.max() - y.min()) / (x.max() - x.min())
					  )

	# Left plot: spectrum
	axs[0].scatter(other_idxs, eigenvalues[other_idxs], c='black', s=20)
	axs[0].scatter(LDOS_idxs, eigenvalues[LDOS_idxs], c='red', s=30)
	axs[0].set_xlabel('n')
	axs[0].set_ylabel('E', rotation=0)
	axs[0].set_title('Spectrum')
	axs[0].set_aspect(spectrum_aspect)

	# Ticks for the x-axis (index n) of the spectrum.
	n_ticks = np.linspace(0, eigenvalues.size - 1, num_ticks_n)
	axs[0].set_xticks(n_ticks)
	axs[0].set_xticklabels([f'{tick:.{decimals_n}f}' for tick in n_ticks])

	# Ticks for the y-axis (energy E).
	E_ticks = np.linspace(eigenvalues.min(), eigenvalues.max(), num_ticks_E)
	axs[0].set_yticks(E_ticks)
	axs[0].set_yticklabels([f'{tick:.{decimals_E}f}' for tick in E_ticks])

	# Right plot: LDOS in real space.
	LDOS_scatter = axs[1].scatter(x, y, c=LDOS, cmap=cmap, s=7.5)
	axs[1].set_aspect('equal')
	axs[1].set_xlabel('x')
	axs[1].set_ylabel('y', rotation=0)
	axs[1].set_title('LDOS')

	# Ticks for x-coordinate.
	x_ticks = np.linspace(x.min(), x.max(), num_ticks_x)
	axs[1].set_xticks(x_ticks)
	axs[1].set_xticklabels([f'{tick:.{decimals_x}f}' for tick in x_ticks])

	# Ticks for y-coordinate.
	y_ticks = np.linspace(y.min(), y.max(), num_ticks_y)
	axs[1].set_yticks(y_ticks)
	axs[1].set_yticklabels([f'{tick:.{decimals_y}f}' for tick in y_ticks])

	# Colorbar for LDOS.
	bbox = axs[1].get_position()
	cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
	cbar = fig.colorbar(LDOS_scatter, cax=cbar_ax)
	LDOS_ticks = np.linspace(LDOS.min(), LDOS.max(), num_ticks_LDOS)
	cbar.set_ticks(LDOS_ticks)
	cbar.set_ticklabels([f'{tick:.{decimals_LDOS}e}' for tick in LDOS_ticks])

	plt.show()


def plot_band_gap_and_width(band_data, cmap='inferno'):
	"""
	Plot 2D diagrams of the band gap and band width as functions of M and phi,
	with consistent tick formatting and axis labels (including color bars).
	"""

	band_gap_array = band_data['band_gap_array'].T
	band_width_array = band_data['band_width_array'].T
	M_range = band_data['M_range']
	phi_range = band_data['phi_range']

	# Number of ticks and decimal places for axes
	num_ticks_M, decimals_M = 5, 2
	num_ticks_phi, decimals_phi = 5, 2
	# Number of ticks and decimal places for color bars
	num_ticks_cb, decimals_cb = 5, 2

	fig, axs = plt.subplots(1, 2, figsize=(14, 6))

	# ---------------------------------------
	# Left subplot: band gap
	# ---------------------------------------
	im1 = axs[0].imshow(
		band_gap_array,
		extent=[M_range[0], M_range[1], phi_range[0], phi_range[1]],
		origin='lower',
		aspect='auto',
		cmap=cmap
	)
	axs[0].set_title('Band Gap')
	axs[0].set_xlabel('M')
	axs[0].set_ylabel('Phi', rotation=0)
	cbar1 = fig.colorbar(im1, ax=axs[0])

	# Axis ticks (M and phi)
	M_ticks = np.linspace(M_range[0], M_range[1], num_ticks_M)
	axs[0].set_xticks(M_ticks)
	axs[0].set_xticklabels([f'{tick:.{decimals_M}f}' for tick in M_ticks])

	phi_ticks = np.linspace(phi_range[0], phi_range[1], num_ticks_phi)
	axs[0].set_yticks(phi_ticks)
	axs[0].set_yticklabels([f'{tick:.{decimals_phi}f}' for tick in phi_ticks])

	# Color bar tick formatting for band gap
	band_gap_min, band_gap_max = np.nanmin(band_gap_array), np.nanmax(band_gap_array)
	cbar1_ticks = np.linspace(band_gap_min, band_gap_max, num_ticks_cb)
	cbar1.set_ticks(cbar1_ticks)
	cbar1.set_ticklabels([f'{tick:.{decimals_cb}f}' for tick in cbar1_ticks])

	# ---------------------------------------
	# Right subplot: band width
	# ---------------------------------------
	im2 = axs[1].imshow(
		band_width_array,
		extent=[M_range[0], M_range[1], phi_range[0], phi_range[1]],
		origin='lower',
		aspect='auto',
		cmap=cmap
	)
	axs[1].set_title('Band Width')
	axs[1].set_xlabel('M')
	axs[1].set_ylabel('Phi', rotation=0)
	cbar2 = fig.colorbar(im2, ax=axs[1])

	# Axis ticks (M and phi)
	axs[1].set_xticks(M_ticks)
	axs[1].set_xticklabels([f'{tick:.{decimals_M}f}' for tick in M_ticks])
	axs[1].set_yticks(phi_ticks)
	axs[1].set_yticklabels([f'{tick:.{decimals_phi}f}' for tick in phi_ticks])

	# Color bar tick formatting for band width
	band_width_min, band_width_max = np.nanmin(band_width_array), np.nanmax(band_width_array)
	cbar2_ticks = np.linspace(band_width_min, band_width_max, num_ticks_cb)
	cbar2.set_ticks(cbar2_ticks)
	cbar2.set_ticklabels([f'{tick:.{decimals_cb}f}' for tick in cbar2_ticks])

	plt.tight_layout()
	plt.show()


def plot_phase_diagram(phase_data, cmap='viridis'):
	"""
	Plot the Bott index as a function of M and phi from precomputed phase diagram data,
	with consistent tick formatting and axis labels (including color bar).
	"""

	# Transpose to match the way imshow expects data (y, x).
	bott_index_array = phase_data['bott_index_array'].T
	M_range = phase_data['M_range']
	phi_range = phase_data['phi_range']

	# Number of ticks and decimal places for axes
	num_ticks_M, decimals_M = 5, 2
	num_ticks_phi, decimals_phi = 5, 2
	# Number of ticks and decimal places for color bar
	num_ticks_cb, decimals_cb = 5, 2

	fig, ax = plt.subplots(figsize=(7, 6))
	im = ax.imshow(
		bott_index_array,
		extent=[M_range[0], M_range[1], phi_range[0], phi_range[1]],
		origin='lower',
		aspect='auto',
		cmap=cmap
	)
	ax.set_title('Phase Diagram (Bott Index)')
	ax.set_xlabel('M')
	ax.set_ylabel('Phi', rotation=0)
	cbar = fig.colorbar(im, ax=ax)

	# Axis ticks (M and phi)
	M_ticks = np.linspace(M_range[0], M_range[1], num_ticks_M)
	ax.set_xticks(M_ticks)
	ax.set_xticklabels([f'{tick:.{decimals_M}f}' for tick in M_ticks])

	phi_ticks = np.linspace(phi_range[0], phi_range[1], num_ticks_phi)
	ax.set_yticks(phi_ticks)
	ax.set_yticklabels([f'{tick:.{decimals_phi}f}' for tick in phi_ticks])

	# Color bar tick formatting for Bott index
	bott_min, bott_max = np.nanmin(bott_index_array), np.nanmax(bott_index_array)
	cbar_ticks = np.linspace(bott_min, bott_max, num_ticks_cb)
	cbar.set_ticks(cbar_ticks)
	cbar.set_ticklabels([f'{tick:.{decimals_cb}f}' for tick in cbar_ticks])

	plt.tight_layout()
	plt.show()


def plot_lattice_sites(n, fractal=False):
	"""
	Plot the lattice sites and nearest-neighbor bonds for a hexagonal or fractal geometry,
	with consistent tick formatting and axis labels.
	"""
	geometric_data = compute_geometric_data(n, False)
	x, y, sublattice, NN, hexaflake = [
		geometric_data[key] for key in ['x', 'y', 'sublattice', 'NN', 'hexaflake']
	]

	# If fractal, restrict to the hexaflake portion of the lattice.
	if fractal:
		NN = NN[np.ix_(hexaflake, hexaflake)]
		x = x[hexaflake]
		y = y[hexaflake]
		sublattice = sublattice[hexaflake]

	# Extract indices of bonds (i, j) where j > i to avoid double-plotting.
	i, j = np.where(NN)
	upper_triangle = j > i
	i, j = i[upper_triangle], j[upper_triangle]

	# Color sublattice A (True) as red, sublattice B (False) as cyan/blue.
	point_colors = np.empty((sublattice.size, 4))
	point_colors[sublattice, :] = np.array([1., 0., 0., 1.])  # Red for sublattice A
	point_colors[~sublattice, :] = np.array([0., 0.75, 1., 1.])  # Cyan/blue for sublattice B

	line_color = (0., 0., 0., 1.)  # Black lines for bonds
	background_color = (0.75, 0.75, 0.75, 1.)  # Light gray background

	fig, ax = plt.subplots()
	ax.set_facecolor(background_color)

	# Plot bonds.
	ax.plot([x[i], x[j]], [y[i], y[j]], c=line_color, linewidth=2, zorder=1)
	# Plot sites.
	ax.scatter(x, y, c=point_colors, s=3 ** (5 - n), zorder=2)

	# Set axis labels
	ax.set_xlabel('x')
	ax.set_ylabel('y', rotation=0)

	# Define the number of ticks and decimal format for x and y.
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
	"""
	Plot the relative distances (in x, y, and radial) from a chosen origin site
	to every other site in the lattice (or fractal subset if `fractal=True`).

	This function uses precomputed discrete coordinate differences stored in
	`delta_x_discrete` and `delta_y_discrete`, along with the real-space
	coordinates x, y, to visualize how far each site is from an origin site.
	The origin site is chosen by specifying a relative position `rel_origin`,
	which is mapped onto the lattice's range in x and y.

	Args:
		n (int): System size or fractal iteration parameter for constructing
			the lattice geometry.
		PBC (bool): Flag indicating whether periodic boundary conditions are
			applied during geometry construction.
		rel_origin (tuple): A pair (x_rel, y_rel) with each component in [-0.5, 0.5].
			These values map to a position in the real-space domain from which
			the closest lattice site is identified as the 'origin site'.
		abs_dist (bool): If True, plot the absolute values of Δx and Δy; if False,
			the signed values are displayed.
		fractal (bool): If True, restrict the plot to sites in the fractal
			(hexaflake) subset of the lattice. Otherwise, plot all sites.

	Returns:
		None
	"""
	geometric_data = compute_geometric_data(n, PBC)

	# Extract relevant arrays: real-space coordinates and discrete differences.
	x, y, delta_x_discrete, delta_y_discrete = [
		geometric_data[key] for key in ['x', 'y', 'delta_x_discrete', 'delta_y_discrete']
	]

	# Restrict to fractal (hexaflake) subset if requested.
	if fractal:
		hexaflake = geometric_data['hexaflake']
		x, y = x[hexaflake], y[hexaflake]
		delta_x_discrete = delta_x_discrete[np.ix_(hexaflake, hexaflake)]
		delta_y_discrete = delta_y_discrete[np.ix_(hexaflake, hexaflake)]

	# Map relative origin (rel_origin) onto the real-space domain to find the site closest to that point.
	x_origin_rel, y_origin_rel = rel_origin
	x_origin = (x_origin_rel / 2) * (x.max() - x.min())
	y_origin = (y_origin_rel / 2) * (y.max() - y.min())
	origin_site = np.argmin((x - x_origin)**2 + (y - y_origin)**2)

	# Calculate real-space origin for reference.
	x_true_origin, y_true_origin = x[origin_site], y[origin_site]

	# Compute x, y, and radial distances from the origin site to all other sites.
	x_dist = (1 / 2) * delta_x_discrete[origin_site]
	y_dist = (np.sqrt(3) / 2) * delta_y_discrete[origin_site]
	euclidean_dist = np.sqrt(x_dist**2 + y_dist**2)
	r_span = euclidean_dist.max()

	# Use absolute or signed distances for plotting.
	if abs_dist:
		x_dist_plot = np.abs(x_dist)
		y_dist_plot = np.abs(y_dist)
		norm_xy = plt.Normalize(0, r_span)
	else:
		x_dist_plot = x_dist
		y_dist_plot = y_dist
		norm_xy = plt.Normalize(-r_span, r_span)

	norm_r = plt.Normalize(0, r_span)

	# Identify all sites except the chosen origin.
	other_sites = np.delete(np.arange(x.size), origin_site)

	# Set up figure and axes.
	num_cbar_ticks = 7
	num_axes_ticks = 5
	fig, axs = plt.subplots(1, 3, figsize=(16, 4))

	# Choose color maps depending on whether abs_dist is used.
	xy_cmap = 'inferno' if abs_dist else 'jet'
	xy_origin_color = 'lime' if abs_dist else 'magenta'

	# Configure axis labels and positions.
	for i in range(3):
		axs[i].set_xlabel('x')
		axs[i].set_ylabel('y', rotation=0)
		axs[i].xaxis.set_label_coords(0.5, -0.1)
		axs[i].yaxis.set_label_coords(-0.133, 0.483)

	# Plot Δx or |Δx|.
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

	# Plot Δy or |Δy|.
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

	# Plot radial distance Δr.
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
	"""
	Plot a single hexagon lattice, along with its periodically shifted copies
	(tiles), in the original coordinate system or an optional transformed basis.

	The function computes a set of shifts that represent the fundamental
	translations needed to wrap the lattice under periodic boundary
	conditions. It applies each shift to all lattice sites, collecting the
	coordinates and color labels for visualization. Optionally, if `fractal`
	is True, the plot is restricted to the hexaflake (fractal) portion of the
	lattice only. If a `change_basis` function is provided, the lattice is
	plotted side-by-side in both the original and the transformed coordinate
	systems.

	Args:
		n (int):
			Size parameter controlling the lattice dimensions (and fractal
			construction if `fractal` is True).
		fractal (bool):
			If True, only plot the subset of sites belonging to the hexaflake
			fractal. Defaults to False (plot the entire hexagon).
		change_basis (callable, optional):
			A function of the form f(x_coords, y_coords) -> (x_new, y_new),
			which transforms coordinates from the original basis to a new one.
			If provided, two plots are shown side-by-side (original and
			transformed). Defaults to None (no transform).
		s (float):
			Marker size for the scatter plot points.

	Returns:
		None
	"""
	# Define colors for the lattice copies and a background color.
	colors = ((0, 0, 0),
			  (1, 0, 0),
			  (0, 1, 0),
			  (0, 0, 1),
			  (1, 1, 0),
			  (0, 1, 1),
			  (1, 0, 1))
	plot_background_color = (0.5, 0.5, 0.5)

	# Compute the discrete hexagon coordinates for order n.
	y_discrete, x_discrete = np.where(compute_hexagon(n))

	# Estimate system dimensions for shifting the lattice copies.
	a = round(np.sqrt(2 * x_discrete.size - 3))
	b = (a + 3) // 2
	c = (a - 3) // 2
	d = 2 * a - b
	e = 2 * a - c

	# Define shifts that replicate the lattice under periodic boundary conditions.
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

	# Apply each shift and collect coordinates + color labels.
	for i, (sx, sy) in enumerate(shifts):
		x_shifted = x_discrete + sx
		y_shifted = y_discrete + sy

		all_x_original += list(x_shifted)
		all_y_original += list(y_shifted)
		all_c += [colors[i]] * x_shifted.size

	# Convert lists to arrays.
	all_x_original, all_y_original, all_c = [
		np.array(this_list) for this_list in
		[all_x_original, all_y_original, all_c]
	]

	# Map discrete (x, y) to real-space coordinates.
	all_x_original = (1 / 2) * (all_x_original - 3 ** (n + 1) + 1)
	all_y_original = (np.sqrt(3) / 4) * (2 * all_y_original - 3 ** (n + 1) + 1)

	# If fractal is True, restrict to hexaflake sites.
	if fractal:
		# Construct the boolean array of hexaflake sites for the unshifted lattice.
		hexaflake = compute_hexaflake(n)[y_discrete, x_discrete]
		# Repeat that boolean array for the 7 shifts.
		hexaflake = np.tile(hexaflake, 7)
		# Filter arrays to keep only fractal sites.
		all_x_original, all_y_original, all_c = [
			arr[hexaflake] for arr in [all_x_original, all_y_original, all_c]
		]

	if change_basis:
		# Transform coordinates using the user-provided function.
		all_x_transformed, all_y_transformed = change_basis(all_x_original, all_y_original)

		fig, axes = plt.subplots(1, 2, figsize=(16, 8))

		# Plot the original basis.
		axes[0].scatter(all_x_original, all_y_original, c=all_c, cmap='jet', s=s)
		original_aspect = (
			(all_x_original.max() - all_x_original.min()) /
			(all_y_original.max() - all_y_original.min())
		)
		axes[0].set_aspect(original_aspect)
		axes[0].set_title("Original Basis")
		axes[0].set_facecolor(plot_background_color)

		# Plot the transformed basis.
		axes[1].scatter(all_x_transformed, all_y_transformed, c=all_c, cmap='jet', s=s)
		transformed_aspect = (
			(all_x_transformed.max() - all_x_transformed.min()) /
			(all_y_transformed.max() - all_y_transformed.min())
		)
		axes[1].set_aspect(transformed_aspect)
		axes[1].set_title("Transformed Basis")
		axes[1].set_facecolor(plot_background_color)

		plt.tight_layout()

	else:
		# Single-plot version (no basis change).
		plt.figure(figsize=(10, 10))
		plt.scatter(all_x_original, all_y_original, c=all_c, cmap='jet', s=s)
		original_aspect = (
			(all_x_original.max() - all_x_original.min()) /
			(all_y_original.max() - all_y_original.min())
		)
		plt.gca().set_aspect(original_aspect)
		plt.gca().set_facecolor(plot_background_color)
		plt.tight_layout()

	plt.show()


def main():
	"""
	Example main function demonstrating various computations:
	1) Compute and optionally save eigenvalues/eigenvectors.
	2) Plot those eigenvalues/eigenvectors if requested.
	3) Compute a band diagram over a range of parameters.
	4) Plot the band gap/width data if requested.
	5) Compute and plot a phase diagram by scanning M and phi (Bott index).
	6) Plot the lattice geometry (optionally just the fractal).
	7) Plot distances from a chosen origin site.
    8) Plot a 'tiled' lattice under periodic boundary shifts (with or without a basis transform).

	Modify the boolean flags below to enable or disable each step.
	"""
	# Set flags for which computations/plots to perform.
	compute_eigen = 0
	plot_eigen = 0
	compute_band = 0
	plot_band = 0
	compute_phase = 1
	plot_phase = 1
	plot_lattice = 0
	plot_distances = 0
	plot_tiled = 0

	if compute_eigen:
		# Parameters for the system.
		n = 3  # System size / fractal iteration order.
		PBC = False  # Periodic boundary condition flag.
		method = 'site_elim'  # One of 'hexagon', 'site_elim', or 'renorm'.
		M_rel, phi = 1 / 2, np.pi / 2  # Sublattice potential ratio and phase. Topological region: |M_rel| < 1
		t1, t2 = 1., 1.  # Nearest- and next-nearest-neighbor hopping amplitudes.

		# Compute the actual on-site potential M from M_rel and phi.
		M = M_rel * 3 * np.sqrt(3) * t2 * np.sin(phi)

		# Compute eigenvalues and eigenvectors.
		eigen_data = compute_eigen_data(n, PBC, method, M, phi, t1, t2)

		# Save computed data to a file for later retrieval.
		np.savez('eigen_data.npz', **eigen_data)

	if plot_eigen:
		# Load saved data and plot the spectrum and LDOS.
		eigen_data = np.load('eigen_data.npz')
		plot_spectrum_and_LDOS(eigen_data)

	if compute_band:
		# Compute band diagram across ranges of M and phi, then save.
		band_data = compute_band_data(n=2, resolution=50)
		np.savez('band_data.npz', **band_data)

	if plot_band:
		# Load band diagram data and plot gap and width.
		data = np.load('band_data.npz')
		plot_band_gap_and_width(data)

	if compute_phase:
		# Compute phase diagram (Bott index) for a range of M and phi.
		phase_data = compute_phase_diagram(n=2, resolution=50)
		np.savez('phase_data.npz', **phase_data)

	if plot_phase:
		# Load phase diagram data and plot Bott index.
		data = np.load('phase_data.npz')
		plot_phase_diagram(data)

	if plot_lattice:
		# Plot the lattice geometry for a given n.
		# Toggle fractal=True to show only the hexaflake sites.
		plot_lattice_sites(n=3, fractal=False)

	if plot_distances:
		# Plot distances from a chosen origin site in the lattice.
		plot_relative_distances(n=3, PBC=True, rel_origin=(-1/4, 2/3), abs_dist=True, fractal=True)

	if plot_tiled:
		# Plot a tiled view of the lattice with periodic shifts,
		# optionally restricting to fractal sites or using a coordinate transform.
		# Example usage with no basis transform and fractal=False:
		plot_tiled_lattice(n=2, fractal=False, change_basis=None, s=8)


if __name__ == '__main__':
	main()
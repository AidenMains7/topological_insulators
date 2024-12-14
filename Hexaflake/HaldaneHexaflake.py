import numpy as np
from scipy.linalg import eigh, eigvals
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from joblib import Parallel, delayed
from itertools import product
import sys
sys.path.append(".")
from Carpet.plotting import reshape_imshow_data, plot_imshow

def compute_hopping_array(coordinates_discrete, PBC):
	"""
    Computes the hopping arrays (delta_x and delta_y) between lattice sites,
    accounting for periodic boundary conditions (PBC) if specified.

    Parameters:
    - coordinates_discrete (np.ndarray): Array of discrete x and y coordinates of lattice sites.
    - PBC (bool): Flag indicating whether to apply periodic boundary conditions.

    Returns:
    - delta_x (np.ndarray): Array of x-coordinate differences between sites.
    - delta_y (np.ndarray): Array of y-coordinate differences between sites.
    """
	# Extract x and y coordinates
	x, y = coordinates_discrete.T

	if not PBC:
		# Compute differences in x and y coordinates without PBC
		delta_x = x[:, np.newaxis] - x
		delta_y = y[:, np.newaxis] - y
	else:
		# Compute differences with PBC using predefined shifts
		a = np.max(y) + 1
		b = (a + 3) // 2
		c = (a - 3) // 2
		d = 2 * a - b
		e = 2 * a - c

		# Define shift vectors for PBC
		shifts = np.array([
			[0, 0],
			[-3, a],
			[3, -a],
			[d, b],
			[-d, -b],
			[-e, c],
			[e, -c]
		])

		delta_x_stack = []
		delta_y_stack = []

		# Apply shifts and compute coordinate differences
		for shift in shifts:
			shifted_delta_x = x[:, np.newaxis] - (x + shift[0])
			shifted_delta_y = y[:, np.newaxis] - (y + shift[1])

			delta_x_stack.append(shifted_delta_x)
			delta_y_stack.append(shifted_delta_y)

		# Convert lists to arrays
		delta_x_stack = np.array(delta_x_stack)
		delta_y_stack = np.array(delta_y_stack)

		# Find indices of minimal distance (closest images)
		idx_array = np.argmin(delta_x_stack ** 2 + delta_y_stack ** 2, axis=0)

		# Create indices for selecting minimal differences
		i_indices, j_indices = np.indices(idx_array.shape)

		# Select minimal coordinate differences
		delta_x = delta_x_stack[idx_array, i_indices, j_indices]
		delta_y = delta_y_stack[idx_array, i_indices, j_indices]

	# Convert differences to integers
	return delta_x.astype(np.int64), delta_y.astype(np.int64)


def compute_geometric_data(iterations, PBC, arbitrary_hex=None):
	"""
    Computes the geometric data required for constructing the Haldane model
    on a hexagonal lattice (hexaflake structure).

    Parameters:
    - iterations (int): Number of iterations for constructing the hexaflake.
    - PBC (bool): Flag indicating whether to apply periodic boundary conditions.

    Returns:
    - geometric_data (dict): Dictionary containing geometric arrays and masks.
    """
	# Initial coordinates for a single hexagon
	initial_x = np.cos((np.pi / 3) * np.arange(6))
	initial_y = np.sin((np.pi / 3) * np.arange(6))

	def vertices(n):
		"""
        Recursively computes vertices of the hexaflake at iteration n.

        Parameters:
        - n (int): Iteration level.

        Returns:
        - new_x (np.ndarray): x-coordinates of vertices.
        - new_y (np.ndarray): y-coordinates of vertices.
        """
		if n == 0:
			return initial_x, initial_y

		new_x, new_y = [], []
		previous_x, previous_y = vertices(n - 1)

		new_scale = 3 ** n
		# Centers for new hexagons
		new_x_centers = [0] + list(new_scale * np.cos((np.pi / 3) * np.arange(6)))
		new_y_centers = [0] + list(new_scale * np.sin((np.pi / 3) * np.arange(6)))

		# Shift previous vertices to new centers
		for new_x_center, new_y_center in zip(new_x_centers, new_y_centers):
			new_x.extend(previous_x + new_x_center)
			new_y.extend(previous_y + new_y_center)

		return np.array(new_x), np.array(new_y)

	# Get vertices for the specified number of iterations
	x, y = vertices(iterations)

	# Convert coordinates to a discrete integer grid
	x_ints = np.round(2 * (x - np.min(x))).astype(int)
	y_ints = np.round((y - np.min(y)) / (np.sqrt(3) / 2)).astype(int)

	# Initialize hexaflake mask
	hexaflake_mask = np.full((np.max(y_ints) + 1, np.max(x_ints) + 1), False, dtype=bool)
	y_unique_ints = np.unique(y_ints)

	# Populate hexaflake mask
	for y_int in y_unique_ints:
		these_x_ints = np.unique(x_ints[y_ints == y_int])
		hexaflake_mask[y_int, these_x_ints] = True

	# Template for a single hexagon layer
	temp = np.array([
		[0, 1, 0, 1, 0, 0],
		[1, 0, 0, 0, 1, 0],
		[0, 1, 0, 1, 0, 0]
	])
	layers = (3 ** iterations - 1) // 2 + 1
	x_width = 2 * (3 ** (iterations + 1)) - 1
	piece1 = np.zeros((3 * layers, x_width), dtype=int)
	start = 0
	num_pieces = 3 ** iterations

	# Build up the hexagon mask
	for j in range(layers):
		section = np.zeros((3, x_width + 1), dtype=int)
		for i in range(num_pieces):
			section[:, start + 6 * i:start + 6 * (i + 1)] = temp
		piece1[3 * j:3 * (j + 1), :] = section[:, :-1]
		start += 3
		num_pieces -= 1

	# Flip and stack to complete the hexagon
	piece2 = np.flipud(piece1[3:, :])
	hexagon_mask = np.vstack((piece2, piece1)).astype(bool)

	if arbitrary_hex is not None:
		arbitrary_hex[0] = arbitrary_hex[0]*2
		arbitrary_hex[1] = arbitrary_hex[1]*2/np.sqrt(3)
		arbitrary_hex[0] -= np.min(arbitrary_hex[0])
		arbitrary_hex[1] -= np.min(arbitrary_hex[1])
		arbitrary_hex = np.unique(arbitrary_hex, axis=1)
		arbitrary_hex = arbitrary_hex.astype(int)
		idxs = np.lexsort((arbitrary_hex[0, :], arbitrary_hex[1, :]))
		arbitrary_hex = arbitrary_hex[:, idxs]
		lattice = np.full((np.max(arbitrary_hex[1]).astype(int)+1, np.max(arbitrary_hex[0]).astype(int)+1), 0, dtype=int)
		lattice[arbitrary_hex[1], arbitrary_hex[0]] = np.arange(arbitrary_hex.shape[1])+1
		hexagon_mask = lattice.astype(bool)
		hexaflake_mask = np.full(hexagon_mask.shape, False, dtype=bool)

	# Identify vacancies (sites in hexagon_mask not in hexaflake_mask)
	if arbitrary_hex is not None:
		vacancies_mask = hexagon_mask
	else:
		vacancies_mask = hexagon_mask & (~hexaflake_mask)

	Ly, Lx = hexagon_mask.shape

	# Initialize sublattice A mask
	sublattice_A_mask = np.full((Ly, Lx), False, dtype=bool)

	# Assign sublattice A and B sites
	for y in range(Ly):
		# Determine starting index for sublattice A based on row
		start_idx = (y + 1 + (1 - ((y // 3) % 2))) % 2
		slice_sites = np.where(hexagon_mask[y])[0][start_idx::2]
		sublattice_A_mask[y][slice_sites] = True

	# Sublattice B is the complement within hexagon_mask
	sublattice_B_mask = hexagon_mask & (~sublattice_A_mask)

	# Get coordinates of sites in the mask
	y_coordinates_discrete, x_coordinates_discrete = np.where(hexagon_mask)

	# Convert discrete coordinates to actual positions
	x_coordinates = x_coordinates_discrete / 2
	y_coordinates = (np.sqrt(3) / 2) * y_coordinates_discrete
	x_coordinates -= (x_coordinates.max() + x_coordinates.min()) / 2
	y_coordinates -= (y_coordinates.max() + y_coordinates.min()) / 2

	coordinates = np.array([x_coordinates, y_coordinates]).T

	# Discrete coordinates for hopping calculations
	coordinates_discrete = np.array([x_coordinates_discrete, y_coordinates_discrete]).T.astype(np.int64)

	# Compute hopping arrays
	delta_x, delta_y = compute_hopping_array(coordinates_discrete, PBC)

	# Get sublattice and site type masks for the sites
	sublattice_A = sublattice_A_mask[y_coordinates_discrete, x_coordinates_discrete]
	sublattice_B = sublattice_B_mask[y_coordinates_discrete, x_coordinates_discrete]
	hexaflake = hexaflake_mask[y_coordinates_discrete, x_coordinates_discrete]
	vacancies = vacancies_mask[y_coordinates_discrete, x_coordinates_discrete]

	N = hexagon_mask.sum()
	all_idxs = np.arange(N)

	# Reorder indices to have hexaflake sites first
	reordered_idxs = np.concatenate((all_idxs[hexaflake], all_idxs[vacancies]))

	# Reorder arrays accordingly
	coordinates, sublattice_A, sublattice_B, hexaflake, vacancies = [
		arr[reordered_idxs] for arr in [coordinates, sublattice_A, sublattice_B, hexaflake, vacancies]
	]

	# Reorder hopping arrays
	delta_x = delta_x[np.ix_(reordered_idxs, reordered_idxs)]
	delta_y = delta_y[np.ix_(reordered_idxs, reordered_idxs)]

	# Create masks for sublattice interactions
	A_to_A = sublattice_A[:, None] & sublattice_A[None, :]
	B_to_B = sublattice_B[:, None] & sublattice_B[None, :]

	# Define winding directions for NNN hopping
	wind_1 = ((delta_x == 0) & (delta_y < 0)) | \
			 ((delta_x > 0) & (delta_y > 0)) | \
			 ((delta_x < 0) & (delta_y > 0))

	wind_2 = ((delta_x == 0) & (delta_y > 0)) | \
			 ((delta_x < 0) & (delta_y < 0)) | \
			 ((delta_x > 0) & (delta_y < 0))

	# Counter-clockwise and clockwise winding masks
	CCW = (A_to_A & wind_1) | (B_to_B & wind_2)
	CW = (A_to_A & wind_2) | (B_to_B & wind_1)

	# Nearest neighbor (NN) and next-nearest neighbor (NNN) masks
	NN = ((np.abs(delta_x) == 2) & (delta_y == 0)) | \
		 ((np.abs(delta_x) == 1) & (np.abs(delta_y) == 1))

	NNN = ((delta_x == 0) & (np.abs(delta_y) == 2)) | \
		  ((np.abs(delta_x) == 3) & (np.abs(delta_y) == 1))

	# NNN hopping with directionality
	NNN_CCW = NNN & CCW
	NNN_CW = NNN & CW

	# Collect geometric data into a dictionary
	geometric_data = {
		'coordinates': coordinates,
		'hexaflake': hexaflake,
		'vacancies': vacancies,
		'sublattice_A': sublattice_A,
		'sublattice_B': sublattice_B,
		'NN': NN,
		'NNN_CCW': NNN_CCW,
		'NNN_CW': NNN_CW
	}

	return geometric_data


def haldane_hamiltonian_full(M, phi, t1, t2, geometric_data):
	"""
    Constructs the full Haldane Hamiltonian matrix for the given parameters
    and geometric data.

    Parameters:
    - M (float): Semiconducting mass term.
    - phi (float): Phase of the complex next-nearest neighbor hopping.
    - t1 (float): Nearest neighbor hopping amplitude.
    - t2 (float): Next-nearest neighbor hopping amplitude.
    - geometric_data (dict): Dictionary containing geometric arrays and masks.

    Returns:
    - H (np.ndarray): The Hamiltonian matrix.
    """
	# Unpack necessary data from geometric_data
	sublattice_A = geometric_data['sublattice_A']
	sublattice_B = geometric_data['sublattice_B']
	NN = geometric_data['NN']
	NNN_CCW = geometric_data['NNN_CCW']
	NNN_CW = geometric_data['NNN_CW']

	# Initialize Hamiltonian matrix
	H = np.zeros(NN.shape, dtype=np.complex128)

	# On-site energies: +M for sublattice A, -M for sublattice B
	H_diag = M * sublattice_A.astype(np.complex128) - M * sublattice_B.astype(np.complex128)
	np.fill_diagonal(H, H_diag)

	# Nearest neighbor hopping
	H[NN] = -t1

	# Next-nearest neighbor hopping with complex phase
	H[NNN_CCW] = -t2 * np.exp(-1j * phi)  # Counter-clockwise
	H[NNN_CW] = -t2 * np.exp(1j * phi)  # Clockwise

	return H


def mat_inv_hermitian(A):
	"""
    Computes the inverse of a Hermitian matrix A using eigenvalue decomposition,
    handling possible zero eigenvalues.

    Parameters:
    - A (np.ndarray): The Hermitian matrix to invert.

    Returns:
    - A_inv (np.ndarray): The inverse of matrix A.
    """
	# Compute eigenvalues (D) and eigenvectors (P) of A
	D, P = eigh(A, overwrite_a=True)

	# Find indices of non-zero eigenvalues
	non_zeros = np.where(np.abs(D) > 0)

	# Initialize inverse eigenvalues array
	D_inv = np.zeros(D.size, dtype=np.complex128)
	D_inv[non_zeros] = 1 / D[non_zeros]
	D_inv = np.diag(D_inv)

	# Reconstruct inverse matrix A_inv
	A_inv = np.dot(P, np.dot(D_inv, P.T.conj()))

	return A_inv


def data_wrapper(M, PBC, method, iterations=3, t1=1., t2=1., phi=np.pi / 2, ahex_L = None):
	"""
    Wraps the data preparation steps: computes geometric data, constructs the
    Hamiltonian, and computes eigenvalues and eigenvectors.

    Parameters:
    - M (float): Semiconducting mass term.
    - PBC (bool): Flag indicating whether to apply periodic boundary conditions.
    - method (str): Method to handle vacancies ('hexagon', 'site_elim', 'renorm').
    - iterations (int, optional): Number of iterations for constructing the hexaflake.
    - t1 (float, optional): Nearest neighbor hopping amplitude.
    - t2 (float, optional): Next-nearest neighbor hopping amplitude.
    - phi (float, optional): Phase of the complex next-nearest neighbor hopping.

    Returns:
    - data_dict (dict): Dictionary containing coordinates, eigenvalues, and eigenvectors.
    """
	if method not in ['hexagon', 'site_elim', 'renorm']:
		raise ValueError("Invalid method. Options are 'hexagon', 'site_elim', and 'renorm'.")

	# Compute geometric data
	if ahex_L is not None:
		ahex = honeycomb_lattice(ahex_L)
		geometric_data = compute_geometric_data(iterations, PBC, ahex)
	else:
		geometric_data = compute_geometric_data(iterations, PBC)

	# Unpack required data
	coordinates = geometric_data['coordinates']
	hexaflake = geometric_data['hexaflake']
	vacancies = geometric_data['vacancies']

	# Construct the full Hamiltonian
	H = haldane_hamiltonian_full(M, phi, t1, t2, geometric_data)

	if method == 'renorm':
		# Partition Hamiltonian into submatrices
		H_aa = H[np.ix_(hexaflake, hexaflake)]
		H_bb = H[np.ix_(vacancies, vacancies)]
		H_ab = H[np.ix_(hexaflake, vacancies)]
		H_ba = H[np.ix_(vacancies, hexaflake)]

		# Schur complement to eliminate vacancies (renormalization)
		H = H_aa - H_ab @ mat_inv_hermitian(H_bb) @ H_ba

	elif method == 'site_elim':
		# Remove vacancy rows and columns from Hamiltonian
		H = H[np.ix_(hexaflake, hexaflake)]

	# Compute eigenvalues and eigenvectors
	eigenvalues, eigenvectors = eigh(H, overwrite_a=True)

	if method in ['site_elim', 'renorm']:
		# Update coordinates to only include hexaflake sites
		coordinates = coordinates[hexaflake]

	return {
		'coordinates': coordinates,
		'eigenvalues': eigenvalues,
		'eigenvectors': eigenvectors,
		'hamiltonian': H
	}


def plot_data(data_dict, n_closest=2):
	"""
    Plots the eigenvalues and local density of states (LDOS) for the closest
    n_closest eigenvalues to zero energy.

    Parameters:
    - data_dict (dict): Dictionary containing coordinates, eigenvalues, and eigenvectors.
    - n_closest (int, optional): Number of eigenvalues closest to zero to consider.
    """
	# Unpack data
	coordinates = data_dict['coordinates']
	eigenvalues = data_dict['eigenvalues']
	eigenvectors = data_dict['eigenvectors']

	x, y = coordinates.T

	# Find indices of eigenvalues closest to zero
	sorted_eigs = np.sort(eigenvalues)
	pos_idxs = np.argwhere(sorted_eigs > 0)[:n_closest//2]
	neg_idxs = np.argwhere(sorted_eigs < 0)[-n_closest//2:]
	closest_to_zero_idxs = np.concatenate((pos_idxs.flatten(), neg_idxs.flatten()), axis=0)

	#closest_to_zero_idxs = np.argsort(np.abs(eigenvalues))[:n_closest]
	other_idxs = np.delete(np.arange(eigenvalues.size), closest_to_zero_idxs)





	# Compute LDOS from selected eigenvectors
	LDOS = (np.abs(eigenvectors[:, closest_to_zero_idxs]) ** 2).sum(axis=1)

	# Set up plots
	fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.2})

	# Adjust aspect ratio for eigenvalue plot
	x_range = x.max() - x.min()
	y_range = y.max() - y.min()
	E_range = eigenvalues.max() - eigenvalues.min()
	n_range = eigenvalues.size - 1

	spect_aspect = (n_range / E_range) * (y_range / x_range)

	# Plot eigenvalues
	axs[0].scatter(other_idxs, eigenvalues[other_idxs], c='black', s=20)
	axs[0].scatter(closest_to_zero_idxs, eigenvalues[closest_to_zero_idxs], c='red', s=30)
	axs[0].set_aspect(spect_aspect)
	axs[0].set_xlabel('State Index')
	axs[0].set_ylabel('Energy')
	axs[0].set_title('Eigenvalues of the Hamiltonian')

	# Plot LDOS
	LDOS_max = LDOS.max()
	LDOS_scatter = axs[1].scatter(x, y, c=LDOS, cmap='inferno', s=7.5, vmin=0, vmax=LDOS_max)
	axs[1].set_aspect('equal')
	axs[1].set_xlabel('x-coordinate')
	axs[1].set_ylabel('y-coordinate')
	axs[1].set_title('Local Density of States (LDOS)')

	# Add colorbar inset
	axins = inset_axes(
		axs[1],
		width="5%",  # width of colorbar
		height="100%",  # height of colorbar
		loc='center left',
		bbox_to_anchor=(1.05, 0., 1, 1),
		bbox_transform=axs[1].transAxes,
		borderpad=0
	)

	ticks = [0, LDOS_max / 2, LDOS_max]
	cbar = plt.colorbar(LDOS_scatter, cax=axins, ticks=ticks)
	cbar.ax.set_yticklabels([f"{tick:.1e}" for tick in ticks])
	cbar.set_label('LDOS')

	fig.subplots_adjust(left=0.1, right=0.85)

# ---
def projector_exact(H:np.ndarray, E_F:float) -> np.ndarray:
    '''
    Constructs the projector of the Hamiltonian onto the states below the Fermi energy

    Parameters: 
    H (ndarray): Hamiltonian operator
    fermi_energy (float): Fermi energy

    Returns: 
    P (ndarray): Projector operator  
    '''
    #eigenvalues and eigenvectors of the Hamiltonian
    eigvals, eigvecs = eigh(H, overwrite_a=True)

    #diagonal matrix 
    D = np.where(eigvals < E_F, 1.0 + 0.0j, 0.0 + 0.0j)
    D_dagger = np.einsum('i,ij->ij', D, eigvecs.conj().T)

    #projector given by matrix multiplication of eigenvectors and D_dagger
    P = eigvecs @ D_dagger

    return P


def bott_index(P:np.ndarray, lattice:np.ndarray) -> float:
	'''
	Computes the Bott Index for a given lattice and projector
	'''
	Y, X = np.where(lattice >= 0)[:]

	system_size = np.max(lattice) + 1
	states_per_site = P.shape[0] // system_size
	X = np.repeat(X, states_per_site)
	Y = np.repeat(Y, states_per_site)
	Ly, Lx = lattice.shape

	# Unitary matrices
	Ux = np.exp(1j*2*np.pi*X/Lx)
	Uy = np.exp(1j*2*np.pi*Y/Ly)

	UxP = np.einsum('i,ij->ij', Ux, P)
	UyP = np.einsum('i,ij->ij', Uy, P)
	Ux_daggerP = np.einsum('i,ij->ij', Ux.conj(), P)
	Uy_daggerP = np.einsum('i,ij->ij', Uy.conj(), P)

	A = np.eye(P.shape[0], dtype=np.complex128) - P + P.dot(UxP).dot(UyP).dot(Ux_daggerP).dot(Uy_daggerP)
	bott = round(np.imag(np.sum(np.log(eigvals(A)))) / (2 * np.pi))

	return bott


def bott_index_coordinates(P, coordinates):
	X = coordinates[0]
	Y = coordinates[1]
	system_size = coordinates.size//2
	states_per_site = P.shape[0] // system_size
	X = np.repeat(X, states_per_site)
	Y = np.repeat(Y, states_per_site)
	Ly, Lx = int(np.max(Y) - np.min(Y)), int(np.max(X) - np.min(X))

	# Unitary matrices
	Ux = np.exp(1j*2*np.pi*X/Lx)
	Uy = np.exp(1j*2*np.pi*Y/Ly)

	UxP = np.einsum('i,ij->ij', Ux, P)
	UyP = np.einsum('i,ij->ij', Uy, P)
	Ux_daggerP = np.einsum('i,ij->ij', Ux.conj(), P)
	Uy_daggerP = np.einsum('i,ij->ij', Uy.conj(), P)

	A = np.eye(P.shape[0], dtype=np.complex128) - P + P.dot(UxP).dot(UyP).dot(Ux_daggerP).dot(Uy_daggerP)
	bott = round(np.imag(np.sum(np.log(eigvals(A)))) / (2 * np.pi))

	return bott


def honeycomb_lattice(side_length: int) -> np.ndarray:
    """
    Generate a 2D honeycomb lattice with the specified side length.

    Parameters:
    side_length (int): Number of hexagon tiles on each side.

    Returns:
    np.ndarray: A 2xN array representing the coordinates of the lattice.
    """
    # Define the angles and generate a single hexagon
    angles = 2 * np.pi * np.arange(1, 7) / 6
    hexagon = np.array([np.cos(angles), np.sin(angles)])

    def _row_of_n_hexagons(n: int) -> np.ndarray:
        """Generate a row of `n` hexagons."""
        row = hexagon.copy()
        for i in range(1, n):
            row = np.append(row, hexagon + np.array([[3 * i], [0]]), axis=1)
        row[0] -= np.mean(row[0])
        row[1] -= np.mean(row[1])
        return row

    # Initialize the lattice
    hexagon_lattice = np.empty((2, 0))
    offset = np.sqrt(3) * 3 / 2

    # Generate the upper half of the honeycomb
    for i, num_hexagons in enumerate(range(side_length, 2 * side_length)):
        row = _row_of_n_hexagons(num_hexagons)
        row += np.array([[0], [i * offset]])
        hexagon_lattice = np.append(hexagon_lattice, row, axis=1)

    # Reflect the upper half to generate the lower half
    lower_half = np.array([[0], [2 * np.max(hexagon_lattice[1]) - np.sqrt(3)]]) - hexagon_lattice
    hexagon_lattice = np.append(hexagon_lattice, lower_half, axis=1)

    # Center the lattice around the origin
    hexagon_lattice[0] -= np.mean(hexagon_lattice[0])
    hexagon_lattice[1] -= np.mean(hexagon_lattice[1])

    # Ensure unique points
    hexagon_lattice = np.unique(hexagon_lattice, axis=1)

    return hexagon_lattice


def lattice_from_coords(coords):
    idxs = np.lexsort((coords[0, :], coords[1, :]))
    coords = coords[:, idxs]

    lattice = np.ones((np.max(coords[1]).astype(int)+1, np.max(coords[0]).astype(int)+1), dtype=int)*(-1)
    lattice[coords[1], coords[0]] = np.arange(coords.shape[1])
    return lattice
# ---


def bott_range(method, fname):
	e = 3*np.sqrt(3)	
	M_vals = [-e, -e/2, 0, e/2, e]
	p = np.pi
	phi_vals = [-p, -p/2, 0, p/2, p]
	params = tuple(product(M_vals, phi_vals))

	def worker(i):
		M, phi = params[i]
		data_dict = data_wrapper(M, True, method, phi=phi)
		coords = data_dict['coordinates'].T
		P = projector_exact(data_dict['hamiltonian'], 0.0)
		bott = bott_index_coordinates(P, coords)
		print(f"Completed {100*(i+1)/len(params):.2f}% : BI = {bott}")
		return [phi, M, bott]
	
	data = np.array(Parallel(n_jobs=4)(delayed(worker)(j) for j in range(len(params)))).T
	
	np.savez(fname, data)

	X, Y, Z = reshape_imshow_data(data)
	fig, ax = plt.subplots(1, 1, figsize=(10,10))
	fig, ax, cbar = plot_imshow(fig, ax, X, Y, Z, doDiscreteCmap=True)
	theta = np.linspace(-np.pi, np.pi, 100)
	ax.plot(theta, np.sin(theta))
	plt.savefig("haldane_bott_site_elim.png")



def main():
	"""
    Main function to execute the data preparation, eigenvalue computation,
    and plotting for the Haldane model on a hexaflake structure.
    """
	# Define parameters
	# Topological region: |M| < |3 * sqrt(3) * t2 * sin(phi)|
	p = np.pi/2
	M = (1 / 2) * 3 * np.sqrt(3)*np.sin(p) # Semiconducting mass term
	PBC = True  # Periodic boundary conditions
	method = 'hexagon'  # Method to handle vacancies
	# Start timing
	t0 = time.time()
	
	# Prepare data and compute eigenvalues and eigenvectors
	data_dict = data_wrapper(M, PBC, method, iterations=2, t1=1., t2=1., phi=p, ahex_L=28)
	return
	coords = data_dict['coordinates'].T
	coords_init = coords.copy()
	coords[0] = coords[0]*2
	coords[1] = coords[1]*2/np.sqrt(3)
	coords[0] -= np.min(coords[0])
	coords[1] -= np.min(coords[1])
	coords = np.unique(coords, axis=1)
	coords = coords.astype(int)
	lat = lattice_from_coords(coords)

	print(coords.shape)
	print(lat.shape)

	P = projector_exact(data_dict['hamiltonian'], 0.0)
	bi = bott_index(P, lat)
	bi2 = bott_index_coordinates(P, coords_init)
	print(f"BI = {bi}")
	print(f"BI (coords) = {bi2}")
	
	#from pybott import bott as pb_bott
	#pyb = pb_bott(coords_init.T, data_dict['hamiltonian'], 0.0)
	#print(f"pybott bott = {round(pyb, 0)}")

	dt = time.time() - t0
	print(f'Computation time: {round(dt)}s')

	# Plot results
	plot_data(data_dict)
	plt.show()


if __name__ == '__main__':
	main()

def extra():
	data = np.load('Hexaflake/hexagon_finer_bott.npz', allow_pickle=True)['arr_0']
	print(data[:, :5])
	flipped = -data
	flipped[1] *= -1		
	data = np.concatenate((data, flipped), axis=1)
	idxs = np.lexsort((data[0], data[1]))

	data = data[:, idxs]
	print(data[:, :5])

	X, Y, Z = reshape_imshow_data(data)
	fig, ax = plt.subplots(1, 1, figsize=(10,10))
	fig, ax, cbar = plot_imshow(fig, ax, X, Y, Z, cmap='viridis', doDiscreteCmap=True)
	theta = np.linspace(-np.pi, np.pi, 100)
	ax.plot(theta, 3*np.sqrt(3)*np.sin(theta), c='black', ls='--', label='bound')
	ax.plot(theta, -3*np.sqrt(3)*np.sin(theta), c='black', ls='--', label='bound2')
	ax.set_xlabel("$\\phi$ (radians)")
	ax.set_ylabel("M")
	ax.set_yticks([-np.sqrt(3)*3, 0, np.sqrt(3)*3])
	ax.set_yticklabels(['$-3\\sqrt{3}$', '0', '$3\\sqrt{3}$'])
	ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
	ax.set_xticklabels(['$-\\pi$', '$-\\frac{\\pi}{2}$', '0', '$\\frac{\\pi}{2}$', '$\\pi$'])
	ax.set_title("Hexagon Lattice, Parent of Gen. 3")
	cbar.set_label(label="Bott Index", size=15)

	ax.title.set_fontsize(20)

	ax.tick_params(labelsize=13)

	for item in [ax.xaxis.label, ax.yaxis.label]:
		item.set_fontsize(15)

	plt.show()

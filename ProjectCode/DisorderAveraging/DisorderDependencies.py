"""
Module for Analyzing Non-Interacting Crystalline Topological Insulators on Fractal Lattices

This module provides tools for constructing and analyzing Hamiltonians for non-interacting crystalline topological insulators on 2D fractal lattices, specifically the Sierpinski carpet. The methods implemented include both real-space and momentum-space techniques to handle the lack of translational symmetry in fractal lattices.

Functions:
- generate_lattices(order, pad_width): Generates square and fractal lattices (Sierpinski carpet) with specified order and padding.
- geometry_arrays(lattice, pbc, n): Constructs geometry arrays for distances and angles between lattice sites.
- wannier_matrices_symmetry_method(lattice, pbc, n, r0): Constructs Wannier matrices using the symmetry method.
- wannier_matrices_FT(lattice, pbc): Constructs Wannier matrices using Fourier transform method.
- Hamiltonian(M, B_tilde, wannier_matrices, t1, t2, B): Constructs the Hamiltonian matrix from given Wannier matrices and parameters.
- H_site_elim(H, fills, holes): Constructs an effective Hamiltonian by eliminating sites corresponding to holes.
- mat_inv(matrix, hermitian, alt, overwrite_a, tol): Computes the inverse of a matrix, with options for handling Hermitian matrices and small eigenvalues.
- mat_solve_iterative(matrix, tol): Solves a linear system iteratively using the Conjugate Gradient method. Used as an alternative method in 'H_renorm' for computing the inverse of H_bb.
- H_renorm(H, fills, holes): Constructs a renormalized effective Hamiltonian using the Schur complement.
- H_and_lattice_wrapper(lattice_order, method, M, B_tilde, pbc, pad_width, n, **kwargs): Wrapper function to generate Hamiltonian and lattice using specified method.
- uniform_mass_disorder(disorder_strength, system_size, internal_freedoms, sparse): Generates a uniform mass disorder operator.

This module allows for the study of topological properties and disorder effects on fractal lattices, enabling a comparison between real-space and momentum-space approaches. The tools provided facilitate the construction of Hamiltonians, the application of disorder, and the computation of properties of interest in topological insulators.
"""

import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse import dok_matrix, csr_matrix, diags
from scipy.sparse.linalg import cg


def generate_lattices(order, pad_width):
	"""
	Generates the square and fractal (Sierpinski carpet) lattices.

	Parameters:
	order (int): Order of the Sierpinski carpet.
	pad_width (int): Padding width for the carpet lattice.

	Returns:
	tuple: A tuple containing:
		- square_lattice (ndarray): Square lattice array.
		- carpet_lattice (ndarray): Fractal lattice array (Sierpinski carpet).
		- fills (ndarray): Indices of filled (non-vacant) sites in the carpet lattice.
		- holes (ndarray): Indices of vacant sites in the carpet lattice.
	"""

	def sierpinski_carpet(order_):
		# Recursively generate Sierpinski carpet. The base case is a single filled site.
		if order_ == 0:
			return np.array([[1]], dtype=int)

		# Generate a smaller carpet for the previous order.
		smaller_carpet = sierpinski_carpet(order_ - 1)
		size = smaller_carpet.shape[0]

		# Initialize the new carpet with all sites filled.
		new_carpet = np.ones((3 * size, 3 * size), dtype=int)
		# Create the central hole in the new carpet.
		new_carpet[size:(2 * size), size:(2 * size)] = np.zeros((size, size), dtype=int)

		# Copy the smaller carpet into each of the 8 surrounding blocks of the new carpet.
		for i in range(3):
			for j in range(3):
				if i != 1 or j != 1:
					new_carpet[i * size:(i + 1) * size, j * size:(j + 1) * size] = smaller_carpet

		return new_carpet

	# Generate the Sierpinski carpet of the specified order.
	carpet = sierpinski_carpet(order)

	# Add padding around the carpet if specified.
	if pad_width > 0:
		carpet = np.pad(carpet, pad_width, mode='constant', constant_values=1)

	# Flatten the carpet lattice to get a 1D array of sites.
	flat_carpet = carpet.flatten()
	# Get indices of filled sites in the flat carpet.
	ones_indices = np.flatnonzero(flat_carpet)
	# Initialize the carpet lattice with all sites marked as vacant.
	carpet_lattice = np.full(flat_carpet.shape, -1, dtype=int)
	# Assign indices to the filled sites in the carpet lattice.
	carpet_lattice[ones_indices] = np.arange(ones_indices.size)
	# Reshape the carpet lattice to match the original carpet shape.
	carpet_lattice = carpet_lattice.reshape(carpet.shape)

	# Determine the size of the carpet lattice.
	L = carpet.shape[0]
	# Generate a square lattice of the same size.
	square_lattice = np.arange(L ** 2).reshape((L, L))

	# Get indices of filled and vacant sites in the flat carpet.
	fills = np.where(flat_carpet == 1)[0]
	holes = np.where(flat_carpet == 0)[0]

	return square_lattice, carpet_lattice, fills, holes


def geometry_arrays(lattice, pbc, n):
	"""
	Generates geometry arrays for the given lattice.

	Parameters:
	lattice (ndarray): Lattice array.
	pbc (bool): Periodic boundary conditions.
	n (int): Cutoff length for distances.

	Returns:
	tuple: A tuple containing:
		- dr (ndarray): Array of distances.
		- cos_dphi (ndarray): Array of cosines of angles.
		- sin_dphi (ndarray): Array of sines of angles.
		- prncpl_mask (ndarray): Mask for principal axes directions.
		- diags_mask (ndarray): Mask for diagonal directions.
	"""
	side_length = lattice.shape[0]

	# Validate the cutoff length to ensure it is within bounds for periodic boundary conditions.
	if pbc and (n >= side_length // 2):
		raise ValueError("Cutoff length must be less than half the system size when PBC is applied.")

	# Find indices of filled lattice sites.
	filled_indices = np.argwhere(lattice >= 0)
	# Compute the difference between all pairs of filled site indices.
	difference = filled_indices[None, :, :] - filled_indices[:, None, :]
	dy, dx = difference[..., 0], difference[..., 1]

	# Apply periodic boundary conditions by wrapping distances that exceed half the lattice size.
	if pbc:
		dx = np.where(np.abs(dx) > side_length / 2, dx - np.sign(dx) * side_length, dx)
		dy = np.where(np.abs(dy) > side_length / 2, dy - np.sign(dy) * side_length, dy)

	# Mask for distances within the specified cutoff length.
	distance_mask = np.maximum(np.abs(dx), np.abs(dy)) <= n

	# Create masks for principal axes and diagonal directions within the cutoff distance.
	prncpl_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & distance_mask
	diags_mask = ((np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0))) & distance_mask

	# Combined mask for valid principal and diagonal directions.
	prelim_mask = prncpl_mask | diags_mask

	# Compute distances for valid site pairs.
	dr = np.where(prelim_mask, np.maximum(np.abs(dx), np.abs(dy)), 0)
	# Compute cosines of angles between valid site pairs.
	cos_dphi = np.where(prelim_mask, np.cos(np.arctan2(dy, dx)), 0.)
	# Compute sines of angles between valid site pairs.
	sin_dphi = np.where(prelim_mask, np.sin(np.arctan2(dy, dx)), 0.)

	return dr, cos_dphi, sin_dphi, prncpl_mask, diags_mask


def wannier_matrices_symmetry_method(lattice, pbc, n, r0=1):
	"""
	Constructs Wannier matrices using the symmetry method.

	Parameters:
	lattice (ndarray): Lattice array.
	pbc (bool): Periodic boundary conditions.
	n (int): Cutoff length for distances.
	r0 (float): Characteristic length for exponential decay.

	Returns:
	tuple: A tuple containing Wannier matrices:
		- I (ndarray): Identity matrix.
		- Sx (ndarray): Wannier matrix for x-direction sine term.
		- Sy (ndarray): Wannier matrix for y-direction sine term.
		- Cx_plus_Cy (ndarray): Wannier matrix for cosine terms.
		- CxSy (ndarray): Wannier matrix for x-sine and y-cosine term.
		- SxCy (ndarray): Wannier matrix for x-cosine and y-sine term.
		- CxCy (ndarray): Wannier matrix for x-cosine and y-cosine term.
	"""
	# Generate geometry arrays for distances and angles between lattice sites.
	dr, cos_dphi, sin_dphi, prncpl_mask, diags_mask = geometry_arrays(lattice, pbc, n)
	system_size = np.max(lattice) + 1
	# Identity matrix representing on-site terms.
	I = np.eye(system_size, dtype=np.complex128)

	# Exponential decay function for principal and diagonal directions.
	F_p = np.where(prncpl_mask, np.exp(1 - dr / r0), 0. + 0.j)
	F_d = np.where(diags_mask, np.exp(1 - dr / r0), 0. + 0.j)

	# Construct Wannier matrices for different terms based on geometry arrays and decay functions.
	Sx = 1j * cos_dphi * F_p / 2
	Sy = 1j * sin_dphi * F_p / 2
	Cx_plus_Cy = F_p / 2

	CxSy = 1j * sin_dphi * F_d / (2 * np.sqrt(2))
	SxCy = 1j * cos_dphi * F_d / (2 * np.sqrt(2))
	CxCy = F_d / 4

	return I, Sx, Sy, Cx_plus_Cy, CxSy, SxCy, CxCy


def wannier_matrices_FT(lattice, pbc):
	"""
	Constructs Wannier matrices using Fourier transform (FT) method.

	Parameters:
	lattice (ndarray): Lattice array.
	pbc (bool): Periodic boundary conditions.

	Returns:
	tuple: A tuple containing Wannier matrices:
		- I (ndarray): Identity matrix.
		- Sx (ndarray): Wannier matrix for x-direction sine term.
		- Sy (ndarray): Wannier matrix for y-direction sine term.
		- Cx (ndarray): Wannier matrix for x-direction cosine term.
		- Cy (ndarray): Wannier matrix for y-direction cosine term.
		- CxSy (ndarray): Wannier matrix for x-sine and y-cosine term.
		- SxCy (ndarray): Wannier matrix for x-cosine and y-sine term.
		- CxCy (ndarray): Wannier matrix for x-cosine and y-cosine term.
	"""
	system_size = np.max(lattice) + 1
	L_y, L_x = lattice.shape

	# Initialize Wannier matrices as sparse matrices to save memory and computation time.
	I = np.eye(system_size, dtype=np.complex128)
	Cx = dok_matrix((system_size, system_size), dtype=np.complex128)
	Sx = dok_matrix((system_size, system_size), dtype=np.complex128)
	Cy = dok_matrix((system_size, system_size), dtype=np.complex128)
	Sy = dok_matrix((system_size, system_size), dtype=np.complex128)
	CxSy = dok_matrix((system_size, system_size), dtype=np.complex128)
	SxCy = dok_matrix((system_size, system_size), dtype=np.complex128)
	CxCy = dok_matrix((system_size, system_size), dtype=np.complex128)

	# Iterate over lattice points to fill Wannier matrices.
	for y in range(L_y):
		for x in range(L_x):
			i = lattice[y, x]
			if i > -1:
				x_neg, x_pos = (x - 1) % L_x, (x + 1) % L_x
				y_pos = (y + 1) % L_y

				j_x = lattice[y, x_pos]
				x_hop = (pbc or x_pos != 0) and j_x > -1

				j_y = lattice[y_pos, x]
				y_hop = (pbc or y_pos != 0) and j_y > -1

				j_xy1 = lattice[y_pos, x_pos]
				xy1_hop = (pbc or x_pos != 0) and (pbc or y_pos != 0) and j_xy1 > -1

				j_xy2 = lattice[y_pos, x_neg]
				xy2_hop = (pbc or x_neg != L_x - 1) and (pbc or y_pos != 0) and j_xy2 > -1

				# Fill Wannier matrices with hopping terms based on the lattice structure.
				if x_hop:
					Cx[i, j_x] = 1 / 2
					Sx[i, j_x] = 1j / 2

				if y_hop:
					Cy[i, j_y] = 1 / 2
					Sy[i, j_y] = 1j / 2

				if xy1_hop:
					CxSy[i, j_xy1] = 1j / 4
					SxCy[i, j_xy1] = 1j / 4
					CxCy[i, j_xy1] = 1 / 4

				if xy2_hop:
					CxSy[i, j_xy2] = 1j / 4
					SxCy[i, j_xy2] = -1j / 4
					CxCy[i, j_xy2] = 1 / 4

	# Symmetrize Wannier matrices to ensure Hermiticity.
	Cx += Cx.conj().T
	Sx += Sx.conj().T
	Cy += Cy.conj().T
	Sy += Sy.conj().T
	CxSy += CxSy.conj().T
	SxCy += SxCy.conj().T
	CxCy += CxCy.conj().T

	# Convert sparse matrices to dense arrays for further computations.
	Sx, Sy, Cx, Cy, CxSy, SxCy, CxCy = [arr.toarray() for arr in [Sx, Sy, Cx, Cy, CxSy, SxCy, CxCy]]

	return I, Sx, Sy, Cx + Cy, CxSy, SxCy, CxCy


def Hamiltonian(M, B_tilde, wannier_matrices, t1=1., t2=1., B=1.):
	"""
	Constructs the Hamiltonian matrix.

	Parameters:
	M (float): On-site mass.
	B_tilde (float): Diagonal hopping amplitude between same orbitals.
	wannier_matrices (tuple): Tuple containing Wannier matrices.
	t1 (float): Principal hopping amplitude between opposite orbitals.
	t2 (float): Diagonal hopping amplitude between opposite orbitals.
	B (float): Principal hopping amplitude between same orbitals.

	Returns:
	ndarray: Hamiltonian matrix.
	"""
	# Pauli matrices used to construct the Hamiltonian.
	tau1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
	tau2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
	tau3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

	I, Sx, Sy, Cx_plus_Cy, CxSy, SxCy, CxCy = wannier_matrices

	# Construct the components of the Hamiltonian using Wannier matrices and parameters.
	d1 = t1 * Sx + t2 * CxSy
	d2 = t1 * Sy + t2 * SxCy
	d3 = (M - 4 * B - 4 * B_tilde) * I + 2 * B * Cx_plus_Cy + 4 * B_tilde * CxCy

	# Combine components into the full Hamiltonian matrix using Kronecker products with Pauli matrices.
	H = np.kron(d1, tau1) + np.kron(d2, tau2) + np.kron(d3, tau3)

	return H


def H_site_elim(H, fills, holes):
	"""
	Constructs the effective Hamiltonian by eliminating sites corresponding to holes.

	Parameters:
	H (ndarray): Full Hamiltonian matrix.
	fills (ndarray): Indices of filled (non-vacant) sites.
	holes (ndarray): Indices of vacant sites.

	Returns:
	ndarray: Effective Hamiltonian matrix.
	"""
	num_fills = fills.size
	num_holes = holes.size

	num_sites = num_fills + num_holes

	states_per_site = H.shape[0] // num_sites

	fill_idxs = np.empty(states_per_site * num_fills, dtype=int)
	hole_idxs = np.empty(states_per_site * num_holes, dtype=int)

	# Generate index arrays for filled and hole sites, considering internal degrees of freedom.
	for i in range(states_per_site):
		fill_idxs[i::states_per_site] = states_per_site * fills + i
		hole_idxs[i::states_per_site] = states_per_site * holes + i

	# Concatenate index arrays for reordering the Hamiltonian matrix.
	reorder_idxs = np.concatenate((fill_idxs, hole_idxs))

	# Reorder the Hamiltonian matrix to separate filled and vacant site blocks.
	H_reordered = H.copy()[np.ix_(reorder_idxs, reorder_idxs)]

	H_eff_size = states_per_site * num_fills

	# Extract the effective Hamiltonian matrix corresponding to filled sites.
	H_eff = H_reordered[:H_eff_size, :H_eff_size]

	return H_eff


def mat_inv(matrix, hermitian=True, alt=True, overwrite_a=True, tol=1e-10):
	"""
	Computes the inverse of a matrix.

	Parameters:
	matrix (ndarray): Input matrix.
	hermitian (bool): Whether the matrix is Hermitian.
	alt (bool): Whether to use an alternative method for inversion.
	overwrite_a (bool): Whether to overwrite the input matrix.
	tol (float): Tolerance for small eigenvalues.

	Returns:
	ndarray: Inverse of the matrix.
	"""
	if not alt:
		try:
			return np.linalg.inv(matrix)
		except np.linalg.LinAlgError:
			return np.linalg.pinv(matrix, hermitian=hermitian)
	else:
		if hermitian:
			# Use eigenvalue decomposition for Hermitian matrices.
			D, P = eigh(matrix, overwrite_a=overwrite_a)
			# Invert eigenvalues, setting small ones to zero.
			D_inv = np.where(np.abs(D) > tol, 1 / D, 0. + 0.j)
			D_inv = np.diag(D_inv)
			# Reconstruct the inverse matrix from eigenvectors and inverted eigenvalues.
			return np.dot(P, np.dot(D_inv, P.T.conj()))
		else:
			# Use generalized eigenvalue decomposition for non-Hermitian matrices.
			D, P_right, P_left = eig(matrix, left=True, right=True, overwrite_a=overwrite_a)

			zero_value = 0. + 0.j if np.iscomplexobj(D) else 0.
			# Invert eigenvalues, setting small ones to zero.
			D_inv = np.where(np.abs(D) > tol, 1 / D, zero_value)
			D_inv = np.diag(D_inv)

			# Reconstruct the inverse matrix from left and right eigenvectors and inverted eigenvalues.
			return np.dot(P_right, np.dot(D_inv, P_left.conj().T))


def mat_solve_iterative(matrix, tol=1e-5):
	"""
	Solves a linear system iteratively using the Conjugate Gradient method.

	Parameters:
	matrix (ndarray): Coefficient matrix.
	tol (float): Tolerance for convergence.

	Returns:
	function: Function that solves the linear system for a given right-hand side vector.

	This function is used as an alternative method in 'H_renorm' for computing the inverse of H_bb.
	"""

	def solve(b):
		# Solve the linear system using the Conjugate Gradient method.
		x, info = cg(csr_matrix(matrix), b, tol=tol)
		if info != 0:
			raise np.linalg.LinAlgError("Conjugate gradient solver did not converge")
		return x

	return solve


def H_renorm(H, fills, holes):
	"""
	Constructs the renormalized effective Hamiltonian using the Schur complement.

	Parameters:
	H (ndarray): Full Hamiltonian matrix.
	fills (ndarray): Indices of filled (non-vacant) sites.
	holes (ndarray): Indices of vacant sites.

	Returns:
	ndarray: Renormalized effective Hamiltonian matrix.
	"""
	num_fills = fills.size
	num_holes = holes.size

	num_sites = num_fills + num_holes

	states_per_site = H.shape[0] // num_sites

	fill_idxs = np.empty(states_per_site * num_fills, dtype=int)
	hole_idxs = np.empty(states_per_site * num_holes, dtype=int)

	# Generate index arrays for filled and hole sites, considering internal degrees of freedom.
	for i in range(states_per_site):
		fill_idxs[i::states_per_site] = states_per_site * fills + i
		hole_idxs[i::states_per_site] = states_per_site * holes + i

	# Concatenate index arrays for reordering the Hamiltonian matrix.
	reorder_idxs = np.concatenate((fill_idxs, hole_idxs))

	# Reorder the Hamiltonian matrix to separate filled and vacant site blocks.
	H_reordered = H[np.ix_(reorder_idxs, reorder_idxs)]

	H_eff_size = states_per_site * num_fills

	# Extract sub-blocks of the reordered Hamiltonian corresponding to filled and vacant sites.
	H_aa = H_reordered[:H_eff_size, :H_eff_size]
	H_bb = H_reordered[H_eff_size:, H_eff_size:]
	H_ab = H_reordered[:H_eff_size, H_eff_size:]
	H_ba = H_reordered[H_eff_size:, :H_eff_size]

	try:
		# Use iterative solver for inverting H_bb if possible.
		solve_H_bb = mat_solve_iterative(H_bb)
		H_ba_solved = np.hstack([solve_H_bb(H_ba[:, i].ravel()).reshape(-1, 1) for i in range(H_ba.shape[1])])
		H_eff = H_aa - H_ab @ H_ba_solved
	except:
		# Fallback to direct inversion if iterative solver fails.
		H_eff = H_aa - H_ab @ mat_inv(H_bb) @ H_ba

	return H_eff


def H_and_lattice_wrapper(lattice_order, method, M, B_tilde, pbc=True, pad_width=0, n=None, **kwargs):
	"""
	Wrapper function to generate Hamiltonian and lattice using specified method.

	Parameters:
	lattice_order (int): Order of the Sierpinski carpet.
	method (str): Method to use ('symmetry', 'square', 'site_elim', 'renorm').
	M (float): On-site mass.
	B_tilde (float): Diagonal hopping amplitude between same orbitals.
	pbc (bool): Periodic boundary conditions.
	pad_width (int): Padding width for the carpet lattice.
	n (int): Cutoff length for distances (required for 'symmetry' method).

	Returns:
	tuple: A tuple containing:
		- H (ndarray): Hamiltonian matrix.
		- lattice (ndarray): Lattice array.
	"""
	if method not in ['symmetry', 'square', 'site_elim', 'renorm']:
		raise ValueError(f"Invalid method {method}: options are ['symmetry', 'square', 'site_elim', 'renorm'].")
	if method == 'symmetry' and not isinstance(n, int):
		raise ValueError(f"Parameter 'n' must be specified and must be an integer.")

	# Generate the square and fractal lattices for the specified order and padding.
	square_lattice, carpet_lattice, fills, holes = generate_lattices(lattice_order, pad_width)

	if method == 'symmetry':
		# Use the symmetry method to construct Wannier matrices and Hamiltonian.
		wannier_matrices = wannier_matrices_symmetry_method(carpet_lattice, pbc, n)
		H = Hamiltonian(M, B_tilde, wannier_matrices)
		return H, carpet_lattice
	else:
		# Use the Fourier transform method to construct Wannier matrices and Hamiltonian.
		wannier_matrices = wannier_matrices_FT(square_lattice, pbc)
		H = Hamiltonian(M, B_tilde, wannier_matrices)
		if method == 'square':
			return H, square_lattice
		elif method == 'site_elim':
			# Construct effective Hamiltonian by eliminating sites corresponding to holes.
			H_eff = H_site_elim(H, fills, holes)
			return H_eff, carpet_lattice
		else:
			# Construct renormalized effective Hamiltonian using the Schur complement.
			H_eff = H_renorm(H, fills, holes)
			return H_eff, carpet_lattice


def uniform_mass_disorder(disorder_strength, system_size, internal_freedoms, sparse):
	"""
	Generates a uniform mass disorder operator.

	Parameters:
	disorder_strength (float): Disorder strength.
	system_size (int): Size of the system.
	internal_freedoms (int): Internal degrees of freedom per site.
	sparse (bool): Whether to return a sparse matrix.

	Returns:
	ndarray or sparse matrix: Disorder operator.
	"""
	# Generate random disorder values uniformly distributed within the specified range.
	disorder_array = np.random.uniform(-disorder_strength / 2, disorder_strength / 2, size=system_size)
	# Normalize the disorder values to have zero mean.
	delta = np.sum(disorder_array) / system_size
	disorder_array -= delta
	# Repeat disorder values for internal degrees of freedom per site.
	disorder_array = np.repeat(disorder_array, internal_freedoms)
	# Create a diagonal matrix with disorder values.
	disorder_operator = np.diag(disorder_array).astype(np.complex128) if not sparse else diags(disorder_array,
																							   dtype=np.complex128,
																							   format='csr')
	return disorder_operator


def main():
	# Placeholder for main function implementation
	pass


if __name__ == '__main__':
	main()

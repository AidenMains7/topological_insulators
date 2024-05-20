"""
Module for Analyzing Non-Interacting Crystalline Topological Insulators on Fractal Lattices without Disorder

This module provides tools for constructing and analyzing Hamiltonians for non-interacting crystalline topological insulators on 2D fractal lattices, specifically the Sierpinski carpet, without introducing disorder. It allows for the analysis of Hamiltonians with various M and B_tilde combinations to generate phase diagrams.

Functions:
- generate_lattices(order, pad_width): Generates square and fractal lattices (Sierpinski carpet) with specified order and padding.
- geometry_arrays(lattice, pbc, n): Constructs geometry arrays for distances and angles between lattice sites.
- wannier_matrices_symmetry_method(lattice, pbc, n, r0): Constructs Wannier matrices using the symmetry method.
- wannier_matrices_FT(lattice, pbc): Constructs Wannier matrices using Fourier transform method.
- hamiltonian_components(wannier_matrices, t1, t2, B, sparse): Constructs Hamiltonian components without M or B_tilde dependence.
- decompose_matrix(H, fills, holes): Decomposes a Hamiltonian matrix into sub-blocks for filled and vacant sites.
- double_decompose(wannier_matrices, fills, holes): Double-decomposes the Hamiltonian components.
- precompute_data(order, method, pbc, n, pad_width): Precomputes Hamiltonian data for a given lattice order and method.
- mat_inv(matrix, hermitian, alt, overwrite_a, tol): Computes the inverse of a matrix.
- mat_solve_iterative(matrix, tol): Solves a linear system iteratively using the Conjugate Gradient method.
- H_renorm(H_parts): Constructs a renormalized effective Hamiltonian using the Schur complement.
- reconstruct_hamiltonian(method, precomputed_data, M, B_tilde, sparse): Reconstructs the Hamiltonian with specified M and B_tilde values.
- main(): Placeholder for main function implementation.
"""

import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse.linalg import cg
import pickle


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
        if order_ == 0:
            return np.array([[1]], dtype=int)

        # Generate a smaller carpet for the previous order
        smaller_carpet = sierpinski_carpet(order_ - 1)
        size = smaller_carpet.shape[0]

        # Initialize the new carpet with all sites filled
        new_carpet = np.ones((3 * size, 3 * size), dtype=int)
        # Create the central hole in the new carpet
        new_carpet[size:(2 * size), size:(2 * size)] = np.zeros((size, size), dtype=int)

        # Copy the smaller carpet into each of the 8 surrounding blocks of the new carpet
        for i in range(3):
            for j in range(3):
                if i != 1 or j != 1:
                    new_carpet[i * size:(i + 1) * size, j * size:(j + 1) * size] = smaller_carpet

        return new_carpet

    # Generate the Sierpinski carpet of the specified order
    carpet = sierpinski_carpet(order)

    # Add padding around the carpet if specified
    if pad_width > 0:
        carpet = np.pad(carpet, pad_width, mode='constant', constant_values=1)

    # Flatten the carpet lattice to get a 1D array of sites
    flat_carpet = carpet.flatten()
    ones_indices = np.flatnonzero(flat_carpet)
    carpet_lattice = np.full(flat_carpet.shape, -1, dtype=int)
    carpet_lattice[ones_indices] = np.arange(ones_indices.size)
    carpet_lattice = carpet_lattice.reshape(carpet.shape)

    L = carpet.shape[0]
    # Generate a square lattice of the same size
    square_lattice = np.arange(L ** 2).reshape((L, L))

    # Get indices of filled and vacant sites in the flat carpet
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

    if pbc and (n >= side_length // 2):
        raise ValueError("Cutoff length must be less than half the system size when PBC is applied.")

    # Find indices of filled lattice sites
    filled_indices = np.argwhere(lattice >= 0)
    # Compute the difference between all pairs of filled site indices
    difference = filled_indices[None, :, :] - filled_indices[:, None, :]
    dy, dx = difference[..., 0], difference[..., 1]

    if pbc:
        # Apply periodic boundary conditions by wrapping distances that exceed half the lattice size
        dx = np.where(np.abs(dx) > side_length / 2, dx - np.sign(dx) * side_length, dx)
        dy = np.where(np.abs(dy) > side_length / 2, dy - np.sign(dy) * side_length, dy)

    # Mask for distances within the specified cutoff length
    distance_mask = np.maximum(np.abs(dx), np.abs(dy)) <= n

    # Create masks for principal axes and diagonal directions within the cutoff distance
    prncpl_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & distance_mask
    diags_mask = ((np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0))) & distance_mask

    # Combined mask for valid principal and diagonal directions
    prelim_mask = prncpl_mask | diags_mask

    # Compute distances for valid site pairs
    dr = np.where(prelim_mask, np.maximum(np.abs(dx), np.abs(dy)), 0)
    # Compute cosines of angles between valid site pairs
    cos_dphi = np.where(prelim_mask, np.cos(np.arctan2(dy, dx)), 0.)
    # Compute sines of angles between valid site pairs
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
    dr, cos_dphi, sin_dphi, prncpl_mask, diags_mask = geometry_arrays(lattice, pbc, n)
    system_size = np.max(lattice) + 1
    # Identity matrix representing on-site terms
    I = np.eye(system_size, dtype=np.complex128)

    # Exponential decay function for principal and diagonal directions
    F_p = np.where(prncpl_mask, np.exp(1 - dr / r0), 0. + 0.j)
    F_d = np.where(diags_mask, np.exp(1 - dr / r0), 0. + 0.j)

    # Construct Wannier matrices for different terms based on geometry arrays and decay functions
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

    # Initialize Wannier matrices as sparse matrices to save memory and computation time
    I = np.eye(system_size, dtype=np.complex128)
    Cx = dok_matrix((system_size, system_size), dtype=np.complex128)
    Sx = dok_matrix((system_size, system_size), dtype=np.complex128)
    Cy = dok_matrix((system_size, system_size), dtype=np.complex128)
    Sy = dok_matrix((system_size, system_size), dtype=np.complex128)
    CxSy = dok_matrix((system_size, system_size), dtype=np.complex128)
    SxCy = dok_matrix((system_size, system_size), dtype=np.complex128)
    CxCy = dok_matrix((system_size, system_size), dtype=np.complex128)

    for y in range(L_y):
        for x in range(L_x):
            i = lattice[y, x]
            if i > -1:
                # Determine neighboring indices with periodic boundary conditions
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

    # Symmetrize Wannier matrices to ensure Hermiticity
    Cx += Cx.conj().T
    Sx += Sx.conj().T
    Cy += Cy.conj().T
    Sy += Sy.conj().T
    CxSy += CxSy.conj().T
    SxCy += SxCy.conj().T
    CxCy += CxCy.conj().T

    # Convert sparse matrices to dense arrays for further computations
    Sx, Sy, Cx, Cy, CxSy, SxCy, CxCy = [arr.toarray() for arr in [Sx, Sy, Cx, Cy, CxSy, SxCy, CxCy]]

    return I, Sx, Sy, Cx + Cy, CxSy, SxCy, CxCy


def hamiltonian_components(wannier_matrices, t1=1., t2=1., B=1., sparse=True):
    """
    Constructs Hamiltonian components without M or B_tilde dependence.

    Parameters:
    wannier_matrices (tuple): Tuple containing Wannier matrices.
    t1 (float): Principal hopping amplitude between opposite orbitals.
    t2 (float): Diagonal hopping amplitude between opposite orbitals.
    B (float): Principal hopping amplitude between same orbitals.
    sparse (bool): Whether to return sparse matrices.

    Returns:
    tuple: Hamiltonian components H_0, M_hat, B_tilde_hat.
    """
    # Pauli matrices used to construct the Hamiltonian
    tau1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    tau2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    tau3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    I, Sx, Sy, Cx_plus_Cy, CxSy, SxCy, CxCy = wannier_matrices

    # Construct the components of the Hamiltonian using Wannier matrices and parameters
    d1 = t1 * Sx + t2 * CxSy
    d2 = t1 * Sy + t2 * SxCy
    d3 = -4 * B * I + 2 * B * Cx_plus_Cy

    # Hamiltonian components without M or B_tilde dependence
    M_hat = np.kron(I, tau3)
    B_tilde_hat = np.kron(4 * (CxCy - I), tau3)

    H_0 = np.kron(d1, tau1) + np.kron(d2, tau2) + np.kron(d3, tau3)

    if sparse:
        return csr_matrix(H_0), csr_matrix(M_hat), csr_matrix(B_tilde_hat)

    return H_0, M_hat, B_tilde_hat


def decompose_matrix(H, fills, holes):
    """
    Decomposes a Hamiltonian matrix into sub-blocks for filled and vacant sites.

    Parameters:
    H (ndarray or csr_matrix): Hamiltonian matrix.
    fills (ndarray): Indices of filled (non-vacant) sites.
    holes (ndarray): Indices of vacant sites.

    Returns:
    tuple: Sub-blocks H_aa, H_bb, H_ab, H_ba.
    """
    num_fills = fills.size
    num_holes = holes.size

    num_sites = num_fills + num_holes

    states_per_site = H.shape[0] // num_sites

    # Create arrays to hold the indices of filled and vacant sites
    fill_idxs = np.empty(states_per_site * num_fills, dtype=int)
    hole_idxs = np.empty(states_per_site * num_holes, dtype=int)

    # Populate the arrays with the corresponding indices
    for i in range(states_per_site):
        fill_idxs[i::states_per_site] = states_per_site * fills + i
        hole_idxs[i::states_per_site] = states_per_site * holes + i

    # Concatenate the indices to get the reordering indices
    reorder_idxs = np.concatenate((fill_idxs, hole_idxs))

    # Reorder the Hamiltonian matrix based on the reordering indices
    H_reordered = H[np.ix_(reorder_idxs, reorder_idxs)]

    # Determine the size of the effective Hamiltonian matrix
    H_eff_size = states_per_site * num_fills

    # Extract the sub-blocks of the Hamiltonian matrix
    H_aa = H_reordered[:H_eff_size, :H_eff_size]
    H_bb = H_reordered[H_eff_size:, H_eff_size:]
    H_ab = H_reordered[:H_eff_size, H_eff_size:]
    H_ba = H_reordered[H_eff_size:, :H_eff_size]

    return H_aa, H_bb, H_ab, H_ba


def double_decompose(wannier_matrices, fills, holes):
    """
    Double-decomposes the Hamiltonian components.

    Parameters:
    wannier_matrices (tuple): Tuple containing Wannier matrices.
    fills (ndarray): Indices of filled (non-vacant) sites.
    holes (ndarray): Indices of vacant sites.

    Returns:
    tuple: Decomposed Hamiltonian components H_0_parts, M_hat_parts, B_tilde_hat_parts.
    """
    # Construct the Hamiltonian components without M or B_tilde dependence
    H_0, M_hat, B_tilde_hat = hamiltonian_components(wannier_matrices, sparse=False)

    # Decompose each component into sub-blocks for filled and vacant sites
    H_0_parts = decompose_matrix(H_0, fills, holes)
    M_hat_parts = decompose_matrix(M_hat, fills, holes)
    B_tilde_hat_parts = decompose_matrix(B_tilde_hat, fills, holes)

    return H_0_parts, M_hat_parts, B_tilde_hat_parts


def precompute_data(order, method, pbc, n=None, pad_width=0):
    """
    Precomputes Hamiltonian data for a given lattice order and method.

    Parameters:
    order (int): Order of the Sierpinski carpet.
    method (str): Method to use ('symmetry', 'square', 'site_elim', 'renorm').
    pbc (bool): Periodic boundary conditions.
    n (int, optional): Cutoff length for distances (required for 'symmetry' method).
    pad_width (int, optional): Padding width for the carpet lattice.

    Returns:
    tuple: Precomputed Hamiltonian data and lattice.
    """
    # Validate method parameter
    if method not in ['symmetry', 'square', 'site_elim', 'renorm']:
        raise ValueError(f"Invalid method {method}: options are ['symmetry', 'square', 'site_elim', 'renorm'].")
    # Validate 'n' parameter for 'symmetry' method
    if method == 'symmetry' and not isinstance(n, int):
        raise ValueError(f"Parameter 'n' must be specified and must be an integer.")

    # Generate the square and fractal lattices for the specified order and padding
    square_lattice, carpet_lattice, fills, holes = generate_lattices(order, pad_width)

    if method == 'symmetry':
        # Use the symmetry method to construct Wannier matrices and Hamiltonian components
        wannier_matrices = wannier_matrices_symmetry_method(carpet_lattice, pbc, n)
        H_components = hamiltonian_components(wannier_matrices)
        return H_components, carpet_lattice
    else:
        # Use the Fourier transform method to construct Wannier matrices
        wannier_matrices = wannier_matrices_FT(square_lattice, pbc)
        if method == 'square':
            H_components = hamiltonian_components(wannier_matrices)
            return H_components, square_lattice
        else:
            # Double-decompose the Wannier matrices for site elimination or renormalization methods
            parts_groups = double_decompose(wannier_matrices, fills, holes)
            if method == 'site_elim':
                # For site elimination, only keep the non-vacant site sub-blocks
                H_components = []
                for parts_group in parts_groups:
                    H_components.append(csr_matrix(parts_group[0]))
                H_components = tuple(H_components)
                return H_components, carpet_lattice
            else:
                # For renormalization, return the decomposed parts
                return parts_groups, carpet_lattice


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
            # Try to compute the inverse using the standard method
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if the standard method fails
            return np.linalg.pinv(matrix, hermitian=hermitian)
    else:
        if hermitian:
            # Use eigenvalue decomposition for Hermitian matrices
            D, P = eigh(matrix, overwrite_a=overwrite_a)
            # Invert eigenvalues, setting small ones to zero
            D_inv = np.where(np.abs(D) > tol, 1 / D, 0. + 0.j)
            D_inv = np.diag(D_inv)
            # Reconstruct the inverse matrix from eigenvectors and inverted eigenvalues
            return np.dot(P, np.dot(D_inv, P.T.conj()))
        else:
            # Use generalized eigenvalue decomposition for non-Hermitian matrices
            D, P_right, P_left = eig(matrix, left=True, right=True, overwrite_a=overwrite_a)
            zero_value = 0. + 0.j if np.iscomplexobj(D) else 0.
            # Invert eigenvalues, setting small ones to zero
            D_inv = np.where(np.abs(D) > tol, 1 / D, zero_value)
            D_inv = np.diag(D_inv)
            # Reconstruct the inverse matrix from left and right eigenvectors and inverted eigenvalues
            return np.dot(P_right, np.dot(D_inv, P_left.conj().T))


def mat_solve_iterative(matrix, tol=1e-5):
    """
    Solves a linear system iteratively using the Conjugate Gradient method.

    Parameters:
    matrix (csr_matrix): Coefficient matrix.
    tol (float): Tolerance for convergence.

    Returns:
    function: Function that solves the linear system for a given right-hand side vector.
    """
    def solve(b):
        # Solve the linear system using the Conjugate Gradient method
        x, info = cg(csr_matrix(matrix), b, tol=tol)
        if info != 0:
            raise np.linalg.LinAlgError("Conjugate gradient solver did not converge")
        return x

    return solve


def H_renorm(H_parts):
    """
    Constructs the renormalized effective Hamiltonian using the Schur complement.

    Parameters:
    H_parts (tuple): Tuple containing sub-blocks H_aa, H_bb, H_ab, H_ba of the Hamiltonian.

    Returns:
    csr_matrix: Renormalized effective Hamiltonian matrix.
    """
    H_aa, H_bb, H_ab, H_ba = H_parts

    try:
        # Use iterative solver for inverting H_bb if possible
        solve_H_bb = mat_solve_iterative(H_bb)
        # Compute H_bb^{-1} * H_ba using the iterative solver
        H_ba_solved = np.hstack([solve_H_bb(H_ba[:, i].ravel()).reshape(-1, 1) for i in range(H_ba.shape[1])])
        # Compute the renormalized effective Hamiltonian using the Schur complement
        H_eff = H_aa - H_ab @ H_ba_solved
    except:
        # Fallback to direct inversion if iterative solver fails
        H_eff = H_aa - H_ab @ mat_inv(H_bb) @ H_ba

    return H_eff


def reconstruct_hamiltonian(method, precomputed_data, M, B_tilde, sparse=True):
    """
    Reconstructs the Hamiltonian with specified M and B_tilde values.

    Parameters:
    method (str): Method to use ('symmetry', 'square', 'site_elim', 'renorm').
    precomputed_data (tuple): Precomputed Hamiltonian data.
    M (float): On-site mass.
    B_tilde (float): Diagonal hopping amplitude between same orbitals.
    sparse (bool): Whether to return a sparse matrix.

    Returns:
    csr_matrix or ndarray: Reconstructed Hamiltonian matrix.
    """
    if method == 'renorm':
        H_0_parts, M_hat_parts, B_tilde_hat_parts = precomputed_data
        H_parts = []

        # Reconstruct each part of the Hamiltonian
        for i in range(len(H_0_parts)):
            H_part = H_0_parts[i] + M * M_hat_parts[i] + B_tilde * B_tilde_hat_parts[i]
            H_parts.append(H_part)

        H_parts = tuple(H_parts)
        # Renormalize the effective Hamiltonian using the Schur complement
        H = H_renorm(H_parts)
        if sparse:
            H = csr_matrix(H)
    else:
        H_0, M_hat, B_tilde_hat = precomputed_data
        # Construct the full Hamiltonian by adding the M and B_tilde dependent components
        H = H_0 + M * M_hat + B_tilde * B_tilde_hat
        if not sparse:
            H = H.toarray()

    return H


def main():
    # Placeholder for main function implementation
    pass


if __name__ == '__main__':
    main()

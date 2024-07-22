"""
Module for Computing the Bott Index of Topological Insulators on Fractal Lattices

This module provides tools for calculating the Bott index, a topological invariant
used to classify non-interacting crystalline topological insulators on 2D fractal lattices,
specifically the Sierpinski carpet. The Bott index is the real-space equivalent of the
Chern number and is particularly useful in systems with disorder or other sources of
translational symmetry breaking.

Functions:
- _rescaling_factors(H, epsilon, k): Compute rescaling factors for the Hamiltonian.
- _jackson_kernel_coefficients(N): Compute Jackson kernel coefficients for the KPM.
- _projector_moments(E_tilde, N): Compute moments for the Chebyshev expansion of the projector.
- projector_KPM(H, E_F, N): Approximate the projector onto the states below Fermi energy using KPM.
- projector_exact(H, E_F): Compute the exact projector onto the states below Fermi energy.
- _trace_logm_power_series(A, order): Compute the trace of the matrix logarithm using a power series expansion.
- bott_index(P, lattice, order): Compute the Bott index using the projector and lattice.

This module allows for the study of topological properties and disorder effects on fractal lattices,
enabling the classification of systems using the Bott index. The tools provided facilitate the
construction of projectors, the computation of the Bott index, and the application of methods
to handle systems with broken translational symmetry.

The Bott index is computed from the projector onto the valence band states and two unitary operators
(Ux and Uy) that represent momentum translations. The index is found from the trace of the matrix
logarithm of an operator representing a closed plaquette loop in momentum space.

Two methods are provided for finding the projector:
1. Exact diagonalization.
2. Kernel polynomial method (KPM) approximation.

The Bott index can be computed using either the direct matrix logarithm or a power series expansion.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, logm


def _rescaling_factors(H, epsilon=0.01, k=12):
    """
    Compute rescaling factors for the Hamiltonian to map its eigenvalues
    into the range [-1 + epsilon, 1 - epsilon].

    Parameters:
    H (csr_matrix): The Hamiltonian matrix.
    epsilon (float): Small parameter to avoid eigenvalues reaching -1 or 1.
    k (int): Number of eigenvalues to compute for estimation.

    Returns:
    tuple: Rescaling factors 'a' and 'b'.
    """
    # Compute the largest and smallest eigenvalues of H
    eigenvalues = eigsh(H, which='LM', k=k)[0]
    E_min, E_max = np.min(eigenvalues), np.max(eigenvalues)

    # Calculate rescaling factors
    a = (E_max - E_min) / (2 - epsilon)
    b = (E_max + E_min) / 2

    return a, b

def _jackson_kernel_coefficients(N):
    """
    Compute Jackson kernel coefficients for the KPM.

    Parameters:
    N (int): Number of terms in the Chebyshev expansion.

    Returns:
    ndarray: Jackson kernel coefficients.
    """
    n = np.arange(N)
    return (1 / (N + 1)) * ((N - n + 1) * np.cos(np.pi * n / (N + 1)) +
                            np.sin(np.pi * n / (N + 1)) / np.tan(np.pi / (N + 1))).astype(np.complex128)

def _projector_moments(E_tilde, N):
    """
    Compute moments for the Chebyshev expansion of the projector.

    Parameters:
    E_tilde (float): Rescaled Fermi energy.
    N (int): Number of terms in the Chebyshev expansion.

    Returns:
    ndarray: Moments of the projector.
    """
    n = np.arange(1, N).astype(np.complex128)
    moments = np.empty(N, dtype=np.complex128)
    moments[0] = 1 - np.arccos(E_tilde) / np.pi
    moments[1:] = -2 * np.sin(n * np.arccos(E_tilde)) / (n * np.pi)

    return moments

def projector_KPM(H, E_F, N):
    """
    Approximate the projector onto the states below Fermi energy using KPM.

    Parameters:
    H (csr_matrix): The Hamiltonian matrix.
    E_F (float): Fermi energy.
    N (int): Number of moments in the Chebyshev expansion.

    Returns:
    ndarray: Projector matrix.
    """
    # Rescale the Hamiltonian and Fermi energy
    a, b = _rescaling_factors(H)
    H_tilde = (H - b * sparse.eye(H.shape[0], dtype=np.complex128, format='csr')) / a
    E_tilde = (E_F - b) / a

    # Compute Jackson kernel coefficients and projector moments
    jack_coefs = _jackson_kernel_coefficients(N)
    proj_mnts = _projector_moments(E_tilde, N)
    moments = jack_coefs * proj_mnts

    # Initialize the projector matrix
    P = np.zeros(H_tilde.shape, dtype=np.complex128)

    # Initialize the Chebyshev polynomials
    Tn_2 = np.eye(H_tilde.shape[0], dtype=np.complex128)
    Tn_1 = H_tilde.toarray()

    # Compute the projector using the Chebyshev expansion
    P += moments[0] * Tn_2
    P += moments[1] * Tn_1

    for n in range(2, N):
        Tn = 2 * H_tilde.dot(Tn_1) - Tn_2
        P += moments[n] * Tn
        Tn_2, Tn_1 = Tn_1, Tn

    return P

def projector_exact(H, E_F):
    """
    Compute the exact projector onto the states below Fermi energy.

    Parameters:
    H (ndarray): The Hamiltonian matrix.
    E_F (float): Fermi energy.

    Returns:
    ndarray: Projector matrix.
    """
    # Compute eigenvalues and eigenvectors of H
    eigenvalues, eigenvectors = eigh(H, overwrite_a=True)

    # Construct the diagonal matrix D with 1 below Fermi energy and 0 above
    D = np.where(eigenvalues < E_F, 1. + 0.j, 0. + 0.j)
    D_mult_eigenvectors_dagger = np.einsum('i,ij->ij', D, eigenvectors.conj().T)

    # Construct the projector matrix
    P = eigenvectors @ D_mult_eigenvectors_dagger

    return P

def _trace_logm_power_series(A, order):
    """
    Compute the trace of the matrix logarithm using a power series expansion.

    Parameters:
    A (ndarray): Matrix to compute the logarithm of.
    order (int): Order of the power series expansion.

    Returns:
    complex: Trace of the matrix logarithm.
    """
    # Initialize the power series expansion
    A_minus_I = A - np.eye(A.shape[0], dtype=np.complex128)
    current_pow = A - np.eye(A.shape[0], dtype=np.complex128)
    trace_logm_A = 0. + 0.j

    # Compute the power series expansion term by term
    for n in range(1, order):
        trace_logm_A += (((-1)**(n-1)) / n) * np.trace(current_pow)
        current_pow = A_minus_I.dot(current_pow)
    trace_logm_A += (((-1)**(order-1)) / order) * np.trace(current_pow)

    return trace_logm_A

def bott_index(P, lattice, order=None):
    """
    Compute the Bott index, a topological invariant, using the projector and lattice.

    Parameters:
    P (ndarray): Projector matrix onto the valence band states.
    lattice (ndarray): Lattice array indicating site positions.
    order (int, optional): Order of the power series expansion for logm.

    Returns:
    int: Bott index.
    """
    # Extract the X and Y coordinates of the lattice sites
    Y, X = np.where(lattice >= 0)[:]
    states_per_site = P.shape[0] // (np.max(lattice) + 1)
    X = np.repeat(X, states_per_site)
    Y = np.repeat(Y, states_per_site)
    ly, lx = lattice.shape

    # Compute the Ux and Uy unitary operators
    Ux = np.exp(1j * 2 * np.pi * X / lx)
    Uy = np.exp(1j * 2 * np.pi * Y / ly)

    # Apply the unitary operators to the projector
    UxP = np.einsum('i,ij->ij', Ux, P)
    UyP = np.einsum('i,ij->ij', Uy, P)
    Ux_daggerP = np.einsum('i,ij->ij', Ux.conj(), P)
    Uy_daggerP = np.einsum('i,ij->ij', Uy.conj(), P)

    # Construct the matrix A used to compute the Bott index
    A = np.eye(P.shape[0], dtype=np.complex128) - P + P.dot(UxP).dot(UyP).dot(Ux_daggerP).dot(Uy_daggerP)

    # Compute the Bott index using either the power series expansion or the direct matrix logarithm
    if order is not None:
        bott = round(np.imag(_trace_logm_power_series(A, order)) / (2 * np.pi))
    else:
        bott = round(np.imag(np.trace(logm(A))) / (2 * np.pi))

    return bott

def main():
    # Placeholder for main function implementation
    pass

if __name__ == '__main__':
    main()

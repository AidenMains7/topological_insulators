"""
Influenced by and taken from Dan Salib's code
"""

import sys
sys.path.append(".")

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, diags, dok_matrix
from scipy.linalg import eigh, logm, eig, eigvals
from scipy.sparse.linalg import cg, eigsh
from numba import jit

def sierpinski_lattice(order:int, pad_width:int) -> tuple:
    """
    Generates a Sierpinski carpet lattice of specified order.

    Parameters:
    order (int): order of the fractal
    pad_width (int): width of padding

    Returns: 
    square_lat (ndarray): square lattice of the same size
    fractal_lat (ndarray): the fractal lattice
    holes (ndarray): indices of the empty sites
    filled (ndarray): indices of the filled sites
    """

    #check that order is proper
    if (order < 0):
        raise ValueError("Order of lattice must be >= 0.")

    def _sierpinski_carpet(order_:int):
        """
        Generates a Sierpinski carpet fractal of degree order_ using recursion.
        """
        if(order_ == 0):
            return np.array([1], dtype=int)
    
        #carpet of one lower degree; recursion
        carpet_lower = _sierpinski_carpet(order_-1)

        #concatenate to make current degree
        top = np.hstack((carpet_lower,carpet_lower,carpet_lower))
        mid = np.hstack((carpet_lower,carpet_lower*0,carpet_lower))
        carpet = np.vstack((top,mid,top))

        return carpet
    
    #side length
    L = 3**order

    #square lattice
    square_lat = np.arange(L*L).reshape((L,L))
    carpet = _sierpinski_carpet(order)

    #pad width
    if (pad_width > 0):
        carpet = np.pad(carpet,pad_width,mode='constant',constant_values=1)

    #get indices of empty and filled sites 
    flat = carpet.flatten()
    holes = np.where(flat==0)[0]
    filled = np.flatnonzero(flat)

    #construct fractal lattice
    fractal_lat = np.full(flat.shape, -1, dtype=int)
    fractal_lat[filled] = np.arange(filled.size)
    fractal_lat = fractal_lat.reshape(carpet.shape)

    return square_lat, fractal_lat, holes, filled


def geometry(lattice:np.ndarray, pbc:bool, n:int) -> tuple:
    """
    Finds the distance between sites, the angles, and principal and diagonal masks.

    Parameters:

    Returns: 

    """
    side_length = lattice.shape[0]

    if (pbc and n >= side_length//2):
        raise ValueError("With periodic boundary conditions, n must be less than half of the system size.")
    
    filled = np.argwhere(lattice >= 0)


    diff = filled[None, :, :] - filled[:, None, :]
    dy, dx = diff[...,0], diff[...,1]

    if pbc:
        dx = np.where(np.abs(dx) > side_length / 2, dx - np.sign(dx) * side_length, dx)
        dy = np.where(np.abs(dy) > side_length / 2, dy - np.sign(dy) * side_length, dy)
    
    mask_dist = np.maximum(np.abs(dx), np.abs(dy)) <= n

    mask_principal = (((dx == 0) & (dy != 0))    | ((dx != 0) & (dy == 0))) & mask_dist
    mask_diagonal  = ((np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0))) & mask_dist

    mask_both = mask_principal | mask_diagonal

    d_r   = np.where(mask_both, np.maximum(np.abs(dx), np.abs(dy)), 0)
    d_cos = np.where(mask_both, np.cos(np.arctan2(dy, dx)),         0.)
    d_sin = np.where(mask_both, np.sin(np.arctan2(dy, dx)),         0.)

    return d_r, d_cos, d_sin, mask_principal, mask_diagonal


def wannier_symmetry(lattice:np.ndarray, pbc:bool, n:int, r0:float=1.) -> tuple:

    d_r, d_cos, d_sin, mask_principal, mask_diagonal = geometry(lattice, pbc, n)

    num_sites = np.max(lattice) + 1

    I = np.eye(num_sites, dtype=np.complex128)

    # Exponential decay function for principal and diagonal directions
    F_p = np.where(mask_principal, np.exp(1 - d_r / r0), 0. + 0.j)
    F_d = np.where(mask_diagonal,  np.exp(1 - d_r / r0), 0. + 0.j)

    # Construct Wannier matrices for different terms based on geometry arrays and decay functions
    Sx = 1j * d_cos * F_p / 2
    Sy = 1j * d_sin * F_p / 2
    Cx_plus_Cy = F_p / 2

    CxSy = 1j * d_sin * F_d / (2 * np.sqrt(2))
    SxCy = 1j * d_cos * F_d / (2 * np.sqrt(2))
    CxCy = F_d / 4

    return I, Sx, Sy, Cx_plus_Cy, CxSy, SxCy, CxCy


def wannier_fourier(lattice:np.ndarray, pbc:bool) -> tuple:
    """
    Constructs the wannier matrices using the fourier transform method.
    """
    num_sites = np.max(lattice) + 1
    L_y, L_x = lattice.shape

    I = np.eye(num_sites, dtype=np.complex128)
    Cx = dok_matrix((num_sites, num_sites), dtype=np.complex128)
    Sx = dok_matrix((num_sites, num_sites), dtype=np.complex128)
    Cy = dok_matrix((num_sites, num_sites), dtype=np.complex128)
    Sy = dok_matrix((num_sites, num_sites), dtype=np.complex128)
    CxSy = dok_matrix((num_sites, num_sites), dtype=np.complex128)
    SxCy = dok_matrix((num_sites, num_sites), dtype=np.complex128)
    CxCy = dok_matrix((num_sites, num_sites), dtype=np.complex128)

    for y in range(L_y):
        for x in range(L_x):
            pos = lattice[y, x]
            if pos > -1:
                #if not a hole

                x_neg, x_pos = (x-1)%L_x, (x+1)%L_x
                y_pos = (y+1)%L_y

                pos_xp = lattice[y, x_pos]
                x_hop = (pbc or x_pos != 0) and pos_xp > -1
                pos_yp = lattice[y_pos, x]
                y_hop = (pbc or y_pos != 0) and pos_yp > -1

                pos_ypxp = lattice[y_pos, x_pos]
                ypxp_hop = (pbc or x_pos != 0) and (pbc or y_pos != 0) and pos_ypxp > -1

                pos_ypxn = lattice[y_pos, x_neg]
                ypxn_hop = (pbc or x_neg != L_x - 1) and (pbc or y_pos != 0) and pos_ypxn > -1

                if x_hop:
                    Cx[pos, pos_xp] = 1 / 2
                    Sx[pos, pos_xp] = 1j / 2
                if y_hop:
                    Cy[pos, pos_yp] = 1 / 2
                    Sy[pos, pos_yp] = 1j / 2
                if ypxp_hop:
                    CxSy[pos, pos_ypxp] = 1j / 4
                    SxCy[pos, pos_ypxp] = 1j / 4
                    CxCy[pos, pos_ypxp] = 1 / 4
                if ypxn_hop:
                    CxSy[pos, pos_ypxn] = 1j / 4
                    SxCy[pos, pos_ypxn] = -1j / 4
                    CxCy[pos, pos_ypxn] = 1 / 4

    Cx += Cx.conj().T
    Sx += Sx.conj().T
    Cy += Cy.conj().T
    Sy += Sy.conj().T
    CxSy += CxSy.conj().T
    SxCy += SxCy.conj().T
    CxCy += CxCy.conj().T

    Sx, Sy, Cx, Cy, CxSy, SxCy, CxCy = [arr.toarray() for arr in [Sx, Sy, Cx, Cy, CxSy, SxCy, CxCy]]

    return I, Sx, Sy, Cx + Cy, CxSy, SxCy, CxCy


def Hamiltonian_components(wannier:tuple, t1:float=1., t2:float=1., B:float=1., sparse:bool=True) -> tuple:
    """
    Constructs the components of the  Hamiltonian without dependencey on M or B_tilde

    Parameters:
    wannier (tuple): The wannier matrices given by wannier_symmetry or wannier_fourier
    t1 (float): Amplitude of principal hop between opposite orbitals
    t2 (float): Amplitude of diagonal hop between opposite orbitals
    B (float):  Amplitude of principal hop between same orbitals
    sparse (bool): Whether to return as sparse

    Returns: 
    H_0 (ndarray): Components of the Hamiltonian 
    M_hat (ndarray): 
    B_tilde_hat (ndarray): 
    """
    #Pauli matrices
    pauli1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    I, Sx, Sy, Cx_plus_Cy, CxSy, SxCy, CxCy = wannier

    #components of the hamilltonian
    d1 = t1 * Sx + t2 * CxSy
    d2 = t1 * Sy + t2 * SxCy
    d3 = -4 * B * I + 2 * B * Cx_plus_Cy

    #not dependent on values of M or B_tilde
    M_hat = np.kron(I, pauli3)
    B_tilde_hat = np.kron(4 * (CxCy - I), pauli3)

    #Hamiltonian
    H_0 = np.kron(d1, pauli1) + np.kron(d2, pauli2) + np.kron(d3, pauli3)

    if sparse:
        return csr_matrix(H_0), csr_matrix(M_hat), csr_matrix(B_tilde_hat)

    return H_0, M_hat, B_tilde_hat


def decompose(H:np.ndarray, fills:np.ndarray, holes:np.ndarray) -> tuple:
    """
    Decomposes a Hamiltonian matrix into sub-blocks dependent on filled and not filled sites

    Parameters:
    H (ndarray): Hamiltonian
    fills (ndarray): Indices of filled sites.
    holes (ndarray): Indices of not filled sites.

    Returns:
    H_aa (ndarray): Sub-blocks of the Hamiltonian
    H_bb (ndarray):
    H_ab (ndarray):
    H_ba (ndarray):
    """
    num_fills = fills.size
    num_holes = holes.size

    num_sites = num_fills + num_holes

    states_per_site = H.shape[0] // num_sites

    fill_idx = np.empty(states_per_site * num_fills, dtype=int)
    hole_idx = np.empty(states_per_site * num_holes, dtype=int)

    for i in range(states_per_site):
        fill_idx[i::states_per_site] = states_per_site * fills + i
        hole_idx[i::states_per_site] = states_per_site * holes + i

    reorder_idxs = np.concatenate((fill_idx, hole_idx))
    H_reordered = H[np.ix_(reorder_idxs, reorder_idxs)]
    H_eff_size = states_per_site * num_fills

    H_aa = H_reordered[:H_eff_size, :H_eff_size] #filled to filled
    H_bb = H_reordered[H_eff_size:, H_eff_size:] #holes to holes
    H_ab = H_reordered[:H_eff_size, H_eff_size:] #filled to holes
    H_ba = H_reordered[H_eff_size:, :H_eff_size] #holes to filled

    return H_aa, H_bb, H_ab, H_ba


def decompose_parts(wannier:tuple, holes:np.ndarray, fills:np.ndarray) -> tuple:
    """
    Decompose the components of the Hamiltonian

    Parameters:
    wannier
    holes
    fills

    Returns: 
    H_0_parts
    M_hat_parts
    B_tilde_parts
    """

    H_0, M_hat, B_tilde_hat = Hamiltonian_components(wannier, sparse=False)
    H_0_parts         = decompose(H_0,         fills, holes)
    M_hat_parts       = decompose(M_hat,       fills, holes)
    B_tilde_hat_parts = decompose(B_tilde_hat, fills, holes)

    return H_0_parts, M_hat_parts, B_tilde_hat_parts


def mat_inv(matrix:np.ndarray, hermitian:bool=True, alt:bool=True, overwrite_a:bool=True, tol:float=1e-10) -> np.ndarray:

    if not alt:
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix, hermitian=hermitian)
    else:
        if hermitian:
            D, P = eigh(matrix, overwrite_a=overwrite_a)
            D_inv = np.where(np.abs(D) > tol, 1 / D, 0. + 0.j)
            D_inv = np.diag(D_inv)


            return np.dot(P, np.dot(D_inv, P.T.conj()))
        else:
            D, P_right, P_left = eig(matrix, left=True, right=True, overwrite_a=overwrite_a)
            zero_value = 0. + 0.j if np.iscomplexobj(D) else 0.
            D_inv = np.where(np.abs(D) > tol, 1 / D, zero_value)
            D_inv = np.diag(D_inv)
            return np.dot(P_right, np.dot(D_inv, P_left.conj().T))


def mat_solve_iterative(matrix:np.ndarray, tol:float=1e-5):
    def solve(b):
        x, info = cg(csr_matrix(matrix), b, tol=tol)
        if info != 0:
            raise np.linalg.LinAlgError("Conjugate gradient solver did not converge")
        return x
    return solve


def H_renorm(H_parts:tuple) -> np.ndarray:
    H_aa, H_bb, H_ab, H_ba = H_parts

    try:
        solve_H_bb = mat_solve_iterative(H_bb)
        H_ba_solved = np.hstack([solve_H_bb(H_ba[:, i].ravel()).reshape(-1, 1) for i in range(H_ba.shape[1])])
        H_eff = H_aa - H_ab @ H_ba_solved
    except:
        H_eff = H_aa - H_ab @ mat_inv(H_bb) @ H_ba

    return H_eff


def precompute(method:str, order:int, pad_width:int, pbc:bool, n:int) -> tuple:
    """
    Precomputes the lattice and the parts of its Hamiltonian
    """

    if method not in ['symmetry', 'square', 'renorm', 'site_elim']:
        raise ValueError("Method must be one of ['symmetry', 'square', 'renorm', 'site_elim']")
    if method == 'symmetry' and not isinstance(n, int):
        raise ValueError(f"When using the method of symmetry, n must be defined. It is currently {n}, of type {type(n)}.")
    
    sq_lat, frac_lat, holes, fills = sierpinski_lattice(order, pad_width)


    if method == "symmetry":
        wannier = wannier_symmetry(frac_lat, pbc=pbc, n=n)
        H_components = Hamiltonian_components(wannier=wannier)
        return H_components, frac_lat
    elif method == "square":
        wannier = wannier_fourier(sq_lat, pbc=pbc)
        H_components = Hamiltonian_components(wannier=wannier)
        return H_components, sq_lat
    elif method == "site_elim":
        wannier = wannier_fourier(sq_lat, pbc=pbc)
        parts_groups = decompose_parts(wannier, holes, fills)
        H_components = []
        for parts_group in parts_groups:
            H_components.append(csr_matrix(parts_group[0]))
        H_components = tuple(H_components)
        return H_components, frac_lat
    elif method == "renorm":
        wannier = wannier_fourier(sq_lat, pbc=pbc)
        parts_groups = decompose_parts(wannier, holes, fills)
        return parts_groups, frac_lat


def Hamiltonian_reconstruct(method:str, precomputed_data:tuple, M:float, B_tilde:float, sparse:bool=True) -> np.ndarray:
    """
    
    """

    if method == "renorm":
        H_0_parts, M_hat_parts, B_tilde_hat_parts = precomputed_data
        H_parts = []

        for i in range(len(H_0_parts)):
            H_part = H_0_parts[i] + M * M_hat_parts[i] + B_tilde * B_tilde_hat_parts[i]
            H_parts.append(H_part)

        H_parts = tuple(H_parts)
        H = H_renorm(H_parts)
        if sparse:
            H = csr_matrix(H)

    else:
        H_0, M_hat, B_tilde_hat = precomputed_data
        H = H_0 + M*M_hat + B_tilde*B_tilde_hat
        if not sparse:
            H = H.toarray()
    
    return H


def mass_disorder(strength:float, system_size:int, df:int, sparse:bool, type:str='uniform') -> np.ndarray:
    '''
    Generates a disorder operator of random values for a given strength.

    Parameters:
    type (str): type of random distribution to sample from
    strength (float): strength of disorder
    system_size (int): number of sites
    df (int): degrees of internal freedom
    sparse (bool): generate as sparse matrix?

    Returns: 
    disorder_operator (ndarray): Diagonal operator to give disorder

    '''

    if type not in ['uniform', 'normal']:
        raise ValueError("type must be one of: ['uniform', 'normal']")

    if type == 'uniform':
        disorder_array = np.random.uniform(-strength/2, strength/2, size=system_size)
        #ensure mean = 0
        delta = np.sum(disorder_array) / system_size
        disorder_array -= delta

    elif type == 'normal':
        disorder_array = np.random.normal(loc=0, scale=strength/2, size=system_size)
        #mean is already 0

    #repeat for all freedoms
    disorder_array = np.repeat(disorder_array, df)

    #diagonal matrix operator
    disorder_operator = np.diag(disorder_array).astype(np.complex128) if not sparse else diags(disorder_array, dtype=np.complex128, format='csr')
    return disorder_operator


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

    #projector given by matrix multiplaction of eigenvectors and D_dagger
    P = eigvecs @ D_dagger

    return P


def _rescaling_factors(H:np.ndarray, epsilon:float=0.01, k:int=12) -> "tuple[float, float]":

    eigenvals = eigsh(H, which='LM', k=k)[0]
    E_min, E_max = np.min(eigenvals), np.max(eigenvals)

    a = (E_max - E_min)/(2-epsilon)
    b = (E_max + E_min)/2

    return a, b


def _jackson_kernel_coefficients(N:int) -> np.ndarray:
    n = np.arange(N)

    return (1 / (N + 1)) * ((N - n + 1) * np.cos(np.pi * n / (N + 1)) +
                            np.sin(np.pi * n / (N + 1)) / np.tan(np.pi / (N + 1))).astype(np.complex128)


def _projector_moments(E_tilde:float, N:int) -> np.ndarray:

    n = np.arange(1, N).astype(np.complex128)
    moments = np.empty(N, dtype=np.complex128)
    moments[0] = 1 - np.arccos(E_tilde)/np.pi
    moments[1:] = -2*np.sin(n*np.arccos(E_tilde))/(n*np.pi)

    return moments


def projector_KPM(H:np.ndarray, E_F:float, N:int) -> np.ndarray:

    a, b = _rescaling_factors(H)
    H_tilde = (H - b*sparse.eye(H.shape[0], dtype=np.complex128, format='csr'))/a
    E_tilde = (E_F-b)/a

    jack_coefs = _jackson_kernel_coefficients(N)
    proj_moments = _projector_moments(E_tilde, N)
    moments = jack_coefs * proj_moments

    P = np.zeros(H_tilde.shape, dtype=np.complex128)

    Tn_2 = np.eye(H_tilde.shape[0], dtype=np.complex128)
    Tn_1 = H_tilde.toarray()

    P += moments[0]*Tn_2
    P += moments[1]*Tn_1

    for n in range(2, N):
        Tn = 2*H_tilde.dot(Tn_1)-Tn_2
        P += moments[n] * Tn
        Tn_2, Tn_1 = Tn_1, Tn

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

    #
    Ux = np.exp(1j*2*np.pi*X/Lx)
    Uy = np.exp(1j*2*np.pi*Y/Ly)

    UxP = np.einsum('i,ij->ij', Ux, P)
    UyP = np.einsum('i,ij->ij', Uy, P)
    Ux_daggerP = np.einsum('i,ij->ij', Ux.conj(), P)
    Uy_daggerP = np.einsum('i,ij->ij', Uy.conj(), P)

    A = np.eye(P.shape[0], dtype=np.complex128) - P + P.dot(UxP).dot(UyP).dot(Ux_daggerP).dot(Uy_daggerP)
   
    #Tr(logm(A)) = sum of log of eigvals of A
    #bott = round(np.imag(np.sum(np.log(eigvals(A)))) / (2 * np.pi))
    bott = round(np.imag(np.trace(logm(A))) / (2 * np.pi))

    return bott


#-------main function implementation-----------------
def main():
    pass

if __name__ == "__main__":
    main()


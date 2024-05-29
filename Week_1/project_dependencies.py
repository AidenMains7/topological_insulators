"""

"""
import sys
sys.path.append(".")
from ProjectCode.PhaseDiagram import PhaseDiagramDependencies as pdd

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.linalg import eigh, logm


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

def Hamiltonian_components(wannier:tuple, t1:float=1., t2:float=1., B:float=1., sparse:bool=True) -> tuple:
    """
    Constructs the components of the Hamiltonian without dependencey on M or B_tilde

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

def decompose(H:np.ndarray, holes:np.ndarray, fills:np.ndarray) -> tuple:
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

def precompute(order:int, pad_width:int, pbc:bool, n:int, sparse:bool=True) -> tuple:
    """
    Precomputes the lattice and the parts of its Hamiltonian
    """

    #currently only using symmetry so no method will be included

    sq_lat, frac_lat, holes, fills = sierpinski_lattice(order, pad_width)

    wannier = wannier_symmetry(frac_lat, pbc, n)
    H_parts = Hamiltonian_components(wannier, sparse=sparse)
    return H_parts, frac_lat

def Hamiltonian_reconstruct(precomputed_data:tuple, M:float, B_tilde:float, sparse:bool=True) -> np.ndarray:
    """
    
    """

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

def projector(H:np.ndarray, E_F:float) -> np.ndarray:
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
   
    bott = round(np.imag(np.trace(logm(A))) / (2 * np.pi))

    return bott


def main():
    pass

if __name__ == "__main__":
    main()


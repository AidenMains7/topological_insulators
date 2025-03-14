import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh, eigh, eigvals
from joblib import Parallel, delayed
from itertools import product
import sys
sys.path.append(".")
from Carpet.plotting import reshape_imshow_data, plot_imshow



def honeycomb_pos(side_length:int) -> np.ndarray:
    """
    side_length (int): Number of hexagon tiles on each side
    
    """
    angles = np.array([2*np.pi*(i+1)/6 for i in range(6)])
    hexagon = np.array([[(np.cos(a)) for a in angles], 
                        [(np.sin(a)) for a in angles]]) 

    def _row_of_n_hexagons(n:int) -> np.ndarray:
        hex_row = hexagon
        amount_to_add = n-1
        for i in range(amount_to_add):
            hex_row = np.append(hex_row, hexagon+np.array([[3*(i+1)], [0]]), axis=1)
        hex_row[0] -= np.mean(hex_row[0])
        hex_row[1] -= np.mean(hex_row[1])
        return hex_row
                
    hexagon_lattice = np.empty((2, 1))
    counter = 0
    for num_hexagons in range(side_length, 2*side_length, 1):

        if counter == side_length-1:
            hexagon_lattice = np.append(hexagon_lattice, _row_of_n_hexagons(num_hexagons)+np.array([[0], [counter*np.sqrt(3)*3/2]]), axis=1)
        else:
            hexagon_lattice = np.append(hexagon_lattice, _row_of_n_hexagons(num_hexagons)+np.array([[0], [counter*np.sqrt(3)*3/2]]), axis=1)
        counter += 1

    hexagon_lattice = np.unique(hexagon_lattice[:, 1:], axis=1)
    hexagon_lattice = np.append(hexagon_lattice, np.array([[0], [2*np.max(hexagon_lattice[1])-np.sqrt(3)]])-hexagon_lattice, axis=1)
    hexagon_lattice[0] -= np.mean(hexagon_lattice[0])
    hexagon_lattice[1] -= np.mean(hexagon_lattice[1])
    return np.unique(hexagon_lattice, axis=1)


def hexaflake_pos(generation:int) -> np.ndarray:
    def fractal_iteration(_gen):
        if _gen == 0:
            angles = np.array([2*np.pi*(i+1)/6 for i in range(6)])
            return np.array([[(np.cos(a)) for a in angles], 
                             [(np.sin(a)) for a in angles]])  
        else:
            smaller = fractal_iteration(_gen-1)
            r = 3**_gen
            points = r*np.append(fractal_iteration(0), np.array([[0], [0]]), axis=1)

            new = np.empty((2, 1))
            for i in range(7):
                new = np.append(new, points[:, i].reshape(2, 1)+smaller, axis=1)
            
            new = new[:, 1:]
            return np.unique(new, axis=1)
    return fractal_iteration(generation)


def position_mask(positions):
    """
    positions (ndarray): A 2xN array such that the first row is x positions and the second is y positions.
    """
    # positions[0] == x; positions[1] == y
    # Scaling values to be at integer points and entirely nonzero
    positions[0] *= 2.0
    positions[1] *= 2/np.sqrt(3)
    positions[0] -= np.min(positions[0])
    positions[1] -= np.min(positions[1])
    x, y = positions[0], positions[1]

    positions = np.round(positions, 0).astype(int)
    positions = np.unique(positions, axis=1)

    # Sort positions by y, then x
    idxs = np.lexsort((positions[0, :], positions[1, :]))
    positions = positions[:, idxs]

    lattice = np.ones((np.max(positions[1]).astype(int)+1, np.max(positions[0]).astype(int)+1), dtype=int)*(-1)
    lattice[positions[1], positions[0]] = np.arange(positions.shape[1])

    mask = lattice >= 0
    return mask, x.astype(int), y.astype(int)


def hopping(coords, pbc):
    x, y = coords[0], coords[1]

    if pbc:
        #handle this-----------------------------
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
    
    else:
        delta_x = x[:, np.newaxis] - x
        delta_y = y[:, np.newaxis] - y

    return delta_x.astype(int), delta_y.astype(int)


def geometry(generation, pbc):
    hf = hexaflake_pos(generation)
    hc = honeycomb_pos(int((3**generation+1)/2))
    
    hf_mask, hf_x, hf_y = position_mask(hf)
    hc_mask, hc_x, hc_y = position_mask(hc)

    vacancies = hc_mask & (~hf_mask)
    Ly, Lx = hc_mask.shape

    # Create a sublattice for A and B sites
    subl_A_mask = np.full(hc_mask.shape, False, dtype=bool)

    subl_B_mask = hc_mask & (~subl_A_mask)

    coords = np.concatenate((hc_x[np.newaxis, :],hc_y[np.newaxis, :]),axis=0)
    delta_x, delta_y = hopping(coords, pbc)

    sublat_A = subl_A_mask[hc_y, hc_x]
    sublat_B = subl_B_mask[hc_y, hc_x]
    hexaflake = hf_mask[hc_y, hc_x]
    vacancies = vacancies[hc_y, hc_x]

    idxs = np.arange(hc_mask.sum())
    reordered_idxs = np.concatenate((idxs[hexaflake], idxs[vacancies]))
    coords, sublat_A, sublat_B, hexaflake, vacancies = [arr[reordered_idxs] for arr in [coords.T, sublat_A, sublat_B, hexaflake, vacancies]]
    delta_x = delta_x[np.ix_(reordered_idxs, reordered_idxs)]
    delta_y = delta_y[np.ix_(reordered_idxs, reordered_idxs)]

    #-------------------
    A_A = sublat_A[None, :] & sublat_A[:, None]
    B_B = sublat_B[None, :] & sublat_B[:, None]

    W1 = ((delta_x == 0) & (delta_y < 0)) | ((delta_x > 0) & (delta_y > 0)) | ((delta_x < 0) & (delta_y > 0))
    W2 = ((delta_x == 0) & (delta_y > 0)) | ((delta_x < 0) & (delta_y < 0)) | ((delta_x > 0) & (delta_y < 0))

    CCW = (A_A & W1) | (B_B & W2)
    CW  = (A_A & W2) | (B_B & W1)

    NN = ((np.abs(delta_x) == 2) & (delta_y == 0)) | ((np.abs(delta_x) == 1) & (np.abs(delta_y) == 1))
    NNN = ((delta_x == 0) & (np.abs(delta_y) == 2)) | ((np.abs(delta_x) == 3) & (np.abs(delta_y) == 1))

    NNN_CCW = NNN & CCW
    NNN_CW = NNN & CW

    return {'coords':coords,'hexaflake':hexaflake,'sublat_A':sublat_A, 'sublat_B':sublat_B,'vacancies':vacancies,'NN':NN,'NNN_CCW':NNN_CCW,'NNN_CW':NNN_CW}


def haldane(data, M, t1, t2, phi):
    sublat_A = data['sublat_A']
    sublat_B=data['sublat_B']
    NN = data['NN']
    NNN_CCW = data['NNN_CCW']
    NNN_CW = data['NNN_CW']

    H = np.zeros(NN.shape, dtype=np.complex128)

    H_diag = M*sublat_A.astype(np.complex128) - M*sublat_B.astype(np.complex128)
    np.fill_diagonal(H, H_diag)

    H[NN] = -t1
    H[NNN_CCW] = -t2*np.exp(-1j*phi)
    H[NNN_CW] = -t2*np.exp(1j*phi)
    return H


def fractal_hamiltonian(method, gen, pbc, M, t1=1.0, t2=1.0, phi=np.pi/2):
    d = geometry(gen, pbc)
    coords = d['coords']
    hf = d['hexaflake'] 
    vac = d['vacancies']

    H = haldane(d, M, t1, t2, phi)

    match method:
        case 'renorm':
            H_ff = H[np.ix_(hf, hf)]
            H_fv = H[np.ix_(hf, vac)]
            H_vf = H[np.ix_(vac, hf)]
            H_vv = H[np.ix_(vac, vac)]

            H = H_ff - H_fv @ H_vv**(-1) @ H_vf
            coords = coords[hf]
        case 'site_elim':
            H = H[np.ix_(hf, hf)]
            coords = coords[hf]
        case 'honeycomb':
            pass
    
    return H, coords


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

    #
    Ux = np.exp(1j*2*np.pi*X/Lx)
    Uy = np.exp(1j*2*np.pi*Y/Ly)

    UxP = np.einsum('i,ij->ij', Ux, P)
    UyP = np.einsum('i,ij->ij', Uy, P)
    Ux_daggerP = np.einsum('i,ij->ij', Ux.conj(), P)
    Uy_daggerP = np.einsum('i,ij->ij', Uy.conj(), P)

    A = np.eye(P.shape[0], dtype=np.complex128) - P + P.dot(UxP).dot(UyP).dot(Ux_daggerP).dot(Uy_daggerP)
   
    #Tr(logm(A)) = sum of log of eigvals of A
    bott = round(np.imag(np.sum(np.log(eigvals(A)))) / (2 * np.pi))
    
    # Old, slower method.
    #bott = round(np.imag(np.trace(logm(A))) / (2 * np.pi))

    return bott


def lattice_from_coords(coords):
    idxs = np.lexsort((coords[0, :], coords[1, :]))
    coords = coords[:, idxs]

    lattice = np.ones((np.max(coords[1]).astype(int)+1, np.max(coords[0]).astype(int)+1), dtype=int)*(-1)
    lattice[coords[1], coords[0]] = np.arange(coords.shape[1])
    return lattice




#_______________________________________________
def main():
    M_vals = np.linspace(-2, 12, 1)
    vals = tuple(M_vals)
    def single(i):
        M = vals[i]
        H, coords = fractal_hamiltonian('honeycomb', 3, False, M)
        lattice = lattice_from_coords(coords)
        P = projector_exact(H, 10.0)
        bott = bott_index(P, lattice)
        return [M, bott]

    data = np.array(Parallel(n_jobs=4)(delayed(single)(i) for i in range(len(vals)))).T


    print(data)
    X, Y, Z = reshape_imshow_data(data)
    fig, ax = plt.subplots(1,1)
    fig, ax, cbar = plot_imshow(fig, ax, X, Y ,Z)
    plt.show()




if __name__ == "__main__":
        # Topological region: |M| < |3 * sqrt(3) * t2 * sin(phi)|
        H, coords = fractal_hamiltonian('honeycomb', 3, True, np.sqrt(3)*3/2)
        lattice = lattice_from_coords(coords)
        P = projector_exact(H, 0.0)
        bott = bott_index(P, lattice)
        print(bott)





def extra():
    gen = 3
    sl = int((gen**3+1)/2)
    hc = honeycomb_pos(sl)
    hf = hexaflake_pos(gen)



    def explain(pts):
        x = (pts[0]*2).astype(int)
        y = (pts[1]*2/np.sqrt(3)).astype(int)
        for arr in [x,y]:
            print(f"min : {np.min(arr)}")
            print(f"max : {np.max(arr)}")
            print(f"range : {np.max(arr)-np.min(arr)}")
            print(f"amount : {np.size(arr)}")
            print()

    for arr in [hc, hf]:
        arr = np.unique(arr, axis=1)
        explain(arr)

    fig, axs = plt.subplots(1,2)
    axs[0].scatter(hc[0], hc[1])
    axs[1].scatter(hf[0],hf[1])

    plt.show()









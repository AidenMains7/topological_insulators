import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse import dok_matrix, csr_matrix, diags
from scipy.sparse.linalg import cg

import sys
sys.path.append(".")
from ProjectCode.DisorderAveraging.DisorderDependencies import mat_inv


def create_lattice(order:int, pad_width:int=0) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Generates a Sierpinski carpet fractal lattice and its corresponding square lattice, as well as the index locations for the empty and non-empty sites.
    '''
    if (order < 0):
        raise ValueError("Order of lattice must be >= 0.")

    #A sierpinski carpet fractal is infinitely self-similar, however we may compute it only up to a certain order
    def sierpinski_carpet(order_):
        if(order_ == 0):
            return np.array([1], dtype=int)
    
        #carpet of one lower degree; recursion
        carpet_lower = sierpinski_carpet(order_-1)

        #concatenate to make current degree
        top = np.hstack((carpet_lower,carpet_lower,carpet_lower))
        mid = np.hstack((carpet_lower,carpet_lower*0,carpet_lower))
        carpet = np.vstack((top,mid,top))

        return carpet
    
    #'side length' in one dimension
    L = 3**order

    #square lattice
    square_lat = np.arange(L*L).reshape((L,L))

    carpet = sierpinski_carpet(order)

    #now, we account for padding (if wanted)
    if(pad_width > 0):
        carpet = np.pad(carpet,pad_width,mode='constant',constant_values=1)
    
    
    #Determining indecies for holes and otherwise
    #flattened array
    flat = carpet.flatten()

    #locations of the filled and empty sites
    filled_indices = np.flatnonzero(flat)
    empty_indices = np.where(flat==0)[0]

    #lattice 
    fractal_lat = np.full(flat.shape, -1, dtype=int)
    fractal_lat[filled_indices] = np.arange(filled_indices.size)
    fractal_lat = fractal_lat.reshape(carpet.shape)

    return square_lat, fractal_lat, empty_indices, filled_indices



def geometry(lattice: np.ndarray, pbc: bool, n: int) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Finds the distance and angle between sites. Polar.

    Also provides masks for the principal axes (xy plane) and the diagonal (rotated 45 deg)
    '''

    side_len = lattice.shape[0]

    if pbc and (n >= side_len//2):
        raise ValueError("Cutoff length must be half of system size while periodic boundary conditions.")

    #where the lattice does not contain a hole
    filled_idx = np.argwhere(lattice > -1)

    #this is wizardry
    diff = filled_idx[None,:,:] - filled_idx[:,None,:]
    dy, dx = diff[..., 0], diff[..., 1]

    #if periodic, set dx and dy such that if they exit the lattice, wrap
    if pbc:
        dx = np.where(np.abs(dx) > side_len / 2, dx - np.sign(dx) * side_len, dx)
        dy = np.where(np.abs(dy) > side_len / 2, dy - np.sign(dy) * side_len, dy)

    #mask as to only consider sites in which the cutoff is not exceeded.
    dist_mask = np.maximum(np.abs(dx), np.abs(dy)) <= n

    #separate masks for principal and diagonal directions
    prin_mask = ( ((dx == 0) and (dy != 0)) or ((dx != 0) and (dy == 0)) ) and dist_mask #only one diff is nonzero
    diag_mask = ( (np.abs(dx) == np.abs(dy)) and (dx != 0) and (dy != 0) ) and dist_mask #45 deg diag; absolute value of diff is the same for each axes and they are both nonzero

    pre_mask = prin_mask or diag_mask

    #get distance between pairs within mask
    dr = np.where(pre_mask, np.maximum(np.abs(dx), np.abs(dy)), 0)

    #calculate angles between pairs within mask
    cos_dphi = np.where(pre_mask, np.cos(np.arctan2(dy, dx)), 0.)
    sin_dphi = np.where(pre_mask, np.sin(np.arctan2(dy, dx)), 0.)

    return dr, cos_dphi, sin_dphi, prin_mask, diag_mask


#do this one
def wannier_symmetry(lattice: np.ndarray, pbc: bool, n: int, r0:float=1):
    '''
    check this again
    '''

    dr, cos_dphi, sin_dphi, prncpl_mask, diag_mask = geometry(lattice, pbc, n)

    system_size = np.max(lattice)+1

    #identity matrix
    I = np.eye(system_size, dtype=np.complex128)

    #exponential decay for principal & diagonal dir
    F_p = np.where(prncpl_mask, np.exp(1 - dr/r0), 0.+0.j)
    F_d = np.where(diag_mask, np.exp(1 - dr/r0), 0.+0.j)

    Sx = 1j/2 * cos_dphi*F_p
    Sy = 1j/2 * sin_dphi*F_p
    Cx_p_Cy = F_d/4



def wannier_fourier(lattice:np.ndarray, pbc:bool) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    system_size = np.max(lattice) + 1
    L_y, L_x = lattice.shape

    # sparse, save time
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
            position = lattice[y,x]
            
            #checking that the current position is not a hole
            if position > -1:
                x_neg = (x-1)%L_x
                x_pos = (x+1)%L_x
                y_pos = (y+1)%L_y

                #principal x dir
                lxp = lattice[y, x_pos] #location x positive
                x_hop = (pbc or x_pos != 0) and lxp > -1 #only hop if location is not hole and only wraps if pbc is true

                #principal y dir
                lyp = lattice[y_pos, x] #location y positive
                y_hop = (pbc or y_pos != 0) and lyp > -1

                #diag 1
                lypxp = lattice[y_pos, x_pos] #location y pos. x pos.
                xy1_hop = (pbc or x_pos != 0) and (pbc or y_pos != 0) and lypxp > -1

                #diag 2
                lypxn = lattice[y_pos, x_neg] #location y pos. x neg.
                xy2_hop = (pbc or x_neg != 0) and (pbc or y_pos != 0) and lypxn > -1

                if x_hop:
                    Cx[position, lxp] = 1/2
                    Sx[position, lxp] = 1j/2

                if y_hop:
                    Cy[position, lyp] = 1/2
                    Sy[position, lyp] = 1j/2

                if xy1_hop:
                    CxSy[position, lypxp] = 1j/4
                    SxCy[position, lypxp] = 1j/4
                    CxCy[position, lypxp] = 1/4

                if xy2_hop:
                    CxSy[position, lypxn] = 1j/4
                    SxCy[position, lypxn] = -1j/4
                    CxCy[position, lypxn] = 1/4

    #ensure hermitian
    Cx += Cx.conj().T
    Sx += Sx.conj().T
    Cy += Cy.conj().T
    Sy += Sy.conj().T
    CxSy += CxSy.conj().T
    SxCy += SxCy.conj().T
    CxCy += CxCy.conj().T
        
    #to numpy array
    Sx, Sy, Cx, Cy, CxSy, SxCy, CxCy = [arr.toarray() for arr in [Sx, Sy, Cx, Cy, CxSy, SxCy, CxCy]]

    return I, Sx, Sy, Cx + Cy, CxSy, SxCy, CxCy



def Hamiltonian(M:float, B_til:float, wannier_matrices:tuple, t1:float=1., t2:float=1., B:float=1.):
    '''
    Construct hamiltonian using wannier matrices
    '''
    #pauli matrices
    theta = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]]) 

    #wannier
    I, Sx, Sy, Cx_plus_Cy, CxSy, SxCy, CxCy = wannier_matrices

    d = tuple()
    d[0] = t1*Sx + t2*CxSy
    d[1] = t1*Sy + t2*SxCy
    d[2] = (M-4*B - 4*B_til)*I + 2*B*Cx_plus_Cy + 4*B_til*CxCy

    #construct hamiltonain
    H = 0
    for i in range(3):
        H += np.kron(d[i],theta[i])

    return H


        
def H_site_elim(H:np.ndarray, fills:np.ndarray, holes:np.ndarray) -> np.ndarray:
    '''
    First method to construct the hamiltonian for only filled sites
    -via site elimination
    '''

    #get amount of holes and fills
    amt_fill = fills.size
    amt_hole = holes.size

    #total number of sites
    amt_sites = amt_fill + amt_hole

    #how many states per site?
    states_per = H.shape[0]//amt_sites

    #empty matrices to give the indices of each site type
    fill_idx = np.empty(states_per*amt_fill, dtype=int)
    hole_idx = np.empty(states_per*amt_hole, dtype=int)

    #wizardry
    for i in range(states_per):
        fill_idx[i::states_per] = states_per*fills + i
        hole_idx[i::states_per] = states_per*holes + i

    #concatenate the index matrices
    reorder = np.concatenate((fill_idx, hole_idx))

    #separate the fill and hole into blocks, reorder
    H_reorder = H.copy()[np.ix_(reorder, reorder)]

    #size of our hamiltonian only considering the fills
    H_eff_size = states_per*amt_fill

    #only the fill section of the hamiltonian
    H_eff = H_reorder[:H_eff_size, :H_eff_size]

    return H_eff


#(self) come back do better
def matrix_inverse(matrix:np.ndarray, hermitian:bool=True, alt:bool=True, overwrite_a:bool=True, tol:float=1e-10) -> np.ndarray:
    #unnecessary to rewrite mat_inv, just done below to maintain consistency
    #this is surely bad practice
    return mat_inv(matrix, hermitian, alt, overwrite_a, tol)


#(self) come back do better
from ProjectCode.DisorderAveraging.DisorderDependencies import mat_solve_iterative



def H_renorm(H:np.ndarray, fills:np.ndarray, holes:np.ndarray) -> np.ndarray:
    '''
    Second method to construct the hamiltonian for only the filled sites
    -via Schur complement, maintains consideration of the properties of the holes
    '''
    #get amount of holes and fills
    amt_fill = fills.size
    amt_hole = holes.size

    #total number of sites
    amt_sites = amt_fill + amt_hole

    #how many states per site?
    states_per = H.shape[0]//amt_sites

    #empty matrices to give the indices of each site type
    fill_idx = np.empty(states_per*amt_fill, dtype=int)
    hole_idx = np.empty(states_per*amt_hole, dtype=int)

    #wizardry
    for i in range(states_per):
        fill_idx[i::states_per] = states_per*fills + i
        hole_idx[i::states_per] = states_per*holes + i

    #concatenate the index matrices
    reorder = np.concatenate((fill_idx, hole_idx))

    #separate the fill and hole into blocks, reorder
    H_reorder = H.copy()[np.ix_(reorder, reorder)]

    #size of our hamiltonian only considering the fills
    H_eff_size = states_per*amt_fill

    #H_reorder is of form [[aa,ab],[ba,bb]].
    H_aa = H_reorder[:H_eff_size, :H_eff_size]
    H_bb = H_reorder[H_eff_size:, H_eff_size:]
    H_ab = H_reorder[:H_eff_size, H_eff_size:]
    H_ba = H_reorder[H_eff_size:, :H_eff_size]

    #more computationally efficient to use mat_solve_iterative
    try:
        # Use iterative solver for inverting H_bb if possible.
        solve_H_bb = mat_solve_iterative(H_bb)
        H_ba_solved = np.hstack([solve_H_bb(H_ba[:, i].ravel()).reshape(-1, 1) for i in range(H_ba.shape[1])])
        H_eff = H_aa - H_ab @ H_ba_solved
    except:
        # Fallback to direct inversion if iterative solver fails.
        #using Schur complement
        H_eff = H_aa - H_ab @ mat_inv(H_bb) @ H_ba

    return H_eff



def H_wrapper(sierpinski_order:int, method:str, M:float, B_tilde:float, pbc:bool=True, pad_width:int=0, n:int=None, **kwagrs) -> tuple[np.ndarray, np.ndarray]:
    
    #check that method is proper
    if method not in ['symmetry', 'square', 'site_elim', 'renorm']:
        raise ValueError(f"Invalid method {method}: options are ['symmetry', 'square', 'site_elim', 'renorm'].")
    if method == 'symmetry' and not isinstance(n, int):
        raise ValueError("Parameter 'n' must be specified and must be an integer.")
    

    #create lattices, fills, holes
    square_lat, fractal_lat, fills, holes = create_lattice(sierpinski_order, pad_width)

    if method == 'symmetry':
        #do symmetry
        wannier = wannier_symmetry(fractal_lat,pbc,n)
        H = Hamiltonian(M,B_tilde,wannier)
        return H, fractal_lat
    else:
        #do fourier
        wannier = wannier_fourier()
        H = Hamiltonian(M, B_tilde, wannier)

        if method == 'square':
            #do square
            return H, square_lat
        elif method == 'site elim':
            #use site elimination
            H_eff = H_site_elim(H, fills, holes)
            return H_eff, fractal_lat
        else:
            #use Schur complement
            H_eff = H_renorm(H, fills, holes)
            return H_eff, fractal_lat



def disorder(disorder_strength:float, system_size:int, df:int, sparse:bool) -> np.ndarray:
    
    #random values in [-d]
    disorder_arr = np.random.uniform(-disorder_strength / 2, disorder_strength / 2, size=system_size)

    #normalize so mean=0
    delta = np.sum(disorder_arr) / system_size
    disorder_arr -= delta

    #repeat disorder values for df
    disorder_arr = np.repeat(disorder_arr, df)

    #create diagonal matrix
    disorder_op = np.diag(disorder_arr).astype(np.complex128) if not sparse else diags(disorder_arr, dtype=np.complex128, format='csr')

    return disorder_op


def main():
    
    
    pass


if __name__ == "__main__":
    main()
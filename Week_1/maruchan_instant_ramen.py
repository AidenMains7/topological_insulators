'''



create_lattice



'''


import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse import dok_matrix, csr_matrix, diags
from scipy.sparse.linalg import cg

def create_lattice(order:int, pad_width:int=0) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Generates a Sierpinski carpet fractal lattice and its corresponding square lattice, as well as the index locations for the empty and filled sites.
    
    Parameters:
    order (int): the order of the fractal
    pad_width (int): amount of padding on the perimeter of the lattice

    Returns:
    square_lat (np.ndarray): a square lattice of the same size
    fractal_lat (np.ndarray): the fractal lattice, with holes having value -1
    hole_indices (np.ndarray): the indices of the holes (empty sites)
    filled_indices (np.ndarray): the indices of the filled sites
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
    hole_indices = np.where(flat==0)[0]

    #lattice 
    fractal_lat = np.full(flat.shape, -1, dtype=int)
    fractal_lat[filled_indices] = np.arange(filled_indices.size)
    fractal_lat = fractal_lat.reshape(carpet.shape)

    return square_lat, fractal_lat, hole_indices, filled_indices




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

    print("filled idx shape=",filled_idx.shape)
    print(filled_idx[None,:,:].shape)
    print(filled_idx[:,None,:].shape)
    print(diff.shape)
    print(dy.shape)
    print(dx.shape)

    #if periodic, set dx and dy such that if they exit the lattice, wrap
    if pbc:
        dx = np.where(np.abs(dx) > side_len / 2, dx - np.sign(dx) * side_len, dx)
        dy = np.where(np.abs(dy) > side_len / 2, dy - np.sign(dy) * side_len, dy)

    #mask as to only consider sites in which the cutoff is not exceeded.
    dist_mask = np.maximum(np.abs(dx), np.abs(dy)) <= n

    #separate masks for principal and diagonal directions
    prin_mask = ( ((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0)) ) & dist_mask #only one diff is nonzero
    diag_mask = ( (np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0)) ) & dist_mask #45 deg diag; absolute value of diff is the same for each axes and they are both nonzero

    pre_mask = prin_mask | diag_mask

    #get distance between pairs within mask
    dr = np.where(pre_mask, np.maximum(np.abs(dx), np.abs(dy)), 0)

    #calculate angles between pairs within mask
    cos_dphi = np.where(pre_mask, np.cos(np.arctan2(dy, dx)), 0.)
    sin_dphi = np.where(pre_mask, np.sin(np.arctan2(dy, dx)), 0.)

    return dr, cos_dphi, sin_dphi, prin_mask, diag_mask






def main():
    sq, frac, fills, holes = create_lattice(2)
    dr, cos_dphi, sin_dphi, prin_mask, diag_mask = geometry(frac, True, 2)

if __name__ == "__main__":
    main()








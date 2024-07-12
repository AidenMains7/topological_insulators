import numpy as np
from time import time


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


def new_gemometry(lattice:np.ndarray, pbc:bool, n:int):
    """
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

    # Previous methods are used in the previous iteration of the geometry function.------------------------
    # Create a mask which does not cross over a vacancy
    def hop_mask(idx):
        row, col = tuple(idx)
        rows = [i%side_length for i in range(row-n, row+n+1)]
        cols = [i%side_length for i in range(col-n, col+n+1)]

        square = lattice[np.ix_(rows, cols)]

        # A direction and the value to move in that direction
        direction = {
            "LEFT": np.array([-1,0]),
            "RIGHT": np.array([1,0]),
            "UP": np.array([0,-1]),
            "DOWN": np.array([0,1]),
            "UP_LEFT": np.array([-1,-1]),
            "UP_RIGHT": np.array([1, -1]),
            "DOWN_LEFT": np.array([-1, 1]),
            "DOWN_RIGHT": np.array([1,1])
        }

        # Dictionary to control whether a vacancy has been encountered in a certain direction
        direction_flag = {
            "LEFT": True,
            "RIGHT": True,
            "UP": True,
            "DOWN": True,
            "UP_LEFT": True,
            "UP_RIGHT": True,
            "DOWN_LEFT": True,
            "DOWN_RIGHT": True
        }      

        # Center of the square
        center = np.array([square.shape[0]//2]*2)
        square_mask = np.empty(square.shape, dtype=bool)

        # For each 'ring' around the center
        for dist in range(n):
            for dir in direction:
                hop = tuple(center + direction[dir]*(dist+1))

                # If vacant, false
                if square[hop] == -1:
                    val = False
                    direction_flag[dir] = False

                # Controls whether there has been a vacancy in this direction
                elif direction_flag[dir] == False:
                    val = False
                # Otherwise, true
                else:
                    val = True
                
                # Update value
                # Values not attempted to hop to are automatically false.
                square_mask[hop] = val

        
        return square.flatten(), square_mask.flatten()


    # Original method to determine cutoff
    mask_dist = np.maximum(np.abs(dx), np.abs(dy)) <= n


    # Calculate for all sites
    for site in range(np.max(lattice)+1):
        square, square_mask = hop_mask(filled[site])
        non_vacant = np.argwhere(square >= 0)
        square, square_mask = square[non_vacant], square_mask[non_vacant]
        mask_dist[site, square] = square_mask

    # Following methods are used in previous iteration of the geometry function.----------------------

    mask_principal = (((dx == 0) & (dy != 0))    | ((dx != 0) & (dy == 0))) & mask_dist
    mask_diagonal  = ((np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0))) & mask_dist

    mask_both = mask_principal | mask_diagonal

    d_r   = np.where(mask_both, np.maximum(np.abs(dx), np.abs(dy)), 0)
    d_cos = np.where(mask_both, np.cos(np.arctan2(dy, dx)),         0.)
    d_sin = np.where(mask_both, np.sin(np.arctan2(dy, dx)),         0.)

    return d_r, d_cos, d_sin, mask_principal, mask_diagonal

        


def main():
    sq, frac, fills, holes = sierpinski_lattice(2, 0)
    print(frac)

    t0 = time()
    d_r, d_cos, d_sin, mask_principal, mask_diagonal = new_gemometry(frac, True, 2)
    print(f't={time()-t0}s')

    t0 = time()
    d_r, d_cos, d_sin, mask_principal_2, mask_diagonal_2 = geometry(frac, True, 2)
    print(f't={time()-t0}s')

    print("Using method to elimiate hopping across vacancy.")
    print(np.argwhere(mask_principal[10, :] == True).flatten())

    print("Old method.")
    print(np.argwhere(mask_principal_2[10, :] == True).flatten())






if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: "%.3g" % x))
    main()
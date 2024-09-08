import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import dok_matrix
from numba import jit

import sys
sys.path.append(".")
from time import time
from Carpet.plotting import plot_imshow

# naive method to prevent trying to import scienceplots on wkstn
if sys.version_info[1] > 9:
    import scienceplots
    plt.style.use(['pgf'])


# Creates a 2xN array of point position values
def honeycomb_lattice(side_length:int) -> np.ndarray:
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

    return hexagon_lattice

# Creates a 2xN array of point position values
def hexaflake_lattice(generation:int) -> np.ndarray:
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

# Turns point position values into lattices
def fractal_and_honeycomb_lattices(generation:int, honeycomb_side_length:int) -> tuple:
    """
    Parameters:
    generation (int): the generation of the fractal
    honeycomb_side_length (int): Length of lattice in unit cells
    """
    def _create_lattice(positions:np.ndarray):
        positions[0] *= 2.0
        positions[1] *= 2/np.sqrt(3)

        positions[0] -= np.min(positions[0])
        positions[1] -= np.min(positions[1])

        positions = np.round(positions, 0).astype(int)
        positions = np.unique(positions, axis=1)
        idxs = np.lexsort((positions[0, :], positions[1, :]))
        positions = positions[:, idxs]

        lattice = np.ones((np.max(positions[1]).astype(int)+1, np.max(positions[0]).astype(int)+1), dtype=int)*(-1)
        lattice[positions[1], positions[0]] = np.arange(positions.shape[1])
        fills = np.argwhere(lattice >= 0)
        holes = np.argwhere(lattice < 0)

        return positions, lattice, fills, holes
    
    hexaflake = hexaflake_lattice(generation)
    honeycomb = honeycomb_lattice(honeycomb_side_length)

    hexaflake_int_positions, fractal_lattice, fractal_fills, fractal_holes = _create_lattice(hexaflake)
    honeycomb_int_positions, pristine_lattice, pristine_fills, pristine_holes = _create_lattice(honeycomb)

    return fractal_lattice, fractal_fills, fractal_holes, pristine_lattice, pristine_fills, pristine_holes

# Generates list of all valid hops
def get_hopping_sites(fills:np.ndarray, pbc:bool=False) -> list:
    """
    Find the list of all valid hops from each site on the lattice.

    Returns: 
    valid_hops (list): List of lists. Each interior list is [initial site, final site, hop type]. Both sites are positions on the lattice as [x, y]
    """
    def _pbc_tile(init_pos:np.ndarray) -> "list[np.ndarray]":  
        """
        Presumes scaling such that positions are integer values. 
        Parameters:  
        init_lattice_positions (ndarray): 2 x N array, such that the first row is x-positions and the second is y-positions.
        """
        x_max, y_max = np.max(init_pos[0]), np.max(init_pos[1])

        lab = ['tr', 'tl', 'br', 'bl', 't', 'b']
        displacements = [None]*6
        displacements[0] = [x_max*3/4+3, y_max/2-1]
        displacements[1] = [-x_max*3/4, y_max/2+2]
        displacements[2] = [x_max*3/4, -y_max/2-2]
        displacements[3] = [-x_max*3/4-3, -y_max/2+1]
        displacements[4] = [3, y_max+1]
        displacements[5] = [-3, -y_max-1]
        
        tiles = []
        for d in displacements:
            tiles.append(init_pos+np.array(d).reshape(2, 1))

        # MAKE TRUE TO VISUAL PLOT PBC
        if False:
            cmap = plt.colormaps['viridis']
            colors = cmap(np.linspace(0, 1, 7))

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            for i in range(len(tiles)):
                ax.scatter(tiles[i][0], tiles[i][1], label=lab[i], c=colors[i], alpha=1.0)
            ax.scatter(init_pos[0], init_pos[1], label='init', c=colors[-1], alpha=1.0)
            ax.legend()
            plt.show()

        return tiles
    
    def _get_mean_xy(_p):
        # xmin, xmax, xmean, ymin, ymax, ymean
        values = [np.min(_p[0]), np.max(_p[0]), np.mean(_p[0]),
                  np.min(_p[1]), np.max(_p[1]), np.mean(_p[1])]
        return [values[2], values[5]]


    # Six hop types, labels for each hop
    hops = [[0, -2], [1, 1],   [-1, 1], [0, 2],  [-1, -1], [1, -1]] # A1, A2, A3, B1, B2, B3
    hop_type = [f'{l}{i}' for l in ['A', 'B'] for i in [1, 2, 3]]

    # The integer-array positions of every point in the lattice
    init_lattice_positions = np.flipud(fills.T)

    # Six arrays of all possible hops from one lattice site to another
    all_possible_hops = []
    for hop in hops:
        all_possible_hops.append(init_lattice_positions + np.array(hop).reshape(2, 1))

    # Create six identical lattices tiled around the initial lattice.
    pbc_arrays = _pbc_tile(init_lattice_positions)

    # Calculate the hopping for each site
    valid_hops = []
    pbc_hops = []
    init_means = _get_mean_xy(init_lattice_positions)
    for i in range(len(hops)):
        for j in range(all_possible_hops[i].size//2):
            # Starting site, hop candidate
            init_site = init_lattice_positions[:, j].flatten()
            site = all_possible_hops[i][:, j].reshape(2, 1)

            # If site is in the lattice
            if any(np.equal(init_lattice_positions, site).all(0)):
                valid_hops.append([init_site.tolist(), site.flatten().tolist(), hop_type[i]])
            else:
                pass

            # Must also check tiled lattices surrounding with pbc
            if pbc:
                for k in range(len(pbc_arrays)):
                    new_means = _get_mean_xy(pbc_arrays[k])
                    if any(np.equal(pbc_arrays[k], site).all(0)):
                        # Wrap into inital lattice
                        displacement = np.array([init_means[0]-new_means[0], init_means[1]-new_means[1]])
                        site_in_init = (site.flatten() + displacement.flatten()).reshape(2, 1).astype(int)

                        # Check, likely unnecessary
                        if any(np.equal(init_lattice_positions, site_in_init).all(0)):
                            pass
                        else:
                            raise ValueError(f"Issue regarding site_in_init: init, site, site_in_init = {init_site}, {site}, {site_in_init}")
                        
                        pbc_hops.append([init_site.tolist(), site_in_init.flatten().tolist(), hop_type[i]])
                        break
                    else:
                        pass
    
    if not pbc:
        return valid_hops
    else:
        return valid_hops+pbc_hops


def construct_hamiltonian(lattice:np.ndarray, fills:np.ndarray, hops:list, t:float, M:float, B:float):
    """
    Directing construct the Hamiltonian using hopping.
    """

    # Initialize lattices
    n_sites = fills.size//2
    I = np.eye(n_sites, dtype=np.complex128)
    d_tilde1 = dok_matrix((n_sites, n_sites), dtype=np.complex128)
    d_tilde2 = dok_matrix((n_sites, n_sites), dtype=np.complex128)
    d_tilde3 = dok_matrix((n_sites, n_sites), dtype=np.complex128)

    for hop in hops:
        site_i, site_f, hop_type = hop
        idx_i = lattice[(site_i[1], site_i[0])]
        idx_f = lattice[(site_f[1], site_f[0])]

        if hop_type[0] == "A":
            if hop_type[1] == "1":
                d_tilde1[idx_i, idx_f] = 1j/2
                d_tilde2[idx_i, idx_f] = 0.0
                d_tilde3[idx_i, idx_f] = 1.0
            elif hop_type[1] == "2":
                d_tilde1[idx_i, idx_f] = -1j/4
                d_tilde2[idx_i, idx_f] = np.sqrt(3)*1j/4
                d_tilde3[idx_i, idx_f] = 1.0
            else:
                d_tilde1[idx_i, idx_f] = -1j/4
                d_tilde2[idx_i, idx_f] = -np.sqrt(3)*1j/4
                d_tilde3[idx_i, idx_f] = 1.0
        else:
            if hop_type[1] == "1":
                d_tilde1[idx_i, idx_f] = -1j/2
                d_tilde2[idx_i, idx_f] = 0.0
                d_tilde3[idx_i, idx_f] = 1.0
            elif hop_type[1] == "2":
                d_tilde1[idx_i, idx_f] = 1j/4
                d_tilde2[idx_i, idx_f] = -np.sqrt(3)*1j/4
                d_tilde3[idx_i, idx_f] = 1.0
            else:
                d_tilde1[idx_i, idx_f] = 1j/4
                d_tilde2[idx_i, idx_f] = np.sqrt(3)*1j/4
                d_tilde3[idx_i, idx_f] = 1.0

    d_tilde1, d_tilde2, d_tilde3 = d_tilde1.toarray(), d_tilde2.toarray(), d_tilde3.toarray()

    pauli1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    d1 = t*d_tilde1
    d2 = t*d_tilde2
    d3 = (M-4*B)*I + B*d_tilde3

    H = np.kron(d1, pauli1) + np.kron(d2, pauli2) + np.kron(d3, pauli3)
    return H
        


if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=10000, formatter=dict(float=lambda x: "%.3g" % x))

    generation = 3
    side_length = 14
    t0 = time()
    fractal_lattice, fractal_fills, fractal_holes, pristine_lattice, pristine_fills, pristine_holes = fractal_and_honeycomb_lattices(generation, side_length)
    print(f"time to generate lattices: {(time()-t0)*1000:.3f}ms")
    t0 = time()
    hops = get_hopping_sites(pristine_fills, False)
    print(f"time to get valid hops: {(time()-t0)*1000:.3f}ms")
    
    t = 1.0
    M = 1.0
    B = 1.0

    H = construct_hamiltonian(pristine_lattice, pristine_fills, hops, t, M, B)

    if False:
        fractal_lattice[fractal_fills[:, 0], fractal_fills[:, 1]] = np.ones(fractal_fills.shape[0])
        pristine_lattice[pristine_fills[:, 0], pristine_fills[:, 1]] = np.ones(pristine_fills.shape[0])

        def naive_plot(lat):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_imshow(fig, ax, np.arange(lat.shape[1]), np.arange(lat.shape[0]), lat)
            plt.show()

        naive_plot(fractal_lattice)
        naive_plot(pristine_lattice)




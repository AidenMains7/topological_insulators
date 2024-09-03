import numpy as np
from matplotlib import pyplot as plt
from time import time
from numba import jit

import sys
sys.path.append(".")
from Carpet.plotting import plot_imshow

import sys
if sys.version_info[1] > 9 and False:
    import scienceplots
    plt.style.use(['science', 'pgf'])


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


def fractal_and_honeycomb_lattices(generation:int, honeycomb_side_length:int):
    """
    
    """
    def _create_lattice(positions:np.ndarray):
        positions[0] *= 2.0
        positions[1] *= 2/np.sqrt(3)

        positions[0] -= np.min(positions[0])
        positions[1] -= np.min(positions[1])

        positions = np.round(positions, 0).astype(int)

        lattice = np.ones((np.max(positions[1]).astype(int)+1, np.max(positions[0]).astype(int)+1), dtype=int)*(-1)
        lattice[positions[1], positions[0]] = np.ones(positions.shape[1]) 

        fills = np.argwhere(lattice >= 0)
        holes = np.argwhere(lattice < 0)
        return positions, lattice, fills, holes
    
    hexaflake = hexaflake_lattice(generation)
    honeycomb = honeycomb_lattice(honeycomb_side_length)

    hexaflake_int_positions, fractal_lattice, fractal_fills, fractal_holes = _create_lattice(hexaflake)
    honeycomb_int_positions, pristine_lattice, pristine_fills, pristine_holes = _create_lattice(honeycomb)

    return fractal_lattice, fractal_fills, fractal_holes, pristine_lattice, pristine_fills, pristine_holes

def get_hopping_sites(fills:np.ndarray, pbc:bool=False):
    """
    
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

        if True:
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
    pbc_labels = ['tr', 'tl', 'br', 'bl', 't', 'b']

    # Calculate the hopping for each site
    valid_hops = []
    pbc_hops = []
    init_means = _get_mean_xy(init_lattice_positions)
    for i in range(len(hops)):
        for j in range(all_possible_hops[i].size//2):
            init_site = init_lattice_positions[:, j].flatten()
            site = all_possible_hops[i][:, j].reshape(2, 1)


            if any(np.equal(init_lattice_positions, site).all(0)):
                valid_hops.append([init_site.tolist(), site.flatten().tolist(), hop_type[i]])
            else:
                pass

            if pbc:
                for k in range(len(pbc_arrays)):
                    new_means = _get_mean_xy(pbc_arrays[k])
                    if any(np.equal(pbc_arrays[k], site).all(0)):
                        # Wrap into inital lattice
                        displacement = np.array([init_means[0]-new_means[0], init_means[1]-new_means[1]])
                        site_in_init = (site.flatten() + displacement.flatten()).reshape(2, 1).astype(int)

                        # Check
                        if any(np.equal(init_lattice_positions, site_in_init).all(0)):
                            pass
                        else:
                            raise ValueError(f"Issue regarding site_in_init: init, site, site_in_init = {init_site}, {site}, {site_in_init}")
                        
                        pbc_hops.append([init_site.tolist(), site_in_init.flatten().tolist(), hop_type[i]])
                        break
                    else:
                        pass
    
    for e in pbc_hops:
        print(e)
    
    if not pbc:
        return valid_hops
    else:
        return valid_hops+pbc_hops



if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=10000, formatter=dict(float=lambda x: "%.3g" % x))

    generation = 3
    side_length = 14
    t0 = time()
    fractal_lattice, fractal_fills, fractal_holes, pristine_lattice, pristine_fills, pristine_holes = fractal_and_honeycomb_lattices(generation, side_length)
    print(f"time to generate lattices: {time()-t0:.3f}s")
    get_hopping_sites(fractal_fills, True)


    if False:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        #fig, ax, cbar = plot_imshow(fig, ax, np.arange(pbc_lattice_arrs.shape[1]), np.arange(pbc_lattice_arrs.shape[0]), pbc_lattice_arrs, doDiscreteCmap=True)
        #plt.savefig('pristine_imshow.png')
        plt.show()

    if False:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))


        ax.scatter(hexaflake_positions[0], hexaflake_positions[1], label='hexaflake_lattice')
        ax.set_title(f'Hexaflake - Generation {generation:.0f}')
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(honeycomb_positions[0], honeycomb_positions[1], label='honeycomb_lattice')
        ax.set_title(f"Honeycomb Lattice - Side Length {side_length}")
        plt.show()

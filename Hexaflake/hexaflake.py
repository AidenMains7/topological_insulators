import numpy as np
from matplotlib import pyplot as plt
from time import time

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


def fractal_and_honeycomb_lattices(generation:int, honeycomb_side_length:int):
    hexaflake = hexaflake_lattice(generation)
    honeycomb = honeycomb_lattice(honeycomb_side_length)

    hexaflake_int_positions, fractal_lattice, fractal_fills, fractal_holes = _create_lattice(hexaflake)
    honeycomb_int_positions, pristine_lattice, pristine_fills, pristine_holes = _create_lattice(honeycomb)

    return fractal_lattice, fractal_fills, fractal_holes, pristine_lattice, pristine_fills, pristine_holes

def get_hopping_sites(lattice:np.ndarray, fills:np.ndarray, pbc:bool=False):
    # List of directions (delta y, delta x):
    # From red to blue A1, A2, A3:
    # 0, -2 
    # 1, 1
    # -1, 1
    # From blue to red B1, B2, B3:
    # 0, 2
    # -1, -1
    # 1, -1

    hops = [[0, -2], [1, 1],   [-1, 1], [0, 2],  [-1, -1], [1, -1]] # A1, A2, A3, B1, B2, B3
    hop_type = [f'{l}{i}' for l in ['A', 'B'] for i in [1, 2, 3]]

    full_arr_hops = []
    for hop in hops:
        full_arr_hops.append(fills + np.array(hop).reshape(1, 2))


    def pbc_tile():
        points = np.flip(fills.T, axis=0)   
        plt.scatter(points[0], points[1])

        plt.show()
        "points (ndarray): 2xN array, such that the first row is x-positions and the second is y-positions."
        range_y = np.max(points[1]) - np.min(points[1])
        a = 3
        b = range_y+1
        theta = np.pi/2 - np.arctan(a/b)
        r = np.sqrt(a**2 + b**2)

        tile_arrs = []
        for i in range(6):
            new_points = points + np.array([[r*np.cos(theta+np.pi/3*i)], [r*np.sin(theta+np.pi/3*i)]])
            tile_arrs.append(new_points)
            
        return tile_arrs
    

    points = np.flip(fills.T, axis=0)
    possible_hop_locations = []
    for hop in hops:
        possible_hop_locations.append(points + np.array(hop).reshape(2, 1))


    
    tiles = pbc_tile()

    site_hops = []
    for i in range(len(possible_hop_locations)):
        for j in range(possible_hop_locations[i].shape[1]):
            site = possible_hop_locations[i][:, j].reshape(2, 1)
            if any(np.equal(points, site).all(1)):
                site_hops.append([list(fills[j, :]), list(site), hop_type[i]])
            elif pbc:
                for tile in tiles:
                    print(tile)
                    if any(np.equal(tile, site).all(1)):
                        print('hit')
                        found = np.all(tile == site, axis=1)
                        

                        tile[0] -= np.min(tile[0])
                        tile[1] -= np.min(tile[1])



    return

    site_hops = []
    for i in range(len(full_arr_hops)):
        for j in range(len(full_arr_hops[i])):
            site = full_arr_hops[i][j]
            if site[0] < lattice.shape[0] and site[1] < lattice.shape[1]:
                if lattice[site[0], site[1]] >= 0:
                    site_hops.append([list(fills[j, :]), list(site), hop_type[i]])
            elif pbc:
                for tile in tiles:
                    pass
            else:
                pass


    for e in site_hops:
        if e[0] == [70, 58]:
            print(e)
    
    
    



if __name__ == "__main__":

    #np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=10000, formatter=dict(float=lambda x: "%.3g" % x))

    generation = 3
    side_length = 14
    fractal_lattice, fractal_fills, fractal_holes, pristine_lattice, pristine_fills, pristine_holes = fractal_and_honeycomb_lattices(generation, side_length)

    if True:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig, ax, cbar = plot_imshow(fig, ax, np.arange(pristine_lattice.shape[1]), np.arange(pristine_lattice.shape[0]), pristine_lattice, doDiscreteCmap=True)
        plt.savefig('pristine_imshow.png')
        plt.show()

    get_hopping_sites(pristine_lattice, pristine_fills, True)

    if False:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))


        ax.scatter(hexaflake_positions[0], hexaflake_positions[1], label='hexaflake_lattice')
        ax.set_title(f'Hexaflake - Generation {generation:.0f}')
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(honeycomb_positions[0], honeycomb_positions[1], label='honeycomb_lattice')
        ax.set_title(f"Honeycomb Lattice - Side Length {side_length}")
        plt.show()

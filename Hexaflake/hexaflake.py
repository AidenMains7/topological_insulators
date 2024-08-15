import numpy as np
from matplotlib import pyplot as plt
from time import time
from itertools import permutations

import sys
if sys.version_info[1] > 9:
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

    
def hexaflake_lattice(generation:int, honeycomb_side_length:int=14) -> tuple[tuple, tuple]:
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

    
    def create_lattice(hexaflake):
        hexaflake[0] *= 2.0
        hexaflake[1] *= 2/np.sqrt(3)

        hexaflake[0] -= np.min(hexaflake[0])
        hexaflake[1] -= np.min(hexaflake[1])

        hexaflake = np.round(hexaflake, 0).astype(int)

        lattice = np.ones((np.max(hexaflake[1]).astype(int)+1, np.max(hexaflake[0]).astype(int)+1), dtype=int)*(-1)
        lattice[hexaflake[1], hexaflake[0]] = np.arange(hexaflake.shape[1]) 

        fills = np.where(lattice >= 0)[0] 
        holes = np.where(lattice < 0)[0]

        return hexaflake, lattice, fills, holes

    hexaflake = fractal_iteration(generation)
    honeycomb = honeycomb_lattice(honeycomb_side_length)

    hexaflake_positions, fractal_lattice, fractal_fills, fractal_holes = create_lattice(hexaflake)
    honeycomb_positions, pristine_lattice, pristine_fills, pristine_holes = create_lattice(honeycomb)

    return hexaflake_positions, fractal_lattice, fractal_fills, fractal_holes, honeycomb_positions, pristine_lattice, pristine_fills, pristine_holes


if __name__ == "__main__":
    generation = 3
    side_length = 14
    hexaflake_positions, fractal_lattice, fractal_fills, fractal_holes, honeycomb_positions, pristine_lattice, pristine_fills, pristine_holes = hexaflake_lattice(generation, side_length)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(hexaflake_positions[0], hexaflake_positions[1], label='hexaflake_lattice')
    ax.set_title(f'Hexaflake - Generation {generation:.0f}')
    plt.savefig('hexaflake_lattice.png')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(honeycomb_positions[0], honeycomb_positions[1], label='honeycomb_lattice')
    ax.set_title(f"Honeycomb Lattice - Side Length {side_length}")
    plt.savefig('honeycomb_lattice.png')
    plt.show()

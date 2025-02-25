import numpy as np
from matplotlib import pyplot as plt


def compute_hexaflake(n):
    """
    Construct a boolean 2D array that represents a hexaflake pattern of order n.
    The hexaflake is created by recursively appending smaller hexagons around
    an initial hexagon shape, scaled by factors of 3.

    Args:
        n (int): The iteration order of the hexaflake. Higher values produce
            more fractal detail.

    Returns:
        np.ndarray: A 2D boolean array marking the presence of sites in the
            hexaflake (True) and empty space (False).
    """

    # Directions in which to replicate the smaller hexagons.
    directions = np.array([[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]])
    # Scale factors determine how far to offset for each recursion level.
    scale_factors = 3 ** np.arange(1, n + 1)

    # Start with the 6 directions and build up by adding scaled copies.
    discrete_coordinates = directions.copy()
    for scale in scale_factors:
        offsets = scale * directions
        new_coordinates = []
        for offset in [[0, 0]] + offsets.tolist():
            new_coordinates.extend(discrete_coordinates + offset)
        discrete_coordinates = np.array(new_coordinates)

    x_discrete, y_discrete = discrete_coordinates.T

    # Shift coordinates so there are no negative indices.
    x_discrete += 3 ** (n + 1) - 1
    y_discrete += (3 ** (n + 1) - 1) // 2

    # Create the array for the hexaflake pattern.
    hexaflake_array = np.full(
        (3 ** (n + 1), 2 * 3 ** (n + 1) - 1),
        False,
        dtype=bool
    )
    hexaflake_array[y_discrete, x_discrete] = True

    return hexaflake_array


def old():
     
    def recursive_ordering(coords, generation):
        if generation == 0:
            pass           
        else:
            reordered = order_by_subflake(coords, generation)
            subflakes = [coords[:, reordered[len(reordered)//7*i:len(reordered)//7*(i+1)]] for i in range(7)]

            for subflake_coords in subflakes:
                sfr = order_by_subflake(subflake_coords, generation-1)


                

            #subflakes_smaller = [subflake[:, get_subflake(subflake, generation-1)] for subflake in subflakes]
    
def plot_tiered_color(n_tiers, coords):
        x, y = coords
        unit = np.linspace(0, 1, n_tiers)
        values = np.sort(np.repeat(unit, len(x)//n_tiers))[:len(x)]
        plt.scatter(x,y, c=plt.get_cmap('viridis')(values))
        plt.show()



def sort_hexaflake_by_subflake(coords, generation):


    def order_by_subflake(coords, generation): 
            centers = (3**generation)*np.array([np.array(([2*np.cos(np.pi/3*a)], [2/np.sqrt(3)*np.sin(np.pi/3*a)])) for a in range(6)])[..., 0].T
            if generation >= 1:
                centers = np.append(centers, [[0],[0]], axis=1)[:, [1,2,0,6,3,5,4]]
            else:
                 centers = centers[:, [1, 2, 0, 3, 5, 4]]
            centers[0] += np.mean(coords[0])
            centers[1] += np.mean(coords[1])

            dx, dy = np.power(coords[..., np.newaxis] - centers[:, np.newaxis, :], 2)
            difference = dx + dy
            which_subflake = np.argmin(difference, axis=-1) # Assign each point to a subflake

            return np.argsort(which_subflake)

    def sort_by_x_then_y(coords):
        idxs = np.lexsort((-coords[0], -coords[1]), axis=0)
        return idxs
    
    reordered = order_by_subflake(coords, generation)
    subflake_idxs = [reordered[np.arange(len(reordered)//7*i,len(reordered)//7*(i+1))] for i in range(7)]
    
    sorted_idxs = np.concatenate([sfidx[sort_by_x_then_y(coords[:, sfidx])] for sfidx in subflake_idxs])
    return sorted_idxs


if __name__ == "__main__":
    generation = 3
    arr = np.where(compute_hexaflake(generation))
    y, x = arr
    coords = np.empty((2, x.size))
    coords[0, :] = x
    coords[1, :] = y

    sorted_idxs = sort_hexaflake_by_subflake(coords, generation)
    plot_tiered_color(7*7, coords[:, sorted_idxs])
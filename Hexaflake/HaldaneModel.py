import numpy as np
import scipy.linalg as spla
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, issparse
import matplotlib.pyplot as plt

_NN_OFFSETS = np.array(
    [[1, 1], [1, -1], [-2, 0], [2, 0], [-1, 1], [-1, -1]], dtype=np.int64
)

_NNN_OFFSETS = np.array(
    [[0, 2], [3, 1], [3, -1], [0, -2], [-3, -1], [-3, 1]], dtype=np.int64
)


def compute_hexagon(n):
    """
    Construct a boolean 2D array that represents a large hexagonal honeycomb lattice composed of.
    The size of the array is determined by the parameter n, which influences the dimensions.

    Args:
        n (int): An integer controlling the size of the resulting hexagon.

    Returns:
        np.ndarray: A 2D boolean array where True indicates the presence of
            a lattice site in the hexagon, and False indicates empty space.
    """

    # Create end_piece array which forms the left and right edges of the lattice.
    end_piece = np.full((3 ** (n + 1), 3 * ((3 ** n - 1) // 2)), False, dtype=bool)
    for i in range((3 ** n - 1) // 2):
        start = (3 ** (n + 1) - 1) // 2 - 3 * i
        indices_1 = start + 2 * np.arange(3 * i + 1)
        indices_2 = start - 1 + 2 * np.arange(3 * i + 2)
        end_piece[indices_1, 3 * i] = True
        end_piece[indices_2, 3 * i + 1] = True

    # Create a repeated column of unit cell width to fill the middle section of the lattice.
    column = np.full((3 ** (n + 1), 6), False, dtype=bool)
    column[1::2, (0, 4)] = 1
    column[::2, (1, 3)] = 1

    # Repeat column to fill horizontal space, minus a strip at the end.
    middle = np.tile(column, (1, (3 ** n + 1) // 2))[:, :-1]

    # Combine the end pieces and the middle portion horizontally.
    hexagon_array = np.hstack((end_piece, middle, np.fliplr(end_piece)))

    return hexagon_array


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


def _build_coord_index(x_discrete: np.ndarray,
                       y_discrete: np.ndarray) -> dict:
    """Return {(x, y): site_index} for fast O(1) look-ups."""
    return {(int(x), int(y)): idx
            for idx, (x, y) in enumerate(zip(x_discrete, y_discrete))}


def compute_hopping(x_discrete: np.ndarray, y_discrete: np.ndarray, pbc: bool = True):
    # Each site has 3 (but there are 6 total displacements) nearest-neighbors and 6 next-nearest-neighbors
    # We construct arrays of size (N, 3) and (N, 6) respectively
    # Whose value is the connected site

    coord_index = _build_coord_index(x_discrete, y_discrete)
    N = x_discrete.size
    NN_idxs, NNN_idxs, NNN_CCW_idxs = [], [], []

    for i, (xi, yi) in enumerate(zip(x_discrete, y_discrete)):
        nn_sites = []
        for dx, dy in _NN_OFFSETS:
            j = coord_index.get((xi + dx, yi + dy))
            nn_sites.append(j)
        NN_idxs.append(nn_sites)
    
    for i, (xi, yi) in enumerate(zip(x_discrete, y_discrete)):
        nnn_sites = []
        nnn_ccw_sites = []
        for k, (dx, dy) in enumerate(_NNN_OFFSETS):
            j = coord_index.get((xi + dx, yi + dy))
            nnn_sites.append(j)

            if (i + k) % 2 == 1:
                nnn_ccw_sites.append(j)
            else:
                nnn_ccw_sites.append(None)
        NNN_idxs.append(nnn_sites)
        NNN_CCW_idxs.append(nnn_ccw_sites)

    if not pbc:
        return np.array(NN_idxs), np.array(NNN_idxs), np.array(NNN_CCW_idxs)
 
    a = round(np.sqrt(2 * N - 3))
    b = (a + 3) // 2
    c = (a - 3) // 2
    d = 2 * a - b
    e = 2 * a - c
    shifts = np.array([
        [-3, a], [d, b], [e, -c], [3, -a], [-d, -b], [-e, c]
    ])


    Cs = [_build_coord_index(x_discrete + s[0], y_discrete + s[1]) for s in shifts]

    for i, (xi, yi) in enumerate(zip(x_discrete, y_discrete)):
        for k, (dx, dy) in enumerate(_NN_OFFSETS):
            js = np.array([ci.get((xi + dx, yi + dy)) for ci in Cs])
            valid_js = np.array(js) != None
            if any(valid_js):
                NN_idxs[i][k] = js[valid_js][0]
            
    for i, (xi, yi) in enumerate(zip(x_discrete, y_discrete)):
        for k, (dx, dy) in enumerate(_NNN_OFFSETS):
            js = np.array([ci.get((xi + dx, yi + dy)) for ci in Cs])
            valid_js = np.array(js) != None
            if any(valid_js):
                NNN_idxs[i][k] =  js[valid_js][0]
                if (i + k) % 2 == 1:
                    NNN_CCW_idxs[i][k] =  js[valid_js][0]

    

    return np.array(NN_idxs), np.array(NNN_idxs), np.array(NNN_CCW_idxs)


def compute_geometric_data(n, PBC):
    hexagon_array = compute_hexagon(n)
    y_discrete, x_discrete = np.where(hexagon_array)

    sublattice_array = np.zeros_like(hexagon_array)
    sublattice_array[:, ::3] = hexagon_array[:, ::3] #one of the sublattices
    sublattice = ~sublattice_array[y_discrete, x_discrete] #opposite of former
    
    stagger_sublattices = np.empty(sublattice.size, dtype=np.int64)
    stagger_sublattices[::2] = np.sort(np.where(sublattice)[0])
    stagger_sublattices[1::2] = np.sort(np.where(~sublattice)[0])

    x_discrete, y_discrete = x_discrete[stagger_sublattices], y_discrete[stagger_sublattices]
    hexaflake_array = compute_hexaflake(n)
    hexaflake = hexaflake_array[y_discrete, x_discrete]

    NN, _, NNN_CCW = compute_hopping(x_discrete, y_discrete, PBC)

    x = (1 / 2) * (x_discrete - 3 ** (n + 1) + 1)
    y = (np.sqrt(3) / 4) * (2 * y_discrete - 3 ** (n + 1) + 1)

    geometric_data = {
        'x': x,
        'y': y,
        'hexaflake': hexaflake,
        'NN': NN,
        'NNN_CCW': NNN_CCW,
        'x_discrete': x_discrete,
        'y_discrete': y_discrete
    }
    return geometric_data


def compute_disorder_array(strength, system_size, degrees_of_freedom=1):
    """
    Generate a disorder array for the Hamiltonian.

    Parameters:
    strength (float): The strength of the disorder.
    system_size (int): The size of the system.
    degrees_of_freedom (int): Degrees of freedom

    Returns:
    np.ndarray: A diagonal matrix representing the disorder.
    """
    disorder_array = np.random.uniform(-strength/2, strength/2, size=system_size)
    delta = np.sum(disorder_array)/system_size
    disorder_array -= delta
    disorder_array = np.repeat(disorder_array, degrees_of_freedom)
    return np.diag(disorder_array).astype(np.complex128)


def compute_hamiltonian(method, M, phi, t1, t2, geometric_data, disorder_strength=0.0, disorderBeforeRenorm:bool = False):

    valid_methods = ['hexagon', 'site_elim', 'renorm1', 'renorm2']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Options are {valid_methods}.")

    NN = geometric_data['NN']
    NNN_CCW = geometric_data['NNN_CCW']
    hexaflake = geometric_data['hexaflake']

    N = NN.shape[0]

    H = np.zeros((N, N), dtype=np.complex128)
    for i in np.arange(N):
        nnn_ccws = NNN_CCW[i]
        H[i, nnn_ccws[nnn_ccws != None].astype(int)] = -t2 * np.sin(phi) * 1j
    H += H.conj().T

    for i in np.arange(N):
        nns = NN[i]
        H[i, nns[nns != None].astype(int)] = -t1

    np.fill_diagonal(H, M * ((-1)**(np.arange(H.shape[0]))))

    if disorder_strength != 0.0 and disorderBeforeRenorm:
        disorder_array = compute_disorder_array(disorder_strength, H.shape[0])
        H += disorder_array

    if 'renorm' in method:
        H_aa = H[np.ix_(hexaflake, hexaflake)]
        H_bb = H[np.ix_(~hexaflake, ~hexaflake)]
        H_ab = H[np.ix_(hexaflake, ~hexaflake)]
        H_ba = H[np.ix_(~hexaflake, hexaflake)]

        H = H_aa - H_ab @ spla.solve(H_bb,H_ba,assume_a='her',check_finite=False,overwrite_a=True,overwrite_b=True)

    elif method == 'site_elim':
        H = H[np.ix_(hexaflake, hexaflake)]

    if disorder_strength != 0.0 and not disorderBeforeRenorm:
        disorder_array = compute_disorder_array(disorder_strength, H.shape[0])
        H += disorder_array

    return H


def compute_sparse_hamiltonian(method, M, phi, t1, t2, geometric_data, disorder_strength=0.0, disorderBeforeRenrom:bool = False):
    valid_methods = ['hexagon', 'site_elim']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Options are {valid_methods}.")

    NN = geometric_data['NN']
    NNN_CCW = geometric_data['NNN_CCW']
    hexaflake = geometric_data['hexaflake']

    N = NN.shape[0]

    data = []
    row_idxs = []
    column_idxs = []
    for i in range(N):
        for j in NN[i]:
            if j != None:
                row_idxs.append(i)
                column_idxs.append(j)
                data.append(-t1)

        for j in NNN_CCW[i]:
            if j != None:
                row_idxs.append(i)
                column_idxs.append(j)
                data.append(-t2 * np.sin(phi) * 1j)

                row_idxs.append(j)
                column_idxs.append(i)
                data.append(t2 * np.sin(phi) * 1j)

    row_idxs = np.array(row_idxs)
    column_idxs = np.array(column_idxs)
    data = np.array(data)

    if disorder_strength != 0.0 and disorderBeforeRenrom:
        disorder_array = compute_disorder_array(disorder_strength, N).flatten()
        onsite_potential = disorder_array + M * (-1) ** np.arange(N)
    else:
        onsite_potential = M * (-1) ** np.arange(N)
    row_idxs = np.concatenate((row_idxs, np.arange(N)))
    column_idxs = np.concatenate((column_idxs, np.arange(N)))
    data = np.concatenate((data, onsite_potential))

    H = coo_matrix((data, (row_idxs, column_idxs)), shape=(N, N), dtype=np.complex128).tocsr()

    if method == 'site_elim':
        H = H[hexaflake, :][:, hexaflake]

    return H


def triangular_basis(x, y):

    a1 = (np.sqrt(3) / 2) * x - 0.5 * y
    a2 = (np.sqrt(3) / 2) * x + 0.5 * y

    return a1, a2


def compute_bott_index(eigen_data):

    eigenvalues, eigenvectors, x, y, S = [eigen_data[key] for key in 'eigenvalues, eigenvectors, x, y, S'.split(', ')]
    lower_band = np.argsort(eigenvalues)[:eigenvalues.size // 2]
    V = eigenvectors[:, lower_band]

    N = round((np.sqrt(2 * S - 3) - 3) / 2 + 2)
    L = np.sqrt(3) * N

    a1, a2 = triangular_basis(x, y)

    U1 = np.exp(1j * 2 * np.pi * a1 / L)[:, np.newaxis]
    U2 = np.exp(1j * 2 * np.pi * a2 / L)[:, np.newaxis]

    U1_proj = V.conj().T @ (V * U1)
    U2_proj = V.conj().T @ (V * U2)

    A = U2_proj @ U1_proj @ U2_proj.conj().T @ U1_proj.conj().T

    eigenvaluesA = spla.eigvals(A, overwrite_a=True)
    trace_logA = np.sum(np.log(eigenvaluesA))

    bott = np.imag(trace_logA) / (2 * np.pi)

    return bott


def compute_bott_from_hamiltonian(H, method, geometry_data):
    x, y = geometry_data['x'], geometry_data['y']
    if issparse(H):
        H = H.toarray()
    eigenvalues, eigenvectors = spla.eigh(H, overwrite_a=True)

    if method in ['site_elim', 'renorm1', 'renorm2']:
        hexaflake = geometry_data['hexaflake']
        x, y = x[hexaflake], y[hexaflake]
    return compute_bott_index({'x':x, 'y':y, 'eigenvalues':eigenvalues, 'eigenvectors':eigenvectors, 'S':geometry_data['x'].size})



def main():
    gd = compute_geometric_data(2, True)
    H = compute_sparse_hamiltonian('hexagon', 0., np.pi, 1., 1., gd)



if __name__ == '__main__':
    main()
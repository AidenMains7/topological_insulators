import numpy as np
import scipy as sp
from scipy.sparse import coo_array, issparse


# ---------------------------------------------------------------------------
# Lattice offsets (in discrete (x, y) coordinates)
# ---------------------------------------------------------------------------
 
# 6 nearest-neighbour displacements
_NN_OFFSETS = np.array(
    [[2, 0], [-2, 0], [1, 1], [-1, 1], [1, -1], [-1, -1]], dtype=np.int64
)
 
# 6 next-nearest-neighbour displacements
_NNN_OFFSETS = np.array(
    [[0, 2], [0, -2], [3, 1], [-3, 1], [3, -1], [-3, -1]], dtype=np.int64
)
 
# Counter-clockwise NNN direction signature:
# Each row is [sign(x_i→k), sign(y_i→k), sign(x_k→j), sign(y_k→j)]
_CCW_DIRECTIONS = np.array(
    [[1, -1, 1, 0], [1, 1, -1, 1], [-1, 0, -1, -1],
     [-1, 1, -1, 0], [-1, -1, 1, -1], [1, 0, 1, 1]],
    dtype=np.int8,
)
 
 
# ---------------------------------------------------------------------------
# Lattice construction
# ---------------------------------------------------------------------------
 
def compute_hexagon(n: int) -> np.ndarray:
    """
    Construct the boolean 2-D array for a hexagonal honeycomb lattice of
    order n.  Identical output to the original; minor micro-optimisations
    only.
    """
    rows = 3 ** (n + 1)
    half = (3 ** n - 1) // 2          # number of 'end' column-pairs
 
    end_piece = np.zeros((rows, 3 * half), dtype=bool)
 
    for i in range(half):
        start = (rows - 1) // 2 - 3 * i
        end_piece[start + 2 * np.arange(3 * i + 1),     3 * i]     = True
        end_piece[start - 1 + 2 * np.arange(3 * i + 2), 3 * i + 1] = True
 
    # Unit-cell column (period 6 in x)
    column = np.zeros((rows, 6), dtype=bool)
    column[1::2, [0, 4]] = True
    column[0::2, [1, 3]] = True
 
    middle = np.tile(column, (1, (3 ** n + 1) // 2))[:, :-1]
 
    return np.hstack((end_piece, middle, np.fliplr(end_piece)))
 
 
def compute_hexaflake(n: int) -> np.ndarray:
    """
    Construct the boolean 2-D hexaflake array of order n.
 
    Vectorised replacement for the original Python loop: all 6 offsets at
    each recursion level are applied in a single broadcast operation, cutting
    Python-loop overhead from O(6^n) iterations to O(n) iterations.
    """
    directions = np.array(
        [[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]], dtype=np.int64
    )
 
    coords = directions.copy()                          # shape (6, 2)
 
    for scale in 3 ** np.arange(1, n + 1):
        offsets = scale * directions                    # (6, 2)
        # Broadcast: (1, M, 2) + (7, 1, 2)  →  (7, M, 2)
        all_offsets = np.concatenate([[[0, 0]], offsets])   # (7, 2)
        coords = (coords[np.newaxis] + all_offsets[:, np.newaxis]).reshape(-1, 2)
 
    x_d, y_d = coords.T
    x_d += 3 ** (n + 1) - 1
    y_d += (3 ** (n + 1) - 1) // 2
 
    arr = np.zeros((3 ** (n + 1), 2 * 3 ** (n + 1) - 1), dtype=bool)
    arr[y_d, x_d] = True
    return arr
 
 
# ---------------------------------------------------------------------------
# O(N) neighbour finding via coordinate hash-map
# ---------------------------------------------------------------------------
 
def _build_coord_index(x_discrete: np.ndarray,
                       y_discrete: np.ndarray) -> dict:
    """Return {(x, y): site_index} for fast O(1) look-ups."""
    return {(int(x), int(y)): idx
            for idx, (x, y) in enumerate(zip(x_discrete, y_discrete))}
 
 
def compute_hopping_arrays_fast(
    x_discrete: np.ndarray,
    y_discrete: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    O(N) replacement for compute_hopping_arrays.
    Returns
    -------
    NN      : (N, N) bool  — nearest-neighbour connectivity
    NNN_CCW : (N, N) bool  — CCW next-nearest-neighbour connectivity
    """
    coord_index = _build_coord_index(x_discrete, y_discrete)
    N = x_discrete.size
 
    nn_rows, nn_cols = [], []
    nnn_rows, nnn_cols, nnn_ccw_flags = [], [], []
 
    # Pre-compute a NN adjacency list (needed to determine CCW for NNN)
    nn_neighbours: list[set] = [set() for _ in range(N)]
 
    for i, (xi, yi) in enumerate(zip(x_discrete, y_discrete)):
        for dx, dy in _NN_OFFSETS:
            j = coord_index.get((int(xi + dx), int(yi + dy)))
            if j is not None:
                nn_rows.append(i)
                nn_cols.append(j)
                nn_neighbours[i].add(j)
 
    # NNN: iterate over NNN offsets; determine CCW via shared NN intermediate
    for i, (xi, yi) in enumerate(zip(x_discrete, y_discrete)):
        for dx, dy in _NNN_OFFSETS:
            j = coord_index.get((int(xi + dx), int(yi + dy)))
            if j is None:
                continue
 
            # Find the intermediate site k: NN of both i and j
            common = nn_neighbours[i] & nn_neighbours[j]
            if not common:
                continue
            k = next(iter(common))          # there is exactly one such site
 
            xk, yk = x_discrete[k], y_discrete[k]
            sig = np.array([
                np.sign(xk - xi), np.sign(yk - yi),
                np.sign(x_discrete[j] - xk), np.sign(y_discrete[j] - yk),
            ], dtype=np.int8)
 
            ccw = bool(np.any(np.all(_CCW_DIRECTIONS == sig[np.newaxis], axis=1)))
 
            nnn_rows.append(i)
            nnn_cols.append(j)
            nnn_ccw_flags.append(ccw)
 
    # Assemble dense arrays (feasible for moderate N; see note above)
    NN = np.zeros((N, N), dtype=bool)
    if nn_rows:
        NN[nn_rows, nn_cols] = True
 
    NNN_CCW = np.zeros((N, N), dtype=bool)
    if nnn_rows:
        ccw_arr = np.array(nnn_ccw_flags, dtype=bool)
        r = np.array(nnn_rows)
        c = np.array(nnn_cols)
        NNN_CCW[r[ccw_arr], c[ccw_arr]] = True
 
    return NN, NNN_CCW
 
 
def compute_hopping_sparse(
    x_discrete: np.ndarray,
    y_discrete: np.ndarray,
):
    """
    Like compute_hopping_arrays_fast but returns scipy.sparse.csr_array
    objects.  Much more memory-efficient for large N (N > ~5 000).
 
    Returns
    -------
    NN      : scipy.sparse.csr_array, shape (N, N), dtype bool
    NNN_CCW : scipy.sparse.csr_array, shape (N, N), dtype bool
    """
    from scipy.sparse import csr_array
 
    coord_index = _build_coord_index(x_discrete, y_discrete)
    N = x_discrete.size
 
    nn_rows, nn_cols = [], []
    nnn_rows, nnn_cols, nnn_ccw_flags = [], [], []
    nn_neighbours: list[set] = [set() for _ in range(N)]
 
    for i, (xi, yi) in enumerate(zip(x_discrete, y_discrete)):
        for dx, dy in _NN_OFFSETS:
            j = coord_index.get((int(xi + dx), int(yi + dy)))
            if j is not None:
                nn_rows.append(i)
                nn_cols.append(j)
                nn_neighbours[i].add(j)
 
    for i, (xi, yi) in enumerate(zip(x_discrete, y_discrete)):
        for dx, dy in _NNN_OFFSETS:
            j = coord_index.get((int(xi + dx), int(yi + dy)))
            if j is None:
                continue
            common = nn_neighbours[i] & nn_neighbours[j]
            if not common:
                continue
            k = next(iter(common))
            xk, yk = x_discrete[k], y_discrete[k]
            sig = np.array([
                np.sign(xk - xi), np.sign(yk - yi),
                np.sign(x_discrete[j] - xk), np.sign(y_discrete[j] - yk),
            ], dtype=np.int8)
            ccw = bool(np.any(np.all(_CCW_DIRECTIONS == sig[np.newaxis], axis=1)))
            nnn_rows.append(i)
            nnn_cols.append(j)
            nnn_ccw_flags.append(ccw)
 
    data_nn = np.ones(len(nn_rows), dtype=bool)
    NN = csr_array(
        (data_nn, (nn_rows, nn_cols)), shape=(N, N), dtype=bool
    )
 
    ccw_arr = np.array(nnn_ccw_flags, dtype=bool)
    r, c = np.array(nnn_rows), np.array(nnn_cols)
    data_nnn = np.ones(ccw_arr.sum(), dtype=bool)
    NNN_CCW = csr_array(
        (data_nnn, (r[ccw_arr], c[ccw_arr])), shape=(N, N), dtype=bool
    )
 
    return NN, NNN_CCW



    



def compute_dx_and_dy_discrete(
    x_discrete: np.ndarray,
    y_discrete: np.ndarray,
    PBC: bool,
    chunk_size: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise discrete differences with optional PBC.
 
    Memory-efficient chunk-wise implementation (unchanged from the previous
    refactor).  Only call this when the full N×N matrix is truly needed; for
    hopping connectivity use compute_hopping_arrays_fast or
    compute_hopping_sparse instead.
    """
    N = x_discrete.size
 
    if not PBC:
        delta_x = (x_discrete[np.newaxis, :] - x_discrete[:, np.newaxis]).astype(np.int64)
        delta_y = (y_discrete[np.newaxis, :] - y_discrete[:, np.newaxis]).astype(np.int64)
        return delta_x, delta_y
 
    a = round(np.sqrt(2 * N - 3))
    b, c = (a + 3) // 2, (a - 3) // 2
    d, e = 2 * a - b, 2 * a - c
    shifts = np.array(
        [[0, 0], [-3, a], [3, -a], [d, b], [-d, -b], [-e, c], [e, -c]],
        dtype=np.int64,
    )
 
    C1, C2 = 0.25, 0.75   # (1/2)² and (√3/2)²
 
    dx_final = np.empty((N, N), dtype=np.int64)
    dy_final = np.empty((N, N), dtype=np.int64)
 
    for i in range(0, N, chunk_size):
        sl = slice(i, min(i + chunk_size, N))
        xi = x_discrete[sl, np.newaxis]
        yi = y_discrete[sl, np.newaxis]
 
        dx = x_discrete[np.newaxis, :] - xi      # (chunk, N)
        dy = y_discrete[np.newaxis, :] - yi
 
        min_d2 = C1 * dx ** 2 + C2 * dy ** 2
 
        for sx, sy in shifts[1:]:
            cdx, cdy = dx - sx, dy - sy
            d2 = C1 * cdx ** 2 + C2 * cdy ** 2
            mask = d2 < min_d2
            min_d2[mask] = d2[mask]
            dx[mask] = cdx[mask]
            dy[mask] = cdy[mask]
 
        dx_final[sl, :] = dx
        dy_final[sl, :] = dy
 
    return dx_final, dy_final
 
 
def compute_hopping_arrays(
    delta_x_discrete: np.ndarray,
    delta_y_discrete: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Original O(N²) implementation — use only when delta arrays already exist."""
    NN = (
        ((np.abs(delta_x_discrete) == 2) & (delta_y_discrete == 0)) |
        ((np.abs(delta_x_discrete) == 1) & (np.abs(delta_y_discrete) == 1))
    )
    NNN = (
        ((delta_x_discrete == 0) & (np.abs(delta_y_discrete) == 2)) |
        ((np.abs(delta_x_discrete) == 3) & (np.abs(delta_y_discrete) == 1))
    )
 
    i, j = np.where(NNN)
    k = np.argmax(NN[i] & NN[j], axis=1)
 
    x_ik = delta_x_discrete[i, k];  y_ik = delta_y_discrete[i, k]
    x_kj = delta_x_discrete[k, j];  y_kj = delta_y_discrete[k, j]
 
    dirs = np.sign(np.stack([x_ik, y_ik, x_kj, y_kj], axis=1)).astype(np.int8)
    ccw = np.any(np.all(_CCW_DIRECTIONS[None] == dirs[:, None], axis=2), axis=1)
 
    NNN_CCW = np.zeros_like(NNN)
    NNN_CCW[i[ccw], j[ccw]] = True
 
    return NN, NNN_CCW
 

def compute_geometric_data(
    n: int,
    PBC: bool,
    return_dx_dy: bool = False,
    sublatticeMethod: str = 'hexaflake',
    print_info: bool = False,
    sparse: bool | None = None,
) -> dict:
    """
    Compute all geometric data for a hexagonal honeycomb lattice of order n.
    """
    hexagon_array = compute_hexagon(n)
    y_discrete, x_discrete = np.where(hexagon_array)
 
    sublattice_array = np.zeros_like(hexagon_array)
    sublattice_array[:, ::3] = hexagon_array[:, ::3]
    sublattice = ~sublattice_array[y_discrete, x_discrete]
 
    stagger = np.empty(sublattice.size, dtype=np.int64)
    stagger[::2]  = np.sort(np.where(sublattice)[0])
    stagger[1::2] = np.sort(np.where(~sublattice)[0])
    x_discrete = x_discrete[stagger]
    y_discrete = y_discrete[stagger]
 
    hexaflake_array = compute_hexaflake(n)
    hexaflake = hexaflake_array[y_discrete, x_discrete]
 
    N = x_discrete.size
    use_sparse = (N >= 50000) if sparse is None else sparse
 
    if print_info:
        print(f"n={n}: N={N} sites, {'sparse' if use_sparse else 'dense'} hopping arrays")
 
    # --- Hopping arrays via O(N) hash-map method ---
    if use_sparse:
        NN, NNN_CCW = compute_hopping_sparse(x_discrete, y_discrete)
    else:
        NN, NNN_CCW = compute_hopping_arrays_fast(x_discrete, y_discrete)
 
    # Physical coordinates
    x = 0.5 * (x_discrete - 3 ** (n + 1) + 1)
    y = (np.sqrt(3) / 4) * (2 * y_discrete - 3 ** (n + 1) + 1)
 
    geometric_data = {
        'x': x,
        'y': y,
        'hexaflake': hexaflake,
        'NN': NN,
        'NNN_CCW': NNN_CCW,
        'x_discrete': x_discrete,
        'y_discrete': y_discrete,
    }
 
    if return_dx_dy:
        dx, dy = compute_dx_and_dy_discrete(x_discrete, y_discrete, PBC)
        geometric_data['delta_x_discrete'] = dx
        geometric_data['delta_y_discrete'] = dy
 
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

    N = len(geometric_data['x'])
    NN = geometric_data['NN']
    NNN_CCW = geometric_data['NNN_CCW']
    hexaflake = geometric_data['hexaflake']

    H = np.zeros(NN.shape, dtype=np.complex128)
    np.fill_diagonal(H, M*((-1)**(np.arange(H.shape[0]))))
    if issparse(NN) and issparse(NNN_CCW):
        rows = np.arange(N)[:, None]
        mask_nn = NN >= 0
        H[rows[mask_nn], NN[mask_nn]] = -t1

    else:
        H[NN] = -t1
        H[NNN_CCW] = -t2 * np.sin(phi)*1j
        H[NNN_CCW.T] = t2 * np.sin(phi)*1j

    if disorder_strength != 0.0 and disorderBeforeRenorm:
        disorder_array = compute_disorder_array(disorder_strength, H.shape[0])
        H += disorder_array

    if method == 'renorm':
        H_aa = H[np.ix_(hexaflake, hexaflake)]
        H_bb = H[np.ix_(~hexaflake, ~hexaflake)]
        H_ab = H[np.ix_(hexaflake, ~hexaflake)]
        H_ba = H[np.ix_(~hexaflake, hexaflake)]

        H = H_aa - H_ab @ sp.linalg.solve(H_bb,H_ba,assume_a='her',check_finite=False,overwrite_a=True,overwrite_b=True)

    elif method == 'site_elim':
        H = H[np.ix_(hexaflake, hexaflake)]

    if disorder_strength != 0.0 and not disorderBeforeRenorm:
        disorder_array = compute_disorder_array(disorder_strength, H.shape[0])
        H += disorder_array

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

    eigenvaluesA = sp.linalg.eigvals(A, overwrite_a=True)
    trace_logA = np.sum(np.log(eigenvaluesA))

    bott = np.imag(trace_logA) / (2 * np.pi)
    return bott



def main():
    pass


if __name__ == '__main__':
    main()


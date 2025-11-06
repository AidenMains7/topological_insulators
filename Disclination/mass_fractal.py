import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh, eigvals
import scipy.sparse as spsp
from itertools import product
from joblib import Parallel, delayed
from tqdm_joblib import tqdm, tqdm_joblib
import os, cProfile, pstats, time, h5py
from matplotlib import colors

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)
        return result
    return wrapper

def mean_time(func, n_iterations: int = 100, warmup: int = 0, *args, **kwargs):
    """
    Measure the execution time of `func(*args, **kwargs)` over `n_iterations`.
    Optional `warmup` runs are executed first and not timed.
    Returns a dict with mean, median, std and the raw times array.
    """
    # Warmup runs (not measured)
    for _ in range(int(warmup)):
        func(*args, **kwargs)

    times = []
    for _ in range(int(n_iterations)):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times, dtype=float)
    return {
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "std": float(times.std(ddof=0)),
        "times": times
    }


# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# region Geometry


def generate_sierpinski_lattice(generation:int, pad_width:int, excludeCenter:bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (generation < 0):
        raise ValueError("Order of lattice must be >= 0.")

    def _sierpinski_carpet(gen_:int):
        if(gen_ == 0):
            return np.array([1], dtype=int)
    
        #carpet of one lower degree; recursion
        carpet_lower = _sierpinski_carpet(gen_-1)

        #concatenate to make current degree
        top = np.hstack((carpet_lower,carpet_lower,carpet_lower))
        mid = np.hstack((carpet_lower,carpet_lower*0,carpet_lower))
        carpet = np.vstack((top,mid,top))

        return carpet
    
    L = 3**generation

    square_lattice = np.arange(L*L).reshape((L,L))
    carpet_structure = _sierpinski_carpet(generation)

    if (pad_width > 0):
        carpet_structure = np.pad(carpet_structure,pad_width,mode='constant',constant_values=1)

    flattened_lattice = carpet_structure.flatten()
    hole_indices = np.where(flattened_lattice == 0)[0]
    fill_indices = np.flatnonzero(flattened_lattice)

    fractal_lattice = np.full(flattened_lattice.shape, -1, dtype=int)
    fractal_lattice[fill_indices] = np.arange(fill_indices.size)
    fractal_lattice = fractal_lattice.reshape(carpet_structure.shape)

    if excludeCenter:
        center = (L // 2, L // 2)
        radius = (L // 3 - 1) // 2
        center_indices = square_lattice[center[0] - radius : center[0] + radius + 1, 
                                        center[1] - radius : center[1] + radius + 1].flatten()
        
        mask = np.ones(L*L, dtype=bool)
        mask[center_indices] = False
    else:
        mask = np.ones(L*L, dtype=bool)


    return {"lattice": square_lattice, 
            "hole_indices": hole_indices, 
            "fill_indices": fill_indices,
            "center_mask": mask}


def calculate_square_hopping(lattice: np.ndarray, pbc: bool) -> dict:
    Ly, Lx = lattice.shape
    N = Lx * Ly
    
    # Data for building sparse matrices (row_ind, col_ind, value)
    Cx_data, Sx_data = [], []
    Cy_data, Sy_data = [], []

    for r in range(Ly):
        for c in range(Lx):
            i = r * Lx + c # Current site index

            # Neighbor to the right (X+1)
            c_right = (c + 1) % Lx if pbc else c + 1
            if c_right < Lx:
                j_right = r * Lx + c_right
                # C_x = 0.5 * (T_x + T_x^\dagger)
                Cx_data.append((i, j_right, 0.5))
                Cx_data.append((j_right, i, 0.5))
                # S_x = 0.5j * (T_x - T_x^\dagger)
                Sx_data.append((i, j_right, 0.5j))
                Sx_data.append((j_right, i, -0.5j))

            # Neighbor below (Y+1)
            r_down = (r + 1) % Ly if pbc else r + 1
            if r_down < Ly:
                j_down = r_down * Lx + c
                # C_y = 0.5 * (T_y + T_y^\dagger)
                Cy_data.append((i, j_down, 0.5))
                Cy_data.append((j_down, i, 0.5))
                # S_y = 0.5j * (T_y - T_y^\dagger)
                Sy_data.append((i, j_down, 0.5j))
                Sy_data.append((j_down, i, -0.5j))

    # Create sparse matrices from the collected data
    def to_coo(data, shape):
        if not data: return spsp.coo_matrix(shape, dtype=complex)
        rows, cols, vals = zip(*data)
        return spsp.coo_matrix((vals, (rows, cols)), shape=shape)

    Cx = to_coo(Cx_data, (N, N)).tocsr()
    Sx = to_coo(Sx_data, (N, N)).tocsr()
    Cy = to_coo(Cy_data, (N, N)).tocsr()
    Sy = to_coo(Sy_data, (N, N)).tocsr()
    I = spsp.eye(N, dtype=complex, format='csr')

    # NNN terms can be built from NN terms
    CxCy = Cx @ Cy
    CySx = Cy @ Sx
    CxSy = Cx @ Sy

    wannier_dict = {
        'Cx': Cx, 'Cy': Cy, 'Sx': Sx, 'Sy': Sy,
        'CxCy': CxCy, 'CySx': CySx, 'CxSy': CxSy, 'I': I
    }
    return wannier_dict


def old_calculate_square_hopping(lattice, pbc):
    y, x = np.where(lattice >= 0)[:]

    side_length = lattice.shape[0]
    
    dx = x - x[:, None]
    dy = y - y[:, None]
    if pbc:
        multipliers = tuple(product([-1, 0, 1], repeat=2))
        shifts = [(i * side_length, j * side_length) for i, j in multipliers]

        x_shifted = np.empty((dx.shape[0], dx.shape[1], len(shifts)), dtype=dx.dtype)
        y_shifted = np.empty((dy.shape[0], dy.shape[1], len(shifts)), dtype=dy.dtype)
        for i, (dx_shift, dy_shift) in enumerate(shifts):
            x_shifted[:, :, i] = dx + dx_shift
            y_shifted[:, :, i] = dy + dy_shift

        distances = x_shifted**2 + y_shifted**2
        minimal_hop = np.argmin(distances, axis = -1)
        i_idxs, j_idxs = np.indices(minimal_hop.shape)

        dx = x_shifted[i_idxs, j_idxs, minimal_hop]
        dy = y_shifted[i_idxs, j_idxs, minimal_hop]


    xp_mask = (dx == 1) & (dy == 0)
    yp_mask = (dx == 0) & (dy == 1)

    xpyp_mask = (dx == 1) & (dy == 1)
    xnyp_mask = (dx == -1) & (dy == 1)

    Cx =   spsp.dok_matrix((side_length**2, side_length**2), dtype=complex)
    Sx =   spsp.dok_matrix((side_length**2, side_length**2), dtype=complex)
    Cy =   spsp.dok_matrix((side_length**2, side_length**2), dtype=complex)
    Sy =   spsp.dok_matrix((side_length**2, side_length**2), dtype=complex)
    CxCy = spsp.dok_matrix((side_length**2, side_length**2), dtype=complex)
    CySx = spsp.dok_matrix((side_length**2, side_length**2), dtype=complex)
    CxSy = spsp.dok_matrix((side_length**2, side_length**2), dtype=complex)
    I =    np.eye(side_length**2, dtype=complex)

    Sx[xp_mask] = 1j / 2
    Cx[xp_mask] = 1 / 2
    Cy[yp_mask] = 1 / 2
    Sy[yp_mask] = 1j / 2

    CxCy[xpyp_mask] = 1 / 4
    CxCy[xnyp_mask] = 1 / 4

    CxSy[xpyp_mask] = 1j / 4
    CxSy[xnyp_mask] = 1j / 4

    CySx[xpyp_mask] = 1j / 4
    CySx[xnyp_mask] = -1j / 4

    wannier_dict = {
        'Cx': Cx,
        'Cy': Cy,
        'Sx': Sx,
        'Sy': Sy,
        'CxCy': CxCy,
        'CySx': CySx,
        'CxSy': CxSy
    }

    wannier_dict = {k: (v + v.conj().T).tocsr() for k, v in wannier_dict.items()}
    wannier_dict["I"] = spsp.dok_matrix(I)
    return wannier_dict


def generate_square_lattice(N:int):
    def _find_good_factors(n):
        factors = []
        for i in range(1, int(np.sqrt(n)) + 1):
            if n % i == 0:
                factors.append((i, n // i))
        return factors
    factors = _find_good_factors(N)
    return np.arange(N).reshape(factors[-1])


def generate_lattice_glass(generation:int, n_fractal_sites:int):
    N_total = 9 ** generation
    N_fractal_max = 8 ** generation

    lattice = generate_square_lattice(N_total)
    fills = np.random.choice(np.arange(N_total), size=min(n_fractal_sites, N_fractal_max), replace=False)
    holes = np.setdiff1d(np.arange(N_total), fills)

    return lattice, fills, holes


# endregion
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# region Observables

def calculate_hamiltonian(wannier_matrices:dict, geometry_dict:dict, 
                          M_background:float, M_substitution:float, 
                          t:float = 1.0, t0:float = 1.0):
    pauli1 = spsp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    pauli2 = spsp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
    pauli3 = spsp.csr_matrix([[1, 0], [0, -1]], dtype=complex)

    Cx, Cy, Sx, Sy, CxCy, CySx, CxSy, I = wannier_matrices.values()
    lattice, hole_indices, fill_indices, mask = geometry_dict.values()

    mass_values = np.full(I.shape[0], M_background)
    mass_values[fill_indices] = M_substitution
    onsite_mass = spsp.diags(mass_values, format='csr')

    d1 = t * Sx
    d2 = t * Sy
    d3 = onsite_mass - t0 * (Cx + Cy)
    hamiltonian = (spsp.kron(d1, pauli1) + spsp.kron(d2, pauli2) + spsp.kron(d3, pauli3)).tocsr()

    parity_mask = np.repeat(mask, 2)
    hamiltonian = hamiltonian[parity_mask, :][:, parity_mask]
    return hamiltonian


def calculate_projector(hamiltonian:np.ndarray, useGPU:bool = False, verbose:bool = False):
    def _projector_from_eigendecomp(xp, eigenvalues, eigenvectors):
        highest_lower_band = xp.sort(eigenvalues)[:eigenvalues.size // 2][-1]
        D = xp.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
        D_dagger = eigenvectors.conj().T * D[:, None]
        return eigenvectors @ D_dagger

    start_time = time.time()

    xp = np
    eigh_fn = eigh
    backend = 'cpu'

    if useGPU: 
        try:
            xp = cp
            eigh_fn = cp.linalg.eigh
            backend = 'gpu'
        except (ImportError, ModuleNotFoundError):
            if verbose:
                print("CuPy is not installed. Falling back to CPU computation.")

    if spsp.issparse(hamiltonian):
        hamiltonian = hamiltonian.toarray()
    
    try:  
        H = xp.asarray(hamiltonian)
        eigenvalues, eigenvectors = eigh_fn(H)
        projector = _projector_from_eigendecomp(xp, eigenvalues, eigenvectors)
        if backend == 'gpu':
            projector = xp.asnumpy(projector)
    except Exception as e:
        if backend == 'gpu':
            if verbose:
                print(f"GPU computation failed ({e}). Falling back to CPU computation.")
            xp = np
            eigh_fn = eigh
            H = np.asarray()
            eigenvalues, eigenvectors = eigh_fn(H)
            projector = _projector_from_eigendecomp(xp, eigenvalues, eigenvectors)
        else:
            raise
            
    if verbose:
        elapsed = time.time() - start_time
        print(f"Projector computed using {backend} in {elapsed:.2f} seconds.")

    return projector


def calculate_bott_index(projector:np.ndarray, lattice:np.ndarray, mask:np.ndarray):
    Y, X = np.where(lattice >= 0)[:]
    X = X[mask]
    Y = Y[mask]

    X = np.repeat(X, 2)
    Y = np.repeat(Y, 2)
    Lx, Ly = lattice.shape

    Ux = np.exp(1j * 2 * np.pi * X / Lx)
    Uy = np.exp(1j * 2 * np.pi * Y / Ly)
    UxP = projector * Ux[:, None]
    UyP = projector * Uy[:, None]
    Ux_daggerP = projector * Ux.conj()[:, None]
    Uy_daggerP = projector * Uy.conj()[:, None]

    A = (
        np.eye(projector.shape[0], dtype=np.complex128) 
        - projector 
        + projector @ UxP @ UyP @ Ux_daggerP @ Uy_daggerP
    )
    
    A_eigenvalues = eigvals(A)
    bott_index = np.imag( np.sum( np.log(A_eigenvalues) ) ) / (2 * np.pi)
    return float(round(bott_index, 3))


def compute_LDOS(hamiltonian:np.ndarray, number_of_states:int = 2):
    eigenvalues, eigenvectors = eigh(hamiltonian, overwrite_a=True)
    number_of_states += number_of_states % 2 # Ensure even number of states
    mid_index = len(eigenvalues) // 2
    lower_idxs = np.arange(len(eigenvalues))[:mid_index][-number_of_states // 2:]
    upper_idxs = np.arange(len(eigenvalues))[mid_index:][:number_of_states // 2]
    selected_indices = np.concatenate((lower_idxs, upper_idxs)) # Indices of the selected states to be used in LDOS

    LDOS = np.sum(np.abs(eigenvectors[:, selected_indices])**2, axis=1)
    LDOS = LDOS[0::2] + LDOS[1::2] # Sum the two parities
    LDOS = LDOS / np.sum(LDOS) # Normalize the LDOS
    gap = abs(np.max(eigenvalues[lower_idxs]) - np.min(eigenvalues[upper_idxs]))
    bandwidth = np.max(eigenvalues) - np.min(eigenvalues)

    data_dict = {
        "LDOS": LDOS,
        "eigenvalues": eigenvalues,
        "gap": gap,
        "bandwidth": bandwidth,
        "ldos_idxs": selected_indices
    }
    return data_dict


# endregion
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# region Phase


def compute_phase_data(generation:int, M_background:np.ndarray, M_substitution:np.ndarray, t:float = 1.0, t0:float = 1.0, n_jobs:int = -1, pad_width:int = 0, excludeCenter:bool = False):
    lattice_dict = generate_sierpinski_lattice(generation=generation, pad_width=pad_width, excludeCenter=excludeCenter)
    lattice, mask = lattice_dict["lattice"], lattice_dict["center_mask"]
    wannier_matrices = calculate_square_hopping(lattice, pbc=True)
    parameter_values = tuple(product(M_background, M_substitution))

    def _compute_single_point(M_background_val, M_substitution_val):
        H = calculate_hamiltonian(wannier_matrices, lattice_dict, M_background_val, M_substitution_val, t, t0)
        P = calculate_projector(H)
        bott = calculate_bott_index(P, lattice, mask)
        return (M_background_val, M_substitution_val, bott)

    with tqdm_joblib(tqdm(total=len(parameter_values), desc="Computing Phase Data")) as progress_bar:
        results = np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_single_point)(M_background_val, M_substitution_val) for M_background_val, M_substitution_val in parameter_values))

    phase_data = {
        "M_background": results[:, 0],
        "M_substitution": results[:, 1],
        "Bott": results[:, 2]
    }
    return phase_data


def save_phase_data(phase_data:dict, filename:str, doOverwrite:bool = False, directory:str = "./Disclination/PhaseData/"):
    if not doOverwrite and os.path.exists(os.path.join(directory, filename)):
        print(f"File {filename} already exists. Use doOverwrite=True to overwrite.")
        return os.path.join(directory, filename)
    np.savez(os.path.join(directory, filename), **phase_data)
    return os.path.join(directory, filename)


def open_phase_data(filename:str, directory:str = "./Disclination/PhaseData/") -> dict:
    data = np.load(os.path.join(directory, filename))
    phase_data = {key: data[key] for key in data}
    return phase_data


def plot_phase_data(filename, ax:plt.axes = None, directory:str = "./Disclination/PhaseData/", doSave:bool = True, doShow:bool = False):
    phase_data = open_phase_data(filename)

    M_background_unique = np.unique(phase_data["M_background"])
    M_substitution_unique = np.unique(phase_data["M_substitution"])

    if (len(M_background_unique) == 1) or (len(M_substitution_unique) == 1):
        # 1D scatter plot
        doScat = True
    else:
        bott_data = phase_data["Bott"].reshape((M_background_unique.size, M_substitution_unique.size))
        doScat = False

    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if not doScat:
        im = ax.imshow(bott_data, extent=(M_substitution_unique.min(), M_substitution_unique.max(), M_background_unique.min(), M_background_unique.max()),
                       origin='lower', cmap='viridis', vmin=-1, vmax=1)

        cmap = plt.get_cmap('viridis', 3)
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        im.set_cmap(cmap)
        im.set_norm(norm)
        cbar = fig.colorbar(im, ax=ax, boundaries=bounds, ticks=[-1, 0, 1], spacing='proportional')
        cbar.set_label('Bott Index', labelpad=15, fontsize=12)
        ax.set_xlabel(f'$m^{{\\text{{sub}}}}$', fontsize=16)
        ax.set_ylabel(f'$m^{{\\text{{back}}}}$', fontsize=16, rotation=0, labelpad=20)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
            
        t = np.linspace(xmin, xmax, 101)
        ax.plot(t, t, color='black', linestyle='--', linewidth=0.8, alpha=0.25)
        for v in [-2.5, -1.0, 1.0, 2.5]:
            ax.axvline(v, color='black', linestyle='--', linewidth=0.8, alpha=0.25)
            ax.axhline(v, color='black', linestyle='--', linewidth=0.8, alpha=0.25)

        ax.set_xticks([xmin, -2.5, -1.0, 0.0, 1.0, 2.5, xmax])
        ax.set_yticks([ymin, -2.5, -1.0, 0.0, 1.0, 2.5, ymax])

    else:
        if len(M_substitution_unique) == 1:
            sc = ax.scatter(phase_data["M_background"], phase_data["Bott"], label=f"$M^{{\\text{{sub}}}}={M_substitution_unique[0]}$") 
            ax.set_xlabel('$m^{{\\text{{back}}}}$')     
        else:
            sc = ax.scatter(phase_data["M_substitution"], phase_data["Bott"], label=f"$M^{{\\text{{back}}}}={M_background_unique[0]}$")
            ax.set_xlabel('$m^{{\\text{{sub}}}}$')
        ax.set_ylabel('Bott Index')
        ax.set_yticks([-1, 0, 1])
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.legend()

    plt.tight_layout()
    ax.set_title(filename[:-4], fontsize=20, pad=15)

    if doSave:
        plt.savefig(os.path.join(directory, filename.replace('.npz', '.png')))
    if doShow:
        plt.show()


def compute_glass_phase_data(M_background_values:np.ndarray, M_substitution_values:np.ndarray, n_fractal_sites:int, n_iterations:int, generation:int = 2, n_jobs:int = -1):
    parameter_values = np.array(tuple(product(M_background_values, M_substitution_values)))
    parameter_values = np.repeat(parameter_values, n_iterations, axis=0)
    total_computations = len(M_background_values) * len(M_substitution_values) * n_iterations

    def _compute_single_point(M_background_val, M_substitution_val):
        lattice_dict = generate_lattice_glass(generation=generation, n_fractal_sites=n_fractal_sites)
        lattice, mask = lattice_dict["lattice"], lattice_dict["center_mask"]
        wannier_matrices = calculate_square_hopping(lattice, pbc=True)
        H = calculate_hamiltonian(wannier_matrices, lattice_dict["hole_indices"], lattice_dict["fill_indices"], M_background_val, M_substitution_val)
        P = calculate_projector(H)
        bott = calculate_bott_index(P, lattice, mask)
        return (M_background_val, M_substitution_val, bott)
    
    with tqdm_joblib(tqdm(total=total_computations, desc="Computing Glass Phase Data")) as progress_bar:
        results = np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_single_point)(M_background_val, M_substitution_val) for M_background_val, M_substitution_val in parameter_values))


    # Reshape results to (len(M_background_values) * len(M_substitution_values), n_iterations, 3)
    results = results.reshape(len(M_background_values) * len(M_substitution_values), n_iterations, 3)
    phase_data = {
        "M_background": results[:, 0, 0],
        "M_substitution": results[:, 0, 1],
        "Bott_mean": np.mean(results[:, :, 2], axis=1)
    }
    print(results)


# endregion
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #


def compute_save_plot_phase_diagram(generation:int, m_back_vals:np.ndarray, m_sub_vals:np.ndarray, filename:str="temp.npz", directory:str = "./Disclination/PhaseData/", doOverwrite:bool = False, excludeCenter:bool = False):
    if os.path.exists(os.path.join(directory, filename)) and not doOverwrite:
        print(f"File {filename} already exists")
    else:
        phase_data = compute_phase_data(generation=generation, M_background=m_back_vals, M_substitution=m_sub_vals, n_jobs=-1, excludeCenter=excludeCenter)
        save_phase_data(phase_data, filename, doOverwrite=doOverwrite, directory=directory)
    phase_data = open_phase_data(filename, directory)
    plot_phase_data(filename)


def main_single():
    lattice_dict = generate_sierpinski_lattice(generation=3, pad_width=0)
    lattice, mask = lattice_dict["lattice"], lattice_dict["center_mask"]
    wannier_matrices = calculate_square_hopping(lattice, pbc=True)

    H = calculate_hamiltonian(wannier_matrices, lattice_dict, 1.0, 1.0)
    P = calculate_projector(H)
    bott = calculate_bott_index(projector=P, lattice=lattice, mask=mask)
    return bott


def make_big_plot(generation:int):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for i, f in enumerate([f"g{generation}_ms={v}.npz" for v in [-2.5, -1.0, 1.0, 2.5]]):
        plot_phase_data(f, axs.flatten()[i], doSave=False, doShow=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("./Disclination/PhaseData/" + f"g{generation}_big_plot.png")
    plt.show()


def main(M_background, M_substitution, excludeCenter:bool):
    lattice_dict = generate_sierpinski_lattice(generation=3, pad_width=0, excludeCenter=excludeCenter)
    lattice, mask = lattice_dict["lattice"], lattice_dict["center_mask"]
    H = calculate_hamiltonian(calculate_square_hopping(lattice, pbc=True), lattice_dict, M_background=M_background, M_substitution=M_substitution)
    P = calculate_projector(H, useGPU=True, verbose=False)
    bott = calculate_bott_index(P, lattice, mask)

    LDOS = compute_LDOS(H.toarray(), number_of_states=2)["LDOS"]
    Y, X = np.where(lattice >= 0)[:]
    plt.figure(figsize=(8, 8))

    fill_indices, hole_indices = lattice_dict["fill_indices"], lattice_dict["hole_indices"]
    fill_mask = np.logical_and(np.isin(lattice.flatten(), fill_indices), mask)
    hole_mask = np.logical_and(np.isin(lattice.flatten(), hole_indices), mask)

    # Map LDOS (which is ordered over active sites where mask==True) to the lattice coordinates
    Y_flat, X_flat = np.where(lattice >= 0)
    active_site_indices = np.flatnonzero(mask)          # flattened indices of sites kept by mask
    X_active = X_flat[mask]                             # X coords in same order as LDOS
    Y_active = Y_flat[mask]                             # Y coords in same order as LDOS

    # Select the active sites that are filled and index LDOS accordingly
    LDOS /= np.max(LDOS)

    filled_active_mask = np.isin(active_site_indices, fill_indices)
    LDOS_filled = LDOS[filled_active_mask]

    hole_active_mask = np.isin(active_site_indices, hole_indices)
    LDOS_hole = LDOS[hole_active_mask]

    plt.scatter(X_active[filled_active_mask], Y_active[filled_active_mask],
                c=LDOS_filled, s=100, cmap='viridis', alpha=1.0, marker='^')
    plt.scatter(X_active[hole_active_mask], Y_active[hole_active_mask],
                c=LDOS_hole, s=100, cmap='viridis', alpha=1.0, marker='o')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"LDOS ($M_{{back}}={M_background:.2f}, M_{{sub}}={M_substitution:.2f}, $BI$={bott:.0f}$)")
    plt.colorbar(label='LDOS')
    plt.show()
    return bott


if __name__ == "__main__":
    plot1D = False
    plot2D = True


    if plot1D:
        m_sub_values = [-2.5, -1.0, 1.0, 2.5]
        m_back = np.linspace(-2.0, 2.0, 251)
        generation = 2
        for m_sub in m_sub_values:
            phase_data = compute_phase_data(generation, m_back, [m_sub])
            save_phase_data(phase_data, filename=f"g{generation}_ms={m_sub}.npz", doOverwrite=True)
            pass
        make_big_plot(generation=generation)

    if plot2D:
        generation = 3
        fname = [f"gen{generation}_mb_vs_ms.npz"]
        vals = np.linspace(-2.5, 2.5, 101)
        for fname in fname:
            compute_save_plot_phase_diagram(generation, vals, vals, filename=fname, doOverwrite=True)

        generation = 4
        fname = [f"gen{generation}_mb_vs_ms.npz"]
        vals = np.linspace(-2.5, 2.5, 25)
        for fname in fname:
            compute_save_plot_phase_diagram(generation, vals, vals, filename=fname, doOverwrite=True)
















    
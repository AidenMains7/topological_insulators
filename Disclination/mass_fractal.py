import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh, eigvals
import scipy.sparse as spsp
from itertools import product
from joblib import Parallel, delayed
from tqdm_joblib import tqdm, tqdm_joblib
import os, cProfile, pstats, time, h5py


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


# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# region Geometry


def generate_sierpinski_lattice(generation:int, pad_width:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    
    #side length
    L = 3**generation

    #square lattice
    square_lat = np.arange(L*L).reshape((L,L))
    carpet = _sierpinski_carpet(generation)

    #pad width
    if (pad_width > 0):
        carpet = np.pad(carpet,pad_width,mode='constant',constant_values=1)

    #get indices of empty and filled sites 
    flat = carpet.flatten()
    holes = np.where(flat==0)[0]
    fills = np.flatnonzero(flat)

    #construct fractal lattice
    fractal_lat = np.full(flat.shape, -1, dtype=int)
    fractal_lat[fills] = np.arange(fills.size)
    fractal_lat = fractal_lat.reshape(carpet.shape)

    return square_lat, holes, fills


def calculate_square_hopping(lattice, pbc):
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

def calculate_hamiltonian(wannier_matrices, holes, fills, M_background, M_substitution, t:float = 1.0, t0:float = 1.0, doSparse:bool = True):
    pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)

    Cx, Cy, Sx, Sy, CxCy, CySx, CxSy, I = wannier_matrices.values()

    onsite_mass = I.copy()
    for idx in holes:
        onsite_mass[idx, idx] = M_background
    for idx in fills:
        onsite_mass[idx, idx] = M_substitution

    if doSparse:
        d1_sparse = t * Sx
        d2_sparse = t * Sy
        d3_sparse = onsite_mass.tocsr() - t0 * (Cx + Cy)

        pauli1_sparse = spsp.csr_matrix(pauli1)
        pauli2_sparse = spsp.csr_matrix(pauli2)
        pauli3_sparse = spsp.csr_matrix(pauli3)

        hamiltonian = spsp.kron(d1_sparse, pauli1_sparse) + spsp.kron(d2_sparse, pauli2_sparse) + spsp.kron(d3_sparse, pauli3_sparse)
        return hamiltonian
    
    d1 = t * Sx.toarray()
    d2 = t * Sy.toarray()
    d3 = onsite_mass.toarray() - t0 * (Cx.toarray() + Cy.toarray())
    hamiltonian = np.kron(d1, pauli1) + np.kron(d2, pauli2) + np.kron(d3, pauli3)
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
            import cupy as cp
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


def calculate_bott_index(projector:np.ndarray, lattice:np.ndarray):
    Y, X = np.where(lattice >= 0)[:]
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


def compute_phase_data(generation:int, M_background:float, M_substitution:float, t:float = 1.0, t0:float = 1.0, n_jobs:int = -1, pad_width:int = 0):
    lattice, holes, fills = generate_sierpinski_lattice(generation=generation, pad_width=pad_width)
    wannier_matrices = calculate_square_hopping(lattice, pbc=True)
    parameter_values = tuple(product(M_background, M_substitution))

    def _compute_single_point(M_background_val, M_substitution_val):
        H = calculate_hamiltonian(wannier_matrices, holes, fills, M_background_val, M_substitution_val, t, t0)
        P = calculate_projector(H)
        bott = calculate_bott_index(P, lattice)
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


def plot_phase_data(filename, ax:plt.axes = None):
    phase_data = open_phase_data(filename)

    M_background_unique = np.unique(phase_data["M_background"])
    M_substitution_unique = np.unique(phase_data["M_substitution"])

    if (len(M_background_unique) == 1) or (len(M_substitution_unique) == 1):
        # 1D scatter plot
        doScat = True
    else:
        Bott = phase_data["Bott"].reshape((M_background_unique.size, M_substitution_unique.size))
        doScat = False

    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if not doScat:
        im = ax.imshow(Bott, extent=(M_substitution_unique.min(), M_substitution_unique.max(), M_background_unique.min(), M_background_unique.max()), 
                       origin='lower', cmap='RdBu', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label='Bott Index')
        ax.set_xlabel('Mass on Filled Sites (M_substitution)')
        ax.set_ylabel('Mass on Empty Sites (M_background)')
    else:
        if len(M_substitution_unique) == 1:
            sc = ax.scatter(phase_data["M_background"], phase_data["Bott"], label=f"$M^{{\\text{{sub}}}}={M_substitution_unique[0]}$") 
            ax.set_xlabel('Mass on Empty Sites (M_background)')     
        else:
            sc = ax.scatter(phase_data["M_substitution"], phase_data["Bott"], label=f"$M^{{\\text{{back}}}}={M_background_unique[0]}$")
            ax.set_xlabel('Mass on Filled Sites (M_substitution)')
        ax.set_ylabel('Bott Index')
        ax.set_yticks([-1, 0, 1])
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.legend()


    ax.set_title('Phase Diagram')
    ax.grid(False)
    plt.show()

def compute_glass_phase_data(M_background_values:np.ndarray, M_substitution_values:np.ndarray, n_fractal_sites:int, n_iterations:int, generation:int = 2, n_jobs:int = -1):
    parameter_values = np.array(tuple(product(M_background_values, M_substitution_values)))
    parameter_values = np.repeat(parameter_values, n_iterations, axis=0)
    total_computations = len(M_background_values) * len(M_substitution_values) * n_iterations

    def _compute_single_point(M_background_val, M_substitution_val):
        lattice, holes, fills = generate_lattice_glass(generation=generation, n_fractal_sites=n_fractal_sites)
        wannier_matrices = calculate_square_hopping(lattice, pbc=True)
        H = calculate_hamiltonian(wannier_matrices, holes, fills, M_background_val, M_substitution_val, t=1.0, t0=1.0)
        P = calculate_projector(H)
        bott = calculate_bott_index(P, lattice)
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


def compute_save_plot_phase_diagram(m_substitution:float, directory:str = "./Disclination/PhaseData/", doOverwrite:bool = False):
    M_substitution = [m_substitution]
    M_background = np.linspace(-100.0, 100.0, 11)

    filename = f"test.npz"

    if os.path.exists(os.path.join(directory, filename)) and not doOverwrite:
        print(f"File {filename} already exists")
    else:
        phase_data = compute_phase_data(generation=3, M_background=M_background, M_substitution=M_substitution, n_jobs=-1)
        save_phase_data(phase_data, filename, doOverwrite=doOverwrite, directory=directory)
    phase_data = open_phase_data(filename, directory)
    plot_phase_data(filename)


def main_single():
    lattice, holes, fills = generate_sierpinski_lattice(generation=3, pad_width=0)
    wannier_matrices = calculate_square_hopping(lattice, pbc=True)

    H = calculate_hamiltonian(wannier_matrices, holes, fills, 1.0, 1.0)
    P = calculate_projector(H)
    bott = calculate_bott_index(projector=P, lattice=lattice)
    return bott


def make_big_plot():
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for i, f in enumerate([f"ms={v}.npz" for v in [-2.5, -1.0, 1.0, 2.5]]):
        plot_phase_data(f, axs.flatten()[i])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("./Disclination/PhaseData/" + "temp.png")
    plt.show()


if __name__ == "__main__":

    compute_save_plot_phase_diagram(m_substitution=1.0, doOverwrite=True)


    lat, fills, holes = generate_sierpinski_lattice(generation=3, pad_width=0)
    H = calculate_hamiltonian(calculate_square_hopping(lat, pbc=True), holes, fills, M_background=4.0, M_substitution=1.0)
    P = calculate_projector(H, useGPU=False, verbose=True)
    bott = calculate_bott_index(P, lat)
    print(bott)

    LDOS = compute_LDOS(H.toarray(), number_of_states=2)["LDOS"]

    Y, X = np.where(lat >= 0)[:]
    plt.scatter(X, Y, c=LDOS, cmap='inferno')
    plt.colorbar(label='LDOS')
    plt.show()
    
    #compute_glass_phase_data([1.0], [-1.0, 0.0, 1.0], n_fractal_sites=10, n_iterations=5, generation=2, n_jobs=4)
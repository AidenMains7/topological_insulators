import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib, tqdm
from itertools import product
import os, h5py
from matplotlib.colors import ListedColormap, BoundaryNorm


# ---------------------------------------------------------------------------
# d-vector construction
# ---------------------------------------------------------------------------

def compute_d_vector(kx: np.ndarray, ky: np.ndarray, m0: float, h_vector: np.ndarray,
                     t: float, t0: float) -> np.ndarray:
    """
    Construct the (complex) d-vector for the non-Hermitian QWZ model.

        d1 = t*sin(kx) + i*hx
        d2 = t*sin(ky) + i*hy
        d3 = m0 + t0*(cos(kx)+cos(ky)) + i*hz

    Returns shape (3, N) for N k-points.
    """
    d1 = t * np.sin(kx) + 1j * h_vector[0]
    d2 = t * np.sin(ky) + 1j * h_vector[1]
    d3 = m0 + t0 * (np.cos(kx) + np.cos(ky)) + 1j * h_vector[2]
    return np.array([d1, d2, d3])


# ---------------------------------------------------------------------------
# Derivatives and d-hat
# ---------------------------------------------------------------------------

def _vector_derivative(kx: np.ndarray, ky: np.ndarray, direction: str,
                        m0: float, h_vector: np.ndarray, t: float, t0: float,
                        dk: float = 1e-5) -> np.ndarray:
    """
    Central-difference partial derivative of compute_d_vector.
    direction: 'x' or 'y'.
    """
    kwargs = dict(m0=m0, h_vector=h_vector, t=t, t0=t0)
    if direction == 'x':
        return (compute_d_vector(kx + dk, ky, **kwargs)
                - compute_d_vector(kx - dk, ky, **kwargs)) / (2 * dk)
    elif direction == 'y':
        return (compute_d_vector(kx, ky + dk, **kwargs)
                - compute_d_vector(kx, ky - dk, **kwargs)) / (2 * dk)
    else:
        raise ValueError(f"direction must be 'x' or 'y', got '{direction}'")


def compute_d_hat_and_derivatives(kx: np.ndarray, ky: np.ndarray,
                                   m0: float, h_vector: np.ndarray,
                                   t: float, t0: float):
    """
    Return d_hat, ∂d_hat/∂kx, ∂d_hat/∂ky.

    For a complex vector d, the bilinear norm is
        |d|² = d† · d  (complex inner product, not d·d)
    so the derivative of d_hat = d/|d| is

        ∂d_hat/∂ki = (∂d/∂ki)/|d| - d/(2|d|³) * (∂d†/∂ki · d + d† · ∂d/∂ki)

    which keeps ∂d_hat/∂ki orthogonal to d_hat in the Hermitian sense.
    This is what is implemented below.
    """
    kwargs = dict(m0=m0, h_vector=h_vector, t=t, t0=t0)
 
    d     = compute_d_vector(kx, ky, **kwargs)          # (3, N)
    d_dkx = _vector_derivative(kx, ky, 'x', **kwargs)  # (3, N)
    d_dky = _vector_derivative(kx, ky, 'y', **kwargs)  # (3, N)
 
    # Bilinear norm: d·d = sum_i d_i^2  (complex, no conjugation); shape (1, N)
    d_dot_d     = np.einsum('ij,ij->j', d, d)[np.newaxis, :]
    d_norm      = np.sqrt(d_dot_d)                      # complex square root
    d_norm_safe = np.where(np.abs(d_norm) < 1e-10, 1.0, d_norm)
 
    d_hat = d / d_norm_safe  # (3, N)
 
    def _d_hat_deriv(d_dki):
        # ∂(d·d)/∂ki = 2 * d_dki · d  (bilinear, no conjugation)
        d_dki_dot_d = np.einsum('ij,ij->j', d_dki, d)[np.newaxis, :]
        return d_dki / d_norm_safe - d_hat * d_dki_dot_d / d_dot_d
 
    return d_hat, _d_hat_deriv(d_dkx), _d_hat_deriv(d_dky)



# ---------------------------------------------------------------------------
# Berry curvature and Chern number (smooth / continuum method)
# ---------------------------------------------------------------------------

def compute_berry_curvature(kx: np.ndarray, ky: np.ndarray,
                             m0: float, h_vector: np.ndarray,
                             t: float, t0: float) -> np.ndarray:
    """
    Berry curvature Ω(k) = d_hat · (∂d_hat/∂kx × ∂d_hat/∂ky).

    For the non-Hermitian case d_hat may be complex, so we take the real part
    after contracting with the conjugate of d_hat to get a real-valued curvature.
    Returns shape (N,).
    """
    d_hat, d_hat_dx, d_hat_dy = compute_d_hat_and_derivatives(
        kx, ky, m0, h_vector, t, t0)

    # Cross product (∂xd_hat) × (∂yd_hat), shape (3, N)
    cross = np.cross(d_hat_dx, d_hat_dy, axis=0)

    # Ω = Re( d_hat* · cross )
    berry_curvature = np.einsum('ij,ij->j', d_hat, cross).real
    return berry_curvature


def compute_chern_number(m0: float, h_vector: np.ndarray, t: float, t0: float,
                          resolution: tuple = (201, 201)) -> float | None:
    """
    Chern number via numerical integration of Berry curvature over the BZ:

        C = (1/4π) ∬ Ω(k) dkx dky

    Uses a uniform mesh with endpoint=False so the BZ is not double-counted.
    The flat (N,) layout avoids a nested loop and keeps NumPy broadcasting fast.
    """
    kx_vals = np.linspace(-np.pi, np.pi, resolution[0], endpoint=False)
    ky_vals = np.linspace(-np.pi, np.pi, resolution[1], endpoint=False)
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)   # (Ny, Nx) each
    kx_flat = kx_grid.ravel()
    ky_flat = ky_grid.ravel()

    berry_curvature = compute_berry_curvature(kx_flat, ky_flat, m0, h_vector, t, t0)

    dkx = 2.0 * np.pi / resolution[0]
    dky = 2.0 * np.pi / resolution[1]

    # Single sum over all N k-points, then multiply by the area element dkx*dky
    chern = np.sum(berry_curvature) * dkx * dky / (4.0 * np.pi)
    chern = np.round(chern)
    
    if np.isnan(chern) or np.isinf(chern) or np.abs(chern) not in [1.0, 0.0]:
        #print(f"Warning: unphysical Chern number for m0={m0:.3f}, "
        #      f"h_vector={h_vector}: {chern:.3e}")
        chern = 0.

    if chern == 1.0:
        chern = 0.0
    

    return chern

# ---------------------------------------------------------------------------
# Phase diagram scan
# ---------------------------------------------------------------------------

def compute_chern_phase_diagram(m0_range: tuple, h_range: tuple, h_type: str,
                                 output_file: str = None, directory: str = '',
                                 overwrite: bool = False,
                                 resolution: tuple = (25, 25)) -> str:
    """
    Scan (m0, h) parameter space and save a grid of Chern numbers to HDF5.

    Parameters
    ----------
    m0_range  : (min, max) for the topological mass m0
    h_range   : (min, max) for the imaginary-field amplitude
    h_type    : which component the field acts on – 'x', 'y', or 'z'
    output_file : path to save HDF5 (auto-generated if None)
    directory  : directory for auto-generated filename
    overwrite  : re-compute even if the file exists
    resolution : (N_m0, N_h) grid size
    """
    m0_values = np.linspace(m0_range[0], m0_range[1], resolution[0])
    h_values  = np.linspace(h_range[0],  h_range[1],  resolution[1])
    # Store (i_m0, i_h) indices alongside params so results can be placed correctly
    # regardless of the order Parallel returns them.
    parameter_values = list(product(enumerate(m0_values), enumerate(h_values)))

    if output_file is None:
        tag = 'fhs' if use_fhs else 'berry'
        output_file = os.path.join(
            directory, f"square_chern_phase_diagram_{tag}_{resolution[0]}x{resolution[1]}.h5")

    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Use overwrite=True to recompute.")
        return output_file

    def _h_vector(h: float) -> list:
        match h_type:
            case 'x': return [h, 0.0, 0.0]
            case 'y': return [0.0, h, 0.0]
            case 'z': return [0.0, 0.0, h]
            case _:   raise ValueError(f"h_type must be 'x', 'y', or 'z', got '{h_type}'")

    def compute_single(params):
        (i_m0, m0), (i_h, h) = params
        hv = _h_vector(h)
        chern = compute_chern_number(m0, hv, 1.0, 1.0)
        return i_m0, i_h, chern if chern is not None else np.nan

    with tqdm_joblib(tqdm(total=len(parameter_values),
                          desc="Computing Chern phase diagram")) as _:
        results = Parallel(n_jobs=-2)(delayed(compute_single)(p) for p in parameter_values)

    # Place each result at its correct (i_m0, i_h) grid position.
    # chern_grid shape: (N_h, N_m0) — rows=h (y-axis), cols=m0 (x-axis) for imshow.
    chern_grid = np.full((resolution[1], resolution[0]), np.nan)
    for i_m0, i_h, chern_val in results:
        chern_grid[i_h, i_m0] = chern_val

    with h5py.File(output_file, "w") as f:
        f.create_dataset("m0",    data=m0_values)
        f.create_dataset("h",     data=h_values)
        f.create_dataset("chern", data=chern_grid)

    return output_file


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phase_diagram(fig, ax,
                        X_values: np.ndarray, Y_values: np.ndarray, Z_values: np.ndarray,
                        labels: list = None, title: str = None,
                        X_ticks=None, Y_ticks=None,
                        X_tick_labels=None, Y_tick_labels=None,
                        cmap: str = 'Spectral',
                        plot_colorbar: bool = True,
                        discrete_colormap: bool = True):
    """
    Display a 2D phase diagram as a colour-mapped image.

    Z_values is shown rounded to the nearest integer (Chern numbers are integers).
    NaN entries are masked.
    """
    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]

    # Treat -0 as 0 and round to nearest integer
    Z_plot = np.where(Z_values == -0.0, 0.0, Z_values)
    Z_rounded = np.round(Z_plot)

    not_nan_mask = ~np.isnan(Z_rounded)
    unique_values = np.sort(np.unique(Z_rounded[not_nan_mask]).astype(int))

    if discrete_colormap and len(unique_values) < 25:
        original_cmap = plt.get_cmap(cmap)
        discrete_colors = original_cmap(np.linspace(0, 1, len(unique_values)))
        plot_cmap = ListedColormap(discrete_colors)
        norm = BoundaryNorm(
            boundaries=np.append(unique_values, unique_values[-1] + 1),
            ncolors=len(unique_values))
    else:
        plot_cmap = cmap
        norm = None

    im = ax.imshow(
        Z_rounded,
        extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]],
        origin='lower', aspect='auto',
        cmap=plot_cmap, norm=norm,
        interpolation='none', rasterized=True)

    if title is not None:
        ax.set_title(title)
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1], rotation=0)
    if X_ticks is not None:
        ax.set_xticks(X_ticks)
    if Y_ticks is not None:
        ax.set_yticks(Y_ticks)
    if X_tick_labels is not None:
        ax.set_xticklabels(X_tick_labels)
    if Y_tick_labels is not None:
        ax.set_yticklabels(Y_tick_labels)

    if plot_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if discrete_colormap and len(unique_values) < 25:
            cbar.set_ticks(unique_values)
            cbar.set_ticklabels([str(v) for v in unique_values])

    return fig, ax


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    h_dir      = 'z'
    resolution = (25, 25)

    file = compute_chern_phase_diagram(
        m0_range=(0.0, 2.0),
        h_range=(0.0, 1.25),
        h_type=h_dir,
        overwrite=True,
        output_file=f"chern_pd_{h_dir}_{resolution[0]}x{resolution[1]}.h5",
        resolution=resolution,
    )

    with h5py.File(file, "r") as f:
        m0    = f["m0"][:]
        h     = f["h"][:]
        chern = f["chern"][:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # m0 and h are now clean 1D axis vectors; chern has shape (N_h, N_m0)
    plot_phase_diagram(fig, ax, m0, h, chern,
                       labels=[r"$m_0$", r"$h$"],
                       discrete_colormap=True)
    

    # Annotate expected phase boundary
    if h_dir in 'xy':
        ax.plot([0, 1, 2, 0, -2, -1, 0], [0, 1, 0, 0, 0, 1, 0], 'k--', lw=1.5)
    elif h_dir == 'z':
        t = np.linspace(0., 2., 101)
        ax.plot(t, np.sqrt(1 - (1 - t) ** 2), c='k', ls='--')
        ax.plot(-t, np.sqrt(1 - (1 - t) ** 2), c='k', ls='--')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    #points = [(1., 0.), (1., 0.5), (1., 1.0), (1., 1.5), (0.5, 0.), (0.5, 0.25), (0.5, 0.5), (0.5, 0.55)]
    #cherns = []
    #for (m0, h) in points:
    #    chern = compute_chern_number(m0, [h, 0., 0.], 1., 1., (101, 101))
    #    cherns.append(chern)
    #    print(f"m0={m0}, h={h}, C={chern}")
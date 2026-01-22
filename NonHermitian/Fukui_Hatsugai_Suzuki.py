"""
Fukui–Hatsugai–Suzuki (FHS) lattice algorithm for computing the Chern number
of a single isolated 2D Bloch band on a discrete Brillouin-zone (BZ) mesh.

Overview
--------
The Brillouin zone is discretized into an N×N grid (N = `k_space_resolution`)
forming a torus with spacing Δk = 2π/N in each direction. For a chosen band,
U(1) link variables between neighboring k-points are constructed from overlaps
of normalized Bloch eigenvectors. The discrete Berry curvature (lattice field
strength) on each plaquette is the oriented product of these links; summing
over all plaquettes yields an integer Chern number in the gapped case.

References
----------
T. Fukui, Y. Hatsugai, and H. Suzuki,
"Chern Numbers in Discretized Brillouin Zone: Efficient Method of Computing
(Spin) Hall Conductances", J. Phys. Soc. Jpn. 74, 1674 (2005).

Typical use
-----------
>>> import numpy as np
>>> def h_func(kx, ky, m=0.5):
...     # Return a 2×2 Hermitian matrix H(k) for your model
...     # that is 2π-periodic in kx, ky.
...     raise NotImplementedError
>>> # Extra parameters can be passed via h_args/h_kwargs:
>>> C = compute_chern(h_func, band_index=0, k_space_resolution=51,
...                   h_kwargs={'m': 0.5})

Notes
-----
- The target band must be *isolated* (spectrally gapped from neighbors) over
  the entire BZ. Near-degeneracies can make overlaps tiny and phases unstable.
- The Hamiltonian must be periodic on kx, ky ∈ [−π, π).
- The returned integer is rounded from the complex sum to protect against
  small floating-point drift.
"""

import numpy as np
from scipy.linalg import eigh


def compute_chern(h_func, *, band_index=0, k_space_resolution=30, h_args=(), h_kwargs=None):
    """
    Compute the Chern number of a specified Bloch band via the FHS lattice algorithm.

    Parameters
    ----------
    h_func : callable
        Function returning an (nband × nband) Hermitian numpy array for
        the Bloch Hamiltonian at (kx, ky):
            H = h_func(kx, ky, *h_args, **h_kwargs)
        - (kx, ky) are floats in radians and should be 2π-periodic on [−π, π).
        - Eigenpairs are assumed to be obtained in ascending energy order via
          `scipy.linalg.eigh`.

    band_index : int, optional
        Zero-based index of the band whose Chern number is evaluated
        (0 = lowest energy). Default is 0.

    k_space_resolution : int, optional
        Number of k-points per dimension in the uniform BZ mesh (N). The
        algorithm cost scales roughly as O(N^2 × diagonalization_cost).
        Default is 30.

    h_args : tuple, optional
        Additional positional arguments forwarded to `h_func` *after* (kx, ky).
        Useful for model parameters that you want to hold fixed while scanning
        the BZ. Default is an empty tuple.

    h_kwargs : dict, optional
        Additional keyword arguments forwarded to `h_func`. Default is None.

    Returns
    -------
    int
        The Chern number of the selected band. The sum of lattice field strengths
        is converted to an integer by rounding its real part to guard against
        small numerical drift.

    Implementation details
    ----------------------
    On a uniform N×N grid with spacing Δk = 2π/N, define normalized U(1) link
    variables using the eigenvector |u(k)⟩ of the chosen band:
        U_x(k) = ⟨u(k)|u(k + Δk_x)⟩ / |⟨u(k)|u(k + Δk_x)⟩|
        U_y(k) = ⟨u(k)|u(k + Δk_y)⟩ / |⟨u(k)|u(k + Δk_y)⟩|
    The lattice field strength on each plaquette with lower-left corner at k is
        F(k) = log[ U_x(k)
                    · U_y(k + Δk_x)
                    · U_x(k + Δk_y)^*
                    · U_y(k)^* ]
    where log denotes the principal complex logarithm. The Chern number is then
        C = (1 / (2π i)) * Σ_k F(k)
    where the sum runs over all plaquettes of the discrete torus (i.e., all grid
    points as lower-left corners). Periodicity at the boundaries is implicit.

    Caveats
    -------
    - Very small overlaps ⟨u(k)|u(k + δk)⟩ (e.g., near degeneracies) make phases
      ill-defined; increasing `k_space_resolution` typically improves robustness.
    - Ensure `h_func` is smooth and periodic over the BZ; discontinuities in gauge
      choice can manifest as noisy phases on coarse meshes.
    """
    if h_kwargs is None:
        h_kwargs = {}

    dk = 2 * np.pi / k_space_resolution

    def psi_band_transpose(kx_, ky_):
        """Return the eigenvector of the selected band at (kx_, ky_)."""
        h = h_func(kx_, ky_, *h_args, **h_kwargs)
        vals, vecs = eigh(h, overwrite_a=True)
        return vecs[:, band_index]

    def U(x_direction, kx_0, ky_0):
        """
        Compute a U(1) link variable along +x or +y from (kx_0, ky_0).

        Parameters
        ----------
        x_direction : bool
            True → step along +x (kx + Δk, ky); False → step along +y (kx, ky + Δk).

        kx_0, ky_0 : float
            Base-point coordinates in radians.

        Returns
        -------
        complex
            Unimodular complex number exp(i Δφ) extracted from the normalized
            overlap ⟨u(k)|u(k + δk)⟩.
        """
        kx_f, ky_f = (kx_0 + dk, ky_0) if x_direction else (kx_0, ky_0 + dk)
        vecs_0_dagger = np.conj(psi_band_transpose(kx_0, ky_0))
        vecs_f_transpose = psi_band_transpose(kx_f, ky_f)
        inner_product = np.dot(vecs_0_dagger, vecs_f_transpose)
        return inner_product / np.abs(inner_product)

    # Discrete torus: uniform grid on [−π, π) without the endpoint.
    kxs = np.linspace(-np.pi, np.pi, k_space_resolution, endpoint=False)
    kys = np.linspace(-np.pi, np.pi, k_space_resolution, endpoint=False)

    # Sum the lattice field strength over all plaquettes.
    sum_F_d2k = sum(
        np.log(
            U(True,  kx,      ky)
            * U(False, kx + dk, ky)
            * np.conj(U(True,  kx,      ky + dk))
            * np.conj(U(False, kx,      ky))
        )
        for kx in kxs for ky in kys
    )

    C = (1 / (2 * np.pi * 1j)) * sum_F_d2k
    return round(np.real(C))


def compute_chern_phase_diagram(
    h_func,
    param1_range,
    param2_range,
    param1_resolution,
    param2_resolution,
    *,
    band_index=0,
    k_space_resolution=30,
    h_args=(),
    h_kwargs=None,
):
    """
    Build a 2D phase diagram of Chern numbers over a rectangular parameter grid.

    This routine scans two model parameters (p1, p2). At each grid point it calls
    `compute_chern` on the same k-mesh and returns a map of integer Chern numbers.

    Expected model signature
    ------------------------
    The Hamiltonian factory is called as
        H = h_func(kx, ky, p1, p2, *h_args, **h_kwargs)
    i.e., (p1, p2) are forwarded as the first two extra positional arguments after
    (kx, ky) by setting `h_args=(p1, p2, *h_args)` in the internal call.

    Parameters
    ----------
    h_func : callable
        See `compute_chern`. Must accept (kx, ky) followed by (p1, p2, ...).

    param1_range : tuple[float, float]
        Inclusive (lower, upper) bounds for the first scanned parameter p1.

    param2_range : tuple[float, float]
        Inclusive (lower, upper) bounds for the second scanned parameter p2.

    param1_resolution : int
        Number of p1 samples. `np.linspace(lower, upper, param1_resolution)`
        is used; index 0 maps to `lower` and index `resolution-1` maps to `upper`.

    param2_resolution : int
        Number of p2 samples, defined analogously.

    band_index : int, optional
        Passed through to `compute_chern`. Default is 0.

    k_space_resolution : int, optional
        BZ mesh resolution used in each `compute_chern` call. Default is 30.

    h_args : tuple, optional
        Extra fixed positional arguments (beyond p1, p2) forwarded to `h_func`.
        Default is an empty tuple.

    h_kwargs : dict, optional
        Extra fixed keyword arguments forwarded to `h_func`. Default is None.

    Returns
    -------
    chern_map : ndarray of shape (param1_resolution, param2_resolution), dtype=int
        Chern number at each (p1, p2) grid point. Axis 0 sweeps p1, axis 1 sweeps p2.

    p1_values : ndarray of shape (param1_resolution,)
        The sampled p1 values (inclusive linspace).

    p2_values : ndarray of shape (param2_resolution,)
        The sampled p2 values (inclusive linspace).

    Notes
    -----
    - Uses inclusive sampling (`np.linspace`) along both parameter axes.
    - Each entry is the rounded output of `compute_chern`, which itself implements
      the FHS algorithm on a discrete torus with periodic boundaries.
    """
    p1_values = np.linspace(param1_range[0], param1_range[1], int(param1_resolution))
    p2_values = np.linspace(param2_range[0], param2_range[1], int(param2_resolution))

    chern_map = np.empty((p1_values.size, p2_values.size), dtype=int)

    for i, p1 in enumerate(p1_values):
        for j, p2 in enumerate(p2_values):
            C = compute_chern(
                h_func,
                band_index=band_index,
                k_space_resolution=k_space_resolution,
                h_args=(p1, p2, *h_args),
                h_kwargs=h_kwargs,
            )
            chern_map[i, j] = int(C)

    return chern_map, p1_values, p2_values


if __name__ == '__main__':
    # Module is intended to be imported.
    pass

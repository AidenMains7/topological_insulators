"""Topological indices and the near-zero-mode LDOS observable.

All functions take an already-built ``hypercubic.Model`` plus a ``params`` dict
and consume the *same* model — the ``observable`` axis is orthogonal to how the
model was built.

  polarization(model, params)   -> float   (1D Cantor; real-space Zak phase)
  bott_index(model, params)     -> int      (2D carpet)
  near_zero_ldos(model, params, k_center, *, method, solver, k_solve) -> dict
"""

import numpy as np
import scipy.linalg as la


# ── shared: occupied lower-half eigenvectors ─────────────────────────────────

def _occupied_eigenvectors(model, params):
    """Lowest N//2 eigenvectors and the active display coordinates."""
    n_occ = model.active_hilbert_dim // 2
    r = model.solve(
        hermitian=True,
        return_eigenvalues=False,
        return_eigenvectors=True,
        return_LDOS=False,
        solver_kwargs={"subset_by_index": [0, n_occ - 1]},
        params=params,
    )
    coords = model.coordinates(frame="display", active_only=True)
    return r["eigenvectors"], coords


# ── 1D polarization (Zak phase) ──────────────────────────────────────────────

def polarization(model, params):
    V, coords = _occupied_eigenvectors(model, params)
    dim, n_occ = V.shape
    if n_occ in (0, dim):
        return 0.0
    X = coords[:, 0].astype(float)
    Lx = X.max() - X.min() + 1.0
    internal = dim // len(X)
    n_al = np.sum(X) / (2 * Lx)
    if internal > 1:
        X = np.repeat(X, internal)
    ex = np.exp(1j * 2 * np.pi * X / Lx)
    W = V.conj().T @ (V * ex[:, None])
    sign, _ = np.linalg.slogdet(W)
    n = np.angle(sign) / (2 * np.pi)
    p = abs(n - n_al)
    return float(p)


# ── 2D Bott index ────────────────────────────────────────────────────────────

def bott_index(model, params):
    V, coords = _occupied_eigenvectors(model, params)
    dim, n_occ = V.shape
    if n_occ in (0, dim):
        return 0
    X = coords[:, 0].astype(float)
    Y = coords[:, 1].astype(float)
    lx, ly = X.max() - X.min() + 1.0, Y.max() - Y.min() + 1.0
    internal = dim // len(X)
    if internal > 1:
        X, Y = np.repeat(X, internal), np.repeat(Y, internal)
    ex = np.exp(1j * 2 * np.pi * X / lx)
    ey = np.exp(1j * 2 * np.pi * Y / ly)
    U = V.conj().T @ (V * ex[:, None])
    W = V.conj().T @ (V * ey[:, None])
    comm = U @ W @ U.conj().T @ W.conj().T
    eigs = la.eigvals(comm, overwrite_a=True)
    return int(round(float(np.real(np.sum(np.angle(eigs)))) / (2 * np.pi)))


# ── near-zero-mode summed LDOS ───────────────────────────────────────────────

def near_zero_ldos(model, params, k_center, *, method="substituted",
                   solver="ed", k_solve=None):
    """Summed LDOS over the central ``k_center`` near-zero modes (background, g=0).

    Selection is by mid-spectrum index; the models' particle-hole symmetry
    makes this identical to smallest-|E|.

    Parameters
    ----------
    k_center : int
        Number of central modes to keep (not half).  Forced even, since the
        particle-hole-symmetric band around E=0 is retained symmetrically.
    method : str
        Hole treatment of the run: ``'substituted'`` / ``'site_elim'`` /
        ``'cube'`` solve the built Hamiltonian directly; ``'renorm'`` solves the
        Schur-complement effective Hamiltonian (sector-0 eliminated at E=0).
    solver : {'ed', 'sparse'}
        ``'ed'``    -> full dense spectrum, slice the central band.
        ``'sparse'`` -> shift-invert ``k_solve`` states near E=0, keep the
        central ``k_center``.  ``k_solve`` (and the sparse path) are ignored when
        ``solver='ed'``.  ``'renorm'`` requires ``'ed'`` because its effective
        Hamiltonian is dense (elimination destroys the original sparsity).

    Returns ``dict(ldos=<grid>, energies=<k_center,>)``.
    """
    if solver not in ("ed", "sparse"):
        raise ValueError(f"unknown solver {solver!r}; expected 'ed' or 'sparse'.")
    if method == "renorm" and solver != "ed":
        raise ValueError(
            "method 'renorm' requires solver='ed': its Schur-complement "
            "effective Hamiltonian is dense, so sparse shift-invert is invalid."
        )

    # k_center directly sets the number of central modes; force even.
    k_center = int(k_center)
    if k_center < 1:
        raise ValueError(f"k_center must be a positive integer; got {k_center}.")
    k_center += k_center % 2
    half = k_center // 2

    if method == "renorm":
        r = model.solve_schur(eliminate_label="sector", eliminate_value=0, energy=0.0,
                              hermitian=True, k=None,
                              return_eigenvalues=True, return_eigenvectors=False,
                              return_LDOS=True, params=params)
    elif solver == "ed":
        r = model.solve(hermitian=True, k=None,
                        return_eigenvalues=True, return_eigenvectors=False,
                        return_LDOS=True, params=params)
    else:  # sparse shift-invert near E=0
        k_solve = k_center + 2 if k_solve is None else int(k_solve)
        k_solve += k_solve % 2                       # force even
        if k_solve < k_center + 2:
            raise ValueError(
                f"k_solve={k_solve} too small for k_center={k_center}; "
                f"use at least {k_center + 2} to keep an ARPACK margin."
            )
        r = model.solve(hermitian=True, k=k_solve, sigma=0.0, which="LM",
                        return_eigenvalues=True, return_eigenvectors=False,
                        return_LDOS=True, params=params)

    w = r["eigenvalues"]            # sorted ascending on all paths
    mid = w.size // 2
    lo, hi = mid - half, mid + half

    # Guard: under PH symmetry the kept band must straddle zero.
    if not (w[lo] <= 0.0 <= w[hi - 1]):
        raise RuntimeError(
            f"central band [{w[lo]:.3e}, {w[hi-1]:.3e}] does not straddle E=0; "
            "check k_solve / particle-hole assumption."
        )

    ldos_sum = np.nan_to_num(r["LDOS"][lo:hi], nan=0.0).sum(axis=0)
    return dict(ldos=ldos_sum, energies=w[lo:hi])


"""3D Menger-sponge Witten-effect response (monopole-induced charge).

One ``method`` switch selects how sector-0 sites are handled:

    'substituted' -> full ED, complement mass = M_alt   (staggered)
    'site_elim'   -> vacancies_from_label(sector=0), then ED
    'renorm'      -> solve_schur, eliminating sector 0 at E=0   (exploratory)
    'cube'        -> solid-block reference                       (exploratory)

The half-filling charge is the LDOS summed over the lowest N//2 states, solved
once at g=0 (background) and once at g=1 (monopole); the induced density is the
difference, which is integrated to delta_Q(R) by an exact sphere-on-cubic-grid
capture (partial cells handled analytically).
"""

import numpy as np

from . import model as _model, lattice as _lattice


# ── charge solve (method-dispatched) ─────────────────────────────────────────

def half_filling_charge(m, params, method):
    """Return ``(eigenvalues, charge_grid)`` = LDOS summed over lowest N//2 states."""
    if method in ("substituted", "site_elim", "cube"):
        r = m.solve(hermitian=True, k=None,
                    return_eigenvalues=True, return_eigenvectors=False,
                    return_LDOS=True, params=params,
                    solver_kwargs={"overwrite_a": True})
    elif method == "renorm":
        r = m.solve_schur(eliminate_label="sector", eliminate_value=0, energy=0.0,
                          hermitian=True, k=None,
                          return_eigenvalues=True, return_eigenvectors=False,
                          return_LDOS=True, params=params)
    else:
        raise ValueError(f"unknown method {method!r}.")
    m._invalidate_cache()
    evals = r["eigenvalues"]
    charge = r["LDOS"][: evals.size // 2].sum(axis=0)
    return evals, charge


def witten_run(fractal="sponge", *, n, M, M_alt=None, method="substituted",
               block_scale=2, pasted=True, M_prime=0.01, gauge="N",
               upscale_to_n=None, t=1.0, B=1.0, pbc=False):
    """Build once, solve g=0 and g=1, return charges + induced dQ field + metadata."""
    if method == "substituted":
        if M_alt is None:
            M_alt = M
    else:
        # cube / site_elim / renorm use a single mass M throughout — there is no
        # independent complement mass, so M_alt is pinned to M.
        M_alt = M
    m = _model.build_model(fractal, n, hole_treatment=method, pbc=pbc,
                           upscale_to_n=upscale_to_n, block_scale=block_scale,
                           pasted=pasted, pseudo_scalar=True)
    base = dict(M=M, M_alt=M_alt, M_prime=M_prime, t=t, B=B,
                disorder_strength=0, disorder_seed=0, gauge=gauge)
    evals_bg, charge_bg = half_filling_charge(m, dict(base, g=0.0), method)
    evals_mono, charge_mono = half_filling_charge(m, dict(base, g=1.0), method)
    L = _lattice.system_length(n, upscale_to_n=upscale_to_n,
                               block_scale=block_scale, pasted=pasted)
    return dict(
        charge_bg=charge_bg, charge_mono=charge_mono,
        eigenvalues_bg=evals_bg, eigenvalues_mono=evals_mono,
        dQ_field=charge_mono - charge_bg,
        fractal=fractal, n=n, L=L, M=M, M_alt=M_alt, M_prime=M_prime,
        method=method, pasted=pasted, gauge=gauge,
    )


# ── exact sphere-on-cubic-grid integrator ────────────────────────────────────
# Computes the exact volume overlap of each unit cell with a sphere of radius R
# so that delta_Q(R) is smooth rather than a staircase.

def _P(rho, y):
    rho2 = rho * rho
    yc = np.clip(y, -rho, rho)
    sq = np.sqrt(np.maximum(rho2 - yc * yc, 0.0))
    safe_rho = np.where(rho > 0, rho, 1.0)
    ratio = np.where(rho > 0, yc / safe_rho, 0.0)
    return yc * sq + rho2 * np.arcsin(np.clip(ratio, -1.0, 1.0))


def _C(rho, a, b):
    rho2 = rho * rho
    A = np.clip(a, -rho, rho)
    bc = np.clip(b, -rho, rho)
    yb = np.sqrt(np.maximum(rho2 - bc * bc, 0.0))
    L = -rho
    B1 = -yb
    B2 = yb
    P_L = -np.pi * rho2 / 2
    lo_v = np.minimum(A, B1)
    mid_u = B1
    mid_v = np.minimum(A, B2)
    ro_u = B2
    I_lo = np.where(lo_v > L, _P(rho, lo_v) - P_L, 0.0)
    I_mid = np.where(mid_v > mid_u,
                     b * (mid_v - mid_u) + (_P(rho, mid_v) - _P(rho, mid_u)) / 2,
                     0.0)
    I_ro = np.where(A > ro_u, _P(rho, A) - _P(rho, ro_u), 0.0)
    f_outer = np.where(b >= 0, 1.0, 0.0)
    result = f_outer * (I_lo + I_ro) + I_mid
    full_seg = _P(rho, A) - P_L
    result = np.where(b >= rho, full_seg, result)
    result = np.where((b <= -rho) | (a <= -rho) | (rho <= 0), 0.0, result)
    return np.maximum(result, 0.0)


def _circle_rect_area(rho, y0, y1, z0, z1):
    return _C(rho, y1, z1) - _C(rho, y0, z1) - _C(rho, y1, z0) + _C(rho, y0, z0)


def _sphere_box_volumes(R, cx, cy, cz, n_quad=20):
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    x_lo = np.maximum(cx - 0.5, -R)
    x_hi = np.minimum(cx + 0.5, R)
    half_w = (x_hi - x_lo) * 0.5
    x_mid = (x_hi + x_lo) * 0.5
    x_q = x_mid[np.newaxis, :] + nodes[:, np.newaxis] * half_w[np.newaxis, :]
    w_q = weights[:, np.newaxis] * half_w[np.newaxis, :]
    rho_sq = R * R - x_q * x_q
    rho = np.sqrt(np.maximum(rho_sq, 0.0))
    areas = _circle_rect_area(rho, cy - 0.5, cy + 0.5, cz - 0.5, cz + 0.5)
    areas = np.where(rho_sq > 0, areas, 0.0)
    return np.clip(np.sum(w_q * areas, axis=0), 0.0, 1.0)


def _dQ_fractional_sphere(Q_flat, disp_flat, resolution):
    abs_d = np.abs(disp_flat)
    r_min = np.sqrt(np.sum(np.maximum(abs_d - 0.5, 0.0) ** 2, axis=0))
    r_max = np.sqrt(np.sum((abs_d + 0.5) ** 2, axis=0))
    R_vals = np.linspace(0, r_max.max(), num=resolution)
    dQ = np.empty(resolution)
    for i, R in enumerate(R_vals):
        full = r_max <= R
        bnd = (r_min <= R) & ~full
        total = Q_flat[full].sum()
        if bnd.any():
            idx = np.where(bnd)[0]
            fracs = _sphere_box_volumes(R, disp_flat[0, idx], disp_flat[1, idx], disp_flat[2, idx])
            total += (Q_flat[idx] * fracs).sum()
        dQ[i] = total
    return R_vals, dQ


def dQ_of_R(dQ_field, resolution=300):
    """Cumulative induced charge delta_Q(R) vs sphere radius R.

    ``dQ_field`` is the 3D induced-charge grid (``charge_mono - charge_bg``).
    Returns ``(R_vals, dQ)``.  NaNs in the field (vacancy sites) are treated 0.
    """
    Q = np.nan_to_num(np.asarray(dQ_field), nan=0.0)
    origin = (np.array(Q.shape, dtype=float) - 1.0) / 2.0
    disp = np.indices(Q.shape, dtype=float) - origin[:, None, None, None]
    return _dQ_fractional_sphere(Q.ravel(), disp.reshape(3, -1), resolution)


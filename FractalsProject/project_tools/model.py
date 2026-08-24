"""The single Wilson-Dirac model builder (Layer 0b).

Everything that knows about the d-vector, the pseudo-scalar ``d_5`` term, the
monopole Peierls modifier, and the sector mass lives here and nowhere else.

The ``hole_treatment`` parameter selects how sector-0 (complement) sites are handled:

    'substituted' -> kept, given mass M_alt   (the main case)
    'site_elim'   -> removed (vacancies)
    'renorm'      -> Schur-eliminated at E=0  (exploratory)
    'cube'        -> full solid-block lattice  (reference)
"""

import numpy as np

from hypercubic import hypercubic_grid_model
from . import lattice as _lattice


# ── real-space callbacks (matched by __name__ in the DSL) ────────────────────

def sector_mass(labels):
    """Closure: sector-1 sites -> mass ``M``; sector-0 sites -> mass ``M_alt``."""
    arr = np.asarray(labels, dtype=int)

    def sector_mass(coords, M, M_alt):        # noqa: N802 (DSL matches __name__)
        return np.where(arr == 1, M, M_alt).astype(complex)

    return sector_mass


def uniform_disorder(coords, disorder_seed, disorder_strength):
    """Box disorder with variance 1/12; negative seed (-1) means 'random'."""
    N = coords.shape[1]
    if disorder_strength == 0.0:
        return None
    seed = None if (disorder_seed is None or disorder_seed < 0) else int(disorder_seed)
    rng = np.random.default_rng(seed)
    vals = rng.uniform(-0.5, 0.5, size=N)
    vals -= vals.mean()
    std = vals.std()
    if std == 0.0:
        raise ValueError("disorder has zero std; cannot rescale.")
    vals *= (1.0 / np.sqrt(12.0)) / std
    return disorder_strength * vals


def peierls_phase(base_value, p0, pf, g, gauge):
    """Dirac-monopole Peierls phase on a hop (3D only)."""
    x1, y1, z1 = p0
    x2, y2, z2 = pf
    xm, ym, zm = 0.5 * (x1 + x2), 0.5 * (y1 + y2), 0.5 * (z1 + z2)
    r = np.sqrt(xm ** 2 + ym ** 2 + zm ** 2)
    z_over_r = np.divide(zm, r, out=np.zeros_like(r, dtype=float), where=r != 0)
    rho2 = xm ** 2 + ym ** 2
    y_over_rho = np.divide(-ym, rho2, out=np.zeros_like(rho2, dtype=float), where=rho2 != 0)
    x_over_rho = np.divide(xm, rho2, out=np.zeros_like(rho2, dtype=float), where=rho2 != 0)
    sign = -1.0 if gauge == "S" else 1.0
    A_x = -(x2 - x1) * (1 + sign * z_over_r) * y_over_rho
    A_y = -(y2 - y1) * (1 + sign * z_over_r) * x_over_rho
    return np.exp(1j * g * (A_x + A_y) / 2) * base_value


# ── d-vector assembly ────────────────────────────────────────────────────────

def wilson_dirac_dvector(D, *, pseudo_scalar=False):
    """Wilson-Dirac d-vector as a DSL dict.

    d_0 = disorder, d_1..d_D = t*sin(k), d_{D+1} = sector-mass + Wilson cos terms.
    If ``pseudo_scalar`` (the 3D Witten case), add d_{D+2} = M_prime (Gamma_5),
    which does not raise the internal dimension.
    """
    sym = ("x", "y", "z")
    dv = {"d_0": "[uniform_disorder, disorder_seed={disorder_seed}, "
                 "disorder_strength={disorder_strength}]"}
    for i in range(D):
        dv[f"d_{i+1}"] = f"{{t}}*sin(k{sym[i]})"
    cos_terms = "".join(f" + {{2*B}}*cos(k{sym[i]})" for i in range(D))
    dv[f"d_{D+1}"] = f"[sector_mass, M={{M}}, M_alt={{M_alt}}] + {{-2*B*{D}}}" + cos_terms
    if pseudo_scalar:
        dv[f"d_{D+2}"] = "{M_prime}"
    return dv


# ── public builder ───────────────────────────────────────────────────────────

HOLE_TREATMENTS = ("substituted", "site_elim", "renorm", "cube")


def build_model(fractal, n, *, hole_treatment="substituted", pbc=False,
                upscale_to_n=None, block_scale=2, pasted=False,
                pseudo_scalar=None):
    """Build the fractal-mass Wilson-Dirac model.

    ``pseudo_scalar`` defaults to True in 3D (needed for the Witten
    construction) and False otherwise; pass explicitly to override.

    ``hole_treatment`` selects how sector-0 sites are handled (see module
    docstring); ``'renorm'`` builds the full lattice here and is downfolded at
    solve time via ``model.solve_schur``.
    """
    if hole_treatment not in HOLE_TREATMENTS:
        raise ValueError(
            f"unknown hole_treatment {hole_treatment!r}; "
            f"expected one of {HOLE_TREATMENTS}."
        )
    D = _lattice._seed(fractal).ndim
    if pseudo_scalar is None:
        pseudo_scalar = (D == 3)

    lat = _lattice.build_lattice(fractal, n, upscale_to_n=upscale_to_n,
                                 block_scale=block_scale, pasted=pasted)
    if hole_treatment == "cube":
        lat = np.ones_like(lat)

    sector_labels = lat.ravel(order="F")
    origin = (np.asarray(lat.shape, dtype=float) - 1.0) / 2.0

    m = hypercubic_grid_model(
        lattice=lat,
        pbc_flags=pbc,
        origin=origin,
        hopping_modifier=peierls_phase if D == 3 else None,
        real_space_functions=(uniform_disorder, sector_mass(sector_labels)),
        **wilson_dirac_dvector(D, pseudo_scalar=pseudo_scalar),
    )
    if hole_treatment == "site_elim":
        m.vacancies_from_label(sector=0)
    return m

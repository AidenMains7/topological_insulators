"""Plot the near-zero-mode local density of states (LDOS) on the 1D Cantor
fractal, for a given (M, M_alt), with an explicit check + visual indicator of
whether the lattice was built with open (OBC) or periodic (PBC) boundary
conditions.

Run directly (edit the parameters in ``main()``), or import
``compute_cantor_ldos`` / ``plot_cantor_ldos`` for use elsewhere.

Boundary-condition check
-------------------------
``model.build_model(..., pbc=...)`` sets a flag deep in the hop-graph
construction; nothing stops that flag from silently disagreeing with what you
think you asked for (e.g. if it's overridden downstream, or the wrong model
is loaded from cache). ``check_boundary_condition`` sidesteps trusting the
flag and instead inspects the *assembled Hamiltonian itself*: it looks at the
matrix block connecting the first and last active site. For a 1D nearest-
neighbor Wilson-Dirac chain, that block is nonzero if and only if the +1
hopping channel actually wraps around -- i.e. PBC is really in effect.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from project_tools import model, lattice, observables


# ── boundary-condition check ─────────────────────────────────────────────────

def check_boundary_condition(m, pbc_expected, *, t=1.0):
    """Inspect H for a hop directly linking the first and last active site.

    Returns ``(pbc_detected, ok)`` where ``pbc_detected`` is True iff that
    corner block of the Hamiltonian is nonzero, and ``ok`` is True iff this
    matches ``pbc_expected``.
    """
    probe_params = dict(M=1.0, M_alt=1.0, t=t, B=1.0,
                         disorder_strength=0.0, disorder_seed=0)
    H = m.assemble(True, format="csr", **probe_params)

    d = m.internal.dim
    n_active = m.active_hilbert_dim // d
    if n_active < 2:
        raise ValueError("need at least 2 active sites to check boundary conditions")

    corner = H[0:d, (n_active - 1) * d: n_active * d]
    corner = corner.toarray() if hasattr(corner, "toarray") else np.asarray(corner)
    pbc_detected = bool(np.abs(corner).sum() > 1e-10)
    return pbc_detected, (pbc_detected == pbc_expected)


# ── near-zero state selection (degeneracy-aware) ─────────────────────────────

def select_near_zero_indices(eigenvalues, k_center, tol=1e-8):
    """Indices of (at least) the ``k_center`` states closest to E=0.

    ``k_center`` is a *target*, not a hard cutoff: sorting purely by |E| and
    slicing at ``k_center`` risks cutting a degenerate multiplet in half
    (e.g. several exact zero modes, or a doubly-degenerate edge state pair
    under PBC). Instead, find the |E| of the k_center-th closest state and
    then include *every* state within ``tol`` of that boundary energy, so a
    degenerate subspace at the edge of the selection is always kept whole.
    """
    eigenvalues = np.asarray(eigenvalues)
    if k_center < 1:
        raise ValueError(f"k_center must be a positive integer; got {k_center}.")
    k_center = min(int(k_center), eigenvalues.size)

    abs_e = np.abs(eigenvalues)
    order = np.argsort(abs_e)
    boundary_energy = abs_e[order[k_center - 1]]

    keep = abs_e <= boundary_energy + tol
    idxs = np.nonzero(keep)[0]
    return idxs[np.argsort(abs_e[idxs])]


# ── compute ───────────────────────────────────────────────────────────────

def compute_cantor_ldos(n, M, M_alt, *, block_scale=1, pbc=False,
                         hole_treatment="substituted", k_center=2,
                         degeneracy_tol=1e-8, t=1.0, B=1.0):
    """Build the 1D Cantor model, solve for near-zero LDOS, run the BC check.

    ``k_center`` is the *target* number of near-zero states to sum LDOS over;
    if the states at that cutoff are degenerate (within ``degeneracy_tol``),
    the entire degenerate subspace is kept, so the actual number of states
    used (``len(data['highlighted_idxs'])``) may exceed ``k_center``.

    Returns a dict with the fractal mask ``l``, summed near-zero ``ldos``,
    the ``energies`` of the states summed over, the full spectrum, the
    selected-state indices, and the boundary-condition check results.
    """
    m = model.build_model("cantor", n, hole_treatment=hole_treatment,
                           pbc=pbc, block_scale=block_scale)

    pbc_detected, bc_ok = check_boundary_condition(m, pbc, t=t)

    params = dict(M=M, M_alt=M_alt, t=t, B=B,
                  disorder_strength=1e-5, disorder_seed=0)

    if hole_treatment == "renorm":
        r = m.solve_schur(eliminate_label="sector", eliminate_value=0, energy=0.0,
                          hermitian=True, k=None,
                          return_eigenvalues=True, return_eigenvectors=False,
                          return_LDOS=True, params=params)
    else:
        r = m.solve(hermitian=True, k=None, return_eigenvalues=True,
                    return_eigenvectors=False, return_LDOS=True, params=params)


    p = observables.polarization(m, params)
    print(f"M={M} : M_alt={M_alt} : p={p:.3e}")

    w = np.asarray(r["eigenvalues"])          # ascending (eigh convention)
    ldos_all = np.asarray(r["LDOS"])           # shape (n_states, L), aligned with w

    ph_symmetric = bool(np.allclose(np.sort(w), -np.sort(w)[::-1], atol=1e-6))

    idxs = select_near_zero_indices(w, k_center, tol=degeneracy_tol)
    ldos = np.nan_to_num(ldos_all[idxs], nan=0.0).sum(axis=0)

    l = lattice.build_lattice("cantor", n, block_scale=block_scale)

    return dict(
        l=l, ldos=ldos, energies=w[idxs],
        eigenvalues=w, highlighted_idxs=idxs,
        pbc_requested=bool(pbc), pbc_detected=pbc_detected, bc_ok=bc_ok,
        ph_symmetric=ph_symmetric,
        M=M, M_alt=M_alt, n=n, block_scale=block_scale,
        hole_treatment=hole_treatment, k_center_requested=k_center,
    )


# ── plot ──────────────────────────────────────────────────────────────────

def plot_cantor_ldos(data, ax=None):
    l = data["l"]
    ldos = np.nan_to_num(np.asarray(data["ldos"]), nan=0.0)
    L = l.size
    x = np.arange(L)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(11, 4.5))

    ymax = float(ldos.max()) * 1.2 if ldos.max() > 0 else 1.0

    # Background shading: dark = fractal site present, light = removed "hole".
    ax.imshow(l[np.newaxis, :], aspect="auto", cmap="Greys", alpha=0.15,
              extent=(-0.5, L - 0.5, 0, ymax), zorder=-1)

    ax.plot(x, ldos, lw=1.3, marker='.', ms=4, color='crimson', zorder=2)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("site index")
    ax.set_ylabel("summed near-zero LDOS")

    bc_label = "PBC" if data["pbc_requested"] else "OBC"
    check_word = "OK" if data["bc_ok"] else "MISMATCH!"
    ph_word = "OK" if data["ph_symmetric"] else "MISMATCH!"
    ax.set_title(
        f"Cantor set  n={data['n']}  L={L}  method={data['hole_treatment']}\n"
        f"M={data['M']:.2f}, M_alt={data['M_alt']:.2f}   |   "
        f"requested {bc_label}, wrap-hop detected={data['pbc_detected']} [{check_word}]   |   "
        f"particle-hole symmetry [{ph_word}]"
    )

    # Visual boundary-condition indicator.
    y_top = ymax
    if data["pbc_requested"]:
        arrow = FancyArrowPatch((L - 1, y_top * 0.93), (0, y_top * 0.93),
                                 connectionstyle="arc3,rad=-0.35",
                                 arrowstyle="-|>", mutation_scale=15,
                                 color="steelblue", lw=1.8, zorder=3,
                                 clip_on=False)
        ax.add_patch(arrow)
        ax.annotate("periodic wrap (site 0 <-> site L-1)",
                    xy=((L - 1) / 2, y_top * 1.03), ha="center",
                    color="steelblue", fontsize=9)
    else:
        for xi in (0, L - 1):
            ax.plot(xi, y_top * 0.93, marker='x', ms=10,
                    color='steelblue', mew=2, zorder=3, clip_on=False)
        ax.annotate("open boundaries (no wrap)",
                    xy=((L - 1) / 2, y_top * 1.03), ha="center",
                    color="steelblue", fontsize=9)

    if not data["bc_ok"]:
        ax.text(0.5, 0.5, "BOUNDARY-CONDITION CHECK FAILED",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=18, color="red", alpha=0.55, rotation=20, zorder=5)

    fig.tight_layout()
    return fig, ax


def plot_spectrum(data, ax=None):
    """Scatter the full eigenvalue spectrum, highlighting the near-zero
    states whose LDOS was summed in the companion LDOS panel."""
    w = data["eigenvalues"]
    idxs = data["highlighted_idxs"]
    x = np.arange(w.size)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 4.5))

    ax.scatter(x, w, s=10, color="0.5", zorder=2, label="full spectrum")
    ax.scatter(x[idxs], w[idxs], s=45, color="crimson", zorder=3,
               label=f"{len(idxs)} near-zero states "
                     f"(requested {data['k_center_requested']})")
    ax.axhline(0.0, c="k", lw=0.8, ls="--", zorder=1)

    ax.set_xlabel("state index")
    ax.set_ylabel("energy")
    ax.set_title("Spectrum")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig, ax


def plot_cantor_summary(data):
    """Two-panel figure: LDOS (with BC checks) on top, spectrum below."""
    fig, (ax_ldos, ax_spec) = plt.subplots(
        2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1.1, 1.0]}
    )
    plot_cantor_ldos(data, ax=ax_ldos)
    plot_spectrum(data, ax=ax_spec)
    fig.tight_layout()
    return fig, (ax_ldos, ax_spec)


# ── entry point ───────────────────────────────────────────────────────────

def main():
    # ── parameters ────────────────────────────────────────────────────────
    n = 4                     # Cantor generation
    block_scale = 5           # linear-size multiplier -> L = block_scale * 3**n
    M = -1.0         # mass on fractal (sector-1) sites
    M_alt = 2.0              # mass on complement (sector-0) sites
    pbc = True                # False -> OBC, True -> PBC
    hole_treatment = "substituted"   # 'substituted' | 'site_elim' | 'renorm'
    k_center = 2               # target number of near-zero states to sum LDOS over
    degeneracy_tol = 1e-8       # |E| tolerance for treating states as degenerate

    data = compute_cantor_ldos(n, M, M_alt, block_scale=block_scale, pbc=pbc,
                                hole_treatment=hole_treatment, k_center=k_center,
                                degeneracy_tol=degeneracy_tol)

    print(f"Boundary condition requested : {'PBC' if pbc else 'OBC'}")
    print(f"Wrap-around hop detected     : {data['pbc_detected']}")
    print(f"BC check                     : {'PASSED' if data['bc_ok'] else 'FAILED'}")
    print(f"Particle-hole symmetry check : {'PASSED' if data['ph_symmetric'] else 'FAILED'}")
    n_sel = len(data['highlighted_idxs'])
    extra = f" (expanded from {k_center} due to degeneracy)" if n_sel != k_center else ""
    print(f"Near-zero states summed      : {n_sel}{extra}")
    print(f"Near-zero energies summed    : {np.round(data['energies'], 6)}")

    plot_cantor_summary(data)
    plt.show()


if __name__ == "__main__":
    main()
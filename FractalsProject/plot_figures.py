import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import scipy.linalg as spla

from project_tools import model, lattice, io
from compute_ltm_sponge import plot_3d_voxels

rcParams["axes.linewidth"] = 3.5
rcParams["xtick.major.width"] = 3.5
rcParams["ytick.major.width"] = 3.5
rcParams["xtick.major.size"] = 5.0
rcParams["ytick.major.size"] = 5.0


def compute_projector(eigenvalues, eigenvectors):
    """Compute the projector onto the lower band of the Hamiltonian."""
    lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2] # Lower band eigenvalues
    highest_lower_band = lower_band[-1] # Highest eigenvalue in the lower band

    D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j) # Projector diagonal matrix
    D_herm_conj = np.einsum('i,ij->ij', D, eigenvectors.conj().T)
    projector = eigenvectors @ D_herm_conj # Projector matrix
    return projector


def compute_bott_index(X, Y, projector:np.ndarray):
    """Compute the Bott index for the given projector."""

    # Repeated (two orbitals)
    X = np.repeat(X, 2)
    Y = np.repeat(Y, 2)
    Lx = np.max(X) - np.min(X) # length of the x-direction
    Ly = np.max(Y) - np.min(Y)

    x_unitary = np.exp(1j * 2 * np.pi * X / Lx) # unitary operator in the x-direction
    y_unitary = np.exp(1j * 2 * np.pi * Y / Ly)
    x_unitary_proj = np.einsum('i,ij->ij', x_unitary, projector) # projector in the x-direction
    y_unitary_proj = np.einsum('i,ij->ij', y_unitary, projector)
    x_unitary_dagger_proj = np.einsum('i,ij->ij', x_unitary.conj(), projector)  # projector in the x-direction (dagger)
    y_unitary_dagger_proj = np.einsum('i,ij->ij', y_unitary.conj(), projector)

    I = np.eye(projector.shape[0], dtype=np.complex128) 
    A = I - projector + projector @ x_unitary_proj @ y_unitary_proj @ x_unitary_dagger_proj @ y_unitary_dagger_proj # BI operator given in arxiv:2407.13767 [Eq. (5)]
    bott_index = np.imag(np.sum(np.log(spla.eigvals(A)))) / (2 * np.pi)
    return bott_index


def compute_ldos(eigenvalues, ldos, tol=1e-10):
    abs_energies = np.abs(eigenvalues)
    min_E = np.min(abs_energies)

    # 2. Degeneracy failsafe: Capture ALL states within numerical tolerance 
    # of the closest energy level (e.g., multiple edge modes or flat bands).
    idxs = np.where(abs_energies <= min_E + tol)[0]

    # Fallback: If the gap is clean, non-degenerate, and shifted, ensure 
    # we still capture at least the two closest states (HOMO/LUMO equivalent).
    if len(idxs) < 2 and len(eigenvalues) >= 2:
        idxs = np.argsort(abs_energies)[:2]

    # Slice COLUMNS (eigenstates), and sum across the entire degenerate subspace
    ldos = np.sum(np.abs(ldos[idxs, ...]) ** 2, axis=0)

    return ldos, idxs


# ── caching ──────────────────────────────────────────────────────────────────
# ``compute()`` below is the expensive step (full diagonalization, plus a
# Bott-index eigenvalue solve for d==2), so its results are cached to disk
# under the project's standard data/ layout via ``project_tools.io``
# (kind="ldos"; see io.py for the on-disk layout and file format).
#
# The cache key is every parameter that can change the result: d/fractal, n,
# b (block_scale), M, M_alt, pasted, pbc, and (for d==3) M_prime/gauge. The
# other model params (disorder_seed, disorder_strength, t, B, and g for d==3)
# are hardcoded constants in this module rather than arguments to compute(),
# so they can't vary and are intentionally left out of the key — if they
# ever become configurable here, add them to ``_cache_meta`` too, or cached
# results will silently mix runs from different parameter values.
_CACHE_KIND = "ldos"
_CACHE_KEYS = ("l", "eigenvalues", "ldos", "idxs", "bott")


def _cache_meta(d, fractal, n, b, M, M_alt, pasted, pbc):
    """Build the metadata dict identifying one ``compute()`` run, used both
    as the cache filename (via ``io.case_tag``) and the embedded ``__meta__``."""
    meta = dict(
        fractal=fractal, method="substituted", n=int(n),
        L=lattice.system_length(n, block_scale=b, pasted=pasted),
        M=float(M), M_alt=float(M_alt),
        pasted=bool(pasted), pbc=bool(pbc),
    )
    if d == 3:
        meta.update(M_prime=0.0, gauge="N")
    return meta


def _load_cached(meta, *, root=None):
    """Return the cached ``(l, eigenvalues, ldos, idxs, bott)`` for ``meta``,
    or ``None`` on any cache miss — file absent, unreadable, or missing an
    expected array — so the caller can transparently fall back to recomputing
    rather than crashing on a stale or partially-written cache file."""
    path = io.result_path(_CACHE_KIND, meta, root=root)
    if not path.exists():
        return None
    try:
        arrays, _saved_meta = io.load_result(path)
        missing = [k for k in _CACHE_KEYS if k not in arrays]
        if missing:
            raise KeyError(f"cache is missing array(s) {missing}")
        return tuple(arrays[k] for k in _CACHE_KEYS)
    except Exception as e:                                  # noqa: BLE001
        print(f"  WARNING: could not read cache {path} ({e}); recomputing.")
        return None


def _save_cached(meta, *, root=None, **arrays):
    """Best-effort cache write: a failure here (e.g. read-only filesystem,
    disk full) should never prevent ``compute()`` from returning its result."""
    try:
        path = io.save_result(_CACHE_KIND, meta, root=root, **arrays)
        print(f"  Cached: {path}")
    except Exception as e:                                  # noqa: BLE001
        print(f"  WARNING: failed to write cache for {meta}: {e}")


def compute(d, n, M, M_alt, b=1, pasted=False, pbc=False, *,
            use_cache=True, force_recompute=False, cache_root=None):
    """Diagonalize the model and return ``(l, eigenvalues, ldos, idxs)``.

    Results are cached under the project's ``data/ldos/`` tree (see
    ``project_tools.io``), keyed by every parameter that affects the
    diagonalization. By default a matching cache entry is read instead of
    recomputing, and a freshly-computed result is written back for next time.

    Parameters
    ----------
    use_cache : bool
        If ``False``, neither read nor write the cache — always recompute.
    force_recompute : bool
        If ``True``, ignore any existing cache entry but still overwrite it
        with the freshly-computed result (use this to refresh a stale cache).
    cache_root : path-like, optional
        Overrides ``io.DATA_ROOT`` for both reading and writing, e.g. to
        point at a scratch directory for a one-off run.
    """
    if d == 1:
        fractal = 'cantor'
    elif d == 2:
        fractal = 'carpet'
    elif d == 3:
        fractal = 'sponge'
    else:
        raise ValueError(f"unsupported d={d!r}; expected 1 (cantor), 2 (carpet), or 3 (sponge).")

    meta = _cache_meta(d, fractal, n, b, M, M_alt, pasted, pbc)

    if use_cache and not force_recompute:
        cached = _load_cached(meta, root=cache_root)
        if cached is not None:
            l, eigenvalues, ldos, idxs, bott = cached
            if d == 2:
                print(f"{pbc} {M:+.2f} {M_alt:+.2f} {bott:+.2f}  [cached]")
            return l, eigenvalues, ldos, idxs

    m = model.build_model(fractal, n=n, block_scale=b, pasted=pasted, pbc=pbc, hole_treatment='substituted')

    params = {"M": M, "M_alt": M_alt, "disorder_seed": 0, "disorder_strength": 0.0, "t": 1.0, "B": 1.0}
    if d == 3:
        params["gauge"] = "N"
        params["g"] = 0
        params["M_prime"] = 0.0
    res = m.solve(**params, return_LDOS=True)

    l = lattice.build_lattice(fractal, n, block_scale=b, pasted=pasted)

    eigenvalues, ldos = res["eigenvalues"], res["LDOS"]

    bott = np.nan
    if d == 2:
        eigenvectors = res["eigenvectors"]
        P = compute_projector(eigenvalues, eigenvectors)
        X, Y = np.where(l >= 0)[:]
        bott = compute_bott_index(X, Y, P)
        print(f"{pbc} {M:+.2f} {M_alt:+.2f} {bott:+.2f}")

    ldos, idxs = compute_ldos(eigenvalues, ldos)

    if use_cache:
        _save_cached(meta, root=cache_root, l=l, eigenvalues=eigenvalues,
                     ldos=ldos, idxs=idxs, bott=bott)

    return l, eigenvalues, ldos, idxs


def plot_spectrum(ax, eigenvalues, idxs):
    ax.scatter(np.arange(eigenvalues.size), eigenvalues, zorder=4)
    ax.scatter(np.arange(eigenvalues.size)[idxs], eigenvalues[idxs], zorder=5, c='r')


def get_ndim_diagonal_indices(arr):
    smallest_dim = np.min(arr.shape)

    mask = np.full(arr.shape, False)
    for i in range(smallest_dim):
        mask[*[v for v in [i]*arr.ndim]] = True
    return mask


def plot_ldos(ax, D, l, ldos, cmap='bwr', do_imshow=False, do_zoom=True, zoom_center=False):
    if not do_imshow or D == 1:
        mask = get_ndim_diagonal_indices(l)
        t = np.arange(np.sum(mask))
        lattice_coloring = l[mask]
        ax.scatter(t, ldos[mask], s=50, c=lattice_coloring, cmap=cmap)
        ax.set_xticks([t.min(), (t.max() + t.min()) / 2, t.max()])
        ax.set_xticklabels([t.min() + 1, (t.max() + t.min()) / 2 + 1, t.max() + 1])


        if do_zoom and D == 1:
            proportion_of_sites = 1.5 / 27 
            inset = ax.inset_axes([0.56, 0.56, 0.40, 0.38])#, xlim=(xmin, xmax), xticklabels=[], ylim=(ymin, ymax), yticklabels=[])
            inset_idxs = np.arange(int(len(t) * proportion_of_sites))
            if zoom_center:
                inset_idxs -= int(np.mean(inset_idxs)) + len(t) // 2
            
            t_inset = t[inset_idxs]
            ldos_inset = ldos[mask][inset_idxs]
            lattice_coloring = l[mask][inset_idxs]

            inset.scatter(t_inset, ldos_inset, s=28, c=lattice_coloring, cmap=cmap)
            inset.margins(y=0.2, x=0.05)

            inset.set_xticks([t_inset.min(), (t_inset.max() + t_inset.min()) / 2, t_inset.max()])
            inset.set_xticklabels([t_inset.min() + 1, (t_inset.max() + t_inset.min()) / 2 + 1, t_inset.max() + 1])




        return

    if D == 2 and do_imshow:
        X, Y = np.where(l > -100)[:]
        mappable = ax.imshow(ldos, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
        ax.set_xticks([X.min(), X.max()])
        ax.set_xticklabels([X.min() + 1, X.max() + 1], fontsize=16)
        ax.set_yticks([Y.min(), Y.max()])
        ax.set_yticklabels([Y.min() + 1, Y.max() + 1], fontsize=16)

    if D == 3 and do_imshow:
        mappable = plot_3d_voxels(l > -100, ldos, cmap, ax=ax, edgecolors='k', plot_diagonal_half=False, alpha=1.0)

    if do_imshow: plt.colorbar(mappable, ax=ax, ticks=[ldos.min(), ldos.max()])
    return


def make_figure(D, fig_index=0, spectrum_plot=False, do_imshow=False):
    if do_imshow and D == 3:
        fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    else:
        fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharey=True, sharex=True)

    cantor_params1 = {"n": 4, "b": 5, "pasted": False, "m_topological1": 2.0, "m_trivial1": -0.5, "m_topological2": -0.5, "m_trivial2": 2.0}
    cantor_params2 = {"n": 4, "b": 5, "pasted": False, "m_topological1": 2.0, "m_trivial1": 5.0, "m_topological2": 5.0, "m_trivial2": 2.0}

    carpet_params1 = {"n": 3, "b": 2, "pasted": False, "m_topological1": 2.0, "m_trivial1": -0.5, "m_topological2": -1.0, "m_trivial2": 1.0}
    carpet_params2 = {"n": 3, "b": 2, "pasted": False, "m_topological1": 6.0, "m_trivial1": 8.5, "m_topological2": 9.0, "m_trivial2": 7.0}

    sponge_params1 = {"n": 1, "b": 2, "pasted": False, "m_topological1": 2.0, "m_trivial1": -0.5, "m_topological2": 2.0, "m_trivial2": -0.5}
    sponge_params2 = {"n": 1, "b": 2, "pasted": False, "m_topological1": 2.0, "m_trivial1": -0.5, "m_topological2": 2.0, "m_trivial2": -0.5}

    p1s = [cantor_params1, carpet_params1, sponge_params1]
    p2s = [cantor_params2, carpet_params2, sponge_params2]
    n, b, pasted, m_topological1, m_trivial1, m_topological2, m_trivial2 = [p1s, p2s][fig_index][D - 1].values()




    bcs = np.array([[False, True],
                    [False, True]])
    m0s = np.array([[m_topological1, m_topological1],
                    [m_topological2, m_topological2]])
    
    m_alts = np.array([[m_trivial1, m_trivial1],
                       [m_trivial2, m_trivial2]])

    for i in range(2):
        for j in range(2):
            print(i,j,bcs[i,j])
            l, eigenvalues, ldos, idxs = compute(D, n, m0s[i, j], m_alts[i, j], b, pasted, pbc=bool(bcs[i, j]))
            if spectrum_plot:
                plot_spectrum(axs[i, j], eigenvalues, idxs)
            else:
                plot_ldos(axs[i, j], D, l, ldos, do_imshow=do_imshow, zoom_center=False if (i+j == 0) else False)
            axs[i, j].annotate(
                xy=(0.05, 0.95),
                ha='left',
                va='top',
                text=f"{eigenvalues[idxs]}",
                xycoords='axes fraction',
            )
            bc_label = "OBC" if bcs[i, j] == False else "PBC"
            axs[i, j].set_title(f"{bc_label} $M_0={m0s[i, j]}$ and $M_{{\\rm alt}}={m_alts[i, j]}$")


    label = ['cantor', 'carpet', 'sponge'][D-1]
    plt.savefig(f'figures/ldos/{label}_ldos{fig_index+1:.0f}.svg', dpi=300)
    #plt.show()


def plot_all_carpet(spectrum_plot=False, do_imshow=True):
    n = 3; b = 2
    m0s =   [+2.00, +0.25, ]#+2.00, +0.25, -0.50, -0.50]
    malts = [+6.00, +6.00, ]#-0.50, +8.50, +2.00, +6.00]

    fig, axs = plt.subplots(len(m0s), 2, figsize=(8 * 2, 4 * len(m0s)), sharey=True, sharex=True)
    
    for i in range(len(m0s)):
        for j in range(2):
            l, eigenvalues, ldos, idxs = compute(2, n, m0s[i], malts[i], b, False, pbc=[False, True][j])
            if spectrum_plot:
                plot_spectrum(axs[i, j], eigenvalues, idxs)
            else:
                plot_ldos(axs[i, j], 2, l, ldos, do_imshow=do_imshow, zoom_center=False, cmap='cividis')
            axs[i, j].annotate(
                xy=(0.05, 0.95),
                ha='left',
                va='top',
                text=f"{eigenvalues[idxs]}",
                xycoords='axes fraction',
            )

            bc_label = "OBC" if j == 0 else "PBC"
            axs[i, j].set_title(f"{bc_label} $M_0={m0s[i]}$ and $M_{{\\rm alt}}={malts[i]}$")

    plt.savefig(f'figures/ldos/carpet_test.svg', dpi=300)


plot_all_carpet(False, True)
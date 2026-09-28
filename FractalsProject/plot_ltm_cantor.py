"""Plotting for the 1D Cantor-chain local topological marker (LTM) and LDOS.

Reads results from ``compute_ltm_cantor.compute_wrapper`` (which handles its
own on-disk caching) and lays them out on the fractal site axis, using
``brokenaxes`` to skip over the large stretches of the Cantor chain that
carry no signal.

Public API (unchanged names/signatures from the original module):
    get_ltm_data(...), get_ldos_data(...)      -- pull one (M, M_alt) result
    plot_local_topological_marker(...)         -- single-run broken-axis plot
    plot_ldos_imshow(...)                      -- LDOS heatmap over a M sweep
    compute_broken_axes_limits(...), compute_cantor_lims(...)
    plot_on_cantor_set(...)                    -- the main multi-M figure
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from brokenaxes import brokenaxes

from project_tools import lattice
from compute_ltm_cantor import compute_wrapper

FIGURE_ROOT = Path("./figures")


# ── per-run data extraction ─────────────────────────────────────────────────

def _project_to_lattice(values, l, method):
    """Place per-active-site ``values`` onto the full L-site lattice axis.

    For ``site_elim``/``renorm`` (and their ``_alt`` variants) the model only
    has degrees of freedom on the kept (sector-1) sites, so ``values`` is
    scattered into those positions and everything else is left as NaN. For
    ``substituted``/``substituted_alt`` every site is active, so ``values``
    already covers the full chain.
    """
    L = l.size
    if method in ("site_elim", "site_elim_alt", "renorm", "renorm_alt"):
        y = np.full(L, np.nan)
        y[l.astype(bool)] = values
    else:
        y = np.asarray(values)
    return np.arange(L), y


def get_ltm_data(n, b, M, method, pbc, M_alt=None, overwrite=False):
    """Local topological marker C(x) vs site index for one (M, M_alt) run."""
    C, _, _ = compute_wrapper(n, b, M, method, M_alt=M_alt, overwrite=overwrite, pbc=pbc)
    c_diag = np.diag(C)
    c_diag = c_diag[::2] + c_diag[1::2]  # trace over the 2-component internal index
    l = lattice.build_lattice("cantor", n, block_scale=b)
    return _project_to_lattice(c_diag, l, method)


def get_ldos_data(n, b, M, method, pbc, M_alt=None, overwrite=False):
    """Near-zero-mode LDOS vs site index for one (M, M_alt) run."""
    _, eigvals, ldos = compute_wrapper(n, b, M, method, M_alt=M_alt, overwrite=overwrite, pbc=pbc)
    fig, ax = plt.subplots(1,1)
    ax.scatter(np.arange(eigvals.size), eigvals)
    plt.show()
    l = lattice.build_lattice("cantor", n, block_scale=b)
    return _project_to_lattice(ldos, l, method)


# ── single-run plot ──────────────────────────────────────────────────────

def plot_local_topological_marker(n, b, M, method, pbc, M_alt=None, fig=None):
    """Broken-axis plot of the local topological marker for one run.

    ``fig`` is passed through to ``brokenaxes`` (a new figure is created if
    omitted); the x-axis breaks are computed the same way as in
    ``plot_on_cantor_set``, via ``compute_cantor_lims``, so both figures skip
    over the same "dead" stretches of the chain consistently.
    """
    l = lattice.build_lattice("cantor", n, block_scale=b)
    t, y = get_ltm_data(n, b, M, method, pbc, M_alt=M_alt)

    xlims, ylims = compute_cantor_lims(n, b, [np.abs(y)])
    if method in ("substituted", "substituted_alt"):
        xlims = [(0, l.size)]

    fig = fig or plt.figure()
    bax = brokenaxes(xlims=xlims, ylims=ylims, fig=fig)
    extent = (0.0, l.size, ylims[0][0], ylims[-1][1])
    bax.imshow(l[np.newaxis], aspect="auto", cmap="Greys", alpha=0.25,
               zorder=-1, extent=extent)
    bax.plot(t, y)
    bax.axhline(-1.0, c="k", ls="--", zorder=-10, alpha=0.5)
    bax.fig.suptitle(f"{method}\nn={n}, L={l.size}, M={M}, M_alt={M_alt}")
    return bax


# ── LDOS heatmap over a mass sweep ──────────────────────────────────────

def plot_ldos_imshow(n, b, M_values, method, pbc, M_alt=None, ax=None, cmap="viridis"):
    """Heatmap of near-zero LDOS vs (M, site index) over a sweep of M_values."""
    l = lattice.build_lattice("cantor", n, block_scale=b)
    L = l.size

    grid = np.full((len(M_values), L), np.nan)
    for i, M in enumerate(M_values):
        _, y = get_ldos_data(n, b, M, pbc, method, M_alt=M_alt)
        grid[i] = y

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    masked = np.ma.masked_invalid(grid)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")

    im = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap_obj,
                    extent=(0, L, M_values[0], M_values[-1]))
    fig.colorbar(im, ax=ax, label="LDOS")
    ax.set_xlabel("site index")
    ax.set_ylabel("$M$")
    ax.set_title(f"{method} : n={n} : L={L}")
    return fig, ax


# ── broken-axis limit helpers ────────────────────────────────────────────

def compute_broken_axes_limits(arr, keep_points=None, threshold=0.01,
                                jump_threshold=0.05, extrema_pad: float = 0.01):
    """Auto-detect large gaps in ``arr`` and return broken-axis segments.

    Splits the range of ``arr`` at every jump between consecutive unique
    values that exceeds ``jump_threshold`` (as a fraction of the full
    range), leaving a small margin (``threshold``, also a fraction of the
    range) on either side of each break. ``keep_points`` are values (e.g.
    ``0.0`` for a y-axis) that must end up strictly inside some segment,
    even if that means inserting an extra break around them.
    """
    keep_points = keep_points or []
    arr = arr[~np.isnan(arr)]
    arr_unique = np.unique(np.round(arr, 6))
    arr_range = np.nanmax(arr) - np.nanmin(arr)

    jump_idxs = np.argwhere(np.diff(arr_unique) / arr_range >= jump_threshold)
    edges = np.sort(np.concatenate((arr_unique[jump_idxs], arr_unique[jump_idxs + 1])).flatten())
    edges += np.tile([threshold * arr_range, -threshold * arr_range], edges.size // 2)
    edges = [arr_unique[0] - arr_range * extrema_pad, arr_unique[-1] + arr_range * extrema_pad] + edges.tolist()
    edges = np.sort(edges)

    add_points, remove_points = [], []
    for kp in keep_points:
        already_inside = any(edges[i] < kp < edges[i + 1] for i in range(0, edges.size, 2))
        if already_inside:
            continue
        xi, xj = kp - threshold * arr_range, kp + threshold * arr_range
        conflicting = [e for e in edges if xi < e < xj]
        remove_points.extend(conflicting)
        add_points.append(xj if conflicting else xi)
        if not conflicting:
            add_points.append(xj)

    edges = [e for e in edges if e not in remove_points] + add_points
    edges = np.sort(edges)
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]


def compute_cantor_lims(n: int, b: int, C_r_arrays: list, x_threshold=0.01, y_threshold=0.01):
    """Broken x/y-axis limits for a Cantor-chain plot: x breaks skip the
    fractal's removed (hole) sites, y breaks skip large jumps in the data."""
    l = lattice.build_lattice("cantor", n, block_scale=b)
    x = np.flatnonzero(l.astype(bool)).astype(float)
    xlims = compute_broken_axes_limits(x, threshold=x_threshold,
                                        jump_threshold=3 ** (-n), extrema_pad=0.001)

    all_values = np.concatenate(C_r_arrays).flatten()
    ylims = compute_broken_axes_limits(all_values, keep_points=[0.0],
                                        threshold=y_threshold, jump_threshold=0.2,
                                        extrema_pad=0.01)
    return xlims, ylims


# ── zoom inset ────────────────────────────────────────────────────────────

def _add_region_inset(fig, l, data, M_values, region, *, bbox=(0.64, 0.6, 0.33, 0.33),
                       colors=("k", "r", "b", "g"), markers=(".", "s", "^", "v")):
    """Add an inset axes (placed at ``bbox`` in figure-fraction coords) that
    zooms into ``region = (x0, x1)`` of the site-index axis, replotting every
    M-value series there. Independent of the broken-axis panels (brokenaxes
    doesn't expose one continuous coordinate space to anchor a normal
    inset_axes/mark_inset call to), so it's placed directly on the figure.
    """
    x0, x1 = region
    t = np.arange(l.size)
    mask = (t >= x0) & (t <= x1)

    axins = fig.add_axes(bbox)
    y_local_max = 0.0
    for i, y in enumerate(data):
        y = np.asarray(y)
        axins.plot(t[mask], y[mask], c=colors[i % len(colors)], marker=markers[i % len(markers)], zorder=(i+1)%2)
        finite = y[mask][~np.isnan(y[mask])]
        if finite.size:
            y_local_max = max(y_local_max, finite.max())
    y_top = y_local_max * 1.2 if y_local_max > 0 else 1.0

    #axins.imshow(l[np.newaxis, :], aspect="auto", cmap="Greys", alpha=0.2,
    #             extent=(x0, x1, 0, y_top), zorder=-1)
    axins.set_xlim(x0, x1)
    axins.set_ylim(-0.25, y_top)
    axins.set_title(f"zoom: sites {int(x0)}–{int(x1)}", fontsize=8)
    axins.tick_params(labelsize=6)
    axins.set_xticks([x0+1, (x1 + x0) / 2, x1-1])
    axins.set_xticklabels([str(int(t + 1)) for t in axins.get_xticks()])
    #for spine in axins.spines.values():
    #    spine.set_edgecolor("steelblue")
    #    spine.set_linewidth(1.1)
    return axins


# ── main multi-M figure ──────────────────────────────────────────────────

def plot_on_cantor_set(method, n, b, pbc, break_xax=True, break_yax=True,
                        data_func=get_ltm_data, M_values=(-1, 1, 3, 5),
                        overwrite=True, save=True,
                        zoom_region=None, zoom_bbox=(0.5-0.33/2, 0.25, 0.33, 0.33)):
    """Scatter |data_func(M)| vs site index for several M values, on one
    broken-axis figure. ``data_func`` is ``get_ltm_data`` or ``get_ldos_data``.

    ``zoom_region``, if given as ``(x0, x1)`` site indices, adds an inset
    (at ``zoom_bbox`` in figure-fraction coords) zooming into that window for
    all M_values series -- handy for resolving peak structure that the main
    broken-axis view compresses too much to see.
    """
    l = lattice.build_lattice("cantor", n, block_scale=b)
    data = [np.abs(data_func(n, b, M, method, pbc, overwrite=overwrite, M_alt=M)[1]) for M in M_values]
    if 'ltm' in data_func.__name__: data = [np.where((d > -0.25) & (d < 1.25), d, np.nan) for d in data]

    print(data[0])

    xlims, ylims = [(0, l.size)], None
    if break_xax or break_yax:
        broken_xlims, broken_ylims = compute_cantor_lims(n, b, data, x_threshold=0.001, y_threshold=0.001)
        if break_xax and method not in ("substituted", "substituted_alt"):
            xlims = broken_xlims
        if break_yax:
            ylims = broken_ylims
    if ylims is None:
        cmin, cmax = np.nanmin(data), np.nanmax(data)
        crange = cmax - cmin
        ylims = [(cmin - 0.1 * crange, cmax + 0.1 * crange)]

    fig = plt.figure(figsize=(20, 10))
    bax = brokenaxes(xlims=xlims, ylims=ylims, d=0.005, despine=True, fig=fig)

    colors = ["k", "r", "b", "g"]
    markers = ["^", "s", ".", "v"]
    markers = [".", "s", ".", "s"]
    sizes = [36, 50, 36, 50]
    zorders = [1, 0, 1, 0]
    for i, M in enumerate(M_values):
        #bax.scatter(np.arange(l.size), data[i],
        #            c=colors[i % len(colors)], marker=markers[i % len(markers)],
        #            s=sizes[i % len(sizes)], zorder=zorders[i % len(zorders)],
        #            label=f"$M={M}$")
        bax.plot(np.arange(l.size), data[i], c=colors[i % len(colors)], marker=markers[i % len(markers)], zorder = zorders[i % len(markers)], label=f"$M={M}$")

    axs = np.array(bax.axs).reshape(len(ylims), len(xlims))
    if break_xax:
        for ax in axs[-1, :]:
            xmin, xmax = ax.get_xlim()
            r = (xmax - xmin) / 5
            r = 2
            ax.set_xticks(np.round([xmin + r, xmax - r], 0))
            ax.set_xticklabels([str(int(t + 1)) for t in ax.get_xticks()])

    axs[0, -1].legend()

    if zoom_region is not None:
        _add_region_inset(fig, l, data, M_values, zoom_region, bbox=zoom_bbox, colors=colors, markers=markers)

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    func_dir = "ltm" if "ltm" in data_func.__name__ else "ldos"
    title = "Local Topological Marker" if func_dir == "ltm" else "Local Density of States"
    bc_tag = "PBC" if pbc else "OBC"
    fig.suptitle(f"{title}\n{method} : n={n} : L={l.size} : " + bc_tag)

    if save:
        out_dir = FIGURE_ROOT / func_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / (f"{method}_n={n}_L={l.size}_" + bc_tag)
        fig.savefig(f"{stem}.svg", transparent=True)
    return fig


# ── entry point ───────────────────────────────────────────────────────────

def main():
    rcParams["axes.linewidth"] = 3.5
    rcParams["xtick.major.width"] = 3.5
    rcParams["ytick.major.width"] = 3.5
    rcParams["xtick.major.size"] = 5.0
    rcParams["ytick.major.size"] = 5.0

    for n in (4,):
        for b in (27,):
            for method in ("renorm",):
                plot_on_cantor_set(method, n, b, pbc=False, break_xax=True, break_yax=False,
                                    data_func=get_ldos_data, overwrite=True, zoom_region=(0 - 1, b))
                plt.close()


if __name__ == "__main__":
    main()
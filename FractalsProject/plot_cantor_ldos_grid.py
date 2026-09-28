"""(n, 2) grid of Cantor-set near-zero LDOS panels.

Layout:
    columns: OBC (left), PBC (right)
    rows:    one row per (M, M_alt) pair, in the order given

Reuses ``compute_cantor_ldos`` from ``plot_cantor_ldos.py`` (degeneracy-aware
near-zero state selection + the boundary-condition sanity check), so every
panel here carries the same checks as the single-panel version.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib import rcParams

from plot_cantor_ldos import compute_cantor_ldos
from project_tools import observables

rcParams['axes.linewidth'] = 2.5
rcParams['xtick.major.width'] = 2.5
rcParams['ytick.major.width'] = 2.5

def _add_zoom_inset(ax, data, side="left", zoom_frac=0.15, bbox=None, x_center=-0.5):
    """Add an inset axes zooming into the LDOS peaks on one side of the chain.

    ``side`` is ``'left'``, ``'right'``, ``'middle'``, or ``'auto'`` (picks
    whichever half of the lattice has the larger peak). ``zoom_frac`` sets
    how much of the lattice (as a fraction of L) the inset window covers,
    from that edge (or centered on ``x_center`` for ``'middle'``/default).

    The connector corners (``loc1``/``loc2``) passed to ``mark_inset`` are
    computed automatically from the actual positions of the inset axes'
    bounding box and the highlighted (zoomed) region, rather than being
    hardcoded per ``side`` — see ``_auto_mark_inset_locs``.
    """
    l = data["l"]
    ldos = np.nan_to_num(np.asarray(data["ldos"]), nan=0.0)
    L = l.size
    x = np.arange(L)
    ymax = ax.get_ylim()[1]

    if side == "auto":
        mid = L // 2
        side = "left" if ldos[:mid].max() >= ldos[mid:].max() else "right"

    if side == "left":
        x0, x1 = -0.5, L * zoom_frac
        loc = "upper left"
    elif side == "right":
        x0, x1 = L * (1 - zoom_frac), L + -0.5
        loc = "upper right"
    elif side == "middle":
        x_center = L // 2
        x0 = x_center - L * zoom_frac / 2
        x1 = x_center + L * zoom_frac / 2
        loc = "upper left"
    else:
        x0 = x_center - L * zoom_frac / 2
        x1 = x_center + L * zoom_frac / 2
        loc = "upper left"

    lo = max(int(np.floor(x0)), 0)
    hi = min(int(np.ceil(x1)), L)
    y_local_max = ldos[lo:hi].max() if hi > lo else ymax
    y_inset_max = y_local_max * 1.2 if y_local_max > 0 else ymax

    if bbox is not None:
        axins = inset_axes(ax, width="100%", height="100%", loc="lower left",
                            bbox_to_anchor=bbox, bbox_transform=ax.transAxes,
                            borderpad=0)
        inset_bbox_axes_frac = bbox  # (x0, y0, width, height) in ax.transAxes fraction
    else:
        axins = inset_axes(ax, width="45%", height="45%", loc=loc, borderpad=1.1)
        inset_bbox_axes_frac = None  # resolved from `loc` inside the helper below
    axins.imshow(l[np.newaxis, :], aspect="auto", cmap="Greys", alpha=0.15,
                 extent=(-0.5, L - 0.5, 0, y_inset_max), zorder=-1)
    axins.plot(x, ldos, lw=2.0, marker='.', ms=10, color='crimson', zorder=2, rasterized=True)
    axins.set_xlim(x0, x1)
    axins.set_ylim(0, y_inset_max)
    axins.set_xticks(np.arange(x1, x1//3))
    axins.set_xticklabels([int(t + 1) for t in axins.get_xticks()])
    axins.tick_params(labelsize=6)
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2.0)

    loc1, loc2 = _auto_mark_inset_locs(
        ax, rect=(x0, x1, 0, y_inset_max),
        inset_bbox_axes_frac=inset_bbox_axes_frac, loc=loc,
    )
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="black", lw=1.5, ls='--')
    return axins


# Corners adjacent to (i.e. sharing an edge with) each numbered corner, in
# the {1: upper right, 2: upper left, 3: lower left, 4: lower right}
# convention shared by ``mark_inset``/``Legend``.
_ADJACENT_CORNERS = {1: (2, 4), 2: (1, 3), 3: (2, 4), 4: (1, 3)}

# Rough default centers (axes fraction) for the loc strings ``inset_axes``
# accepts, used only as a fallback when the inset's bbox isn't known exactly
# (i.e. it was placed via a ``loc`` string rather than an explicit bbox).
_LOC_STRING_CENTERS = {
    "upper left": (0.225, 0.775), "upper right": (0.775, 0.775),
    "lower left": (0.225, 0.225), "lower right": (0.775, 0.225),
    "upper center": (0.5, 0.775), "lower center": (0.5, 0.225),
    "center left": (0.225, 0.5), "center right": (0.775, 0.5),
    "center": (0.5, 0.5),
}


def _auto_mark_inset_locs(ax, rect, inset_bbox_axes_frac, loc="upper right"):
    """Pick ``mark_inset``'s ``loc1``/``loc2`` from where the inset axes and
    the highlighted region actually sit, instead of a hardcoded pair.

    ``mark_inset`` connects corner ``loc1`` of the highlighted rectangle to
    corner ``loc1`` of the inset axes' bbox (likewise for ``loc2``), using
    the usual 1=upper-right, 2=upper-left, 3=lower-left, 4=lower-right
    numbering. To avoid the connector lines crossing through the highlighted
    region or the inset itself, we use the two rectangle corners *adjacent*
    to whichever corner most directly faces the inset (i.e. the corners on
    the near edges, not the far corner or the near corner itself) — this is
    the same pairing matplotlib's own gallery examples use (e.g. an inset
    placed at ``loc=1`` pairs with ``mark_inset(..., loc1=2, loc2=4)``).

    Parameters
    ----------
    ax : the parent axes (used for its current data limits).
    rect : (x0, x1, y0, y1) of the highlighted region, in ``ax`` data coords.
    inset_bbox_axes_frac : (x0, y0, width, height) in ``ax``-fraction
        coordinates, or ``None`` to fall back to ``loc``'s rough position.
    loc : the ``loc`` string used to place the inset axes, used only when
        ``inset_bbox_axes_frac`` is ``None``.
    """
    x0, x1, y0, y1 = rect
    ax_x0, ax_x1 = ax.get_xlim()
    ax_y0, ax_y1 = ax.get_ylim()
    rect_cx_frac = ((x0 + x1) / 2 - ax_x0) / (ax_x1 - ax_x0)
    rect_cy_frac = ((y0 + y1) / 2 - ax_y0) / (ax_y1 - ax_y0)

    if inset_bbox_axes_frac is not None:
        ix0, iy0, iw, ih = inset_bbox_axes_frac
        inset_cx_frac = ix0 + iw / 2
        inset_cy_frac = iy0 + ih / 2
    else:
        inset_cx_frac, inset_cy_frac = _LOC_STRING_CENTERS.get(loc, (0.775, 0.775))

    right = inset_cx_frac >= rect_cx_frac
    up = inset_cy_frac >= rect_cy_frac
    # near corner of the rect that most directly faces the inset
    near_corner = {(True, True): 1, (False, True): 2, (False, False): 3, (True, False): 4}[(right, up)]
    loc1, loc2 = _ADJACENT_CORNERS[near_corner]
    return loc1, loc2


def _style_xaxis(ax, L, n_ticks=6, pad_frac=0.02):
    """Uniform x-ticks spanning [0, L], labeled 1-indexed, with edge padding
    so the data doesn't sit flush against the axes spines."""
    ticks = np.linspace(0, L, n_ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(round(t)) + 1) for t in ticks])
    pad = pad_frac * L
    ax.set_xlim(-0.5 - pad, L - 0.5 + pad)


def _plot_panel(ax, data, zoom_side="left", zoom_frac=0.15, zoom_bbox=(0.1, 0.2, 0.3, 0.3),
                 x_center=-0.5, ymax=None):
    l = data["l"]
    ldos = np.nan_to_num(np.asarray(data["ldos"]), nan=0.0)
    L = l.size
    x = np.arange(L)
    if ymax is None:
        ymax = float(ldos.max()) * 1.28 if ldos.max() > 0 else 1.0

    # background shading: fractal site (sector 1) vs complement (sector 0)
    ax.imshow(l[np.newaxis, :], aspect="auto", cmap="Greys", alpha=0.15,
              extent=(-0.5, L - 0.5, 0, ymax), zorder=-1)
    ax.plot(x, ldos, lw=2.0, marker='.', ms=10, color='crimson', zorder=2, rasterized=True)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(0, ymax)
    _style_xaxis(ax, L)

    bc_label = "PBC" if data["pbc_requested"] else "OBC"
    check_word = "OK" if data["bc_ok"] else "MISMATCH!"
    n_sel = len(data["highlighted_idxs"])

    ax.set_title(
        f"{bc_label}   $M={data['M']:.2f}$, $M_{{\\rm alt}}={data['M_alt']:.2f}$"
        f"   [{check_word}]",
        fontsize=10,
    )
    ax.annotate(
        f"{n_sel} near-zero state(s)\n$E$ = {np.round(data['energies'], 5)}",
        xy=(0.03, 0.95), xycoords="axes fraction", ha="left", va="top", fontsize=7.5,
    )

    if not data["bc_ok"]:
        ax.text(0.5, 0.5, "BC CHECK FAILED", transform=ax.transAxes, ha="center",
                va="center", fontsize=14, color="red", alpha=0.55, rotation=20, zorder=5)

    _add_zoom_inset(ax, data, side=zoom_side, zoom_frac=zoom_frac, bbox=zoom_bbox, x_center=x_center)


def plot_cantor_ldos_grid(cantor_n, block_scale, mass_pairs, *,
                           hole_treatment="substituted", k_center=2,
                           degeneracy_tol=1e-8, figsize=None,
                           zoom_side="left", zoom_frac=0.15, x_center=-0.5,
                           zoom_bbox=(0.1, 0.2, 0.3, 0.3),
                           inset_overrides=None):
    """Build an (n, 2) grid: columns=[OBC, PBC], one row per (M, M_alt) pair.

    Parameters
    ----------
    cantor_n : int
        The Cantor-set generation depth, passed straight through to
        ``compute_cantor_ldos`` (this is the ``n`` of the single-panel
        version, kept distinct here from the number of grid rows).
    block_scale : passed through to ``compute_cantor_ldos``.
    mass_pairs : sequence of (M, M_alt) tuples
        One entry per row. Row ``i`` is built from ``mass_pairs[i]``.
    figsize : (width, height), optional
        Defaults to ``(12, 4.5 * n_rows)`` if not given.

    Each panel gets an inset zooming into the near-zero LDOS peaks. By
    default every panel uses ``zoom_side`` (``'left'``, ``'right'``, or
    ``'auto'`` to pick whichever side has the larger peak in that panel),
    ``zoom_frac`` (how much of the lattice the inset window covers, from
    that edge) and ``zoom_bbox`` (the inset's location/size, in axes
    fraction coordinates ``(x0, y0, width, height)``).

    ``inset_overrides`` : dict, optional
        Per-panel overrides, keyed by ``(row, col)`` (``row`` = index into
        ``mass_pairs``, ``col`` = 0 for OBC / 1 for PBC). Each value is a
        dict that may set any of ``side``, ``zoom_frac``, ``bbox`` to
        override the corresponding global default for just that panel;
        keys omitted from a panel's dict fall back to the global default.
        For example, to zoom into the right edge of the PBC panel in row 2
        and use a wider window for the OBC panel in row 0::

            inset_overrides = {
                (2, 1): {"side": "right", "zoom_frac": 0.25},
                (0, 0): {"zoom_frac": 0.3, "bbox": (0.55, 0.2, 0.3, 0.3)},
            }

        Panels not present in ``inset_overrides`` use the global
        ``zoom_side`` / ``zoom_frac`` / ``zoom_bbox`` values unchanged.

    Returns ``(fig, axs, all_data)`` where ``all_data[(row, col)]`` is the
    dict returned by ``compute_cantor_ldos`` for that panel (row = index
    into ``mass_pairs``, col 0/1 = OBC/PBC). ``axs`` is always 2D, shape
    ``(n_rows, 2)``, even when ``n_rows == 1``.
    """
    n_rows = len(mass_pairs)
    if n_rows < 1:
        raise ValueError("mass_pairs must contain at least one (M, M_alt) pair")
    bc_cols = [False, True]  # col 0 = OBC, col 1 = PBC

    if figsize is None:
        figsize = (12, 4.5 * n_rows)

    fig, axs = plt.subplots(n_rows, 2, figsize=figsize, sharex=True, sharey=False,
                             squeeze=False)

    inset_overrides = inset_overrides or {}

    all_data = {}
    for i, (M, M_alt) in enumerate(mass_pairs):
        # Compute both columns of this row first, so OBC/PBC can share a
        # single y-axis range (and therefore the same tick locations).
        row_data = {}
        row_ymax = 1.0
        for j, pbc in enumerate(bc_cols):
            data = compute_cantor_ldos(cantor_n, M, M_alt, block_scale=block_scale, pbc=pbc,
                                        hole_treatment=hole_treatment, k_center=k_center,
                                        degeneracy_tol=degeneracy_tol)
            row_data[j] = data
            all_data[(i, j)] = data
            ldos = np.nan_to_num(np.asarray(data["ldos"]), nan=0.0)
            panel_max = float(ldos.max()) * 1.28 if ldos.max() > 0 else 1.0
            row_ymax = max(row_ymax, panel_max)

        for j, data in row_data.items():
            override = inset_overrides.get((i, j), {})
            panel_side = override.get("side", zoom_side)
            panel_zoom_frac = override.get("zoom_frac", zoom_frac)
            panel_bbox = override.get("bbox", zoom_bbox)
            panel_xcenter = override.get("x_center", x_center)
            _plot_panel(axs[i, j], data, zoom_side=panel_side,
                        zoom_frac=panel_zoom_frac, zoom_bbox=panel_bbox,
                        x_center=panel_xcenter, ymax=row_ymax)

        # Explicitly link the row's y-axes so limits/ticks stay identical
        # even if anything downstream (e.g. tight_layout) nudges one of them.
        axs[i, 1].sharey(axs[i, 0])

    for ax in axs[:, 0]:
        ax.set_ylabel("summed near-zero LDOS")
    for ax in axs[-1, :]:
        ax.set_xlabel("site index")

    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig, axs, all_data


def main(idx):
    # ── parameters ────────────────────────────────────────────────────────
    cantor_n = 4
    block_scale = 5
    k_center = 2
    hole_treatment = "substituted"

    if idx == 1:
        mass_pairs = [
            (0.5, 3.5),
            (0.925, -0.05),
            (2.0, -0.5),
            (-0.5, 2.0),
            (-0.75, 0.1)
        ]
    elif idx == 2:
        mass_pairs = [
            (-0.75, 0.1),
            (-0.75, 0.05),
            (-0.75, 0.04),
            (-0.75, 0.03),
            (-0.75, 0.025),
        ]
    else:
        raise ValueError()

    fig, axs, all_data = plot_cantor_ldos_grid(
        cantor_n, block_scale, mass_pairs,
        hole_treatment=hole_treatment, k_center=k_center, zoom_frac=1/18,
        figsize=(12, 4.5 / 2 * len(mass_pairs)),
        inset_overrides={
            (0, 1): {"zoom_frac": 1/9 + 1/27, "side": np.nan, "x_center": 405 * (2/3 + 1/18)},
            (1, 1): {"zoom_frac": 1/9, "side": np.nan, "x_center": 405 * (2/3)},
            #(2, 0): {"zoom_frac": 1/6},
            #(2, 1): {"zoom_frac": 1/6},
            (4, 0): {"zoom_frac": 1/9, "side": np.nan, "x_center": 405 * (2/3)},
            (4, 1): {"zoom_frac": 1/9, "side": np.nan, "x_center": 405 * (2/3)},
        }
    )
    fig.suptitle(f"Cantor set n={cantor_n}, block_scale={block_scale}, method={hole_treatment}",
                 y=1.02)

    for (i, j), data in all_data.items():
        bc = "PBC" if data["pbc_requested"] else "OBC"
        print(f"row{i} {bc}: bc_ok={data['bc_ok']}  ph_symmetric={data['ph_symmetric']}  "
              f"n_selected={len(data['highlighted_idxs'])}  "
              f"energies={np.round(data['energies'], 6)}")

    plt.savefig(f'./figures/ldos/cantor_ldos{idx}.svg', dpi=300)


if __name__ == "__main__":
    main(1)
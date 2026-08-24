"""Plotting orchestration — run this file in your IDE.

Holds all plotting for the project: the reusable per-result plot functions and,
at the bottom, a ``main()`` with flags for assembling specific figures.  Compute
is done separately in ``compute_data.py``; this file only reads saved results via
``project_tools.io`` and draws them.

  plot_phase_diagrams(fractal)  # grid of cantor polarization / carpet Bott index
  plot_dQ_vs_R(result)         # sponge/cube Witten delta_Q(R)
  plot_ldos(result)            # near-zero LDOS: 1D line, 2D image, 3D z-slices
  plot_spectrum(result)        # background/monopole eigenvalue scatter
  plot_result(path)            # dispatch by kind (data/<kind>/...)

The per-result plot functions accept either a path to a saved ``.npz`` result or
an ``(arrays, meta)`` pair, and return ``(fig, ax)``; ``plot_phase_diagrams``
instead takes a fractal name and assembles a grid from all its saved results.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from project_tools import io


def _resolve(result):
    """Return ``(arrays, meta)`` from a path or an already-loaded pair."""
    if isinstance(result, (str, Path)):
        return io.load_result(result)
    arrays, meta = result
    return arrays, meta


def _system_L(meta):
    """Linear size L of a run, read directly from metadata."""
    L = meta.get("L")
    return int(L) if L is not None else None


# ── phase diagrams ───────────────────────────────────────────────────────────

def _plot_phase_diagram_panel(result, ax=None):
    """Imshow one (M, M_alt) phase diagram; polarization (cantor) or Bott (carpet)."""
    arrays, meta = _resolve(result)
    grid = np.asarray(arrays["grid"])
    M, A = arrays["M_vals"], arrays["M_alt_vals"]
    extent = (float(M[0]), float(M[-1]), float(A[0]), float(A[-1]))
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()

    if meta.get("fractal") == "carpet" or grid.dtype.kind in "iu":
        g = np.ma.masked_equal(grid.astype(int), 100)
        lo, hi = -3, 3
        g = np.ma.clip(g, lo, hi)
        vals = np.arange(lo, hi + 1)
        cmap = mcolors.ListedColormap([plt.colormaps["coolwarm"](x)
                                       for x in np.linspace(0, 1, len(vals))])
        cmap.set_bad("black")
        norm = mcolors.BoundaryNorm([v - 0.5 for v in vals] + [vals[-1] + 0.5], len(vals))
        im = ax.imshow(g, origin="lower", extent=extent, aspect="auto",
                       interpolation="nearest", cmap=cmap, norm=norm)
        cb = fig.colorbar(im, ax=ax, label="Bott index", ticks=vals)
        cb.set_ticklabels([str(v) for v in vals])
    else:
        g = np.ma.masked_invalid(grid.astype(float))
        cmap = plt.colormaps["Blues"].copy()
        cmap.set_bad("black")
        im = ax.imshow(g, origin="lower", extent=extent, aspect="auto",
                       interpolation="nearest", cmap=cmap)
        cb = fig.colorbar(im, ax=ax, label="polarization")

    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$M_{\rm alt}$")
    ax.set_title(f"{meta.get('fractal', '')} n={meta.get('n', '?')}")
    return fig, ax


def _phase_diagram_entries(fractal):
    """Return ``[(n, L, path), ...]`` for every phase-diagram file of ``fractal``."""
    entries = []
    folder = io.DATA_ROOT / "phase_diagram"
    for path in sorted(folder.rglob("*.npz")) if folder.is_dir() else []:
        try:
            _, meta = io.load_result(path)
        except Exception:                                       # noqa: BLE001
            continue
        if meta.get("fractal") != fractal:
            continue
        L = _system_L(meta)
        entries.append((int(meta["n"]), int(L), path))
    return entries


def plot_phase_diagrams(fractal, pairs=None):
    """Grid of (M, M_alt) phase diagrams for one fractal: columns = n, rows = L.

    Loads every phase-diagram result for ``fractal`` and lays them out with one
    column per unique ``n`` and one row per unique system length ``L``.  Missing
    ``(n, L)`` combinations are left blank.

    ``pairs`` optionally restricts the plot to specific ``(n, L)`` tuples; by
    default all available pairs for the fractal are shown.
    """
    entries = _phase_diagram_entries(fractal)
    if pairs is not None:
        wanted = {(int(n), int(L)) for n, L in pairs}
        entries = [e for e in entries if (e[0], e[1]) in wanted]
    if not entries:
        raise FileNotFoundError(f"no phase-diagram data found for fractal {fractal!r}")

    ns = sorted({n for n, _, _ in entries})
    Ls = sorted({L for _, L, _ in entries})
    col_of = {n: j for j, n in enumerate(ns)}
    row_of = {L: i for i, L in enumerate(Ls)}

    fig, axs = plt.subplots(
        len(Ls), len(ns),
        figsize=(3.5 * len(ns), 3.0 * len(Ls)),
        squeeze=False,
    )
    for ax in axs.ravel():          # start every cell blank
        ax.axis("off")

    for n, L, path in entries:
        ax = axs[row_of[L]][col_of[n]]
        ax.axis("on")
        _plot_phase_diagram_panel(path, ax=ax)
        ax.set_title(f"{fractal} n={n}, L={L}")

    fig.suptitle(f"{fractal} phase diagrams")
    fig.tight_layout()
    return fig, axs


# ── Witten delta_Q(R) ────────────────────────────────────────────────────────

def plot_dQ_vs_R(result, ax=None, normalize_L=True, label=None):
    """Plot cumulative induced charge delta_Q vs radius (optionally R/L)."""
    arrays, meta = _resolve(result)
    R, dQ = np.asarray(arrays["R"]), np.asarray(arrays["dQ"])
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()

    L = _system_L(meta) if normalize_L else None
    x = R / L if L else R
    if label is None:
        label = f"L={L}" if L else meta.get("method", "")
    ax.plot(x, dQ, lw=1.5, label=label)
    ax.axhline(-0.5, c="k", ls="--", lw=1, zorder=0)
    ax.set_xlabel(r"$R/L$" if L else r"$R$")
    ax.set_ylabel(r"$\delta Q(R)$")
    ax.set_title(f"{meta.get('fractal', '')} {meta.get('method', '')}")
    ax.legend()
    return fig, ax


# ── near-zero LDOS ───────────────────────────────────────────────────────────

def plot_ldos(result, cmap="viridis"):
    """Plot summed near-zero LDOS: 1D line, 2D image, or 3D z-slice grid."""
    arrays, meta = _resolve(result)
    ldos = np.asarray(arrays["ldos"])
    title = f"{meta.get('fractal', '')} near-zero LDOS"

    if ldos.ndim == 1:
        fig, ax = plt.subplots()
        ax.plot(np.arange(ldos.size), ldos, lw=1)
        ax.set_xlabel("site")
        ax.set_ylabel("LDOS")
        ax.set_title(title)
        return fig, ax

    if ldos.ndim == 2:
        fig, ax = plt.subplots()
        im = ax.imshow(ldos.T, origin="lower", cmap=cmap, aspect="equal")
        fig.colorbar(im, ax=ax, label="LDOS")
        ax.set_title(title)
        return fig, ax

    # 3D: grid of z-slices
    Lz = ldos.shape[2]
    cols = int(np.ceil(np.sqrt(Lz)))
    rows = int(np.ceil(Lz / cols))
    vmax = float(np.nanmax(ldos)) if ldos.size else 1.0
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axs = np.atleast_1d(axs).ravel()
    for i, ax in enumerate(axs):
        if i < Lz:
            ax.imshow(ldos[:, :, i].T, origin="lower", vmin=0, vmax=vmax, cmap=cmap)
            ax.set_title(f"z={i}", fontsize=8)
            ax.tick_params(labelsize=6)
        else:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axs


# ── spectra ──────────────────────────────────────────────────────────────────

def plot_spectrum(result):
    """Scatter background and monopole eigenvalue spectra side by side."""
    arrays, meta = _resolve(result)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axs, ("eigenvalues_bg", "eigenvalues_mono"),
                              ("background (g=0)", "monopole (g=1)")):
        w = np.asarray(arrays[key])
        ax.scatter(np.arange(w.size), w, s=2)
        ax.axhline(0, c="k", lw=0.5, ls="--")
        ax.set_xlabel("state index")
        ax.set_ylabel("energy")
        ax.set_title(title)
    fig.tight_layout()
    return fig, axs


# ── dispatch by kind ─────────────────────────────────────────────────────────

def plot_result(path):
    """Plot a saved result, choosing the plot from its ``data/<kind>/`` folder."""
    # kind is the top-level folder under data/ (the path is now nested below it).
    parts = Path(path).parts
    kind = next((k for k in io.KINDS if k in parts), None)
    if kind == "phase_diagram":
        _, meta = io.load_result(path)
        return plot_phase_diagrams(meta.get("fractal"),
                                   pairs=[(int(meta["n"]), int(_system_L(meta)))])
    if kind == "witten":
        return plot_dQ_vs_R(path)
    if kind == "ldos":
        return plot_ldos(path)
    raise ValueError(f"cannot infer plot for kind {kind!r}")


# ── figure orchestration ─────────────────────────────────────────────────────

def main():
    # ── plot flags ───────────────────────────────────────────────────────────
    PLOT_PHASE_DIAGRAMS = 1
    PLOT_WITTEN         = 1
    PLOT_LDOS           = 1

    if PLOT_PHASE_DIAGRAMS:
        # All available phase diagrams for one fractal: columns = n, rows = L.
        # Pass ``pairs=[(n, L), ...]`` to restrict to specific combinations.
        plot_phase_diagrams("cantor")
        plt.show()

    if PLOT_WITTEN:
        # δQ(R) with multiple L curves on one axis
        fig, ax = plt.subplots(figsize=(7, 5))
        for L in (12, 18, 24):
            plot_dQ_vs_R(
                io.result_path("witten", dict(
                    fractal="sponge", method="substituted",
                    n=1, L=L, pasted=True,
                    M=2.0, M_alt=-0.10, M_prime=0.01, t=1.0, B=1.0, gauge="N",
                )),
                ax=ax,
            )
        ax.set_title("Sponge substituted, M_alt = -0.10")
        plt.show()

    if PLOT_LDOS:
        plot_ldos(
            io.result_path("ldos", dict(
                fractal="sponge", method="substituted",
                n=1, L=12, M=2.0, M_alt=-0.10,
            ))
        )
        plt.show()


if __name__ == "__main__":
    main()


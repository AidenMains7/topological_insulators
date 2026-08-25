import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from brokenaxes import brokenaxes

from project_tools import lattice, model
from compute_ltm import compute_wrapper, compute_ldos


def get_ltm_data(n, b, M, method, M_alt=None):
    m = model.build_model("cantor", n, hole_treatment=method, block_scale=b)
    C, _, _ = compute_wrapper(m, n, b, M, method, M_alt, True)
    c_diag = np.diag(C)
    c_diag = c_diag[::2] + c_diag[1::2]

    l = lattice.build_lattice("cantor", n, block_scale=b)
    L = l.size
    t = np.arange(L)
    y = np.full(L, np.nan)
    if method in ["site_elim", "renorm"]:
        site_mask = l.astype(bool)
        y[site_mask] = c_diag

    return t, y


def plot_local_topological_marker(n, b, M, method, ax=None, M_alt=None):
    l = lattice.build_lattice("cantor", n, block_scale=b)
    L = l.size

    t, y = get_ltm_data(n, b, M, method, M_alt)

    if ax is None:
        fig, ax = plt.subplots(1, 1)


    xlims = [(i * L / 9, (i + 1) * L / 9) for i in range(9)]
    for i in [7, 5, 4, 3, 1]:
        xlims.pop(i)
    bax = brokenaxes(
        xlims=tuple(xlims)
    )

    extent = (0., L, max(-3.0, np.min(y)), min(3.0, np.max(y)))
    bax.imshow(l[np.newaxis], aspect='auto', cmap='Greys', alpha=0.25, zorder=-1, extent=extent)
    bax.plot(t, y)
    bax.set_ylim(extent[2], extent[3])
    #bax.figure.suptitle(f"{method}\nn={n}, L={l.size}")
    bax.axhline(-1.0, c='k', ls='--', zorder=-10, alpha=0.5)
    ax = bax
    return ax


def get_ldos_scatter_data(n, b, M, method, M_alt=None):
    l = lattice.build_lattice("cantor", n, block_scale=b)
    ldos = compute_ldos(n, b, M, method, M_alt)
    
    L = l.size
    t = np.arange(L)
    y = np.full(L, np.nan)
    if method in ["site_elim", "renorm"]:
        site_mask = l.astype(bool)
        y[site_mask] = ldos

    return t, y


def plot_ldos_imshow(n, b, M_values, method, M_alt=None, *args, **kwargs):
    l = lattice.build_lattice("cantor", n, block_scale=b)
    L = l.size
    t = np.arange(L)
    y = np.full(L, np.nan)
    grid = np.full((len(M_values), l.size), np.nan)
    for i, M in enumerate(M_values):
        ldos = compute_ldos(n, b, M, method, M_alt)
        if method in ["site_elim", "renorm"]:
            site_mask = l.astype(bool)
            grid[i, site_mask] = ldos

    plt.imshow(grid, aspect='auto')
    cbar = plt.colorbar()
    plt.show()


def make_broken_axes(x, y, cut_points, axs, forgiveness=0.01, **scatter_kwargs):
    assert len(cut_points) % 2 == 0

    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.tick_params(width=2.0)

    edges = [-0.01] + [p + j * forgiveness for p, j in zip(cut_points, [1, -1] * (len(cut_points) // 2))] + [1.01]
    x_range = (np.min(x), np.max(x))
    edges = [e * (x_range[1] - x_range[0]) + x_range[0] for e in edges]
    xlims = [tuple(edges[i:i+2]) for i in range(0, len(edges), 2)]
    
    d = 0.01

    for i, ax in enumerate(axs):
        kwargs = dict(transform=ax.transAxes, color='black', clip_on=False, lw=2.0)
        if i != 0:
            ax.spines['left'].set_visible(False)
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)

            ax.set_yticks([])
            ax.set_yticklabels([])
        if i != len(axs) - 1:
            ax.spines['right'].set_visible(False)
            ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax.set_xlim(xlims[i])
        ax.scatter(x, y, **scatter_kwargs)

    return axs




def plot_on_cantor(n, b, method):
    cut_points = [1/9, 2/9, 1/3, 2/3, 7/9, 8/9]
    fig, axs = plt.subplots(1, len(cut_points) // 2 + 1, figsize=(10, 5))

    ms = [-1, 1, 3, 5]
    cs = ['k', 'r', 'b', 'g']
    shapes = ['.', 's', '^', 'v']
    offsets = [-0.1, 0, -0.1, 0]
    sizes = [36, 50, 36, 50]
    zorders = [1, 0, 1, 0]
    ys = []
    for i in range(len(ms)):
        x, y = get_ltm_data(n, b, ms[i], method)
        ys.append(y)
        axs = make_broken_axes(x, y, cut_points, axs, c=cs[i], marker=shapes[i], label=f"$M={ms[i]}$", s=sizes[i], zorder=zorders[i])

    for ax in axs: ax.axhline(-1.0, c='k', zorder=-10, ls='--')

    ymax, ymin = np.nanmax(ys[1]), np.nanmin(ys[1])
    yticks = [ymax, 0., -1.0]
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels([round(t, 2) for t in yticks])
    axs[0].legend(loc='upper left')
    axs[0].set_ylabel("C(r)", fontsize=16)
    fig.text(0.5, 0.02, "Site Index", ha='center', fontsize=16)

    l = lattice.build_lattice("cantor", n, block_scale=b)
    L = l.size
    fig.suptitle(f"{method} n={n} L={L}")
    plt.savefig(f"./figures/{method}_n={n}_L={L}.png")

if __name__ == "__main__":

    ns = [2, 3, 4]
    bs = [1, 3, 9, 27]
    for n in ns:
        for b in bs:
            for m in ["renorm"]:
                plot_on_cantor(n, b, m)
                plt.close()

    #n = 4; b = 9
    #C, eigenvalues, eigenvectors = compute_wrapper(model.build_model("cantor", n, hole_treatment="renorm", block_scale=b), n, b, 1.0, "site_elim", None, True)
    #plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
    #plt.show()
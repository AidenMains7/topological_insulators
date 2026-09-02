import numpy as np
from scipy import linalg as spla
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
from brokenaxes import brokenaxes

from project_tools import lattice
from compute_ltm_cantor import compute_wrapper, compute_ldos


def get_ltm_data(n, b, M, method, M_alt=None, overwrite=False):
    C, eigenvalues, V = compute_wrapper(n, b, M, method, M_alt, overwrite)

    c_diag = np.diag(C)
    c_diag = c_diag[::2] + c_diag[1::2]

    l = lattice.build_lattice("cantor", n, block_scale=b)
    L = l.size
    t = np.arange(L)
    y = np.full(L, np.nan)
    if method in ["site_elim", "site_elim_alt", "renorm", "renorm_alt"]:
        site_mask = l.astype(bool)
        y[site_mask] = c_diag
    else:
        y = c_diag
    return t, y


def get_ldos_data(n, b, M, method, M_alt=None, overwrite=False):
    _, _, ldos = compute_wrapper(n, b, M, method, M_alt, overwrite)
    l = lattice.build_lattice("cantor", n, block_scale=b)
    
    L = l.size
    t = np.arange(L)
    y = np.full(L, np.nan)
    if method in ["site_elim", "site_elim_alt", "renorm", "renorm_alt"]:
        site_mask = l.astype(bool)
        y[site_mask] = ldos
    else:
        y = ldos
    return t, y


def get_ldos_scatter_data(n, b, M, method, M_alt=None):
    m = model.build_model("cantor", n, hole_treatment=method, block_scale=b)
    _, _, ldos = compute_wrapper(m, n, b, M, method, M_alt, True)
    l = lattice.build_lattice("cantor", n, block_scale=b)
    
    L = l.size
    t = np.arange(L)
    y = np.full(L, np.nan)
    if method in ["site_elim", "renorm"]:
        site_mask = l.astype(bool)
        y[site_mask] = ldos

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


def compute_broken_axes_limits(arr, keep_points=[], threshold=0.01, jump_threshold=0.05, extrema_pad:float=0.01):
    arr = arr[~np.isnan(arr)]
    arr_unique = np.unique(np.round(arr, 6))
    arr_range = np.nanmax(arr) - np.nanmin(arr)

    large_jumps_idxs = np.argwhere(np.diff(arr_unique) / arr_range >= jump_threshold)
    edges = np.sort(np.concatenate((arr_unique[large_jumps_idxs], arr_unique[large_jumps_idxs + 1])).flatten())
    edges += np.tile([+threshold * arr_range, -threshold * arr_range], edges.size // 2)
    edges = [arr_unique[0] - arr_range * extrema_pad, arr_unique[-1] + arr_range * extrema_pad] + edges.tolist()
    edges = np.sort(edges)

    add_points = []
    remove_points = []
    for kp in keep_points:
        skip = False
        for i in range(0, edges.size, 2):
            if edges[i] < kp and kp < edges[i+1]:
                skip = True
        if not skip:
            conflicting_edges = []
            xi = kp - threshold * arr_range
            xj = kp + threshold * arr_range
            for e in edges:
                if e > xi and e < xj:
                    conflicting_edges.append(e)

            for p in conflicting_edges: remove_points.append(p)
            if conflicting_edges:
                add_points.append(xj)
            else:
                add_points.append(xi)
                add_points.append(xj)

    edges = list(edges) + add_points
    for p in remove_points:
        edges.remove(p)
    edges = np.sort(edges)
    lims = [(edges[i], edges[i+1]) for i in range(0, len(edges) - 1, 2)]
    return lims


def compute_cantor_lims(n:int, b:int, C_r_arrays:list[np.ndarray], x_threshold = 0.01, y_threshold = 0.01):
    l = lattice.build_lattice("cantor", n, block_scale=b)
    x = np.arange(l.size)[np.argwhere(l)].flatten().astype(float)
    xlims = compute_broken_axes_limits(x, threshold=x_threshold, jump_threshold=3 ** (-n), extrema_pad=0.001)

    all_c_values = np.concatenate(C_r_arrays).flatten()
    ylims = compute_broken_axes_limits(all_c_values, keep_points=[0.], threshold=y_threshold, jump_threshold = 0.2, extrema_pad=0.01)
    return xlims, ylims


def plot_on_cantor_set(method, n, b, break_xax=True, break_yax=True, data_func=get_ltm_data, overwrite=True):
    l = lattice.build_lattice("cantor", n, block_scale=b)
    ms = [-1, 1, 3, 5]
    data = []
    for m in ms:
        _, d = data_func(n, b, m, method, overwrite)
        data.append(np.abs(d))

    cmin, cmax = np.nanmin(data), np.nanmax(data)
    crange = cmax - cmin
    xlims = [(0, l.size)]
    ylims = [(cmin - 0.1 * crange, cmax + 0.1 * crange)]
    if break_xax or break_yax:
        broken_xlims, broken_ylims = compute_cantor_lims(n, b, data, x_threshold=0.001, y_threshold=0.001)
    if break_xax and method not in ["substituted", "substituted_alt"]:
        xlims = broken_xlims
    if break_yax:
        ylims = broken_ylims
    
    fig = plt.figure(figsize=(20, 10))
    bax = brokenaxes(xlims=xlims, ylims=ylims, d=0.005, despine=True, fig=fig)

    cs = ['k', 'r', 'b', 'g']
    shapes = ['.', 's', '^', 'v']
    offsets = [-0.1, 0, -0.1, 0]
    sizes = [36, 50, 36, 50]
    zorders = [1, 0, 1, 0]

    for i in range(len(ms)):
        bax.scatter(np.arange(l.size), data[i], 
                    c=cs[i], marker=shapes[i], s=sizes[i], zorder=zorders[i],
                    label=f'$M={ms[i]}$')
    axs = np.array(bax.axs).reshape(len(ylims), len(xlims))

    if break_xax:
        for ax in axs[-1, :]:
            xmin, xmax = ax.get_xlim()
            r = (xmax - xmin) / 5
            ax.set_xticks(np.round([xmin + r, xmax - r], 0))
            #ax.set_xticks([np.round(np.mean(ax.get_xlim()), 0)])

    axs[0, -1].legend()

    fig = bax.fig
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    func_dir = 'ltm' if 'ltm' in data_func.__name__ else 'ldos'
    tit = 'Local Topological Marker' if func_dir == 'ltm' else 'Local Density of States'
    fig.suptitle(f"{tit}\n{method} : n={n} : L={l.size}")
    plt.savefig(f'./figures/{func_dir}/{method}_n={n}_L={l.size}.svg')
    plt.savefig(f'./figures/{func_dir}/{method}_n={n}_L={l.size}.png')



if __name__ == "__main__":
    rcParams['axes.linewidth'] = 2.0
    rcParams['xtick.major.width'] = 2.0
    rcParams['ytick.major.width'] = 2.0
    ns = [4]
    bs = [27]
    for n in ns:
        for b in bs:
            for m in ["substituted"]:
                plot_on_cantor_set(m, n, b, True, True, get_ldos_data, overwrite=False)
                plt.close() 
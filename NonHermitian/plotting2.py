import numpy as np
import scipy.linalg as spla
from typing import Any
from decimal import Decimal
import h5py

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
from matplotlib import colormaps, patches
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from nonhermitian_defects import DefectLattice, compute_eigenvectors_eigenvalues


def save_ipr_data(Ls:list[int], eig_dicts:list[dict], fname:str, directory:str = "./NonHermitian/Data/"):
    with h5py.File(directory + fname, "w") as f:
        f.create_dataset(name="Ls", data=Ls)
        for L, ed in zip(Ls, eig_dicts):
            for k, v in zip(ed.keys(), ed.values()):
                f.create_dataset(name = f"L={L}_"+k, data=v)


def read_ipr_data(fname:str, directory:str = "./NonHermitian/Data/"):
    data = {}
    eds = []
    supergroups = []
    with h5py.File(directory + fname, "r") as f:
        Ls = f["Ls"][()]
        for k, v in zip(f.keys(), f.values()):
            grouping = k.split("_")[0]
            if grouping == "Ls":
                continue
            supergroups.append(grouping)
            if grouping not in data:
                data[grouping] = {}
            data[grouping][k[len(grouping) + 1:]] = v[()]

    return Ls, [v for v in data.values()]
    

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + int(exponent) - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()

def format_colorbar(cbar:Colorbar):
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_powerlimits((0,  0))
    cbar.formatter = formatter
    cbar.update_ticks()


def plot_ldos(Lattice:DefectLattice, ax:Axes, color_array:np.ndarray, cbar_ax:None|Axes=None, cmap="Greys",
              extent:tuple=(), scatter_size = 100):

    X, Y = Lattice.X, Lattice.Y
    X_d, Y_d = np.array(Lattice.defect_positions).T

    plot = ax.scatter(X, Y, c=color_array, cmap=cmap, s=scatter_size, vmin=0., vmax=1., zorder=1)
    plot_defects = ax.scatter(X_d, Y_d, zorder=2, lw=1, s=scatter_size, edgecolor='r', facecolor='w')

    if cbar_ax != None:
        cax_box = cbar_ax.get_position()
        hw_ratio = cax_box.height / cax_box.width
        cbar = plt.colorbar(plot, cbar_ax, orientation="vertical" if hw_ratio >= 1. else "horizontal")
        if cbar.orientation == "horizontal":
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")

        cbar.set_label("Local Density of States")
        format_colorbar(cbar)

        vmin, vmax = plot.get_clim()
        ticks = (vmin, vmax)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tuple([str(round(t, 1)) for t in ticks]))

    if extent == ():
        xticks = [0, Lattice.Lx - 1]
        yticks = [0, Lattice.Ly - 1]
    else:
        xticks = [extent[0], extent[1]]
        yticks = [extent[2], extent[3]]

    xtick_labels = [f"${int(t+1)}$" for t in xticks]
    ytick_labels = [f"${int(t+1)}$" for t in yticks]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")


def plot_spectrum(ax:Axes, eigenvalues:np.ndarray, color_array:np.ndarray|None = None, cbar_ax:Axes|None = None, 
                  selected_idxs:np.ndarray|None = None, cmap:str='jet', scatter_size:int = 50):

    if not isinstance(color_array, np.ndarray):
        color_array = np.zeros(eigenvalues.shape)

    if np.issubdtype(eigenvalues.dtype, np.complexfloating):
        x, y = eigenvalues.real, eigenvalues.imag # type: ignore
        ax.set_xlabel("$\\Re (E)$")
        ax.set_ylabel("$\\Im (E)$")
    else:
        x, y = np.arange(eigenvalues.size), eigenvalues
        ax.set_xlabel("$n$")
        ax.set_ylabel("$E_n$")

    sort_idxs = np.argsort(color_array)
    plot = ax.scatter(x[sort_idxs], y[sort_idxs], c=color_array[sort_idxs], cmap=cmap, zorder=1, s=scatter_size)


    xticks = np.linspace(np.min(x), np.max(x), 3)
    yticks = np.linspace(np.min(y), np.max(y), 3)
    xticks = tuple([round(t, 1) for t in xticks])
    yticks = tuple([round(t, 1) for t in yticks])
    
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(tuple([str(t) for t in xticks]))
    ax.set_yticklabels(tuple([str(t) for t in yticks]))

    vmin, vmax = plot.get_clim()
    if not isinstance(selected_idxs, np.ndarray): 
        ax.scatter(x[selected_idxs], y[selected_idxs], c=color_array[selected_idxs], zorder=2, marker="*", vmin=vmin, vmax=vmax, s=2 * scatter_size)

    if not isinstance(cbar_ax, np.ndarray):
        cbar = plt.colorbar(plot, cbar_ax)
        format_colorbar(cbar)

        ticks = (vmin, (vmin + vmax) / 2, vmax)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tuple([str(round(t, 1)) for t in ticks]))


def plot_ipr(ax:Axes, eigenvalues:tuple, iprs:tuple, Ls:"list|tuple", selected_idxs:"np.ndarray|None" = None):

    L_sort_idxs = np.argsort(Ls)
    Ls = [Ls[i] for i in L_sort_idxs]
    eigenvalues = tuple([eigenvalues[i] for i in L_sort_idxs])
    iprs = tuple([iprs[i] for i in L_sort_idxs])

    if np.issubdtype(eigenvalues[0].dtype, np.complexfloating):
        xs = [np.abs(ev) * np.sign(ev.real) for ev in eigenvalues] # type: ignore
        ax.set_xlabel("$|E| \\times {\\rm sign}(\\Re E)$")
    else:
        xs = [ev for ev in eigenvalues]
        ax.set_xlabel("$E$")
    ax.set_ylabel("IPR")

    all_colors = ['tab:purple', 'tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:olive', 'tab:cyan', 'r']
    assert len(iprs) <= len(all_colors)
    colors = [all_colors[i] for i in range(len(iprs))]
    colors[-1] = 'r'

    mask = np.full(xs[-1].size, False)
    if isinstance(selected_idxs, np.ndarray): 
        mask[selected_idxs] = True
        ax.scatter(xs[-1][mask], iprs[-1][mask], c=colors[-1], marker="*", zorder=100, s = 100)
    ax.scatter(xs[-1][~mask], iprs[-1][~mask], s=100, c=colors[-1], alpha=0.5, label=f"$L={Ls[-1]}$", zorder=99)


    for i in range(len(iprs) - 1):
        ax.scatter(xs[i], iprs[i], s=100, c=colors[i], alpha=0.5, label=f"$L={Ls[i]}$", zorder=i)

    ax.legend()

    xmax = max(np.max(x) for x in xs)
    xmin = min(np.min(x) for x in xs)
    ymax = max(np.max(y) for y in iprs)
    ymin = min(np.min(y) for y in iprs)
    xrange = xmax - xmin
    yrange = ymax - ymin

    thresh = 0.0
    xticks = np.linspace(xmin + thresh * xrange, xmax - thresh  * xrange, 3)
    yticks = np.linspace(ymin + thresh * yrange, ymax - thresh * yrange, 3)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(tuple([f"{t:.1f}" for t in xticks]))
    ax.set_yticklabels(tuple([f"${fman(t):.1f}\\times 10^{{{fexp(t):.0f}}}$" for t in yticks]))
    

def figure_layout():
    fig = plt.figure()

    # Master Gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1., 1.5, 1.5], hspace=0.3)

    # Gridspec for IPR
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 0])
    ax1 = fig.add_subplot(gs0[0, 0], label="a")
    ax2 = fig.add_subplot(gs0[0, 1], label="b", sharey=ax1)

    # Gridspec for first spectrum
    gs1_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], width_ratios=[1, .1])
    ax3 = fig.add_subplot(gs1_a[0, 0], label="c")
    ax3_cb = fig.add_subplot(gs1_a[0, 1], label="c_cb")

    # Gridspec for second spectrum
    gs1_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], width_ratios=[1, .1])
    ax4 = fig.add_subplot(gs1_b[0, 0], label="d")
    ax4_cb = fig.add_subplot(gs1_b[0, 1], label="d_cb")

    if True:
        # Gridspec for LDOS
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, 1:], width_ratios=[1, 1, 1], height_ratios=[.1, 1.])
        ax5 = fig.add_subplot(gs2[1, 0], label="e")
        ax5_cb = fig.add_subplot(gs2[0, 0], label="e_cb")

        ax6 = fig.add_subplot(gs2[1, 1], label="f")
        ax7 = fig.add_subplot(gs2[1, 2], label="g")
        ax67_cb = fig.add_subplot(gs2[0, 1:], label="67_cb")
    else:
        # Gridspec for LDOS of topological modes
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[.1, 1])
        ax5 = fig.add_subplot(gs2[1, 0], label="e")
        ax5_cb = fig.add_subplot(gs2[0, 0], label="e_cb")

        # Gridspec for LDOS of skin effect
        gs3 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 2], width_ratios=[1, 1], height_ratios=[.1, 1.])
        ax6 = fig.add_subplot(gs3[1, 0], label="f")
        ax7 = fig.add_subplot(gs3[1, 1], label="g")
        ax_6_7_cb = fig.add_subplot(gs3[0, :], label="67_cb")

    ax3_cb_box = ax3_cb.get_position()
    ax3_box = ax3.get_position()
    ax3_cb.set_position((ax3_cb_box.x0, ax3_cb_box.y0, ax3_cb_box.width, ax3_box.height))
    ax4_cb_box = ax4_cb.get_position()
    ax4_box = ax4.get_position()
    ax4_cb.set_position((ax4_cb_box.x0, ax4_cb_box.y0, ax4_cb_box.width, ax4_box.height))


    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(width=1.5)
        for k in ax.spines.keys():
            ax.spines[k].set_linewidth(1.5)

    return fig


def main(method:str, Ls:np.ndarray):
    fig = figure_layout()
    ax_a, ax_b, ax_c, ax_c_cb, ax_d, ax_d_cb, ax_e, ax_e_cb, ax_f, ax_g, ax_fg_cb = fig.axes

    if isinstance(Ls, int):
        Ls = (Ls)
    
    lattices = [DefectLattice(L, L, method, True) for L in Ls]
    ed1s = [compute_eigenvectors_eigenvalues(Lattice, -1., (0.5, 0., 0.), (1.5, 0., 0.)) for Lattice in lattices]
    ed2s = [compute_eigenvectors_eigenvalues(Lattice, -1., (1.5, 0., 0.), (0.5, 0., 0.)) for Lattice in lattices]

    # Plot the IPRs
    for i, (eds, ipr_ax) in enumerate(zip([ed1s, ed2s], [ax_a, ax_b])):
        evs = tuple([ed["eigenvalues"] for ed in eds])
        iprs = tuple([ed["left_ipr"] for ed in eds])
        plot_ipr(ipr_ax, evs, iprs, Ls, eds[-1]["selected_idxs"] if i == 0 else None)

    ipr_max = max([np.max(ed["left_ipr"]) for ed in ed1s] + [np.max(ed["left_ipr"]) for ed in ed2s])
    ipr_min = min([np.min(ed["left_ipr"]) for ed in ed1s] + [np.min(ed["left_ipr"]) for ed in ed2s])
    for ax in [ax_a, ax_b]:
        ipr_ticks = np.linspace(ipr_min, ipr_max, 3)
        ax.set_yticks(ipr_ticks)
    ax_b.set_yticklabels([])
    ax_a.set_yticklabels(tuple([f"${fman(t):.1f}\\times 10^{{{fexp(t):.0f}}}$" for t in ipr_ticks]))

    # Plot the complex spectrums
    for eds, spectrum_ax, cb_ax in zip([ed1s, ed2s], [ax_c, ax_d], [ax_c_cb, ax_d_cb]):
        eigenvalues = eds[-1]["eigenvalues"]
        ipr = eds[-1]["left_ipr"]
        idxs = eds[-1]["selected_idxs"]
        plot_spectrum(spectrum_ax, eigenvalues, ipr, cb_ax, idxs)

    # Plot the LDOS
    topo_ldos = ed1s[-1]["selected_left_eigenvectors"]
    L1 = ed1s[-1]["L"]
    L2 = ed2s[-1]["L"]

    topo_ldos -= np.min(topo_ldos)
    L1 -= np.min(L1)
    L2 -= np.min(L2)

    topo_ldos /= np.max(topo_ldos)

    skin_max = max(np.max(L1), np.max(L2))
    L1 /= skin_max
    L2 /= skin_max

    plot_ldos(lattices[-1], ax_e, topo_ldos, ax_e_cb)
    plot_ldos(lattices[1], ax_f, L1, ax_fg_cb)
    plot_ldos(lattices[-1], ax_g, L2)


    ax_b.set_ylabel("")
    ax_b.legend().remove()
    ax_d.set_ylabel("")
    ax_g.set_ylabel("")

    for ax in [ax_e, ax_f, ax_g]:
        ax.set_aspect("equal")

    plt.show()


def compute_ipr_data(method:str, Ls:list[int], hdir:str, **kwargs):
    lattices = [DefectLattice(L, L, method, True, **kwargs) for L in Ls]
    ed1s = ]


if __name__ == "__main__":
    if 0:
        Ls = [10, 12, 14]
        method = "vacancy"
        lattices = [DefectLattice(L, L, method, True) for L in Ls]
        ed1s = [compute_eigenvectors_eigenvalues(Lattice, -1., (0.5, 0., 0.), (1.5, 0., 0.)) for Lattice in lattices]
        ed2s = [compute_eigenvectors_eigenvalues(Lattice, -1., (1.5, 0., 0.), (0.5, 0., 0.)) for Lattice in lattices]

        save_ipr_data(Ls, ed1s, "vacancy_ipr_data_hx0=0.5.h5")
        save_ipr_data(Ls, ed2s, "vacancy_ipr_data_hx0=1.5.h5")


    Ls, eds = read_ipr_data("vacancy_ipr_data_hx0=0.5.h5")

    
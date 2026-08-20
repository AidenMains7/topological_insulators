import numpy as np
import scipy.linalg as spla
from typing import Any, cast
from decimal import Decimal
import h5py, os

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
from matplotlib import rcParams

from nonhermitian_defects import DefectLattice, compute_eigenvectors_eigenvalues, get_close_to_zero_idxs

# region Saving and Reading Data
def save_ipr_data(Ls:list[int], eig_dicts:list[dict], fname:str, directory:str = "./NonHermitian/Data/"):
    with h5py.File(directory + fname, "w") as f:
        f.create_dataset(name="Ls", data=Ls)
        for L, ed in zip(Ls, eig_dicts):
            for k, v in zip(ed.keys(), ed.values()):
                f.create_dataset(name = f"L={L}_"+k, data=v)


def read_ipr_data(fname:str, directory:str = "./NonHermitian/Data/") -> tuple[list, list[dict]]:
    data = {}
    eds = []
    supergroups = []
    with h5py.File(directory + fname, "r") as f:
        Ls = cast(list, f["Ls"])
        for k, v in zip(f.keys(), f.values()):
            grouping = k.split("_")[0]
            if grouping == "Ls":
                continue
            supergroups.append(grouping)
            if grouping not in data:
                data[grouping] = {}
            data[grouping][k[len(grouping) + 1:]] = cast(np.ndarray, v)[()]

    return Ls, [v for v in data.values()]


def get_selected_modes(eig_dict, n_idxs, Lattice:DefectLattice):
    def _ensure_eigenvector_shape_for_schottky(eigenvector):
        mask = np.full(eigenvector.shape[0] + len(Lattice.defect_indices), True) 
        for i, idx in enumerate(Lattice.defect_indices):    
            mask[2 * idx + i % 2] = False
        resized_eigenvector = np.zeros(mask.shape, dtype=eigenvector.dtype)
        resized_eigenvector[mask] = eigenvector
        return resized_eigenvector
    
    left_eigenvectors = eig_dict["left_eigenvectors"]
    idxs = get_close_to_zero_idxs(left_eigenvectors, n_idxs)
    selected = left_eigenvectors[:, idxs]
    ldos = np.sum(np.abs(selected) ** 2, axis = 1)

    if Lattice.defect_type == "schottky":
        ldos = _ensure_eigenvector_shape_for_schottky(ldos)
    selected_left_eigenvectors = ldos[::2] + ldos[1::2]
    eig_dict["selected_left_eigenvectors"] = selected_left_eigenvectors
    eig_dict["selected_idxs"] = idxs

    return eig_dict
  

def compute_ipr_data(method:str, Ls:list[int], n_idxs:int, hdir:str, directory:str = "./NonHermitian/Data/", **kwargs):
    file1 = directory + f"{method}_ipr_data_h{hdir}=0.5.h5"
    file2 = directory + f"{method}_ipr_data_h{hdir}=1.5.h5"

    if method in ["substitution", "interstitial"]:
        n_defects = 1
        if "defect_radius" in kwargs.keys():
            n_defects += np.sum(4 * np.arange(kwargs["defect_radius"]))
        if "break_c4" in kwargs.keys():
            n_defects += 2
        file1 = directory + f"{method}_ipr_data_h{hdir}=0.5_nd={n_defects}.h5"
        file2 = directory + f"{method}_ipr_data_h{hdir}=1.5_nd={n_defects}.h5"

    schottky_separations = [L // 4 + (L // 4 + 1) % 2 for L in Ls]
    fpxs = fpys = [-s -0.5 for s in schottky_separations]

    lattices = [DefectLattice(L, L, method, True, schottky_separation=s, frenkel_x_disp=fpx, frenkel_y_disp=fpy, **kwargs) 
                for L, s, fpx, fpy in zip(Ls, schottky_separations, fpxs, fpys)]
    
    if all([os.path.exists(file) for file in [file1, file2]]):
        print(f"IPR data files already exist for method {method} and hdir {hdir}.")
        new_Ls1, data1 = read_ipr_data(file1, "")
        new_Ls2, data2 = read_ipr_data(file2, "")
        assert new_Ls1 == new_Ls2, "Mismatch in Ls between the two IPR data files."
        data1 = [get_selected_modes(ed, n_idxs, l) for ed, l in zip(data1, lattices)]
        data2 = [get_selected_modes(ed, n_idxs, l) for ed, l in zip(data2, lattices)]
        return Ls, data1, data2, lattices[-1]
    hdir_map = {"x":0, "y":1, "z":2}
    v1, v2 = [0.0] * 3, [0.0] * 3
    v1[hdir_map[hdir]] = 0.5
    v2[hdir_map[hdir]] = 1.5

    if method == "schottky": print(schottky_separations)
    if method == "frenkel_pair": print(fpxs)


    v1, v2 = np.array(v1), np.array(v2)
    print(f"Starting computation for {method} h{hdir}")
    ed1s = [compute_eigenvectors_eigenvalues(Lattice, -1.0, v1, v2, n_idxs) for Lattice in lattices]
    print("Computed eigenvectors and eigenvalues for (v1, v2).")
    ed2s = [compute_eigenvectors_eigenvalues(Lattice, -1.0, v2, v1, n_idxs) for Lattice in lattices]
    print("Computed eigenvectors and eigenvalues for (v2, v1).")
    save_ipr_data(Ls, ed1s, file1, "")
    save_ipr_data(Ls, ed2s, file2, "")
    return Ls, ed1s, ed2s, lattices[-1]

# endregion

# region Misc. Functions
def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + int(exponent) - 1


def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()


def find_ldos_view_area(Lattice, ldos_array):
    X, Y = Lattice.X, Lattice.Y
    X_d, Y_d = np.array(Lattice.defect_positions)

    x_center = int(np.mean(X_d))
    y_center = int(np.mean(Y_d))
    radius = (max(np.max(X_d - x_center), np.max(Y_d - y_center)) + 10)
    extent = (x_center - radius, x_center + radius, y_center - radius, y_center + radius)

    ldos_mask = ldos_array >  np.mean(ldos_array) + 2 * np.std(ldos_array)
    ldos_mask = [False, False]
    if np.any(ldos_mask):
        X_masked = X[ldos_mask]
        Y_masked = Y[ldos_mask]

        xmin, xmax = int(np.min(X_masked)), int(np.max(X_masked))
        ymin, ymax = int(np.min(Y_masked)), int(np.max(Y_masked))

        d_xmin, d_xmax, d_ymin, d_ymax = extent
        extent = (min(xmin, d_xmin), max(xmax, d_xmax), min(ymin, d_ymin), max(ymax, d_ymax))
        radius = max(extent[1] - extent[0], extent[3] - extent[2]) // 2 + 3
        x_center = (extent[0] + extent[1]) // 2
        y_center = (extent[2] + extent[3]) // 2
        extent = (x_center - radius, x_center + radius, y_center - radius, y_center + radius)

    extent = (max(extent[0], 0), min(extent[1], Lattice.Lx - 1), max(extent[2], 0), min(extent[3], Lattice.Ly - 1))

    return tuple(np.ravel(extent).astype(int))

# endregion

# region Plotting
def format_colorbar(cbar:Colorbar):
    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_powerlimits((0,  0))
    cbar.formatter = formatter
    cbar.update_ticks()
    cbar.outline.set_linewidth(1.5)


def plot_ldos(Lattice:DefectLattice, ax:Axes, color_array:np.ndarray, cbar_ax:None|Axes=None, cmap="Greys",
              extent:tuple=(), scatter_size = 100):

    if extent == ():
        extent = (0, Lattice.Lx - 1, 0, Lattice.Ly - 1)
    xticks = [extent[0] + 1, extent[1] - 1]
    yticks = [extent[2] + 1 , extent[3] - 1]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    X, Y = Lattice.X, Lattice.Y
    X_d, Y_d = np.array(Lattice.defect_positions)

    mask = (X >= extent[0]) & (X <= extent[1]) & (Y >= extent[2]) & (Y <= extent[3])

    plot = ax.scatter(X, Y, c=color_array, cmap=cmap, s=scatter_size, zorder=1, rasterized=True)
    if Lattice.defect_type == "vacancy":
        ax.scatter(X_d, Y_d, zorder=2, lw=1, s=scatter_size, edgecolor='r', facecolor='none', rasterized=True)
    elif Lattice.defect_type == "frenkel_pair":
        pass

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
        #cbar.set_ticklabels(tuple([str(round(t, 1)) for t in ticks]))

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
    x = x[sort_idxs]
    y = y[sort_idxs]
    color_array = color_array[sort_idxs]
    mask = np.full(color_array.shape, False)
    if isinstance(selected_idxs, np.ndarray):
        mask[selected_idxs] = True
    mask = mask[sort_idxs]

    plot = ax.scatter(x[~mask], y[~mask], c=color_array[~mask], cmap=cmap, zorder=1, s=scatter_size, rasterized=True)

    vmin, vmax = plot.get_clim()
    ax.scatter(x[mask], y[mask], c=color_array[mask], zorder=2, marker="*", vmin=vmin, vmax=vmax, s=2 * scatter_size, cmap=cmap, rasterized=True)

    xticks = np.linspace(np.min(x), np.max(x), 3)
    yticks = np.linspace(np.min(y), np.max(y), 3)
    xticks = tuple([round(t, 1) for t in xticks])
    yticks = tuple([round(t, 1) for t in yticks])
    
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(tuple([str(t) for t in xticks]))
    ax.set_yticklabels(tuple([str(t) for t in yticks]))

    if not isinstance(cbar_ax, np.ndarray):
        cbar = plt.colorbar(plot, cbar_ax)
        format_colorbar(cbar)
        ticks = (vmin, (vmin + vmax) / 2, vmax)
        cbar.set_ticks(ticks)
        #cbar.set_ticklabels(tuple([str(round(t, 1)) for t in ticks]))


def plot_ipr(ax:Axes, eigenvalues:tuple, iprs:tuple, Ls:"list|tuple", selected_idxs:"np.ndarray|None" = None, scatter_size:int = 75):

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
        ax.scatter(xs[-1][mask], iprs[-1][mask], c=colors[-1], marker="*", zorder=100, s = scatter_size, rasterized=True)
    ax.scatter(xs[-1][~mask], iprs[-1][~mask], s = 1.5*scatter_size, c=colors[-1], alpha=0.25, label=f"$L={Ls[-1]}$", zorder=99, rasterized=True)


    for i in range(len(iprs) - 1):
        ax.scatter(xs[i], iprs[i], s=scatter_size, c=colors[i], alpha=0.25, label=f"$L={Ls[i]}$", zorder=i, rasterized=True)

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
    fig = plt.figure(figsize=(20, 10))

    # Master Gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2., 1.5, 1.5], hspace=0.3)

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
        ax6_cb = fig.add_subplot(gs2[0, 1], label="f_cb")
        ax7_cb = fig.add_subplot(gs2[0, 2], label="g_cb")
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


    print(fig.axes)
    for ax in fig.axes:
        ax.tick_params(width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(length=5.0)
    return fig

# endregion

def main(method:str, Ls:list, n:int, hdir:str, **kwargs):
    fig = figure_layout()
    ax_a, ax_b, ax_c, ax_c_cb, ax_d, ax_d_cb, ax_e, ax_e_cb, ax_f, ax_g, ax_f_cb, ax_g_cb = fig.axes

    if isinstance(Ls, int):
        Ls = (Ls)

    # Compute / read the data
    Ls, ed1s, ed2s, LargestLattice = compute_ipr_data(method, Ls, n, hdir, **kwargs)

    # Plot the IPRs
    for i, (eds, ipr_ax) in enumerate(zip([ed1s, ed2s], [ax_a, ax_b])):
        evs = tuple([ed["eigenvalues"] for ed in eds])
        iprs = tuple([ed["left_ipr"] for ed in eds])
        plot_ipr(ipr_ax, evs, iprs, Ls, eds[-1]["selected_idxs"] if i == 0 else None)

    # Set the y-ticks for the IPR plots
    ipr_max = max([np.max(ed["left_ipr"]) for ed in ed1s] + [np.max(ed["left_ipr"]) for ed in ed2s])
    ipr_min = min([np.min(ed["left_ipr"]) for ed in ed1s] + [np.min(ed["left_ipr"]) for ed in ed2s])
    for ax in [ax_a, ax_b]:
        ipr_ticks = np.linspace(ipr_min, ipr_max, 3)
        ax.set_yticks(ipr_ticks)
    ax_b.set_yticklabels([])
    ax_a.set_yticklabels(tuple([f"${fman(t):.1f}\\times 10^{{{fexp(t):.0f}}}$" for t in ipr_ticks]))

    # Plot the complex spectrums
    for i, (eds, spectrum_ax, cb_ax) in enumerate(zip([ed1s, ed2s], [ax_c, ax_d], [ax_c_cb, ax_d_cb])):
        eigenvalues = eds[-1]["eigenvalues"]
        ipr = eds[-1]["left_ipr"]
        idxs = eds[-1]["selected_idxs"]
        plot_spectrum(spectrum_ax, eigenvalues, ipr, cb_ax, idxs if i == 0 else None, cmap='jet' if i == 0 else 'jet')

    # Plot the LDOS
    topo_ldos = ed1s[-1]["selected_left_eigenvectors"]
    L1 = ed1s[-1]["L"]
    L2 = ed2s[-1]["L"]

    if False:
        # Normalize the LDOS of the topological modes to the range (0, 1)
        topo_ldos -= np.min(topo_ldos)
        topo_ldos /= np.max(topo_ldos)

        # Normalize the LDOS of the skin effect to the range (0, 1)
        L1 -= np.min(L1)
        L2 -= np.min(L2)
        skin_max = max(np.max(L1), np.max(L2))
        L1 /= skin_max
        L2 /= skin_max

    # Plot the LDOS
    if method == "frenkel_pair":
        defect_pos = LargestLattice.defect_positions
        extent = ()
    elif method in ["vacancy", "substitution"]:
        r = 10
        extent = (Ls[-1] // 2 - r, Ls[-1] // 2 + r, Ls[-1] // 2 - r, Ls[-1] // 2 + r)
    elif method == "interstitial":
        r = 5
        extent = (Ls[-1] // 2 - r - 0.5, Ls[-1] // 2 + r, Ls[-1] // 2 - r - 0.5, Ls[-1] // 2 + r)

    plot_ldos(LargestLattice, ax_e, topo_ldos, ax_e_cb, extent=extent, scatter_size=75)
    plot_ldos(LargestLattice, ax_f, L1, ax_f_cb, extent=extent, scatter_size=75)
    plot_ldos(LargestLattice, ax_g, L2, ax_g_cb, extent=extent, scatter_size=75)

    # Remove some labels and legends for clarity
    ax_b.set_ylabel("")
    ax_b.legend().remove()
    ax_d.set_ylabel("")
    ax_g.set_ylabel("")

    for ax in [ax_e, ax_f, ax_g]:
        ax.set_aspect("equal")

    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_IPR_h{hdir}.svg", dpi=96)

if __name__ == "__main__":
    Ls = [10, 20, 30, 40, 50]
    #main("vacancy", Ls, 2, "z", defect_radius = 1, break_c4 = False)

    for m in ["interstitial"]:
        for hd, n in zip("xz", [4,2]):
            main(m, Ls, n, hd)
    

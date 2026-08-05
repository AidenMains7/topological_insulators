import numpy as np
import scipy.linalg as spla
from typing import Any
from decimal import Decimal

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
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib .patches import Circle

from nonhermitian_defects import DefectLattice, compute_eigenvectors_eigenvalues
from plotting2 import plot_spectrum, plot_ldos, plot_ipr


def add_rotated_ax(fig, rotation_angle):
    transform = Affine2D().rotate_deg(rotation_angle)

    bounds = (0, 12, 0, 12)
    grid_helper = floating_axes.GridHelperCurveLinear(transform, bounds)

    ax_r = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    ax_r.set_aspect("equal")

    ax_r.set_position((0.35, 0.25, 0.5, 0.5))
    fig.add_subplot(ax_r, label="r")


def war_crime():
    fig = plt.figure()

    # Master Gridspec
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1., 1., 1., 1.], wspace=0.4)

    # Gridspec for IPR
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 0])
    ax1 = fig.add_subplot(gs0[0, 0], label="a")
    ax2 = fig.add_subplot(gs0[0, 1], label="b", sharey=ax1)

    # Gridspec for first spectrum
    gs1_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], width_ratios=[1, .1])
    ax3 = fig.add_subplot(gs1_a[0, 0], label="c")
    ax3_cb = fig.add_subplot(gs1_a[0, 1], label="c_cb")

    # Gridspec for second spectrum
    gs1_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, -1], width_ratios=[1, .1])
    ax4 = fig.add_subplot(gs1_b[0, 0], label="d")
    ax4_cb = fig.add_subplot(gs1_b[0, 1], label="d_cb")

    ## Gridspec for LDOS of topological modes
    #gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[.1, 1])
    #ax5 = fig.add_subplot(gs2[1, 0], label="e")
    #ax5_cb = fig.add_subplot(gs2[0, 0], label="e_cb")

    # Gridspec for first LDOS of skin effect 
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 1], width_ratios=[1., .1])
    ax6 = fig.add_subplot(gs3[0, 0], label="f")
    ax6_cb = fig.add_subplot(gs3[0, 1], label="f_cb")

    # Gridspec for second LDOS of skin effect
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, -1], width_ratios=[1., .1])
    ax7 = fig.add_subplot(gs4[0, 0], label="g")
    ax7_cb = fig.add_subplot(gs4[0, 1], label="g_cb")



    # Now to add the rotated center ax
    add_rotated_ax(fig, 45)

    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(width=1.)
        for k in ax.spines.keys():
            ax.spines[k].set_linewidth(1.)

    return fig



def main(method:str, Ls:np.ndarray):
    fig = war_crime()
    ax_a, ax_b, ax_c, ax_c_cb, ax_d, ax_d_cb, ax_f, ax_f_cb, ax_g, ax_g_cb, ax_e = fig.axes
    
    lattices = [DefectLattice(L, L, method, True) for L in Ls]
    ed1s = [compute_eigenvectors_eigenvalues(Lattice, -1., (0.5, 0., 0.), (1.5, 0., 0.)) for Lattice in lattices]
    ed2s = [compute_eigenvectors_eigenvalues(Lattice, -1., (1.5, 0., 0.), (0.5, 0., 0.)) for Lattice in lattices]


    # Plot the IPRs
    for i, (eds, ipr_ax) in enumerate(zip([ed1s, ed2s], [ax_a, ax_b])):
        evs = tuple([ed["eigenvalues"] for ed in eds])
        iprs = tuple([ed["left_ipr"] for ed in eds])
        plot_ipr(ipr_ax, evs, iprs, Ls, eds[-1]["selected_idxs"] if i == 0 else None)

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

    aux = ax_e.get_aux_axes(Affine2D().rotate_deg(45))
    plot_ldos(lattices[-1], aux, topo_ldos)
    plot_ldos(lattices[1], ax_f, L1, ax_f_cb)
    plot_ldos(lattices[-1], ax_g, L2, ax_g_cb)

    ax_b.set_ylabel("")
    ax_b.set_yticklabels([])
    ax_b.legend().remove()
    ax_d.set_ylabel("")
    ax_g.set_ylabel("")

    for ax in [ax_e, ax_f, ax_g]:
        ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    main("vacancy", (10, 12))

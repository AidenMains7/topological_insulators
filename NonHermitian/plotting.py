import numpy as np
import scipy.linalg as spla
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
from matplotlib import colormaps, patches
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from nonhermitian_defects import DefectLattice, compute_ipr, compute_eigenvectors_eigenvalues, compute_hamiltonian


def plot_on_lattice(ldos_ax, Lattice:DefectLattice, color_array:np.ndarray, plot_type:str, 
                    cmap:str = 'cividis', title:"str|None" = None, tick_fontsize:int = 16, label_fontsize:int = 20, scatter_size:int=10,
                    rasterized:bool = True, extent=None, plot_colorbar:bool = True):

    fig = ldos_ax.get_figure()
    lattice = Lattice.lattice
    X = Lattice.X
    Y = Lattice.Y
    if extent is None:
        extent = (np.min(X), np.max(X), np.min(Y), np.max(Y))

    if plot_type == 'trisurf':
        ax_pos = ldos_ax.get_position()
        ldos_ax.remove()
        ldos_ax = fig.add_axes(ax_pos, projection="3d") 
        plot = ldos_ax.plot_trisurf(X, Y, color_array, cmap=cmap, linewidth=0.2, antialiased=False, rasterized=rasterized) 
    elif plot_type == 'scatter':
        plot = ldos_ax.scatter(X, Y, c=color_array, cmap=cmap, s=scatter_size, marker='.', rasterized=rasterized)
    elif plot_type == 'imshow':
        Z = np.full(lattice.size, np.nan)
        filled_idxs = np.argwhere(lattice.flatten() >= 0).flatten()
        Z[filled_idxs] = color_array
        plot = ldos_ax.imshow(Z.reshape(lattice.shape), cmap=cmap, origin='lower', extent=extent, rasterized=rasterized) 
    elif plot_type == "tripcolor":
        triang = tri.Triangulation(X, Y)
        xtri = triang.x[triang.triangles]
        ytri = triang.y[triang.triangles]
        l01 = np.sqrt((xtri[:,1] - xtri[:,0])**2 + (ytri[:,1] - ytri[:,0])**2)
        l12 = np.sqrt((xtri[:,2] - xtri[:,1])**2 + (ytri[:,2] - ytri[:,1])**2)
        l20 = np.sqrt((xtri[:,0] - xtri[:,2])**2 + (ytri[:,0] - ytri[:,2])**2)
        lmax = np.maximum.reduce([l01, l12, l20])
        mask = lmax > np.sqrt(2) + 1e-6
        triang.set_mask(mask)
        plot = ldos_ax.tripcolor(triang, color_array, cmap=cmap, shading='flat', rasterized=rasterized)
        ldos_ax.set_xlim(extent[0], extent[1])
        ldos_ax.set_ylim(extent[2], extent[3])
    elif plot_type == "tricontourf":
        plot = ldos_ax.tricontourf(X, Y, color_array, 10, cmap=cmap, rasterized=rasterized)
    else:
        raise ValueError("Plot type not provided correctly. It is:", plot_type)

    # Colorbar
    if plot_colorbar:
        divider = make_axes_locatable(ldos_ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = fig.colorbar(plot, cax=cax)

        formatter = ticker.ScalarFormatter(useMathText = True)
        formatter.set_powerlimits((0,  0))
        cbar.formatter = formatter
        cbar.update_ticks()

        vmin, vmax = plot.get_clim()
        ticks = np.linspace(vmin, vmax, 3)
        cbar.set_ticks(ticks) 

        cbar.ax.yaxis.offsetText.set_fontsize(tick_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
    else:
        cax = None

    # Ticks
    xticks = [extent[0], extent[1]]
    yticks = [extent[2], extent[3]]

    ldos_ax.set_xticks(xticks)
    ldos_ax.set_yticks(yticks)
    ldos_ax.set_xticklabels([int(extent[0] + 1), int(extent[1] + 1)], fontsize=tick_fontsize) 
    ldos_ax.set_yticklabels([int(extent[2] + 1), int(extent[3] + 1)], fontsize=tick_fontsize) 

    ldos_ax.set_xlabel("$x$", fontsize=label_fontsize, labelpad=-15)
    ldos_ax.set_ylabel("$y$", rotation=0, fontsize=label_fontsize, labelpad=-10)

    ldos_ax.set_title(title, fontsize=16) 
    return ldos_ax, cax


def plot_complex_spectrum(spectrum_ax:Axes, eigenvalues:np.ndarray[Any, np.dtype[np.complexfloating]], scatter_kwargs = {}, highlighted_idxs:"int|None" = None, zoomGap:bool = False):
    eig_real, eig_imag = eigenvalues.real, eigenvalues.imag 
    scat = spectrum_ax.scatter(eig_real, eig_imag, **scatter_kwargs, rasterized = False)
    #scat_real = spectrum_ax.scatter(np.arange(len(eig_real)), eig_real, c='blue', s=25, zorder=2, rasterized = False)
    #scat_imag = spectrum_ax.scatter(np.arange(len(eig_imag)), eig_imag, c='orange', s=25, zorder=2, rasterized = False)

    if isinstance(highlighted_idxs, (np.ndarray, list, tuple)):
        scat2 = spectrum_ax.scatter(eig_real[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=2, rasterized = False)
        #scat_real2 = spectrum_ax.scatter(np.arange(len(eig_real))[highlighted_idxs], eig_real[highlighted_idxs], c='red', s=25, zorder=3, rasterized = False)
        #scat_imag2 = spectrum_ax.scatter(np.arange(len(eig_imag))[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=3, rasterized = False)

    xmax, ymax = np.round(np.max(eig_real), 1), np.round(np.max(eig_imag), 1)
    xticks = [-xmax, 0.0, xmax]
    yticks = [-ymax, 0.0, ymax]
    spectrum_ax.set_xticks(xticks)
    spectrum_ax.set_yticks(yticks)
    spectrum_ax.set_xticklabels(xticks, fontsize=16)
    spectrum_ax.set_yticklabels(yticks, fontsize=16)
    spectrum_ax.set_xlabel("$\\Re(E)$", fontsize=20)
    spectrum_ax.set_ylabel("$\\Im(E)$", fontsize=20, rotation=0)


    if zoomGap:
        highlighted_eigenvalues = eigenvalues[highlighted_idxs]
        min_real = np.min(highlighted_eigenvalues.real) 
        max_real = np.max(highlighted_eigenvalues.real) 
        min_imag = np.min(highlighted_eigenvalues.imag) 
        max_imag = np.max(highlighted_eigenvalues.imag) 

        width_real = max_real - min_real
        width_imag = max_imag - min_imag

        dx = width_real + 1e-3
        dy = width_imag + 1e-3
        axins = spectrum_ax.inset_axes(
            (0.7, 0.05, 0.25, 0.25), 
            xlim = (min_real - dx, max_real + dx), ylim = (min_imag - dy, max_imag + dy),
            xticklabels = [], yticklabels = [])
        axins.scatter(eig_real, eig_imag, c='k', s=25, zorder=1)
        axins.scatter(eig_real[highlighted_idxs], eig_imag[highlighted_idxs], c='red', s=25, zorder=2)
        #axins.get_xaxis().set_visible(False)
        #axins.get_yaxis().set_visible(False)
        spectrum_ax.indicate_inset_zoom(axins, edgecolor='black')

    return spectrum_ax


def plot_spectrum_ldos(fig, axs, Lattice:DefectLattice, m0:float, h0_vector:np.ndarray, hsub_vector:"np.ndarray|None" = None, zoomGap:bool = False):
    eigvec_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h0_vector, hsub_vector, 4) 
    eigenvalues = eigvec_dict['eigenvalues']
    L, R = eigvec_dict['L'], eigvec_dict['R']
    close_idxs = eigvec_dict['selected_idxs']
    close_L, close_R = eigvec_dict['selected_left_eigenvectors'], eigvec_dict['selected_right_eigenvectors']

    if Lattice.defect_type in ["interstitial", "frenkel_pair"]:
        plot_type = "tripcolor"
    else:
        plot_type = "tripcolor"

    plot_complex_spectrum(axs[0], eigenvalues, {'c':'k'}, close_idxs, zoomGap = zoomGap)
    plot_on_lattice(axs[1], Lattice, close_L, plot_type, scatter_size = 100)
    plot_on_lattice(axs[2], Lattice, close_R, plot_type, scatter_size = 100)
    plot_on_lattice(axs[3], Lattice,       L, plot_type, scatter_size = 100)
    plot_on_lattice(axs[4], Lattice,       R, plot_type, scatter_size = 100)

    axs[0].set_title("Eigenvalue Spectra", fontsize=16)
    axs[1].set_title("Selected $|\\Psi^L_i|^2$", fontsize=16)
    axs[2].set_title("Selected $|\\Psi^R_i|^2$", fontsize=16)
    axs[3].set_title("All $|\\Psi^L_i|^2$", fontsize=16)
    axs[4].set_title("All $|\\Psi^R_i|^2$", fontsize=16)

    axs[0].set_box_aspect(0.95)

    axs[0].yaxis.set_label_coords(-0.225, 0.4625)
    for ax in axs[1:]:
        ax.set_box_aspect(1)
        ax.set_aspect('equal')

    for ax in axs[2:]:
        ax.set_ylabel("")

    plt.subplots_adjust(wspace=0.25)

    return fig, axs


def plot_many_spectrum_lr(fig, axs, defect_type: str, L: int, m0_values: list[float], 
                          h_dir: str, h0_values: list[float], hsub_values: "list[float]|None" = None, 
                          defect_radius:int = 1, break_c4:bool = False):
    assert defect_type in ['none', 'vacancy', 'schottky', 'substitution', 'interstitial', 'frenkel_pair'], "defect_type must be one of 'none', 'vacancy', 'schottky', 'substitution', 'interstitial', or 'frenkel_pair'"
    assert h_dir in 'xyz', "h_dir must be 'x', 'y', or 'z'"
    if hsub_values is not None:
        assert len(h0_values) == len(hsub_values) == len(m0_values), "h0_values, hsub_values, and m0_values must be of equal length"

    if hsub_values == None:
        hsub_values = [] * len(h0_values)

    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc('xtick.major', width=2.5)
    plt.rc('ytick.major', width=2.5)

    assert axs.shape[0] >= len(m0_values) and axs.shape[1] >= 5, "Shape of input axs array is not of suitable size"

    if len(m0_values) == 1:
        axs = np.array(axs).reshape(1, 5)

    for i, (m0, h0, hsub) in enumerate(zip(m0_values, h0_values, hsub_values)): 
        Lattice = DefectLattice(L, L, defect_type, True, schottky_separation = L // 4, 
                                frenkel_x_disp = -3.5, frenkel_y_disp = -3.5, defect_radius = defect_radius, break_c4 = break_c4)

        if h_dir == 'x':
            h_vector = np.array([h0, 0.0, 0.0])
            hsub_vector = np.array([hsub, 0.0, 0.0]) if hsub is not None else None
        elif h_dir == 'y':
            h_vector = np.array([0.0, h0, 0.0])
            hsub_vector = np.array([0.0, hsub, 0.0]) if hsub is not None else None
        elif h_dir == 'z':
            h_vector = np.array([0.0, 0.0, h0])
            hsub_vector = np.array([0.0, 0.0, hsub]) if hsub is not None else None

        fig, axs[i, :] = plot_spectrum_ldos(fig, axs[i, :], Lattice, m0, h_vector, hsub_vector)

        if Lattice.defect_type in ["vacancy", "schottky", "none"]:
            annotation_text = f'$m_0 = {m0}$\n$h_0={h0} \\hat{{{h_dir}}}$'
        elif Lattice.defect_type == "substitution":
            annotation_text = f'$m_0 = {m0}$\n$h_0={h0} \\hat{{{h_dir}}}$\n$h_0^{{\\rm sub}}={hsub} \\hat{{{h_dir}}}$'
        elif Lattice.defect_type in ["interstitial", "frenkel_pair"]:
            annotation_text = f'$m_0 = {m0}$\n$h_0={h0} \\hat{{{h_dir}}}$\n$h_0^{{\\rm int}}={hsub} \\hat{{{h_dir}}}$'

        axs[i, 0].annotate(
            annotation_text,
            xy = (0.025, 0.975),
            xycoords='axes fraction', 
            ha='left', 
            fontsize=16, 
            rotation=0,
            va='top',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
        )

    for ax in axs[:-1, :].flatten():
        ax.set_xlabel("")

    for ax in axs.flatten():
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title(ax.get_title(), fontsize=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)


def cluster_figure(defect_type:str, L, h_dir, m0_values, h0_values, hsub_values):
    
    num_rows = len(m0_values) * 3

    fig, axs = plt.subplots(num_rows, 5, figsize=(6 * 5, 6 * num_rows))

    plot_many_spectrum_lr(fig, axs[:2,  :], defect_type, L, m0_values, h_dir, h0_values, hsub_values=hsub_values)
    plot_many_spectrum_lr(fig, axs[2:4, :], defect_type, L, m0_values, h_dir, h0_values, hsub_values=hsub_values, defect_radius = 2)
    plot_many_spectrum_lr(fig, axs[4:,  :], defect_type, L, m0_values, h_dir, h0_values, hsub_values=hsub_values, defect_radius = 2, break_c4 = True)
    
    row_labels = 'abcdefghijklmnopqrstuvwxyz'[:num_rows]
    column_labels = ['i', 'ii', 'iii', 'iv', 'v']

    for i in range(num_rows):
        for j in range(5):
            axs[i, j].annotate(
            "(" + row_labels[i] + "." + column_labels[j] + ")",
            xy = (0.975, 0.975),
            xycoords='axes fraction', 
            ha='right', 
            fontsize=24, 
            rotation=0,
            va='top',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
            )

    plt.savefig(f"./NonHermitian/Plots/{defect_type.capitalize()}/{defect_type.lower()}_cluster_h{h_dir}.png", bbox_inches='tight')


def plot_ldos_ipr(Lattice, m0, h0, hsub):

    eig_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h0, hsub)
    eigenvalues = eig_dict['eigenvalues']
    left_eigenvectors, right_eigenvectors = eig_dict['left_eigenvectors'], eig_dict['right_eigenvectors']
    L, R = eig_dict['L'], eig_dict['R']
    close_L, close_R = eig_dict['selected_left_eigenvectors'], eig_dict['selected_right_eigenvectors']
    left_ipr = compute_ipr(left_eigenvectors)
    
    fig, axs = plt.subplots(1, 4, figsize=(6 * 4, 6))

    selected_idxs = eig_dict['selected_idxs']
    x_energy = np.abs(eigenvalues) * np.sign(eigenvalues.real)

    axs[0].scatter(x_energy, left_ipr, c='k', s=25, zorder=0)
    axs[0].scatter(x_energy[selected_idxs], left_ipr[selected_idxs], c='r', s=25, zorder=1)
    axs[1].scatter(eigenvalues.real, eigenvalues.imag, c='k', s=25, zorder=0)
    axs[1].scatter(eigenvalues.real[selected_idxs], eigenvalues.imag[selected_idxs], c='r', s=25, zorder=1)
    plot_on_lattice(axs[2], Lattice, close_L, plot_type = "tripcolor", title = "Left Eigenvectors")
    plot_on_lattice(axs[3], Lattice, close_R, plot_type = "tripcolor", title = "Right Eigenvectors")

    plt.show()


def plot_spectrum_and_ldos():
    L = 20
    Lattice = DefectLattice(L, L, 'substitution', True, defect_radius = 1)

    H = compute_hamiltonian(Lattice, -1., np.array([0.5, 0.0, 0.0]), 1.0, 1.0, np.array([1.5, 0.0, 0.0]))
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(H, left=True, right=True, overwrite_a=True) 
    sort_idxs = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sort_idxs]
    left_eigenvectors = left_eigenvectors[:, sort_idxs]
    right_eigenvectors = right_eigenvectors[:, sort_idxs]

    eigenvalue_dict = compute_eigenvectors_eigenvalues(Lattice, -1., 
                        np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0]), n_closest_to_zero = 2)
  
    eigenvalues = eigenvalue_dict['eigenvalues']
    selected_idxs = eigenvalue_dict['selected_idxs']
    selected_left_eigenvectors = eigenvalue_dict['selected_left_eigenvectors']
    selected_right_eigenvectors = eigenvalue_dict['selected_right_eigenvectors']
    L = eigenvalue_dict['L']
    R = eigenvalue_dict['R']
    n = np.arange(len(eigenvalues))
    
    # Spectrum & LDOS
    if 0:
        fig, axs = plt.subplots(1, 7, figsize=(6*7, 6))
        axs[0].scatter(n, eigenvalues.real, c='black', s=25, zorder=0)
        axs[0].scatter(n[selected_idxs], eigenvalues.real[selected_idxs], c='red', s=25, zorder=1)
        axs[1].scatter(n, eigenvalues.imag, c='black', s=25, zorder=0)
        axs[1].scatter(n[selected_idxs], eigenvalues.imag[selected_idxs], c='red', s=25, zorder=1)

        for ax in axs[:2]:
            ax.set_xlabel("$n$")
            ax.set_ylabel("$E_n$")
        
        axs[0].set_title("Real Part of Eigenvalues")
        axs[1].set_title("Imaginary Part of Eigenvalues")

        axs[2].scatter(eigenvalues.real, eigenvalues.imag, c='black', s=25, zorder=0)
        axs[2].scatter(eigenvalues.real[selected_idxs], eigenvalues.imag[selected_idxs], c='red', s=25, zorder=1)
        axs[2].set_xlabel("$\\Re(E)$")
        axs[2].set_ylabel("$\\Im(E)$")
        axs[2].set_title("Complex Eigenvalue Spectrum")

        axs[3].tripcolor(Lattice.X, Lattice.Y, selected_left_eigenvectors, shading='flat', cmap='cividis')
        axs[4].tripcolor(Lattice.X, Lattice.Y, selected_right_eigenvectors, shading='flat', cmap='cividis')
        axs[5].tripcolor(Lattice.X, Lattice.Y, L, shading='flat', cmap='cividis')
        axs[6].tripcolor(Lattice.X, Lattice.Y, R, shading='flat', cmap='cividis')
        plt.savefig('./NonHermitian/Plots/temp.png')


def plot_ipr(method:str, Ls: np.typing.ArrayLike, m0: float, h0: float, hsub: float, h_dir: str, ax, defect_radius: int = 1, 
             break_c4:bool = False, plot_inset:bool = True, n_selected: int = 4, 
             left_or_right:str = 'left', extent=None) -> Axes:
    left_IPRs = []
    eigenvalues_list = []
    left_eigvecs_list = []
    idxs_list = []
    close_list = []
    close_list = []

    h_dir_mapping = {'x': 0, 'y': 1, 'z': 2}
    h0_vector = np.zeros(3)
    hsub_vector = np.zeros(3)
    h0_vector[h_dir_mapping[h_dir]] = h0
    hsub_vector[h_dir_mapping[h_dir]] = hsub

    Ls = np.sort(Ls)

    all_colors = np.array(['tab:purple', 'tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:olive', 'tab:cyan'])
    colors = all_colors[np.arange(Ls.size)]
    colors[-1] = 'tab:red'
    
    for L, c in zip(Ls, colors):
        schottky_sep = L // 4
        if schottky_sep % 2 == 0:
            schottky_sep += 1
        print(f"L={L} : schottky_separation={schottky_sep}")
        Lattice = DefectLattice(L, L, method, True, defect_radius = defect_radius, schottky_separation = schottky_sep, break_c4 = break_c4)

        eigval_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h0_vector, hsub_vector, n_closest_to_zero = n_selected)
        eigenvalues = eigval_dict['eigenvalues']
        left_eigenvectors = eigval_dict['left_eigenvectors']
        right_eigenvectors = eigval_dict['right_eigenvectors']

        if left_or_right == 'left':
            IPR = compute_ipr(left_eigenvectors)
            close_list.append(eigval_dict['selected_left_eigenvectors'])
        else:
            IPR = compute_ipr(right_eigenvectors)
            close_list.append(eigval_dict['selected_right_eigenvectors'])

        left_IPRs.append(IPR)
        eigenvalues_list.append(eigenvalues)
        left_eigvecs_list.append(left_eigenvectors)
        idxs_list.append(eigval_dict['selected_idxs'])
        print(f"Completed computation for L = {L}")

        if L == Ls[-1] and plot_inset:  
            inset_ax = ax.inset_axes([0.5, 0.1, 0.35, 0.35])
            inset_ax.set_zorder(100)


            close_eigvec = close_list[-1]
            selected_idxs = idxs_list[-1]

            extent = (L / 2 - defect_radius - 3, L / 2 + defect_radius + 3, L / 2 - defect_radius - 3, L / 2 + defect_radius + 3)
            extent = (L / 2 - 3.5 - 5, L / 2 + 3, L / 2 - 3.5 - 2, L / 2 + 3) if method == 'frenkel_pair' else extent
            plot_on_lattice(inset_ax, Lattice, close_eigvec, 
                            plot_type = "tripcolor", title = "",
                            tick_fontsize = 10, label_fontsize = 10,
                            extent = extent if method != 'schottky' else None)
    
            t = np.abs(eigenvalues) * np.sign(eigenvalues.real)
            ax.scatter(t[selected_idxs], left_IPRs[-1][selected_idxs], c=c, marker='*', s=100, alpha=1.0, zorder=100)


    for i, (L, eigs, left_IPR, c) in enumerate(zip(Ls, eigenvalues_list, left_IPRs, colors)):
        t = np.abs(eigs) * np.sign(eigs.real)
        label = f"$L={L}$"
        ax.scatter(t, left_IPR,  s=25, alpha=0.5, label=label, color=c, zorder=i)
        
        if plot_inset:
            rect = patches.Rectangle((0.4, 0.0), 0.6, 0.6, transform = ax.transAxes, edgecolor='none', facecolor='white', zorder=99, alpha=0.5)
            ax.add_patch(rect)

    ax.set_ylabel("IPR")
    ax.set_title("Left Eigenvector IPR")
    ax.set_xlabel("$|E|\\times {\\rm sign}(\\Re(E))$")
    ax.legend(bbox_to_anchor=(0.0, 0.5, 1.0, 0.5), loc="best", fontsize=12)

    return ax


def plot_ipr_figure(method, h_dir, Ls, m0_values, h0_values, hsub_values, n_selected=4, left_or_right='left', out_fname:str = "", extensions:list[str]=[".png"]):
    fig, axs = plt.subplots(1, len(m0_values), figsize=(6 * len(m0_values), 6))
    axs = np.array(axs).reshape(len(m0_values))
    all_annotations = [f"({letter})" for letter in 'abcdefghijklmnopqrstuvwxyz']

    for i, (m0, h0, hsub) in enumerate(zip(m0_values, h0_values, hsub_values)):
        print(f"Plotting for m0={m0}, h0={h0}, hsub={hsub}")
        axs[i] = plot_ipr(method, Ls, m0, h0, hsub, h_dir, axs[i], plot_inset = True if i == 0 else False, n_selected = n_selected, left_or_right = left_or_right)
        axs[i].annotate(
            all_annotations[i],
            xy = (0.025, 0.975),
            xycoords='axes fraction', 
            ha='left', 
            fontsize=24, 
            rotation=0,
            va='top',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        )

    if out_fname == "":
        for ext in extensions: plt.savefig(f'./NonHermitian/Plots/IPR/{method}_ipr_{h_dir}' + ext, bbox_inches='tight')
    else:
        for ext in extensions: plt.savefig(out_fname + ext, bbox_inches='tight')
        

def plot_ipr_figure_cluster(method, h_dir, Ls, m0_values, h0_values, hsub_values, n_idxs_values = [4, 4, 4], left_or_right='left'):
    fig, axs = plt.subplots(3, len(m0_values), figsize=(6 * len(m0_values), 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    row_labels = "abc"
    column_labels = ["i", "ii"]

    radiuses = [1, 2, 2]
    break_c4_flags = [False, False, True]
    for i in range(3):
        for j, (m0, h0, hsub, nidx) in enumerate(zip(m0_values, h0_values, hsub_values, n_idxs_values)):
            plot_inset = True if (j == 0) else False
            axs[i, j] = plot_ipr(method, Ls, m0, h0, hsub, h_dir, axs[i, j], defect_radius = radiuses[i], 
                                 break_c4 = break_c4_flags[i], plot_inset = plot_inset, n_selected = nidx, 
                                 left_or_right = left_or_right)
            axs[i, j].annotate(
                "(" + row_labels[i] + "." + column_labels[j] + ")",
                xy = (0.025, 0.975),
                xycoords='axes fraction', 
                ha='left', 
                fontsize=24, 
                rotation=0,
                va='top',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            )

    #plt.tight_layout()
    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_cluster_ipr_{h_dir}.png", bbox_inches='tight')
    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_cluster_ipr_{h_dir}.svg", bbox_inches='tight')
    

def compare_ipr_vs_radius(Ls, radii:np.ndarray[Any, np.dtype[np.integer]]=np.arange(1, 5, dtype=int), cmap='cividis'):

    assert np.max(radii) <= np.min(Ls) // 2, "Maximum defect radius must be less than or equal to half the minimum lattice size"


    fig, axs = plt.subplots(2, len(radii), figsize=(6 * len(radii), 12), sharex=True, sharey=True)

    colors = colormaps.get_cmap(cmap)(np.linspace(0, 1, len(Ls)))

    for k, L in enumerate(Ls):
        for i, r in enumerate(radii):
            Sub_Lattice = DefectLattice(L, L, 'substitution', True, defect_radius = r)
            sub_eig_dict = compute_eigenvectors_eigenvalues(Sub_Lattice, -1., np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0]), n_closest_to_zero = 4)
            
            Int_Lattice = DefectLattice(L, L, 'interstitial', True, defect_radius = r)
            int_eig_dict = compute_eigenvectors_eigenvalues(Int_Lattice, -1., np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0]), n_closest_to_zero = 4)

            sub_ipr = compute_ipr(sub_eig_dict['left_eigenvectors'])
            int_ipr = compute_ipr(int_eig_dict['left_eigenvectors'])

            axs[0, i].scatter(np.abs(sub_eig_dict['eigenvalues']) * np.sign(sub_eig_dict['eigenvalues'].real), sub_ipr, label=f"L={L}", s=25, alpha=0.5, color=colors[k])
            axs[0, i].set_xlabel("$|E|\\times {\\rm sign}(\\Re(E))$")
            axs[0, i].set_ylabel("IPR")

            axs[1, i].scatter(np.abs(int_eig_dict['eigenvalues']) * np.sign(int_eig_dict['eigenvalues'].real), int_ipr, label=f"L={L}", s=25, alpha=0.5, color=colors[k])
            axs[1, i].set_xlabel("$|E|\\times {\\rm sign}(\\Re(E))$")
            axs[1, i].set_ylabel("IPR")

            left_eigvecs = [sub_eig_dict['selected_left_eigenvectors'], int_eig_dict['selected_left_eigenvectors']]
            lattices = [Sub_Lattice, Int_Lattice]
        
            if L == Ls[-1]: 
                axs[0, i].scatter(np.abs(sub_eig_dict['eigenvalues'][sub_eig_dict['selected_idxs']]) * np.sign(sub_eig_dict['eigenvalues'][sub_eig_dict['selected_idxs']].real), sub_ipr[sub_eig_dict['selected_idxs']], c='red', s=25, alpha=0.5)
                axs[1, i].scatter(np.abs(int_eig_dict['eigenvalues'][int_eig_dict['selected_idxs']]) * np.sign(int_eig_dict['eigenvalues'][int_eig_dict['selected_idxs']].real), int_ipr[int_eig_dict['selected_idxs']], c='red', s=25, alpha=0.5)
                for j in range(2):
                    axs[j, i].annotate(
                        "(" + "ab"[j] + "." + ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"][i] + ")",
                        xy = (0.025, 0.975),
                        xycoords='axes fraction',
                        ha='left',
                        fontsize=24,
                        rotation=0,
                        va='top',
                    )

                    inset_ax = axs[j, i].inset_axes([0.5, 0.1, 0.35, 0.35])
                    inset_ax.set_zorder(100)

                    plot_on_lattice(inset_ax, lattices[j], left_eigvecs[j], plot_type = "tripcolor", title = "", tick_fontsize = 10, label_fontsize = 10)
                    
    for ax in axs.flatten():
        ax.legend(loc='upper right')

    plt.savefig(f"./NonHermitian/Plots/IPR/substitution_ipr_vs_radius.png", bbox_inches='tight')


def compare_ipr_vs_hsub(Ls, m0, h0, hsub_values, h_dir, defect_type='substitution', cmap='cividis'):
    fig, axs = plt.subplots(1, len(hsub_values), figsize=(6 * len(hsub_values), 6))
    colors = colormaps.get_cmap(cmap)(np.linspace(0, 1, len(Ls)))

    for k, L in enumerate(Ls):
        for i, hsub in enumerate(hsub_values):
            Lattice = DefectLattice(L, L, defect_type, True, defect_radius = 1)
            h_dir_mapping = {'x': 0, 'y': 1, 'z': 2}
            h0_vector = np.zeros(3)
            hsub_vector = np.zeros(3)
            h0_vector[h_dir_mapping[h_dir]] = h0
            hsub_vector[h_dir_mapping[h_dir]] = hsub

            eig_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h0_vector, hsub_vector, n_closest_to_zero=4)
            left_ipr = compute_ipr(eig_dict['left_eigenvectors'])
            eigenvalues = eig_dict['eigenvalues']

            axs[i].scatter(np.abs(eigenvalues) * np.sign(eigenvalues.real), left_ipr, label=f"L={L}", s=25, alpha=0.5, color=colors[k])
            axs[i].set_xlabel("$|E|\\times {\\rm sign}(\\Re(E))$")
            axs[i].set_ylabel("IPR")
            axs[i].set_title(f"$h_{{\\rm sub}}={hsub}$")

    for ax in axs:
        ax.legend(loc='upper right')

    plt.savefig(f"./NonHermitian/Plots/IPR/{defect_type}_ipr_vs_hsub.png", bbox_inches='tight')


def figure_layout(fig, master_gs, do_inset:bool = True, epsilon:float = 0.1):
    # Create figure and axes
    golden_ratio = (1 + np.sqrt(5)) / 2

    small_width = 1.0
    large_width = small_width * golden_ratio
    colorbar_width = small_width / 10
    
    #gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=(large_width, small_width + colorbar_width))

    gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=master_gs, width_ratios=(large_width, small_width + colorbar_width))
    ax_a = fig.add_subplot(gs[:, 0], label="a")

    sub_gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], width_ratios = (small_width, small_width / 10), wspace=0.1  )
    ax_b = fig.add_subplot(sub_gs0[0, 0], label="b")
    ax_b_cb = fig.add_subplot(sub_gs0[0, 1], label="b_colorbar")
    ax_c = fig.add_subplot(gs[1, 1], label="c")

    plt.subplots_adjust(hspace=0.25, wspace=0.25)

    if do_inset:
        ax_a_inset = ax_a.inset_axes((0.0 + epsilon, epsilon, 0.5 - 2 * epsilon, 0.5 - 2 * epsilon), label="a_inset")
        fig.add_axes(ax_a_inset)


def plot_spectrum_ipr_color(ax,  cbar_ax, eigenvalues, ipr, selected_idxs=[], cmap='jet'):
    scat1 = ax.scatter(eigenvalues.real, eigenvalues.imag, c=ipr, cmap=cmap, zorder=0)
    if len(selected_idxs) > 0:
        print(selected_idxs)
        scat2 = ax.scatter(eigenvalues.real[selected_idxs], eigenvalues.imag[selected_idxs], c=ipr[selected_idxs], marker="*", s=100, cmap=cmap)
    cbar = plt.colorbar(scat1, cbar_ax)


def remove_ax_ticks_and_labels(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("auto")


def format_colorbar(cbar, vmin, vmax):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    formatter.set_scientific(True)
    cbar.formatter = formatter
    cbar.update_ticks()
    
    cbar.ax.yaxis.set_ticks([vmin, (vmax + vmin) / 2, vmax])
    cbar.ax.yaxis.offsetText.set_position((0., 0.0))
    cbar.ax.yaxis.offsetText.set_fontsize(14)
    cbar.ax.tick_params(labelsize=14)


def spectrum_and_ipr_figure(axes, method, m0, h0_vector, hsub_vector, Ls, ns, do_additional_plotting:bool = True):
    Ls = np.sort(Ls)

    ldos_plot_radius = max(Ls[-1] // 4, 5)
    ldos_plot_radius = min(ldos_plot_radius, Ls[-1] // 2)
    all_colors = np.array(['tab:purple', 'tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:olive', 'tab:cyan'])
    colors = all_colors[np.arange(Ls.size)]
    colors[-1] = 'r'

    # Compute all observables
    lattices = []
    eig_dicts = []
    for L, n in zip(Ls, ns):
        schottky_sep = L // 4
        if schottky_sep % 2 == 0:
            schottky_sep += 1
        if method == "schottky": print(f"L={L} : schottky_separation={schottky_sep}")
        Lattice = DefectLattice(L, L, method, True, schottky_separation=schottky_sep)
        ed = compute_eigenvectors_eigenvalues(Lattice, m0, h0_vector, hsub_vector, n)
        lattices.append(Lattice)
        eig_dicts.append(ed)

    if do_additional_plotting:
        ax_a, ax_b, ax_b_cb, ax_c, ax_a_inset = axes  
    else:
        ax_a, ax_b, ax_b_cb, ax_c = axes

    # Axis (a)
    for L, ed, c in zip(Ls, eig_dicts, colors):
        ipr = ed["left_ipr"]
        eigenvalues = ed["eigenvalues"]
        selected_idxs = ed["selected_idxs"]

        a_xaxis = np.abs(eigenvalues) * np.sign(eigenvalues.real)
        ax_a.scatter(a_xaxis, 1e2*ipr, zorder=0, c=c, alpha=0.5, label=f"$L={L}$")
        if do_additional_plotting and L == np.max(Ls): ax_a.scatter(a_xaxis[selected_idxs], 1e2*ipr[selected_idxs], marker="*", c="r", s=100, zorder=1)
        ax_a.set_xlabel("$|E| \\times {\\rm sign}(E)$")
        ax_a.set_ylabel("${\\rm IPR} \\times 10^2$")
        ax_a.legend(loc = 'upper right')

    # Axis (a_inset)
    if do_additional_plotting:
        ax_a_inset, a_inset_cbar = plot_on_lattice(ax_a_inset, Lattice, ed["selected_left_eigenvectors"], 
                                                "tripcolor", "cividis", plot_colorbar=False,
                                                extent = (L // 2 - ldos_plot_radius, L // 2 + ldos_plot_radius, L // 2 - ldos_plot_radius, L // 2 + ldos_plot_radius))
        #remove_ax_ticks_and_labels(ax_a_inset)
        ax_a_inset.tick_params(axis="both", labelsize=10)
        ax_a_inset.set_xlabel("$X$", fontsize=10)
        ax_a_inset.set_ylabel("$Y$", fontsize=10)

    # Axis (b) and (b_colorbar)
    selected_mask = np.full(eigenvalues.size, False)
    selected_mask[selected_idxs] = True 
    b_scat1 = ax_b.scatter(eigenvalues.real[~selected_mask], eigenvalues.imag[~selected_mask] * 1e1, c=ipr[~selected_mask], cmap='jet', zorder=0)
    vmin, vmax = b_scat1.get_clim()
    if do_additional_plotting: b_scat2 = ax_b.scatter(eigenvalues.real[selected_idxs], eigenvalues.imag[selected_idxs] * 1e1, c=ipr[selected_idxs], marker="*", s=100, cmap='jet', vmin=vmin, vmax=vmax)
    print(eigenvalues.real[selected_idxs], eigenvalues.imag[selected_idxs], ipr[selected_idxs])

    b_scat1.set_clim(vmin=0)
    b_cbar = plt.colorbar(b_scat1, ax_b_cb)

    format_colorbar(b_cbar, *b_scat1.get_clim())
    ax_b.set_xlabel("$\\Re (E)$")
    ax_b.set_ylabel("$\\Im (E) \\times 10^1$")

    # Axis (c) and (c_colorbar)
    ax_c, ax_c_cb = plot_on_lattice(ax_c, Lattice, ed["L"], 
                                    "tripcolor", "cividis", plot_colorbar=False,
                                    extent = (L // 2 - ldos_plot_radius, L // 2 + ldos_plot_radius, L // 2 - ldos_plot_radius, L // 2 + ldos_plot_radius))
    ax_c.tick_params(axis="both", labelsize=10)
    ax_c.set_xlabel("$X$", fontsize=10)
    ax_c.set_ylabel("$Y$", fontsize=10)
    
# ============================
# ============================
# ============================


def plot_figs_12_to_21(method, Ls, ns, hdir):
    fig = plt.figure(figsize=(20, 10))
    plt.subplots_adjust(wspace=0.1)
    master_gs = gridspec.GridSpec(1, 2, figure=fig)

    figure_layout(fig, master_gs[0, 0])
    figure_layout(fig, master_gs[0, 1], False)

    axes = np.array(fig.axes)
    axes_chunks = [axes[:5], axes[5:]]

    
    h_dir_mapping = {'x': 0, 'y': 1, 'z': 2}
    v1 = np.zeros(3)
    v2 = np.zeros(3)
    v1[h_dir_mapping[hdir]] = 0.5
    v2[h_dir_mapping[hdir]] = 1.5

    spectrum_and_ipr_figure(axes_chunks[0], method, -1.0, v1, v2, Ls, ns)
    spectrum_and_ipr_figure(axes_chunks[1], method, -1.0, v2, v1, Ls, ns, do_additional_plotting = False)

    labels = ["(" + l + ")" for l in "ab_c_de_f"]
    for ax, lab in zip(fig.axes, labels):
        if lab[1] == "_":
            continue
        ax.annotate(
            lab,
            xy = (0.025, 0.975),
            xycoords='axes fraction',
            ha='left',
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
        )

    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_ipr_h{hdir}.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    Ls = [10, 20, 30, 40]
    ns = [2, 2, 2, 2]
    for hd in "xy":
        for m  in ["vacancy", "schottky", "frenkel_pair"]:
            plot_figs_12_to_21(m, Ls, ns, hd)
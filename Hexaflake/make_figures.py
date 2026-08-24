import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import h5py
from typing import cast
from HaldaneModel import compute_geometric_data, compute_hamiltonian

def get_numbers(phi, M, bis, disorder, tol=1e-6, plot_disorder=False):
    bis = np.round(bis.flatten(), int(-np.log10(tol)))
    disorder = np.round(disorder.flatten(), int(-np.log10(tol)))


    in_topo_mask = (M < 3 * np.sqrt(3) * np.sin(phi))
    # Number of points within the topological regime given by M < 3 * sqrt(3) * sin(phi)
    kdxs = np.argwhere(in_topo_mask)
    n_in_topo = kdxs.shape[0]

    # Number of points inside the topological regime with bott index < 0.0
    jdxs = np.argwhere(in_topo_mask & (bis < 0.0))
    n_in_topo_bott = jdxs.shape[0]

    # Number of points with disorder <= -1.0
    n_disorder_is_1 = np.argwhere(disorder <= -1.0 + tol).shape[0]

    if plot_disorder:
        plt.scatter(phi[kdxs], M[kdxs], c=disorder[kdxs], label="Computed Points")
        plt.colorbar()
        t = np.linspace(0, np.pi, 101)
        plt.plot(t, 3 * np.sqrt(3) * np.sin(t), 'k--', label="$M = 3 \\sqrt{3} \\sin(\\phi)$")
        plt.legend()
        plt.show()

    return n_in_topo, n_in_topo_bott, n_disorder_is_1


def get_unique_coords_from_grids(arrays:list[np.ndarray], tol=1e-6) -> np.ndarray:
    """
    Returns values of given arrays on the same grid.
    
    Assumes the second dimension of each array is the same, and that the arrays are 1D or 2D
    Assumes the grid position is given by (arr[:, 0], arr[:, 1]) for each array, and that the values are given by arr[:, 2:]
    """
    assert np.all([arr.ndim in [1, 2] for arr in arrays]), "All arrays must be 1D or 2D"
    assert np.all([arr.shape[1] == arrays[0].shape[1] for arr in arrays]), "All arrays must have the same first dimension"

    stack = np.concatenate(arrays, axis=0)
    coords = np.round(stack[:, :2], -int(np.log10(tol)))
    values = stack[:, 2:]
    unique_coords = np.unique(coords, axis=0)

    values_on_unique_coords = []
    for (x, y) in unique_coords:
        idxs = np.argwhere((coords[:, 0] == x) & (coords[:, 1] == y)).flatten()
        not_nan_counts = np.count_nonzero(~np.isnan(values[idxs]), axis=0)
        assert np.all(not_nan_counts <= 1), f"Multiple non-NaN values found for coordinate ({x}, {y}) : {values[idxs]}"
        v = np.nanmax(values[idxs], axis=0)
        values_on_unique_coords.append(v)

    return np.column_stack((unique_coords, np.array(values_on_unique_coords)))


def get_values_from_file(generation: int):
    clean_filename = f"./Hexaflake/Data/Generation {generation}/site_elim_g{generation}_(25_by_25).h5"
    disorder_filename = clean_filename.replace(".h5", "_w1.0.h5")
    with h5py.File(clean_filename, "r") as f:
        phi = cast(np.ndarray, f["phi"])[()]
        M = cast(np.ndarray, f["M"])[()]
        bott_index = np.round(cast(np.ndarray, f["bott_index"])[()].flatten(), 6)
    with h5py.File(disorder_filename, "r") as f:
        try:
            idxs = cast(np.ndarray, f["computed_idxs"])[()]
            disorder = np.round(cast(np.ndarray, f["disorder_flat"])[()], 6)
            unique = np.column_stack((phi[idxs], M[idxs], bott_index[idxs], disorder[idxs]))
        except: 
            phi_disorder = cast(np.ndarray, f["phi"])[()].flatten()
            M_disorder = cast(np.ndarray, f["M"])[()].flatten()
            disorder = np.round(cast(np.ndarray, f["disorder"])[()].flatten(), 6)
            arr1 = np.column_stack((phi, M, bott_index, np.full(phi.shape, np.nan)))
            arr2 = np.column_stack((phi_disorder, M_disorder, np.full(phi_disorder.shape, np.nan), disorder))
            unique = get_unique_coords_from_grids([arr1, arr2])
    return get_numbers(*unique.T), unique


def plot_percentages(generations = np.array([2, 3, 4]), ax=None, verbose=False):
    """
    We assume that the data files for all generations lies on the same grid of phi and M values. 
    """

    if ax == None:
        fig, ax = plt.subplots(1, 1)

    numbers = []
    values = []
    for g in generations:
        ns, vals = get_values_from_file(g)
        numbers.append(ns)
        values.append(vals)

    numbers = np.array(numbers)

    n_in_top = numbers[0][0]
    pristine_ratios = numbers[:, 1] / n_in_top
    disorder_ratios = numbers[:, 2] / n_in_top
    ratio_ratios = pristine_ratios / disorder_ratios

    if verbose:
        print("Pristine Ratios (n_in_topo_bott / n_in_topo):", pristine_ratios)
        print("Disorder Ratios (n_disorder_is_1 / n_in_topo):", disorder_ratios)
        print("Ratio of Ratios (Pristine / Disorder):", ratio_ratios)

    n_sites = 6 * (7 ** np.array(generations))
    x = 1/n_sites * 1e3
    ax.scatter(x, pristine_ratios, label="W=0.0", color='blue', marker="*")
    ax.scatter(x, disorder_ratios, label="W=1.0", color='red', marker="^")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0., 1.2, 2.4, 3.6])
    ax.set_xlim(-0.1, 3.7)
    ax.set_yticks([0., 0.25, 0.5, 0.75, 1.])
    ax.set_xlabel("1/N")
    ax.set_ylabel("Ratios")

    for spine in ax.spines.values():
        spine.set_linewidth(2.)
    ax.tick_params(width=2.)
    #plt.xscale('log')
    ax.legend()
    return ax


def plot_w1_comparison(axs:np.ndarray, cbar_ax, generations = np.array([2, 3, 4])):
    axs = np.array(axs)
    if axs.shape == (2, len(generations)):
        pass
    else:
        try: 
            axs = axs.reshape(2, len(generations))
        except:
            raise ValueError(f"axs shape must be (2, {len(generations)}). It is {axs.shape}.")

    fig = axs[0, 0].figure

    values = []
    for g in generations:
        ns, vals = get_values_from_file(g)
        values.append(vals)

    new_values = []
    n_diff_vals = np.sum([v.shape[1] - 2 for v in values]) + 2
    counter = 0
    fill_value = np.nan

    for v in values:
        coords = v[:, :2]

        if counter == 0:
            # columns: coords | vals | fill
            fill = np.full((v.shape[0], n_diff_vals - v.shape[1]), fill_value)
            stack = np.column_stack((coords, v[:, 2:], fill))
        else:
            # columns coords | fill | vals | fill
            fill1 = np.full((v.shape[0], counter), fill_value)
            fill2 = np.full((v.shape[0], n_diff_vals - v.shape[1] - counter), fill_value)
            stack = np.column_stack((coords, fill1, v[:, 2:], fill2))
        counter += v.shape[1] - 2
        new_values.append(stack)
    
    unique = get_unique_coords_from_grids(new_values)
    bis = unique[:, np.arange(2, unique.shape[1], 2)]
    disorders = unique[:, np.arange(3, unique.shape[1], 2)]

    vmin, vmax = -1.0, 0.0
    row_label = ['i', 'ii']
    col_label = 'abc'

    for i in range(2):
        for j, g in enumerate(generations):
            arr = bis[:, j] if i == 0 else disorders[:, j]
            arr[np.isnan(arr)] = 0.0
            #plot = axs[i, j].scatter(phi, M, c=arr, vmin=vmin, vmax=vmax)
            Z = arr.reshape(25, 25).T
            plot = axs[i, j].imshow(Z, vmin=vmin, vmax=vmax, cmap='Greys_r', origin='lower', aspect='auto',
                                    extent=(0., np.pi, 0., 3 * np.sqrt(3)))

            axs[i, j].set_xticks([0., np.pi/2, np.pi])
            axs[i, j].set_yticks([0., 3 * np.sqrt(3)])
            axs[i, j].set_xticklabels(["0", "$\\pi / 2$", "$\\pi$"], fontsize=16)
            axs[i, j].set_yticklabels(["0", "$3\\sqrt{3}$"], fontsize=16)

            axs[-1, j].set_xlabel("$\\phi$", fontsize=16)
            axs[i, 0].set_ylabel("$M$", fontsize=16, rotation=0)
            axs[i, j].annotate(
                f"({col_label[j]}.{row_label[i]})",
                xy=(0.05, 0.95), xycoords="axes fraction",
                ha="left", va="top",
                fontsize=24
            )

            for spine in axs[i, j].spines.values():
                spine.set_linewidth(3.0)

            axs[i, j].tick_params(width=3.0, length=5.0)

    cbar = fig.colorbar(plot, cax=cbar_ax)
    cbar.set_ticks([-1., -0.5, 0.])
    cbar.ax.tick_params(length=5.0, width=3.0)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(3.)

    return axs   


def plot_full_w1_comp(generations = np.array([2, 3, 4])):

    fig = plt.figure(figsize=(12, 8))

    if 0:
        gs = gridspec.GridSpec(2, 4, width_ratios=(1,1,1,.5), height_ratios=(1,1))
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 3, gs[:, :-1])
        ax1 = fig.add_subplot(gs0[0, 0])
        ax2 = fig.add_subplot(gs0[0, 1])
        ax3 = fig.add_subplot(gs0[0, 2])
        ax4 = fig.add_subplot(gs0[1, 0])
        ax5 = fig.add_subplot(gs0[1, 1])
        ax6 = fig.add_subplot(gs0[1, 2])
        imshow_axs = np.array([ax1, ax2, ax3, ax4, ax5, ax6]).reshape(2, 3)
        ax7 = fig.add_subplot(gs[:, -1])
    else:
        gs = gridspec.GridSpec(3, 4, width_ratios=(1, 1, 1, .1), height_ratios=(1, 1, .5), hspace=0.4, wspace=0.1)
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 3, gs[:-1, :-1], hspace=0.1, wspace=0.1)
        ax1 = fig.add_subplot(gs0[0, 0])
        ax2 = fig.add_subplot(gs0[0, 1])
        ax3 = fig.add_subplot(gs0[0, 2])
        ax4 = fig.add_subplot(gs0[1, 0])
        ax5 = fig.add_subplot(gs0[1, 1])
        ax6 = fig.add_subplot(gs0[1, 2])
        imshow_axs = np.array([ax1, ax2, ax3, ax4, ax5, ax6]).reshape(2, 3)
        cbar_ax = fig.add_subplot(gs[:-1, -1])

        ax7 = fig.add_subplot(gs[2, :])

    plot_percentages(generations=generations, ax=ax7)
    plot_w1_comparison(axs=imshow_axs, cbar_ax=cbar_ax, generations=generations)

    for ax in imshow_axs.flatten():
        ax.set_aspect('auto')
    for ax in imshow_axs[:, 1:].flatten():
        ax.set_yticklabels([])
    for ax in imshow_axs[0, :].flatten():
        ax.set_xticklabels([])
    plt.savefig("./Hexaflake/figures/generation_comparison.svg")


def f_asymptotic_exp(x: np.ndarray, a: float, b: float):
    """Returns `f(x) = a - (a + 1) * np.exp(-b * x)`, which is an asymptotic exponential function that approaches `a` as `x` increases."""
    return a - (a + 1) * np.exp(-b * x)


def plot_iterations():
    with h5py.File("./Hexaflake/Data/site_elim_g4_w1.0_i50.h5", 'r') as f:
        phi = cast(np.ndarray, f["phi"])[()]
        M = cast(np.ndarray, f["M"])[()]
        disorder = cast(np.ndarray, f["disorder"])[()]
        disorder_all = cast(np.ndarray, f["disorder_all"])[()]

    # Array to hold the mean disorder values for the first i iterations
    # Row index is the specific parameter set (phi, M), column index is the number of iterations considered
    arr = []
    for i in range(disorder_all.shape[1]):
        arr.append(np.nanmean(disorder_all[:, :i + 1], axis=1))
    arr = np.array(arr).T

    t = np.arange(arr.shape[1])
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Axs[0]: disorder converges to near topological (-1.0)
    # Axs[1]: otherwise

    # Remove parameter sets that have any values > 0.0 
    valid_idxs = np.argwhere(np.all(disorder_all <= 0.1, axis=1)).flatten()
    arr = arr[valid_idxs, :]
    disorder_all = disorder_all[valid_idxs, :]

    convergent_values = arr[:, -1]
    sort_idxs = np.argsort(convergent_values)
    arr = arr[sort_idxs, :]
    disorder_all = disorder_all[sort_idxs, :]

    convergent_values = convergent_values[sort_idxs]
    topological_idxs = np.argwhere(np.isclose(convergent_values, -1.0, atol=1e-2)).flatten()
    other_idxs = np.argwhere(~np.isclose(convergent_values, -1.0, atol=1e-2)).flatten()
  
    starting_n_iterations = 1 - 1
    for i, idx_arr in enumerate([topological_idxs, other_idxs]):
        colors = plt.cm.jet(np.linspace(0, 1, idx_arr.size))
        for j in range(idx_arr.size):
            y = arr[idx_arr[j], :]
            axs[i].plot(t[starting_n_iterations:], y[starting_n_iterations:], alpha=0.5, color=colors[j], marker='.', label=str(j))
    
        axs[i].set_xlim(starting_n_iterations, arr.shape[1])
    plt.ticklabel_format(axis='y', useOffset=False)

    axs[0].set_title("Convergent to Topological (-1.0)")
    axs[1].set_title("Convergent to Other Values")
    plt.savefig("./Hexaflake/Figures/g4_iterations.svg")
    plt.show()


def make_haldane_ldos(generation):

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, layout='constrained')

    # (a) two closest to zero energy modes on honeycomb w OBC
    # (b) two closest to zero energy modes on corresponding hexaflake (renorm) w OBC
    # (c) two closest to zero energy modes on hexaflake (site elim) w OBC
    # (d) two closest to zero energy modes on hexaflake (site elim) w PBC

    g_data_obc = compute_geometric_data(generation, False)
    g_data_pbc = compute_geometric_data(generation, True)

    H_a = compute_hamiltonian("hexagon", 3 * np.sqrt(3) / 2, np.pi / 2, 1., 1., g_data_obc)
    H_b = compute_hamiltonian("renorm1", 3 * np.sqrt(3) / 2, np.pi / 2, 1., 1., g_data_obc)
    H_c = compute_hamiltonian("site_elim", 3 * np.sqrt(3) / 2, np.pi / 2, 1., 1., g_data_obc)
    H_d = compute_hamiltonian("site_elim", 3 * np.sqrt(3) / 2, np.pi / 2, 1., 1., g_data_pbc)

    def get_closest_to_zero_eigvectors(H, n=2):
        eigvals, eigvecs = np.linalg.eigh(H)
        idxs = np.argsort(np.abs(eigvals))
        evs = eigvecs[:, idxs[:n]]
        ldos = np.sum(np.abs(evs) ** 2, axis=1)
        return eigvals[idxs[:n]], ldos

    eigvals_a, eigvecs_a = get_closest_to_zero_eigvectors(H_a)
    eigvals_b, eigvecs_b = get_closest_to_zero_eigvectors(H_b)
    eigvals_c, eigvecs_c = get_closest_to_zero_eigvectors(H_c)
    eigvals_d, eigvecs_d = get_closest_to_zero_eigvectors(H_d)

    eigvals = [eigvals_a, eigvals_b, eigvals_c, eigvals_d]
    for lab, eigvals in zip(["(a)", "(b)", "(c)", "(d)"], eigvals):
        print(f"{lab} Eigenvalues closest to zero: {eigvals}")

    x, y = g_data_obc["x"], g_data_obc["y"]
    x -= np.min(x)
    y -= np.min(y)
    hexaflake_mask = g_data_obc["hexaflake"]

    ldoses = [eigvecs_a, eigvecs_b, eigvecs_c, eigvecs_d]
    ldoses = [ld - np.min(ld) for ld in ldoses]
    ldoses = [ld / np.max(ld) for ld in ldoses]

    xticks = [0, np.max(x)]
    yticks = [0, np.max(y)]

    for i, (ax, ldos) in enumerate(zip(axs.flatten(), ldoses)):
        if i == 0:
            #plot = ax.scatter(x, y, c=ldos, cmap='inferno', s=50, vmin=0., vmax=1.)
            plot = ax.scatter(x, y, c=ldos, cmap='Greys', vmin=0., vmax=1., s=2.25)
        else:
            plot = ax.scatter(x[hexaflake_mask], y[hexaflake_mask], c=ldos, cmap='Greys', vmin=0., vmax=1., s=2.25)
        ax.set_aspect('equal')
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([str(int(t + 1)) for t in xticks])
        ax.set_yticklabels([str(int(t + 1)) for t in yticks])
        cbar = plt.colorbar(plot, ax=ax)
        
    for ax in fig.axes:
        ax.tick_params(width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.savefig("./Figures/haldane_ldos.svg", bbox_inches='tight', transparent=False)
    #plt.show()


if __name__ == "__main__":
    #plot_full_w1_comp()

    width_mm = 14.429 * 1.2
    height_mm = 726.798 * 1.3

    fig, ax = plt.subplots(1, 1, figsize=(lambda x, y: (x / 25.4, y / 25.4))(width_mm, height_mm))

    plot = plt.scatter([0, 0], [0, 1], c=[0., -1.], vmin=0., vmax=1., cmap='jet')
    cbar = fig.colorbar(plot, cax=ax)

    plt.savefig('./Hexaflake/Figures/cbar.svg')
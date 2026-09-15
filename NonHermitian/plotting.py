from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar

from nonhermitian_defects import (
    DefectLattice,
    compute_eigenvectors_eigenvalues,
    get_close_to_zero_idxs,
)

DEFAULT_DATA_SAVE_DIRECTORY = Path("./data/")
DEFAULT_FIGURE_SAVE_DIRECTORY = Path("./figures/")
DEFAULT_N_SAVE = 16  # Buffer size of near-zero modes saved to disk


# region Saving and Reading Data
def prepare_eig_dict_for_saving(
    eig_dict: dict[str, Any], n_save: int = DEFAULT_N_SAVE
) -> dict[str, Any]:
    """Stores global spectrum index mapping and trims eigenvectors to n_save buffer size."""
    if "left_eigenvectors" in eig_dict:
        evecs = eig_dict["left_eigenvectors"]
        if evecs.ndim == 2 and evecs.shape[1] > n_save:
            saved_idxs = get_close_to_zero_idxs(evecs, n_save)
            eig_dict["saved_idxs"] = saved_idxs
            eig_dict["left_eigenvectors"] = evecs[:, saved_idxs]
            if "right_eigenvectors" in eig_dict and eig_dict["right_eigenvectors"].ndim == 2:
                eig_dict["right_eigenvectors"] = eig_dict["right_eigenvectors"][:, saved_idxs]
        elif "saved_idxs" not in eig_dict:
            eig_dict["saved_idxs"] = np.arange(evecs.shape[1])
    return eig_dict


def get_selected_modes(
    eig_dict: dict[str, Any], n_idxs: int, lattice: DefectLattice
) -> dict[str, Any]:
    """Dynamically extracts n_idxs modes from the saved buffer, computes LDOS, and maps global spectrum indices."""

    def _ensure_eigenvector_shape_for_schottky(eigenvector: np.ndarray) -> np.ndarray:
        """Pads eigenvector to account for missing site indices in Schottky defects."""
        mask = np.full(eigenvector.shape[0] + len(lattice.defect_indices), True)
        for i, idx in enumerate(lattice.defect_indices):
            mask[2 * idx + i % 2] = False
        resized_eigenvector = np.zeros(mask.shape, dtype=eigenvector.dtype)
        resized_eigenvector[mask] = eigenvector
        return resized_eigenvector

    left_eigenvectors = eig_dict["left_eigenvectors"]
    num_available_modes = left_eigenvectors.shape[1]

    if n_idxs > num_available_modes:
        raise ValueError(
            f"Requested {n_idxs} modes, but only {num_available_modes} modes were saved in data. "
            "Recompute with a larger n_save buffer."
        )

    # Determine local column indices relative to saved buffer
    if num_available_modes == n_idxs:
        local_idxs = np.arange(n_idxs)
    else:
        local_idxs = get_close_to_zero_idxs(left_eigenvectors, n_idxs)

    selected = left_eigenvectors[:, local_idxs]

    # Map local column indices back to global 1D spectrum indices
    if "saved_idxs" in eig_dict:
        global_idxs = np.asarray(eig_dict["saved_idxs"])[local_idxs]
    elif "selected_idxs" in eig_dict and len(eig_dict["selected_idxs"]) == num_available_modes:
        global_idxs = np.asarray(eig_dict["selected_idxs"])[local_idxs]
    else:
        global_idxs = local_idxs

    ldos = np.sum(np.abs(selected) ** 2, axis=1)

    if lattice.defect_type == "schottky":
        ldos = _ensure_eigenvector_shape_for_schottky(ldos)

    # Sum local density of states across internal degrees of freedom per lattice site
    selected_left_eigenvectors = ldos[::2] + ldos[1::2]
    eig_dict["selected_left_eigenvectors"] = selected_left_eigenvectors
    eig_dict["selected_idxs"] = global_idxs

    return eig_dict


def save_ipr_data(
    Ls: list[int],
    eig_dicts: list[dict[str, Any]],
    fname: str,
    directory: Path | str = DEFAULT_DATA_SAVE_DIRECTORY,
) -> None:
    """Saves full spectrum/IPR arrays and the buffered eigenvector slice to HDF5."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / fname

    KEYS_TO_SAVE = {
        "eigenvalues",
        "L",
        "R",
        "left_ipr",
        "right_ipr",
        "saved_idxs",
        "selected_idxs",
        "selected_left_eigenvectors",
        "selected_right_eigenvectors",
        "left_eigenvectors",
        "right_eigenvectors",
    }

    with h5py.File(filepath, "w") as f:
        f.create_dataset(name="Ls", data=Ls)
        for L, ed in zip(Ls, eig_dicts):
            for k, v in ed.items():
                if k in KEYS_TO_SAVE:
                    f.create_dataset(name=f"L={L}_{k}", data=v)


def read_ipr_data(
    fname: str, directory: Path | str = DEFAULT_DATA_SAVE_DIRECTORY
) -> tuple[list[int], list[dict[str, Any]]]:
    """Reads inverse participation ratio (IPR) dataset from an HDF5 file."""
    filepath = Path(directory) / fname
    data: dict[str, dict[str, Any]] = {}

    with h5py.File(filepath, "r") as f:
        Ls = cast(list[int], list(f["Ls"]))  # type: ignore
        for k, v in f.items():
            if k == "Ls":
                continue
            grouping, prop_name = k.split("_", 1)
            if grouping not in data:
                data[grouping] = {}
            data[grouping][prop_name] = cast(np.ndarray, v)[()]

    return Ls, list(data.values())


def compute_ipr_data(
    method: str,
    Ls: list[int],
    n_idxs: int,
    hdir: str,
    directory: Path | str = DEFAULT_DATA_SAVE_DIRECTORY,
    n_save: int = DEFAULT_N_SAVE,
    **kwargs: Any,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]], DefectLattice]:
    """Computes or loads IPR data, keeping a buffer of n_save near-zero modes on disk."""
    directory = Path(directory)
    file1 = f"{method}_ipr_data_h{hdir}=0.5.h5"
    file2 = f"{method}_ipr_data_h{hdir}=1.5.h5"

    if method in ["substitution", "interstitial"]:
        n_defects = 1
        if "defect_radius" in kwargs:
            n_defects += int(np.sum(4 * np.arange(kwargs["defect_radius"])))
        if "break_c4" in kwargs:
            n_defects += 2
        file1 = f"{method}_ipr_data_h{hdir}=0.5_nd={n_defects}.h5"
        file2 = f"{method}_ipr_data_h{hdir}=1.5_nd={n_defects}.h5"

    schottky_separations = [L // 4 + (L // 4 + 1) % 2 for L in Ls]
    fpxs = fpys = [-s - 0.5 for s in schottky_separations]

    lattices = [
        DefectLattice(
            L,
            L,
            method,
            True,
            schottky_separation=s,
            frenkel_x_disp=fpx,
            frenkel_y_disp=fpy,
            **kwargs,
        )
        for L, s, fpx, fpy in zip(Ls, schottky_separations, fpxs, fpys)
    ]

    path1, path2 = directory / file1, directory / file2
    if path1.exists() and path2.exists():
        print(f"IPR data files already exist for method {method} and hdir {hdir}.")
        new_Ls1, data1 = read_ipr_data(file1, directory)
        new_Ls2, data2 = read_ipr_data(file2, directory)
        assert new_Ls1 == new_Ls2, "Mismatch in Ls between the two IPR data files."
        data1 = [get_selected_modes(ed, n_idxs, l) for ed, l in zip(data1, lattices)]
        data2 = [get_selected_modes(ed, n_idxs, l) for ed, l in zip(data2, lattices)]
        return Ls, data1, data2, lattices[-1]

    hdir_map = {"x": 0, "y": 1, "z": 2}
    v1_arr, v2_arr = [0.0] * 3, [0.0] * 3
    v1_arr[hdir_map[hdir]] = 0.5
    v2_arr[hdir_map[hdir]] = 1.5

    v1, v2 = np.array(v1_arr), np.array(v2_arr)
    print(f"Starting computation for {method} h{hdir}")

    # Compute and buffer max(n_idxs, n_save) near-zero modes
    calc_modes = max(n_idxs, n_save)

    ed1s = [compute_eigenvectors_eigenvalues(lat, -1.0, v1, v2, calc_modes) for lat in lattices]
    ed1s = [prepare_eig_dict_for_saving(ed, calc_modes) for ed in ed1s]

    ed2s = [compute_eigenvectors_eigenvalues(lat, -1.0, v2, v1, calc_modes) for lat in lattices]
    ed2s = [prepare_eig_dict_for_saving(ed, calc_modes) for ed in ed2s]

    save_ipr_data(Ls, ed1s, file1, directory)
    save_ipr_data(Ls, ed2s, file2, directory)

    # Slice dynamically to requested n_idxs for immediate plot/analysis
    ed1s = [get_selected_modes(ed, n_idxs, lat) for ed, lat in zip(ed1s, lattices)]
    ed2s = [get_selected_modes(ed, n_idxs, lat) for ed, lat in zip(ed2s, lattices)]

    return Ls, ed1s, ed2s, lattices[-1]


# endregion


# region Formatting Helpers
def fexp(number: float) -> int:
    """Returns exponent component of a floating-point number for scientific formatting."""
    _, digits, exponent = Decimal(str(number)).as_tuple()
    return len(digits) + int(exponent) - 1


def fman(number: float) -> Decimal:
    """Returns mantissa component of a floating-point number for scientific formatting."""
    return Decimal(str(number)).scaleb(-fexp(number)).normalize()


def find_ldos_view_area(
    lattice: DefectLattice, ldos_array: np.ndarray
) -> tuple[int, int, int, int]:
    """Calculates spatial bounding box coordinates around high-density defect regions."""
    X_d, Y_d = np.array(lattice.defect_positions)

    x_center = int(np.mean(X_d))
    y_center = int(np.mean(Y_d))
    radius = max(np.max(X_d - x_center), np.max(Y_d - y_center)) + 10
    extent = (x_center - radius, x_center + radius, y_center - radius, y_center + radius)

    threshold = np.mean(ldos_array) + 2 * np.std(ldos_array)
    ldos_mask = ldos_array > threshold

    if np.any(ldos_mask):
        X, Y = lattice.X, lattice.Y
        X_masked, Y_masked = X[ldos_mask], Y[ldos_mask]

        xmin, xmax = int(np.min(X_masked)), int(np.max(X_masked))
        ymin, ymax = int(np.min(Y_masked)), int(np.max(Y_masked))

        d_xmin, d_xmax, d_ymin, d_ymax = extent
        extent = (min(xmin, d_xmin), max(xmax, d_xmax), min(ymin, d_ymin), max(ymax, d_ymax))
        radius = max(extent[1] - extent[0], extent[3] - extent[2]) // 2 + 3
        x_center = (extent[0] + extent[1]) // 2
        y_center = (extent[2] + extent[3]) // 2
        extent = (x_center - radius, x_center + radius, y_center - radius, y_center + radius)

    bounded_extent = (
        max(extent[0], 0),
        min(extent[1], lattice.Lx - 1),
        max(extent[2], 0),
        min(extent[3], lattice.Ly - 1),
    )

    return cast(tuple[int, int, int, int], tuple(np.ravel(bounded_extent).astype(int)))


# endregion


# region Plotting
def format_colorbar(cbar: Colorbar) -> None:
    """Configures colorbar limits, scientific formatting, and boundary thickness."""
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    cbar.formatter = formatter
    cbar.update_ticks()
    cbar.outline.set_linewidth(1.5)


def plot_ldos(
    lattice: DefectLattice,
    ax: Axes,
    color_array: np.ndarray,
    cbar_ax: Axes | None = None,
    cmap: str = "Greys",
    extent: tuple[int, int, int, int] | tuple[()] = (),
    scatter_size: int = 100,
) -> None:
    """Plots spatial Local Density of States (LDOS) scatter map on the lattice grid."""
    if not extent:
        extent = (0, lattice.Lx - 1, 0, lattice.Ly - 1)

    xticks = [extent[0] + 1, extent[1] - 1]
    yticks = [extent[2] + 1, extent[3] - 1]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    X, Y = lattice.X, lattice.Y
    X_d, Y_d = np.array(lattice.defect_positions)

    plot = ax.scatter(X, Y, c=color_array, cmap=cmap, s=scatter_size, zorder=1, rasterized=True)

    if lattice.defect_type == "vacancy":
        ax.scatter(X_d, Y_d, zorder=2, lw=1, s=scatter_size, edgecolor="r", facecolor="none", rasterized=True)
    elif lattice.defect_type == "frenkel_pair":
        ax.scatter(lattice.Lx // 2, lattice.Ly // 2, zorder=2, lw=1, s=scatter_size, edgecolor="r", facecolor="none", rasterized=True)

    if cbar_ax is not None:
        cax_box = cbar_ax.get_position()
        hw_ratio = cax_box.height / cax_box.width
        orientation = "vertical" if hw_ratio >= 1.0 else "horizontal"
        cbar = plt.colorbar(plot, cax=cbar_ax, orientation=orientation)
        if orientation == "horizontal":
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")

        cbar.set_label("Local Density of States")
        format_colorbar(cbar)

        vmin, vmax = plot.get_clim()
        cbar.set_ticks((vmin, vmax))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"${int(t + 1)}$" for t in xticks])
    ax.set_yticklabels([f"${int(t + 1)}$" for t in yticks])
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$Y$")


def plot_spectrum(
    ax: Axes,
    eigenvalues: np.ndarray,
    color_array: np.ndarray | None = None,
    cbar_ax: Axes | None = None,
    selected_idxs: np.ndarray | None = None,
    cmap: str = "jet",
    scatter_size: int = 50,
) -> None:
    """Plots complex energy spectrum with selected zero-mode markers correctly indexed."""
    if color_array is None:
        color_array = np.zeros(eigenvalues.shape)

    if np.issubdtype(eigenvalues.dtype, np.complexfloating):
        x, y = eigenvalues.real, eigenvalues.imag
        ax.set_xlabel(r"$\Re(E)$")
        ax.set_ylabel(r"$\Im(E)$")
    else:
        x, y = np.arange(eigenvalues.size), eigenvalues
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"$E_n$")

    sort_idxs = np.argsort(color_array)
    x = x[sort_idxs]
    y = y[sort_idxs]
    color_array = color_array[sort_idxs]

    mask = np.full(color_array.shape, False)
    if selected_idxs is not None:
        mask[selected_idxs] = True
    mask = mask[sort_idxs]

    plot = ax.scatter(x[~mask], y[~mask], c=color_array[~mask], cmap=cmap, zorder=1, s=scatter_size, rasterized=True)

    vmin, vmax = plot.get_clim()
    ax.scatter(
        x[mask],
        y[mask],
        c=color_array[mask],
        zorder=2,
        marker="*",
        vmin=vmin,
        vmax=vmax,
        s=2 * scatter_size,
        cmap=cmap,
        rasterized=True,
    )

    xticks = np.round(np.linspace(np.min(x), np.max(x), 3), 1)
    yticks = np.round(np.linspace(np.min(y), np.max(y), 3), 1)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.set_yticklabels([str(t) for t in yticks])

    if cbar_ax is not None:
        cbar = plt.colorbar(plot, cax=cbar_ax)
        format_colorbar(cbar)
        cbar.set_ticks((vmin, (vmin + vmax) / 2, vmax))


def plot_ipr(
    ax: Axes,
    eigenvalues: tuple[np.ndarray, ...] | list[np.ndarray],
    iprs: tuple[np.ndarray, ...] | list[np.ndarray],
    Ls: list[int] | tuple[int, ...],
    selected_idxs: np.ndarray | None = None,
    scatter_size: int = 75,
) -> None:
    """Plots Inverse Participation Ratio vs Eigenvalues across system sizes (Ls)."""
    L_sort_idxs = np.argsort(Ls)
    Ls_sorted = [Ls[i] for i in L_sort_idxs]
    eigenvalues_sorted = [eigenvalues[i] for i in L_sort_idxs]
    iprs_sorted = [iprs[i] for i in L_sort_idxs]

    if np.issubdtype(eigenvalues_sorted[0].dtype, np.complexfloating):
        xs = [np.abs(ev) * np.sign(ev.real) for ev in eigenvalues_sorted]
        ax.set_xlabel(r"$|E| \times \mathrm{sign}(\Re E)$")
    else:
        xs = list(eigenvalues_sorted)
        ax.set_xlabel(r"$E$")
    ax.set_ylabel("IPR")

    all_colors = ["tab:purple", "tab:blue", "tab:green", "tab:orange", "tab:pink", "tab:olive", "tab:cyan", "r"]
    colors = [all_colors[i] for i in range(len(iprs_sorted))]
    colors[-1] = "r"

    mask = np.full(xs[-1].size, False)
    if selected_idxs is not None:
        mask[selected_idxs] = True
        ax.scatter(xs[-1][mask], iprs_sorted[-1][mask], c=colors[-1], marker="*", zorder=100, s=scatter_size, rasterized=True)

    ax.scatter(
        xs[-1][~mask],
        iprs_sorted[-1][~mask],
        s=1.5 * scatter_size,
        c=colors[-1],
        alpha=0.25,
        label=f"$L={Ls_sorted[-1]}$",
        zorder=99,
        rasterized=True,
    )

    for i in range(len(iprs_sorted) - 1):
        ax.scatter(
            xs[i],
            iprs_sorted[i],
            s=scatter_size,
            c=colors[i],
            alpha=0.25,
            label=f"$L={Ls_sorted[i]}$",
            zorder=i,
            rasterized=True,
        )

    ax.legend()

    xmax, xmin = max(np.max(x) for x in xs), min(np.min(x) for x in xs)
    ymax, ymin = max(np.max(y) for y in iprs_sorted), min(np.min(y) for y in iprs_sorted)

    xticks = np.linspace(xmin, xmax, 3)
    yticks = np.linspace(ymin, ymax, 3)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{t:.1f}" for t in xticks])
    ax.set_yticklabels([rf"${fman(t):.1f}\times 10^{{{fexp(t):.0f}}}$" for t in yticks])


def figure_layout() -> Figure:
    """Builds multi-panel figure layout using matplotlib GridSpec."""
    fig = plt.figure(figsize=(20, 10))

    # Master GridSpec
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2.0, 1.5, 1.5], hspace=0.3)

    # Sub-GridSpec: IPR subplots
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 0])
    ax1 = fig.add_subplot(gs0[0, 0], label="a")
    fig.add_subplot(gs0[0, 1], label="b", sharey=ax1)

    # Sub-GridSpec: Complex spectra
    gs1_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], width_ratios=[1, 0.1])
    fig.add_subplot(gs1_a[0, 0], label="c")
    fig.add_subplot(gs1_a[0, 1], label="c_cb")

    gs1_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], width_ratios=[1, 0.1])
    fig.add_subplot(gs1_b[0, 0], label="d")
    fig.add_subplot(gs1_b[0, 1], label="d_cb")

    # Sub-GridSpec: Spatial LDOS plots
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, 1:], width_ratios=[1, 1, 1], height_ratios=[0.1, 1.0])
    fig.add_subplot(gs2[1, 0], label="e")
    fig.add_subplot(gs2[0, 0], label="e_cb")
    fig.add_subplot(gs2[1, 1], label="f")
    fig.add_subplot(gs2[1, 2], label="g")
    fig.add_subplot(gs2[0, 1], label="f_cb")
    fig.add_subplot(gs2[0, 2], label="g_cb")

    for ax in fig.axes:
        ax.tick_params(width=1.5, length=5.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    return fig


def plot_ipr_figure(
    method: str,
    Ls: list[int] | int,
    n_idxs: int,
    hdir: str,
    save_directory: Path | str = DEFAULT_FIGURE_SAVE_DIRECTORY,
    **kwargs: Any,
) -> None:
    """Executes calculations, populates figure layout, and saves output SVG plot."""
    save_directory = Path(save_directory)
    fig = figure_layout()
    ax_a, ax_b, ax_c, ax_c_cb, ax_d, ax_d_cb, ax_e, ax_e_cb, ax_f, ax_g, ax_f_cb, ax_g_cb = fig.axes

    Ls_list = [Ls] if isinstance(Ls, int) else list(Ls)

    # Compute or read IPR data
    Ls_list, ed1s, ed2s, largest_lattice = compute_ipr_data(method, Ls_list, n_idxs, hdir, **kwargs)

    # Plot IPR curves
    for i, (eds, ipr_ax) in enumerate(zip([ed1s, ed2s], [ax_a, ax_b])):
        evs = [ed["eigenvalues"] for ed in eds]
        iprs = [ed["left_ipr"] for ed in eds]
        plot_ipr(ipr_ax, evs, iprs, Ls_list, eds[-1]["selected_idxs"] if i == 0 else None)

    # Calculate unified Y-axis limits for IPR subplots
    ipr_max = max(max(np.max(ed["left_ipr"]) for ed in ed1s), max(np.max(ed["left_ipr"]) for ed in ed2s))
    ipr_min = min(min(np.min(ed["left_ipr"]) for ed in ed1s), min(np.min(ed["left_ipr"]) for ed in ed2s))
    ipr_ticks = np.linspace(ipr_min, ipr_max, 3)

    for ax in [ax_a, ax_b]:
        ax.set_yticks(ipr_ticks)

    ax_b.set_yticklabels([])
    ax_a.set_yticklabels([rf"${fman(t):.1f}\times 10^{{{fexp(t):.0f}}}$" for t in ipr_ticks])

    # Plot complex spectrums
    for i, (eds, spectrum_ax, cb_ax) in enumerate(zip([ed1s, ed2s], [ax_c, ax_d], [ax_c_cb, ax_d_cb])):
        eigenvalues = eds[-1]["eigenvalues"]
        ipr = eds[-1]["left_ipr"]
        idxs = eds[-1]["selected_idxs"]
        plot_spectrum(spectrum_ax, eigenvalues, ipr, cb_ax, idxs if i == 0 else None, cmap="jet")

    # Extract spatial LDOS from selected modes
    topo_ldos1 = ed1s[-1]["selected_left_eigenvectors"]
    topo_ldos2 = ed2s[-1]["selected_left_eigenvectors"]

    if method == "frenkel_pair":
        r = 5
        extent1 = (11.5 - r, 11.5 + r, 11.5 - r, 11.5 + r)
        extent2 = (11.5 - r, Ls_list[-1] // 2 + r, 11.5 - r, Ls_list[-1] // 2 + r)
    else:
        if method in ["vacancy", "substitution"]:
            r = 10
            extent = (Ls_list[-1] // 2 - r, Ls_list[-1] // 2 + r, Ls_list[-1] // 2 - r, Ls_list[-1] // 2 + r)
        elif method == "interstitial":
            r = 5
            extent = (Ls_list[-1] // 2 - r - 0.5, Ls_list[-1] // 2 + r, Ls_list[-1] // 2 - r - 0.5, Ls_list[-1] // 2 + r)
        elif method == "schottky":
            r = 11
            extent = (Ls_list[-1] // 2 - r, Ls_list[-1] // 2 + r, Ls_list[-1] // 2 - r, Ls_list[-1] // 2 + r)
        else:
            extent = (0, largest_lattice.Lx - 1, 0, largest_lattice.Ly - 1)
        extent1 = extent2 = extent

    plot_ldos(largest_lattice, ax_e, topo_ldos1, ax_e_cb, extent=extent1, scatter_size=50)
    plot_ldos(largest_lattice, ax_f, topo_ldos1, ax_f_cb, extent=extent2, scatter_size=50)
    plot_ldos(largest_lattice, ax_g, topo_ldos2, ax_g_cb, extent=extent2, scatter_size=50)

    # Clean up redundant axis labels and legend artifacts
    ax_b.set_ylabel("")
    if ax_b.get_legend():
        ax_b.get_legend().remove()
    ax_d.set_ylabel("")
    ax_g.set_ylabel("")

    for ax in [ax_e, ax_f, ax_g]:
        ax.set_aspect("equal")

    save_directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_directory / f"{method}_ipr_h{hdir}.svg", dpi=96)


# endregion


def main():
    pass


if __name__ == "__main__":
    main()
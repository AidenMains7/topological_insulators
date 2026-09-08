import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

from project_tools import model, lattice
from compute_ltm_sponge import plot_3d_voxels

rcParams['axes.linewidth'] = 2.0
rcParams['xtick.major.width'] = 2.0
rcParams['ytick.major.width'] = 2.0


def compute_ldos(eigenvalues, ldos, tol=1e-10):
    abs_energies = np.abs(eigenvalues)
    min_E = np.min(abs_energies)

    # 2. Degeneracy failsafe: Capture ALL states within numerical tolerance 
    # of the closest energy level (e.g., multiple edge modes or flat bands).
    idxs = np.where(abs_energies <= min_E + tol)[0]

    # Fallback: If the gap is clean, non-degenerate, and shifted, ensure 
    # we still capture at least the two closest states (HOMO/LUMO equivalent).
    if len(idxs) < 2 and len(eigenvalues) >= 2:
        idxs = np.argsort(abs_energies)[:2]

    # Slice COLUMNS (eigenstates), and sum across the entire degenerate subspace
    ldos = np.sum(np.abs(ldos[idxs, ...]) ** 2, axis=0)

    return ldos, idxs


def compute(d, n, M, M_alt, b=1, pasted=False, pbc=False):
    if d == 1:
        fractal = 'cantor'
    elif d == 2:
        fractal = 'carpet'
    elif d == 3:
        fractal = 'sponge'
    m = model.build_model(fractal, n=n, block_scale=b, pasted=pasted, pbc=pbc)

    params = {"M": M, "M_alt": M_alt, "disorder_seed": 0, "disorder_strength": 0.0, "t": 1.0, "B": 1.0}
    if d == 3:
        params["gauge"] = "N"
        params["g"] = 0
        params["M_prime"] = 0.0
    res = m.solve(**params, return_LDOS=True)

    l = lattice.build_lattice(fractal, n, block_scale=b, pasted=pasted)

    eigenvalues, ldos = res["eigenvalues"], res["LDOS"]
    ldos, idxs = compute_ldos(eigenvalues, ldos)

    return l, eigenvalues, ldos, idxs


def plot_spectrum(ax, eigenvalues, idxs):
    ax.scatter(np.arange(eigenvalues.size), eigenvalues, zorder=4)
    ax.scatter(np.arange(eigenvalues.size)[idxs], eigenvalues[idxs], zorder=5, c='r')


def get_ndim_diagonal_indices(arr):
    smallest_dim = np.min(arr.shape)

    mask = np.full(arr.shape, False)
    for i in range(smallest_dim):
        mask[*[v for v in [i]*arr.ndim]] = True
    return mask


def plot_ldos(ax, D, l, ldos, cmap='managua', do_imshow=False):

    if not do_imshow or D == 1:
        mask = get_ndim_diagonal_indices(l)
        ax.scatter(np.arange(np.sum(mask)), ldos[mask], s=50)
        return

    if D == 2 and do_imshow:
        X, Y = np.where(l > -100)[:]
        mappable = ax.imshow(ldos, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
        ax.set_xticks([X.min(), X.max()])
        ax.set_xticklabels([X.min() + 1, X.max() + 1], fontsize=16)
        ax.set_yticks([Y.min(), Y.max()])
        ax.set_yticklabels([Y.min() + 1, Y.max() + 1], fontsize=16)

    if D == 3 and do_imshow:
        mappable = plot_3d_voxels(l > -100, ldos, cmap, ax=ax, edgecolors='k', plot_diagonal_half=False, alpha=1.0)

    #if do_imshow: plt.colorbar(mappable, ax=ax)
    return


def make_figure(D, spectrum_plot=False, do_imshow=False):
    if do_imshow and D == 3:
        fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    else:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    cantor_params = {"n": 4, "b": 5, "pasted": False, "m_topological": 2.0, "m_trivial": -0.5}
    carpet_params = {"n": 3, "b": 1, "pasted": False, "m_topological": 2.0, "m_trivial": -0.5}
    sponge_params = {"n": 1, "b": 2, "pasted": False, "m_topological": 2.0, "m_trivial": -0.5}
    n, b, pasted, m_topological, m_trivial = [cantor_params, carpet_params, sponge_params][D - 1].values()

    bcs = np.array([[np.repeat(False, D), np.repeat(True, D)],
                    [np.repeat(False, D), np.repeat(True, D)]])
    
    m0s = np.array([[m_topological, m_topological],
                    [-0.1, -0.1]])
    
    m_alts = np.array([[m_trivial, m_trivial],
                       [-0.5, -0.5]])

    for i in range(2):
        for j in range(2):
            l, eigenvalues, ldos, idxs = compute(D, n, m0s[i, j], m_alts[i, j], b, pasted, bcs[i, j])
            if spectrum_plot:
                plot_spectrum(axs[i, j], eigenvalues, idxs)
            else:
                plot_ldos(axs[i, j], D, l, ldos, do_imshow=do_imshow)
            axs[i, j].annotate(
                xy=(0.05, 0.95),
                ha='left',
                va='top',
                text=f"{eigenvalues[idxs]}",
                xycoords='axes fraction',
            )
            bc_label = "OBC" if bcs[i, j, 0] == False else "PBC"
            axs[i, j].set_title(f"{bc_label} $M_0={m0s[i, j]}$ and $M_{{\\rm alt}}={m_alts[i, j]}$")
            



make_figure(2, False, True)
plt.savefig('carpet.png')
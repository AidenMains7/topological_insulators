import numpy as np
from matplotlib import pyplot as plt
from real_space import fractal_wrapper
import cProfile
import pstats




def highlight_NNN(generation):
    lattice_hopping_dict = fractal_wrapper(generation, False)
    fractal_lattice = lattice_hopping_dict["fractal_lattice"]
    b1, b2, b2_tilde, c1, c2, c3 = lattice_hopping_dict["fractal_hopping_masks"].values()

    y, x = np.where(fractal_lattice >= 0)[:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    arrs = [c1, c2, c3]
    colors = ['b', 'orange','r']
    linestyles = ['--','-.', ':']
    for i, arr in enumerate(arrs):
        i_idx, j_idx = np.where(arr)
        valid_indices = (i_idx < len(x)) & (j_idx < len(x))  # Ensure indices are valid
        i_idx, j_idx = i_idx[valid_indices], j_idx[valid_indices]
        ax.plot([x[i_idx], x[j_idx]], [y[i_idx], y[j_idx]], c = colors[i], ls=linestyles[i], zorder=2)

    for arr in [b1, b2, b2_tilde]:
        i_idx, j_idx = np.where(arr)
        ax.plot([x[i_idx], x[j_idx]], [y[i_idx], y[j_idx]], c = 'k', zorder=1)


    if False:
        yidx, xidx = np.argwhere(fractal_lattice >= 0).T
        for yidx, xidx in zip(yidx, xidx):
            ax.text(xidx, yidx, str(fractal_lattice[yidx, xidx]), fontsize=12, ha='center', va='top', c='r')

    ax.scatter(x, y, c='k', zorder=0, s=4)
    #plt.show()


    



if __name__ == "__main__":
    with cProfile.Profile() as pr:
        generation = 6
        lattice_hopping_dict = fractal_wrapper(generation, False)
        fractal_lattice = lattice_hopping_dict["fractal_lattice"]
        b1, b2, b2_tilde, c1, c2, c3 = lattice_hopping_dict["fractal_hopping_masks"].values()

    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime').print_stats(10)

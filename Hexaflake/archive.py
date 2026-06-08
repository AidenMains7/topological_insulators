



def get_info(generation):
    geometry_data = compute_geometric_data(generation, True)
    x = geometry_data['x']
    hex = geometry_data['hexaflake']
    print("Pristine  N sites:", x.size)
    print("Hexaflake N sites:", np.sum(hex))


def gen4_points():
    with h5py.File("./Hexaflake/Data/site_elim_g4_(25_by_25).h5", 'r') as f:
        phi_vals = f['phi'][:] # type: ignore
        M_vals = f['M'][:] # type: ignore
        bott_index_vals = f['bott_index'][:] # type: ignore

    idxs = [(0, 4), (0, 5), (1, 6), (1, 7), (2, 9), (2, 10), (3, 11), (3, 12), (4, 13), 
         (5, 15), (6, 16), (7, 17), (8, 18), (6, 13), (7, 14), (8, 14), (9, 14),
         (6, 14), (7, 15), (8, 16), (9, 17), (10, 17)]

    phi_unique = np.unique(phi_vals) # type: ignore
    M_unique = np.unique(M_vals) # type: ignore

    parameters = []
    for i, j in idxs:
        parameters.append((phi_unique[i], M_unique[j]))

    compute_phase('site_elim', 4, dimensions=(25, 25), directory="./Hexaflake/Data/", 
               M_values = M_unique, phi_values = phi_unique, n_jobs=-4,
               outfname = 'site_elim_g4_selected_points.h5')
    

def gen4_disorder_selected_points():
    with h5py.File('./site_elim_g4_selected_points.h5', 'r') as f:
        M = f['M'][:] # type: ignore
        bott = f["bott_index"][:] # type: ignore
        phi = f["phi"][:] # type: ignore


    right_half_idxs = np.argwhere(phi >= np.pi/2)[:, 0] # type: ignore
    phi, M, bott = phi[right_half_idxs], M[right_half_idxs], bott[right_half_idxs] # type: ignore

    nontrivial_idxs = np.argwhere(np.round(bott, 3) < 0)[:, 0] # type: ignore

    phi, M, bott = phi[nontrivial_idxs], M[nontrivial_idxs], np.round(bott[nontrivial_idxs]) # type: ignore

    outf = compute_disorder('g4_disorder.h5', 'site_elim', 2, 1.0, 15, 1.0, 1.0, -4, True, False, phi, M, bott) # type: ignore


def pristine_comparison(generation = 2):
    files = ["./Hexaflake/Data/phase_data_hexagon_gen2.npz", './Hexaflake/Data/phase_data_renorm_gen2.npz', './Hexaflake/Data/phase_data_site_elim_gen2.npz']
    fig, axs = plt.subplots(1, len(files), figsize=(15, 5), sharey=True, sharex=True)
    titles = ['Honeycomb', 'Renormalization', 'Site Elimination']



    for i, file in enumerate(files):
        data = np.load(file)
        phi_values, M_values, bott_values = data['phi_range'], data['M_range'], data['bott_index_array'].T
        bott_values = np.round(bott_values).astype(int)

        unique_values = np.unique(bott_values[~np.isnan(bott_values)])
        base_cmap = plt.get_cmap('viridis')
        discrete_cmap = ListedColormap(base_cmap(np.linspace(0, 1, len(unique_values))))
        boundaries = np.concatenate((
            [unique_values[0] - 0.5],
            (unique_values[:-1] + unique_values[1:]) / 2,
            [unique_values[-1] + 0.5]
        ))
        norm = BoundaryNorm(boundaries, discrete_cmap.N)

        im = axs[i].imshow(
            bott_values.T,
            extent=[phi_values[0], phi_values[-1], M_values[0], M_values[-1]],
            cmap=discrete_cmap,
            norm=norm,
            aspect='auto'
        )
        axs[i].set_title(titles[i], fontsize=16)
        axs[i].set_xlabel("$\\phi / \\pi$", fontsize=16)
        axs[i].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs[i].set_yticks([-3 * np.sqrt(3), 0., 3 * np.sqrt(3)])
        axs[i].tick_params(axis='both', which='both', width=1.5, length=6)
        for spine in axs[i].spines.values():
            spine.set_linewidth(2)
        
    axs[0].set_ylabel("M", fontsize=16, rotation=0)

    cbar = fig.colorbar(im, ax=axs[-1])
    cbar.set_ticks(unique_values)
    cbar.set_label("Bott Index", fontsize=16)
    cbar.ax.tick_params(which='both', width=2, length=6)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    
    plt.savefig('./Hexaflake/Figures/Pristine_Comparison.png', bbox_inches='tight', transparent=False)
    plt.savefig('./Hexaflake/Figures/Pristine_Comparison.svg', bbox_inches='tight')


def curve_fit_percentages():
    y = np.array([0.754929, .504225, 0.32676], dtype=float)
    x = 1 / (np.power(7, [2, 3, 4]) * 6)

    def model(x, a, b, n):
        return a + b * np.exp(n * np.log(x))

    from scipy.optimize import curve_fit
    params, cov = curve_fit(model, x, y, p0=(0.0, 0.2, 1.0))
    a, b, n = params
    print(f"a = {a:.6g}, b = {b:.6g}, n = {n:.6g}")
    print("fit values:", model(x, a, b, n))
    
    plt.scatter(x, y, label='Data', color='red')
    x_fit = np.linspace(min(x), max(x), 100)
    plt.plot(x_fit, model(x_fit, a, b, n), label='Fit', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1/(6 * 7^g)')
    plt.ylabel('Percentage of Nontrivial Points')
    plt.title('Curve Fit of Nontrivial Point Percentages')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


def temp_fit():

    y = [0.754929,.504225,0.32676]
    x = 1 / (np.power(7, [2, 3, 4]) * 6)

    a_values = np.linspace(0.0, np.max(y), 101)
    n_values = np.linspace(0.0, 1.0, 101)

    a_grid, n_grid = np.meshgrid(a_values, n_values)
    def fit_func(x, y, a, n):
        return np.log(y - a) - n * np.log(x)
    

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    grids = [fit_func(xi, yi, a_grid, n_grid) for xi, yi in zip(x, y)]

    sum = np.sum(grids, axis=0)
    plt.imshow(np.abs(sum), extent=[0, 1, 0, 1], aspect='auto') # type: ignore
    plt.show()	


if __name__ == "__main__":
    f = f'./Hexaflake/Data/site_elim_g4_(25_by_25)_w1.0.h5'
    with h5py.File(f, 'r') as file:
        M = file['M'][:] # type: ignore
        phi = file['phi'][:] # type: ignore
        disorder_alls = file['disorder_all'][:] # type: ignore
    vals = np.nanmean(disorder_alls, axis=1) # type: ignore
    phi_unique = np.sort(np.unique(phi)) # type: ignore
    M_unique = np.sort(np.unique(M)) # type: ignore

    grid = np.zeros((M_unique.size, phi_unique.size), dtype=float)
    phi_idx = {v: i for i, v in enumerate(phi_unique)}
    M_idx = {v: i for i, v in enumerate(M_unique)}
    for p, m, v in zip(phi, M, vals): # type: ignore
        grid[M_idx[m], phi_idx[p]] = v

    grid = np.hstack([np.fliplr(grid), grid])

    plt.imshow(
        grid,
        origin='lower',
        aspect='auto',
        cmap='jet',
        extent=[0., np.pi, 0., np.unique(M).max()] # type: ignore
    )
    plt.xlabel('$\\phi$')
    plt.ylabel('M')
    plt.colorbar(label='Disorder')
    plt.title("Site Elimination $g=4$ and $W=1.0$")
    plt.show()

    disorder_alls = np.round(disorder_alls, 3) # type: ignore

    arr = []
    for i in range(20):
        arr.append(np.nanmean(disorder_alls[:, :i], axis=1))

    arr = np.array(arr).T[:, 0:]
    good_idxs = np.where((arr[:, -1] < -0.5) & (arr[:, -1] > -1.0))[0]


    bad_idxs = []
    for i in range(arr.shape[0]):
        if any(arr[i, :] > 0.):
            bad_idxs.append(i)  


    print(arr[0, :].shape)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    print(len(good_idxs), len(bad_idxs))
    for i in good_idxs:
        if i not in bad_idxs:
            ax.plot(np.arange(len(arr[i, :])), arr[i, :], label=f"Point {i}", alpha=0.25)
    ax.set_xlabel("Number of Iterations Averaged")
    ax.set_ylabel("Average Disorder")
    ax.set_xticks([1, 10, 19])
    ax.set_xticklabels([1, 10, 20])
    plt.show()




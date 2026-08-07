import numpy as np


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


def compute_sparse_hamiltonian(method, M, phi, t1, t2, geometric_data):
	NN = geometric_data['NN']
	NNN_CCW = geometric_data['NNN_CCW']

	if method == 'site_elim':
		hexaflake = geometric_data['hexaflake']
		NN = NN[np.ix_(hexaflake, hexaflake)]
		NNN_CCW = NNN_CCW[np.ix_(hexaflake, hexaflake)]

	i_diagonal = np.arange(NN.shape[0])
	data_diagonal = np.empty(NN.shape[0], dtype=np.complex128)
	data_diagonal[::2] = M
	data_diagonal[1::2] = -M

	i_nn, j_nn = np.where(NN)
	data_nn = np.full(i_nn.size, -t1, dtype=np.complex128)

	i_nnn_ccw, j_nnn_ccw = np.where(NNN_CCW)
	data_nnn_ccw = np.full(i_nnn_ccw.size, -t2*np.sin(phi)*1j)

	i_all = np.concatenate([i_diagonal, i_nn, i_nnn_ccw, j_nnn_ccw])
	j_all = np.concatenate([i_diagonal, j_nn, j_nnn_ccw, i_nnn_ccw])
	data_all = np.concatenate([data_diagonal, data_nn, data_nnn_ccw, -data_nnn_ccw])

	H = coo_matrix((data_all, (i_all, j_all)), shape=NN.shape).tocsr()

	return H




def compute_eigen_data(method, M, phi, t1, t2, geometric_data):

	x, y = geometric_data['x'], geometric_data['y']

	H = compute_hamiltonian(method, M, phi, t1, t2, geometric_data)

	eigenvalues, eigenvectors = sp.linalg.eigh(H, overwrite_a=True)
	
	if method in ['site_elim', 'renorm']:
		hexaflake = geometric_data['hexaflake']
		x, y = x[hexaflake], y[hexaflake]

	eigen_data = {
		'S': geometric_data['x'].size,
		'x': x,
		'y': y,
		'eigenvalues': eigenvalues,
		'eigenvectors': eigenvectors
	}

	return eigen_data


def compute_phase_diagram(
        method='hexagon', resolution=100,
        M_resolution=None, phi_resolution=None,
        M_range=(-5.5, 5.5), phi_range=(-np.pi, np.pi),
        n=3, t1=1., t2=1.,  n_jobs=-8, invmethod=None):

	valid_methods = ['hexagon', 'site_elim', 'renorm1', 'renorm2']
	if method not in valid_methods:
		raise ValueError(f"Invalid method '{method}'. Options are {valid_methods}.")

	M_resolution = resolution if M_resolution is None else M_resolution
	phi_resolution = resolution if phi_resolution is None else phi_resolution

	M_resolution += 1 - M_resolution % 2
	phi_resolution += 1 - phi_resolution % 2

	geometric_data = compute_geometric_data(n, True)

	def worker(param_idxs):
		M_idx, phi_idx = param_idxs
		M_val = (M_range[1] - M_range[0]) * (M_idx / (M_resolution - 1)) + M_range[0]
		phi_val = (phi_range[1] - phi_range[0]) * (phi_idx / (phi_resolution - 1)) + phi_range[0]

		try:
			eigen_data = compute_eigen_data(method, M_val, phi_val, t1, t2, geometric_data)
			bott = compute_bott_index(eigen_data)
			return [M_idx, phi_idx, bott]

		except Exception as e:
			print(f'An error occurred for (M, phi)=({M_val:.3f}, {phi_val:.3f}): {e}')
			return [M_idx, phi_idx, np.nan]

	idxs_sets = tuple(product(range(M_resolution), range(phi_resolution)))

	with tqdm_joblib(tqdm(total=len(idxs_sets), desc='Computing phase diagram...')):
		M_data, phi_data, bott_index_data = np.array(
			Parallel(n_jobs=n_jobs)(delayed(worker)(idx_set) for idx_set in idxs_sets)
		).T

	bott_index_array = np.empty((M_resolution, phi_resolution))
	bott_index_array[M_data.astype(np.int32), phi_data.astype(np.int32)] = bott_index_data

	phase_data = dict(
		bott_index_array=bott_index_array,
		M_range=np.asarray(M_range),
		phi_range=np.asarray(phi_range),
		t1=t1,
		t2=t2
	)

	return phase_data


def plot_spectrum_and_LDOS(eigen_data, num_states=2, cmap='inferno'):

	num_ticks_E, decimals_E = 5, 2
	num_ticks_n, decimals_n = 5, 0
	num_ticks_x, decimals_x = 5, 2
	num_ticks_y, decimals_y = 5, 2
	num_ticks_LDOS, decimals_LDOS = 5, 2

	num_states += 1 - num_states % 2

	x, y, eigenvalues, eigenvectors = [eigen_data[key] for key in 'x, y, eigenvalues, eigenvectors'.split(', ')]

	print(eigenvalues.shape)
	print(x.shape)


	all_idxs = np.arange(eigenvalues.size)
	positive = eigenvalues > 0

	positive_idxs = all_idxs[positive][np.argsort(eigenvalues[positive])][:num_states // 2]
	negative_idxs = all_idxs[~positive][np.argsort(eigenvalues[~positive])[::-1]][:num_states // 2]

	LDOS_idxs = np.concatenate((negative_idxs, positive_idxs))
	other_idxs = np.delete(all_idxs, LDOS_idxs)

	LDOS = np.sum(np.abs(eigenvectors[:, LDOS_idxs]) ** 2, axis=1)

	fig, axs = plt.subplots(1, 2, figsize=(12, 6))

	spectrum_aspect = (
							  (eigenvalues.size - 1) / (eigenvalues.max() - eigenvalues.min())
					  ) * (
							  (y.max() - y.min()) / (x.max() - x.min())
					  )

	axs[0].scatter(other_idxs, eigenvalues[other_idxs], c='black', s=20)
	axs[0].scatter(LDOS_idxs, eigenvalues[LDOS_idxs], c='red', s=30)
	axs[0].set_xlabel('n')
	axs[0].set_ylabel('E', rotation=0)
	axs[0].set_title('Spectrum')
	axs[0].set_aspect(spectrum_aspect)

	n_ticks = np.linspace(0, eigenvalues.size - 1, num_ticks_n)
	axs[0].set_xticks(n_ticks)
	axs[0].set_xticklabels([f'{tick:.{decimals_n}f}' for tick in n_ticks])

	E_ticks = np.linspace(eigenvalues.min(), eigenvalues.max(), num_ticks_E)
	axs[0].set_yticks(E_ticks)
	axs[0].set_yticklabels([f'{tick:.{decimals_E}f}' for tick in E_ticks])

	LDOS_scatter = axs[1].scatter(x, y, c=LDOS, cmap=cmap, s=7.5)
	axs[1].set_aspect('equal')
	axs[1].set_xlabel('x')
	axs[1].set_ylabel('y', rotation=0)
	axs[1].set_title('LDOS')

	x_ticks = np.linspace(x.min(), x.max(), num_ticks_x)
	axs[1].set_xticks(x_ticks)
	axs[1].set_xticklabels([f'{tick:.{decimals_x}f}' for tick in x_ticks])

	y_ticks = np.linspace(y.min(), y.max(), num_ticks_y)
	axs[1].set_yticks(y_ticks)
	axs[1].set_yticklabels([f'{tick:.{decimals_y}f}' for tick in y_ticks])

	bbox = axs[1].get_position()
	cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])
	cbar = fig.colorbar(LDOS_scatter, cax=cbar_ax)
	LDOS_ticks = np.linspace(LDOS.min(), LDOS.max(), num_ticks_LDOS)
	cbar.set_ticks(LDOS_ticks)
	cbar.set_ticklabels([f'{tick:.{decimals_LDOS}e}' for tick in LDOS_ticks])

	plt.show()


def plot_phase_diagram(phase_data, cmap='viridis', outputfile='temp.png'):

	bott_index_array = phase_data['bott_index_array']
	phi_range = phase_data['phi_range']
	M_range = phase_data['M_range']

	num_ticks_phi, decimals_phi = 5, 2
	num_ticks_M, decimals_M = 5, 2
	num_ticks_cb, decimals_cb = 5, 2

	fig, ax = plt.subplots(figsize=(7, 6))
	im = ax.imshow(
		bott_index_array,
		extent=[phi_range[0], phi_range[1], M_range[0], M_range[1]],
		origin='lower',
		aspect='auto',
		cmap=cmap
	)

	temp = np.linspace(-np.pi, np.pi, 500)
	ax.plot(temp, np.sin(temp)*np.sqrt(3)*3, c='k', ls='--')
	ax.plot(temp, -np.sin(temp)*np.sqrt(3)*3, c='k', ls='--')

	ax.set_title(f'Phase Diagram (Bott Index)')
	ax.set_xlabel('Phi')
	ax.set_ylabel('M', rotation=0)
	cbar = fig.colorbar(im, ax=ax)

	phi_ticks = np.linspace(phi_range[0], phi_range[1], num_ticks_phi)
	ax.set_xticks(phi_ticks)
	ax.set_xticklabels([f'{tick:.{decimals_M}f}' for tick in phi_ticks])

	M_ticks = np.linspace(M_range[0], M_range[1], num_ticks_M)
	ax.set_yticks(M_ticks)
	ax.set_yticklabels([f'{tick:.{decimals_phi}f}' for tick in M_ticks])

	bott_min, bott_max = np.nanmin(bott_index_array), np.nanmax(bott_index_array)
	cbar_ticks = np.linspace(bott_min, bott_max, num_ticks_cb)
	cbar.set_ticks(cbar_ticks)
	cbar.set_ticklabels([f'{tick:.{decimals_cb}f}' for tick in cbar_ticks])

	plt.tight_layout()
	plt.savefig(outputfile)
     



def comp():
	generations = [0, 1, 2, 3, 4]
	n_honeycomb = [6, 42, 366, 3282, 29526]
	n_hexaflake = [6, 42, 294, 2058, 14406]
	hexaflake_percentages = np.array([100*nhex/nhon for nhex, nhon in zip(n_hexaflake, n_honeycomb)])
	percent_nontrivial = np.array([np.nan, np.nan, 42.88, 28.64, 18.56]
)
	fig, ax = plt.subplots(1, 1, figsize=(8,4))
	ax.grid(True, ls='--', c='k', alpha=0.5)
	ax.scatter(generations, hexaflake_percentages, label='Hexaflake Percentage')
	ax.scatter(generations, percent_nontrivial, label='site_elim Percent Non-trivial')
	ax.scatter(generations, hexaflake_percentages-percent_nontrivial, label="Difference")
	ax.set_xlabel("Generation", fontsize=12)
	ax.set_ylabel("Percentage (%)", fontsize=12)
	ax.set_title("Percentage Comparison of Honeycomb and Hexaflake Lattices")
	ax.set_xticks(generations)
	ax.set_yticks([0, 25, 50, 75, 100])
	plt.legend()
	plt.tight_layout()
	plt.show()


def temp():
	import h5py
	def get_amount(gen):
		f = f'./Hexaflake/Data/site_elim_g{gen}_(25_by_25).h5'
		with h5py.File(f, 'r') as file:
			M = file['M'][:] # type: ignore
			phi = file['phi'][:] # type: ignore
			bi = file['bott_index'][:].flatten() # type: ignore
			bi = np.round(bi).astype(int)

		in_phase = np.where(M <= np.sin(phi) * np.sqrt(3) * 3, True, False)
		amount_good = len(np.where(in_phase & (bi != 0))[0])
		return len(np.where(in_phase)[0]), amount_good

	generations = [2, 3, 4]
	amounts = [get_amount(gen) for gen in generations]
	n_sites = np.array([6 * (7 ** gen) for gen in generations])
	ratios = [amount_good/amounts[0][0] for _, amount_good in amounts]
	
	print(amounts)
	print(np.round(ratios, 3))
	print(n_sites)

	fig, axs = plt.subplots(1, 2, figsize=(12, 5))
	axs[0].grid(zorder=0, alpha=0.5, ls='--')
	axs[1].grid(zorder=0, alpha=0.5, ls='--')



	axs[0].scatter(1/np.array(generations), ratios, zorder=1)
	axs[0].set_xlabel('1/generation')

	axs[1].scatter(1/n_sites, ratios, zorder=1)
	axs[1].set_xlabel('1/n_sites')
	axs[0].set_ylabel('% topological')

	axs[0].set_xlim(0.0 - 0.05, 0.75 + 0.05)
	axs[1].set_ylim(0.0 - 0.05, 1.0 + 0.05)

	fig.suptitle("Scaling of topological region with system size : Site Elimination")
	axs[0].set_title('Percent of region that is topological vs $1/g$')
	axs[1].set_title('Percent of region that is topological vs $1/N$')

	slope, intercept = np.polyfit(1/np.array(generations), ratios, 1)
	xmin, xmax = axs[0].get_xlim()
	x_fit = np.linspace(xmin, xmax, 100)
	y_fit = slope * x_fit + intercept
	axs[0].plot(x_fit, y_fit, c='red', ls='--', label=f'Fit: y={slope:.2f}x + {intercept:.2f}', alpha=0.5)
	axs[0].legend()

	axs[0].set_ylim(y_fit[0] - 0.05, y_fit[-1] + 0.05)
	

	plt.show()
      


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
      

def compute_dx_and_dy_discrete(x_discrete, y_discrete, PBC, chunk_size=500):
    """
    Compute discrete differences in a highly memory-efficient way using chunking.

    This version processes the data in row-based chunks to avoid creating
    large intermediate (N, N) arrays, making it suitable for very large inputs
    with limited RAM.

    Args:
        x_discrete (np.ndarray): 1D array of x-coordinates of points.
        y_discrete (np.ndarray): 1D array of y-coordinates of points.
        PBC (bool): Flag indicating whether to apply periodic boundary conditions.
        chunk_size (int): The number of rows to process in each chunk.
                          Smaller values use less RAM but may be slightly slower.

    Returns:
        tuple:
            - (np.ndarray) delta_x_discrete: 2D array of differences in x-coordinates.
            - (np.ndarray) delta_y_discrete: 2D array of differences in y-coordinates.
    """
    N = x_discrete.size

    # The non-PBC case still creates two (N, N) arrays. If this is too large,
    # the same chunking method would need to be applied here as well.
    if not PBC:
        delta_x_discrete = x_discrete[np.newaxis, :] - x_discrete[:, np.newaxis]
        delta_y_discrete = y_discrete[np.newaxis, :] - y_discrete[:, np.newaxis]
        return delta_x_discrete, delta_y_discrete

    # --- PBC Calculation with Chunking ---


    a = round(np.sqrt(2 * N - 3))
    b = (a + 3) // 2
    c = (a - 3) // 2
    d = 2 * a - b
    e = 2 * a - c
    shifts = np.array([
        [0, 0], [-3, a], [3, -a],
        [d, b], [-d, -b], [-e, c], [e, -c]
    ])

    # Pre-calculate squared constants
    C1_SQ = 0.25  # (1/2)^2
    C2_SQ = 0.75  # (sqrt(3)/2)^2

    # Pre-allocate the final full-size arrays in memory. This will be the
    # largest memory allocation.
    delta_x_final = np.empty((N, N), dtype=np.int64)
    delta_y_final = np.empty((N, N), dtype=np.int64)

    # Process the N x N matrix in horizontal chunks
    for i in range(0, N, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, N)
        
        # Select the 'i' coordinates for the current chunk
        x_i_chunk = x_discrete[start_row:end_row, np.newaxis]
        y_i_chunk = y_discrete[start_row:end_row, np.newaxis]
        
        # Initialize the results for this chunk with the 'no shift' case
        # This creates temporary arrays of shape (chunk_size, N)
        delta_x_chunk = x_discrete[np.newaxis, :] - x_i_chunk
        delta_y_chunk = y_discrete[np.newaxis, :] - y_i_chunk
        
        min_sq_dist_chunk = C1_SQ * delta_x_chunk**2 + C2_SQ * delta_y_chunk**2

        # Iterate through the remaining shifts
        for shift_x, shift_y in shifts[1:]:
            current_dx_chunk = delta_x_chunk - shift_x
            current_dy_chunk = delta_y_chunk - shift_y
            
            current_sq_dist_chunk = C1_SQ * current_dx_chunk**2 + C2_SQ * current_dy_chunk**2
            
            update_mask = current_sq_dist_chunk < min_sq_dist_chunk
            
            # Update the chunk arrays where a shorter distance was found
            min_sq_dist_chunk[update_mask] = current_sq_dist_chunk[update_mask]
            delta_x_chunk[update_mask] = current_dx_chunk[update_mask]
            delta_y_chunk[update_mask] = current_dy_chunk[update_mask]
        
        # Assign the finalized chunk to the correct slice of the output arrays
        delta_x_final[start_row:end_row, :] = delta_x_chunk
        delta_y_final[start_row:end_row, :] = delta_y_chunk

    return delta_x_final, delta_y_final


def compute_hopping_arrays(delta_x_discrete, delta_y_discrete):

	NN = ((np.abs(delta_x_discrete) ==  2) & (delta_y_discrete == 0)) | \
		 ((np.abs(delta_x_discrete) == 1) & (np.abs(delta_y_discrete) == 1))

	NNN = ((delta_x_discrete ==  0) & (np.abs(delta_y_discrete) == 2)) | \
		 ((np.abs(delta_x_discrete) == 3) & (np.abs(delta_y_discrete) == 1))

	CCW_directions = np.array([[1, -1, 1, 0], [1, 1, -1, 1], [-1, 0, -1, -1], [-1, 1, -1, 0], [-1, -1, 1, -1], [1, 0, 1, 1]], dtype=np.int8)

	i, j = np.where(NNN)

	k = np.argmax(NN[i] & NN[j], axis=1)

	x_i_to_k, y_i_to_k = delta_x_discrete[i, k], delta_y_discrete[i, k]
	x_k_to_j, y_k_to_j = delta_x_discrete[k, j], delta_y_discrete[k, j]

	NNN_directions = np.sign(np.array([x_i_to_k, y_i_to_k, x_k_to_j, y_k_to_j]).T).astype(np.int8)

	CCW = np.any(np.all(CCW_directions[None, :, :] == NNN_directions[:, None, :], axis=2), axis=1)

	NNN_CCW = np.full_like(NNN, False)
	NNN_CCW[i, j] = CCW

	return NN, NNN_CCW




# pre sparse hamiltonian construc



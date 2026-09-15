import numpy as np
import scipy.linalg as spla

def compute_non_hermitian_berry_curvature(kx, ky, m0, h_vector, t, t0, n):
    """
    https://doi.org/10.1103/PhysRevB.98.165148
    Left eigenvector \chi_n
    Right eigenvector \phi_n

    Berry Connection
    A_n^i := 1j \langle \chi_n(\vec{k}) | \partial_{k_i} \phi_n (\vec{k}) \rangle

    Berry Curvature
    F_n(\vec{k}) := \partial_{k_x} A_n^y(\vec{k}) - \partial_{k_y} A_n^x(\vec{k})
    
    """
    tau_x = np.array([[0, 1], [1, 0]])
    tau_y = np.array([[0, -1j], [1j, 0]])
    tau_z = np.array([[1, 0], [0, -1]])
    tau = np.array((tau_x, tau_y, tau_z))
    tau = np.rollaxis(tau, 0, 3)

    def compute_eig(kx_, ky_):
        d_vector = compute_d_vector(kx_, ky_, m0, h_vector, t, t0)
        hamiltonian = np.dot(tau, d_vector)
        eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True)
        return eigenvalues[n], left_eigenvectors[n], right_eigenvectors[n]
    
    def compute_eig_right(kx_, ky_):
        return compute_eig(kx_, ky_)[2]

    def compute_berry_connection(kx_, ky_, axis:str, dk=1e-5):
        eigenvalue, left_eigvec, right_eigvec = compute_eig(kx_, ky_)
        partial_right = compute_vector_derivative(compute_eig_right, kx_, ky_, axis, dk)
        return 1j * np.dot(np.conj(left_eigvec), partial_right)
    
    partial_kx_Ay = compute_vector_derivative(compute_berry_connection, kx, ky, 'x', dk=1e-5, axis='y')
    partial_ky_Ax = compute_vector_derivative(compute_berry_connection, kx, ky, 'y', dk=1e-5, axis='x')
    return partial_kx_Ay - partial_ky_Ax


def compute_non_hermitian_berry_curvature2(kx, ky, m0, gamma, t, n):
    """
    https://doi.org/10.1103/PhysRevB.98.165148
    Left eigenvector \chi_n
    Right eigenvector \phi_n

    Berry Connection
    A_n^i := 1j \langle \chi_n(\vec{k}) | \partial_{k_i} \phi_n (\vec{k}) \rangle

    Berry Curvature
    F_n(\vec{k}) := \partial_{k_x} A_n^y(\vec{k}) - \partial_{k_y} A_n^x(\vec{k})
    
    """
    tau_x = np.array([[0, 1], [1, 0]])
    tau_y = np.array([[0, -1j], [1j, 0]])
    tau_z = np.array([[1, 0], [0, -1]])
    tau = np.array((tau_x, tau_y, tau_z))
    tau = np.rollaxis(tau, 0, 3)

    def compute_eig(kx_, ky_):
        d_vector = compute_d_vector2(kx_, ky_, m0, gamma, t)
        hamiltonian = np.dot(tau, d_vector)
        eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True)
        return eigenvalues[n], left_eigenvectors[n], right_eigenvectors[n]
    
    def compute_eig_right(kx_, ky_):
        return compute_eig(kx_, ky_)[2]

    def compute_berry_connection(kx_, ky_, axis:str, dk=1e-5):
        eigenvalue, left_eigvec, right_eigvec = compute_eig(kx_, ky_)
        partial_right = compute_vector_derivative(compute_eig_right, kx_, ky_, axis, dk)
        return 1j * np.dot(np.conj(left_eigvec), partial_right)
    
    partial_kx_Ay = compute_vector_derivative(compute_berry_connection, kx, ky, 'x', dk=1e-5, axis='y')
    partial_ky_Ax = compute_vector_derivative(compute_berry_connection, kx, ky, 'y', dk=1e-5, axis='x')
    return partial_kx_Ay - partial_ky_Ax


def compute_chern_number2(m0, gamma, t, resolution = (101, 101), n=0):
    kx, ky = compute_square_brillouin_zone(resolution)
    berry_curvature = [compute_non_hermitian_berry_curvature2(kx_, ky_, m0, gamma, t, n) for kx_, ky_ in zip(kx, ky)]
    dkx = (np.max(kx) - np.min(kx)) / resolution[0]
    dky = (np.max(ky) - np.min(ky)) / resolution[1]

    sum_kx = np.sum(berry_curvature, axis=0) * dkx
    sum_total = -np.sum(sum_kx) * dky
 
    if np.abs(sum_total) > 10:
        print(f"Warning: Chern2 number is too large for m0 = {m0:.3f}, gamma = {gamma} : {sum_total:.3e}.")
        return None
    elif np.isnan(sum_total):
        return None
    else:
        return np.round(sum_total) / 2 * np.pi


def compute_d_vector2(kx, ky, m0, gamma, t):
    # Method 2: From https://doi.org/10.1103/PhysRevB.98.165148
    d1 = m0 + t * np.cos(kx) + t * np.cos(ky)
    d2 = 1j * gamma + t * np.sin(kx)
    d3 = t * np.sin(ky)
    return np.array([d1, d2, d3])


def compute_fhs_chern(m0:float, h_vector:np.ndarray, t:float, t0:float, n:int=0, Lx:int=31, Ly:int=31, qx:float=1.0, qy:float=1.0):
    dkx = 2 * np.pi / qx / Lx
    dky = 2 * np.pi / qy / Ly
    def compute_hamiltonian(kx, ky):
        paulix = np.array([[0, 1], [1, 0]])
        pauliy = np.array([[0, -1j], [1j, 0]])
        pauliz = np.array([[1, 0], [0, -1]])
        d1, d2, d3 = compute_d_vector(kx, ky, m0, h_vector, t, t0)
        return paulix * d1 + pauliy * d2 + pauliz * d3
    
    def compute_left_right_eigenvectors(kx, ky):
        hamiltonian = compute_hamiltonian(kx, ky)
        eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True)
        return left_eigenvectors[:, n], right_eigenvectors[:, n]

    def U_mu(mu, kx, ky):
        """
        U(1) link variable from the wavefunctions of the nth Bloch band
        
        :param mu: (str) either 'x' or 'y'
        :param k_l: (ndarray) 2xN array s.t. the first column is k_x values and the second is k_y values.
        """
        # N_\mu = |<n(k_l)|n(k_l + \hat{\mu})>|
        # U_\mu(k_l) = <n(k_l)|n(k_l + \hat{\mu})> / N_\mu
        kx2, ky2 = (kx + dkx, ky) if mu == 'x' else (kx, ky + dky)
        lvec1, rvec1 = compute_left_right_eigenvectors(kx, ky)
        lvec2, rvec2 = compute_left_right_eigenvectors(kx2, ky2)

        dot_left = np.dot(lvec1, lvec2)
        dot_right = np.dot(rvec1, rvec2)
        return dot_left/spla.norm(dot_left), dot_right/spla.norm(dot_right)


    def compute_field_strength(kx, ky, left_or_right='left'):
        if left_or_right == 'left':
            idx = 0
        else:
            idx = 1
        term1 = U_mu('x', kx,       ky      )[idx]
        term2 = U_mu('y', kx + dkx, ky      )[idx]
        term3 = U_mu('x', kx,       ky + dky)[idx].conj()
        term4 = U_mu('y', kx,       ky      )[idx].conj()
        return np.log(term1 * term2 * term3 * term4)
  

    # Square Lattice FBZ
    kxs, kys = np.linspace(-np.pi, np.pi, Lx, endpoint=False), np.linspace(-np.pi, np.pi, Ly, endpoint=False)
    kxs, kys = np.meshgrid(kxs, kys)
    kxs, kys = kxs.flatten(), kys.flatten()
    
    chern = np.round(sum(compute_field_strength(kx, ky) for kx, ky in zip(kxs, kys)) / (2 * np.pi * 1j), 3)
    return chern


def compute_FHS_chern_fast(m0, h_vector, t, t0, n_band=0, Lx=31, Ly=31):
    # 1. Setup Grid
    k_vec = np.linspace(-np.pi, np.pi, Lx, endpoint=False)
    kx_grid, ky_grid = np.meshgrid(k_vec, k_vec)
    
    d = compute_d_vector(kx_grid, ky_grid, m0, h_vector, t, t0) 
    
    H = np.zeros((Lx, Ly, 2, 2), dtype=complex)

    # tau \cdot (\vec{d_NH})
    H[:, :, 0, 0] = d[2]
    H[:, :, 1, 1] = -d[2]
    H[:, :, 0, 1] = d[0] - 1j * d[1]
    H[:, :, 1, 0] = d[0] + 1j * d[1]

    H_flat = H.reshape(-1, 2, 2)
    L_vecs = np.zeros((Lx * Ly, 2), dtype=complex)
    R_vecs = np.zeros((Lx * Ly, 2), dtype=complex)
    
    for i in range(len(H_flat)):
        w, vl, vr = spla.eig(H_flat[i], left=True, right=True)
        idx = np.lexsort((np.imag(w), np.real(w)))
        
        # Extract the desired band
        chosen_idx = idx[n_band]
        L_vecs[i] = vl[:, chosen_idx]
        R_vecs[i] = vr[:, chosen_idx]

    # Reshape back to grid
    L_grid = L_vecs.reshape(Lx, Ly, 2)
    R_grid = R_vecs.reshape(Lx, Ly, 2)

    # 4. Compute Link Variables (Biorthogonal)
    # Roll to get neighbors (k+x, k+y)
    R_plus_x = np.roll(R_grid, -1, axis=1)
    R_plus_y = np.roll(R_grid, -1, axis=0)
    
    # Inner products <L(k)|R(k+mu)>
    ov_x = np.einsum('ijk,ijk->ij', np.conj(L_grid), R_plus_x)
    ov_y = np.einsum('ijk,ijk->ij', np.conj(L_grid), R_plus_y)
    
    # Check for Exceptional Points (where overlap is zero)
    # If <L|R> ~ 0, the gap is closed or bands merged -> Chern undefined
    min_overlap = np.min(np.abs(ov_x))
    if min_overlap < 1e-6:
        print(f"Warning: Possible Exceptional Point detected (Overlap ~ {min_overlap:.2e})")

    # Normalize (keep phase only)
    U_x = ov_x / (np.abs(ov_x) + 1e-12)
    U_y = ov_y / (np.abs(ov_y) + 1e-12)

    # 5. Field Strength F = ln( U_x * U_y(x) * U_x(y)* * U_y* )
    U_x_py = np.roll(U_x, -1, axis=0)
    U_y_px = np.roll(U_y, -1, axis=1)
    
    F_plaq = np.log(U_x * U_y_px * np.conj(U_x_py) * np.conj(U_y))
    
    # 6. Sum flux
    total_flux = np.sum(F_plaq)
    chern = np.real(total_flux / (2j * np.pi))
    
    return chern


def plot_nh_figure(fig:plt.Figure, eigval_ax:plt.Axes, Lattice:DefectLattice, eigenvalues:np.ndarray, L_over_R:np.ndarray):
    lattice, defect_indices = Lattice.lattice, Lattice.defect_indices

    box = eigval_ax.get_position()
    zx = box.width / 7.5
    zy = box.width / 20
    n = 3
    eigvec_ax = fig.add_axes([box.x0 + box.width * (1 - 1/n) - zx, box.y0 + zy, box.width / n, box.height / n])

    eigval_ax = plot_complex_spectrum(eigval_ax, eigenvalues, defect_indices, scatter_kwargs={'c':'black','s':25})
    eigvec_ax, colorbar_ax = plot_on_lattice(fig, eigvec_ax, lattice, L_over_R, "scatter" if Lattice.defect_type in ["interstitial", "frenkel_pair"] else "imshow")
    if Lattice.defect_type in ['interstitial', 'frenkel_pair']:
        xticks = [0, Lattice.Lx - 1]
        yticks = [0, Lattice.Ly - 1]
        eigvec_ax.set_xticklabels([str(tick + 1) for tick in xticks], fontsize=12)
        eigvec_ax.set_yticklabels([str(tick + 1) for tick in yticks], fontsize=12)
    return fig, eigval_ax, eigvec_ax, colorbar_ax


def plot_comparison_of_regimes(Lattice:DefectLattice, h_vector, m0_values:np.ndarray, resolution_scale:int = 6):
    # If msub is not applicable, using m0 as columns and only one row.
    # Otherwise, use m0 as rows and msub as columns

    m0_values = np.array(m0_values)
    m0_values = m0_values[np.argsort(m0_values)[::-1]]

    if Lattice.defect_type in ['none', 'vacancy', 'schottky']:
        msub_values = []
        m0_values = m0_values[::-1]
    else:
        msub_values = np.array(m0_values)
        msub_values = msub_values[np.argsort(msub_values)]

    if len(msub_values) <= 1:
        if msub_values == []:
            msub_values = [None]
        n_rows = 1
        n_cols = len(m0_values)

    else:
        n_rows = len(m0_values)
        n_cols = len(msub_values) - 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(resolution_scale * n_cols, resolution_scale * n_rows))
    if n_rows == 1:
        axs = axs.reshape(n_cols, 1)
    for i, m0 in enumerate(m0_values):
        good_msub_values = np.array(msub_values)[np.array(msub_values) != m0]
        for j, msub in enumerate(good_msub_values):
            eigvals, L, R, _, _, _ = compute_eigenvectors_eigenvalues(Lattice, m0, h_vector, msub).values()
            fig, axs[i, j], eigvec_ax, cbar_ax = plot_nh_figure(fig, axs[i, j], Lattice, eigvals, L/R)
            axs[i, j].set_title("")
            match Lattice.defect_type:
                case "vacancy" | "schottky" | "none":
                    annotation = f"$m_0={m0}$"
                case "substitution":
                    annotation = f"$m_0={m0}$\n$m_0^{{\\text{{sub}}}}={msub}$"
                case "interstitial" | "frenkel_pair":
                    annotation = f"$m_0={m0}$\n$m_0^{{\\text{{int}}}}={msub}$"
            if h_vector[-1] == 0.0:
                axs[i, j].annotate(
                    annotation,
                    xy = (0.025, 0.95),
                    xycoords = 'axes fraction',
                    ha = 'left',
                    va = 'top',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.0)
                )
            else:
                axs[i, j].annotate(
                    annotation,
                    xy = (0.025, 0.5),
                    xycoords = 'axes fraction',
                    ha = 'left',
                    va = 'top',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.0)
                )


def plot_nhse_xyz_left_right(Lx:int, Ly:int, m0:float):
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    plt.subplots_adjust(hspace=0.33, wspace=0.33)

    Lattice_obc = DefectLattice(Lx, Ly, "none", False)
    Lattice_pbc = DefectLattice(Lx, Ly, "none", True)
    eig1_obc, L1, R1, _, _, _ = compute_eigenvectors_eigenvalues(Lattice_obc, m0, [0.25, 0.0, 0.0]).values()
    eig2_obc, L2, R2, _, _, _ = compute_eigenvectors_eigenvalues(Lattice_obc, m0, [0.0, 0.25, 0.0]).values()
    eig3_obc, L3, R3, _, _, _ = compute_eigenvectors_eigenvalues(Lattice_obc, m0, [0.0, 0.0, 0.25]).values()
    eig1_pbc, _, _, _, _, _ =   compute_eigenvectors_eigenvalues(Lattice_pbc, m0, [0.25, 0.0, 0.0]).values()
    eig2_pbc, _, _, _, _, _ =   compute_eigenvectors_eigenvalues(Lattice_pbc, m0, [0.0, 0.25, 0.0]).values()
    eig3_pbc, _, _, _, _, _ =   compute_eigenvectors_eigenvalues(Lattice_pbc, m0, [0.0, 0.0, 0.25]).values()

    eigval_obc_list = [eig1_obc, eig2_obc, eig3_obc]
    eigval_pbc_list= [eig1_pbc, eig2_pbc, eig3_pbc]
    L_list = [L1, L2, L3]
    R_list = [R1, R2, R3]

    for i in range(3):
        plot_complex_spectrum(axs[i, 0], eigval_obc_list[i], scatter_kwargs = {'c':'red', 'label':'OBC'})
        plot_complex_spectrum(axs[i, 0], eigval_pbc_list[i], scatter_kwargs = {'c':'blue', 'label':'PBC'})
        axs[i, 0].legend()
        axs[i, 0].set_title('')
        
        plot_on_lattice(fig, axs[i, 1], Lattice_obc.lattice, L_list[i], 'imshow', label_fontsize=24, tick_fontsize=20)
        plot_on_lattice(fig, axs[i, 2], Lattice_obc.lattice, R_list[i], 'imshow', label_fontsize=24, tick_fontsize=20)

    #plt.tight_layout()
    plt.savefig(f'./NonHermitian/Plots/nhse_square_lattice_m0={m0}.png')


def main(defect_type, Lx, Ly, dir, h,
         frenkel_x_disp=None, frenkel_y_disp=None, schottky_separation=None, doOverwrite=True):
    Lattice = DefectLattice(Lx, Ly, defect_type, True, 
                            schottky_separation=schottky_separation, frenkel_x_disp=frenkel_x_disp, frenkel_y_disp=frenkel_y_disp)
    match dir:
        case 'x':
            hv = [h, 0.0, 0.0]
        case 'y':
            hv = [0.0, h, 0.0]
        case 'z':
            hv = [0.0, 0.0, h]
    plot_comparison_of_regimes(Lattice, hv, [-2.5, -1.0, 1.0, 2.5])

    basename = './NonHermitian/Plots/'+f'{Lattice.defect_type}_h{dir}'
    if defect_type == 'frenkel_pair':
        basename = './NonHermitian/Plots/FrenkelPair/'+f'{Lattice.defect_type}_h{dir}_x{frenkel_x_disp}_y{frenkel_y_disp}'

    def recursive_filesave(basename, ext):
        def _save(base, ext, num):
            if not os.path.exists(base+ext):
                plt.savefig(base+ext)
            elif os.path.exists(base+f'_{num}'+ext):
                _save(base, ext, num + 1)
            else:
                plt.savefig(base+f'_{num}'+ext)
        _save(basename, ext, 0)

    if doOverwrite:
        plt.savefig(basename + '.png')
        plt.savefig(basename + '.svg')
    else:
        recursive_filesave(basename, '.png')
        recursive_filesave(basename, '.svg')



    # Hard-coded selection for certain parameters
    if 0:
        if h_vector[0] == 0.25 or h_vector[1] == 0.25 or h_vector[2] == 0.25:
            if all([h_vector[2], not any(h != 0 for h in h_vector[:2]), Lattice.defect_type == 'interstitial']):
                close_to_zero_idxs = get_separated_points(eigenvalues)
                if (m0, msub) in [(1.0, 2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:2]
                if (m0, msub) in [(2.5, -2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[:2]
            elif all([any(h != 0 for h in h_vector[:2]), Lattice.defect_type == "substitution"]):
                if (m0, msub) in [(2.5, -1.0)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:4]
            elif all([h_vector[2], not any(h != 0 for h in h_vector[:2]), Lattice.defect_type == 'frenkel_pair']):
                if (m0, msub) in [(1.0, -1.0), (1.0, 2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:4]
                elif (m0, msub) in [(2.5, -1.0)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[:2]
                elif (m0, msub) in [(2.5, -2.5)]:
                    arr = 10 * np.abs(eigenvalues.imag) - np.abs(eigenvalues.real)
                    close_to_zero_idxs = np.argsort(arr)[:2]
            elif all([any(h != 0 for h in h_vector[:2]), Lattice.defect_type == 'frenkel_pair']):
                if (m0, msub) in [(1.0, -1.0), (1.0, 2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:4]
        elif np.abs(h_vector[2]) >= 1 :
            if Lattice.defect_type in ['substitution', 'interstitial', 'frenkel_pair']:
                close_to_zero_idxs = get_separated_points(eigenvalues)
            else:
                if np.abs(m0) >= 2.0:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[:2]
        elif np.abs(h_vector[0]) >= 1 and Lattice.defect_type != 'none':
            if Lattice.defect_type == 'interstitial':
                if (m0, msub) in [(1.0, 2.5)]:
                    close_to_zero_idxs = np.argsort(np.abs(eigenvalues))[:4]
        elif h_vector[2] == 0.5 and Lattice.defect_type == 'interstitial':
            if (m0, msub) in [(1.0, -3.0)]:
                close_to_zero_idxs = np.argsort(np.abs(eigenvalues.real))[-2:]
            if (m0, msub) in [(1.0, -1.0)]:
                close_to_zero_idxs = get_separated_points(eigenvalues)
        elif h_vector[2] == 0.5 and Lattice.defect_type == 'substitution':
            if (m0, msub) in [(3., -3.)]:
                close_to_zero_idxs = np.argsort(np.abs(eigenvalues.imag))[:2]


def big_nvalues_probe(Lattice, m0_values, h_vector, hsub_values, ext:str = '.png', overwrite:bool = False):
    if h_vector[0]:
        hdir = 'x'
        h = h_vector[0]
    elif h_vector[1]:
        hdir = 'y'
        h = h_vector[1]
    elif h_vector[2]:
        hdir = 'z'
        h = h_vector[2]
    else:
        hdir = 'NA'
        h = 0.0

    directory = "./NonHermitian/Plots/" + Lattice.defect_type.capitalize() + "/"
    basename = f"{Lattice.defect_type}_h{hdir}={h}"
    if Lattice.defect_type == "frenkel_pair":
        basename += f"_fx={Lattice._fp_xdisp}_fy={Lattice._fp_ydisp}"
    basename += f"_L={Lattice.Lx}"

    if os.path.exists(directory + basename + ext) and not overwrite:
        print(f"File '{directory + basename + ext}' exists and overwrite is False")
        return

    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc(('xtick.major', 'ytick.major'), width=2.5) # type: ignore

    fig, axs = plt.subplots(len(m0_values), 5, figsize=(6 * 5, 6 * len(m0_values)))

    if len(m0_values) == 1:
        axs = np.array(axs).reshape(1, 5)

    if len(msub_values) != len(m0_values):
        msub_values = [None] * len(m0_values)

    for i, (m0, msub) in enumerate(zip(m0_values, msub_values)):
        if all([
            abs(m0) == 2.5,
            msub is not None and abs(msub) == 1.0 and m0 * msub == -2.5,
            any(h == 0.25 for h in h_vector[:2]),
            any(h <= 1.0 for h in h_vector[:2]),
            Lattice.defect_type == "substitution",
            ]):
            zoomGap = True
        elif all([Lattice.defect_type == 'frenkel_pair', m0 == -2.5, any(h != 0 for h in h_vector[:2])]):
            zoomGap = True
        else:
            zoomGap = False

        fig, axs[i, :] = plot_spectrum_ldos(fig, axs[i, :], Lattice, m0, h_vector, msub, zoomGap)

        
        if Lattice.defect_type in ["vacancy", "schottky", "none"]:
            annotation_text = f'$m_0 = {m0}$'
        elif Lattice.defect_type == "substitution":
            annotation_text = f'$m_0^{{\\rm back}} = {m0}$' + f"\n$m_0^{{\\rm sub}} = {msub}$"
        elif Lattice.defect_type in ["interstitial", "frenkel_pair"]:
            annotation_text = f'$m_0^{{\\rm back}} = {m0}$' + f"\n$m_0^{{\\rm int}} = {msub}$"

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

    plt.savefig(directory + basename + ext, bbox_inches='tight', pad_inches=0, dpi=96)


def plot_nhse_xyz_left_right(Lx:int, Ly:int, m0:float):
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    plt.subplots_adjust(hspace=0.33, wspace=0.33)

    Lattice_obc = DefectLattice(Lx, Ly, "none", False)
    Lattice_pbc = DefectLattice(Lx, Ly, "none", True)
    eig1_obc, L1, R1, _, _, _ = compute_eigenvectors_eigenvalues(Lattice_obc, m0, [0.25, 0.0, 0.0]).values()
    eig2_obc, L2, R2, _, _, _ = compute_eigenvectors_eigenvalues(Lattice_obc, m0, [0.0, 0.25, 0.0]).values()
    eig3_obc, L3, R3, _, _, _ = compute_eigenvectors_eigenvalues(Lattice_obc, m0, [0.0, 0.0, 0.25]).values()
    eig1_pbc, _, _, _, _, _ =   compute_eigenvectors_eigenvalues(Lattice_pbc, m0, [0.25, 0.0, 0.0]).values()
    eig2_pbc, _, _, _, _, _ =   compute_eigenvectors_eigenvalues(Lattice_pbc, m0, [0.0, 0.25, 0.0]).values()
    eig3_pbc, _, _, _, _, _ =   compute_eigenvectors_eigenvalues(Lattice_pbc, m0, [0.0, 0.0, 0.25]).values()

    eigval_obc_list = [eig1_obc, eig2_obc, eig3_obc]
    eigval_pbc_list= [eig1_pbc, eig2_pbc, eig3_pbc]
    L_list = [L1, L2, L3]
    R_list = [R1, R2, R3]

    for i in range(3):
        plot_complex_spectrum(axs[i, 0], eigval_obc_list[i], scatter_kwargs = {'c':'red', 'label':'OBC'})
        plot_complex_spectrum(axs[i, 0], eigval_pbc_list[i], scatter_kwargs = {'c':'blue', 'label':'PBC'})
        axs[i, 0].legend()
        axs[i, 0].set_title('')
        
        plot_on_lattice(fig, axs[i, 1], Lattice_obc, L_list[i], 'imshow', label_fontsize=24, tick_fontsize=20)
        plot_on_lattice(fig, axs[i, 2], Lattice_obc, R_list[i], 'imshow', label_fontsize=24, tick_fontsize=20)

    #plt.tight_layout()
    plt.savefig(f'./NonHermitian/Plots/nhse_square_lattice_m0={m0}.png')


def compute_figures(L, defect_types, h, h_directions = 'xz', fpd=-3.5,
                    m0_values =   [2.5, 1.0],
                    msub_values = [2.5, 1.0, -1.0, -2.5],
                    overwrite = False,
                    set_values:list = []):
    


    if set_values == []:
        unique_pairs = np.array(list({(xi, yi) for xi in m0_values for yi in msub_values if xi != yi}))
        sort = np.lexsort((unique_pairs[:, 1], -unique_pairs[:, 0]))
        unique_pairs = unique_pairs[sort]
    else:
        unique_pairs = np.array(set_values)

    default = np.array((h, 0.0, 0.0))
    dir_map = {'x':0,'y':1,'z':2}
    h_vectors = [np.roll(default, dir_map[d]) for d in h_directions]
    # 'none', 'vacancy', 'schottky', 'substitution'
    for deftype in defect_types:
        Lattice = DefectLattice(L, L, deftype, True, schottky_separation=7, defect_radius=1)
        for hv in h_vectors:
            if deftype in ['vacancy', 'schottky', 'none']:
                big_nvalues_probe(Lattice, m0_values, hv, [], overwrite=overwrite)
            else:
                big_nvalues_probe(Lattice, unique_pairs[:, 0], hv, unique_pairs[:, 1], overwrite=overwrite)

    if 'frenkel_pair' not in defect_types:
        return

    for (fpx, fpy) in [(fpd, fpd)]:
        Lattice = DefectLattice(L, L, "frenkel_pair", True, frenkel_x_disp=fpx, frenkel_y_disp=fpy)
        for hv in h_vectors:
            big_nvalues_probe(Lattice, unique_pairs[:, 0], hv, unique_pairs[:, 1], overwrite=overwrite)


def big_defect():
    L = 20
    dr = 1
    Lattice = DefectLattice(L, L, "interstitial", True, frenkel_x_disp=-5.5, frenkel_y_disp=-5.5, defect_radius=dr)
    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    m0, hv, msub = -1.0, [0.25, 0., 0.], -2.5
    plot_spectrum_ldos(fig, axs[1:], Lattice, m0, hv, msub, False)
    eigval_dict = compute_eigenvectors_eigenvalues(Lattice, m0, hv, msub)
    eigvals = eigval_dict['eigenvalues']
    #axs[0].scatter(np.arange(len(eigvals)), np.abs(eigvals), c='k', zorder=0)
    axs[0].scatter(np.arange(len(eigvals)), eigvals.real, c='b', zorder=1, alpha=0.5)
    axs[0].scatter(np.arange(len(eigvals)), eigvals.imag, c='r', alpha=0.1, zorder=2)
    points = np.array([(L / 2 - dr + 0.5, L / 2 - 0.5), (L / 2 - 0.5, L / 2 - dr + 0.5), (L / 2 + dr - 1.5, L / 2 - 0.5), (L / 2 - 0.5, L / 2 + dr - 1.5), (L / 2 - dr + 0.5, L / 2 - 0.5)])
    for ax in axs[2:].flatten():
        ax.plot(points[:, 0], points[:, 1], c='r', ls='-', lw=2, alpha=0.25, zorder=100)

    plt.show()


def compute_gap(Lattice, m0, hvec, hsub=None):
    hamiltonian = compute_hamiltonian(Lattice, m0, hvec, 1.0, 1.0, hsub)
    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True) # type: ignore
    
    lowest_energy_eigval = np.sort(np.abs(eigenvalues))[0]

    return lowest_energy_eigval


def compute_gap_over_region(L, deftype, m0_range, h_range, hdir, resolution=(25,25)):
    Lattice = DefectLattice(L, L, deftype, True, schottky_separation=5)
    parameters = tuple(product(np.linspace(m0_range[0], m0_range[1], resolution[0]), 
                               np.linspace(h_range[0], h_range[1], resolution[1])))

    hmap = {'x':0, 'y':1, 'z':2}
    def h_vec_map(h):
        arr = [0., 0., 0.]
        arr[hmap[hdir]] = h
        return arr

    def worker_function(i):
        m0, h = parameters[i]
        h_vec = h_vec_map(h)
        gap = compute_gap(Lattice, m0, h_vec, msub=-1)
        return [m0, h, gap]

    with tqdm_joblib(tqdm(total=len(parameters), desc=f"hi")) as progress_bar:
        data = np.array(Parallel(n_jobs=-1)(delayed(worker_function)(i) for i in range(len(parameters)))).T

    M0, H, GAP = data
    return [M0, H, np.flipud(GAP.reshape(resolution).T)]




def find_defect_points(Lattice, m0, h_vector, hsub):
    PristineLat = DefectLattice(Lattice.Lx, Lattice.Ly, 'none', Lattice.pbc)
    pristine_hamiltonian = compute_hamiltonian(PristineLat, m0, h_vector, 1.0, 1.0)
    defect_hamiltonian = compute_hamiltonian(Lattice, m0, h_vector, 1.0, 1.0, hsub)

    pristine_eigenvalues = spla.eig(pristine_hamiltonian, left=False, right=False, overwrite_a=True)
    sort_idxs = np.argsort(pristine_eigenvalues.real) # type: ignore
    pristine_eigenvalues = pristine_eigenvalues[sort_idxs]

    eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(defect_hamiltonian, left=True, right=True, overwrite_a=True) # type: ignore
    sort_idxs = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sort_idxs]
    left_eigenvectors = left_eigenvectors[:, sort_idxs]
    right_eigenvectors = right_eigenvectors[:, sort_idxs]


    tol = 1e-1

    result = np.array([
        z for z in eigenvalues
        if not np.any(np.abs(pristine_eigenvalues - z) < tol)
    ])
    plt.scatter(result.real, result.imag)
    plt.show()



# region old_chern.py
def compute_d_vector(kx:np.ndarray, ky:np.ndarray, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float) -> np.ndarray:
    """
    Compute the d-vector components for a non-Hermitian system.

    This function calculates the three components of the d-vector used in the 
    Hamiltonian of a topological insulator model with non-Hermitian terms.

    Parameters
    ----------
    kx : ndarray
        Wave vector component along the x-direction with shape (N, )
    ky : ndarray
        Wave vector component along the y-direction with shape (N, )
    m0 : float
        Mass term parameter
    h_vector : ndarray
        Non-Hermitian perturbation terms [h_x, h_y, h_z] with shape (3, )
    t : float
        Hopping parameter for the sine terms in d1 and d2.
    t0 : float
        Hopping parameter for the cosine terms in d3.
    a : float
        Lattice constant.

    Returns
    -------
    d_vector : ndarray
        Array of three complex components [d1, d2, d3] representing the 
        d-vector of the Hamiltonian. Has shape (3, Nx, Ny)
    """
    h_vector = np.array(h_vector)
    assert isinstance(m0, float), "m0 must be a float"
    assert h_vector.size == 3, "h_vector must have size (3, ) or (3, 1)"

    d1 = t * np.sin(kx[:, np.newaxis] * a) + 1j * h_vector[0] + 0 * ky[np.newaxis, :]
    d2 = t * np.sin(ky[np.newaxis, :] * a) + 1j * h_vector[1] + 0 * kx[:, np.newaxis]
    d3 = m0 + t0 * (np.cos(kx[:, np.newaxis] * a) + np.cos(ky[np.newaxis, :] * a)) + 1j * h_vector[2]
    return np.array([d1, d2, d3])

def compute_d_vector_scalar(kx:np.ndarray, ky:np.ndarray, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float) -> np.ndarray:

    h_vector = np.array(h_vector)
    assert isinstance(m0, float), "m0 must be a float"
    assert h_vector.size == 3, "h_vector must have size (3, ) or (3, 1)"

    d1 = t * np.sin(kx * a) + 1j * h_vector[0]
    d2 = t * np.sin(ky * a) + 1j * h_vector[1]
    d3 = m0 + t0 * (np.cos(kx * a) + np.cos(ky * a)) + 1j * h_vector[2]
    return np.array([d1, d2, d3])


def compute_hamiltonian(d_vector:np.ndarray) -> np.ndarray:
    """
    Compute the Hamiltonians for a non-Hermitian system.

    Parameters
    ----------
    d_vector : ndarray
        Hopping vector with shape (3, Nx, Ny)
    Returns
    -------
    hamiltonians : ndarray
        Array of momentum-space Hamiltonians with shape (2, 2, Nx, Ny)
    """
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    tau = np.swapaxes(np.array((pauli_x, pauli_y, pauli_z)), 0, -1) # Shape (2, 2, 3)

    hamiltonians =  np.einsum('ijk,klm->ijlm', tau, d_vector) # Shape (2, 2, Nx, Ny)
    return hamiltonians


def compute_eigenvectors(hamiltonians:np.ndarray, band_index:int=0):
    """
    Compute the eigenvalues, left and right eigenvectors for an array of momentum-space Hamiltonians
    Parameters
    ----------

    Returns
    -------
    eigenvalues : ndarray
        Array of shape (2, Nx, Ny) containing the eigenvalues of each Hamiltonian
    left_eigenvectors : ndarray
        Array of shape (2, 2, Nx, Ny) containing the left_eigenvectors of each Hamiltonian
    right_eigenvectors : ndarray
        Array of shape (2, 2, Nx, Ny) containing the right_eigenvectors of each Hamiltonian
    """
    Nx, Ny = hamiltonians.shape[-2], hamiltonians.shape[-1]
    idxs = np.indices((Nx, Ny))
    idx_i, idx_j = idxs[0].flatten(), idxs[1].flatten()

    eigenvalues = np.full((2, Nx, Ny), np.nan, dtype=complex)
    left_eigenvectors = np.full(hamiltonians.shape, np.nan, dtype=complex)
    right_eigenvectors = np.full(hamiltonians.shape, np.nan, dtype=complex)
    
    for i, j in zip(idx_i, idx_j):
        if False:
            eigs, eigvecs = spla.eigh(hamiltonians[:, :, i, j])
            eigenvalues[:, i, j], left_eigenvectors[:, :, i, j], right_eigenvectors[:, :, i, j] = eigs, eigvecs, eigvecs
        else:
            eigenvalues[:, i, j], left_eigenvectors[:, :, i, j], right_eigenvectors[:, :, i, j] = spla.eig(hamiltonians[:, :, i, j], left=True, right=True)


    return {"eigenvalues": eigenvalues[band_index, :, :],
            "left_eigenvectors": left_eigenvectors[:, band_index, :, :], 
            "right_eigenvectors": right_eigenvectors[:, band_index, :, :]}


def compute_eigenvectors_from_momentum(kx, ky, m0, h_vector, t, t0, a, band_index):
    d_vector = compute_d_vector(kx, ky, m0, h_vector, t, t0, a)
    hamiltonians = compute_hamiltonian(d_vector)
    return compute_eigenvectors(hamiltonians, band_index)


def compute_u1_link_variable(kx:np.ndarray, ky:np.ndarray, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float, direction:str, band_index:int = 0):
    """
    Parameters
    ----------


    Returns
    -------
    u_lower : ndarray
        U(1) link variable for the lower eigenvector with shape (Nx, Ny)
    u_upper : ndarray
        U(1) link variable for the upper eigenvector with shape (Nx, Ny)
    """
    dkx = 2 * np.pi / len(kx)
    dky = 2 * np.pi / len(ky)

    _, left_eigenvectors, right_eigenvectors = compute_eigenvectors_from_momentum(kx, ky, m0, h_vector, t, t0, a, band_index).values()

    if direction == 'none':
        _ = left_eigenvectors
        shifted_right_eigenvectors = right_eigenvectors
    elif direction == 'x':
        _, _, shifted_right_eigenvectors = compute_eigenvectors_from_momentum(kx + dkx, ky, m0, h_vector, t, t0, a, band_index).values()
    elif direction == 'y':
        _, _, shifted_right_eigenvectors = compute_eigenvectors_from_momentum(kx, ky + dky, m0, h_vector, t, t0, a, band_index).values()

    product = np.einsum('ijk,ijk->jk', left_eigenvectors.conj(), shifted_right_eigenvectors)
    u = product / np.abs(product)
    return u

def compute_lattice_field_strength(kx:np.ndarray, ky:np.ndarray, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float, band_index:int=0):
    """
    
    """
    dkx = 2 * np.pi / kx.size
    dky = 2 * np.pi / ky.size

    term1 = compute_u1_link_variable(kx, ky, m0, h_vector, t, t0, a, 'x', band_index)
    term2 = compute_u1_link_variable(kx + dkx, ky, m0, h_vector, t, t0, a, 'y', band_index)
    term3 = compute_u1_link_variable(kx, ky + dky, m0, h_vector, t, t0, a, 'x', band_index).conj()
    term4 = compute_u1_link_variable(kx, ky, m0, h_vector, t, t0, a, 'y', band_index).conj()

    field_strength = np.angle(term1 * term2 * term3 * term4)
    chern = np.sum(field_strength) / (2 * np.pi)
    return chern.real


def compute_chern_number(m0, h_vector, t:float=1.0, t0:float=1.0, a:float=1.0, Nx:int=25, Ny:int=25):
    kx = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, Ny, endpoint=False)
    chern = compute_lattice_field_strength(kx, ky, m0, h_vector, t, t0, a, band_index=1)
    return chern


def compute_chern_phase_diagram(output_file:str, resolution = (51, 51)):
    m0_values = np.linspace(-2.0, 2.0, resolution[0])
    h_values = np.linspace(-1.0, 1.0, resolution[1])

    parameters = tuple(product(m0_values, h_values))

    def worker(i):
        m0, h = parameters[i]
        chern = compute_FHS_chern_fast(m0, [0., 0., h])
        return [m0, h, chern]
    
    with tqdm_joblib(tqdm(total=len(parameters), desc=f"Computing phase diagram for Chern number.")) as progress_bar:
        m0_data, h_data, chern_data = np.array(Parallel(n_jobs=-1)(delayed(worker)(i) for i in range(len(parameters))), dtype=float).T

    with h5py.File(output_file, "w") as f:
        f.create_dataset(name = "m0", data=m0_data)
        f.create_dataset(name = "h", data=h_data)
        f.create_dataset(name =  "chern", data=chern_data.reshape(resolution).T)
    return output_file


if __name__ == "__main__":
    f = compute_chern_phase_diagram('temp.h5', (15,15))

    with h5py.File(f, 'r') as f:
        m0_data = f['m0'][:]
        h_data = f['h'][:]
        chern_data = f['chern'][:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_phase_diagram(fig, ax, m0_data, h_data, chern_data)
    plt.show()


# endregion



# region old_chern_2.py
def compute_d_vector(kx, ky, m0, h_vector, t, t0, a):
    d1 = t * np.sin(kx * a) + 1j * h_vector[0]
    d2 = t * np.sin(ky * a) + 1j * h_vector[1]
    d3 = m0 + t0 * np.cos(kx * a) + np.cos(ky * a) + 1j * h_vector[2]
    return [d1, d2, d3]

def compute_hamiltonian(kx, ky, m0, h_vector, t, t0, a):
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    d_vector = compute_d_vector(kx, ky, m0, h_vector, t, t0, a)
    return pauli_x * d_vector[0] + pauli_y * d_vector[1] + pauli_z * d_vector[2]

def compute_fhs_chern_number(m0, h_vector, Nx = 25, Ny = 25):
    kx_values = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    ky_values = np.linspace(-np.pi, np.pi, Ny, endpoint=False)
    dkx = 2 * np.pi / Nx
    dky = 2 * np.pi / Ny

    def compute_eigenvectors(kx_, ky_, band_index):
            hamiltonian = compute_hamiltonian(kx_, ky_, m0, h_vector, 1.0, 1.0, 1.0)
            eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True)
            
            # --- FIX: Sort eigenvalues and vectors by the real part of eigenvalues ---
            idx = np.argsort(eigenvalues.real)
            eigenvalues = eigenvalues[idx]
            left_eigenvectors = left_eigenvectors[:, idx]
            right_eigenvectors = right_eigenvectors[:, idx]
            # -----------------------------------------------------------------------

            return left_eigenvectors[:, band_index], right_eigenvectors[:, band_index]
    
    def compute_u1_link_variable(kx, ky, direction:str, band_index):
        left_eigenvector, _ = compute_eigenvectors(kx, ky, band_index)
        if direction == 'none':
            pass
        elif direction == 'x':
            _, shifted_right_eigenvector = compute_eigenvectors(kx + dkx, ky, band_index)
        elif direction == 'y':
            _, shifted_right_eigenvector = compute_eigenvectors(kx, ky + dky, band_index)

        product = np.dot(left_eigenvector.conj(), shifted_right_eigenvector)
        return product / np.abs(product)
    
    def compute_field_strength(kx_, ky_, band_index):
        term1 = compute_u1_link_variable(kx_, ky_, 'x', band_index)
        term2 = compute_u1_link_variable(kx_ + dkx, ky_, 'y', band_index)
        term3 = compute_u1_link_variable(kx_, ky_ + dky, 'x', band_index).conj()
        term4 = compute_u1_link_variable(kx_, ky_, 'y', band_index).conj()
        return np.log(term1 * term2 * term3 * term4)
    
    fs_values = []
    for kx in kx_values:
        for ky in ky_values:
            fs_values.append(compute_field_strength(kx, ky, 0))

    return (sum(fs_values) / (2 * np.pi * 1j)).real


def compute_chern_phase_diagram(m0_range, h_range, h_type,
                                output_file=None, directory='', overwrite=False, resolution=(25, 25)):
    m0_values = np.linspace(m0_range[0], m0_range[1], resolution[0])
    h_values = np.linspace(h_range[0], h_range[1], resolution[1])
    parameter_values = tuple(product(m0_values, h_values))

    if output_file is None:
        root_fname = 'square'
        output_file = os.path.join(directory, root_fname+f"_chern_phase_diagram_{resolution[0]}x{resolution[1]}.h5")
    else:
        output_file = os.path.join(directory, output_file)

    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Use overwrite=True to overwrite.")
        return output_file

    def compute_single(i):
        m0, h = parameter_values[i]
        match h_type:
            case 'x':
                h_vector = [h, 0.0, 0.0]
            case 'y':
                h_vector = [0.0, h, 0.0]
            case 'z':
                h_vector = [0.0, 0.0, h]
        chern = compute_fhs_chern_number(m0, h_vector)
        #chern2 = compute_chern_number2(m0, h, 1.0, n=n)
        return [m0, h, chern] #+ [chern2]

    with tqdm_joblib(tqdm(total=len(parameter_values), desc=f"Computing phase diagram for Chern number.")) as progress_bar:
        m0_data, h_data, chern_data = np.array(Parallel(n_jobs=-2)(delayed(compute_single)(i) for i in range(len(parameter_values))), dtype=float).T

    with h5py.File(output_file, "w") as f:
        f.create_dataset(name = "m0", data=m0_data)
        f.create_dataset(name = "h", data=h_data)
        f.create_dataset(name =  "chern", data=chern_data.reshape(resolution).T)
    return output_file



if __name__ == "__main__":
    output_file = compute_chern_phase_diagram((-4.0, 4.0), (-1.0, 1.0), 'x', overwrite=True, output_file='temp.h5', directory='./Non-Hermitian/Data/')
    
    with h5py.File(output_file, 'r') as f:
        m0 = f["m0"][:]
        h = f["h"][:]
        chern = f["chern"][:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    from nonhermitian_chern import plot_phase_diagram

    plot_phase_diagram(fig, ax, m0, h, chern)
    plt.show()


# endregion



# region nonhermitian_bott.py
def compute_lattice(Lx, Ly):
    return np.arange(Lx*Ly).reshape(Ly, Lx)

def compute_distances(lattice, pbc):
    """
    Compute the distances between all pairs of sites in the lattice.
    """
    # Displacement matrices. dx[i, j] is the x-displacement between site i and site j.
    Y, X = np.where(lattice >= 0)
    Ly, Lx = lattice.shape
    dx = X - X[:, None]
    dy = Y - Y[:, None]

    if pbc:
        # Apply periodic boundary conditions
        multipliers = tuple(product([-1, 0, 1], repeat=2))
        shifts = [(i * Lx, j * Ly) for i, j in multipliers]

        x_shifted = np.empty((dx.shape[0], dx.shape[1], len(shifts)), dtype=dx.dtype)
        y_shifted = np.empty((dy.shape[0], dy.shape[1], len(shifts)), dtype=dy.dtype)
        for i, (dx_shift, dy_shift) in enumerate(shifts):
            x_shifted[:, :, i] = dx + dx_shift
            y_shifted[:, :, i] = dy + dy_shift

        distances = x_shifted**2 + y_shifted**2
        minimal_hop = np.argmin(distances, axis = -1)
        i_idxs, j_idxs = np.indices(minimal_hop.shape)

        dx = x_shifted[i_idxs, j_idxs, minimal_hop]
        dy = y_shifted[i_idxs, j_idxs, minimal_hop]
    return dx, dy

def compute_wannier(dx, dy):
    hop_xp = ((dx == 1) & (dy == 0))
    hop_yp = ((dx == 0) & (dy == 1))
    NN_pos = hop_xp | hop_yp
    Cx_plus_Cy = NN_pos * 1/2
    Sx = hop_xp * 1j/2
    Sy = hop_yp * 1j/2
    Cx_plus_Cy += Cx_plus_Cy.conj().T
    Sx += Sx.conj().T
    Sy += Sy.conj().T
    return {"Cx_plus_Cy":Cx_plus_Cy, "Sx":Sx, "Sy":Sy, "I":np.eye(dx.shape[0])}

def compute_wannier_polar(dx, dy):
    """Compute the Wannier polar matrices based on the displacements. While the construction is only necessary for defects containing an interstitial defect (interstitial, frenkek_pair), 
    it is computed for all defect types for consistency. Typical behavior is, of course, recovered in the case of no interstitial defect."""
    theta = np.arctan2(dy, dx)  
    dr = np.sqrt(dx ** 2 + dy ** 2)

    # Create masks for different types of hopping. 
    distance_mask = ((dr <= 1 + 1e-6) & (dr > 1e-6)) # Mask for distances close to 1
    principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & distance_mask 
    diagonal_mask  = ((np.isclose(np.abs(dx), np.abs(dy), atol=1e-4)) & (dx != 0)) & distance_mask
    hopping_mask = principal_mask | diagonal_mask

    # Compute the Wannier matrices based on the masks
    d_cos = np.where(hopping_mask, np.cos(theta), 0. + 0.j)
    d_sin = np.where(hopping_mask, np.sin(theta), 0. + 0.j)
    amplitude = np.where(hopping_mask, np.exp(1. - dr), 0. + 0.j)

    # Momentum space matrices constructed from the real-space displacements based on arxiv.org/abs/2407.13767
    Cx_plus_Cy = amplitude / 2
    Sx = 1j * d_cos * amplitude / 2
    Sy = 1j * d_sin * amplitude / 2
    return {"Cx_plus_Cy":Cx_plus_Cy, "Sx":Sx, "Sy":Sy, "I":np.eye(dx.shape[0])}


def compute_hamiltonian(wannier_matrices:dict, m0:float, h_vector:np.ndarray, t:float = 1.0, t0:float = 1.0):
    # Pauli matrices for Hamiltonian computation
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)     
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    Cx_plus_Cy, Sx, Sy, I = wannier_matrices["Cx_plus_Cy"], wannier_matrices["Sx"], wannier_matrices["Sy"], wannier_matrices["I"]
    
    # Hopping vector terms
    d1 = t * Sx
    d2 = t * Sy
    d3 = m0 * I + t0 * Cx_plus_Cy

    hamiltonian = np.kron(d1, pauli_x) + np.kron(d2, pauli_y) + np.kron(d3, pauli_z)

    # Non hermitian coupling
    hx, hy, hz = h_vector
    H_NH_x = np.kron(1j * hx * I, pauli_x)
    H_NH_y = np.kron(1j * hy * I, pauli_y)
    H_NH_z = np.kron(1j * hz * I, pauli_z)
    H_NH = H_NH_x + H_NH_y + H_NH_z 

    return hamiltonian + H_NH

def compute_projector(hamiltonian:np.ndarray):
    """Compute the projector onto the lower band of the Hamiltonian."""
    eigenvalues, eigenvectors = spla.eigh(hamiltonian, overwrite_a=True)
    lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2] # Lower band eigenvalues
    highest_lower_band = lower_band[-1] # Highest eigenvalue in the lower band

    D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j) # Projector diagonal matrix
    D_herm_conj = np.einsum('i,ij->ij', D, eigenvectors.conj().T)
    projector = eigenvectors @ D_herm_conj # Projector matrix
    return projector

def compute_bott_index(lattice:np.ndarray, projector:np.ndarray):
    """Compute the Bott index for the given projector."""
    Y, X = np.where(lattice >= 0)
    # Repeated (two orbitals)
    X = np.repeat(X, 2)
    Y = np.repeat(Y, 2)
    Lx = np.max(X) - np.min(X) # length of the x-direction
    Ly = np.max(Y) - np.min(Y)

    x_unitary = np.exp(1j * 2 * np.pi * X / Lx) # unitary operator in the x-direction
    y_unitary = np.exp(1j * 2 * np.pi * Y / Ly)
    x_unitary_proj = np.einsum('i,ij->ij', x_unitary, projector) # projector in the x-direction
    y_unitary_proj = np.einsum('i,ij->ij', y_unitary, projector)
    x_unitary_dagger_proj = np.einsum('i,ij->ij', x_unitary.conj(), projector)  # projector in the x-direction (dagger)
    y_unitary_dagger_proj = np.einsum('i,ij->ij', y_unitary.conj(), projector)

    I = np.eye(projector.shape[0], dtype=np.complex128) 
    A = I - projector + projector @ x_unitary_proj @ y_unitary_proj @ x_unitary_dagger_proj @ y_unitary_dagger_proj # BI operator given in arxiv:2407.13767 [Eq. (5)]
    bott_index = np.imag(np.sum(np.log(spla.eigvals(A)))) / (2 * np.pi)
    return bott_index


def plot_phase_diagram(fig, ax, 
                       X_values, Y_values, Z_values, 
                       labels:list=None, title:str=None, 
                       X_ticks=None, Y_ticks=None, X_tick_labels=None, Y_tick_labels=None,
                       cmap='Spectral', plotColorbar=True, doDiscreteColormap=True):
    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]
    Z_values = np.where(Z_values == -0, 0, Z_values)

    not_nan_mask = ~np.isnan(Z_values)
    unique_values = np.sort(np.unique(Z_values[not_nan_mask]).astype(int))
    #unique_values = np.arange(-3, 4, 1)

    if doDiscreteColormap:
        if len(unique_values) < 25:
            cmap = plt.get_cmap(cmap)
            discrete_colors = cmap(np.linspace(0, 1, len(unique_values)))
            cmap = ListedColormap(discrete_colors)
            norm = BoundaryNorm(boundaries=np.append(unique_values, unique_values[-1] + 1), ncolors=len(unique_values))

    im = ax.imshow(Z_values, extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]], 
                   origin='lower', aspect='auto', cmap=cmap, interpolation='none', 
                   rasterized=True, norm=norm)
    
    if title is not None:
        ax.set_title(title)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1], rotation=0)

    if X_ticks is not None:
        ax.set_xticks(X_ticks)
    if Y_ticks is not None:
        ax.set_yticks(Y_ticks)
    if X_tick_labels is not None:
        ax.set_xticklabels(X_tick_labels)
    if Y_tick_labels is not None:
        ax.set_yticklabels(Y_tick_labels)

    if plotColorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_ticks(unique_values+0.5)
        cbar.set_ticklabels([str(val) for val in unique_values], fontsize=16)

    return fig, ax


if __name__ == "__main__":

    resolution = (51, 51)
    m0_values = np.linspace(-2.0, 2.0, resolution[0])
    h_values = np.linspace(-1.0, 1.0, resolution[1])
    h_type = 'x'
    filename = f"NH_bott_{h_type}_{resolution[0]}x{resolution[1]}.h5"

    if not os.path.exists(filename):
        parameters = tuple(product(m0_values, h_values))
        lattice = compute_lattice(15, 15)
        dx, dy = compute_distances(lattice, False)
        wannier_matrices = compute_wannier(dx, dy)
        def worker(i):
            m0, h = parameters[i]

            match h_type:
                case 'x':
                    h_vector = [h, 0., 0.]
                case 'y':
                    h_vector = [0., h, 0.]
                case 'z':
                    h_vector = [0., 0., h]
            hamiltonian = compute_hamiltonian(wannier_matrices, m0, h_vector)
            projector = compute_projector(hamiltonian)
            bott_index = compute_bott_index(lattice, projector)
            return [m0, h, bott_index]

        with tqdm_joblib(tqdm(total=len(parameters), desc="Computing m0 vs h_i phase diagram for the Bott Index")) as progress_bar:
            data = np.array(Parallel(n_jobs=-1)(delayed(worker)(i) for i in range(len(parameters)))).T
        m0 = data[0]
        h = data[1]
        bott_index = np.round(data[2].reshape(resolution).T, 3)
        with h5py.File(filename, 'w') as f:
            f.create_dataset(name = 'm0', data = m0)
            f.create_dataset(name = 'h', data = h)
            f.create_dataset(name = 'bott_index', data = bott_index)

    with h5py.File(filename, 'r') as f:
        m0 = f['m0'][:]
        h = f['h'][:]
        bott_index = f['bott_index'][:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_phase_diagram(fig, ax, m0, h, bott_index)

    x_line = np.linspace(-1.0, 1.0)
    ax.plot(x_line, x_line, ls='--', c='k')
    ax.plot(x_line, -x_line, ls='--', c='k')
    x_line2 = np.linspace(1.0, 2.0)
    ax.plot(x_line2, x_line2 - 2, ls='--', c='k')
    ax.plot(x_line2, 2 - x_line2, ls='--', c='k')
    x_line3 = np.linspace(-2.0, -1.0)
    ax.plot(x_line3, x_line3 + 2, ls='--', c='k')
    ax.plot(x_line3, - 2 - x_line3, ls='--', c='k')
    plt.show()
# endregion



# region chern.py


# region Non-Hermitian d-vector
def compute_d_vector(kx:float, ky:float, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float):
    d1 = t * np.sin(kx * a) + 1.0j * h_vector[0]
    d2 = t * np.sin(ky * a) + 1.0j * h_vector[1]
    d3 = m0 + t0 * np.cos(kx * a) + t0 * np.cos(ky * a) + 1.0j * h_vector[2]
    vector = np.array((d1, d2, d3))
    return vector


def compute_d_vector_conj(kx:float, ky:float, m0:float, h_vector:np.ndarray, t:float, t0:float, a:float):
    d1 = t * np.sin(kx * a) + 1.0j * h_vector[0]
    d2 = t * np.sin(ky * a) + 1.0j * h_vector[1]
    d3 = m0 + t0 * np.cos(kx * a) + t0 * np.cos(ky * a) + 1.0j * h_vector[2]
    return np.array((d1, d2, d3)).conj()

# endregion
# region Chern Number computation

def compute_normalized_vector(vector_generating_function:callable, kx:float, ky:float, vector_kwargs:dict=None, returnNorm:bool=False):
    vector = vector_generating_function(kx, ky, **vector_kwargs)
    norm = spla.norm(vector, axis=0)
    norm = np.where(norm == 0, 1, norm)
    if returnNorm:
        return vector / norm, norm
    return vector / norm


def compute_vector_finite_derivative(vector_generating_function:callable, kx:float, ky:float, direction:str, vector_kwargs:dict=None, dk:float=1e-5):
    if direction == 'x':
        d_v_dk = vector_generating_function(kx + dk, ky, **vector_kwargs) - vector_generating_function(kx - dk, ky, **vector_kwargs)
    elif direction == 'y':
        d_v_dk = vector_generating_function(kx, ky + dk, **vector_kwargs) - vector_generating_function(kx, ky - dk, **vector_kwargs)
    else:
        raise ValueError("Direction must be either 'x' or 'y'")
    return d_v_dk / (2 * dk)


def compute_berry_curvature(vector_generating_function:callable, kx:float, ky:float, vector_kwargs:dict):
    def conjugate_vector_generating_function(kx, ky, **vector_kwargs):
        return np.conj(vector_generating_function(kx, ky, **vector_kwargs))
    v = vector_generating_function(kx, ky, **vector_kwargs)
    vconj = np.conj(v)
    v_hat, norm = compute_normalized_vector(vector_generating_function, kx, ky, vector_kwargs, returnNorm=True)
    d_v_dkx = compute_vector_finite_derivative(vector_generating_function, kx, ky, 'x', vector_kwargs)
    d_v_dky = compute_vector_finite_derivative(vector_generating_function, kx, ky, 'y', vector_kwargs)

    d_vconj_dkx = compute_vector_finite_derivative(conjugate_vector_generating_function, kx, ky, 'x', vector_kwargs)
    d_vconj_dky = compute_vector_finite_derivative(conjugate_vector_generating_function, kx, ky, 'y', vector_kwargs)

    d_vhat_dkx = d_v_dkx / norm - v_hat / (2 * norm ** 2) * (np.dot(d_vconj_dkx, v) + np.dot(vconj, d_v_dkx))
    d_vhat_dky = d_v_dky / norm - v_hat / (2 * norm ** 2) * (np.dot(d_vconj_dky, v) + np.dot(vconj, d_v_dky))

    cross_product = np.cross(d_vhat_dkx, d_vhat_dky)
    berry_curvature = np.dot(v_hat, cross_product) / 2
    return berry_curvature


def compute_chern_number(vector_generating_function:callable, vector_kwargs:dict, brillouin_zone_resolution:int=101,
                         returnGapData:bool = True):
    kx_values = ky_values = np.linspace(-np.pi, np.pi, brillouin_zone_resolution, endpoint=False)
    kx_values, ky_values = np.meshgrid(kx_values, ky_values)
    kx_values, ky_values = kx_values.flatten(), ky_values.flatten()

    if returnGapData:
        min_real_gap = min_imag_gap = min_mag_gap = float('inf')
    berry_curvatures = []
    for kx, ky in zip(kx_values, ky_values):

        if returnGapData:
            v = vector_generating_function(kx, ky, **vector_kwargs)
            gap = np.sqrt(np.dot(v,v))
            real_gap = gap.real
            imag_gap = gap.imag
            mag_gap = spla.norm(v)
            if real_gap < min_real_gap:
                min_real_gap = real_gap
            if imag_gap < min_imag_gap:
                min_imag_gap = imag_gap
            if mag_gap < min_mag_gap:
                min_mag_gap = mag_gap

        bc = compute_berry_curvature(vector_generating_function, kx, ky, vector_kwargs)
        berry_curvatures.append(bc)

    dkx = 2 * np.pi / brillouin_zone_resolution
    chern_number = np.real(np.sum(berry_curvatures) * dkx * dkx / (2 * np.pi))
    if returnGapData:
        return chern_number, min_real_gap, min_imag_gap, min_mag_gap
    return chern_number


def plot_energy_in_complex_plane(m0, h_vector, t, t0):
    def compute_energies(kx, ky, m0, h_vector, t, t0):
        term1 = (t * np.sin(kx)) ** 2 - h_vector[0] ** 2 + 2 * 1.0j * h_vector[0] * (t * np.sin(kx))
        term2 = (t * np.sin(ky)) ** 2 - h_vector[1] ** 2 + 2 * 1.0j * h_vector[1] * (t * np.sin(ky))
        term3 = (m0 + t0 * (np.cos(kx) + np.cos(ky))) ** 2 - h_vector[2] ** 2 + 2 * 1.0j * h_vector[2] * (m0 + t0 * (np.cos(kx) + np.cos(ky)))
        return np.sqrt(term1 + term2 + term3)

    kx_values = ky_values = np.linspace(-np.pi, np.pi, 101, endpoint=False)
    kx_values, ky_values = np.meshgrid(kx_values, ky_values)
    energies = compute_energies(kx_values, ky_values, m0, h_vector, t, t0)
    energies = np.concatenate((energies, -energies))

    real_part = energies.flatten().real
    imaginary_part = energies.flatten().imag

    plt.scatter(real_part, imaginary_part)
    plt.show()



# endregion    
# region General Functions

def compute_phase_diagram_parallel(worker_function:callable, parameter_values, filename:str, 
                                   overwrite:bool=False, dataset_labels:list[str]=None):
    filename_base = filename.split('.')[0]
    filename = filename_base + '.h5'


    if os.path.exists(filename) and not overwrite:
        print(f"Filename '{filename}' already exists.")
        return filename

    with tqdm_joblib(tqdm(total=len(parameter_values))) as progress_bar:
        computed_data = Parallel(n_jobs=-1)(delayed(worker_function)(*params) for params in parameter_values)
    computed_data = np.array(computed_data).T

    if (dataset_labels == None) or (len(dataset_labels) != computed_data.shape[0]):
        dataset_labels = [f"dataset_{i}" for i in range(len(computed_data.shape[0]))]

    with h5py.File(filename, 'w') as f:
        for label, dataset in zip(dataset_labels, computed_data):
            f.create_dataset(name=label, data=dataset)

    return filename


def get_data_from_h5_file(filename:str, dataset_labels:list[str]=None):
    with h5py.File(filename, 'r') as f:
        if dataset_labels != None:
            data = []
            good_labels = []
            for label in dataset_labels:
                try:
                    data.append(f[label][:])
                    good_labels.append(label)
                except:
                    print(f"Label '{label}' not in file '{filename}'")
        else:
            data = [d[:] for d in f.values()]
            good_labels = list(f.keys())
    return data, good_labels


def plot_phase_diagram(fig, ax, 
                       X_values, Y_values, Z_values, 
                       labels:list=None, title:str=None, 
                       X_ticks=None, Y_ticks=None, X_tick_labels=None, Y_tick_labels=None,
                       cmap='Spectral', plotColorbar=True, doDiscreteColormap=True):
    X_range = [np.min(X_values), np.max(X_values)]
    Y_range = [np.min(Y_values), np.max(Y_values)]
    Z_values = np.where(Z_values == -0, 0, Z_values)

    if doDiscreteColormap:
        not_nan_mask = ~np.isnan(Z_values)
        unique_values = np.sort(np.unique(Z_values[not_nan_mask]).astype(int))
        cmap = plt.get_cmap(cmap)
        discrete_colors = cmap(np.linspace(0, 1, len(unique_values)))
        cmap = ListedColormap(discrete_colors)
        norm = BoundaryNorm(boundaries=np.append(unique_values, unique_values[-1] + 1), ncolors=len(unique_values))
    else:
        cmap = plt.get_cmap(cmap)
        norm = None

    im = ax.imshow(Z_values, extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]], 
                   origin='lower', aspect='auto', cmap=cmap, interpolation='none', 
                   rasterized=True, norm=norm)
    
    if title is not None:
        ax.set_title(title)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1], rotation=0)

    if X_ticks is not None:
        ax.set_xticks(X_ticks)
    if Y_ticks is not None:
        ax.set_yticks(Y_ticks)
    if X_tick_labels is not None:
        ax.set_xticklabels(X_tick_labels)
    if Y_tick_labels is not None:
        ax.set_yticklabels(Y_tick_labels)

    if plotColorbar:
        cbar = fig.colorbar(im, ax=ax)
        if doDiscreteColormap:
            cbar.set_ticks(unique_values+0.5)
            cbar.set_ticklabels([str(val) for val in unique_values], fontsize=16)

    return fig, ax


def profile_wrapper(func:callable, *args, **kwargs):
    profiler = Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    return result
    

# endregion
# region Chern Phase Diagram Stuff
def plot_chern_and_gaps(filename:str, labels:list[str]):
    data, labels = get_data_from_h5_file(filename, labels)
    m0, h, chern, min_real_gap, min_imag_gap, min_mag_gap = data

    n_unique_m0 = np.unique(m0).size
    n_unique_h = np.unique(h).size
    
    chern = chern.reshape(n_unique_m0, n_unique_h).T
    min_real_gap, min_imag_gap, min_mag_gap = min_real_gap.reshape(n_unique_m0, n_unique_h).T, min_imag_gap.reshape(n_unique_m0, n_unique_h).T, min_mag_gap.reshape(n_unique_m0, n_unique_h).T

    min_imag_gap = np.abs(min_imag_gap)

    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    fig, axes[0, 0] = plot_phase_diagram(fig, axes[0, 0], m0, h, chern, doDiscreteColormap=False)
    fig, axes[0, 1] = plot_phase_diagram(fig, axes[0, 1], m0, h, min_mag_gap, doDiscreteColormap=False, cmap='RdPu')
    fig, axes[1, 0] = plot_phase_diagram(fig, axes[1, 0], m0, h, min_real_gap, doDiscreteColormap=False, cmap='RdPu')
    fig, axes[1, 1] = plot_phase_diagram(fig, axes[1, 1], m0, h, min_imag_gap, doDiscreteColormap=False, cmap='RdPu')
    fig.suptitle(filename)
    axes[0, 0].set_title("Chern Number")
    axes[0, 1].set_title("Minimum Gap $\\sqrt{\\mathbf{d}^\\dagger \\cdot \\mathbf{d}}$")
    axes[1, 0].set_title("Minimum Gap $\\Re \\sqrt{\\mathbf{d} \\cdot \\mathbf{d}}$")
    axes[1, 1].set_title("Minimum Gap $\\Im \\sqrt{\\mathbf{d} \\cdot \\mathbf{d}}$")

    def plot_line(ax, xrange, yrange, slope, intercept):
        t = np.linspace(xrange[0], xrange[1], 101)
        ax.plot(t, slope * t + intercept, ls='--', c='k', lw=1)
        ax.set_ylim(yrange)
        ax.set_xlim(xrange)

    def plot_quadratic(ax, xrange, yrange):
        t = np.linspace(-2.0, 2.0, 101)
        ax.plot(t, np.sqrt(2 * np.abs(t) - t**2), ls='--', c='k', lw=1)
        ax.plot(t, -np.sqrt(2 * np.abs(t) - t**2), ls='--', c='k', lw=1)
        ax.set_ylim(yrange)
        ax.set_xlim(xrange)

    if   filename.find('_hx') + 1:
        h_dir = 'hx'
    elif filename.find('_hy') + 1:
        h_dir = 'hy'
    elif filename.find('_hz') + 1:
        h_dir = 'hz'
    else:
        h_dir = 'NAN'

    for ax in axes.flatten():
        ax.set_xlabel("$m_0$", fontsize=12)
        ax.set_ylabel(f"$h_{h_dir}$", rotation=0, fontsize=12)
        if h_dir in ['hx', 'hy']:
            slopes = [1, -1]
            intercepts = [2, 0, -2]
            values = tuple(product(slopes, intercepts))
            for s, i in values:
                plot_line(ax, (-3.0, 3.0), (-3.0, 3.0), s, i)
        else:
            plot_quadratic(ax, (0.0, 2.0), (0.0, 1.0))
    plt.tight_layout()
    plt.savefig(filename.replace(".h5", ".svg"))


def compute_chern_phase_diagram():
    def worker(m0:float, h:float, h_dir:str, t:float=1.0, t0:float=1.0, a:float=1.0):
        match h_dir:
            case 'x':
                h_vector = [h, 0.0, 0.0]
            case 'y':
                h_vector = [0.0, h, 0.0]
            case 'z':
                h_vector = [0.0, 0.0, h]
            case _:
                raise ValueError(f"h_dir must be in ['x', 'y', 'z']. It is {h_dir}")
        vector_kwargs = {
            'm0': m0,
            'h_vector': h_vector,
            't': t,
            't0': t0,
            'a': a
        }
        chern_number, min_real_gap, min_imag_gap, min_mag_gap = compute_chern_number(compute_d_vector, vector_kwargs)
        return [m0, h, chern_number, min_real_gap, min_imag_gap, min_mag_gap]

    m0_values = np.linspace(0.0, 2.0, 51)
    h_values = np.linspace(0.0, 1.0, 51)
    h_dir_values = ['x'] 
    parameter_values = tuple(product(m0_values, h_values, h_dir_values))

    labels = ['m0', 'h', 'chern', 'min_real_gap', 'min_imag_gap', 'min_mag_gap']
    directory = "NonHermitian/Data/"
    filename = directory+f"chern_h{h_dir_values[0]}.h5"
    filename = compute_phase_diagram_parallel(worker, parameter_values, filename = filename, 
                                              overwrite=False, dataset_labels=labels)

    plot_chern_and_gaps(filename, labels)
    #plt.show()

# endregion

def find_zeros_of_energy_hz():
    def f(m0, hz, kx, ky):
        m0 = m0[:, np.newaxis, np.newaxis, np.newaxis]
        hz = hz[np.newaxis, :, np.newaxis, np.newaxis]
        kx = kx[np.newaxis, np.newaxis, :, np.newaxis]
        ky = ky[np.newaxis, np.newaxis, np.newaxis, :]
        real = m0**2 - hz**2 + 2 + 2 * np.cos(kx) * np.cos(ky) + 2 * m0 * (np.cos(kx) + np.cos(ky))
        imag = 2 * hz * (m0 + np.cos(kx) + np.cos(ky))
        return real, imag
    
    N = 11
    Nk = 15
    m0 = np.linspace(0.0, 2.0, 25)
    hz = np.linspace(0.0, 1.0, 25)
    kx = ky =  np.linspace(0.0, np.pi, Nk)

    labels = ['m0', 'hz', 'kx', 'ky']
    values = (m0, hz, kx, ky)
    
    real, imag = f(m0, hz, kx, ky)
    real_zero = np.isclose(real, 0.0)
    imag_zero = np.isclose(imag, 0.0)
    both_zero = real_zero & imag_zero

    
    zero_idxs = np.argwhere(both_zero)
    print(zero_idxs)
    for zero_idx in zero_idxs:
        for i, idx in enumerate(zero_idx):
            print(labels[i], values[i][idx])
        print('-'*10)



if __name__ == "__main__":
    vector_kwargs = {
        'm0': 1.0,
        'h_vector': [0.0, 0.0, 0.0],
        't': 1.0,
        't0': 1.0,
        'a': 1.0
    }
    #chern_number = compute_chern_number(compute_d_vector, vector_kwargs, returnGapData=False)
    #print(f"Chern number: {chern_number}")

    compute_chern_phase_diagram()

# endregion



# region Plotting


if 0:
    # Axes width parameters
    large_width = 1.
    small_width = large_width * 2 / (1 + np.sqrt(5))
    colorbar_width = 1.
    alpha = 1.
    beta = 1.
    gamma = 1.

    # Axes height parameters
    large_height = 1.0
    small_height = small_width
    delta = large_height - 2 * small_height

    # Normalize the total width to 1.0
    total_width = 2 * (large_width + alpha + small_width + beta + colorbar_width) + gamma
    small_width /= total_width
    large_width /= total_width
    colorbar_width /= total_width
    alpha /= total_width
    beta /= total_width
    gamma /= total_width

    # Normalize height to 1.0
    total_height = large_height


    if 0:
        L = 6
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        Lattice1 = DefectLattice(L, L, 'substitution', True, defect_radius = 1)
        Lattice1.plot(axs[0])
        axs[0].set_title("Defect Radius = 1", fontsize=16)

        Lattice2 = DefectLattice(L, L, 'substitution', True, defect_radius = 2)
        Lattice2.plot(axs[1])
        axs[1].set_title("Defect Radius = 2", fontsize=16)

        Lattice3 = DefectLattice(L, L, 'substitution', True, defect_radius = 2, break_c4=True)
        Lattice3.plot(axs[2])
        axs[2].set_title("Defect Radius = 2, Break C4", fontsize=16)

        for ax in axs:
            ax.set_xticks([1, 5])
            ax.set_yticks([1, 5])

        plt.tight_layout()
        plt.savefig("./NonHermitian/Plots/defect_radius_comparison.png", bbox_inches='tight')

    if 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_ipr('substitution', [20, 24, 28], -1.0, 0.5, 1.5, 'x', ax, 1)
        plt.show()

    if 0:
        L = 20
        fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        m0 = -1.0
        h0 = 0.5
        hsubs = np.linspace(0.5, 3.5, 19)

        colors = colormaps.get_cmap('cividis')(np.linspace(0, 1, len(hsubs)))

        all_iprs = []
        all_eigenvalues = []
        for hsub, color in zip(hsubs, colors):
            Lattice = DefectLattice(L, L, 'substitution', True, defect_radius = 1)
            h0_vector = np.array([h0, 0.0, 0.0])
            hsub_vector = np.array([hsub, 0.0, 0.0])
            eig_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h0_vector, hsub_vector, n_closest_to_zero = 4)
            left_ipr = compute_ipr(eig_dict['left_eigenvectors'])
            all_iprs.append(left_ipr)
            eigenvalues = eig_dict['eigenvalues']
            all_eigenvalues.append(eigenvalues)
            #axs[0].scatter(np.abs(eigenvalues) * np.sign(eigenvalues.real), left_ipr, label=f"$h_{{0x}}^{{\\rm sub}}={hsub:.2f}$", s=25, alpha=0.5, color=color)
            print(f"Computed for hsub={hsub:.2f}")

        mean_iprs = [np.mean(all_iprs[j]) for j in range(len(hsubs))]
        median_iprs = [np.median(all_iprs[j]) for j in range(len(hsubs))]
        max_iprs = [np.max(all_iprs[j]) for j in range(len(hsubs))]
        axs[1].plot(hsubs, mean_iprs, alpha=1.0, label="Mean IPR", color='black', marker='o', ls='--')
        axs[1].plot(hsubs, median_iprs, alpha=1.0, label="Median IPR", color='red', marker='o', ls='--')

        axs[0].plot(hsubs, max_iprs, alpha=1.0, label="Max IPR", color='blue', marker='o', ls='--')
        axs[1].set_xlabel("$h_{0x}^{\\rm sub}$")
        axs[1].set_ylabel("IPR")
        axs[1].legend()
        axs[1].set_xticks(np.arange(np.min(hsubs), np.max(hsubs) + 0.5, 0.5))

        axs[0].set_ylabel("IPR")
        axs[0].legend()

        plt.tight_layout()

        axs[0].spines['bottom'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[0].xaxis.tick_top()
        axs[0].tick_params(labeltop=False)  # don't put tick labels at the top
        axs[1].xaxis.tick_bottom()

        axs[0].axvline(x=1.5, c='k', ls='--', lw=1, alpha=0.5)
        axs[0].axvline(x=2.5, c='k', ls='--', lw=1, alpha=0.5)
        axs[1].axvline(x=1.5, c='k', ls='--', lw=1, alpha=0.5)
        axs[1].axvline(x=2.5, c='k', ls='--', lw=1, alpha=0.5)

        d = 0.5
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                    linestyle = "none", color='k', mec='k', mew=1, clip_on=False)
        
        axs[0].plot([0, 1], [0, 0], transform=axs[0].transAxes, **kwargs)
        axs[1].plot([0, 1], [1, 1], transform=axs[1].transAxes, **kwargs)
        plt.savefig(f"./NonHermitian/Plots/IPR/mean_ipr_vs_hsub_L={L}_h0={h0}_temp.png", bbox_inches='tight')
        plt.show()

    if 0:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

        markers = ['o', 's', '^']

        defect_sizes = [1, 2, 2]
        bc4s = [False, False, True]
        titles = [f"{size} Defect Sites" for size in [1, 5, 7]]
        left_iprs = []

        for i, L in enumerate([20, 30, 40]):
            for j in range(3):
                Lattice = DefectLattice(L, L, 'substitution', True, defect_sizes[j], break_c4 = bc4s[j])
                eig_dict = compute_eigenvectors_eigenvalues(Lattice, -1.0, np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0]), n_closest_to_zero = 4)
                left_ipr = compute_ipr(eig_dict['left_eigenvectors'])
                left_iprs.append(left_ipr)

                eigenvalues = eig_dict['eigenvalues']
                axs[j].scatter(np.real(eigenvalues), np.imag(eigenvalues), c=left_ipr, s=25, zorder=0, marker=markers[i], label=f"$L={L}$", cmap='cividis', alpha=0.5)
                print(f"Computed L={L}, j={j}")


        for j in range(3):
            axs[j].set_title(titles[j], fontsize=16)
            axs[j].set_xlabel("$\\Re(E)$", fontsize=16)
            if j == 0: axs[j].set_ylabel("$\\Im(E)$", fontsize=16)
            plt.colorbar(axs[j].collections[0], ax=axs[j], label='Left IPR')
        

        for ax in axs:
            ax.legend()

        plt.savefig("./NonHermitian/Plots/Substitution/substitution_spectrum_iprs.png", bbox_inches='tight')

    if 0:
        fig, axs = plt.subplots(6, 5, figsize=(30, 36))
        plot_many_spectrum_lr(fig, axs, 'substitution', 20, [-1.0] * 6, 'x', [0.5] * 6, [1.5, 1.25, 1.125, 1.0625, 1.0, 0.95], defect_radius = 1) 
        plt.savefig("./NonHermitian/Plots/temp.png", bbox_inches='tight')
        compare_ipr_vs_radius(Ls=[20, 30, 40], radii=np.array([1, 4, 7, 10]))

    if 0:
        Lattice = DefectLattice(20, 20, 'substitution', True, defect_radius = 1)
        eig_dict = compute_eigenvectors_eigenvalues(Lattice, -1.0, np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0]), n_closest_to_zero = 4)
        ipr = compute_ipr(eig_dict['left_eigenvectors'])

        selected_idxs = eig_dict['selected_idxs']

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].scatter(np.abs(eig_dict['eigenvalues']) * np.sign(eig_dict['eigenvalues'].real), ipr, c='k', s=25, alpha=1.)

        norm = Normalize(vmin=np.min(ipr), vmax=np.max(ipr))
        axs[1].scatter(np.real(eig_dict['eigenvalues']), np.imag(eig_dict['eigenvalues']), c=ipr, s=25, cmap='jet', alpha=1., norm=norm)
        axs[1].scatter(np.real(eig_dict['eigenvalues'])[selected_idxs], np.imag(eig_dict['eigenvalues'])[selected_idxs], s=200, c=ipr[selected_idxs], cmap='jet', alpha=1., marker="*", zorder=-10, norm=norm)
        
        axs[2].scatter(np.real(eig_dict['eigenvalues']), np.imag(eig_dict['eigenvalues']), s=25, color='k', alpha=1.)
        axs[2].scatter(np.real(eig_dict['eigenvalues'])[selected_idxs], np.imag(eig_dict['eigenvalues'])[selected_idxs], s=25, color='r', alpha=1., zorder=10)
        
        plt.tight_layout()

        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        inset_ax_1 = axs[1].inset_axes([0.05, 0.05, 0.35, 0.35])
        inset_ax_1.set_zorder(100)
        inset_ax_2 = axs[2].inset_axes([0.05, 0.05, 0.35, 0.35])
        inset_ax_2.set_zorder(100)
        plt.colorbar(axs[1].collections[0], ax=axs[1], label='Left IPR')


        for ax in [inset_ax_1, inset_ax_2]:
            plot_on_lattice(ax, Lattice, eig_dict['selected_left_eigenvectors'], plot_type = "tripcolor", title = "", tick_fontsize = 10, label_fontsize = 10)
        

        axs[0].set_xlabel("$|E|\\times {\\rm sign}(\\Re(E))$")
        axs[0].set_ylabel("IPR")
        axs[1].set_xlabel("$\\Re(E)$")
        axs[1].set_ylabel("$\\Im(E)$")
        axs[2].set_xlabel("$\\Re(E)$")
        axs[2].set_ylabel("$\\Im(E)$")

        axs[0].set_title("Left Eigenvector IPR")
        axs[1].set_title("Complex Eigenvalue Spectrum")
        axs[2].set_title("Complex Eigenvalue Spectrum")

        plt.savefig("./NonHermitian/Plots/temp.png", bbox_inches='tight')

    # Show the relative concentration of an eigenvector on each X value
    if 0:
        L = 20
        Lattice = DefectLattice(L, L, 'frenkel_pair', True, 1)

        hamiltonian = compute_hamiltonian(Lattice, -1.0, np.array([0.5, 0., 0.]), 1., 1., np.array([1.5, 0., 0.]))

        eigenvalues, left_eigenvectors, right_eigenvectors = spla.eig(hamiltonian, left=True, right=True, overwrite_a=True)

        # Reshape the eigenvectors to fit on the lattice
        X, Y = Lattice.X, Lattice.Y
        x_values, inverse_indices = np.unique(X, return_inverse = True)
        
        eigval_max_x = []
        for eigval_idx in range(eigenvalues.size):
            selected_indices = [eigval_idx]
            eigvecs = np.sum(np.abs(left_eigenvectors[:, selected_indices]) ** 2, axis=1)
            eigvecs = eigvecs[::2] + eigvecs[1::2]

            arr = np.stack((X, Y, eigvecs)).T
            sums = np.bincount(inverse_indices, weights=arr[:, 2])
            count_along_x = np.bincount(inverse_indices)
            means = sums / count_along_x

            max_xpos = x_values[np.argmax(sums)]
            eigval_max_x.append(max_xpos)



        sort_idxs = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[sort_idxs]
        eigval_max_x = np.array(eigval_max_x)[sort_idxs]
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues.real, c=eigval_max_x, cmap='Spectral')
        plt.colorbar()
        plt.show()




# endregion



# plotting.py
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

        plot = ldos_ax.scatter(Lattice.X, Lattice.Y, c=color_array, cmap='Greys', s=200)
        ldos_ax.scatter(Lattice.Lx / 2, Lattice.Ly / 2, facecolor='w', edgecolor='r', s=200, lw=2, ls='--')
        if 0:
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


def spectrum_and_ipr_figure(axes, method, m0, h0_vector, hsub_vector, Ls, ns, 
                            defect_radius:int = 1, break_c4:bool = False, 
                            do_additional_plotting:bool = True):
    Ls = np.sort(Ls)

    ldos_plot_radius = max(Ls[-1] // 4, 5)
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
        fpx = fpy = -schottky_sep - 0.5
        if method == "frenkel_pair": print(f"L={L}: fpx = {fpx}, fpy = {fpy}")

        Lattice = DefectLattice(L, L, method, True, schottky_separation=schottky_sep, 
                                frenkel_x_disp = fpx, frenkel_y_disp= fpy,
                                defect_radius = defect_radius, break_c4 = break_c4)
        
        ed = compute_eigenvectors_eigenvalues(Lattice, m0, h0_vector, hsub_vector, n)
        lattices.append(Lattice)
        eig_dicts.append(ed)

    if method in ["schottky", "frenkel_pair"]: ldos_plot_radius = max(ldos_plot_radius, 4+-int(round(max(abs(fpx), abs(fpy)), 0)), 2 + schottky_sep // 2)

    ldos_plot_radius = min(ldos_plot_radius, Ls[-1] // 2)

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
        ax_a.set_xlabel("$|E| \\times {\\rm sign}[\\Re (E)]$")
        ax_a.set_ylabel("${\\rm IPR} \\times 10^2$")
        ax_a.legend(loc = 'upper right')

    ext_min = max(L // 2 - ldos_plot_radius, 0)
    ext_max = min(L // 2 + ldos_plot_radius, L)
    ldos_extent = (ext_min, ext_max, ext_min, ext_max)

    # Axis (a_inset)
    if do_additional_plotting:
        ax_a_inset, a_inset_cbar = plot_on_lattice(ax_a_inset, Lattice, ed["selected_left_eigenvectors"], 
                                                "tripcolor", "Greys", plot_colorbar=False,
                                                extent = ldos_extent)
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

    b_scat1.set_clim(vmin=0)
    b_cbar = plt.colorbar(b_scat1, ax_b_cb)

    format_colorbar(b_cbar, *b_scat1.get_clim())
    ax_b.set_xlabel("$\\Re (E)$")
    ax_b.set_ylabel("$\\Im (E) \\times 10^1$")

    # Axis (c) and (c_colorbar)
    ax_c, ax_c_cb = plot_on_lattice(ax_c, Lattice, ed["L"], 
                                    "tripcolor", "Greys", plot_colorbar=False,
                                    extent = ldos_extent)
    ax_c.tick_params(axis="both", labelsize=10)
    ax_c.set_xlabel("$X$", fontsize=10)
    ax_c.set_ylabel("$Y$", fontsize=10)
    
# ============================
# ============================
# ============================


def plot_ipr_figs1(method, Ls, ns, hdir):
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

    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_ipr_{hdir}_greys.png", dpi=300, bbox_inches="tight")


def plot_ipr_figs2(method, Ls, ns, hdir):
    fig = plt.figure(figsize=(20, 30))
    plt.subplots_adjust(wspace=0.1)
    master_gs = gridspec.GridSpec(3, 2, figure=fig)

    figure_layout(fig, master_gs[0, 0])
    figure_layout(fig, master_gs[0, 1], False)
    figure_layout(fig, master_gs[1, 0])
    figure_layout(fig, master_gs[1, 1], False)
    figure_layout(fig, master_gs[2, 0])
    figure_layout(fig, master_gs[2, 1], False)
    
    axes = np.array(fig.axes)
    axes_chunks = [axes[:5], axes[5:9], axes[9:14], axes[14:18], axes[18:23], axes[23:]]

    h_dir_mapping = {'x': 0, 'y': 1, 'z': 2}
    v1 = np.zeros(3)
    v2 = np.zeros(3)
    v1[h_dir_mapping[hdir]] = 0.5
    v2[h_dir_mapping[hdir]] = 1.5

    spectrum_and_ipr_figure(axes_chunks[0], method, -1.0, v1, v2, Ls, ns)
    spectrum_and_ipr_figure(axes_chunks[1], method, -1.0, v2, v1, Ls, ns, do_additional_plotting = False)
    spectrum_and_ipr_figure(axes_chunks[2], method, -1.0, v1, v2, Ls, ns, defect_radius = 2)
    spectrum_and_ipr_figure(axes_chunks[3], method, -1.0, v2, v1, Ls, ns, defect_radius = 2, do_additional_plotting = False)
    spectrum_and_ipr_figure(axes_chunks[4], method, -1.0, v1, v2, Ls, ns, defect_radius = 2, break_c4 =  True)
    spectrum_and_ipr_figure(axes_chunks[5], method, -1.0, v2, v1, Ls, ns, defect_radius = 2, break_c4 =  True, do_additional_plotting = False)

    fig_row_labels = ["i", "ii", "iii"]
    for i in range(3):
        for j, ax in enumerate(fig.axes[9*i:9*(i+1)]):
            lab = "(" + fig_row_labels[i] + "." + "ab_c_de_f"[j] + ")"
            if lab[-2] == "_":
                continue
            ax.annotate(
                lab,
                xy = (0.025, 0.975),
                xycoords='axes fraction',
                ha='left',
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
            )

    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_ipr_{hdir}.png", dpi=300, bbox_inches="tight")


def figure_layout2():
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1., 1.5, 1.5])

    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 0])
    ax1 = fig.add_subplot(gs0[0, 0], label="a")
    ax2 = fig.add_subplot(gs0[0, 1], label="b")

    gs1_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], width_ratios=[1, .1])
    ax3 = fig.add_subplot(gs1_a[0, 0], label="c")
    ax3_cb = fig.add_subplot(gs1_a[0, 1], label="c_cb")
    gs1_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], width_ratios=[1, .1])
    ax4 = fig.add_subplot(gs1_b[0, 0], label="d")
    ax4_cb = fig.add_subplot(gs1_b[0, 1], label="d_cb")

    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], width_ratios=[1, .1])
    ax5 = fig.add_subplot(gs2[0, 0], label="e")
    ax5_cb = fig.add_subplot(gs2[0, 1], label="e_cb")
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 2], width_ratios=[1, 1, .1])
    ax6 = fig.add_subplot(gs3[0, 0], label="f")
    ax7 = fig.add_subplot(gs3[0, 1], label="g")
    ax_6_7_cb = fig.add_subplot(gs3[0, 2], label="67_cb")

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


if __name__ == "__main__":
    #plot_ipr_figs1('vacancy', [10], [2], 'x')
    fig = figure_layout2()

    Lattice = DefectLattice(10, 10, "vacancy", True)
    eig_dict_1 = compute_eigenvectors_eigenvalues(Lattice, -1.0, [0.5, 0., 0.])
    eig_dict_2 = compute_eigenvectors_eigenvalues(Lattice, -1.0, [1.5, 0., 0.])

    ax_a, ax_b, ax_c, ax_c_cb, ax_d, ax_d_cb, ax_e, ax_e_cb, ax_f, ax_g, ax_fg_cb = fig.axes


    axes = fig.axes
    for ed, ipr_ax, s_ax, cb_ax in zip([eig_dict_1, eig_dict_2], [ax_a, ax_b], [ax_c, ax_d], [ax_c_cb, ax_d_cb]):
        eigvals = ed["eigenvalues"]
        ipr = ed["left_ipr"]
        ipr_ax.scatter(np.abs(eigvals) * np.sign(eigvals.real), ipr, c='r', alpha=0.5, zorder=0)
        s_scat = s_ax.scatter(eigvals.real, eigvals.imag, c=ipr, cmap='jet')
        cb = plt.colorbar(s_scat, cb_ax)


    close = eig_dict_1["selected_left_eigenvectors"]
    L1 = eig_dict_1["L"]
    L2 = eig_dict_2["L"]

    close -= np.min(close)
    L1 -= np.min(L1)
    L2 -= np.min(L2)

    close /= np.max(close)

    skin_max = max(np.max(L1), np.max(L2))
    L1 /= skin_max
    L2 /= skin_max

    X, Y = Lattice.X, Lattice.Y
    s1 = ax_e.scatter(X, Y, c=close, cmap="Greys", s=100, vmin=0., vmax=1., zorder=1)
    d_idxs = Lattice.defect_indices
    s1_defects = ax_e.scatter(X[d_idxs], Y[d_idxs], zorder=2, lw=1, facecolor='w', edgecolor='r')
    top_cb = plt.colorbar(s1, ax_e_cb)

    s2 = ax_f.scatter(X, Y, c=L1, cmap="Greys", s=100, vmin=0., vmax=1.)
    s3 = ax_g.scatter(X, Y, c=L2, cmap="Greys", s=100, vmin=0., vmax=1.)
    skin_cb = plt.colorbar(s2, ax_fg_cb)
    plt.show()
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

        plot = ldos_ax.scatter(Lattice.X, Lattice.Y, c=color_array, cmap='Greys', s=200)
        ldos_ax.scatter(Lattice.Lx / 2, Lattice.Ly / 2, facecolor='w', edgecolor='r', s=200, lw=2, ls='--')
        if 0:
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


def spectrum_and_ipr_figure(axes, method, m0, h0_vector, hsub_vector, Ls, ns, 
                            defect_radius:int = 1, break_c4:bool = False, 
                            do_additional_plotting:bool = True):
    Ls = np.sort(Ls)

    ldos_plot_radius = max(Ls[-1] // 4, 5)
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
        fpx = fpy = -schottky_sep - 0.5
        if method == "frenkel_pair": print(f"L={L}: fpx = {fpx}, fpy = {fpy}")

        Lattice = DefectLattice(L, L, method, True, schottky_separation=schottky_sep, 
                                frenkel_x_disp = fpx, frenkel_y_disp= fpy,
                                defect_radius = defect_radius, break_c4 = break_c4)
        
        ed = compute_eigenvectors_eigenvalues(Lattice, m0, h0_vector, hsub_vector, n)
        lattices.append(Lattice)
        eig_dicts.append(ed)

    if method in ["schottky", "frenkel_pair"]: ldos_plot_radius = max(ldos_plot_radius, 4+-int(round(max(abs(fpx), abs(fpy)), 0)), 2 + schottky_sep // 2)

    ldos_plot_radius = min(ldos_plot_radius, Ls[-1] // 2)

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
        ax_a.set_xlabel("$|E| \\times {\\rm sign}[\\Re (E)]$")
        ax_a.set_ylabel("${\\rm IPR} \\times 10^2$")
        ax_a.legend(loc = 'upper right')

    ext_min = max(L // 2 - ldos_plot_radius, 0)
    ext_max = min(L // 2 + ldos_plot_radius, L)
    ldos_extent = (ext_min, ext_max, ext_min, ext_max)

    # Axis (a_inset)
    if do_additional_plotting:
        ax_a_inset, a_inset_cbar = plot_on_lattice(ax_a_inset, Lattice, ed["selected_left_eigenvectors"], 
                                                "tripcolor", "Greys", plot_colorbar=False,
                                                extent = ldos_extent)
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

    b_scat1.set_clim(vmin=0)
    b_cbar = plt.colorbar(b_scat1, ax_b_cb)

    format_colorbar(b_cbar, *b_scat1.get_clim())
    ax_b.set_xlabel("$\\Re (E)$")
    ax_b.set_ylabel("$\\Im (E) \\times 10^1$")

    # Axis (c) and (c_colorbar)
    ax_c, ax_c_cb = plot_on_lattice(ax_c, Lattice, ed["L"], 
                                    "tripcolor", "Greys", plot_colorbar=False,
                                    extent = ldos_extent)
    ax_c.tick_params(axis="both", labelsize=10)
    ax_c.set_xlabel("$X$", fontsize=10)
    ax_c.set_ylabel("$Y$", fontsize=10)
    
# ============================
# ============================
# ============================


def plot_ipr_figs1(method, Ls, ns, hdir):
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

    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_ipr_{hdir}_greys.png", dpi=300, bbox_inches="tight")


def plot_ipr_figs2(method, Ls, ns, hdir):
    fig = plt.figure(figsize=(20, 30))
    plt.subplots_adjust(wspace=0.1)
    master_gs = gridspec.GridSpec(3, 2, figure=fig)

    figure_layout(fig, master_gs[0, 0])
    figure_layout(fig, master_gs[0, 1], False)
    figure_layout(fig, master_gs[1, 0])
    figure_layout(fig, master_gs[1, 1], False)
    figure_layout(fig, master_gs[2, 0])
    figure_layout(fig, master_gs[2, 1], False)
    
    axes = np.array(fig.axes)
    axes_chunks = [axes[:5], axes[5:9], axes[9:14], axes[14:18], axes[18:23], axes[23:]]

    h_dir_mapping = {'x': 0, 'y': 1, 'z': 2}
    v1 = np.zeros(3)
    v2 = np.zeros(3)
    v1[h_dir_mapping[hdir]] = 0.5
    v2[h_dir_mapping[hdir]] = 1.5

    spectrum_and_ipr_figure(axes_chunks[0], method, -1.0, v1, v2, Ls, ns)
    spectrum_and_ipr_figure(axes_chunks[1], method, -1.0, v2, v1, Ls, ns, do_additional_plotting = False)
    spectrum_and_ipr_figure(axes_chunks[2], method, -1.0, v1, v2, Ls, ns, defect_radius = 2)
    spectrum_and_ipr_figure(axes_chunks[3], method, -1.0, v2, v1, Ls, ns, defect_radius = 2, do_additional_plotting = False)
    spectrum_and_ipr_figure(axes_chunks[4], method, -1.0, v1, v2, Ls, ns, defect_radius = 2, break_c4 =  True)
    spectrum_and_ipr_figure(axes_chunks[5], method, -1.0, v2, v1, Ls, ns, defect_radius = 2, break_c4 =  True, do_additional_plotting = False)

    fig_row_labels = ["i", "ii", "iii"]
    for i in range(3):
        for j, ax in enumerate(fig.axes[9*i:9*(i+1)]):
            lab = "(" + fig_row_labels[i] + "." + "ab_c_de_f"[j] + ")"
            if lab[-2] == "_":
                continue
            ax.annotate(
                lab,
                xy = (0.025, 0.975),
                xycoords='axes fraction',
                ha='left',
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
            )

    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_ipr_{hdir}.png", dpi=300, bbox_inches="tight")


def figure_layout2():
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1., 1.5, 1.5])

    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 0])
    ax1 = fig.add_subplot(gs0[0, 0], label="a")
    ax2 = fig.add_subplot(gs0[0, 1], label="b")

    gs1_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], width_ratios=[1, .1])
    ax3 = fig.add_subplot(gs1_a[0, 0], label="c")
    ax3_cb = fig.add_subplot(gs1_a[0, 1], label="c_cb")
    gs1_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], width_ratios=[1, .1])
    ax4 = fig.add_subplot(gs1_b[0, 0], label="d")
    ax4_cb = fig.add_subplot(gs1_b[0, 1], label="d_cb")

    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], width_ratios=[1, .1])
    ax5 = fig.add_subplot(gs2[0, 0], label="e")
    ax5_cb = fig.add_subplot(gs2[0, 1], label="e_cb")
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 2], width_ratios=[1, 1, .1])
    ax6 = fig.add_subplot(gs3[0, 0], label="f")
    ax7 = fig.add_subplot(gs3[0, 1], label="g")
    ax_6_7_cb = fig.add_subplot(gs3[0, 2], label="67_cb")

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


if __name__ == "__main__":
    #plot_ipr_figs1('vacancy', [10], [2], 'x')
    fig = figure_layout2()

    Lattice = DefectLattice(10, 10, "vacancy", True)
    eig_dict_1 = compute_eigenvectors_eigenvalues(Lattice, -1.0, [0.5, 0., 0.])
    eig_dict_2 = compute_eigenvectors_eigenvalues(Lattice, -1.0, [1.5, 0., 0.])

    ax_a, ax_b, ax_c, ax_c_cb, ax_d, ax_d_cb, ax_e, ax_e_cb, ax_f, ax_g, ax_fg_cb = fig.axes


    axes = fig.axes
    for ed, ipr_ax, s_ax, cb_ax in zip([eig_dict_1, eig_dict_2], [ax_a, ax_b], [ax_c, ax_d], [ax_c_cb, ax_d_cb]):
        eigvals = ed["eigenvalues"]
        ipr = ed["left_ipr"]
        ipr_ax.scatter(np.abs(eigvals) * np.sign(eigvals.real), ipr, c='r', alpha=0.5, zorder=0)
        s_scat = s_ax.scatter(eigvals.real, eigvals.imag, c=ipr, cmap='jet')
        cb = plt.colorbar(s_scat, cb_ax)


    close = eig_dict_1["selected_left_eigenvectors"]
    L1 = eig_dict_1["L"]
    L2 = eig_dict_2["L"]

    close -= np.min(close)
    L1 -= np.min(L1)
    L2 -= np.min(L2)

    close /= np.max(close)

    skin_max = max(np.max(L1), np.max(L2))
    L1 /= skin_max
    L2 /= skin_max

    X, Y = Lattice.X, Lattice.Y
    s1 = ax_e.scatter(X, Y, c=close, cmap="Greys", s=100, vmin=0., vmax=1., zorder=1)
    d_idxs = Lattice.defect_indices
    s1_defects = ax_e.scatter(X[d_idxs], Y[d_idxs], zorder=2, lw=1, facecolor='w', edgecolor='r')
    top_cb = plt.colorbar(s1, ax_e_cb)

    s2 = ax_f.scatter(X, Y, c=L1, cmap="Greys", s=100, vmin=0., vmax=1.)
    s3 = ax_g.scatter(X, Y, c=L2, cmap="Greys", s=100, vmin=0., vmax=1.)
    skin_cb = plt.colorbar(s2, ax_fg_cb)
    plt.show()


# plotting2.py


# region Saving and Reading Data
def save_ipr_data(Ls:list[int], eig_dicts:list[dict], fname:str, directory:str = "./NonHermitian/Data/"):
    if not os.path.exists(directory):
        os.makedirs(directory)
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
    file1 = f"{method}_ipr_data_h{hdir}=0.5.h5"
    file2 = f"{method}_ipr_data_h{hdir}=1.5.h5"

    if method in ["substitution", "interstitial"]:
        n_defects = 1
        if "defect_radius" in kwargs.keys():
            n_defects += np.sum(4 * np.arange(kwargs["defect_radius"]))
        if "break_c4" in kwargs.keys():
            n_defects += 2
        file1 = f"{method}_ipr_data_h{hdir}=0.5_nd={n_defects}.h5"
        file2 = f"{method}_ipr_data_h{hdir}=1.5_nd={n_defects}.h5"

    schottky_separations = [L // 4 + (L // 4 + 1) % 2 for L in Ls]
    fpxs = fpys = [-s -0.5 for s in schottky_separations]

    lattices = [DefectLattice(L, L, method, True, schottky_separation=s, frenkel_x_disp=fpx, frenkel_y_disp=fpy, **kwargs) 
                for L, s, fpx, fpy in zip(Ls, schottky_separations, fpxs, fpys)]
    
    if all([os.path.exists(directory + file) for file in [file1, file2]]):
        print(f"IPR data files already exist for method {method} and hdir {hdir}.")
        new_Ls1, data1 = read_ipr_data(file1, directory)
        new_Ls2, data2 = read_ipr_data(file2, directory)
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
    save_ipr_data(Ls, ed1s, file1, "./NonHermitian/Data/")
    save_ipr_data(Ls, ed2s, file2, "./NonHermitian/Data/")
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
        r = 5
        extent1 = (11.5 - r, 11.5 + r, 11.5 - r, 11.5 + r)
        extent2 = (11.5 - r, Ls[-1] // 2 + r, 11.5 - r, Ls[-1] // 2 + r)
    else:
        if method in ["vacancy", "substitution"]:
            r = 10
            extent = (Ls[-1] // 2 - r, Ls[-1] // 2 + r, Ls[-1] // 2 - r, Ls[-1] // 2 + r)
        elif method == "interstitial":
            r = 5
            extent = (Ls[-1] // 2 - r - 0.5, Ls[-1] // 2 + r, Ls[-1] // 2 - r - 0.5, Ls[-1] // 2 + r)
        elif method == "schottky":
            r = 11
            extent = (Ls[-1] // 2 - r, Ls[-1] // 2 + r, Ls[-1] // 2 - r, Ls[-1] // 2 + r)
        extent1 = extent2 = extent


    print(LargestLattice.defect_positions)
    plot_ldos(LargestLattice, ax_e, topo_ldos, ax_e_cb, extent=extent1, scatter_size=50)
    plot_ldos(LargestLattice, ax_f, L1, ax_f_cb, extent=extent2, scatter_size=50)
    plot_ldos(LargestLattice, ax_g, L2, ax_g_cb, extent=extent2, scatter_size=50)

    # Remove some labels and legends for clarity
    ax_b.set_ylabel("")
    ax_b.legend().remove()
    ax_d.set_ylabel("")
    ax_g.set_ylabel("")

    for ax in [ax_e, ax_f, ax_g]:
        ax.set_aspect("equal")

    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_IPR_h{hdir}.svg", dpi=96)



# war crime
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


# region Saving and Reading Data
def save_ipr_data(Ls:list[int], eig_dicts:list[dict], fname:str, directory:str = "./NonHermitian/Data/"):
    if not os.path.exists(directory):
        os.makedirs(directory)
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
    file1 = f"{method}_ipr_data_h{hdir}=0.5.h5"
    file2 = f"{method}_ipr_data_h{hdir}=1.5.h5"

    if method in ["substitution", "interstitial"]:
        n_defects = 1
        if "defect_radius" in kwargs.keys():
            n_defects += np.sum(4 * np.arange(kwargs["defect_radius"]))
        if "break_c4" in kwargs.keys():
            n_defects += 2
        file1 = f"{method}_ipr_data_h{hdir}=0.5_nd={n_defects}.h5"
        file2 = f"{method}_ipr_data_h{hdir}=1.5_nd={n_defects}.h5"

    schottky_separations = [L // 4 + (L // 4 + 1) % 2 for L in Ls]
    fpxs = fpys = [-s -0.5 for s in schottky_separations]

    lattices = [DefectLattice(L, L, method, True, schottky_separation=s, frenkel_x_disp=fpx, frenkel_y_disp=fpy, **kwargs) 
                for L, s, fpx, fpy in zip(Ls, schottky_separations, fpxs, fpys)]
    
    if all([os.path.exists(directory + file) for file in [file1, file2]]):
        print(f"IPR data files already exist for method {method} and hdir {hdir}.")
        new_Ls1, data1 = read_ipr_data(file1, directory)
        new_Ls2, data2 = read_ipr_data(file2, directory)
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
    save_ipr_data(Ls, ed1s, file1, "./NonHermitian/Data/")
    save_ipr_data(Ls, ed2s, file2, "./NonHermitian/Data/")
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
        r = 5
        extent1 = (11.5 - r, 11.5 + r, 11.5 - r, 11.5 + r)
        extent2 = (11.5 - r, Ls[-1] // 2 + r, 11.5 - r, Ls[-1] // 2 + r)
    else:
        if method in ["vacancy", "substitution"]:
            r = 10
            extent = (Ls[-1] // 2 - r, Ls[-1] // 2 + r, Ls[-1] // 2 - r, Ls[-1] // 2 + r)
        elif method == "interstitial":
            r = 5
            extent = (Ls[-1] // 2 - r - 0.5, Ls[-1] // 2 + r, Ls[-1] // 2 - r - 0.5, Ls[-1] // 2 + r)
        elif method == "schottky":
            r = 11
            extent = (Ls[-1] // 2 - r, Ls[-1] // 2 + r, Ls[-1] // 2 - r, Ls[-1] // 2 + r)
        extent1 = extent2 = extent


    print(LargestLattice.defect_positions)
    plot_ldos(LargestLattice, ax_e, topo_ldos, ax_e_cb, extent=extent1, scatter_size=50)
    plot_ldos(LargestLattice, ax_f, L1, ax_f_cb, extent=extent2, scatter_size=50)
    plot_ldos(LargestLattice, ax_g, L2, ax_g_cb, extent=extent2, scatter_size=50)

    # Remove some labels and legends for clarity
    ax_b.set_ylabel("")
    ax_b.legend().remove()
    ax_d.set_ylabel("")
    ax_g.set_ylabel("")

    for ax in [ax_e, ax_f, ax_g]:
        ax.set_aspect("equal")

    plt.savefig(f"./NonHermitian/Plots/IPR/{method}_IPR_h{hdir}.svg", dpi=96)


# plotting3.py

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



# 
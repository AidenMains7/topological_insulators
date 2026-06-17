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





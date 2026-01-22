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










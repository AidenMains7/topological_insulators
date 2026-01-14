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


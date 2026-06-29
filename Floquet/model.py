#import numpy as np
from matplotlib import pyplot as plt
import torch
import gc

from typing import Callable, Any
from functools import wraps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_VRAM_INFO = False

def get_vram_info() -> str:
    return f"VRAM Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB"

def track_vram(func:Callable):
    if PRINT_VRAM_INFO:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print(func.__name__ + " start : " + get_vram_info())
            result = func(*args, **kwargs)
            print(func.__name__ + " end : " + get_vram_info())
            return result
        return wrapper
    else:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper


@track_vram
@torch.no_grad()
def generate_lattice(Lx, Ly):
    return torch.arange(Lx * Ly, device=device).reshape(Ly, Lx)

@track_vram
@torch.no_grad()
def compute_distances(lattice: torch.Tensor, pbc: bool):
    Ly, Lx = lattice.shape
    Y, X = torch.where(lattice >= 0)

    dx = (X - X[:, None])
    dy = (Y - Y[:, None])
    
    if pbc:
        dx = dx - Lx * torch.round(dx / Lx)
        dy = dy - Ly * torch.round(dy / Ly)
        
    return dx, dy

@track_vram
@torch.no_grad()
def compute_wannier_matrices_fourier(dx: torch.Tensor, dy: torch.Tensor,
                                     deleteUnused:bool = False):
    xp_mask = torch.isclose(dx, torch.tensor(1.0, dtype = dx.dtype)) & torch.isclose(dy, torch.tensor(0.0, dtype = dy.dtype))
    yp_mask = torch.isclose(dx, torch.tensor(0.0, dtype = dx.dtype)) & torch.isclose(dy, torch.tensor(1.0, dtype = dy.dtype))

    N = dx.shape[0]
    Cx = torch.zeros((N, N), dtype=torch.complex128, device=device)
    Sx = torch.zeros((N, N), dtype=torch.complex128, device=device)
    Cy = torch.zeros((N, N), dtype=torch.complex128, device=device)
    Sy = torch.zeros((N, N), dtype=torch.complex128, device=device)
    I = torch.eye(N,         dtype=torch.complex128, device=device)

    Sx[xp_mask] = torch.tensor(1j/ 2, dtype = torch.complex128, device=device)
    Cx[xp_mask] = 1 / 2
    Cy[yp_mask] = 1 / 2
    Sy[yp_mask] = torch.tensor(1j/ 2, dtype = torch.complex128, device=device)

    Sx += Sx.conj().T
    Sy += Sy.conj().T
    Cx += Cx.conj().T
    Cy += Cy.conj().T

    if deleteUnused: # is this necessary?
        del xp_mask
        del yp_mask
    
    return I, Sx, Sy, Cx + Cy

@torch.no_grad()
def batched_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    batch_size, n, _ = A.shape
    m = B.shape[0]
    
    # 'bij, kl -> bikjl' creates the block structure, then we reshape to flat matrices
    return torch.einsum('bij,kl->bikjl', A, B).reshape(batch_size, n * m, n * m)

@track_vram
@torch.no_grad()
def compute_hamiltonian(drive_type:str, wannier_matrices:tuple, m0_batch:torch.Tensor, m1_batch:torch.Tensor, omega_batch:torch.Tensor, t:float = 1.0, t0:float = 1.0):
    """
    drive_type (str): the type of time-depdendent drive type. Options are 'kick', 'step', and 'sin'
    
    """
    assert drive_type in ['kick', 'step', 'sinusoidal']
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device) 
    pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)

    m0_b = m0_batch.view(-1, 1, 1)
    m1_b = m1_batch.view(-1, 1, 1)
    T_b = 2 * torch.pi / omega_batch.view(-1, 1, 1)


    I, Sx, Sy, Cx_plus_Cy = wannier_matrices

    dx = (t * Sx).unsqueeze(0)
    dy = (t * Sy).unsqueeze(0)
    dz = m0_b * I.unsqueeze(0) - t0 * Cx_plus_Cy.unsqueeze(0)

    d_vector = torch.stack((dx.expand_as(dz), dy.expand_as(dz), dz)).view(-1, 3, dz.shape[-2], dz.shape[-1])
    d_norm = torch.linalg.norm(d_vector, axis=1)
    d_norm = torch.where(d_norm == 0.0, torch.tensor(1e-6, device=device, dtype=d_norm.dtype), d_norm)

    match drive_type:
        case 'kick':
            abs_term = torch.abs(d_norm * T_b)
            A = torch.cos(abs_term) * torch.cos(m1_b)
            B = torch.sin(abs_term) * torch.cos(m1_b)
            C = torch.cos(abs_term) * torch.sin(m1_b)
            D = torch.sin(abs_term) * torch.sin(m1_b)

            d0 = torch.arccos(A - D * dz / d_norm) / T_b
            coef = d0 / torch.sin(d0 * T_b)

            d1_flq = coef * (B * dx / d_norm - D * dy / d_norm)
            d2_flq = coef * (B * dy / d_norm + D * dx / d_norm)
            d3_flq = coef * (B * dz / d_norm + C)

        case 'step':
            delta_t_mu = torch.stack([T_b / 4, T_b / 2, T_b / 4]).view(-1, 3, 1)
            theta_mu = [torch.stack([torch.abs(d_j) * delta_t_j for d_j, delta_t_j in zip(d_vector[i], delta_t_mu[i])]) for i in range(len(m0_b))]
            theta_mu = torch.stack(theta_mu)
            alpha = torch.prod(torch.cos(theta_mu), dim=1)
            beta = torch.prod(torch.stack([torch.sin(theta_j) / torch.abs(d_j) for (theta_j, d_j) in zip(theta_mu, d_vector)]), dim=1)

            def _A(i, j, k):
                assert all([val in [0, 1, 2] for val in [i, j, k]])
                numerator = torch.cos(theta_mu[i]) * torch.sin(theta_mu[j]) * torch.sin(theta_mu[k])
                denominator = torch.abs(d_vector[j]) * torch.abs(d_vector[k])
                return  numerator / denominator
            
            def _B(i, j, k):
                assert all([val in [0, 1, 2] for val in [i, j, k]])
                numerator = torch.sin(theta_mu[i]) * torch.cos(theta_mu[j]) * torch.cos(theta_mu[k])
                return numerator / torch.abs(d_vector[i])
            
            def _Q(i):
                if i == 0:
                    return d_vector[1] * d_vector[2]
                elif i == 1:
                    return d_vector[2] * d_vector[0]
                elif i == 2:
                    return d_vector[0] * d_vector[1]
                else:
                    raise ValueError
                
            def _R(i):
                return (-1) ** (i + 1) * d_vector[1] * (m_j - m_k)
            

            print(_A(1, 2, 3).shape)


            raise SystemExit

    H_flq = batched_kron(d1_flq, pauli_x) + batched_kron(d2_flq, pauli_y) + batched_kron(d3_flq, pauli_z)

    return H_flq


@track_vram
@torch.no_grad()
def compute_projector(hamiltonian: torch.Tensor) -> torch.Tensor:
    """
    Args:
        hamiltonian (np.ndarray): The full system Hamiltonian matrix.

    Returns:
        np.ndarray: The mathematical projector matrix spanned by the lower band eigenstates.
    """

    eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
    idx = eigenvalues.shape[0] // 2 - 1
    highest_lower_band = eigenvalues[:, idx].unsqueeze(1)

    D = torch.where(eigenvalues <= highest_lower_band, torch.tensor(1.0, dtype=torch.complex128), torch.tensor(0.0, dtype=torch.complex128))
    projector = eigenvectors @ torch.diag_embed(D) @ eigenvectors.conj().transpose(1, 2)
    return projector


@track_vram
@torch.no_grad()
def compute_bott_index(projector: torch.Tensor, lattice :torch.Tensor) -> torch.Tensor:
    """
    Args:
        projector (np.ndarray): The spectral projection operator for the lower band.

    Returns:
        float: The numeric value of the computed Bott Index topological invariant.
    """
    Y, X = torch.where(lattice >= 0)[:]

    X = torch.repeat_interleave(X, 2)
    Y = torch.repeat_interleave(Y, 2)

    Lx = torch.max(X) - torch.min(X)
    Ly = torch.max(Y) - torch.min(Y)

    x_unitary = torch.exp(1j * 2 * torch.pi * X.to(torch.complex128) / Lx)
    y_unitary = torch.exp(1j * 2 * torch.pi * Y.to(torch.complex128) / Ly)

    Ux = x_unitary.view(1, -1, 1)
    Uy = y_unitary.view(1, -1, 1)
    
    Ux_P = Ux * projector
    Uy_P = Uy * projector
    Ux_dag_P = Ux.conj() * projector
    Uy_dag_P = Uy.conj() * projector

    B, N, _ = projector.shape
    I = torch.eye(N, dtype=torch.complex128, device=device).unsqueeze(0).expand(B, -1, -1)
    A = I - projector + projector @ Ux_P @ Uy_P @ Ux_dag_P @ Uy_dag_P
    eigvals = torch.linalg.eigvals(A)
    bott_index = torch.imag(torch.sum(torch.log(eigvals), dim=1)) / (2 * torch.pi)
    return bott_index


@track_vram
@torch.no_grad()
def compute_bott_wrapper(Lx, Ly, pbc, drive_type):

    m0_batch = torch.tensor([3.0, 3.0], dtype=torch.complex128, device=device)
    m1_batch = torch.tensor([0.0, 0.0], dtype=torch.complex128, device=device)
    omega_batch = torch.tensor([4.0, 4.0], dtype=torch.complex128, device=device)

    lattice = generate_lattice(Lx, Ly)
    dx, dy = compute_distances(lattice, pbc)
    wannier_matrices = compute_wannier_matrices_fourier(dx, dy, False)
    H = compute_hamiltonian(drive_type, wannier_matrices, m0_batch, m1_batch, omega_batch)
    projector = compute_projector(H)

    bott = compute_bott_index(projector, lattice)
    print(bott)


if __name__ == "__main__":
    compute_bott_wrapper(20, 20, False, 'step')


if 0:
    import time

    t0 = time.time()
    torch.cuda.empty_cache()
    print(device)

    L = 11

    lattice = generate_lattice(L, L)
    dx, dy = compute_distances(lattice, True)
    wannier = compute_wannier_matrices_fourier(dx, dy)


    dimensions = (1, 11, 11)
    m0_values = torch.linspace(3.0, 3.0,   dimensions[0], device=device)
    m1_values = torch.linspace(-10.0, 2.0, dimensions[1], device=device)
    w_values = torch.linspace(2.0, 12.0,   dimensions[2], device=device)
    m0_values, m1_values, w_values = torch.meshgrid([m0_values, m1_values, w_values], indexing='ij')

    m0_values = m0_values.flatten()
    m1_values = m1_values.flatten()
    w_values = w_values.flatten()

    chunk_size = 121
    m0_chunks = torch.split(m0_values, chunk_size)
    m1_chunks = torch.split(m1_values, chunk_size)
    w_chunks = torch.split(w_values, chunk_size)

    bott_index_results = []

    for i, (m0_c, m1_c, t_c) in enumerate(zip(m0_chunks, m1_chunks, w_chunks)):
        t0 = time.time()
        H_chunk = compute_hamiltonian('kick', wannier, m0_c, m1_c, t_c)
        P_chunk = compute_projector(H_chunk)
        BI_chunk = compute_bott_index(P_chunk, lattice)

        bott_index_results.append(BI_chunk.cpu())
        
        del H_chunk, P_chunk, BI_chunk
        torch.cuda.empty_cache()
        
        print(f"Chunk {i+1}/{len(m0_chunks)} complete. : t={time.time() - t0:.2f}s")


    BI = np.concatenate(bott_index_results).reshape(dimensions)

    plt.imshow(BI[0])
    plt.savefig('temp.png')

        
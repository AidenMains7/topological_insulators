import numpy as np
from matplotlib import pyplot as plt
import torch
import gc

from typing import Callable, Any
from functools import wraps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
printVramInfo = False

def get_vram_info() -> str:
    return f"VRAM Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB"

def track_vram(func:Callable):
    if printVramInfo:
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

    dx = X - X[:, None]
    dy = Y - Y[:, None]
    
    if pbc:
        dx = dx - Lx * torch.round(dx / Lx)
        dy = dy - Ly * torch.round(dy / Ly)
        
    return dx, dy

@track_vram
@torch.no_grad()
def compute_wannier_matrices_fourier(dx: torch.Tensor, dy: torch.Tensor,
                                     deleteUnused:bool = False):
    xp_mask = torch.isclose(dx, torch.tensor(1.0)) & torch.isclose(dy, torch.tensor(0.0))
    yp_mask = torch.isclose(dx, torch.tensor(0.0)) & torch.isclose(dy, torch.tensor(1.0))

    N = dx.shape[0]
    Cx = torch.zeros((N, N), dtype=torch.complex64, device=device)
    Sx = torch.zeros((N, N), dtype=torch.complex64, device=device)
    Cy = torch.zeros((N, N), dtype=torch.complex64, device=device)
    Sy = torch.zeros((N, N), dtype=torch.complex64, device=device)
    I = torch.eye(N,         dtype=torch.complex64, device=device)

    Sx[xp_mask] = torch.tensor(1j/ 2, dtype = torch.complex64, device=device)
    Cx[xp_mask] = 1 / 2
    Cy[yp_mask] = 1 / 2
    Sy[yp_mask] = torch.tensor(1j/ 2, dtype = torch.complex64, device=device)

    Sx += Sx.conj().T
    Sy += Sy.conj().T
    Cx += Cx.conj().T
    Cy += Cy.conj().T

    if deleteUnused:
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
def compute_hamiltonian(drive_type:str, wannier_matrices:tuple, m0_batch:torch.Tensor, m1_batch:torch.Tensor, T_batch:torch.Tensor, t:float = 1.0, t0:float = 1.0):

    assert drive_type in ['kick', 'step', 'sinusoidal']
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device) 
    pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

    m0_b = m0_batch.view(-1, 1, 1)
    m1_b = m1_batch.view(-1, 1, 1)
    T_b = T_batch.view(-1, 1, 1)

    I, Sx, Sy, Cx_plus_Cy = wannier_matrices

    dx = (t * Sx).unsqueeze(0)
    dy = (t * Sy).unsqueeze(0)
    dz = m0_b * I.unsqueeze(0) - t0 * Cx_plus_Cy.unsqueeze(0)

    d_vector = torch.stack((dx.expand_as(dz), dy.expand_as(dz), dz))
    d_norm = torch.linalg.norm(d_vector, axis=0)

    d_norm = torch.where(d_norm == 0.0, torch.tensor(1e-6, device=device, dtype=d_norm.dtype), d_norm)


    match drive_type:
        case 'kick':
            
            A = torch.cos(d_norm * T_b) * torch.cos(m1_b)
            B = torch.sin(d_norm * T_b) * torch.cos(m1_b)
            C = torch.cos(d_norm * T_b) * torch.sin(m1_b)
            D = torch.sin(d_norm * T_b) * torch.sin(m1_b)

            d0 = torch.arccos(A - D * dz / d_norm) / T_b
            coef = d0 / torch.sin(d0 * T_b)

            d1_flq = coef * (B * dx / d_norm - D * dy / d_norm)
            d2_flq = coef * (B * dy / d_norm + D * dx / d_norm)
            d3_flq = coef * (B * dz / d_norm + C)

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

    D = torch.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
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

    x_unitary = torch.exp(1j * 2 * torch.pi * X / Lx)
    y_unitary = torch.exp(1j * 2 * torch.pi * Y / Ly)

    Ux = x_unitary.view(1, -1, 1)
    Uy = y_unitary.view(1, -1, 1)
    
    Ux_P = Ux * projector
    Uy_P = Uy * projector
    Ux_dag_P = Ux.conj() * projector
    Uy_dag_P = Uy.conj() * projector

    B, N, _ = projector.shape
    I = torch.eye(N, dtype=torch.complex64, device=device).unsqueeze(0).expand(B, -1, -1)
    A = I - projector + projector @ Ux_P @ Uy_P @ Ux_dag_P @ Uy_dag_P
    
    eigvals = torch.linalg.eigvals(A)
    bott_index = torch.imag(torch.sum(torch.log(eigvals), dim=1)) / (2 * torch.pi)
    return bott_index



if __name__ == "__main__":
    import time

    t0 = time.time()
    torch.cuda.empty_cache()

    L = 15

    print(device)

    lattice = generate_lattice(L, L)
    dx, dy = compute_distances(lattice, True)
    wannier = compute_wannier_matrices_fourier(dx, dy)


    dimensions = (1, 11, 11)
    m0_values = torch.linspace(3.0, 3.0,   dimensions[0], device=device)
    m1_values = torch.linspace(-10.0, 2.0, dimensions[1], device=device)
    w_values = torch.linspace(3.0, 13.0,   dimensions[2], device=device)
    m0_values, m1_values, w_values = torch.meshgrid([m0_values, m1_values, w_values], indexing='ij')

    m0_values = m0_values.flatten()
    m1_values = m1_values.flatten()
    w_values = w_values.flatten()

    T_values = 2 * torch.pi / w_values

    chunk_size = 11
    m0_chunks = torch.split(m0_values, chunk_size)
    m1_chunks = torch.split(m1_values, chunk_size)
    T_chunks = torch.split(T_values, chunk_size)

    bott_index_results = []

    for i, (m0_c, m1_c, t_c) in enumerate(zip(m0_chunks, m1_chunks, T_chunks)):
        t0 = time.time()
        H_chunk = compute_hamiltonian('kick', wannier, m0_c, m1_c, t_c)
        P_chunk = compute_projector(H_chunk)
        BI_chunk = compute_bott_index(P_chunk, lattice)

        bott_index_results.append(BI_chunk.cpu())
        
        del H_chunk, P_chunk, BI_chunk
        torch.cuda.empty_cache()
        
        print(f"Chunk {i+1}/{len(m0_chunks)} complete.")


    BI = np.concatenate(bott_index_results).reshape(dimensions)
    
    plt.imshow(BI[0])
    plt.show()
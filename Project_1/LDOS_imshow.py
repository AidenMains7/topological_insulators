import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from itertools import product

from project_dependencies import precompute, Hamiltonian_reconstruct, projector_exact, bott_index
from scipy.linalg import eigh

import glob
import contextlib
from PIL import Image



def LDOS(Hamiltonian:np.ndarray) -> np.ndarray:

    # Get eigenvalues, eigenvectors
    eigvals, eigvecs = eigh(Hamiltonian)

    # Index of lowest eigenvalue
    idxs = np.argsort(np.abs(eigvals))

    # Eigenvectors and eigenvalues for lowest energy states
    eigva_one, eigva_two = eigvals[idxs[0]], eigvals[idxs[1]]
    eigve_one, eigve_two = eigvecs[:, idxs[0]], eigvecs[:, idxs[1]]

    # |v|^2
    eigve_one, eigve_two = np.power(np.abs(eigve_one), 2), np.power(np.abs(eigve_two), 2)

    # Sum pairwise
    LDOS_one, LDOS_two = eigve_one[0::2] + eigve_one[1::2], eigve_two[0::2] + eigve_two[1::2]

    # Spectral gap
    gap = np.abs(eigva_one) - np.abs(eigva_two)


    return LDOS_one, LDOS_two, gap



def remap_LDOS(local_density, lattice):
    # Ensure proper size
    if np.max(lattice)+1 != local_density.size:
        raise ValueError(f'Sizes of inputs do not match. Sizes {np.max(lattice)+1} and {local_density.size}')
    
    # Locations of filled site
    fills = np.argwhere(lattice.flatten() >= 0).flatten()

    # Initialize lattifce
    LDOS_lattice = np.full(lattice.size, 0.0)

    # Input into lattice
    LDOS_lattice[fills] = local_density

    # Reshape to proper size
    return LDOS_lattice.reshape(lattice.shape)


# Parallel
def LDOS_imshow(method:str, order:int, pad_width:int, n:int | None, pbc:bool, M_vals:np.ndarray | list | tuple, B_tilde_vals:np.ndarray | list | tuple, t1:float=1.0, t2:float=1.0, B:float=1.0, num_jobs:int=4) -> None:

    # Pre compute
    pre_data, lattice = precompute(method, order, pad_width, pbc, n, t1, t2, B)

    # Get list of parameter combinations
    parameters = tuple(product(M_vals, B_tilde_vals))


    def do_single(i):
        # Do computation
        M, B_tilde = parameters[i]
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)
        P = projector_exact(H, 0.0)
        bott = bott_index(P, lattice)

        # Compute LDOS for two closest to 0 eigenstates
        local_one, local_two, gap = LDOS(H)
        local_both = local_one + local_two

        # Remap to shape of lattice
        LDOS_lattice = remap_LDOS(local_both, lattice)

        # Create figure
        fig = plt.figure(figsize=(10,10))
        plt.imshow(LDOS_lattice, label='Local Density of States', cmap='Purples')
        plt.title(f'M={M:.1f} :: B={B_tilde:.1f} :: BI={bott}')
        plt.colorbar()
        plt.savefig(f'Data/LDOS/Square_Range/{method}_{M:.1f}_{B:.1f}_{bott}.png')

    # Compute in parallel
    Parallel(n_jobs=num_jobs)(delayed(do_single)(i) for i in range(len(parameters)))


# Create a gif :)
def create_gif():
    fp_in = "Data/LDOS/Square_Range/square_*.png"
    fp_out = "Data/LDOS/square.gif"

    image_files = glob.glob(fp_in)
    shortened_names = [fp[len("Data/LDOS/Square_Range/square_"):-4] for fp in image_files]
    number_order = [[float(num_str) for num_str in fp.split('_')][0] for fp in shortened_names]
    number_order = np.array(number_order)

    image_files = np.array(image_files, dtype=str)
    file_order = image_files[np.argsort(number_order)]

    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f))
                for f in file_order)
        
        img = next(imgs)

        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=2000, loop=1)






def main():
    LDOS_imshow('square', 3, 0, None, True, np.linspace(-2.0, 10.0, 16), [0], 1.0, 0.0, 1.0, 4)
    create_gif()

if __name__ == "__main__":
    main()
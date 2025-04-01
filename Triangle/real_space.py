import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from scipy.sparse import dok_matrix
from scipy.linalg import eigh, eigvals
import os
import h5py
from tqdm_joblib import tqdm_joblib, tqdm
from joblib import Parallel, delayed
from momentum_space import plot_phase_diagram

def calculate_square_lattice(length):
    return np.arange(length**2).reshape(length, length)


def calculate_square_hopping(lattice, pbc):
    y, x = np.where(lattice >= 0)[:]

    side_length = lattice.shape[0]
    
    dx = x - x[:, None]
    dy = y - y[:, None]
    if pbc:
        multipliers = tuple(product([-1, 0, 1], repeat=2))
        shifts = [(i * side_length, j * side_length) for i, j in multipliers]

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


    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)

    xp_mask = (dx == 1) & (abs_dy == 0)
    yp_mask = (dx == 0) & (dy == 1)

    xpyp_mask = (dx == 1) & (dy == 1)
    xnyp_mask = (dx == -1) & (dy == 1)

    Cx =   dok_matrix((side_length**2, side_length**2), dtype=complex)
    Sx =   dok_matrix((side_length**2, side_length**2), dtype=complex)
    Cy =   dok_matrix((side_length**2, side_length**2), dtype=complex)
    Sy =   dok_matrix((side_length**2, side_length**2), dtype=complex)
    CxCy = dok_matrix((side_length**2, side_length**2), dtype=complex)
    CySx = dok_matrix((side_length**2, side_length**2), dtype=complex)
    CxSy = dok_matrix((side_length**2, side_length**2), dtype=complex)
    I =    np.eye(side_length**2, dtype=complex)

    Sx[xp_mask] = 1j / 2
    Cx[xp_mask] = 1 / 2
    Cy[yp_mask] = 1 / 2
    Sy[yp_mask] = 1j / 2

    CxCy[xpyp_mask] = 1 / 4
    CxCy[xnyp_mask] = 1 / 4

    CxSy[xpyp_mask] = 1j / 4
    CxSy[xnyp_mask] = 1j / 4

    CySx[xpyp_mask] = 1j / 4
    CySx[xnyp_mask] = -1j / 4

    wannier_dict = {
        'Cx': Cx,
        'Cy': Cy,
        'Sx': Sx,
        'Sy': Sy,
        'CxCy': CxCy,
        'CySx': CySx,
        'CxSy': CxSy
    }

    wannier_dict = {k: (v + v.conj().T).toarray() for k, v in wannier_dict.items()}
    wannier_dict["I"] = I
    return wannier_dict


def calculate_square_hamiltonian(wannier_matrices, M, B_tilde, B, t1, t2):
    pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)

    Cx, Cy, Sx, Sy, CxCy, CySx, CxSy, I = wannier_matrices.values()
    d1 = t1 * Sx + t2 * CxSy
    d2 = t1 * Sy + t2 * CySx
    d3 = (M - 4 * B - 4 * B_tilde) * I + 2 * B * (Cx + Cy) + 2 * B_tilde * CxCy

    hamiltonian = np.kron(d1, pauli1) + np.kron(d2, pauli2) + np.kron(d3, pauli3)

    return hamiltonian


def calculate_projector(hamiltonian):
    eigenvalues, eigenvectors = eigh(hamiltonian, overwrite_a=True)
    lower_band = np.sort(eigenvalues)[:eigenvalues.size // 2]
    highest_lower_band = lower_band[-1]

    D = np.where(eigenvalues <= highest_lower_band, 1.0 + 0.0j, 0.0 + 0.0j)
    D_dagger = np.einsum('i,ij->ij', D, eigenvectors.conj().T)

    projector = eigenvectors @ D_dagger
    return projector


def calculate_square_bott_index(projector, lattice):
    Y, X = np.where(lattice >= 0)[:]

    X = np.repeat(X, 2)
    Y = np.repeat(Y, 2)

    Ux = np.exp(1j * 2 * np.pi * X / lattice.shape[0])
    Uy = np.exp(1j * 2 * np.pi * Y / lattice.shape[1])

    UxP = np.einsum('i,ij->ij', Ux, projector)
    UyP = np.einsum('i,ij->ij', Uy, projector)
    Ux_daggerP = np.einsum('i,ij->ij', Ux.conj(), projector)
    Uy_daggerP = np.einsum('i,ij->ij', Uy.conj(), projector)

    A = np.eye(projector.shape[0], dtype=np.complex128) - projector + projector @ UxP @ UyP @ Ux_daggerP @ Uy_daggerP
    bott_index = round(np.imag(np.sum(np.log(eigvals(A)))) / (2 * np.pi))

    return bott_index

# -------------

def compute_triangular_lattice(generation):
    def recursive_lattice(_gen):
        if _gen == 0:
            return np.array([0.0, -np.sqrt(3)/2, np.sqrt(3)/2, 1.0, -0.5, -0.5]).reshape(2, 3)
        else:
            smaller = recursive_lattice(_gen - 1)
            x_range = np.max(smaller[0]) - np.min(smaller[0])
            y_range = np.max(smaller[1]) - np.min(smaller[1])
            lattice_points = np.zeros((2, smaller.shape[1]*4))

            shifts = {
                "top": np.array([[0], [y_range]]),
                "left": np.array([[-x_range/2], [0]]),
                "right": np.array([[x_range/2], [0]]),
            }

            for i, shift in enumerate(shifts.values()):
                lattice_points[:, i::4] = smaller + shift
            
            flipped_max_y = np.min((smaller+shifts["left"])[1])
            flipped = smaller.copy()
            flipped[1] *= -1
            flipped[1] -= np.min(flipped[1]) - flipped_max_y

            lattice_points[:, 3::4] = flipped
            return np.unique(lattice_points, axis=1)
        
    coordinates = recursive_lattice(generation)
    xmin, ymin = np.min(coordinates[0]), np.min(coordinates[1])
    coordinates[0] -= xmin
    coordinates[1] -= ymin
    coordinates[0] *= 2/np.sqrt(3)
    coordinates[1] *= 2
    coordinates = np.unique(np.round(coordinates).astype(int), axis=1)
    sorted_idxs = np.lexsort((coordinates[0], coordinates[1]))
    coordinates = coordinates[:, sorted_idxs]
    
    lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
    lattice[coordinates[1], coordinates[0]] = np.arange(coordinates.shape[1])
    return lattice


def compute_sierpinski_triangle(generation):
    def recursive_fractal(_gen):
        if _gen == 0:
            return {"lattice_points": np.array([0.0, -np.sqrt(3)/2, np.sqrt(3)/2, 1.0, -0.5, -0.5]).reshape(2, 3), "triangular_hole_locations": None}
        else:
            fractal_dict = recursive_fractal(_gen - 1)
            smaller = fractal_dict["lattice_points"]

            x_range = np.max(smaller[0]) - np.min(smaller[0])
            y_range = np.max(smaller[1]) - np.min(smaller[1])
            fractal = np.zeros((2, smaller.shape[1]*3))

            shifts = {
                "top": np.array([[0], [y_range]]),
                "left": np.array([[-x_range/2], [0]]),
                "right": np.array([[x_range/2], [0]]),
            }

            for i, shift in enumerate(shifts.values()):
                fractal[:, i::3] = smaller + shift

            # Hole size, x location, y location
            location_of_hole = np.array([np.mean(fractal[0]), np.min(fractal[1]), _gen-1]).T

            if fractal_dict["triangular_hole_locations"] is not None:
                smaller_triangular_hole_locations = fractal_dict["triangular_hole_locations"]
                new_hole_locations = np.zeros((3, smaller_triangular_hole_locations.shape[1]*3))
                for i, shift in enumerate(shifts.values()):
                    new_hole_locations[2, i::3] = smaller_triangular_hole_locations[2] # hole size
                    new_hole_locations[[0,1], i::3] = smaller_triangular_hole_locations[[0,1]] + shift # x pos
                
                all_hole_points = np.append(new_hole_locations, location_of_hole.reshape(3, 1), axis=1)
            else:
                all_hole_points = location_of_hole.reshape(3, 1)

            return {"lattice_points": np.unique(fractal, axis=1), "triangular_hole_locations": np.unique(all_hole_points, axis=1)}
    fractal_dict = recursive_fractal(generation)
    coordinates = fractal_dict["lattice_points"]
    
    xmin, ymin = np.min(coordinates[0]), np.min(coordinates[1])
    coordinates[0] -= xmin
    coordinates[1] -= ymin
    coordinates[0] *= 2/np.sqrt(3)
    coordinates[1] *= 2
    coordinates = np.unique(np.round(coordinates).astype(int), axis=1)
    sorted_idxs = np.lexsort((coordinates[0], coordinates[1]))
    coordinates = coordinates[:, sorted_idxs]
    
    lattice = np.full((np.max(coordinates[1])+1, np.max(coordinates[0])+1), -1)
    lattice[coordinates[1], coordinates[0]] = np.arange(coordinates.shape[1])


    if fractal_dict["triangular_hole_locations"] is None:
        return {"lattice": lattice, "triangular_hole_locations": None}
    
    hole_locations = fractal_dict["triangular_hole_locations"]
    hole_locations[0] -= xmin
    hole_locations[1] -= ymin
    hole_locations[0] *= 2/np.sqrt(3)
    hole_locations[1] *= 2
    hole_locations = np.round(hole_locations).astype(int)
    return {"lattice": lattice, "triangular_hole_locations": hole_locations}


def tile_triangle(generation, doFractal=False):
    if doFractal:
        fractal_dict = compute_sierpinski_triangle(generation)
        lattice, triangular_hole_locations = fractal_dict["lattice"], fractal_dict["triangular_hole_locations"]
    else:
        lattice = compute_triangular_lattice(generation)
        triangular_hole_locations = None
    
    y, x = np.where(lattice >= 0)
    coordinates = np.vstack((x, y))

    arrays = [coordinates.copy()]
    if doFractal:
        arrays.append(triangular_hole_locations.copy())
        arrays[1][2] *= -1 # filp hole size as to imply that the hole is an upright triangle now
    
    for arr in arrays:
        arr[1] = np.max(coordinates[1])-arr[1] # flip upside down, adjust y position
        arr[0] += np.max(coordinates[0])//2 # shift x position to the right
  
    tiled_coords = np.hstack((coordinates, arrays[0]))
    
    y_min, x_min = np.min(tiled_coords[1]), np.min(tiled_coords[0])
    tiled_coords[0] -= x_min
    tiled_coords[1] -= y_min
    tiled_coords = np.unique(tiled_coords, axis=1)
    tiled_coords = tiled_coords[:, np.lexsort((tiled_coords[0], tiled_coords[1]))]

    lattice = np.full((np.max(tiled_coords[1])+1, np.max(tiled_coords[0])+1), -1)
    lattice[tiled_coords[1], tiled_coords[0]] = np.arange(tiled_coords.shape[1])

    if doFractal:
        tiled_hole_locations = np.hstack((triangular_hole_locations, arrays[1]))
        tiled_hole_locations[0] -= x_min
        tiled_hole_locations[1] -= y_min
        tiled_hole_locations = np.unique(tiled_hole_locations, axis=1)
    else:
        tiled_hole_locations = None

    return {"lattice" : lattice, "triangular_hole_locations" : tiled_hole_locations}


def calculate_triangle_distances(lattice, pbc):
    y, x = np.where(lattice >= 0)

    dx = x - x[:, np.newaxis]
    dy = y - y[:, np.newaxis]
    if pbc:
        x_range = np.max(x)
        y_range = np.max(y)

        displacements = {
            "center": np.array([0, 0]).T,
            "top_left": np.array([-x_range / 3 - 1, y_range + 3]).T,
            "left": np.array([-x_range * 2 / 3 - 2, 0]).T,
            "bottom_left": np.array([-x_range - 3, -y_range - 3]).T,
            "bottom": np.array([-x_range/3 - 1, -y_range - 3]).T,
            "bottom_right": np.array([x_range / 3 + 1, -y_range - 3]).T,
            "right": np.array([x_range* 2 / 3 + 2, 0]).T,
            "top_right": np.array([x_range + 3, y_range + 3]).T,
            "top": np.array([x_range / 3 + 1, y_range + 3]).T
        }

        x_shifted = np.empty((dx.shape[0], dx.shape[1], len(displacements)), dtype=dx.dtype)
        y_shifted = np.empty((dy.shape[0], dy.shape[1], len(displacements)), dtype=dy.dtype)
        for i, shift in enumerate(displacements.values()):
            x_shifted[:, :, i] = dx + shift[0]
            y_shifted[:, :, i] = dy + shift[1]

        distances = x_shifted**2 + y_shifted**2
        minimal_hop = np.argmin(distances, axis=-1)
        i_idxs, j_idxs = np.indices(minimal_hop.shape)
        
        dx = x_shifted[i_idxs, j_idxs, minimal_hop]
        dy = y_shifted[i_idxs, j_idxs, minimal_hop]
    
    return dx, dy

        
def calculate_triangle_hopping(dx, dy):

    # For NN--------------
    # In real space:
    # b1: (1, 0)
    # b2: (1 / 2, sqrt(3) / 2)
    # b2_tilde: (1 / 2, -sqrt(3) / 2)

    # In integer:
    # b1: (2, 0)
    # b2: (1, 3)
    # b2_tilde: (1, -3)

    b1_mask = (dx == 2) & (dy == 0)
    b2_mask = (dx == 1) & (dy == 3)
    b2_tilde_mask = (dx == 1) & (dy == -3)
    neg_b2_tilde_mask = (dx == -1) & (dy == 3)

    # For NNN--------------
    # In real space:
    # c1: (0,    sqrt(3))
    # c2: (3 / 2,  sqrt(3) / 2)
    # c3: (3 / 2, -sqrt(3) / 2)

    # In integer:
    # c1: (0, 6)
    # c2: (3, 3)
    # c3: (3, -3)

    c1_mask = (dx == 0) & (dy == 6)
    c2_mask = (dx == 3) & (dy == 3)
    c3_mask = (dx == 3) & (dy == -3)
    masks = [b1_mask, b2_mask, b2_tilde_mask, neg_b2_tilde_mask, c1_mask, c2_mask, c3_mask]
    return [m.astype(np.complex128) for m in masks]


def find_removal_sites(lattice, triangular_hole_locations, masks):
    def get_side_length(n):
        if n == 0:
            return 2
        elif n < 0:
            raise ValueError("n must be a non-negative integer")
        else:
            return 2*get_side_length(n-1) - 1
        
    hole_x, hole_y, hole_n = triangular_hole_locations
    remove_n0_holes_mask = (hole_n != 0)
    hole_x, hole_y, hole_n = hole_x[remove_n0_holes_mask], hole_y[remove_n0_holes_mask], hole_n[remove_n0_holes_mask]
    moving_vector_right = np.array([1, 3]).T
    moving_vector_left = np.array([-1, 3]).T
    hole_pos = np.vstack((hole_x, hole_y))

    bonds_to_remove = []
    for pos, n in zip(hole_pos.T, hole_n):
        # n is the relative generation size of the hole. Positive n means the hole is an inverted triangle, negative n means the hole is an upright triangle.
        horizontal_break_pos = pos + np.sign(n) * moving_vector_right
        h_break_end = horizontal_break_pos + np.sign(n) * np.array([-2, 0])

        right_vertical_break_pos = pos + (get_side_length(abs(n))-2) * np.sign(n) * moving_vector_right
        rv_break_end = right_vertical_break_pos + np.sign(n) * np.array([-1, 3])

        left_vertical_break_pos = pos + (get_side_length(abs(n))-2) * np.sign(n) * moving_vector_left
        lv_break_end = left_vertical_break_pos + np.sign(n) * np.array([1, 3])

        hor_i = lattice[horizontal_break_pos[1], horizontal_break_pos[0]]
        hor_j = lattice[h_break_end[1], h_break_end[0]]

        right_ver_i = lattice[right_vertical_break_pos[1], right_vertical_break_pos[0]]
        right_ver_j = lattice[rv_break_end[1], rv_break_end[0]]

        left_ver_i = lattice[left_vertical_break_pos[1], left_vertical_break_pos[0]]
        left_ver_j = lattice[lv_break_end[1], lv_break_end[0]]

        if hor_i >= 0 and hor_j >= 0:
            bonds_to_remove.append((hor_i, hor_j))
        if right_ver_i >= 0 and right_ver_j >= 0:
            bonds_to_remove.append((right_ver_i, right_ver_j))
        if left_ver_i >= 0 and left_ver_j >= 0:
            bonds_to_remove.append((left_ver_i, left_ver_j))

    b1_mask, b2_mask, b2_tilde_mask, neg_b2_tilde_mask, c1_mask, c2_mask, c3_mask = masks

    for i, j in bonds_to_remove:
        b1_mask[i, j] = 0
        b1_mask[j, i] = 0
        b2_mask[i, j] = 0
        b2_mask[j, i] = 0
        b2_tilde_mask[i, j] = 0
        b2_tilde_mask[j, i] = 0
    
    return b1_mask, b2_mask, b2_tilde_mask, neg_b2_tilde_mask, c1_mask, c2_mask, c3_mask
    

def fractal_wrapper(generation, pbc, doFractal=False):

    triangular_dict = tile_triangle(generation, doFractal=False)
    triangular_lattice = triangular_dict["lattice"]

    if doFractal:
        fractal_dict = tile_triangle(generation, True)
        fractal_lattice = fractal_dict["lattice"]
        triangular_hole_locations = fractal_dict["triangular_hole_locations"]

        dx, dy = calculate_triangle_distances(fractal_lattice, pbc)
        hopping_masks = calculate_triangle_hopping(dx, dy)
        hopping_masks = find_removal_sites(fractal_lattice, triangular_hole_locations, hopping_masks)

        Y_tri, X_tri = np.where(triangular_lattice >= 0)[:]
        bool_fractal_lattice = fractal_lattice >= 0
        fractal_site_mask = bool_fractal_lattice[Y_tri, X_tri]

        #hopping_masks = [mask[np.ix_(fractal_site_mask, fractal_site_mask)] for mask in hopping_masks]

    else:
        dx, dy = calculate_triangle_distances(triangular_lattice, pbc)
        hopping_masks = calculate_triangle_hopping(dx, dy)

    geometry_dict = {
        "lattice": triangular_lattice if not doFractal else fractal_lattice,
        "hopping_masks": hopping_masks,
    }
    return geometry_dict


def calculate_triangle_hamiltonian(hopping_masks, M, B_tilde, B = 1.0, A_tilde = 1.0, t = 1.0):
    b1_mask, b2_mask, b2_tilde_mask, neg_b2_tilde_mask, c1_mask, c2_mask, c3_mask = hopping_masks   

    I = np.eye(b1_mask.shape[0], dtype=np.complex128)

    d1 = (t / 2j) * (b1_mask) + (t / 4j) * (b2_mask + b2_tilde_mask)
    d1 += d1.conj().T

    d2 = (-t * np.sqrt(3) / 4j) * (b2_mask + neg_b2_tilde_mask)
    d2 += d2.conj().T

    d3 = (B) * (b1_mask + b2_mask + b2_tilde_mask)
    d3 += d3.conj().T
    d3 += (M - 4 * B) * (I)

    dtilde1 = (np.sqrt(3) * A_tilde / 8j) * (c2_mask + c3_mask)
    dtilde1 += dtilde1.conj().T

    dtilde2 = (-A_tilde / 2j) * (c1_mask) + (A_tilde / 4j) * (c2_mask - c3_mask)
    dtilde2 += dtilde2.conj().T

    dtilde3 = (B_tilde) * (c1_mask + c2_mask + c3_mask)
    dtilde3 += dtilde3.conj().T
    dtilde3 += (-6 * B_tilde) * (I)


    pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)

    H_from_d = np.kron(d1, pauli1) + np.kron(d2, pauli2) + np.kron(d3, pauli3)
    H_from_dtilde = np.kron(dtilde1, pauli1) + np.kron(dtilde2, pauli2) + np.kron(dtilde3, pauli3)
    hamiltonian = H_from_d + H_from_dtilde

    return hamiltonian


def calculate_triangle_bott_index(projector, lattice):
    Y, X = np.where(lattice >= 0)[:]

    X = np.repeat(X, 2)
    Y = np.repeat(Y, 2)

    def convert_from_discrete_to_triangular_basis(x, y):
        # First, convert to real space
        x = x.astype(float) / 2
        y = y.astype(float) / 3 * (np.sqrt(3) / 2)

        # Then, with basis vectors (a=1)
        # a1 = (1, 0)
        # a2 = (1/2, sqrt(3)/2)
        # We get:
        a1 = x - y / np.sqrt(3)
        a2 = y * 2 / np.sqrt(3)
        return a1, a2
    
    X, Y = convert_from_discrete_to_triangular_basis(X, Y)

    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)

    x_unitary = np.exp(1j * 2 * np.pi * X / Lx)
    y_unitary = np.exp(1j * 2 * np.pi * Y / Ly)

    x_unitary_proj = np.einsum('i,ij->ij', x_unitary, projector)
    y_unitary_proj = np.einsum('i,ij->ij', y_unitary, projector)
    x_unitary_dagger_proj = np.einsum('i,ij->ij', x_unitary.conj(), projector)
    y_unitary_dagger_proj = np.einsum('i,ij->ij', y_unitary.conj(), projector)

    I = np.eye(projector.shape[0], dtype=np.complex128)
    A = I - projector + projector @ x_unitary_proj @ y_unitary_proj @ x_unitary_dagger_proj @ y_unitary_dagger_proj

    bott_index = round(np.imag(np.sum(np.log(eigvals(A)))) / (2 * np.pi), 10)

    return bott_index

# ------------------

def square_bottom_line():
    M_values = np.linspace(-2.0, 10.0, 51)
    #M_values = [-2.0, 0.0, 3.0, 5.0, 7.0]

    bott_values = []

    lattice = calculate_square_lattice(9)
    wannier_matrices = calculate_square_hopping(lattice, pbc=True)

    for M in M_values:
        hamiltonian = calculate_square_hamiltonian(wannier_matrices, M, 0.0, 1.0, 1.0, 0.0)    
        projector = calculate_projector(hamiltonian)
        bott = calculate_square_bott_index(projector, lattice)
        bott_values.append(bott)
        print(f'M = {M:.2f}, Bott Index = {bott}')
    
    plt.scatter(M_values, bott_values, marker='o', color='blue')
    plt.xlabel('M')
    plt.ylabel('Bott Index')
    plt.title('Square Lattice :: Bott Index :: B_tilde = t2 = 0.0')
    plt.savefig("./Triangle/PhaseDiagrams/square_bottom_line_bott.png")
    plt.show()


def compute_triangle_bott_phase_diagram(generation, doFractal, M_range, B_tilde_range, resolution=(25, 25), B=1.0, t=1.0, A_tilde=1.0, output_file=None, directory='', overwrite=False):
    M_values = np.linspace(M_range[0], M_range[1], resolution[0])

    M_start = np.arange(-2.0, 6.0, 1.0)
    M_values = np.concatenate((M_start, np.linspace(6.0, 8.0, resolution[0]-len(M_start))))

    B_tilde_values = np.linspace(B_tilde_range[0], B_tilde_range[1], resolution[1])
    parameter_values = tuple(product(M_values, B_tilde_values))

    if output_file is None:
        root_fname = 'triangular_' if not doFractal else 'sierpinski_'
        output_file = os.path.join(directory, root_fname+str(generation)+f"_bott_phase_diagram_{resolution[0]}x{resolution[1]}.h5")
    
    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Use overwrite=True to overwrite.")
        return output_file


    geometry_dict = fractal_wrapper(generation, pbc=True, doFractal=doFractal)
    lattice = geometry_dict["lattice"]
    masks = geometry_dict["hopping_masks"]

    def compute_single(params):
        M, B_tilde = params
        hamiltonian = calculate_triangle_hamiltonian(masks, M, B_tilde, B=B, A_tilde=A_tilde, t=t)
        projector = calculate_projector(hamiltonian)
        bott = calculate_triangle_bott_index(projector, lattice)
        return [M, B_tilde, bott]

    with tqdm_joblib(tqdm(total=len(parameter_values), desc=f"Computing phase diagram for Chern number.")) as progress_bar:
        M_data, B_tilde_data, bott_data = np.array(Parallel(n_jobs=4)(delayed(compute_single)(params) for params in parameter_values), dtype=float).T

    with h5py.File(output_file, "w") as f:
        f.create_dataset(name = "M", data=M_data)
        f.create_dataset(name = "B_tilde", data=B_tilde_data)
        f.create_dataset(name =  "bott_data", data=bott_data.reshape(resolution).T)
    return output_file


def triangle_bottom_line():
    lattice = tile_triangle(4, doFractal=False)["lattice"]
    y, x = np.where(lattice >= 0)
    dx, dy = calculate_triangle_distances(lattice, pbc=True)

    masks = calculate_triangle_hopping(dx, dy)

    M_values = np.linspace(-2.0, 8.0, 51)
    bott_vals = []
    for M in M_values:
        H = calculate_triangle_hamiltonian(masks, M=M, B_tilde=0.0, B=1.0, A_tilde=0.0, t=1.0)

        #eigenvalues, eigenvectors = eigh(H, overwrite_a=True)
        #plt.scatter(np.arange(len(eigenvalues)), eigenvalues, marker='o', color='blue')
        #plt.title(f"M = {M:.2f}")
        #plt.show()

        projector = calculate_projector(H)
        bott_index = calculate_triangle_bott_index(projector, lattice)
        bott_vals.append(bott_index)
    
    plt.scatter(M_values, bott_vals, marker='o', color='blue')
    plt.xlabel('M')
    plt.ylabel('Bott Index')
    plt.title('Triangle Lattice :: Bott Index')
    plt.savefig("./Triangle/PhaseDiagrams/triangle_bott_index.png")
    plt.show()


# ------------------
def test():
    fout = compute_triangle_bott_phase_diagram(4, (-2.0, 8.0), (0.0, 0.5), resolution=(25, 25), B=1.0, t=1.0, A_tilde=0.0, output_file=None, directory='./Triangle/PhaseDiagrams', overwrite=False)
    with h5py.File(fout, 'r') as f:
        M_data = f["M"][:]
        B_tilde_data = f["B_tilde"][:]
        bott_data = f["bott_data"][:]
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    plot_phase_diagram(fig, ax, M_data, B_tilde_data, bott_data, title="Triangle Lattice Phase Diagram", labels=["M", "B_tilde"])   
    plt.savefig("./Triangle/PhaseDiagrams/triangle_bott_index_phase_diagram.png")
    plt.show()

if __name__ == "__main__":
    fout = compute_triangle_bott_phase_diagram(5, True, (-2.0, 8.0), (0.0, 0.5), resolution=(25, 25), B=1.0, t=1.0, A_tilde=0.0, directory='./Triangle/PhaseDiagrams', overwrite=False)
    with h5py.File(fout, 'r') as f:
        M_data = f["M"][:]
        B_tilde_data = f["B_tilde"][:]
        bott_data = f["bott_data"][:]
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    plot_phase_diagram(fig, ax, M_data, B_tilde_data, bott_data, title="Triangle Lattice Phase Diagram", labels=["M", "B_tilde"])
    plt.show()
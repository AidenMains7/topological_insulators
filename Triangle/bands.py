import numpy as np
import matplotlib.pyplot as plt


def plot_bands():
    # Define parameters
    M = 6.
    A_tilde = 0.
    B = 1.0
    B_tilde = M / 8 - 3/4
    B_tilde = 0.
    t1 = 1.0

    # Define d-vector function (your triangular version)
    def triangular_d_vector(kx, ky, M, B_tilde, B, t1, A_tilde):

        sqrt3 = np.sqrt(3)
        d1 = t1 * (np.sin(kx) + np.sin(kx / 2) * 0.5 * np.cos(sqrt3 / 2 * ky))
        d2 = t1 * (-sqrt3 * np.cos(kx / 2) * 0.5 * np.sin(sqrt3 / 2 * ky))
        d3 = M - 2 * B * (2 - np.cos(kx) - 2 * np.cos(kx / 2) * np.cos(sqrt3 * ky / 2))

        dtilde1 =  A_tilde * np.sin(3 * kx / 2) * (np.sqrt(3)/2) * np.cos(sqrt3 / 2 * ky)
        dtilde2 = -A_tilde * (np.sin(sqrt3 * ky) - 2 * np.cos(3 * kx / 2) * 0.5 * np.sin(sqrt3 / 2 * ky))
        dtilde3 = -2 * B_tilde * (3 - np.cos(sqrt3 * ky) - 2 * np.cos(3 / 2 * kx) * np.cos(sqrt3 / 2 * ky))

        d = np.array([d1 + dtilde1, d2 + dtilde2, d3 + dtilde3])
        return d


    # Define high symmetry points
    Gamma = np.array([0, 0])
    Kplus = np.array([4 * np.pi / 3, 0])
    Kminus = np.array([-4 * np.pi / 3, 0])
    M0 = np.array([0, 2 * np.pi / np.sqrt(3)])

    # Path through the BZ
    path = [Gamma, Kplus, M0, Kminus, Gamma]
    labels = [r'$\Gamma$', r'$K_+$', r'$M_0$', r'$K_-$', r'$\Gamma$'] # , 
    k_vals = []
    energies = []
    colors = []
    N = 201

    pauli1 = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli3 = np.array([[1, 0], [0, -1]], dtype=complex)

    for i in range(len(path)-1):
        start, end = path[i], path[i+1]
        for alpha in np.linspace(0, 1, N):
            kx, ky = start + alpha * (end - start)
            d = triangular_d_vector(kx, ky, M, B_tilde, B, t1, A_tilde)
            H = np.kron(d[0], pauli1) + np.kron(d[1], pauli2) + np.kron(d[2], pauli3)
            eigenvalues, eigenvectors = np.linalg.eig(H)
            alpha1, beta1 = eigenvectors[0].real
            alpha2, beta2 = eigenvectors[1].real

            colors.append([(alpha1**2 - beta1**2), (alpha2**2 - beta2**2)])
            k_vals.append(len(k_vals))
            energies.append([eigenvalues[0].real, eigenvalues[1].real])

    energies = np.array(energies)
    colors = np.array(colors)

    # Plot
    fig, ax = plt.subplots(1, 1)
    if True:
        ax.scatter(k_vals, energies[:, 0], label='Band +', c=colors[:, 0], cmap='Spectral', marker='.', zorder=2)
        ax.scatter(k_vals, energies[:, 1], label='Band -', c=colors[:, 1], cmap='Spectral', marker='.', zorder=2)
        ax.plot(k_vals, energies[:, 0],  c='k', ls='--', zorder=1, alpha=0.25)    
        ax.plot(k_vals, energies[:, 1],  c='k', ls='--', zorder=1, alpha=0.25)    
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
        cbar.set_label("Inversion")
    else:
        ax.plot(k_vals, energies[:, 0], label='Band +', c='blue', marker='.', zorder=2)
        ax.plot(k_vals, energies[:, 1], label='Band -', c='red', marker='.', zorder=2)


    ax.set_xticks(np.arange(len(path))*N, labels)
    ax.set_ylabel("Energy")
    ax.set_xlabel("Location in BZ")
    pathlabel = " → ".join(labels)
    ax.set_title(f"Energy Bands on {pathlabel} : (M={M}, " + r"$\tilde{B}$" + f"={B_tilde}, " + r"$\tilde{A}$" + f"={A_tilde})")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

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


def tile2(generation:int, doFractal:bool):
    if doFractal:
        gdict = compute_sierpinski_triangle(generation)
        lattice = gdict["lattice"]
        hole_locations = gdict["triangular_hole_locations"]
    else:
        lattice = compute_triangular_lattice(generation)
        hole_locations = None

    # get upside down triangle also
    y, x = np.where(lattice >= 0)[:]
    upright_coordinates = np.array([x, y])
    flipped_coordinates = upright_coordinates.copy()

    if hole_locations is not None:
        flipped_hole_locations = hole_locations.copy()
        flipped_hole_locations[1] *= -1
        flipped_hole_locations[1] -= np.min(flipped_hole_locations[1]) - 6
        flipped_hole_locations[0] -= np.min(flipped_hole_locations[0]) - np.min(hole_locations[0])
        flipped_hole_locations[2] *= -1

    flipped_coordinates[1] *= -1
    flipped_coordinates[1] -= np.min(flipped_coordinates[1])
    flipped_coordinates[0] -= np.min(flipped_coordinates[0])

    upright_shifts = [np.array([0, 0]).T, np.array([-np.max(x), 0]).T, np.array([-np.max(x)//2, -np.max(y)]).T]
    flipped_shifts = [np.array([-np.max(x)//2, 0]).T, np.array([0, -np.max(y)]).T, np.array([-np.max(x), -np.max(y)]).T]

    hexagon_coordinates = []
    for shift in upright_shifts:
        hexagon_coordinates.append(upright_coordinates + shift.reshape(2, 1))
    for shift in flipped_shifts:
        hexagon_coordinates.append(flipped_coordinates + shift.reshape(2, 1))

    hexagon_coordinates = np.concatenate(hexagon_coordinates, axis=1)
    hexagon_coordinates = np.unique(hexagon_coordinates, axis=1)


    if hole_locations is not None:
        hexagon_hole_locations = []
        for shift in upright_shifts:
            shift = np.array([shift[0], shift[1], 0])
            hexagon_hole_locations.append(hole_locations + shift.reshape(3, 1))
        for shift in flipped_shifts:
            shift = np.array([shift[0], shift[1], 0])
            hexagon_hole_locations.append(flipped_hole_locations + shift.reshape(3, 1))
        
        hexagon_hole_locations = np.concatenate(hexagon_hole_locations, axis=1)
        hexagon_hole_locations = np.unique(hexagon_hole_locations, axis=1)
        hexagon_hole_locations[0] -= np.min(hexagon_coordinates[0])
        hexagon_hole_locations[1] -= np.min(hexagon_coordinates[1])

        
    hexagon_coordinates[0] -= np.min(hexagon_coordinates[0])
    hexagon_coordinates[1] -= np.min(hexagon_coordinates[1])
    lattice = np.full((np.max(hexagon_coordinates[1])+1, np.max(hexagon_coordinates[0])+1), -1)
    lattice[hexagon_coordinates[1], hexagon_coordinates[0]] = np.arange(hexagon_coordinates.shape[1])

    return {"lattice": lattice, "hole_locations": hexagon_hole_locations}




if __name__ == "__main__":
    gdict = tile2(4, True)
    lattice = gdict["lattice"]
    x2, y2 = gdict["hole_locations"][:2]
    color = gdict["hole_locations"][2]
    plt.figure(figsize=(10, 10))
    plt.imshow((lattice+1).astype(bool), cmap="Blues", interpolation="nearest")
    plt.scatter(x2, y2, c=color, cmap="Reds")
    plt.show()
    if False:
        fdict = tile_triangle_into_hexagon(4, True)
        lattice = fdict["lattice"]
        y, x = np.where(lattice >= 0)[:]
        x2, y2 = fdict["hole_locations"][:2]
        color = fdict["hole_locations"][2]
        plt.figure(figsize=(10, 10))
        #plt.imshow(lattice, cmap="Blues", interpolation="nearest")
        plt.scatter(x2, y2, c=color, cmap="Reds")
        
        plt.scatter(x, y, alpha=0.5)
        plt.show()


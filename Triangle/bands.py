import numpy as np
import matplotlib.pyplot as plt


def plot_bands():
    # Define parameters
    M = 8.0
    A_tilde = 1.0
    B = 1.0
    B_tilde = 0.0
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
    point_pos_dict = {
        "G": np.array([0, 0]),
        "Kp": np.array([4 * np.pi / 3, 0]),
        "Km": np.array([-4 * np.pi / 3, 0]),
        "M0": np.array([0, 2 * np.pi / np.sqrt(3)]),
        "Mn": np.array([-np.pi, np.pi / np.sqrt(3)]),
        "Mp": np.array([np.pi, np.pi / np.sqrt(3)]),
    }
    point_label_dict = {
        "G": r"$\Gamma$",
        "Kp": r"$K_+$",
        "Km": r"$K_-$",
        "M0": r"$M_0$",
        "Mn": r"$M_{-1}$",
        "Mp": r"$M_{+1}$",
    }

    # Path through the BZ
    path_keys = ["Kp", "G", "M0", "Kp"]
    path = [point_pos_dict[key] for key in path_keys]
    labels = [point_label_dict[key] for key in path_keys]
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if False:
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


if __name__ == "__main__":
    plot_bands()
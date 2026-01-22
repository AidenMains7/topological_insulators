import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from itertools import product



def compute_d_vector(kx, ky, m0, hx, hy, hz, t:float=1.0, t0:float=1.0, a:float=1.0):
    d1 = t * np.sin(kx * a) + 1j * hx
    d2 = t * np.sin(ky * a) + 1j * hy
    d3 = m0 + t0 * (np.cos(kx * a) + np.cos(ky * a)) + 1j * hz
    return np.array([d1, d2, d3])

def compute_hamiltonian(kx, ky, m0, hx, hy, hz):
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    d = compute_d_vector(kx, ky, m0, hx, hy, hz)
    H = d[0] * pauli_x + d[1] * pauli_y + d[2] * pauli_z
    return H

def get_path_momenta(path, path_res):
    path_points_dict = {
        "G": (0.0, 0.0),
        "X": (np.pi, 0.0),
        "M": (np.pi, np.pi)
    }
    high_res_path = []
    path_values = [path_points_dict[point] for point in path]
    for i in range(len(path_values)):
        kx, ky = path_values[i]
        kx2, ky2 = path_values[(i + 1) % len(path_values)]
        high_res_path.append((np.linspace(kx, kx2, path_res), np.linspace(ky, ky2, path_res)))
    high_res_path = np.concatenate(high_res_path, axis=1)
    kx_path, ky_path = high_res_path
    return kx_path, ky_path

def compute_path_energies(m0, hx, hy, hz, path, path_res):
    kx_path, ky_path = get_path_momenta(path, path_res)
    energies = []
    for kx, ky in zip(kx_path, ky_path):
        H = compute_hamiltonian(kx, ky, m0, hx, hy, hz)
        evals = np.linalg.eigvals(H)
        energies.append(evals)
    return np.array(energies)


def get_energies_at_point(m0, hx, hy, hz, point):
    kx, ky = point
    H = compute_hamiltonian(kx, ky, m0, hx, hy, hz)
    evals = np.linalg.eigvals(H)
    return evals


def plot_interactive_band_structure():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    path = ["G", "X", "M"]
    path_res = 101
    ax_m0 = fig.add_axes([0.2, 0.15, 0.6, 0.03])
    ax_hx = fig.add_axes([0.2, 0.10, 0.6, 0.03])
    ax_hy = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    ax_hz = fig.add_axes([0.2, 0.00, 0.6, 0.03])

    init_m0 = 1.0
    init_hx = 0.0
    init_hy = 0.0
    init_hz = 0.0

    slider_m0 = Slider(ax_m0, 'm0', -2.0, 2.0, valinit=init_m0, valstep=0.1)
    slider_hx = Slider(ax_hx, 'hx', -1.0, 1.0, valinit=init_hx, valstep=0.1)
    slider_hy = Slider(ax_hy, 'hy', -1.0, 1.0, valinit=init_hy, valstep=0.1)
    slider_hz = Slider(ax_hz, 'hz', -1.0, 1.0, valinit=init_hz, valstep=0.1)

    xticks = [i * path_res for i in range(len(path) + 1)]
    xtick_labels = path + [path[0]]

    def update_axes_labels():
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Momentum Path")
        ax.set_ylabel("Energy")
        for x in xticks:
            ax.axvline(x=x, color='black', linestyle='--', linewidth=0.5)

    def update(val):
        m0, hx, hy, hz = slider_m0.val, slider_hx.val, slider_hy.val, slider_hz.val
        energies = compute_path_energies(m0, hx, hy, hz, path, path_res)
        ax.clear()
        ax.plot(energies.real[:, 0], label='Real Part', c='blue')
        ax.plot(energies.imag[:, 0], label='Imaginary Part', linestyle='--', c='blue')
        ax.plot(energies.real[:, 1], label='Real Part', c='red')
        ax.plot(energies.imag[:, 1], label='Imaginary Part', linestyle='--', c='red')
        update_axes_labels()
        ax.legend()
        fig.canvas.draw_idle()
    
    slider_m0.on_changed(update)
    slider_hx.on_changed(update)
    slider_hy.on_changed(update)
    slider_hz.on_changed(update)

    reset_ax = fig.add_axes([0.8, 0.925, 0.1, 0.04])
    button = Button(reset_ax, 'Reset', hovercolor='0.975')

    def reset(event):
        slider_m0.reset()
        slider_hx.reset()
        slider_hy.reset()
        slider_hz.reset()
    button.on_clicked(reset)

    update(None)


    plt.show()


def main():
    resolution = (101, 101)
    m0_extent = (-3.0, 3.0)
    h_extent = (-1.0, 1.0)
    m0_values = np.linspace(m0_extent[0], m0_extent[1], resolution[0])
    h_values = np.linspace(h_extent[0], h_extent[1], resolution[1])

    m0_values, h_values = np.meshgrid(m0_values, h_values)
    m0_values, h_values = m0_values.flatten(), h_values.flatten()


    point_dict = {
        "G": (0.0, 0.0),
        "X": (np.pi, 0.0),
        "M": (np.pi, np.pi)
    }
    point = point_dict["M"]
    energies = []
    for m0, h in zip(m0_values, h_values):
        evals = get_energies_at_point(m0, h, 0.0, 0.0, point)
        energies.append(evals)
    energies = np.round(np.array(energies), 5)


    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    real_part = energies.real[:, 0].reshape(resolution)
    imag_part = energies.imag[:, 0].reshape(resolution)
    magnitude = np.sqrt(energies.real[:, 0]**2 + energies.imag[:, 0]**2).reshape(resolution)

    axs[0].imshow(real_part, extent=(m0_extent[0], m0_extent[1], h_extent[0], h_extent[1]), origin='lower', aspect='auto')
    axs[0].set_title(f'Real part of lowest energy band at ({point[0]:.3f}, {point[1]:.3f})')
    axs[1].imshow(imag_part, extent=(m0_extent[0], m0_extent[1], h_extent[0], h_extent[1]), origin='lower', aspect='auto')
    axs[1].set_title(f'Imaginary part of lowest energy band at ({point[0]:.3f}, {point[1]:.3f})')
    axs[2].imshow(magnitude, extent=(m0_extent[0], m0_extent[1], h_extent[0], h_extent[1]), origin='lower', aspect='auto')
    axs[2].set_title(f'Magnitude of lowest energy band at ({point[0]:.3f}, {point[1]:.3f})')

    plt.colorbar(axs[0].images[0], ax=axs[0], label='Energy')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
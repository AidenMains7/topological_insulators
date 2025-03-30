import numpy as np
from matplotlib import pyplot as plt

m_electron = 9.10938356e-31
c = 299792458
a_const = (-13.6)**2 / (2*m_electron*c*c)
h_ev = 4.1357e-15


def fine_structure_correction(n, nprime, j, jprime):
    E_n_ev = (3 - (4*n)/(j+1/2)) / (n**2)
    E_nprime_ev = (3 - (4*nprime)/(jprime+1/2)) / (nprime**2)
    return E_n_ev - E_nprime_ev


def find_E_n(n):
    return -13.6 / n**2


def E_transition(j, jprime, n, nprime):
    return (find_E_n(n) - find_E_n(nprime)) + fine_structure_correction(n, nprime, j, jprime)


def energy_to_wavelenght(E):
    return h_ev * c / E


def get_quantum_numbers(n):
    l_values = np.arange(0, n)
    j_values = np.full((2, l_values.size), np.nan)
    j_values[0] = l_values - 1/2
    j_values[1] = l_values + 1/2
    fix_values = np.where(j_values <= 0)
    j_values[fix_values] = np.nan
    return l_values, j_values

def get_unique_values(values, removeNan=True):
    values = np.unique(values.flatten())
    if removeNan:
        values = values[~np.isnan(values)]
    return values
    



def main():
    n, nprime = 6, 5
    l_values, all_j_values = get_quantum_numbers(n)
    lprime_values, all_jprime_values = get_quantum_numbers(nprime)

    j_values = get_unique_values(all_j_values)
    jprime_values = get_unique_values(all_jprime_values)

    looped_j = []
    looped_jprime = []
    looped_energy = []
    print(f"Transition from n = {n} to n = {nprime}")
    for j in j_values:
        for jprime in jprime_values:
            if jprime > j:
                continue
            else:
                E = E_transition(j, jprime, n, nprime)
                looped_j.append(j)
                looped_jprime.append(jprime)
                looped_energy.append(E)
       

    looped_energy = np.array(looped_energy)
    looped_j = np.array(looped_j)
    looped_jprime = np.array(looped_jprime)

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = plt.Normalize(min(looped_energy), max(looped_energy))
    colors = plt.cm.viridis(norm(looped_energy))

    # Plot arrows for transitions
    ones_arr = np.ones(len(looped_j))
    ax.scatter(n*ones_arr, looped_j, c='k')
    ax.scatter(nprime*ones_arr, looped_jprime, c='k')

    # Calculate the distances between the points
    distances = np.sqrt((nprime - n)**2 + (looped_jprime - looped_j)**2)

    # Scale the quiver arrows to have magnitude equal to the distances
    ax.quiver(n*ones_arr, looped_j, (nprime-n)*ones_arr, looped_jprime - looped_j, 
              color=colors, scale=1/distances*2, scale_units='xy')

    # Set axis labels and title
    ax.set_xlabel("n (Principal Quantum Number)")
    ax.set_ylabel("j (Total Angular Momentum)")
    ax.set_title("Quantum Transitions with Wavelength-Based Coloring")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
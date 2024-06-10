from project_execute import _many_bott
from disorder_phase_diagram import _read_npz_data, _save_npz_data, compare_data
import matplotlib.pyplot as plt
import numpy as np

def run_computation_bott(filename:str) -> None:
    """
    
    
    """

    method = 'symmetry'
    order = 3
    pad_w = 0
    pbc = True
    n = 10
    M_values = np.linspace(-2, 12, 10)
    B_tilde_values = np.linspace(0, 2, 10)
    E_F = 0.0
    num_jobs = 4
    cores_per_job=1
    sparse=False
    progress=True


    bott_array = _many_bott(method, order, pad_w, pbc, n, M_values, B_tilde_values, E_F, num_jobs, cores_per_job, sparse, progress)

    filename = _save_npz_data(filename, bott_array)


def plot_bott(filename:str) -> None:

    bott_arr = _read_npz_data(filename)

    M = bott_arr[0]
    B_tilde = bott_arr[1]
    bott = bott_arr[2]

    Y, X = np.meshgrid(np.unique(M), np.unique(B_tilde))

    N = np.unique(M).size

    arrs = np.split(bott, N)
    
    Z0 = np.stack(arrs, axis=0)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z0.T, cmap='viridis')

    ax.set_ylabel('M')
    ax.set_xlabel('B_tilde')
    ax.set_zlabel('Bott Index')      

    plt.show()




#---------main func implementation---------

def main():
    filename = f"bott.npz"
    run_computation_bott(filename)
    plot_bott(filename)


def main2():
    filename = "bott_0.npz"
    plot_bott(filename)

def main3():
    file1 = "bott_0.npz"
    file2 = "bott_1.npz"

    v = compare_data(file1, file2)
    print(v)

if __name__ == "__main__":
    main2()
from project_execute import _many_bott
from disorder_phase_diagram import _read_npz_data, _save_npz_data, compare_data, get_all_npz
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


def plot_bott(filename:str, doShow:bool=True, doSave:bool=True) -> None:

    bott_arr = _read_npz_data(filename)

    if bott_arr.shape[0] != 3:
        raise Exception(f"The array from the {filename} is not of proper shape. The first dimension must have size 3. ")

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


    if doSave and filename.endswith(".npz"):
        figname = filename[:-4]+".png"
        plt.savefig(figname)

    if doShow:
        plt.show()



def plot_all_npz_bott(dir:str=".") -> None:
    """
    Plots all .npz files in current directory which start with "bott"
    """
    files = get_all_npz(dir)

    for f in files:
        if f.startswith("disorder"):
            print(f)
            plot_bott(f, False, True)


#---------main func implementation---------

def main():
    filename = f"bott.npz"
    run_computation_bott(filename)
    plot_bott(filename)


def main2():
    try:
        plot_all_npz_bott()
    except Exception as e:
        print(f"Caught exception: {e}")

def main3():
    plot_bott("bott_1.npz")

if __name__ == "__main__":
    main3()
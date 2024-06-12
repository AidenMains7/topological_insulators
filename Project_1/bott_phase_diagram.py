from project_execute import _many_bott
from disorder_phase_diagram import _read_npz_data, _save_npz_data, compare_data, get_all_npz
import matplotlib.pyplot as plt
import numpy as np



def plot_bott(filename:str, doShow:bool=True, doSave:bool=True) -> None:
    """
    Plot the data from given filename; data from _many_bott()
    """

    #read array from file
    bott_arr, params = _read_npz_data(filename)

    #check that array is of proper shape
    if bott_arr.shape[0] != 3:
        raise Exception(f"The array from the {filename} is not of proper shape. The first dimension must have size 3. ")


    #values
    M = bott_arr[0]
    B_tilde = bott_arr[1]
    bott = bott_arr[2]

    #meshgrid of  M and B_tilde
    Y, X = np.meshgrid(np.unique(M), np.unique(B_tilde))

    #number of unique values
    N = np.unique(M).size

    #organize the bott array into a surface over the meshgrid
    arrs = np.split(bott, N)
    Z0 = np.stack(arrs, axis=0)

    #create figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z0.T, cmap='viridis')

    ax.set_ylabel('M')
    ax.set_xlabel('B_tilde')
    ax.set_zlabel('Bott Index')      

    #check proper filename format
    if doSave and filename.endswith(".npz"):
        figname = filename[:-4]+".png"
        plt.savefig(figname)
    elif not filename.endswith(".npz"):
        raise Exception("Filename does not end in '.npz'; trying to save figure as '.png'")

    #whether to plot
    if doShow:
        plt.show()


def plot_all_npz_bott(dir:str=".") -> None:
    """
    Plots all .npz files in current directory which start with "bott"
    """
    files = get_all_npz(dir)

    for f in files:
        if f.startswith("bott"):
            print(f)
            plot_bott(f, False, True)


def run_computation_bott(filename:str, doPlot:bool=True, doShow:bool=True, doSave:bool=True) -> None:
    """
    Run _many_bott() with specified parameters; save file with given filename.
    """

    #set parameters
    parameters = dict(
        method = "symmetry",
        order = 3,
        pad_width = 0,
        pbc = True,
        n = 0,
        M_values =         np.linspace(-2.0, 12.0, 1),
        B_tilde_values =   np.linspace(0.0, 2.0, 1),
        E_F = 0.0,
        num_jobs = 4,
        cores_per_job = 1,
        sparse = False,
        progress = True
    )

    bott_array = _many_bott(**parameters)
    end_f = _save_npz_data(filename, data=bott_array, parameters=parameters)
    
    if doPlot:
        plot_bott(end_f, doShow, doSave)


#---------main func implementation---------

def main():
    filename = f"bott.npz"
    run_computation_bott(filename)



def main2():
    try:
        plot_all_npz_bott()
    except Exception as e:
        print(f"Caught exception: {e}")

def main3():
    plot_bott("bott_3.npz")

if __name__ == "__main__":
    main()
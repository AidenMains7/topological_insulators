import numpy as np
from project_execute import many_bott, computation
from datetime import datetime

import latex
import matplotlib.pyplot as plt
import scienceplots

import cProfile

from os import walk


#plt.style.use([]'science', 'pgf])

def timely_filename() -> str:
    """
    Current date and time in string format
    """
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hr = now.hour
    minute = now.minute
    time_string=f"{month}_{day}_{year}__{hr}_{minute}"
    return time_string


def _read_npz_data(filename:str) -> np.ndarray:
    """
    Will read .npz file

    Parameters:
    filename (str): Base filename with .npz extension
    """
    file_data = np.load(filename)
    data = file_data['arr_0']
    return data


def _phase_plot(filename:str, doShow:bool, doSave:bool) -> None:
    """
    Will read data from a specifed .npz file and plot for which has non-zero inital Bott Index.
    """
    try:
        data = _read_npz_data(filename)
    except Exception as e:
        print(f"Error with {filename}: {e}")
        data = None

    if data is not None:
        num = data.shape[0]

        fig, ax = plt.subplots()
        ax.set_title("Bott Index vs. Disorder Strength")
        ax.set_xlabel("Disorder Strength (W)")
        ax.set_ylabel("Bott Index")
        ax.set_ylim([-2, 2])


        for i in range(num):
            if data[i][0,1] is not np.nan:
                x = data[i][0] #disorder value
                y = data[i][1] #bott index after disorder

                ax.plot(x, y)

        if doSave and filename.endswith(".npz"):
            figname = filename[:-4]+".png"
            plt.savefig(figname)

        if doShow:
            plt.show()
    

def run_profile() -> None:
    pass


def make_figures_dir(dir:str="."):
    """
    Will make a figure using phase_plot() and save it as a png for all .npz files in the directory.
    """
    f = []
    for (dirpath, dirnames, filenames) in walk(dir):
        f.extend(filenames)
        break

    for file in f:
        if file.endswith(".npz"):
            _phase_plot(file, False, True)


def run_computation(filename:str, doPhase:bool=True, doShow:bool=True, doSave:bool=True) -> None:
    """
    Will run computation with specified parameters and save to a .npz file

    Parameters:
    filename (str): Base filename to save data, with extension
    doPhase (bool): Whether to plot the phase diagram

    """

    #set parameters
    method = "symmetry"
    order = 3
    pad_w = 0
    pbc=True
    n=10
    M_values = np.linspace(6,12,50)
    B_tilde_values = np.linspace(1,2,50)
    W_values = np.linspace(0.5,7.5,20)
    iter_p_d = 20
    num_jobs=4 #28 if on workstation

    #run the computation
    data = computation(method, order, pad_w, pbc, n, M_values, B_tilde_values, W_values, iter_p_d, num_jobs=num_jobs, E_F=0.0, progresses=(True, True, False))

    #save the data to a .npz file
    np.savez(filename, data)

    #do phase diagram
    if doPhase:
        _phase_plot(filename, doShow, doSave)


#-------------main function implementation-------------------------
def main():
    filename = f"bott_disorder_{timely_filename()}.npz"
    run_computation(filename)

if __name__ == "__main__":
    main()


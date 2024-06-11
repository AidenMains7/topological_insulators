"""

Functions:
timely_filename() : gets the current date/time as string format
_read_npz_data()  : read data from a npz file into array, only for use within here
_phase_plot()     : plot phase data 
make_figures_dir(): will run _phase_plot() for all .npz files in current working directory
run_computation() : run the computation and create phase plot for specified parameters
"""

import numpy as np
from project_execute import computation, computation_alt
from datetime import datetime
import matplotlib.pyplot as plt
import os

import latex
import scienceplots
plt.style.use(['science', 'pgf'])

def _save_npz_data(filename:str, data:np.ndarray, iter:int=0) -> str:
    """
    Saves data to the specified filename with "_{iter}" appended before extension. If file of iter exists, will increment by one.
    """

    try:
        temp_fname = filename[:-4]+f"_{iter}.npz"
        if os.path.isfile(temp_fname):
            filename = _save_npz_data(filename, data, iter+1)
        else:
            filename = filename[:-4]+f"_{iter}.npz"
            np.savez(filename, data)
        return filename

    except Exception as e:
        print(f"Exception: {e}")
        return None


def _read_npz_data(filename:str) -> np.ndarray:
    """
    Will read .npz file

    Parameters:
    filename (str): Base filename with .npz extension
    """
    file_data = np.load(filename)
    data = file_data['arr_0']
    return data


def plot_disorder(filename:str, doShow:bool, doSave:bool) -> None:
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

        fig, ax = plt.subplots(1,1,figsize=(16,9))
        ax.set_title("Bott Index vs. Disorder Strength")
        ax.set_xlabel("Disorder Strength (W)")
        ax.set_ylabel("Bott Index")
        ax.set_ylim([-3, 3])



        color_list = ['#845B97', '#0C5DA5', '#00B945', '', '#FF9500', '#FF2C00', '#474747']
        ax.hlines(0, 0, 8, colors='k', ls='--')
        for i in range(num):
            if data[i][0,1] is not np.nan:
                x = data[i][0] #disorder value
                y = data[i][1] #bott index after disorder

                if True:
                    if y[0] not in [-3, -2, -1, 1, 2, 3]:
                        print(f"Initial bott index is not in [-3, -2, -1, 1, 2, 3]. Value is {y[0]}")
                        ax.plot(x, y, label=f"{i}", c='black')
                    
                    try:
                        ax.plot(x, y, label=f"{i}", c=color_list[int(y[0]+3)])

                    except Exception as e:
                        print(f"Caught exception: {e}")

                else:
                    ax.plot(x, y, label=f"{i}")

        if False:
            ax.legend(True)

        if doSave and filename.endswith(".npz"):
            figname = filename[:-4]+".png"
            plt.savefig(figname)

        if doShow:
            plt.show()
    

def get_all_npz(dir:str=".") -> list:
    """
    Gets a list of all .npz files in the provided directory
    
    """
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        all_files.extend(filenames)
        break

    files = []
    for f in all_files:
        if f.endswith(".npz"):
            files.append(f)

    return files


def plot_all_npz_disorder(dir:str=".") -> None:
    """
    Plots all .npz files in current directory which start with "disorder"
    """
    files = get_all_npz(dir)

    for f in files:
        if f.startswith("disorder"):
            print(f)
            plot_disorder(f, False, True)


def run_computation_disorder(filename:str, doPhase:bool=True, doShow:bool=True, doSave:bool=True, alternate:bool=False) -> None:
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
    M_values =       np.linspace(6.0, 12.0, 5)
    B_tilde_values = np.linspace(1.0, 2.0,  1)
    W_values =       np.linspace(0.5, 7.5,  5)
    iter_p_d = 5
    num_jobs = 4 #28 if on workstation



    #run the computation
    if not alternate:
        data = computation(method, order, pad_w, pbc, n, M_values, B_tilde_values, W_values, iter_p_d, num_jobs=num_jobs, E_F=0.0, progresses=(True, True, False))
    else:
        data = computation_alt(method, order, pad_w, pbc, n, M_values, B_tilde_values, W_values, iter_p_d, num_jobs=num_jobs, E_F=0.0, progresses=(True, True, False, True))


    #save the data to a .npz file
    filename = _save_npz_data(filename, data)

    #do phase diagram
    if doPhase:
        plot_disorder(filename, doShow, doSave)


def compare_data(f1:str, f2:str) -> float:
    """
    Compare the data between two .npz files. For the same parameter values, one would expect the average to be 0.
    
    """
    data1 = _read_npz_data(f1)
    data2 = _read_npz_data(f2)

    diff = data1-data2
    avg = np.average(diff)
    return avg

#-------------main function implementation-------------------------
def main():
    filename = f"disorder.npz"
    run_computation_disorder(filename, alternate=True)

def main2():
    plot_all_npz_disorder()


if __name__ == "__main__":
    main()

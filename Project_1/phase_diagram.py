"""
Handles saving and displaying data via .npz files.
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os, sys

# Check to use latex style as the workstation pc does not have latex installed; workstation pc is in python 3.9
if sys.version_info[1] > 9:
    import latex, scienceplots
    plt.style.use(['science', 'pgf'])


def _save_npz_data(filename:str, data:np.ndarray, parameters:dict, iter:int=0) -> str:
    """
    Saves data to the specified filename with "_{iter}" appended before extension. If file of iter exists, will increment by one.
    """

    try:
        temp_fname = filename[:-4]+f"_{iter}.npz"
        if os.path.isfile(temp_fname):
            filename = _save_npz_data(filename, data=data, parameters=parameters, iter=iter+1)
        else:
            filename = filename[:-4]+f"_{iter}.npz"
            np.savez(filename, data=data, parameters=parameters)
        return filename

    except Exception as e:
        print(f"Exception: {e}")
        return None


def _read_npz_data(filename:str) -> 'tuple[np.ndarray, dict]':
    """
    Will read .npz file

    Parameters:
    filename (str): Base filename with .npz extension
    """
    file_data = np.load(filename, allow_pickle=True)
    
    #if .npz file is of first form
    if file_data.get('arr_0') is not None:
        data = file_data['arr_0']
        parameters = None

    #if .npz file has parameters (second form)
    else:
        data = file_data['data']
        parameters = file_data['parameters']

    return data, parameters


def plot_disorder(filename:str, doShow:bool, doSave:bool, title:str=None) -> None:
    """
    Will read data from a specifed .npz file and plot for which has non-zero inital Bott Index.
    """
    try:
        data, params = _read_npz_data(filename)
        if params is not None:
            params = params[()]

    except Exception as e:
        print(f"Error with {filename}: {e}")
        data = None

    if data is not None:
        num = data.shape[0]

        fig, ax = plt.subplots(1,1,figsize=(16,9))

        #Set title, labels
        if title is None:
            if params is not None:
                plt.title(f"Bott Index of Fractal Lattice, Method of {params['method']}, Order = {params['order']}")
        else:
            plt.title(title)

        ax.set_xlabel("Disorder Strength (W)")
        ax.set_ylabel("Bott Index")
        ax.set_ylim([-3, 3])


        #list of colors and respective bott index
        colors = ['#845B97', '#0C5DA5', '#00B945', '', '#FF9500', '#FF2C00', '#474747']
        bott_vals = [-3, -2, -1, 0, 1, 2, 3]

        #horizontal line at 0
        if params is not None:
            maxval = params['W_values'].max()
        else:
            maxval = 10
            
        ax.hlines(0, 0, maxval, colors='k', ls='--')

        for i in range(num):
            if data[i][0,1] is not np.nan:
                x = data[i][0] #disorder value
                y = data[i][1] #bott index after disorder

                if True:
                    if y[0] not in bott_vals:
                        print(f"Initial bott index is not in [-3, -2, -1, 1, 2, 3]. Value is {y[0]}")
                        ax.scatter(x, y, c='black', marker='.')
                    
                    try:
                        ax.scatter(x, y, c=colors[int(y[0]+3)], marker='.')

                    except Exception as e:
                        print(f"Caught exception: {e}")

                else:
                    ax.plot(x, y, label=f"{i}")

        #Add legend
        if True:
            #Dummy artists
            colors.remove('')
            bott_vals.remove(0)
            dummy_artists = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
            ax.legend(dummy_artists, bott_vals)

        if doSave and filename.endswith(".npz"):
            figname = filename[:-4]+".png"
            plt.savefig(figname)

        if doShow:
            plt.show()
    

def plot_bott(filename:str, doShow:bool=True, doSave:bool=True, title:str=None) -> None:
    """
    Plot the data from given filename; data from _many_bott()
    """

    #read array from file
    try:
        data, params = _read_npz_data(filename)
        if params is not None:
            params = params[()]

    except Exception as e:
        print(f"Error with {filename}: {e}")
        data = None

    if data is not None:

        #check that array is of proper shape
        if data.shape[0] != 3:
            raise Exception(f"The array from the {filename} is not of proper shape. The first dimension must have size 3. ")


        #values
        M = data[0]
        B_tilde = data[1]
        bott = data[2]

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
        ax.set_title(title)      

        #check proper filename format
        if doSave and filename.endswith(".npz"):
            figname = filename[:-4]+".png"
            plt.savefig(figname)
        elif not filename.endswith(".npz"):
            raise Exception("Filename does not end in '.npz'; trying to save figure as '.png'")

        #whether to plot
        if doShow:
            plt.show()


def plot_bott_imshow(filename:str, doShow:bool=True, doSave:bool=True, title:str=None) -> None:
    """
    Phase diagram using imshow. 
    """
    #read array from file
    try:
        data, params = _read_npz_data(filename)
        if params is not None:
            params = params[()]

    except Exception as e:
        print(f"Error with {filename}: {e}")
        data = None

    if data is not None:

        #check that array is of proper shape
        if data.shape[0] != 3:
            raise Exception(f"The array from the {filename} is not of proper shape. The first dimension must have size 3. ")

        #values
        M = data[0]
        B_tilde = data[1]
        bott = data[2]

        #unique values
        M_vals = np.unique(M)
        B_tilde_vals = np.unique(B_tilde)

        #bounds for plot
        x_bounds = (M_vals.min(), M_vals.max())
        y_bounds = (B_tilde_vals.min(), B_tilde_vals.max())

        #organize the bott array into a surface over the meshgrid
        arrs = np.split(bott, M_vals.size)
        Z = np.stack(arrs, axis=0)

        fig = plt.figure(figsize=(10,10))
        plt.imshow(Z, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], origin='lower', aspect='auto', cmap='viridis')
        cbar = plt.colorbar(label='Bott Index')
        cbar.set_label('Bott Index', rotation=0)

        x_ticks = np.linspace(x_bounds[0], x_bounds[1], 5)
        y_ticks = np.linspace(y_bounds[0], y_bounds[1], 5)
        plt.xticks(ticks=x_ticks, labels=np.round(x_ticks, 2))
        plt.yticks(ticks=y_ticks, labels=np.round(y_ticks, 2))

        plt.xlabel("M")
        plt.ylabel("B_tilde", rotation=0)

        #Set title
        if title is None:
            if params is not None:
                plt.title(f"Bott Index of Fractal Lattice, Method of {params['method']}, Order = {params['order']}")
        else:
            plt.title(title)
        
        #check proper filename format
        if doSave and filename.endswith(".npz"):
            figname = filename[:-4]+".png"
            plt.savefig(figname)
        elif not filename.endswith(".npz"):
            raise Exception("Filename does not end in '.npz'; trying to save figure as '.png'")
        
        # Display figure
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


def plot_all_npz(dir:str=".") -> None:
    """
    Plots all .npz files in current directory, save image as .png
    """
    files = get_all_npz(dir)

    for f in files:
        if f.startswith("disorder"):
            print(f)
            plot_disorder(f, False, True)
        elif f.startswith("bott"):
            print(f)
            plot_bott_imshow(f, False, True)



#-------------main function implementation-------------------------
def main():
    pass


if __name__ == "__main__":
    main()

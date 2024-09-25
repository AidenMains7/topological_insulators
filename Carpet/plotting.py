import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

import sys
sys.path.append(".")
from Carpet.filesaving import return_all_file_type
#import scienceplots
#plt.style.use(['science', 'ieee'])


def plot_imshow(fig:figure.Figure, ax:axes.Axes, X:np.ndarray, Y:np.ndarray, Z:np.ndarray, cmap:str='viridis', doDiscreteCmap:bool=False) -> None:
    """
    
    Parameters:

    
    Returns:
    fig (Figure): The updated figure
    ax (Axes): The updated axes
    cbar (Colorbar): The colorbar
    
    """

    x_bounds = (X.min(), X.max())
    y_bounds = (Y.min(), Y.max())
    cbar_bounds = np.linspace(Z.min(), Z.max(), np.unique(Z).size) 


    cbar_ticks = np.linspace(cbar_bounds[0], cbar_bounds[-1], np.unique(Z).size)

    if doDiscreteCmap:
        cmap = plt.cm.get_cmap(cmap, int(Z.max()-Z.min()+1))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(Z, cmap=cmap, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], aspect='auto')

    cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks, spacing='uniform', cmap=cmap)

    return fig, ax, cbar



def reshape_imshow_data(data:np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Assuming a 3xN array such that row 0 is X, row 1 is Y, and row 2 is Z.
    """
    X = data[0, :]
    Y = data[1, :]
    Z = data[2, :]

    X_unique, Y_unique = np.unique(X), np.unique(Y)
    X_mesh, Y_mesh = np.meshgrid(X_unique, Y_unique)
    Z_surface = np.empty(X_mesh.shape)

    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        z = Z[i]

        x_idx = np.where(X_unique == x)[0][0]
        y_idx = np.where(Y_unique == y)[0][0]
        Z_surface[y_idx, x_idx] = z
    
    return X, Y, np.flipud(Z_surface)



def plot_bott(infile:str, doShow:bool=True, doSave:bool=False, outfile:str=None, cmap:str='viridis', figsize:tuple=(10, 10)) -> None:
    """
    Wrapper function for reshape_imshow_data() tailored for data from computing the Bott Index of a lattice. 
    """

    filedata = np.load(infile, allow_pickle=True)
    data, params = filedata['data'], filedata['parameters'][()]


    boundary_cond_str = 'PBC' if params['pbc'] else 'OBC'
    if params['method'] != 'symmetry':
        title = f"Bott Index of lattice. Method of {params['method']}. {boundary_cond_str}. Generation {params['order']}."
    else:
        title = f"Bott Index of lattice. Method of {params['method']}, n = {params['n']:.0f}. {boundary_cond_str}. Generation {params['order']}."

    X, Y, Z = reshape_imshow_data(data)

    if Y.shape[0] <= 1 or X.shape[0] <= 1:
        raise ValueError("One of either X or Y dimensions has size <= 1.")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    fig, ax, cbar = plot_imshow(fig, ax, X, Y, Z, cmap, True)

    ax.set_xticks(np.linspace(X.min(), X.max(), 8))
    ax.set_yticks(np.linspace(Y.min(), Y.max(), 5))
    ax.set_title(title)
    ax.set_xlabel('M')
    ax.set_ylabel('B_tilde')
    cbar.set_label('Bott Index')


    if doSave:
        if outfile == None:
            plt.savefig(infile[:-4]+'.png')
        else:
            plt.savefig(outfile)

    if doShow:
        plt.show()



def plot_series(fig:figure.Figure, ax:axes.Axes, X:np.ndarray, Y:np.ndarray, series_labels:list, series_colors:list, plot_type:str='scatter', marker:str=None):
    """
    
    Parameters:
    X (ndarray): A 1D or 1xN array of X values.
    Y (ndarary): A MxN array, such that each row is a separate series. 
    series_labels (list): A list of size M, a label for each series.
    series_colors (list) A list of size M, a color for each series.
    
    """

    if plot_type not in ['scatter', 'line']:
        raise ValueError(f"plot_type must be in ['scatter', 'line']. It is currently {plot_type}")

    series_list = []

    if plot_type == 'scatter':
        for i in range(Y.shape[0]):
            s = ax.scatter(X, Y[i, :], label=series_labels[i], color=series_colors[i])
            series_list.append(s)

    elif plot_type == 'line':
        for i in range(Y.shape[0]):
            s = ax.plot(X, Y[i, :], label=series_labels[i], c=series_colors[i], marker=marker)
            series_list.append(s)

    return fig, ax



def plot_disorder(infile:str, doShow:bool=True, doSave:bool=False, outfile:str=None, cmap:str='viridis', figsize:tuple=(10, 10)) -> None:
    filedata = np.load(infile, allow_pickle=True)
    try:
        params = filedata['parameters'][()]
    except:
        params = filedata['p_list'][0]
    
    data = filedata['data']

    print(params)

    if False:
        for k, v in zip(params.keys(), params.values()):
            print(f"{k}  :  {v}")

        print()
        print(data)


    lattice_param_vals = data[1:, 0:2]
    disorder_vals = data[0, 2:]
    bott_vals = data[1:, 2:]

    series_labels = []
    for i in range(lattice_param_vals.shape[0]):
        series_labels.append(f"(M, B_tilde) = ({lattice_param_vals[i, 0]}, {lattice_param_vals[i, 1]})")


    boundary_cond_str = 'PBC' if params['pbc'] else 'OBC'
    if params['method'] != 'symmetry':
        title = f"Method of {params['method']}. {boundary_cond_str}. Generation {params['order']}.\n(t1, t2, B) = ({params['t1']}, {params['t2']}, {params['B']})"
    else:
        title = f"Method of {params['method']}, n = {params['n']:.0f}. {boundary_cond_str}. Generation {params['order']}.\n(t1, t2, B) = ({params['t1']}, {params['t2']}, {params['B']})"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig, ax = plot_series(fig, ax, disorder_vals, bott_vals, series_labels, get_colors_from_cmap(cmap, len(series_labels)), plot_type='scatter', marker='.')

    ax.hlines(0, 0, disorder_vals.max(), colors='k', ls='--')

    x_ticks = np.linspace(disorder_vals.min(), disorder_vals.max(), 11)
    y_ticks = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_ylim(-2.0, 2.0)

    ax.set_xlabel("Disorder Strength")
    ax.set_ylabel("Bott Index")
    ax.set_title(title)
    ax.legend()

    if doSave:
        if outfile == None:
            plt.savefig(infile[:-4]+'.png')
        else:
            plt.savefig(outfile, format='svg')

    if doShow:
        plt.show()



def get_colors_from_cmap(cmap:str, amount:int):
    return np.array([plt.cm.get_cmap(cmap)(val) for val in np.linspace(0.0, 1.0, amount)])



def print_npz_info(f, savetxt):
    fdata = np.load(f, allow_pickle=True)
    data, params = fdata['data'], fdata['parameters'][()]
    print(f)
    print(f"{params['method']} : {params['M_values']} : {params['B_tilde_values']}")
    print(pd.DataFrame(data[:, 0:-35]))
    print()
    if savetxt:
        np.savetxt(f[:-4]+'.txt', data)
#-------------Main Function Implementation-----------------

def test_series_plot():
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x = np.linspace(0, 2*np.pi, 100)
    
    y = np.empty((3, x.size))
    y[0, :] = np.sin(x)
    y[1, :] = np.cos(x)
    y[2, :] = np.cos(x) + np.sin(x)

    colors = get_colors_from_cmap('viridis', 3)

    fig, ax = plot_series(fig, ax, x, y, ['theta', 'function'], ['a', 'b', 'c'], colors, plot_type='line')

    plt.show()


def main():
    direc = './Data/Bott_Disorder/8-15-2024/'
    f = 'disorder_site_elim_crystalline.npz'
    outf = ''+'.png'
    plot_disorder(direc+f, True, False, outf, figsize=(7.5, 7.5))

def main2():
    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=10000, formatter=dict(float=lambda x: "%.2g" % x))
    pd.set_option('display.precision', 1)
    files = return_all_file_type('./zorganizing data', '.npz')

    c_list = []
    se = []
    sy = []
    re = []
    for f in files:
        if "crystalline" in f:
            c_list.append(f)
        elif "site_elim" in f:
            se.append(f)
        elif "renorm" in f:
            re.append(f)
        elif "symmetry" in f:
            sy.append(f)
    
    flists = [c_list, se, sy, re]
    for fl in flists:
        for f in fl:
            print_npz_info(f, False)


if __name__ == "__main__":
    files = return_all_file_type('./gen4/', '.npz')
    for f in files:
        fdata = np.load(f, allow_pickle=True)
        data, params = fdata['data'], fdata['parameters'][()]
        print(f)
        print(data)

    
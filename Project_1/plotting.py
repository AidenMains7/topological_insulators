import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from filesaving import return_all_file_type
import pylab


def plot_imshow(X:np.ndarray, Y:np.ndarray, Z:np.ndarray, xyz_labels:list[str, str, str], xy_ticks:tuple=(10,10), title:str=None, doShow:bool=True, doSave:bool=False, filename:str=None, cmap:str='viridis', doDiscreteCmap:bool=False) -> None:
    """
    
    Parameters:
    data (ndarray): 2D array
    
    """

    if doSave and filename == None:
        raise ValueError("When saving, a filename must be specified.")

    x_bounds = (X.min(), X.max())
    y_bounds = (Y.min(), Y.max())
    cbar_bounds = np.linspace(Z.min(), Z.max(), np.unique(Z).size) 

    x_ticks = np.linspace(x_bounds[0], x_bounds[1], xy_ticks[0])
    y_ticks = np.linspace(y_bounds[0], y_bounds[1], xy_ticks[1])
    cbar_ticks = np.linspace(cbar_bounds[0], cbar_bounds[-1], np.unique(Z).size)

    if doDiscreteCmap:
        cmap = plt.get_cmap(cmap, int(Z.max()-Z.min()+1))


    fig = plt.figure(figsize=(10, 10))
    plt.imshow(Z, cmap=cmap, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], aspect='auto')

    plt.xticks(x_ticks, labels=np.round(x_ticks, 1))
    plt.yticks(y_ticks, labels=np.round(y_ticks, 1))

    plt.xlabel(xyz_labels[0])
    plt.ylabel(xyz_labels[1])

    cbar = plt.colorbar(label=xyz_labels[2], ticks=cbar_ticks, spacing='uniform', cmap=cmap)

    plt.suptitle(title)

    if doSave:
        plt.savefig(filename)

    if doShow:
        plt.show()



def reshape_imshow_data(data:np.ndarray) -> np.ndarray:
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



def plot_bott(infile:str, doShow:bool=True, doSave:bool=False, outfile:str=None, cmap:str='viridis') -> None:
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
    plot_imshow(X, Y, Z, ['M', 'B_tilde', 'Bott Index'], (8, 5), title, doShow, doSave, outfile, cmap, False)




def main():
    plot_bott('bott_0.npz', True, True, 'bott_0.png')


if __name__ == "__main__":
    main()
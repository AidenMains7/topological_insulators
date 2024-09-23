import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from Carpet.compute_bott_disorder import run_computation
from scipy.linalg import eigvalsh
from itertools import product
from joblib import Parallel, delayed

from Carpet.project_dependencies import precompute, Hamiltonian_reconstruct
from Carpet.filesaving import return_all_file_type


def gaps_range(method:str, M_vals, B_tilde_vals, t1, t2, B, num_jobs=4):
    
    params = tuple(product(M_vals, B_tilde_vals))

    pre_data, lattice = precompute(method, 3, 0, True, 1, t1, t2, B)

    def worker(i):
        M, B_tilde = params[i]
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)
        eigvals = eigvalsh(H)

        pos_idxs = np.argwhere(eigvals > 0)
        neg_idxs = np.argwhere(eigvals < 0)

        spectral_gap = np.abs(eigvals[pos_idxs[0]] - eigvals[neg_idxs[-1]])
        bandwidth = np.abs(eigvals[pos_idxs[-1]] - eigvals[neg_idxs[0]])
        return [M, B_tilde, spectral_gap[0], bandwidth[0]]
    
    data = Parallel(n_jobs=num_jobs)(delayed(worker)(j) for j in range(len(params)))
    return np.array(data)

def find_wc(data):
    W_vals = data[0, 2:]
    params = data[1:, :2]
    bott_vals = data[1:, 2:]

    results = []
    for i in range(params.shape[0]):
        M, B_tilde = tuple(params[i])
        series = bott_vals[i]
        bott_init = series[0]

        Wc = W_vals[np.argwhere(np.abs(series) <= np.abs(bott_init/2))[0]][0]
        results.append([M, B_tilde, Wc])

    return np.array(results)

def plot_wc_gap(f:str, num_jobs:int=4):
    # Get data from file
    fdata = np.load(f, allow_pickle=True)
    data, params = fdata['data'], fdata['parameters'][()]

    # Calculate gap data for un-disordered lattice
    gap_data = gaps_range(params['method'], data[1:, 0], data[1:, 1], params['t1'], params['t2'], params['B'], num_jobs)
    
    # Find Wc data for each disorder series
    wc_data = find_wc(data)

    # Calculate the data for each lattice; minimal gap, bandwidth
    gap_data = np.unique(gap_data, axis=0)

    # Sort so that the parameter locations are the same
    def sort_by_two_cols(arr, col_order=(0, 1)):
        idx = np.lexsort((arr[:, col_order[1]], arr[:, col_order[0]]), axis=0)
        return arr[idx, :]
    gap_data = sort_by_two_cols(gap_data)
    wc_data = sort_by_two_cols(wc_data)

    # Concatenate data
    plotting_data = np.append(wc_data, gap_data[:, [2,3]], axis=1)    


    if False:
        for i in range(plotting_data.shape[0]):
            dat = plotting_data[i]
            print(dat)
            # min vs wc
            ax[0].scatter(dat[2], dat[3 ], label=f"({dat[0]}, {dat[1]})")
            # wc vs. M
            ax[2].scatter(dat[0], dat[2], label=f"M. Gap = {dat[3]:.2f}")
            # bw vs. wc
            ax[1].scatter(dat[2], dat[4], label=f"({dat[0]}, {dat[1]})")
        
        ax[0].legend()
        ax[0].set_xlabel('W critical')
        ax[0].set_ylabel('Minimal gap value')
        ax[0].set_title("Minimal gap vs. Wc")

        ax[1].legend()
        ax[1].set_xlabel('W critical')
        ax[1].set_ylabel('Bandwidth')
        ax[1].set_title('Bandwidth vs. W critical')

        ax[2].legend()
        ax[2].set_xlabel("M")
        ax[2].set_ylabel("W critical")
        ax[2].set_title("Wc vs. M")

        fig.suptitle(f)
        plt.show()

        
if __name__ == "__main__":
    direc='./zorganizing data/'
    plot_wc_gap(direc+'disorder_renorm_crystalline.npz')
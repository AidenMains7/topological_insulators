import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from Carpet.critical_w import find_critical_W, min_max_gap
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

def plot_wc_gap(f):

    if f is None:
        parameters = dict(
            method = "symmetry",
            order = 3,
            pad_width = 0,
            pbc = True,
            n = 1,
            t1 = 1.0,
            t2 = 1.0,
            B = 1.0,
            M_values =         np.linspace(1.0, 7.0, 25),
            B_tilde_values =   [0.0],
            W_values =         np.linspace(0.0, 10.0, 15, endpoint=False) + (10.0/15),
            iterations = 20,
            E_F = 0.0,
            KPM = False,
            N = 512,
            progress_bott = True,
            progress_disorder_iter = False, 
            progress_disorder_range = False,
            progress_disorder_many = True,
            doParallelIter = True,
            doParallelRange = False,
            doParallelMany = True,
            num_jobs = 28,
            cores_per_job = 1,
            saveEach = False
        )
        print("Computing disorder")
        f = run_computation(parameters, True, True, False, False)
    else:
        pass

    fdata = np.load(f, allow_pickle=True)
    data, params = fdata['data'], fdata['parameters'][()]
    gap_data = gaps_range(params['method'], params['M_values'], params['B_tilde_values'], params['t1'], params['t2'], params['B'], params['num_jobs'])
    wc_data = find_wc(data)

    gap_data = np.unique(gap_data, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    plotting_data = np.append(wc_data, gap_data[:, 2][:, np.newaxis], axis=1)

    for i in range(plotting_data.shape[0]):
        dat = plotting_data[i]
        ax.scatter(dat[2], dat[3], label=f"({dat[0]}, {dat[1]})")
    
    ax.legend()
    ax.set_xlabel('W critical')
    ax.set_ylabel('Minimal gap value')
    ax.set_title(f)
    plt.show()

        
if __name__ == "__main__":
    plot_wc_gap()
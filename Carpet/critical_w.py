import sys
sys.path.append(".")
from Carpet.project_dependencies import precompute, Hamiltonian_reconstruct
from Carpet.filesaving import return_all_file_type
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

from itertools import product


def min_max_gap(method, t1, t2, B, M, B_tilde):
    pre_data, lat = precompute(method, 3, 0, True, 1, t1, t2, B)
    H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)   
    eigvals = eigvalsh(H)

    pos_idxs = np.argwhere(eigvals > 0)
    neg_idxs = np.argwhere(eigvals < 0)

    spectral_gap = np.abs(eigvals[pos_idxs[0]] - eigvals[neg_idxs[-1]])
    bandwidth = np.abs(eigvals[pos_idxs[-1]] - eigvals[neg_idxs[0]])
    return spectral_gap, bandwidth

def find_critical_W(data, method, t2, title):
    W_vals = data[0, 2:]
    phy_params = data[1:, :2]
    scatter_vals = data[1:, 2:]

    if False:
        scatter_vals = scatter_vals[0:-1:2, :]
        phy_params = phy_params[0:-1:2, :]

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)

    for i in range(scatter_vals.shape[0]):
        M, B_tilde = phy_params[i][0], phy_params[i][1]
        s = scatter_vals[i, :]
        spec_gap, bandw = min_max_gap(method, 1.0, t2, 1.0, M, B_tilde)


        initial_bott = s[0]
        critical_w = W_vals[np.argwhere(np.abs(s) <= np.abs(initial_bott/2))[0]]

        s_lab = f"({M}, {B_tilde})"

        ax[1].scatter(critical_w, spec_gap, label=s_lab)
        ax[0].scatter(W_vals, s, label=s_lab)
        ax[2].scatter(critical_w, bandw, label=s_lab)


        print(f"{M}  {B_tilde}   :   {spec_gap}")
    


    ax[1].set_title("Minimum gap vs. critical W")
    ax[1].set_ylabel("Minimum gap")
    ax[1].set_xlabel("Critical W")

    ax[0].set_title("Bott Index vs. Disorder")
    ax[0].set_ylabel("Bott Index")
    ax[0].set_xlabel("Disorder (W)")

    ax[2].set_title("Bandwidth vs. critical W")
    ax[2].set_ylabel("Bandiwdth")
    ax[2].set_xlabel("Critical W")

    for i in range(3):
        ax[i].legend()  

    fig.suptitle(title)

    #plt.savefig(title+'_gap_w_'+'.svg', format='svg')   


def resave_data(file):
    fdata = np.load(file, allow_pickle=True)
    data, parameters = fdata['data'], fdata['parameters'][()]
    s_name = file[len('disorder_'):-4]

    if s_name == 'renorm':
        data = data[0:-1:2, :]

    params = data[1:, :2]



    if s_name in ['site_elim_crystalline', 'symmetry_crystalline']:
        good_list = [[10.0, 0.95], [10.0, 0.925], [10.0, 0.85]]    
    elif s_name in ['site_elim', 'symmetry']:
        good_list = [[1.0, 0.0], [2.0, 0.0], [2.5, 0.0], [5.5, 0.0], [6.0, 0.0], [6.5, 0.0]]
    elif s_name == 'renorm':
        good_list = [[1.0, 0.0], [3.0, 0.0], [3.5, 0.0], [4.5, 0.0], [5.5, 0.0], [6.0, 0.0]]
    elif s_name == 'renorm_crystalline':
        good_list = [[10.0, 1.0], [10.0, 0.8], [10.0, 0.9]]
    
    good_idxs = []
    for i in range(params.shape[0]):
        if list(params[i, :]) in good_list:
            good_idxs.append(i)
    
    good_idxs = np.array([-1]+good_idxs)+1
    
    new_data = data[good_idxs, :]

    np.savez(file, data=new_data, parameters=parameters)


def main():
    for f in return_all_file_type('.', '.npz'):
        fdata = np.load(f, allow_pickle=True)
        data, params = fdata['data'], fdata['parameters'][()]

        find_critical_W(data, params['method'], params['t2'], f[len('disorder_'):-4])

if __name__ == "__main__":

    print(min_max_gap('renorm', 1.0, 0.0, 1.0, 6.0, 0.0))
import numpy as np
import sys
sys.path.append(".")
from filesaving import return_all_file_type
from project_dependencies import precompute, Hamiltonian_reconstruct, projector_exact, bott_index, LDOS, remap_LDOS
from scipy.linalg import eigvalsh, eigh
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from joblib import Parallel, delayed
from itertools import product
from time import time
from datetime import date
import glob
import contextlib
from PIL import Image
import os
import pandas as pd


# OLD FUNCTIONS
def cbd_resaving_data():
    files = return_all_file_type('zorganizing data/', '.npz')
    

    fs = 'zorganizing data/disorder_'
    fe = '.npz'

    files = [fs+f+fe for f in ['symmetry_crystalline']]

    w_list = []
    s = []
    p_list = []
    for f in files:
        fdata = np.load(f, allow_pickle=True)
        data = fdata['data']
        params = fdata['parameters'][()]
        print(data[:, :2])

        p_list.append(params)
        w_list.append(data[0, :])
        for i in range(data.shape[0]-1):
            s.append(data[i+1, :])
    
    if False:
        if np.average((w_list[0]-w_list[1])[2:]) != 0:
            print(np.average((w_list[0]-w_list[1])[2:]))
            sleep(1)
            raise ValueError



    good_list = []
    added_list = []
    for i in range(len(s)):
        if s[i][1] in [0.95, 0.925, 0.85] and s[i][1] not in added_list:
            added_list.append(s[i][1])
            good_list.append(s[i][:, np.newaxis])

    good_list = [w_list[0][:, np.newaxis]]+good_list
    

    new_arr = np.concatenate(good_list, axis=1).T

    print(np.round(new_arr, 2))

    if False:
        np.savez('./fig1_data/fig1_c.npz', data=new_arr, p_list=p_list)


def cw_resave_data(file):
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

#
# FROM critical_w.py
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

#
# FROM max_min_gap.py
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

#
# From paper_figures.py
# Parallel
def compute_bott_range(method:str, M_vals:float, B_tilde_vals:float, order:int=None, pbc:bool=True, n:int=None, t1:float=1, t2:float=1, B:float=1, num_jobs:int=4) -> np.ndarray:
    
    params = tuple(product(M_vals, B_tilde_vals))
    pre_data, lattice = precompute(method, order, 0, pbc, n, t1, t2, B)
    def find_bott(M:float, B_tilde:float) -> tuple:
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, sparse=False)
        P = projector_exact(H, 0.0)
        bott = bott_index(P, lattice)

        return (M, B_tilde, bott)
    
    data = np.array(Parallel(n_jobs=num_jobs)(delayed(find_bott)(params[j][0], params[j][1]) for j in range(len(params)))).T
    return data


def compute_FIG2(num_jobs:int=28, resolution:int=10, doOrderFour:bool=False, doSave:bool=False, n:int=1, pbc:bool=True):
    
    # a:
    # Order 3, 4 : symmetry
    # t1 = B = 1
    # t2 = B_tilde = 0
    # M : [-1, 9]
    data_a_3 = compute_bott_range('symmetry', np.linspace(-1, 9, resolution), [0], 3, pbc, n, 1.0, 0.0, 1.0, num_jobs)
    if doOrderFour:
        data_a_4 = compute_bott_range('symmetry', np.linspace(-1, 9, resolution), [0], 4, pbc, n, 1.0, 0.0, 1.0, num_jobs)
        data_a = (data_a_3, data_a_4)
    else:
        data_a = (data_a_3, None)
    print('Finished: a')

    # b:
    # Order 3, 4 : symmetry
    # t1 = B = 1.0
    # t2 = 1.0
    # M = 10
    # B_tilde : [0.7, 1.1]
    data_b_3 = compute_bott_range('symmetry', [10], np.linspace(0.7, 1.1, resolution), 3, pbc, n, 1.0, 1.0, 1.0, num_jobs)
    if doOrderFour:
        data_b_4 = compute_bott_range('symmetry', [10], np.linspace(0.7, 1.1, resolution), 4, pbc, n, 1.0, 1.0, 1.0, num_jobs)
        data_b = (data_b_3, data_b_4)
    else:
        data_b = (data_b_3, None)
    print('Finished: b')

    # c: 
    # Square, method 3
    # t1 = B = 1
    # t2 = B_tilde = 0
    # M : [-2, 10]
    if doOrderFour:
        data_c_renorm = compute_bott_range('renorm', np.linspace(-2, 10, resolution), [0], 4, pbc, None, 1.0, 0.0, 1.0, num_jobs)
        data_c_square = compute_bott_range('square', np.linspace(-2, 10, resolution), [0], 4, pbc, None, 1.0, 0.0, 1.0, num_jobs)
    else:
        data_c_renorm = compute_bott_range('renorm', np.linspace(-2, 10, resolution), [0], 3, pbc, None, 1.0, 0.0, 1.0, num_jobs)
        data_c_square = compute_bott_range('square', np.linspace(-2, 10, resolution), [0], 3, pbc, None, 1.0, 0.0, 1.0, num_jobs)

    data_c = (data_c_square, data_c_renorm)
    print('Finished: c')

    # d:
    # Square, method 3
    # t1 = B = 1
    # t2 = 1
    # M = 10
    # B : [0.7, 1.1] 
    if doOrderFour:
        data_d_renorm = compute_bott_range('renorm', [10], np.linspace(0.7, 1.1, resolution), 4, pbc, None, 1.0, 1.0, 1.0, num_jobs)
        data_d_square = compute_bott_range('square', [10], np.linspace(0.7, 1.1, resolution), 4, pbc, None, 1.0, 1.0, 1.0, num_jobs)
    else:
        data_d_renorm = compute_bott_range('renorm', [10], np.linspace(0.7, 1.1, resolution), 3, pbc, None, 1.0, 1.0, 1.0, num_jobs)
        data_d_square = compute_bott_range('square', [10], np.linspace(0.7, 1.1, resolution), 3, pbc, None, 1.0, 1.0, 1.0, num_jobs)

    data_d = (data_d_square, data_d_renorm)
    print('Finished: d')


    if doSave:
        np.savez('FIG_2_data.npz', data_a_3=data_a[0], data_a_4=data_a[1], data_b_3=data_b[0], data_b_4=data_b[1], data_c_square=data_c[0], data_c_renorm=data_c[1], data_d_square=data_d[0], data_d_renorm=data_d[1])


    return data_a, data_b, data_c, data_d


def plot_FIG2(data_a, data_b, data_c, data_d):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # a
    axs[0, 0].scatter(data_a[0][0, :], data_a[0][2, :], label='Generation 3', c='green')
    if data_a[1] is not None and data_a[1].shape != ():
        axs[0, 0].scatter(data_a[1][0, :], data_a[1][2, :], label='Generation 4', c='red')

    # b
    axs[0, 1].scatter(data_b[0][1, :], data_b[0][2, :], label='Generation 3', c='green')
    if data_b[1] is not None and data_b[1].shape != ():
        axs[0, 1].scatter(data_b[1][1, :], data_b[1][2, :], label='Generation 4', c='red')

    # c
    axs[1, 0].scatter(data_c[0][0, :], data_c[0][2, :], label='Square', c='red')
    axs[1, 0].scatter(data_c[1][0, :], data_c[1][2, :], label='Renorm', c='blue')

    # d
    axs[1, 1].scatter(data_d[0][1, :], data_d[0][2, :], label='Square', c='red')
    axs[1, 1].scatter(data_d[1][1, :], data_d[1][2, :], label='Renorm', c='blue')



    axs[0, 0].set_ylabel("BI")
    axs[0, 0].set_title("(a)")
    axs[0, 0].set_yticks([-1, 0, 1])

    axs[0, 1].set_title("(b)")
    axs[0, 1].set_yticks([-2, -1, 0])

    axs[1, 0].set_ylabel("BI")
    axs[1, 0].set_xlabel("M")
    axs[1, 0].set_title("(c)")
    axs[1, 0].set_yticks([-1, 0, 1])

    axs[1, 1].set_xlabel("B_tilde")
    axs[1, 1].set_title("(d)")
    axs[1, 1].set_yticks([-2, -1, 0])

    plt.suptitle('n=1')

    for i in range(2):
        for j in range(2):
            axs[i, j].legend(frameon=True)


    plt.savefig('Data/Paper_Figures/FIG_2_n=1.png')
    plt.show()

# Parallel
def spectral_gap_range(method, M_vals, B_tilde_vals, order, pbc=True, n=None, t1=1.0, t2=1.0, B=1.0, num_jobs=4):

    params = tuple(product(M_vals, B_tilde_vals))

    pre_data, lattice = precompute(method, order, 0, pbc, n, t1, t2, B)

    def do_single(M, B_tilde):
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)
        eigvals = eigvalsh(H)

        pos_idxs = np.argwhere(eigvals > 0)
        neg_idxs = np.argwhere(eigvals < 0)

        spectral_gap = np.abs(eigvals[pos_idxs[0]] - eigvals[neg_idxs[-1]])[0]
        bandwidth = np.abs(eigvals[pos_idxs[-1]] - eigvals[neg_idxs[0]])
        return [M, spectral_gap]


    data = np.array(Parallel(n_jobs=num_jobs)(delayed(do_single)(params[j][0], params[j][1]) for j in range(len(params)))).T
    return data


def right_now():

    def spectral_gap(vals:tuple):
        M, B_tilde = vals
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)
        l, gap = LDOS(H)
        return np.round(gap, 3)
        
    pbc = True
    order = 3

    all_gaps = []
    all_vals = []

    # site elim 
    method = 'site_elim'
    t1, t2, B = (1.0, 0.0, 1.0)
    # m, b_tilde
    yellow = (6.5, 0.0)
    green = (6.0, 0.0)
    green_darker = (5.5, 0.0)
    cyan = (2.5, 0.0)
    blue = (2.0, 0.0)   
    purple = (1.0, 0.0)


    vals = [yellow, green, green_darker, cyan, blue, purple]
    pre_data, lattice = precompute(method, order, 0, pbc, None, t1, t2, B)
    gaps = []
    for v in vals:
        gaps.append(spectral_gap(v))
    all_gaps.append(gaps+[method])
    print(method)
    print(gaps)
    print(vals)
    print()

    # site elim crystalline
    method = 'site_elim'
    t1, t2, B = (1.0, 1.0, 1.0)
    # m, b_tilde
    yellow = (10.0, 0.95)
    green = (10.0, 0.925)
    purple = (10.0, 0.85)
    vals = [yellow, green, purple]
    pre_data, lattice = precompute(method, order, 0, pbc, None, t1, t2, B)
    gaps = []
    for v in vals:
        gaps.append(spectral_gap(v))
    all_gaps.append(gaps+[method])
    all_vals.append(vals + [method])
    print(method)
    print(gaps)
    print(vals)
    print()

    # symmetry
    method = 'symmetry'
    t1, t2, B = (1.0, 0.0, 1.0)
    # m, b_tilde
    yellow = (6.5, 0.0)
    green = (6.0, 0.0)
    green_darker = (5.5, 0.0)
    cyan = (2.5, 0.0)
    blue = (2.0, 0.0)
    purple = (1.0, 0.0)


    vals = [yellow, green, green_darker, cyan, blue, purple]
    pre_data, lattice = precompute(method, order, 0, pbc, 1, t1, t2, B)
    gaps = []
    for v in vals:
        gaps.append(spectral_gap(v))
    all_gaps.append(gaps+[method])
    all_vals.append(vals + [method])
    print(method)
    print(gaps)
    print(vals)
    print()

    # symmetry crystalline
    method = 'symmetry'
    t1, t2, B = (1.0, 1.0, 1.0)
    # m, b_tilde
    yellow = (10.0, 0.95)
    green = (10.0, 0.925)
    purple = (10.0, 0.85)
    vals = [yellow, green, purple]
    pre_data, lattice = precompute(method, order, 0, pbc, 1, t1, t2, B)
    gaps = []
    for v in vals:
        gaps.append(spectral_gap(v))
    all_gaps.append(gaps+[method])
    all_vals.append(vals + [method])
    print(method)
    print(gaps)
    print(vals)
    print()

    # renorm
    method = 'renorm'
    t1, t2, B = (1.0, 0.0, 1.0)
    # m, b_tilde
    yellow = (1.0, 0.0)
    green = (3.0, 0.0)
    green_darker = (3.5, 0.0)

    cyan = (4.5, 0.0)
    blue = (5.5, 0.0)
    purple = (6.5, 0.0)


    vals = [purple, blue, cyan, green_darker, green, yellow]
    pre_data, lattice = precompute(method, order, 0, pbc, None, t1, t2, B)
    gaps = []
    for v in vals:
        gaps.append(spectral_gap(v))
    all_gaps.append(gaps+[method])
    all_vals.append(vals + [method])
    print(method)
    print(gaps)
    print(vals)
    print()

    
    # renorm crystalline
    method = 'renorm'
    t1, t2, B = (1.0, 1.0, 1.0)
    # m, b_tilde
    three = (10.0, 0.8)
    two = (10.0, 0.9)
    one = (10.0, 1.0)
    vals = [one, two, three]
    pre_data, lattice = precompute(method, order, 0, pbc, None, t1, t2, B)
    gaps = []
    for v in vals:
        gaps.append(spectral_gap(v))
    all_gaps.append(gaps+[method])
    all_vals.append(vals + [method])
    print(method)
    print(gaps)
    print(vals)
    print()


def spectral_gap(method, t1, t2, B, M, B_tilde):
    pre_data, lat = precompute(method, 3, 0, True, 1, t1, t2, B)
    H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)   
    
    eigvals = eigvalsh(H)

    pos_idxs = np.argwhere(eigvals > 0)
    neg_idxs = np.argwhere(eigvals < 0)

    spectral_gap = np.abs(eigvals[pos_idxs[0]] - eigvals[neg_idxs[-1]])
    bandwidth = np.abs(eigvals[pos_idxs[-1]] - eigvals[neg_idxs[0]])
    return spectral_gap


def disorder_fig():
    if True:
        # -1 and +1
        e = [1.0, 2.0, 2.5, 5.5, 6.0, 6.5]
        f = [1.0, 3.0, 3.5, 4.5, 5.5, 6.0]
        g = np.linspace(1.0, 6.5, 100)
        params = tuple(product(f, [0.0]))
        t1, t2, B = 1.0, 1.0, 1.0
        y = [p[0] for p in params]
        y_tit = 'M'
        other_tit = 'B_tilde = 0.0'
    else:
        # -2
        params = tuple(product([10.0], [0.85, 0.875, 0.9, 0.925, 0.95]))
        t1, t2, B = 1.0, 0.0, 1.0
        y = [p[1] for p in params]
        y_tit = 'B_tilde'
        other_tit = 'M = 10.0'

    results = []    
    gaps = []
    for p in params:
        M, B_tilde = p
        method = 'renorm'
        g = spectral_gap(method, t1, t2, B, M, B_tilde)
        results.append([p, g])
        gaps.append(g)
        print(f"{p} : {g:.4f}")

    plt.scatter(y, gaps)
    plt.title(f"{method} : t1, t2, B = {t1}, {t2}, {B}\n{other_tit}")
    plt.xlabel('minimum gap')
    plt.ylabel(y_tit)
    plt.show()

    print(results)


def compute_FIG3(resolution:int=112, num_jobs:int=28) -> str:
    
    t0 = time()

    # t1 = B = 1
    # t2 = B_tilde = 0
    # order = 4
    # both pbc and obc

    # a : square
    # b : renorm
    # c : symmetry
    # d : site elim

    # x-axis : M varies on [-2.0, 10.0]
    # y-axis : G varies on [0.0, (4.0, 4.5, 5.0)] 

    
    method_list = ['square', 'renorm', 'symmetry', 'site_elim']
    order = 3; n = 1
    B_tilde_values = [0]
    t1 = 1.0; t2 = 0.0; B = 1.0
    M_values = np.linspace(-2.0, 10.0, resolution)

    save = []

    for i in range(2):
        for j in range(2):
            data_pbc = spectral_gap_range(method_list[2*i+j], M_values, B_tilde_values, order, True, n, t1, t2, B, num_jobs)
            data_obc = spectral_gap_range(method_list[2*i+j], M_values, B_tilde_values, order, False, n, t1, t2, B, num_jobs)

            save.append(data_pbc)
            save.append(data_obc)
            print(f"Completed :: {method_list[2*i+j]}")

    time_now = str(date.today())
    output_filename = f'FIG3_{time_now}.npz'
    np.savez(output_filename, square_pbc = save[0], square_obc = save[1], 
                                     renorm_pbc = save[2], renorm_obc = save[3], 
                                     symmetry_pbc = save[4], symmetry_obc = save[5], 
                                     site_elim_pbc = save[6], site_elim_obc = save[7])
    
    print(f"{time() - t0:.0f}s")

    return output_filename


def plot_FIG3(filepath:str):

    title_list = ['(a) square', '(b) renorm', '(c) symmetry', '(d) site_elim']
    name_list = ['square', 'renorm', 'symmetry', 'site_elim']

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    file_data = np.load(filepath, allow_pickle=True)


    for i in range(2):
        for j in range(2):

            data_pbc = file_data[name_list[2*i+j]+'_pbc']
            data_obc = file_data[name_list[2*i+j]+'_obc']

            axs[i, j].scatter(data_pbc[0, :], data_pbc[1, :], label='pbc', c='black')
            axs[i, j].scatter(data_obc[0, :], data_obc[1, :], label='obc', c='red')

            axs[i, j].set_xticks([-2.0, 1.0, 4.0, 7.0, 10.0])
            axs[i, j].set_yticks([0.0, 2.5, 5.0])

            axs[i, j].set_title(title_list[2*i+j])

    axs[0, 0].set_ylabel("G")
    axs[1, 0].set_ylabel("G")
    axs[1, 0].set_xlabel("M")
    axs[1, 1].set_xlabel("M")

    time_now = str(date.today())
    plt.savefig(f'Data/New_Data/FIG3_{time_now}.png')
    plt.show()


def LDOS_image():

    method = "square"
    order = 3
    pad_width = 0
    pbc = False
    n = 1
    t1 = 1.0
    t2 = 0.0
    B = 1.0

    M = 6.0
    B_tilde = 0.0

    if True:
        pre_data, lattice = precompute(method, order, pad_width, pbc, n, t1, t2, B)
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)
    else:
        # t1, t2, B must be adjusted in the code
        pre_data, lattice = pdd.precompute_data(order, method, pbc, n, pad_width)
        H = pdd.reconstruct_hamiltonian(method, pre_data, M, B_tilde, False)

    local, gap = LDOS(H)
    remapped_LDOS = remap_LDOS(local, lattice)

    remapped_LDOS[lattice < 0] = np.nan

    cmap = plt.cm.jet
    cmap.set_bad((0, 0, 0, 0))
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    im = ax[0].imshow(remapped_LDOS, cmap=cmap)
    ax[0].set_title(f"Method of {method}. Generation {order}. {'PBC' if pbc else 'OBC'}. \n(t1, t2, B) = ({t1:.2f}, {t2:.2f}, {B:.2f}). (M, B_tilde) = ({M:.2f}, {B_tilde:.2f})")
    cbar = fig.colorbar(im, ax=ax[0], cmap=cmap)

    ax[1].scatter(np.arange(local.size), local, label='LDOS')
    ax[1].set_title("LDOS, sum of two lowest energy states.")
    plt.show()


#
# From LDOS_imshow.py
def LDOS(Hamiltonian:np.ndarray) -> np.ndarray:

    # Get eigenvalues, eigenvectors
    eigvals, eigvecs = eigh(Hamiltonian)

    # Index of lowest eigenvalue
    idxs = np.argsort(np.abs(eigvals))

    # Eigenvectors and eigenvalues for lowest energy states
    eigva_one, eigva_two = eigvals[idxs[0]], eigvals[idxs[1]]
    eigve_one, eigve_two = eigvecs[:, idxs[0]], eigvecs[:, idxs[1]]

    # |v|^2
    eigve_one, eigve_two = np.power(np.abs(eigve_one), 2), np.power(np.abs(eigve_two), 2)

    # Sum pairwise
    LDOS_one, LDOS_two = eigve_one[0::2] + eigve_one[1::2], eigve_two[0::2] + eigve_two[1::2]

    # Spectral gap
    gap = np.abs(eigva_one) - np.abs(eigva_two)


    return LDOS_one, LDOS_two, gap


def remap_LDOS(local_density, lattice):
    # Ensure proper size
    if np.max(lattice)+1 != local_density.size:
        raise ValueError(f'Sizes of inputs do not match. Sizes {np.max(lattice)+1} and {local_density.size}')
    
    # Locations of filled site
    fills = np.argwhere(lattice.flatten() >= 0).flatten()

    # Initialize lattifce
    LDOS_lattice = np.full(lattice.size, 0.0)

    # Input into lattice
    LDOS_lattice[fills] = local_density

    # Reshape to proper size
    return LDOS_lattice.reshape(lattice.shape)

# Parallel
def LDOS_imshow(method:str, order:int, pad_width:int, n:int | None, pbc:bool, M_vals:np.ndarray | list | tuple, B_tilde_vals:np.ndarray | list | tuple, t1:float=1.0, t2:float=1.0, B:float=1.0, num_jobs:int=4) -> None:

    # Pre compute
    pre_data, lattice = precompute(method, order, pad_width, pbc, n, t1, t2, B)

    # Get list of parameter combinations
    parameters = tuple(product(M_vals, B_tilde_vals))


    def do_single(i):
        # Do computation
        M, B_tilde = parameters[i]
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)
        P = projector_exact(H, 0.0)
        bott = bott_index(P, lattice)

        # Compute LDOS for two closest to 0 eigenstates
        local_one, local_two, gap = LDOS(H)
        local_both = local_one + local_two

        # Remap to shape of lattice
        LDOS_lattice = remap_LDOS(local_both, lattice)

        # Create figure
        fig = plt.figure(figsize=(10,10))
        plt.imshow(LDOS_lattice, label='Local Density of States', cmap='Purples')
        plt.title(f'M={M:.1f} :: B={B_tilde:.1f} :: BI={bott}')
        plt.colorbar()
        plt.savefig(f'Data/LDOS/Square_Range/{method}_{M:.1f}_{B:.1f}_{bott}.png')

    # Compute in parallel
    Parallel(n_jobs=num_jobs)(delayed(do_single)(i) for i in range(len(parameters)))

# Create a gif :)
def create_gif():
    fp_in = "Data/LDOS/Square_Range/square_*.png"
    fp_out = "Data/LDOS/square.gif"

    image_files = glob.glob(fp_in)
    shortened_names = [fp[len("Data/LDOS/Square_Range/square_"):-4] for fp in image_files]
    number_order = [[float(num_str) for num_str in fp.split('_')][0] for fp in shortened_names]
    number_order = np.array(number_order)

    image_files = np.array(image_files, dtype=str)
    file_order = image_files[np.argsort(number_order)]

    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f))
                for f in file_order)
        
        img = next(imgs)

        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=2000, loop=1)


#
# From interactive_plot.py
def get_LDOS(method, pre_data, M, B_tilde):
    H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)
    one, two, gap = LDOS(H)
    return one, two


def make_plot(method):

    pre_data, lattice = precompute(method, 3, 0, True, None, 1.0, 0.0, 1.0)
    init_M = -2.0
    init_B_tilde = 0.0

    fig, ax = plt.subplots(figsize=(10, 10))

    

    one, two = get_LDOS(method, pre_data, init_M, init_B_tilde)

    x = np.arange(one.size)

    scat_one = ax.scatter(x, one, c='red')
    scat_two = ax.scatter(x, two, c='black')
    ax.set_xlabel('Site Number')

    fig.subplots_adjust(left=0.25, bottom=0.25)


    axM = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    M_slider = Slider(
        ax=axM,
        label='M',
        valmin=-2.0,
        valmax=10.0,
        valinit=init_M,
    )

    axB_tilde = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    B_tilde_slider = Slider(
        ax=axB_tilde,
        label="B_tilde",
        valmin=0.0,
        valmax=2.0,
        valinit=init_B_tilde,
        orientation="vertical"
    )


    def update(val):

        ax.clear()

        one, two = get_LDOS(method, pre_data, M_slider.val, B_tilde_slider.val)
        scat_one = ax.scatter(x, one, c='red')
        scat_two = ax.scatter(x, two, c='black')
        ax.set_xlabel('Site Number')


    M_slider.on_changed(update)
    B_tilde_slider.on_changed(update)

    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        M_slider.reset()
        B_tilde_slider.reset()
    button.on_clicked(reset)

    plt.show()

#
# From phase_diagram.py
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
        parameters = file_data['parameters'][()]

    return data, parameters


def print_params_nice(params:dict):
    for kw in params:
        if isinstance(params[kw], np.ndarray):
            print(f"{kw}: np.linspace({params[kw][0]}, {params[kw][-1]}, {params[kw].size})")
        else:
            print(f"{kw}: {params[kw]}")


def plot_disorder(filename:str, doShow:bool, doSave:bool, title:str=None) -> None:
    """
    Will read data from a specifed .npz file and plot for which has non-zero inital Bott Index.
    """
    try:
        data, params = _read_npz_data(filename)

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

                M = x[0]
                B_tilde = y[0]
                x, y = x[1:], y[1:]

                if True:
                    if y[0] not in bott_vals:
                        print(f"Initial bott index is not in [-3, -2, -1, 1, 2, 3]. Value is {y[0]}")
                        ax.plot(x, y, c='black', marker='.')
                    
                    try:
                        ax.plot(x, y, marker='.', label=f"(M, B_tilde) = ({M:.2f}, {B_tilde:.2f})")

                    except Exception as e:
                        print(f"Caught exception: {e}")

                else:
                    ax.plot(x, y, label=f"(M, B_tilde) = ({M:.2f}, {B_tilde:.2f})")

        plt.legend()

        #Add legend
        if False:
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
        Z = np.stack(arrs, axis=0).T

        fig = plt.figure(figsize=(10,10))
        plt.imshow(Z, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], origin='lower', aspect='auto', cmap='viridis')
        cbar = plt.colorbar(label='Bott Index')
        cbar.set_label('Bott Index', rotation=0)

        x_ticks = np.linspace(x_bounds[0], x_bounds[1], 8)
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

#
# From testing.py
def testing_main1():
    files = return_all_file_type('fig1_data/', '.npz')

    data = np.load('fig1_data/fig1_i.npz', allow_pickle=True)['data']
    print(pd.DataFrame(np.round(data[[0, 2], :30], 1)))

def show_data(f):
    data = np.load(f, allow_pickle=True)['data']
    print(data)


def testing_clean_csv(f):
    dataf = pd.read_csv(f)
    data = np.array(dataf)

    row_list = []
    for i in range(data.shape[0]):
        d = data[i]
        M, B_tilde = d[1], d[2]
        W = d[4]
        method = d[5]

        t2 = 1.0 if d[6] == 'y' else 0.0
        ming, maxg = min_max_gap(method, 1.0, t2, 1.0, M, B_tilde)

        name = 'abcdefghi'[i//3]

        new_row = [name, M, B_tilde, np.round(ming[0], 5), np.round(maxg[0], 5), W]
        row_list.append(new_row)

    labels = ['section', 'M', 'B_tilde', 'Minimal Gap', 'Bandwidth', 'W']
    
    data_vals = pd.DataFrame(row_list, columns=labels)
    data_vals.to_csv(f[:-4]+'_cleaned'+f[-4:])


def testing_main2():
    B_tilde_list = [0.0]
    M_list = np.linspace(0.0, 4.0, 41)
    p = tuple(product(M_list, B_tilde_list))
    minimal, bandwidth = [], []
    for i in range(len(p)):
        M, B_tilde = p[i]
        mg, bw = min_max_gap('renorm', 1.0, 0.0, 1.0, M, B_tilde)
        minimal.append(mg)
        bandwidth.append(bw)

    fig, axs = plt.subplots(1, 2, figsize=(10,10))
    axs[0].scatter(M_list, minimal, label='min gap')
    axs[1].scatter(M_list, bandwidth, label='bandwidth')
    plt.show()


if __name__ == "__main__":
    pass
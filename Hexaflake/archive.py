import numpy as np
from matplotlib import pyplot as plt
from MaybeActualFinalHaldane2 import  compute_hamiltonian, compute_geometric_data, compute_bott_index
from itertools import product
from tqdm_joblib import tqdm_joblib, tqdm
from joblib import Parallel, delayed
import scipy as sp
from multiprocessing import Lock, Manager
import h5py, os, glob
from time import time

def plot_files():
    directory = 'Haldane_Disorder_Data/Res2500_Avg100/'
    val_list = []
    for f in glob.glob(directory+'*.h5'):
        with h5py.File(f, 'r') as file:
            if 'disorder' in file.keys():
                val_list.append(file['disorder'][:])
    for v in val_list:
        plt.plot(range(len(v)), v)

        plt.show()


def plot_comparison(clean_file, disorder_vals, method, generation, disorder_strength):
    with h5py.File(clean_file, 'r') as f:
        clean_dict = {k: v[:] for k, v in zip(f.keys(), f.values())}
    
    with h5py.File(disorder_vals, 'r') as f:
        disorder_vals = f['disorder'][:].T

    phi_vals, M_vals, bott_index_vals = clean_dict['phi'], clean_dict['M'], clean_dict['bott_index'].T

    for Z, tp, fp in zip([bott_index_vals, disorder_vals], ['Clean', f'Disordered (W={disorder_strength})'], ['clean', 'disorder']):
        fig, axs = plt.subplots(1, 1)
        fig, axs = plot_phase_diagram(fig, axs, phi_vals, M_vals, Z, titleparams=f'{tp}, {method}, g{generation}')
        if tp[0] == 'C':
            plt.savefig(f"{method}_g{generation}_{fp}.png")
        else:
            plt.savefig(f"{method}_g{generation}_w{disorder_strength}_{fp}.png")


def probe_files(directory):
    for f in glob.glob(directory+'*.h5'):
        with h5py.File(f, 'r') as file:
            print(f)
            for k in file.keys():
                print(f"{k}: {file[k][:]}")
            print('\n')


def find_edges(Z):
    Z = np.pad(Z, pad_width=1, mode='constant', constant_values=np.nan)
    top = Z - np.roll(Z, 1, axis=0)
    bottom = Z - np.roll(Z, -1, axis=0)
    left = Z - np.roll(Z, 1, axis=1)
    right = Z - np.roll(Z, -1, axis=1)
    labels = ['top', 'bottom', 'left', 'right']
    arrs = [top, bottom, left, right]
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    for i in range(2):
        for j in range(2):
            axs[i,j].imshow(arrs[2*i+j][1:-2, 1:-2])
            axs[i,j].set_title(labels[2*i+j])
    plt.show()  
    plt.imshow(top+left)
    plt.show()




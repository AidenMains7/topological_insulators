import numpy as np
from matplotlib import pyplot as plt
from project_dependencies import bott_index, precompute, Hamiltonian_reconstruct, projector_exact, remap_LDOS, LDOS
from joblib import Parallel, delayed
from itertools import product

import Dan_Code.PhaseDiagram.PhaseDiagramDependencies as pdd

from scipy.linalg import eigvalsh

from datetime import date
from time import time

import sys
if sys.version_info[1] > 9:
    import scienceplots
    plt.style.use(['science', 'pgf'])


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
        local, gap = LDOS(H)

        return [M, gap]

    data = np.array(Parallel(n_jobs=num_jobs)(delayed(do_single)(params[j][0], params[j][1]) for j in range(len(params)))).T
    return data


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


# --------------

def main():
    f = compute_FIG3(200, 4)
    plot_FIG3(f)

if __name__ == "__main__":
    main()
import numpy as np
from matplotlib import pyplot as plt
from project_dependencies import bott_index, precompute, Hamiltonian_reconstruct, projector_exact, remap_LDOS, LDOS
from joblib import Parallel, delayed
from itertools import product

import Dan_Code.PhaseDiagram.PhaseDiagramDependencies as pdd

from scipy.linalg import eigvalsh

from datetime import date
from time import time




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


def new_now():

    pass

        


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
    l, gap = LDOS(H)
    return gap


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


# --------------

def main():
    f = compute_FIG3(200, 4)
    plot_FIG3(f)

if __name__ == "__main__":
    print(np.linspace(1.0, 7.0, 25))
    data = spectral_gap_range('renorm', np.linspace(1.0, 7.0, 25), [0.0], 3, True, None, 1.0, 0.0, 1.0, 4)
    plt.scatter(data[0, :], data[1, :])
    print(data)
    plt.show()

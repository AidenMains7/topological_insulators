import numpy as np
from matplotlib import pyplot as plt
from project_dependencies import bott_index, precompute, Hamiltonian_reconstruct, projector_exact, remap_LDOS, LDOS, spectral_gap
from joblib import Parallel, delayed
from itertools import product
import ProjectCode_1.PhaseDiagram.PhaseDiagramDependencies as dsd
import ProjectCode_1.ComputeBottIndex as dsb

import sys
if sys.version_info[1] > 9:
    import scienceplots
    plt.style.use(['science', 'pgf'])


def compute_bott_range(method:str, M_vals:float, B_tilde_vals:float, order:int=None, n:int=None, t1:float=1, t2:float=1, B:float=1, num_jobs:int=4) -> np.ndarray:


    params = tuple(product(M_vals, B_tilde_vals))

    def find_bott(M:float, B_tilde:float) -> tuple:
        pre_data, lattice = precompute(method, order, 0, True, n, t1, t2, B)
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, sparse=False)
        P = projector_exact(H, 0.0)
        bott = bott_index(P, lattice)

        return (M, B_tilde, bott)
    
    data = np.array(Parallel(n_jobs=num_jobs)(delayed(find_bott)(params[j][0], params[j][1]) for j in range(len(params)))).T
    return data
    

def compute_bott_range(method:str, M_vals:float, B_tilde_vals:float, order:int=None, n:int=None, t1:float=1, t2:float=1, B:float=1, num_jobs:int=4) -> np.ndarray:


    params = tuple(product(M_vals, B_tilde_vals))

    def find_bott(M:float, B_tilde:float) -> tuple:
        pre_data, lattice = dsd.precompute_data(order, method, True, n, 0, t1, t2, B)
        H = dsd.reconstruct_hamiltonian(method, pre_data, M, B_tilde, sparse=False)
        P = dsb.projector_exact(H, 0.0)
        bott = dsb.bott_index(P, lattice)

        return (M, B_tilde, bott)
    
    data = np.array(Parallel(n_jobs=num_jobs)(delayed(find_bott)(params[j][0], params[j][1]) for j in range(len(params)))).T
    return data


def compute_FIG2(num_jobs:int=28, resolution:int=10, doOrderFour:bool=False, doSave:bool=False):
    
    # a:
    # Order 3, 4 : symmetry
    # t1 = B = 1
    # t2 = B_tilde = 0
    # M : [-1, 9]
    data_a_3 = compute_bott_range('symmetry', np.linspace(-1, 9, resolution), [0], 3, 2, 1, 0, 1, num_jobs)
    if doOrderFour:
        data_a_4 = compute_bott_range('symmetry', np.linspace(-1, 9, resolution), [0], 4, 2, 1, 0, 1, num_jobs)
        data_a = (data_a_3, data_a_4)
    else:
        data_a = (data_a_3, None)
    print('Finished: a')

    # b:
    # Order 3, 4 : symmetry
    # t1 = B = 1
    # t2 = 1
    # M = 10
    # B_tilde : [0.7, 1.1]
    data_b_3 = compute_bott_range('symmetry', [10], np.linspace(0.7, 1.1, resolution), 3, 2, 1, 1, 1, num_jobs)
    if doOrderFour:
        data_b_4 = compute_bott_range('symmetry', [10], np.linspace(0.7, 1.1, resolution), 4, 2, 1, 1, 1, num_jobs)
        data_b = (data_b_3, data_b_4)
    else:
        data_b = (data_b_3, None)
    print('Finished: b')

    # c: 
    # Square, method 3
    # t1 = B = 1
    # t2 = B_tilde = 0
    # M : [-2, 10]
    data_c_renorm = compute_bott_range('renorm', np.linspace(-2, 10, resolution), [0], 3, None, 1, 0, 1, num_jobs)
    data_c_square = compute_bott_range('square', np.linspace(-2, 10, resolution), [0], 3, None, 1, 0, 1, num_jobs)
    data_c = (data_c_square, data_c_renorm)
    print('Finished: c')

    # d:
    # Square, method 3
    # t1 = B = 1
    # t2 = 1
    # M = 10
    # B : [0.7, 1.1] 
    data_d_renorm = compute_bott_range('renorm', [10], np.linspace(0.7, 1.1, resolution), 3, None, 1, 1, 1, num_jobs)
    data_d_square = compute_bott_range('square', [10], np.linspace(0.7, 1.1, resolution), 3, None, 1, 1, 1, num_jobs)
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

    for i in range(2):
        for j in range(2):
            axs[i, j].legend(frameon=True)


    plt.savefig('FIGURE_2.png')
    plt.show()


def spectral_gap_range(method, M_vals, B_tilde_vals, order, pbc=True, n=None, t1=1.0, t2=1.0, B=1.0, num_jobs=4):

    params = tuple(product(M_vals, B_tilde_vals))

    def do_single(M, B_tilde):

        pre_data, lattice = precompute(method, order, 0, pbc, n, t1, t2, B)
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, False)
        G = spectral_gap(H)

        return [M, G]

    data = np.array(Parallel(n_jobs=num_jobs)(delayed(do_single)(params[j][0], params[j][1]) for j in range(len(params)))).T
    return data



def compute_FIG3():
    

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
    title_list = ['(a)', '(b)', '(c)', '(d)']
    order = 3; n = 2; num_jobs = 28
    M_values = np.linspace(-2.0, 10.0, 28); B_tilde_values = [0]
    t1 = 1.0; t2 = 0.0; B = 1.0

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(2):
        for j in range(2):
            data_pbc = spectral_gap_range(method_list[2*i+j], M_values, B_tilde_values, order, True, n, t1, t2, B, num_jobs)
            data_obc = spectral_gap_range(method_list[2*i+j], M_values, B_tilde_values, order, False, n, t1, t2, B, num_jobs)

            axs[i, j].scatter(data_pbc[0, :], data_pbc[1, :], label='pbc', c='black')
            axs[i, j].scatter(data_obc[0, :], data_obc[1, :], label='obc', c='red')

            axs[i, j].set_xticks([-2.0, 1.0, 4.0, 7.0, 10.0])
            axs[i, j].set_yticks([0.0, 2.5, 5.0])

            axs[i, j].set_title(title_list[2*i+j])
    
    axs[0, 0].set_ylabel("G")
    axs[1, 0].set_ylabel("G")
    axs[1, 0].set_xlabel("M")
    axs[1, 1].set_xlabel("M")



# Main function implementation------------------

def FIG2_main():
    data_a, data_b, data_c, data_d = compute_FIG2(28, 28, True, True)
    plot_FIG2(data_a, data_b, data_c, data_d)

def FIG2_main2():
    file_data = np.load('FIG_2_data.npz', allow_pickle=True)
    data_a = (file_data['data_a_3'], file_data['data_a_4'])
    data_b = (file_data['data_b_3'], file_data['data_b_4'])
    data_c = (file_data['data_c_square'], file_data['data_c_renorm'])
    data_d = (file_data['data_d_square'], file_data['data_d_renorm'])

    plot_FIG2(data_a, data_b, data_c, data_d)

def FIG3_main():
    method = 'site_elim'; order = 3; pbc = True
    pre_data, lattice = precompute(method, order, 0, pbc, 2, 1, 0, 1)
    H = Hamiltonian_reconstruct(method, pre_data, 2.5, 0.0, False)

    P = projector_exact(H, 0.0)
    bott = bott_index(P, lattice)
    print(f"Bott Index = {bott:.0f}")

    local_density = LDOS(H)

    fills = np.argwhere(lattice >= 0)

    LDOS_lattice = np.full(lattice.shape, 0.0)
    LDOS_lattice[fills[:, 0], fills[:, 1]] = local_density


    fig = plt.figure(figsize=(10, 10))
    plt.scatter(np.arange(local_density.size), local_density)
    plt.show()


    fig = plt.figure(figsize=(10, 10))
    plt.imshow(LDOS_lattice, cmap='jet')
    plt.colorbar()
    plt.title(f"{method}, order = {order}, pbc = {pbc}")
    plt.show()

def FIG3_main2():
    compute_FIG3()

if __name__ == "__main__":
    FIG3_main2()
import numpy as np
from matplotlib import pyplot as plt
import scienceplots
from project_dependencies import bott_index, precompute, Hamiltonian_reconstruct, projector_exact
from joblib import Parallel, delayed
from itertools import product

plt.style.use(['science', 'pgf'])




def FIG_2_data(method:str, M_vals:float, B_tilde_vals:float, order:int=None, n:int=None, t1:float=1, t2:float=1, B:float=1, num_jobs:int=4) -> np.ndarray:


    params = tuple(product(M_vals, B_tilde_vals))

    def find_bott(M:float, B_tilde:float) -> tuple:
        pre_data, lattice = precompute(method, order, 0, True, n, t1, t2, B)
        H = Hamiltonian_reconstruct(method, pre_data, M, B_tilde, sparse=False)
        P = projector_exact(H, 0.0)
        bott = bott_index(P, lattice)

        return (M, B_tilde, bott)
    
    data = np.array(Parallel(n_jobs=num_jobs)(delayed(find_bott)(params[j][0], params[j][1]) for j in range(len(params)))).T
    return data
    



def compute(num_jobs:int=28, resolution:int=10, doOrderFour:bool=False, doSave:bool=False):
    
    # a:
    # Order 3, 4 : symmetry
    # t1 = B = 1
    # t2 = B_tilde = 0
    # M : [-1, 9]
    data_a_3 = FIG_2_data('symmetry', np.linspace(-1, 9, resolution), [0], 3, 2, 1, 0, 1, num_jobs)
    if doOrderFour:
        data_a_4 = FIG_2_data('symmetry', np.linspace(-1, 9, resolution), [0], 4, 2, 1, 0, 1, num_jobs)
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
    data_b_3 = FIG_2_data('symmetry', [10], np.linspace(0.7, 1.1, resolution), 3, 2, 1, 1, 1, num_jobs)
    if doOrderFour:
        data_b_4 = FIG_2_data('symmetry', [10], np.linspace(0.7, 1.1, resolution), 4, 2, 1, 1, 1, num_jobs)
        data_b = (data_b_3, data_b_4)
    else:
        data_b = (data_b_3, None)
    print('Finished: b')

    # c: 
    # Square, method 3
    # t1 = B = 1
    # t2 = B_tilde = 0
    # M : [-2, 10]
    data_c_renorm = FIG_2_data('renorm', np.linspace(-2, 10, resolution), [0], 3, None, 1, 0, 1, num_jobs)
    data_c_square = FIG_2_data('square', np.linspace(-2, 10, resolution), [0], 3, None, 1, 0, 1, num_jobs)
    data_c = (data_c_square, data_c_renorm)
    print('Finished: c')

    # d:
    # Square, method 3
    # t1 = B = 1
    # t2 = 1
    # M = 10
    # B : [0.7, 1.1] 
    data_d_renorm = FIG_2_data('renorm', [10], np.linspace(0.7, 1.1, resolution), 3, None, 1, 1, 1, num_jobs)
    data_d_square = FIG_2_data('square', [10], np.linspace(0.7, 1.1, resolution), 3, None, 1, 1, 1, num_jobs)
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



# Main function implementation------------------

def main():
    data_a, data_b, data_c, data_d = compute(4, 10, False, True)
    plot_FIG2(data_a, data_b, data_c, data_d)

def main2():
    file_data = np.load('FIG_2_data.npz', allow_pickle=True)
    data_a = (file_data['data_a_3'], file_data['data_a_4'])
    data_b = (file_data['data_b_3'], file_data['data_b_4'])
    data_c = (file_data['data_c_square'], file_data['data_c_renorm'])
    data_d = (file_data['data_d_square'], file_data['data_d_renorm'])

    plot_FIG2(data_a, data_b, data_c, data_d)



if __name__ == "__main__":
    main()
import sys
sys.path.append(".")


import numpy as np
from Carpet.filesaving import return_all_file_type
import pandas as pd
from Carpet.critical_w import min_max_gap
from itertools import product
import matplotlib.pyplot as plt


def main():
    files = return_all_file_type('fig1_data/', '.npz')

    data = np.load('fig1_data/fig1_i.npz', allow_pickle=True)['data']
    print(pd.DataFrame(np.round(data[[0, 2], :30], 1)))

def show_data(f):
    data = np.load(f, allow_pickle=True)['data']
    print(data)


def clean_csv(f):
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


def main():
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
    clean_csv('fig1_data/fig1_vals.csv')
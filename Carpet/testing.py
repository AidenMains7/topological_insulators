import sys
sys.path.append(".")


import numpy as np
from Carpet.filesaving import return_all_file_type
import pandas as pd
from Carpet.plotting import plot_disorder

def main():
    files = return_all_file_type('fig1_data/', '.npz')

    data = np.load('fig1_data/fig1_i.npz', allow_pickle=True)['data']
    print(pd.DataFrame(np.round(data[[0, 2], :30], 1)))

def show_data(f):
    data = np.load(f, allow_pickle=True)['data']
    print(data)



if __name__ == "__main__":
    show_data('gen4/disorder_gen4_symmetry_crystalline_0.npz')
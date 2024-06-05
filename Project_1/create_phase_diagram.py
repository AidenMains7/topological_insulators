import numpy as np
from project_execute import many_bott, many_lattices, computation
from datetime import datetime
import latex
import matplotlib.pyplot as plt
import scienceplots

#plt.style.use('science')

def timely_filename() -> str:
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hr = now.hour
    minute = now.minute
    time_string=f"{month}_{day}_{year}__{hr}_{minute}"
    return time_string


def read_npz_data(filename:str):
    file_data = np.load(filename)
    data = file_data['arr_0'] 
    return data


def plot(filename:str):
    data = read_npz_data(filename)

    num = data.shape[0]

    fig, ax = plt.subplots()
    ax.set_title("Bott Index vs. Disorder Strength $\\pi$")
    ax.set_xlabel("Disorder Strength (W)")
    ax.set_ylabel("Bott Index")
    ax.set_ylim([-2, 2])


    for i in range(num):
        if data[i][0,1] is not np.nan:
            x = data[i][0]
            y = data[i][1]

            ax.plot(x, y)

    plt.show()


def run_computation():
    method = "symmetry"
    order = 3
    pad_w = 0
    pbc=True
    n=10
    M_values = np.linspace(-2,12,5)
    B_tilde_values = np.linspace(0,2,1)
    W_values = np.linspace(0.5,7.5,20)
    iter_p_d = 1

    data = computation(method, order, pad_w, pbc, n, M_values, B_tilde_values, W_values, iter_p_d, E_F=0.0)

    filename = f"bott_disorder_{timely_filename()}.npz"
    np.savez(filename, data)

def main():
    plot("bott_disorder_6_5_2024__16_8.npz")


if __name__ == "__main__":
    main()


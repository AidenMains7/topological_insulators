import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from project_dependencies import precompute, Hamiltonian_reconstruct, LDOS

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



if __name__ == "__main__":
    make_plot('square')
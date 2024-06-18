import numpy as np

from phase_diagram import _read_npz_data,plot_bott_imshow


data, params = _read_npz_data("bott_2.npz")

bott_separation = [data[:, mask] for mask in [data[2, :] == bott for bott in [-3, -2, -1, 0, 1, 2, 3]]]
bott_separation = [arr[:, :5] for arr in bott_separation]

nonzero_bott_arr = np.empty((3, 0))
for arr in bott_separation:
    if arr.shape[1] > 0:
        if arr[2,0] != 0:
            nonzero_bott_arr = np.append(nonzero_bott_arr, arr, axis=1)


print(list(np.unique(data[2])))

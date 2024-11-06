import numpy as np
from matplotlib import pyplot as plt

x_num, y_num = 11, 11
X = np.linspace(-2.0, 12.0, x_num)
Y = np.linspace(0.0, 2.0, y_num)

X = np.repeat(X, Y.size)
Y = np.tile(Y, X.size)

Z = np.random.randint(-2, 2, x_num*y_num)

print(Z)
    
# We want to artifically increase the resolution of the imshow plot.



X_unique = np.unique(X)
Y_unique = np.unique(Y)
X_mesh, Y_mesh = np.meshgrid(X_unique, Y_unique)
Z_surf = np.empty(X_mesh.shape)


for i in range(len(X)):
    x = X[i]
    y = Y[i]
    z = Z[i]

    x_idx = np.where(X_unique == x)[0][0]
    y_idx = np.where(Y_unique == y)[0][0]
    Z_surf[y_idx, x_idx] = z
Z_surf = np.flipud(Z_surf)

fig, ax = plt.subplots(2, 1, figsize=(10,10))
im = ax[0].imshow(Z_surf, extent=[X.min(), X.max(), Y.min(), Y.max()], interpolation='gaussian')


if True:
    scaling_factor = 25
    sc_mat_2d = np.ones((scaling_factor, scaling_factor))
    sc_mat_1d = np.ones((scaling_factor, 1))

    Z_surf = np.kron(Z_surf, sc_mat_2d)
    X_scaled, Y_scaled = np.kron(X, sc_mat_1d), np.kron(Y, sc_mat_1d)


def tensor_rescale(X, Y, Z, scaling_factor):
    sc_mat_2d = np.ones((scaling_factor, scaling_factor))
    sc_mat_1d = np.ones((scaling_factor, 1))

    Z_scaled = np.kron(Z, sc_mat_2d)
    X_scaled, Y_scaled = np.kron(X, sc_mat_1d), np.kron(Y, sc_mat_1d)
    return X_scaled, Y_scaled, Z_scaled



im = ax[1].imshow(Z_surf, extent=[X.min(), X.max(), Y.min(), Y.max()], interpolation='gaussian')
plt.show()
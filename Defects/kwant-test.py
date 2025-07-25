import numpy as np
import kwant as kw
from matplotlib import pyplot as plt



system = kw.Builder()
a = 1.0
lat = kw.lattice.square(a, norbs=1)

t = 1.0
W, L = 10, 10
for i in range(W):
    for j in range(L):
        system[lat(i, j)] = 4 * t

        if j > 0:
            system[lat(i, j), lat(i, j - 1)] = -t
        if i > 0:
            system[lat(i, j), lat(i - 1, j)] = -t


kw.plot(system)

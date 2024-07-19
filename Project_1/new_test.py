import numpy as np
from itertools import product


idx = (0, 1)

a = np.arange(25).reshape(25)


print([cell for cell in row] for row in a)
import numpy as np


seeds = []
for i in range(1):
    seeds.append(np.random.randint(1, 2**31))


print(seeds)
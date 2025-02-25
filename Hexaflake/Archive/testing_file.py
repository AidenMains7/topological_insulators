import numpy as np

n=100
arr = np.arange(n).reshape(10, 10)

print(arr)

a = np.random.randint(0, 10, (10, 2))
print(a)
print(arr[a[:, 0], a[:, 1]])
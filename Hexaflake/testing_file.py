import numpy as np

n=20
arr = np.arange(n).reshape(2, n//2).T

print(arr)

s = np.array([4, 14])


if any(np.equal(arr, s).any(1)):
    print("True")
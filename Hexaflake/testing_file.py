import numpy as np

n=20
arr = np.arange(n).reshape(2, n//2)

print(arr)

s = np.array([[4], [14]])

print(arr.shape)
print(s.shape)

if any(np.equal(arr, s).any(1)):
    print("True")
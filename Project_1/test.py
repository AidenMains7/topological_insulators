import numpy as np

from joblib import Parallel, delayed



val1 = [0, 1]

val2 = np.array([0, 1])


def func1():
    return val1


def func2():
    return val2


arr1 = np.array(Parallel(n_jobs=1)(delayed(func1)() for _ in range(10))).T
arr2 = np.array(Parallel(n_jobs=1)(delayed(func2)() for _ in range(10))).T


print(arr1.shape)
print(arr2.shape)
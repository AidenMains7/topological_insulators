import numpy as np
a, b, c = 3, 4, 5
arr = np.arange(a*b*c).reshape(a,b,c)
print(arr)

vals = [a for a in arr]
for v in vals:
    print(v.shape)
import numpy as np

L_x, L_y = 20, 15

# For the lattice, I typically define the order of the dimensions with y first then x  so that if the system is plotted
# using plt.imshow, the y axis is vertical and the x axis is horizontal
# This is a consequence of the fact that arrays are displayed with rows being stacked vertically
system_size = L_x*L_y
lattice = np.arange(system_size).reshape((L_y, L_x))

# I iterate over x in the inner loop so that we move along the x axis in each layer before moving up to the next y layer
# The order of the loop nesting is completely arbitrary though, and all that matters is that 'lattice' is
# indexed as 'lattice[y, x]' and not 'lattice[x, y'
for y in range(L_y):
    for x in range(L_x):
        i = lattice[y, x]
        # ... rest of the code

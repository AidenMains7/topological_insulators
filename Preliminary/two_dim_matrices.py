import numpy as np
import scipy as sp


def hop_2D(L_y,L_x,isPerY,isPerX): #find cos, sin matrices for hopping in the dir of each axis; specify if either axis is periodic.
    #L_y: vertical size (n rows)
    #L_x: horizontal size (n cols)
    #isPerY,X: is periodic (boudnary cond) in respective direction?
    n_points = L_y*L_x
    lattice = np.arange(n_points).reshape((L_y,L_x))

    #initialize matrices as sparse dok matrices
    cosY = sp.sparse.dok_matrix((n_points,n_points), dtype=complex)
    sinY = sp.sparse.dok_matrix((n_points,n_points), dtype=complex)
    cosX = sp.sparse.dok_matrix((n_points,n_points), dtype=complex) 
    sinX = sp.sparse.dok_matrix((n_points,n_points), dtype=complex)

    for y in range(L_y):
        for x in range(L_x):

            y_nhbr = lattice[(y+1)%L_y, x] #next neighbor in positive vert dir (down)
            doHopY = isPerY or y_nhbr != 0 #only hop if periodic or next point is not wrapped 
            if doHopY:
                cosY[lattice[y,x], y_nhbr] = 1/2
                sinY[lattice[y,x], y_nhbr] = 1j/2


            x_nhbr = lattice[y, (x+1)%L_x] #next neighbor in positive hor dir (right)
            doHopX = isPerX or x_nhbr != 0
            if doHopX:
                cosX[lattice[y,x], x_nhbr] = 1/2
                sinX[lattice[y,x], x_nhbr] = 1j/2

    cosX += cosX.conj().T
    cosY += cosY.conj().T
    sinX += sinX.conj().T
    sinY += sinY.conj().T

    return cosX.toarray(), sinX.toarray(), cosY.toarray(), sinY.toarray()
import numpy as np
import scipy as sp

def wannier_matrices(Lx,Ly,periodicX,periodicY): #wannier matrices for uniform rectangular lattice

    n_points = Ly*Lx
    lattice = np.arange(n_points).reshape((Ly,Lx))


    cosY = sp.sparse.dok_matrix((n_points,n_points), dtype=complex) #initialize matrices as sparse dok matrices
    sinY = sp.sparse.dok_matrix((n_points,n_points), dtype=complex)
    cosX = sp.sparse.dok_matrix((n_points,n_points), dtype=complex) 
    sinX = sp.sparse.dok_matrix((n_points,n_points), dtype=complex)

    for y in range(Ly):
        for x in range(Lx):

            y_nhbr = lattice[(y-1)%Ly, x] #next neighbor in positive vert dir (up)
            doHopY = periodicY or y_nhbr != 0 #only hop if periodic or next point is not wrapped 
            if doHopY:
                cosY[lattice[y,x], y_nhbr] = 1/2
                sinY[lattice[y,x], y_nhbr] = 1j/2


            x_nhbr = lattice[y, (x+1)%Lx] #next neighbor in positive hor dir (right)
            doHopX = periodicX or x_nhbr != 0
            if doHopX:
                cosX[lattice[y,x], x_nhbr] = 1/2
                sinX[lattice[y,x], x_nhbr] = 1j/2

    cosX += cosX.conj().T #add hermitian conjugate
    sinX += sinX.conj().T
    cosY += cosY.conj().T
    sinY += sinY.conj().T

    return cosX.toarray(), sinX.toarray(), cosY.toarray(), sinY.toarray()



def wannier_hamiltonian(Lx,Ly,periodicX,periodicY,M):
    '''
    Constructs the hamiltonian for a uniform rectangular lattice
    '''
    t = 1 #constants, default to 1
    B = 1

    cosX, sinX, cosY, sinY = wannier_matrices(Lx,Ly,periodicX,periodicY)
    Id = sp.sparse.identity(Ly*Lx).toarray() #identity matrix for system size

    theta = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]]) #pauli matrices
    d = np.array([t*sinX, t*sinY, (M-4*B)*Id+2*B*(cosX + cosY)]) #

    H = 0 #hamiltonian
    for i in range(3):
        H += np.kron(d[i],theta[i])
    return H
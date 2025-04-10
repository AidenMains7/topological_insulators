import numpy as np
import scipy as sp
import two_dim_matrices


#construct the hamiltonian of a 2D system with width L_x and height L_y.
def hamiltonian_2D(L_y,L_x,periodicY,periodicX,M):
    t = 1 #constants, default to 1
    B = 1

    cosX, sinX, cosY, sinY = two_dim_matrices.hop_2D(L_y,L_x,periodicY,periodicX)
    Id = sp.sparse.identity(L_y*L_x).toarray() #identity matrix for system size

    theta = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]]) #pauli matrices
    d = np.array([t*sinX, t*sinY, (M-4*B)*Id+2*B*(cosX + cosY)]) #

    H = 0 #hamiltonian
    for i in range(3):
        H += np.kron(d[i],theta[i])
    return H

if __name__ == "__main__":
    H = hamiltonian_2D(50,50,False,False,1)
    print(H)
    print(H.shape)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



def construct_matrix(type,size): #construct cos, sin matrices for the 1D lattice.
    '''
    type is boundary type. string; 'open' or 'periodic'
    size is size of the 1D lattice. integer.
    '''
    #1D lattice with N sites
    lattice = np.arange(size) 

    #initialize matrices
    cos = np.zeros((size,size), dtype=complex)
    sin = np.zeros((size,size), dtype=complex)

    #value for "cos, sin to move"
    cosR = 1/2
    cosL = 1/2
    sinR = 1j/2 #j = sqrt(-1)
    sinL = -1j/2

    if (type=='open'):
        for i in range(size): #row index of matrix; lattice position
            #right and left neighbor positions on lattice
            #positive direction is right, negative direction is left
            neighborR = lattice[(i+1)%(size)]
            neighborL = lattice[(i-1)%(size)]

            #set value in matrices
            if (i-1 < 0): #if left neighbor is wrapped, do not set it
                cos[i][neighborR] = cosR 
                sin[i][neighborR] = sinR
            elif(i+1 > size-1): #if right neighbor is wrapped, do not set it 
                cos[i][neighborL] = cosL
                sin[i][neighborL] = sinL
            else:
                cos[i][neighborR] = cosR 
                sin[i][neighborR] = sinR
                cos[i][neighborL] = cosL
                sin[i][neighborL] = sinL

        return cos, sin


    elif (type=='periodic'):
        for i in range(size):
                #right and left neighbor positions on lattice
                neighborR = lattice[(i+1)%(size)]
                neighborL = lattice[(i-1)%(size)]

                #set value in matrices
                cos[i][neighborR] = cosR
                cos[i][neighborL] = cosL
                sin[i][neighborR] = sinR
                sin[i][neighborL] = sinL
        
        return cos, sin
            
    else:
        print('Not an accepted boundary condition type.')



def get_eigvalsh(matrix_list): #find the eigenvalues of a list of hermitian matrices
    '''
    matrix_list is a list of hermitiain matrices.
    '''
    matrix_amount = len(matrix_list)

    eigvalsh = [None]*matrix_amount
    for i in range(matrix_amount):
        eigvalsh[i] = np.linalg.eigvalsh(matrix_list[i])

    return eigvalsh



def plot_eigvals(eigvals_list,lattice_size,plot_info): #plot the eigen values vs. the lattice position
    '''
    eigvals_list is a list of arrays containing eigenvalues
    lattice_size is an integer, length of lattice
    plot_info is a dictionary of plotnames; perhaps not the 'best' way to do this
    
    '''
    amount = len(eigvals_list) #amount of lists of eigvals

    for i in range(amount): #iterate over amoutn of 
        t = np.arange(lattice_size) #lattice_size equal to eigvals_list[k] for any defined index k
        
        fig = plt.figure(figsize=(10,2))
        plt.plot(t,eigvals_list[i],label='eigenvalues') #plot
        plt.gca().update(dict(title=plot_info[i]['title'],
                              xlabel='Nth Lattice Position', 
                              ylabel='Eigenvalue value',)) #update plot with given info
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) #make x ticks integer

        derivative = np.diff(eigvals_list[i]) #derivative approximation; has length lattice_size-1
        derivative = np.concatenate(([0],derivative)) #fix length error
        plt.plot(t,derivative,label='derivative')

        plt.grid()
        plt.legend()
        plt.show() #show plot









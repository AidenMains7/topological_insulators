from nonhermitian_defects import DefectLattice, compute_eigenvectors_eigenvalues
from plotting import plot_ipr_figure

# Initialization parameters
defect_method = "vacancy"
Lx = 5
Ly = 7
m0 = -1.0 # on-site mass $m_0$
h0 = [0.5, 0.0, 0.0] # non-hermitian perturbation term $\vec{h}_0$

# Initialize the Lattice object
Lattice = DefectLattice(Lx, Ly, defect_method, pbc=True) # lattice object

# Plot the lattice with its defects highlighted
Lattice.plot()

# Compute the necessary observables
data_dict = compute_eigenvectors_eigenvalues(Lattice, m0, h0)
H = data_dict["hamiltonian"]
eigenvalues = data_dict["eigenvalues"]
L = data_dict["L"] # This is sum of all eigenvectors

# Plot the desired figure. Data files are automatically saved and read for future use. 
# Here, we automatically set the values for the (i) topological regime and (b) trivial regime.
# So for hdir='x', the function sets h0=[0.5, 0., 0.] while for hdir='z' --> h0=[0., 0., 0.5]
plot_ipr_figure(defect_method, [10, 20, 30], n_idxs=2, hdir='x')
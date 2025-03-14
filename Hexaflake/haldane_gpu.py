import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.linalg import eigh
from disorder_haldane import compute_geometric_data
import cProfile, pstats
from itertools import product


# Things to do on the GPU:
# 1. Compute the Hamiltonian
# 2. Compute disorder array
# 3. Compute the Bott index

def profile_function(func, *args, **kwargs):
	profiler = cProfile.Profile()
	profiler.enable()
	
	output = func(*args, **kwargs)

	profiler.disable()
	stats = pstats.Stats(profiler) 	
	stats = stats.sort_stats('cumtime')
	stats.print_stats(10)
	return output


def compute_hamiltonian_cuda(method, M_values, phi_values, t1, t2, geometric_data, device):
	
	NN = torch.from_numpy(geometric_data['NN'])
	NNN_CCW = torch.from_numpy(geometric_data['NNN_CCW'])
	hexaflake = torch.from_numpy(geometric_data['hexaflake'])

	parameters = torch.tensor(tuple(product(M_values, phi_values)), device=device)

	H = torch.zeros((len(parameters), NN.shape[0], NN.shape[1]), dtype=torch.complex64, device=device)
	indices = torch.arange(H.shape[0], device=device)
	H[:, indices, indices] = (parameters[:, 0] * ((-1)**indices))
	H[:, NNN_CCW] = -t2 * torch.sin(parameters[:, 1]) * 1j
	H[:, NNN_CCW.T] = t2 * torch.sin(parameters[:, 1]) * 1j

	if method == 'renorm':
		pass
	elif method == 'site_elim':
		H = H[:, hexaflake][:, :, hexaflake]
	
	return H








@profile_function
def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	geo_data = compute_geometric_data(2, True)
	H = compute_hamiltonian_cuda('site_elim', [1., 2.], [0., 1.], 1.0, 1.0, geo_data, device)
	print(H.shape)

if __name__ == "__main__":
	main()
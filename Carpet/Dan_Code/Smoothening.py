import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def mono_sigmoid(x_data, y_data, A):
	def sigmoid(x, k, x0):
		return A / (1 + np.exp(k * (x - x0)))

	popt, _ = curve_fit(sigmoid, x_data, y_data, p0=[1, np.median(x_data)])
	k_fit, x0_fit = popt

	return lambda x: A / (1 + np.exp(k_fit * (x - x0_fit)))


def bi_sigmoid(x_data, y_data, A):
	def sigmoid_double(x, A1, k1, x01, k2, x02):
		A2 = A - A1
		return A1 / (1 + np.exp(k1 * (x - x01))) + A2 / (1 + np.exp(k2 * (x - x02)))

	popt, _ = curve_fit(sigmoid_double, x_data, y_data, p0=[A / 2, 1, np.median(x_data) - 1, 1, np.median(x_data) + 1])
	A1_fit, k1_fit, x01_fit, k2_fit, x02_fit = popt
	A2_fit = A - A1_fit

	return lambda x: A1_fit / (1 + np.exp(k1_fit * (x - x01_fit))) + A2_fit / (1 + np.exp(k2_fit * (x - x02_fit)))


def sigmoid_fit(which, x_data, y_data, A):
	if which == 'mono':
		return mono_sigmoid(x_data, y_data, A)
	elif which == 'bi':
		return bi_sigmoid(x_data, y_data, A)
	else:
		raise ValueError("Parameter 'which' must be either 'mono' or 'bi'.")


def custom_fit_wrapper(x_data, y_data):
	y_max_mag_idx = np.argmax(np.abs(y_data))
	y_max_mag = y_data[y_max_mag_idx]

	which = 'mono' if round(np.abs(y_max_mag)) == 1 else 'bi'

	lambda_fit = sigmoid_fit(which, x_data, y_data, y_max_mag)

	return lambda_fit


def smooth_and_plot(data, title):
	x = data[0, :]
	num_y_sets = data.shape[0] - 1

	plt.figure(figsize=(10, 6))

	for i in range(1, num_y_sets + 1):
		y = data[i, :]

		cs = custom_fit_wrapper(x, y)

		x_fine = np.linspace(x.min(), x.max(), 500)
		y_fine = cs(x_fine)

		plt.scatter(x, y, label=f'Raw data {i}', alpha=0.6)

		plt.plot(x_fine, y_fine, label=f'Smoothened data {i}')

	plt.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Raw Data and Smoothened Curves')
	plt.savefig(title+'.svg', format='svg')
	plt.show()


def main():
	file = 'disorder_3.npz'
	fdata = np.load(file, allow_pickle=True)
	data = fdata['data'][:, 2:]
	smooth_and_plot(data, title=file[:-4])


if __name__ == '__main__':
	main()

import numpy as np
import matplotlib.pyplot as plt

def create_z_array(data):
    """
    Create a z_array from the given 2D data array.

    Parameters:
    data (ndarray): 2D numpy array where each row is [x, y, z].

    Returns:
    z_array (ndarray): 2D array of z values.
    x_bounds (tuple): (min_x, max_x)
    y_bounds (tuple): (min_y, max_y)
    """
    x_values = np.unique(data[:, 0])
    y_values = np.unique(data[:, 1])
    z_array = np.zeros((len(y_values), len(x_values)))
    z_array[(np.searchsorted(y_values, data[:, 1]), np.searchsorted(x_values, data[:, 0]))] = data[:, 2]

    x_bounds = (x_values.min(), x_values.max())
    y_bounds = (y_values.min(), y_values.max())

    return z_array, x_bounds, y_bounds

def plot_z_array(z_array, x_bounds, y_bounds, N=5, cmap='viridis', xlabel='x', ylabel='y', title='Phase Diagram'):
    """
    Plot the z_array using imshow with specified bounds and labels.

    Parameters:
    z_array (ndarray): 2D array of z values.
    x_bounds (tuple): (min_x, max_x)
    y_bounds (tuple): (min_y, max_y)
    N (int): Number of labels on each axis (including end points).
    cmap (str): Colormap to use for the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title of the plot.
    **kwargs: Additional keyword arguments for plt.imshow.
    """
    plt.imshow(z_array, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], origin='lower', aspect='auto', cmap=cmap)
    cbar = plt.colorbar(label='z value')
    cbar.set_label('z value', rotation=0)

    x_ticks = np.linspace(x_bounds[0], x_bounds[1], N)
    y_ticks = np.linspace(y_bounds[0], y_bounds[1], N)
    plt.xticks(ticks=x_ticks, labels=np.round(x_ticks, 2))
    plt.yticks(ticks=y_ticks, labels=np.round(y_ticks, 2))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel, rotation=0)
    plt.title(title)
    plt.show()

def main():
    # Example data array where each row is [x, y, z]
    data = np.array([
        [-2, -3, 0.1], [-2, -2.5, 0.2], [-2, -2, 0.3],
        [-1.75, -3, 0.4], [-1.75, -2.5, 0.5], [-1.75, -2, 0.6],
        [0, 0, 0.7], [0.5, 0.5, 0.8], [1, 1, 0.9]
        # Add more data points as needed
    ])

    # Create z_array and bounds
    z_array, x_bounds, y_bounds = create_z_array(data)

    # Plot the z_array
    plot_z_array(z_array, x_bounds, y_bounds, N=5, cmap='plasma', xlabel='X-axis', ylabel='Y-axis', title='Customized Phase Diagram')

if __name__ == "__main__":
    main()

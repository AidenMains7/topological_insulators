import numpy as np
import os
import h5py



def iterative_filename(filename):
    base, extension = os.path.splitext(filename)
    counter = 0
    new_filename = f"{base}_{counter}{extension}"
    while os.path.exists(new_filename):
        counter += 1
        new_filename = f"{base}_{counter}{extension}"
    return new_filename


def disorder_file(filename:str, shape:tuple, arr_name:str='disorder_vals'):
    filename = iterative_filename(filename)
    if filename[-3:] != '.h5':
        raise ValueError(f"Filename should end with '.h5'. Given filename is: {filename} ")
    with h5py.File(filename, 'w') as f:
        f.create_dataset(arr_name, data=np.empty(shape, dtype=np.float64))
    return filename



def read_file(filename:str):
    with h5py.File(filename, 'r') as f:
        pass


def main():
    disorder_file(10, np.empty(5), 'test.h5')
    read_file('test.h5')

if __name__ == "__main__":
    main()
import numpy as np


def _read_npz_data(filename:str) -> 'tuple[np.ndarray, dict]':
    """
    Will read .npz file

    Parameters:
    filename (str): Base filename with .npz extension
    """
    file_data = np.load(filename, allow_pickle=True)
    
    #if .npz file is of first form
    if file_data.get('arr_0') is not None:
        data = file_data['arr_0']
        parameters = None

    #if .npz file has parameters (second form)
    else:
        data = file_data['data']
        parameters = file_data['parameters'][()]

    return data, parameters


def print_params_nice(params:dict):
    for kw in params:
        if isinstance(params[kw], np.ndarray):
            print(f"{kw}: np.linspace({params[kw][0]}, {params[kw][-1]}, {params[kw].size})")
        else:
            print(f"{kw}: {params[kw]}")



def main():
    #Specify filename
    filename = 'Data/disorder_symmetry.npz'

    data, params = _read_npz_data(filename)
    print_params_nice(params)


if __name__ == "__main__":
    main()
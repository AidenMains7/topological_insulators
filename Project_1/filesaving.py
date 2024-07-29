import numpy as np
from pathlib import Path


def generate_filenames(base_filename:str, extensions:"list[str]") -> list:
    """
    Will return a list of filenames such that each extension is concatenated to the base_filename.
    """
    for ext in extensions:
        if ext[0] != '.':
            raise ValueError("Extensions must be of form '.png', for example.") 
        
    return [base_filename+ext for ext in extensions]


def generate_save_filename(filenames:"str|list") -> "list[str]":
    """
    Will check if a file name of the same exists. 

    Parameters:
    filenames (str|list|tuple): Must either be a single filename or a list of filenames

    Returns:
    new_filename (list): Will add an index suffix to the base filename. Begins at 0. If "filename_0.ext" exists, then will return "filename_1.ext", etc. Returns a list of end filenames
    """
    def generate_single(filename:str):
        base_filename, ext = filename[:-4], filename[-4:]
        
        def index_file(base_filename:str, idx:int=0):
            f = Path(base_filename+f'_{idx}'+ext)
            if f.is_file():
                result_filename = index_file(base_filename, idx+1)
            else:
                return str(f)
            return result_filename
        
        return index_file(base_filename)

    if isinstance(filenames, str):
        filenames = [filenames]

    if isinstance(filenames, list):
        return [generate_single(filename) for filename in filenames]


def make_directories(directories:"str|list") -> None:
    """
    Will check if each directory exists. 


    """

    if isinstance(directories, str):
        directories = [directories]

    if isinstance(directories, list):
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    
def return_all_file_type(directory:str, extension:str) -> list:
    """
    Will return a list of all specified file type in the provided directory.
    
    """

    p = Path(directory)
    files = list(p.iterdir())
    files = [str(file) if str(file)[-4:] == extension else None for file in files]
    files = list(filter(lambda a: a != None, files))
    return files


def print_dict_nice(dictionary:dict):
    for kw in dictionary:
        if isinstance(dictionary[kw], np.ndarray):
            print(f"{kw}: np.linspace({dictionary[kw][0]}, {dictionary[kw][-1]}, {dictionary[kw].size})")
        else:
            print(f"{kw}: {dictionary[kw]}")


def save_to_npz_intermittently(filename:str, data:np.ndarray, data_name:str):
    # If file exists
    if Path(filename).is_file():
        with np.load(filename, allow_pickle=True) as file_data:
            file_arrays = [file_data[name] for name in file_data.files]
            
            file_dict = {key: value for key, value in zip(file_data.files, file_arrays)}
            new_arr = {data_name: data}

        np.savez(filename, **new_arr, **file_dict)

    else:
        new_arr = {data_name: data}
        np.savez(filename, **new_arr)


def add_to_npz_file(filename:str, data:"np.ndarray | dict | list", data_name:str):
    with np.load(filename, allow_pickle=True) as file_data:
        arrs = [file_data[name] for name in file_data.files]
        file_dict = {key: value for key, value in zip(file_data.files, arrs)}
        new_arr = {data_name: data}

    np.savez(filename, **new_arr, **file_dict)
    

def reorder_npz_disorder(filename:str):
    with np.load(filename, allow_pickle=True) as file_data:
        
        names = file_data.files
        arrs = [file_data[name] for name in file_data.files]
        if "parameters" in names:
            arrs.pop(names.index("parameters"))
            parameters = file_data["parameters"][()]


        arrs = [arr.reshape(2, arr.size//2) for arr in arrs]
        X = arrs[0][1, :].reshape(1, arrs[0].shape[1])
        arrs = [arr[0, :] for arr in arrs]

        full_array = np.concatenate((X, arrs), axis=0)
    
    if "parameters" in names:
        np.savez(filename, data=full_array, parameters=parameters)
    else:
        np.savez(filename, data=full_array)




def main():
    reorder_npz_disorder('disorder_1.npz')


def main2():
    filedata = np.load('disorder_1.npz', allow_pickle=True)
    print([filedata[arr] for arr in filedata.files])



if __name__ == "__main__":
    main()
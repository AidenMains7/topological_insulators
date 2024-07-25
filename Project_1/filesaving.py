import numpy as np
import os, sys
from pathlib import Path

def generate_filenames(base_filename:str, extensions:list[str]) -> list:
    """
    Will return a list of filenames such that each extension is concatenated to the base_filename.
    """
    for ext in extensions:
        if ext[0] != '.':
            raise ValueError("Extensions must be of form '.png', for example.") 
        
    return [base_filename+ext for ext in extensions]


def generate_save_filename(filenames:str|list) -> "tuple[bool, str]":
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


def make_directories(directories:str|list) -> None:
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






def main():
    files = return_all_file_type("./Project_1", '.py')
    print(files)


if __name__ == "__main__":
    main()
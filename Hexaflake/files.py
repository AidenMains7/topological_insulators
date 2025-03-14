import glob, os
import numpy as np
import pandas as pd



def list_files_and_sizes(directory, min_size_mb):
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames]
    files = [file for file in files if os.path.isfile(file)]
    sizes = [os.path.getsize(file) for file in files]
    files_and_sizes = [(file, size) for file, size in zip(files, sizes) if size > min_size_mb * 1024 * 1024]
    files, sizes = zip(*files_and_sizes) if files_and_sizes else ([], [])
    return files, sizes

def main():
    files, sizes = list_files_and_sizes('.', 1)
    df = pd.DataFrame({'File': files, 'Size': sizes})
    df = df.sort_values(by='Size', ascending=False)
    print(df)


if __name__ == "__main__":
    main()
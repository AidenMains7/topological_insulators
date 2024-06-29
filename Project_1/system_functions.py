import cProfile
import pstats
import sys
import inspect

import psutil
from time import time, perf_counter


def profile_print_save(func:object, saveFile:bool=True, **kwargs):

    with cProfile.Profile() as profile:
        func_params = inspect.signature(func).parameters
        filtered_dict = {k: v for k, v in kwargs.items() if k in func_params}
        func(**filtered_dict)
        

    if saveFile:
        filename = 'profile.pstats'
        profile.dump_stats(filename)
    else: 
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.print_stats()



if sys.version_info[1] >= 11:

    def display_pstats_file(filename:str='profile.pstats'):
        results = pstats.Stats(filename)
        results.sort_stats(pstats.SortKey.TIME)
        results.print_stats(50)
        







def main():
    display_pstats_file()

if __name__ == "__main__":
    main()
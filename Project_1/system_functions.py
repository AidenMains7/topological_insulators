import cProfile
import pstats
import sys

import psutil
from time import time, perf_counter


def profile_print_save(funcs:"list[object]", saveFile:bool=True, **kwargs:dict):

    with cProfile.Profile() as profile:
        for func in funcs:
            func(**kwargs)
        

    if saveFile:
        filename = 'profile.pstats'
        profile.dump_stats(filename)
    else: 
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.print_stats()



if sys.version_info >= 11:

    def display_pstats_file(filename:str='profile.pstats'):
        results = pstats.Stats(filename)
        results.sort_stats(pstats.SortKey.TIME)
        results.print_stats()
        







def main():
    pass

if __name__ == "__main__":
    main()
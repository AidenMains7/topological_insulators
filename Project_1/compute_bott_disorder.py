import numpy as np
import inspect 
from time import time
import sys

from project_execute import bott_many, disorder_many
from phase_diagram import _save_npz_data, plot_disorder, plot_bott, plot_all_npz, plot_bott_imshow, _read_npz_data
from system_functions import profile_print_save

def run_computation(parameters:dict, computeBott:bool=True, computeDisorder:bool=True, plotBott:bool=False, plotDisorder:bool=False, bottFile:str=None) -> None:
    """
    Will run disorder computation with parameters specified in internal dictionary. Will save data from both Bott Index calculation and disorder calculation. May plot.
    """
    if not computeBott and not computeDisorder:
        raise ValueError("There is no point if both computeBott and computeDisorder are False.")
    
    if not computeBott and bottFile == None:
        raise ValueError("If computeBott=False, then a filename must be specified to read from")

    t0 = time()

    # Use applicable **kwargs
    bott_params = inspect.signature(bott_many).parameters
    filtered_dict = {k: v for k, v in parameters.items() if k in bott_params}

    if computeBott:
        # Compute the Bott Index [USES PARALLELIZATION]
        bott_arr = bott_many(**filtered_dict)

        # Save the Bott Index data
        end_filename_bott = _save_npz_data("bott.npz", data=bott_arr, parameters=parameters)
        print(f"Bott Index data saved as {end_filename_bott}")

        # Plot Bott Index data
        if plotBott:
            plot_bott_imshow(end_filename_bott, False, True, f"Bott Index, Method of {parameters['method']}, Order = {parameters['order']}")
    else:

        bott_arr, file_params = _read_npz_data(bottFile)

        # If read from file, will use the parameters specified above, barring what is necessary to maintain correctness.
        print("Reading from file:")
        for kw in ['method', 'order', 'pad_width', 'pbc', 'n', 'M_values', 'B_tilde_values']:
            parameters[f"{kw}"] = file_params[f"{kw}"]

            if kw in ['M_values', 'B_tilde_values']:
                val = parameters[f"{kw}"]
                print(f"{kw}: np.linspace{np.min(val), np.max(val), val.size}")
            else:
                print(f"{kw} = {parameters[kw]}")

        
    # Compute disorder
    if computeDisorder:
        # Use applicable **kwargs
        disorder_params = inspect.signature(disorder_many).parameters
        filtered_dict_2 = {k: v for k, v in parameters.items() if k in disorder_params}
        
        # Compute disorder data [USES PARALLELIZATION]
        disorder_arr = disorder_many(bott_arr=bott_arr, **filtered_dict_2)

        # Save the disorder data
        end_filename_disorder = _save_npz_data("disorder.npz", data=disorder_arr, parameters=parameters)
        print(f"Disorder data saved as {end_filename_disorder}")

        # Plot the disorder data
        if plotDisorder:
            plot_disorder(end_filename_disorder, False, True, f"Bott Index vs. Disorder, Method of {parameters['method']}, Order = {parameters['order']}")
    
    print(f"Total time taken: {time()-t0:.0f}s")



#----------main function implementation--------

def main():
    parameters = dict(
        method = "symmetry",
        order = 4,
        pad_width = 0,
        pbc = True,
        n = 2,
        M_values =         np.linspace(-2.0, 12.0, 51),
        B_tilde_values =   np.linspace(0.0, 2.0, 51),
        W_values =         np.linspace(0.0, 12.5, 1, endpoint=False) + (12.5/1),
        iterations = 1,
        E_F = 0.0,
        amount_per_idx = 1,
        num_jobs = 28,
        cores_per_job = 1,
        progress_bott = True,
        progress_disorder_iter = False, 
        progress_disorder_range = False,
        progress_disorder_many = True,
        KPM = False,
        N = 1024,
        task_timeout = None
    )

    run_computation(parameters, True, False, True, False)



if __name__ == "__main__":
    main()

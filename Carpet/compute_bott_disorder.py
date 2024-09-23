import numpy as np
import inspect 
from time import time, sleep
from filesaving import generate_filenames, generate_save_filename, add_to_npz_file, reorder_npz_disorder, return_all_file_type

from project_execute import bott_many, disorder_many
from plotting import plot_bott, plot_disorder
from system_functions import profile_print_save


def _read_from_npz(filename:str):
    filedata = np.load(filename, allow_pickle=True)
    data, params = filedata['data'], filedata['parameters'][()]
    return data, params


def run_computation(parameters:dict, computeBott:bool=True, computeDisorder:bool=True, plotBott:bool=False, plotDisorder:bool=False, bottFile:str=None) -> float:
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
        bott_outfile = generate_save_filename('bott.npz')[0]
        np.savez(bott_outfile, data=bott_arr, parameters=parameters)
        print(f"Bott Index data saved as {bott_outfile}")

        # Plot Bott Index data
        if plotBott:
            plot_bott(bott_outfile, False, True)
    else:

        bott_arr, file_params = _read_from_npz(bottFile)

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
        
        # Generate name for savefile
        disorder_outfile = generate_save_filename('disorder.npz')[0]
        filtered_dict_2.update({'disorder_outfile': disorder_outfile})

        # Compute disorder data
        disorder_arr = disorder_many(bott_arr=bott_arr, **filtered_dict_2)


        # Saving the disorder data
        if parameters["saveEach"]:
            add_to_npz_file(disorder_outfile, parameters, 'parameters')
            reorder_npz_disorder(disorder_outfile)
        else:
            np.savez(disorder_outfile, data=disorder_arr, parameters=parameters)
        print(f"Disorder data saved as {disorder_outfile}")

        # Plot the disorder data
        if plotDisorder:
            plot_disorder(disorder_outfile, False, True)
    
    print(f"Total time taken: {time()-t0:.0f}s")
    return disorder_outfile

#----------main function implementation--------

def main():
    parameters = dict(
        method = "symmetry",
        order = 3,
        pad_width = 0,
        pbc = True,
        n = 1,
        t1 = 1.0,
        t2 = 0.0,
        B = 1.0,
        M_values =         [6.5],
        B_tilde_values =   [0.0],
        W_values =         np.array([4.29]),
        iterations = 25,
        E_F = 0.0,
        KPM = False,
        N = 512,
        progress_bott = True,
        progress_disorder_iter = True, 
        progress_disorder_range = False,
        progress_disorder_many = False,
        doParallelIter = True,
        doParallelRange = False,
        doParallelMany = False,
        num_jobs = 4,
        cores_per_job = 1,
        saveEach = True
    )

    run_computation(parameters, True, True, False, False, None)



if __name__ == "__main__":
    main()
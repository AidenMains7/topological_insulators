import numpy as np
import inspect 
from time import time, sleep
from filesaving import generate_filenames, generate_save_filename, add_to_npz_file, reorder_npz_disorder, return_all_file_type
import pandas as pd

from project_execute import bott_many, disorder_many
from plotting import plot_bott, plot_disorder
from system_functions import profile_print_save


def _read_from_npz(filename:str):
    filedata = np.load(filename, allow_pickle=True)
    data, params = filedata['data'], filedata['parameters'][()]
    return data, params


def run_computation(parameters:dict, computeBott:bool=True, computeDisorder:bool=True, plotBott:bool=False, plotDisorder:bool=False, bottFile:str=None, disorderFile:str=None) -> float:
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
        if disorderFile is None:
            disorder_outfile = generate_save_filename('disorder.npz')[0]
        else:
            disorder_outfile = generate_save_filename(disorderFile)[0]
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



def resaving_data():
    files = return_all_file_type('zorganizing data/', '.npz')
    

    fs = 'zorganizing data/disorder_'
    fe = '.npz'

    files = [fs+f+fe for f in ['symmetry_crystalline']]

    w_list = []
    s = []
    p_list = []
    for f in files:
        fdata = np.load(f, allow_pickle=True)
        data = fdata['data']
        params = fdata['parameters'][()]
        print(data[:, :2])

        p_list.append(params)
        w_list.append(data[0, :])
        for i in range(data.shape[0]-1):
            s.append(data[i+1, :])
    
    if False:
        if np.average((w_list[0]-w_list[1])[2:]) != 0:
            print(np.average((w_list[0]-w_list[1])[2:]))
            sleep(1)
            raise ValueError



    good_list = []
    added_list = []
    for i in range(len(s)):
        if s[i][1] in [0.95, 0.925, 0.85] and s[i][1] not in added_list:
            added_list.append(s[i][1])
            good_list.append(s[i][:, np.newaxis])

    good_list = [w_list[0][:, np.newaxis]]+good_list
    

    new_arr = np.concatenate(good_list, axis=1).T

    print(np.round(new_arr, 2))

    if False:
        np.savez('./fig1_data/fig1_c.npz', data=new_arr, p_list=p_list)


#----------main function implementation--------

def main():
    parameters = dict(
        method = None,
        order = 4,
        pad_width = 0,
        pbc = True,
        n = 1,
        t1 = 1.0,
        t2 = 0.0,
        B = 1.0,
        M_values =         None,
        B_tilde_values =   None,
        W_values =         None,
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
        num_jobs = 28,
        cores_per_job = 1,
        saveEach = False
    )

    fdata = np.array(pd.read_csv('gen4/gen4_computations.csv'))
    for i in range(fdata.shape[0]):
        d = fdata[i]
        parameters['M_values'] = np.array(d[1]).astype(int)
        parameters['B_tilde_values'] = np.array(d[2]).astype(int)
        parameters['W_values'] = np.array(d[3])
        parameters['method'] = d[4]
        parameters['t2'] = 1.0 if d[5] == 'y' else 0.0
        
        f = f"disorder_gen4_{parameters['method']}_crystalline.npz" if d[5] == 'y' else f"disorder_gen4_{parameters['method']}.npz"
        run_computation(parameters, True, True, False, False, None, f)




if __name__ == "__main__":
    pass


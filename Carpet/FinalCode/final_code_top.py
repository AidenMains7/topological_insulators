import numpy as np
import inspect
from time import time
from filesaving import generate_save_filename, add_to_npz_file, reorder_npz_disorder
from project_execute import bott_many, disorder_many
from plotting import plot_bott, plot_disorder

def run_computation(parameters: dict, 
                    computeBott: bool = True, 
                    computeDisorder: bool = True, 
                    plotBott: bool = False, 
                    plotDisorder: bool = False, 
                    bottFile: str = None, 
                    disorderFile: str = None) -> str:
    """
    Run computations for the Bott Index and disorder, based on specified parameters.
    The data can be optionally saved and plotted.

    Parameters:
    ----------
    parameters : dict
        Dictionary containing parameters for the computations.
    computeBott : bool, optional
        Whether to compute the Bott Index (default is True).
    computeDisorder : bool, optional
        Whether to compute disorder (default is True).
    plotBott : bool, optional
        Whether to plot Bott Index data (default is False).
    plotDisorder : bool, optional
        Whether to plot disorder data (default is False).
    bottFile : str, optional
        Filename to read Bott Index data from if computeBott is False (default is None).
    disorderFile : str, optional
        Filename to save the disorder data (default is None).

    Returns:
    -------
    str
        The filename of the saved data (either for Bott Index or disorder).

    Raises:
    ------
    ValueError
        If both computeBott and computeDisorder are False.
        If computeBott is False and bottFile is not provided.
    """

    if not computeBott and not computeDisorder:
        raise ValueError("Both computeBott and computeDisorder cannot be False.")
    
    if not computeBott and bottFile is None:
        raise ValueError("If computeBott is False, bottFile must be specified.")
    
    start_time = time()

    # Bott Index computation
    if computeBott:
        bott_outfile = generate_save_filename('bott.npz')[0]
        bott_params = {k: v for k, v in parameters.items() if k in inspect.signature(bott_many).parameters}
        bott_many(**bott_params, saveEach=parameters.get("saveEachBott", False), filename=bott_outfile)
        add_to_npz_file(bott_outfile, {"parameters": parameters})
        print(f"Bott Index data saved as {bott_outfile}")
        
        if plotBott:
            plot_bott(bott_outfile, save=False, show=True)
    else:
        with np.load(bottFile, allow_pickle=True) as fdata:
            try: 
                bott_arr = fdata['data']
                file_params = fdata['parameters'][()]
            except KeyError:
                bott_arr = fdata['bott_index']
                file_params = fdata['parameters'][()]

        print("Reading Bott Index data from file:")
        for key in ['method', 'order', 'pad_width', 'pbc', 'n', 'M_values', 'B_tilde_values']:
            parameters[key] = file_params[key]
            if key in ['M_values', 'B_tilde_values']:
                print(f"{key}: np.linspace({np.min(parameters[key])}, {np.max(parameters[key])}, {parameters[key].size})")
            else:
                print(f"{key} = {parameters[key]}")
    
    # Disorder computation
    if computeDisorder:
        disorder_outfile = disorderFile or generate_save_filename('disorder.npz')[0]
        disorder_params = {k: v for k, v in parameters.items() if k in inspect.signature(disorder_many).parameters}
        disorder_arr = disorder_many(bott_arr=bott_arr, **disorder_params)
        
        if parameters.get("saveEachDisorder", False):
            add_to_npz_file(disorder_outfile, parameters, 'parameters')
            reorder_npz_disorder(disorder_outfile)
        else:
            np.savez(disorder_outfile, data=disorder_arr, parameters=parameters)
        print(f"Disorder data saved as {disorder_outfile}")
        
        if plotDisorder:
            plot_disorder(disorder_outfile, save=False, show=True)
    
    print(f"Total time taken: {time() - start_time:.0f} seconds")

    return disorder_outfile if computeDisorder else bott_outfile


def main():
    """
    Main function to generate a heatmap for Bott Index values over a range of parameters.
    It sets up parameters and runs the computation.
    """
    parameters = dict(
        method='symmetry',
        order=3,
        pad_width=0,
        pbc=True,
        n=1,
        t1=1.0,
        t2=1.0,
        B=1.0,
        M_values=np.linspace(-2.0, 12.0, 15),
        B_tilde_values=np.linspace(0.0, 2.0, 3),
        W_values=None,
        iterations=None,
        E_F=0.0,
        KPM=False,
        N=512,
        progress_bott=True,
        progress_disorder_iter=True, 
        progress_disorder_range=False,
        progress_disorder_many=False,
        doParallelIter=True,
        doParallelRange=False,
        doParallelMany=False,
        num_jobs=4,
        cores_per_job=1,
        saveEachDisorder=False,
        saveEachBott=True
    )

    for method in ['symmetry']:
        parameters['method'] = method
        run_computation(parameters, computeBott=True, computeDisorder=False, plotBott=False, plotDisorder=False)

if __name__ == "__main__":
    main()

    
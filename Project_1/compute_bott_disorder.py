import numpy as np
from project_execute import _init_environment, _many_bott, _many_disorder
from phase_diagram import _save_npz_data, plot_disorder, plot_bott, plot_all_npz, plot_bott_imshow

import inspect

def run_computation(doDisorder:bool=True, plotBott:bool=False, plotDisorder:bool=False) -> None:
    """
    Will run disorder computation with parameters specified in internal dictionary. Will save data from both Bott Index calculation and disorder calculation. May plot.
    """

    # Define parameters
    parameters = dict(
        method = "symmetry",
        order = 3,
        pad_width = 0,
        pbc = True,
        n = 1,
        M_values =         np.linspace(-2.0, 12.0, 2),
        B_tilde_values =   np.linspace(0.0, 2.0, 2),
        W_values =         np.linspace(0.0, 12.5, 3, endpoint=False) + (12.5/28),
        iterations_per_disorder = 1,
        E_F = 0.0,
        num_jobs = 4,
        cores_per_job = 1,
        sparse = False,
        progress_bott = True,
        progress_disorder_range = True,
        progress_disorder_iter = False
    )

    # Initialize the environment to avoid cannibalization
    _init_environment(cores_per_job=parameters["cores_per_job"])

    # Use applicable **kwargs
    bott_params = inspect.signature(_many_bott).parameters
    filtered_dict = {k: v for k, v in parameters.items() if k in bott_params}

    # Compute the Bott Index [USES PARALLELIZATION]
    bott_arr = _many_bott(**filtered_dict, progress=parameters['progress_bott'])

    # Save the Bott Index data
    end_filename_bott = _save_npz_data("bott.npz", data=bott_arr, parameters=parameters)
    print(f"Bott Index data saved as {end_filename_bott}")

    # Plot Bott Index data
    if plotBott:
        plot_bott_imshow(end_filename_bott, False, True, f"Bott Index, Method of {parameters['method']}, Order = {parameters['order']}")


    # Compute disorder
    if doDisorder:
        # Use applicable **kwargs
        disorder_params = inspect.signature(_many_disorder).parameters
        filtered_dict_2 = {k: v for k, v in parameters.items() if k in disorder_params}
        
        # Compute disorder data [USES PARALLELIZATION]
        disorder_arr = _many_disorder(bott_arr=bott_arr, **filtered_dict_2)

        # Save the disorder data
        end_filename_disorder = _save_npz_data("disorder.npz", data=disorder_arr, parameters=parameters)
        print(f"Disorder data saved as {end_filename_disorder}")

        # Plot the disorder data
        if plotDisorder:
            plot_disorder(end_filename_disorder, False, True, f"Bott Index vs. Disorder, Method of {parameters['method']}, Order = {parameters['order']}")



#----------main function implementation--------

def main():
    run_computation()


def main2():
    plot_all_npz()


if __name__ == "__main__":
    main2()
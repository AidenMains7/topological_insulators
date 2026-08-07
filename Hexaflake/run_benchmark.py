from disorder_haldane import compute_disorder

n_jobs = -2 # number of cores to run parallel on
n_chunks = 1 # number of chunks; saves in batches of each chunk then concatenates it all at the end
data_file = "./Hexaflake/Data/" + "site_elim_g4_selected_points_for_w1.h5"
generation = 4
n_iterations = 30
compute_disorder(data_file, 'site_elim', generation, 1.0, n_iterations, 
                 1.0, 1.0, n_jobs, True, False, False, n_chunks)

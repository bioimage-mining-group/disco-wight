import numpy as np
from dwight.utils import load_csv_coords
from dwight.simulation_corrupter import SimulationCorrupter
import os
import csv
from tqdm import tqdm
np.random.seed(2022)

"""
Loads the coordinates of the ground truth in CSV format, corrupts them,
then save the results in CSV format.
"""

path_to_root = ...

path_to_data = f'{path_to_root}/simulations/coords/gt'
path_to_save = f'{path_to_root}/simulations/coords'

files = os.listdir(path_to_data)

corrupter = SimulationCorrupter()


# fnfp_rates = np.linspace(0,0.2,5)
fn_rates = np.arange(0.0, 0.5, 0.05)
fp_rates = np.arange(0.0, 0.5, 0.05)

for ind_fn, fn_rate in enumerate(tqdm(fn_rates)):
    for ind_fp, fp_rate in enumerate(fp_rates):

        fnfp_rate_str = f'fn{ind_fn}_fp{ind_fp}'

        path_to_corrupted_coords = f'{path_to_save}/coords_{fnfp_rate_str}'
        path_to_dicts = f'{path_to_save}/dicts_{fnfp_rate_str}'
        

        os.mkdir(path_to_corrupted_coords)
        os.mkdir(path_to_dicts)

        for filename in files:

            coords = load_csv_coords(
                path_to_csv=f'{path_to_data}/{filename}'
            )

            coords = corrupter.remove_fn_from_coords(
                coords, 
                fn_rate=fn_rate
            )

            coords, dict_new_old = corrupter.add_fp_to_coords(
                coords, 
                fp_rate=fp_rate,
                return_dict=True
            )


            np.savetxt(
                f'{path_to_corrupted_coords}/{filename}',
                coords,
                delimiter=','
            )

            w = csv.writer(open(f'{path_to_dicts}/dict_{filename}', "w"))

            # loop over dictionary keys and values
            for key, val in dict_new_old.items():

                # write every key and value to file
                w.writerow([key, val])










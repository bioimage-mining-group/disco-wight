import numpy as np
from dwight.utils import load_csv_coords
from dwight.simulation_corrupter import SimulationCorrupter
import os
import csv
from functools import partial
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

rates = np.arange(0.0, 0.5, 0.05)

N_tests = 10

error_types = [
    'fn',
    'fp',
    'merge',
    'split'
]

error_functions = [
    corrupter.remove_fn_from_coords,
    corrupter.add_fp_to_coords,
    partial(corrupter.add_merge_to_coords, max_distance=1.1*(2*8)), # corresponds to 1.1 * d0
    corrupter.add_split_to_coords
]

for etype, efunc in tqdm(zip(error_types, error_functions)):

    os.mkdir(f'{path_to_save}/{etype}')

    for i in range(N_tests):

        os.mkdir(f'{path_to_save}/{etype}/{i}')

        for rate in rates:

            rate_str = f'{rate:.2f}'

            filename = str(int(100 * rate)).zfill(3)

            path_to_corrupted_coords = f'{path_to_save}/{etype}/{i}/coords_{etype}_{filename}'
            path_to_dicts = f'{path_to_save}/{etype}/{i}/dicts_{etype}_{filename}'
            

            os.mkdir(path_to_corrupted_coords)
            os.mkdir(path_to_dicts)

            for filename in files: # each time point

                coords = load_csv_coords(
                    path_to_csv=f'{path_to_data}/coords/{filename}'
                )

                coords, dict_new_old = efunc(coords, rate, return_dict=True)

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










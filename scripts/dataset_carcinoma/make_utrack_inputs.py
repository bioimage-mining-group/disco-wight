import tifffile
import numpy as np
from dwight.utils import delete_content_of_dir, array_to_utrack3d_inputs, rescale_array
from pathlib import Path

path_to_root = ...

path_to_data = f'{path_to_root}/dataset_carcinoma/data'

models = ['sd', 'cyto', 'cyto2', 'gt']

for dataset_index in [1,2]:

	for model in models:

		if model == 'gt':
			path_to_labels = f'{path_to_data}/labels_{dataset_index}.tif'
		elif model == 'sd':
			path_to_labels = f'{path_to_data}/labels_predicted_sd_pretrained_{dataset_index}.tif'
		else:
			path_to_labels = f'{path_to_data}/labels_predicted_cp_pretrained_{model}_{dataset_index}.tif' 

		path_to_save = f'{path_to_root}/dataset_carcinoma/utrack/{dataset_index}/{model}/utrack_inputs'

		Path(path_to_save).mkdir(parents=True, exist_ok=True)

		labels = tifffile.imread(path_to_labels)

		if model == 'gt':

			labels = np.array(
				[rescale_array(elem,zoom_factor=(2.5,0.5,0.5), order=0) for elem in labels]
			)

		delete_content_of_dir(path_to_dir=path_to_save, content_type='tif')

		array_to_utrack3d_inputs(
			array = labels,
			path_to_save=path_to_save,
			transpose=True
		)






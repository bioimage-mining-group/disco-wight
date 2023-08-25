import tifffile
from dwight.utils import delete_content_of_dir, array_to_utrack3d_inputs
from pathlib import Path

path_to_root = ...
path_to_data = f'{path_to_root}/dataset_organoid/data'


names = [
	'labels_predicted_cp_pretrained_cyto2.tif',
	'labels_predicted_sd_pretrained.tif',
	'labels_predicted_best_cp.tif',
	'labels_predicted_best_sd.tif',
	'square_annotations_gt.tif'
]

models = [
	'cp_cyto2',
	'sd',
	'cp_trained',
	'sd_trained',
	'gt'
]

for name,model in zip(names, models):

	path_to_save = f'{path_to_root}/dataset_carcinoma/utrack/{model}/utrack_inputs'

	Path(path_to_save).mkdir(parents=True, exist_ok=True)
	delete_content_of_dir(path_to_dir=path_to_save, content_type='tif')

	array_to_utrack3d_inputs(
		tifffile.imread(f'{path_to_data}/{name}'),
		path_to_save
	)



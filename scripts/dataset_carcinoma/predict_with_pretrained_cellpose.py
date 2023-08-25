import numpy as np
import tifffile
from tqdm import tqdm
import tifffile
from cellpose import models
import napari


"""
Loads organoid data and predict with Cellpose pretrained models
'cyto' and 'cyto2'.
"""


predict = True

path_to_root = ...

path_to_dataset = f'{path_to_root}/dataset_carcinoma/data'
path_to_save = f'{path_to_root}/dataset_carcinoma/data'

viewer = napari.Viewer()

for data_index in [1,2]:

    data = tifffile.imread(f'{path_to_dataset}/data_{data_index}.tif')
    labels_gt = tifffile.imread(f'{path_to_dataset}/labels_{data_index}.tif')

    viewer.add_image(data)
    viewer.add_labels(labels_gt)

    for model in tqdm(['cyto', 'cyto2', 'nuclei']):
        model = models.Cellpose(gpu=False, model_type=model)   

        labels = []

        for image in tqdm(data):
            # image = np.expand_dims
            out = model.eval(image, diameter=15,
                                    channels = [0,0],
                                    net_avg=False,
                                    augment=False,do_3D=True,anisotropy=5, z_axis=0, normalize=True)

            labels.append(out[0])
        labels = np.array(labels)

        tifffile.imwrite(f'{path_to_save}/labels_predicted_cp_pretrained_{model}_{data_index}.tif', labels)
        viewer.add_labels(labels)


napari.run()

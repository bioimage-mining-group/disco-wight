import numpy as np
import tifffile
from tqdm import tqdm
import tifffile
from cellpose import models


"""
Loads organoid data and predict with Cellpose pretrained model 'nuclei'.
"""

 
path_to_root = ...

path_to_dataset = f'{path_to_root}/dataset_carcinoma/data'
path_to_save = f'{path_to_root}/dataset_carcinoma/data'


data = tifffile.imread(f'{path_to_dataset}/square.tif')
model = models.Cellpose(gpu=True, model_type='nuclei')   

labels = []

for image in tqdm(data):
    # image = np.expand_dims
    out = model.eval(image, diameter=12,
                            channels = [0,0],
                            net_avg=False,
                            augment=False,do_3D=True,anisotropy=1.6, z_axis=0, normalize=True)

    labels.append(out[0])
labels = np.array(labels)

tifffile.imwrite(f'{path_to_save}/labels_predicted_cp_pretrained_nuclei.tif', labels)


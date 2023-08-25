import os 
# Suppress some level of logs
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from stardist.models import StarDist3D
from csbdeep.utils import normalize

import numpy as np
import tifffile
from tqdm import tqdm

from dwight.utils import rescale_array

"""
Loads organoid data, normalize and rescale, then predict with Stardist3D pretrained model.
"""

path_to_root = ...

path_to_dataset = f'{path_to_root}/dataset_carcinoma/data'
path_to_save = f'{path_to_root}/dataset_carcinoma/data'


model = StarDist3D.from_pretrained('3D_demo')
model.config.use_gpu=True

for data_index in [1,2]:

    data = tifffile.imread(f'{path_to_dataset}/data_{data_index}.tif')

    labels_gt = tifffile.imread(f'{path_to_dataset}/labels_{data_index}.tif')

    data = [normalize(x,1,99.8,axis=(0,1,2)) for x in data]
    data = np.array([rescale_array(elem,zoom_factor=(2.5,0.5,0.5), order=1) for elem in tqdm(data)])

    labels = np.array(
        [model.predict_instances(elem)[0] for elem in tqdm(data)]
    )

    labels_gt_scaled = [rescale_array(elem,zoom_factor=(2.5,0.5,0.5), order=0) for elem in tqdm(labels_gt)]

    tifffile.imwrite(f'{path_to_save}/labels_predicted_sd_pretrained_{data_index}.tif', labels)

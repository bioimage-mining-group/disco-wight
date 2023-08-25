import os 
# Suppress some level of logs
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from stardist.models import StarDist3D
from csbdeep.utils import normalize

import numpy as np
import tifffile
from tqdm import tqdm
from csbdeep.utils import normalize

"""
Loads organoid data and predict with Stardist3D pretrained model.
"""


path_to_root = ...

path_to_dataset = f'{path_to_root}/dataset_organoid/data'
path_to_save = f'{path_to_root}/dataset_organoid/data'

data = tifffile.imread(f'{path_to_dataset}/square.tif')


# prints a list of available models
StarDist3D.from_pretrained()
model = StarDist3D.from_pretrained('3D_demo')
model.config.use_gpu=True

norm_data = [normalize(x,1,99.8,axis=(0,1,2)) for x in tqdm(data)]
labels = np.array(
    [model.predict_instances(elem,n_tiles=model._guess_n_tiles(elem))[0] for elem in tqdm(norm_data)]
)

tifffile.imwrite(f'{path_to_save}/labels_predicted_sd_pretrained.tif', labels)

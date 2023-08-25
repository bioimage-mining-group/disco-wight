import napari
import numpy as np
import tifffile
from dwight.utils import make_bounding_box
import matplotlib.pyplot as plt

"""
Loads predictions from all detectors and plot them along 
each other for comparison with the ground truth. 
"""

dataset_index = 2

path_to_root = ...

path_to_data = f'{path_to_root}/dataset_carcinoma/data'

data = tifffile.imread(f'{path_to_data}/data_isoscaled_{dataset_index}.tif')

labels_gt = tifffile.imread(f'{path_to_data}/labels_isoscaled_{dataset_index}.tif')

labels_cp_cyto = tifffile.imread(f'{path_to_data}/labels_predicted_cp_pretrained_cyto_{dataset_index}.tif')
labels_cp_cyto2 = tifffile.imread(f'{path_to_data}/labels_predicted_cp_pretrained_cyto2_{dataset_index}.tif')

labels_sd = tifffile.imread(f'{path_to_data}/labels_predicted_sd_pretrained_{dataset_index}.tif')

crop_side = int(data.shape[2]/4)
sls = (
    slice(0, None),
    slice(0, None),
    slice(2*crop_side+10, 3*crop_side+10),
    slice(1*crop_side+10, 2*crop_side+10)    
)
scale_crop=(1,1,1.9,1.9)

mask = np.zeros(data.shape)

np.random.seed(20231)

def add_data(viewer, array, image=False):
    array[:,:,:int(np.ceil(crop_side*1.9)),:int(np.ceil(crop_side*1.9))] = 0
    if not image:
        cmap = plt.get_cmap('Paired')
        cols = [cmap(i) for i in range(12)]

        colors = {i: cols[np.random.randint(0,12)] for i in np.unique(array)[1:]}
        viewer.add_labels(array, scale=(1,1,1,1),color=colors)
    else:
        viewer.add_image(array, scale=(1,1,1,1))
        viewer.layers[-1].interpolation = 'nearest'
    
    viewer.add_image(make_bounding_box(data.shape), blending='additive')
    viewer.layers[-1].interpolation = 'nearest'

    viewer.add_image(
        make_bounding_box((*data.shape[:2],crop_side, crop_side)),
        blending='additive',
        translate=(0,0,2*crop_side+10,1*crop_side+10),
        colormap='reds'
    )
    viewer.layers[-1].interpolation = 'nearest'

    if not image:
        viewer.add_labels(array[sls],scale=scale_crop, color=colors)
    else:
        viewer.add_image(array[sls],scale=scale_crop)
        viewer.layers[-1].interpolation = 'nearest'
    
    viewer.add_image(make_bounding_box(array[sls].shape, bb_width=1), blending='opaque',scale=scale_crop, colormap='reds')
    viewer.layers[-1].interpolation = 'nearest'


###
viewer = napari.Viewer(ndisplay=3)

add_data(viewer, data, image=True)
add_data(viewer, labels_cp_cyto2)
add_data(viewer, labels_cp_cyto)
add_data(viewer, labels_sd)
add_data(viewer, labels_gt)

###
# viewer = napari.Viewer(ndisplay=3)




viewer.grid.enabled = True
viewer.grid.shape = (3,2)
viewer.grid.stride=5
viewer.dims.set_current_step(0,3)


viewer.layers.move_selected(-1,-2)
viewer.layers.move_selected(-1,-2)

napari.run()
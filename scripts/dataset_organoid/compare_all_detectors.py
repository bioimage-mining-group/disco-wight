import numpy as np
import tifffile
import napari
from dwight.utils import make_bounding_box

"""
Loads predictions from all detectors and plot them along 
each other for comparison with the ground truth. 
"""


scale4d = (1,1.6,1,1)

path_to_root = ...

path_to_data = f'{path_to_root}/dataset_organoid/data'

data = tifffile.imread(f'{path_to_data}/square.tif')
print(data.shape)
labels_gt = tifffile.imread(f'{path_to_data}/square_annotations_gt.tif')

all_names = [
    'cp_cyto2',
    'cp_trained',
    'sd',
    'sd_trained'
]
label_names = [
    'labels_predicted_cp_pretrained_cyto2.tif',
    'labels_predicted_best_cp.tif',
    'labels_predicted_sd_pretrained.tif',
    'labels_predicted_best_sd.tif'
]

viewer = napari.Viewer(ndisplay=3)
viewer2 = napari.Viewer()


viewer.add_image(data[11],scale=scale4d[1:])
bb_3D = make_bounding_box(data[0].shape)

viewer.camera.angles = (-3.9,15,75)

roi_centers = [
    (19,48,78),
    (12,154,110),
    (17,116,82),
    (42,25,50),
]

w=25
crops = [
    (slice(z,z+1),slice(y-w,y+w),slice(x-w,x+w)) for z,y,x in roi_centers
]
colors = {1:(1,0,0,1), 2:(0,1,1,1), 3:(0,1,0,1), 4:(1,0,1,1)}


for i,(crop,col) in enumerate(zip(crops[::-1],colors),start=1):
    bb_crop = make_bounding_box(bb_shape=[sl.stop-sl.start for sl in crop[1:]],bb_width=3)
    bb_crop = np.expand_dims(bb_crop, 0)
    viewer.add_labels(
        (i*bb_crop).astype(int),
        color={i:colors[i]},
        translate=[sl.start*s for sl,s in zip(crop, scale4d[1:])],
        blending='translucent',
        rendering='translucent',
        opacity=1
    )

    crop_data = data[11][crop]
    crop_labels_gt = labels_gt[11][crop]

    bb = make_bounding_box(bb_crop.shape,bb_width=2)
    bb_col = make_bounding_box(bb_crop.shape,bb_width=4)

    

    for name, label_name in zip(all_names[::-1], label_names[::-1]):

        labels = tifffile.imread(f'{path_to_data}/{label_name}')

        
        crop_labels = labels[11][crop]

        viewer2.add_image(crop_data)
        viewer2.add_labels(crop_labels, name=name)
        viewer2.add_image(bb, blending='additive')

    viewer2.add_image(crop_data)
    viewer2.add_labels(crop_labels_gt, name='gt')
    viewer2.add_image(bb, blending='additive')
    
    viewer2.add_image(crop_data)
    viewer2.add_labels(crop_labels_gt, name='gt',visible=False)
    viewer2.add_labels(
        (i*bb_col).astype(int),
        color={i:colors[i]}
    )


viewer.add_image(bb_3D, scale=scale4d[1:], blending='opaque')


viewer2.grid.shape =(len(crops),len(all_names)+2) 
viewer2.grid.stride = 3
viewer2.grid.enabled = True

napari.run()


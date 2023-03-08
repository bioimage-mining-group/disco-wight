import numpy as np
import napari
import tifffile
import matplotlib.pyplot as plt

import packajules.utils as jutils
from packajules.dynROI import DynROI
from packajules.render import Renderer

from packajules.segmentation_classes import Tracks





scale4d = (5,1,0.621,0.621)

path2data = '/home/jvanaret/data/data_trackability_study'

data = tifffile.imread(f'{path2data}/raw_data/square.tif')



annotations_gt = tifffile.imread(
    f'{path2data}/raw_data/square_annotations_gt.tif'
)
tracks_gt = Tracks()
ty_gt = tracks_gt.load_utrack3d_json(
    path_to_json=f'{path2data}/utrack/gt/saved_tracks/tracks.json',
    scale=scale4d,
    path_to_trackability_json=f'{path2data}/utrack/gt/saved_tracks/trackability.json'
)


renderer = Renderer()

roi_fringes=(28,40,40)

ROI_coordinates = [
    (range(8,13), [int(elem/s) for elem,s in zip([55-28, 30, 20], scale4d[1:])])
]


list_of_tinds, list_of_center_inds = ROI_coordinates[0]

dynROI = DynROI().from_coords(
        list_of_tinds=list_of_tinds,
        list_of_center_inds=list_of_center_inds,
        world_zyx_shape=data.shape[1:],
        roi_fringes=roi_fringes
    )
print('starting orthorender')

global_dynROI = DynROI().from_full_array(world_array=data)

coherent_annotations_gt = tracks_gt.coherent_segmentation_from_tracks(
    raw_segmented_labels=annotations_gt
)


print('starting 3D render')


viewer_render3D_local = renderer.render_dynROI(
        dynROI=dynROI,
        data=data,
        labels_data=coherent_annotations_gt,
        tracks=tracks_gt,
        scale4d=scale4d,
        window_name=f'ROI 3D render',
        view3D=True
    )

viewer_ortho_local = renderer.orthorender_dynROI(
    dynROI=dynROI,
    data=data,
    labels_data=coherent_annotations_gt,
    tracks=tracks_gt,
    scale4d=scale4d,
    window_name=f'ROI orthorender',
    view3D=False
)




for layer in viewer_render3D_local.layers:
    if 'tracks' in layer.name:

        tracks_props = layer.properties 

        tracks_indices = np.unique(layer.data[:,0])

        sub_track = tracks_gt.get_subset_of_tracks(
            selected_track_indices=tracks_indices,
            filters_dict={
                'tinds':(8, 13),
                'xinds':(ROI_coordinates[0][1][2]-roi_fringes[2],ROI_coordinates[0][1][2]+roi_fringes[2]),
                'yinds':(ROI_coordinates[0][1][1]-roi_fringes[1],ROI_coordinates[0][1][1]+roi_fringes[1]),
                'zinds':(ROI_coordinates[0][1][0]-roi_fringes[0],ROI_coordinates[0][1][0]+roi_fringes[0])
            }
        )

        subtrack_indices = [track.index for track in sub_track]

        foo = [[np.min(track.nodes_trackabilities)]*track.number_of_frames for track in sub_track ]
        trackability_min_local = [elem for l in foo for elem in l]

        layer.properties = {
            'trackability_min':trackability_min_local
        }

        layer.color_by = 'trackability_min'
        layer.colormap = 'plasma'




napari.run()
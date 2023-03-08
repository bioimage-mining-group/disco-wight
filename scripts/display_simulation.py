import tifffile
import numpy as np
import napari
import os

from packajules.segmentation_classes import Tracks
from packajules.dynROI import DynROI
from packajules.render import Renderer
from organo_simulator.utils import load_csv_coords
from packajules.utils import make_bounding_box

scale = (1,)*4 # min/frame, um/pix, um/pix, um/pix

path2coords = '/home/jvanaret/data/data_trackability_study/simulations/test_long/coords'

files = sorted(os.listdir(path2coords))


gt_tracks = []
gt_points = []

for indt,filename in enumerate(files):
    coords = load_csv_coords(
        path_to_csv=f'{path2coords}/{filename}'
    )

    for ind_track, coord in enumerate(coords, start=1):


        rescaled_coords = [c*300/(2*130)-40 for c in coord]

        gt_tracks.append(
            [int(ind_track),int(indt), *rescaled_coords]
        )
        gt_points.append(
            [int(indt), *rescaled_coords]
        )

gt_tracks = sorted(gt_tracks, key=lambda l: l[0])
total_time = len(np.unique(np.array(gt_tracks)[:,1]))

# 3D Render
viewer = napari.Viewer() 
bb = make_bounding_box((total_time,)+(300-80,)*3,bb_width=2)

viewer.add_tracks(
    np.array(gt_tracks),
    name='gt_tracks',
    color_by='track_id',
    properties={}
)
viewer.add_image(bb,blending='additive')

cols = np.zeros(shape=(400,4))
# cols[:,2]=1
# cols[:,1]=1
cols[:,[0,1,2]] = np.clip(np.random.normal(loc=0.5,scale=0.25, size=(400,3)),0,1)
cols[:,3] = np.clip(np.random.normal(loc=0.8,scale=0.1, size=400),0,1)

cols = np.concatenate([cols]*total_time,axis=0)

viewer.add_points(
    gt_points,
    name='gt_points',
    size=np.ones(shape=(len(gt_points),4))*16,
    face_color=cols
)
viewer.add_image(bb,blending='additive')




viewer.grid.shape =(1,2) 
viewer.grid.stride = 2
viewer.grid.enabled = True
roi_fringes=(15,15,15)
list_of_center_inds = [100,100,100]

utracks = Tracks().load_napari_tracks(napari_tracks=gt_tracks)

dynROI = DynROI().from_coords(
    list_of_tinds=range(total_time),
    list_of_center_inds=list_of_center_inds,
    world_zyx_shape=(300-80,)*3,
    roi_fringes=roi_fringes
)


renderer = Renderer()


renderer.render_dynROI(
    dynROI=dynROI,
    tracks=utracks,
    points=gt_points,
    scale4d=scale,
    view3D=True
)


viewer.dims.ndisplay = 3
viewer.grid.shape = (1,len(viewer.layers))

for layer in viewer.layers: layer.scale = scale


napari.run()
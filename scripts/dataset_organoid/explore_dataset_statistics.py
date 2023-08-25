from dwight.tracking_classes import Tracks
from scipy.spatial import KDTree

import numpy as np
import tifffile
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from labellines import labelLines

"""
Loads the ground truth labels, extracts the centroid coordinates at each frame, 
computes the distributions of the nearest neighbor distances and of the frame 
to frame displacements, and plots the results.
"""

scale = [5,1,0.621,0.621]


path_to_root = ...
path_to_dataset = f'{path_to_root}/dataset_organoid/data'
path_to_tracks = f'{path_to_root}/dataset_organoid/utrack/gt'

labels = tifffile.imread(f'{path_to_dataset}/square_annotations_gt.tif')

utracks = Tracks()
utracks.load_utrack3d_json(
    path_to_json=f'{path_to_tracks}/tracks.json',
    scale=scale
)

centers = [np.array([prop.centroid for prop in regionprops(lab)])*scale[1:] for lab in labels]

utracks.compute_velocities()

velocities = [
    np.linalg.norm(utracks[track_index].velocity_vectors,axis=1) for track_index in utracks.indices
]

dists = []

for cs in centers:
    tree = KDTree(cs)
    nn_dists, _ = tree.query(cs ,k=2)

    dists.append(nn_dists[:,1])

test_velocities = [elem for vel in velocities for elem in vel]
test_dists = [elem for dist in dists for elem in dist]

fig = plt.figure(figsize=(6,2.2))
ax = fig.add_subplot(121)
ax.set_xlabel('displacement (um)', fontsize=12)
ax.set_ylabel('PDF', fontsize=12)
ax.set_ylim(0,2.2)
ax.get_yaxis().set_ticks([])
ax.hist(test_velocities, bins=50, density=True)
ax.plot([np.mean(test_velocities)]*100,np.linspace(0,2.2,100),'k', label=f'{np.mean(test_velocities):.2f}')
labelLines(ax.get_lines(), zorder=2.5,yoffsets=1/4, fontsize=16,align=False)
print(np.mean(test_velocities), np.std(test_velocities), np.median(test_velocities))

ax2 = fig.add_subplot(122)
ax2.set_xlabel('nearest neighbor distance (um)', fontsize=12)
ax2.set_ylabel('PDF', fontsize=12)
ax2.get_yaxis().set_ticks([])
ax2.set_ylim(0,0.21)
ax2.hist(test_dists, bins=50, density=True)
ax2.plot([np.mean(test_dists)]*100,np.linspace(0,0.21,100),'k', label=f'{np.mean(test_dists):.1f}')
labelLines(ax2.get_lines(), zorder=2.5,yoffsets=0.13*1/1.4/4, fontsize=16,align=False)
print(np.mean(test_dists), np.std(test_dists), np.median(test_dists))
fig.tight_layout()
plt.show()


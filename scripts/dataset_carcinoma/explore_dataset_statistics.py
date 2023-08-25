import tifffile
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from dwight.tracking_classes import Tracks

"""
Loads the ground truth labels, extracts the centroid coordinates at each frame, 
computes the distributions of the nearest neighbor distances and of the frame 
to frame displacements, and plots the results.
"""

path_to_root = ...
path_to_dataset = f'{path_to_root}/dataset_carcinoma/data'

scale = np.array([5,1,1])
scale4d = np.array([1, *scale])

for data_index in [1,2]:

    data = tifffile.imread(f'{path_to_dataset}/data_{data_index}.tif')
    print('shape', data.shape)
    labels_gt = tifffile.imread(f'{path_to_dataset}/labels_{data_index}.tif')

    dists = []
    sizes = []
    for label in labels_gt:
        props = regionprops(label)

        centroids = [prop.centroid*scale for prop in props]
        diams = [2 * np.power(scale[0]*  3*prop.area/(4*np.pi),1/3) for prop in props]

        tree = KDTree(centroids)
        nn_dists, _ = tree.query(centroids ,k=2)

        dists = dists + [elem for elem in nn_dists[:,1]]
        sizes = sizes+diams

    plt.figure()
    plt.hist(dists, bins=20)
    print('mean dists', np.mean(dists), np.median(dists))
    plt.xlabel("nn distance")

    plt.figure()
    plt.hist(sizes, bins=20)
    print('mean sizes', np.mean(sizes), np.median(sizes))
    plt.xlabel("typical diameter")


    tracks = Tracks().load_coherent_labels(labels_gt, scale=scale4d)
    tracks_list = [track for track in tracks if track.number_of_frames > 1]
    subtracks = Tracks()
    subtracks.load_tracks_list(tracks_list)

    subtracks.compute_velocities()

    velocities = np.array([
        np.linalg.norm(subtracks[track_index].velocity_vectors,axis=1) for track_index in subtracks.indices
    ])

    velocities = [elem for vel in velocities for elem in vel]

    plt.figure()
    plt.hist(velocities, bins=40)
    print('mean velocities', np.mean(velocities), np.median(velocities))
    plt.xlabel("displacement")



plt.show()
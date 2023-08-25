from scipy.spatial import KDTree
from dwight.utils import load_csv_coords
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines
import os 

"""
Loads the coordinates of the simulation at each frame, computes the distributions
of the nearest neighbor distances and of the frame to frame displacements, and
plots the results.
"""

path_to_root = ...

path_to_coords = f'{path_to_root}/simulations/gt'


files = os.listdir(path_to_coords)

centers = [load_csv_coords(f'{path_to_coords}/{file}') for file in files]

all_nn_dists = []
for cs in centers[:]:
    # use KDTree to compute the nearest neighbor distances
    tree = KDTree(cs)
    nn_dists, _ = tree.query(cs, k=2)
    all_nn_dists = all_nn_dists + nn_dists[:,1].tolist()


# compute mean nearest neighbor distance
d0 = np.mean(all_nn_dists)

# normalize positions
centers = np.array(centers) / d0
all_nn_dists = np.array(all_nn_dists) / d0

velocities = np.diff(centers,axis=0)
velocities = np.linalg.norm(velocities, axis=2)
all_velocities = [elem for vel in velocities for elem in vel]





fig = plt.figure(figsize=(6,2.2))
ax = fig.add_subplot(121)
ax.set_xlabel('displacement ('+r'$d_0^{-1}$'+')', fontsize=12)
ax.set_ylabel('PDF', fontsize=12)
ax.set_xlim(-0.5/d0,5/d0)
ax.set_ylim(0,1.1*d0)
ax.get_yaxis().set_ticks([])
ax.hist(all_velocities, bins=200, density=True)
ax.plot([np.mean(all_velocities)]*100,np.linspace(0,1.1*d0,100),'k', label=f'{np.mean(all_velocities):.2f}'+r'$d_0$')
labelLines(ax.get_lines(), zorder=2.5,yoffsets=1.1/3.5*d0, fontsize=16, align=False)
print(np.mean(all_velocities), np.std(all_velocities))

ax2 = fig.add_subplot(122)
ax2.set_xlabel('nearest neighbor distance ('+r'$d_0^{-1}$'+')', fontsize=12)
ax2.set_ylabel('PDF', fontsize=12)
ax2.get_yaxis().set_ticks([])
ax2.set_ylim(0,0.4*d0)
ax2.hist(all_nn_dists, bins=50, density=True)
ax2.plot([np.mean(all_nn_dists)]*100,np.linspace(0,0.4*d0,100),'k', label=r'$d_0$')
labelLines(ax2.get_lines(), zorder=2.5,yoffsets=0.4/3.5*d0, fontsize=16, align=False)
print(np.mean(all_nn_dists), np.std(all_nn_dists))

fig.tight_layout()
plt.show()


import numpy as np
import dwight.utils as simulator_utils
from scipy.spatial import KDTree as scipy_KDTree


class SimulationCorrupter:
    """
    Provides methods to add errors to a collection of points.
    Functions return the corrupted coordinates and a mapping dictionary.
    
    The mapping dictionary is a dictionary of the form:
        {new index in the returned coordinates: old index in the input coordinates, ...}
    If a point is removed, the old index is not present in the mapping dictionary.
    If a point is added, the new index is replaced by a string indicating the type of error.
    
    Their arguments are all of the form:
        'coords': np.array of shape (N, d)
        'rate': a float value between 0 and 1
        'return_dict': a boolean value indicating whether to return the mapping dictionary
    """

    def add_fp_to_coords(self, coords, fp_rate: float, return_dict: bool = False):
        coords = np.array(coords)
        N_part, d = coords.shape
        
        N_FP = int(N_part * fp_rate)

        # generate random points in a sphere around the average position
        average_pos = np.mean(coords, axis=0)
        typical_radius = 3/4 * np.max(np.linalg.norm(coords-average_pos, axis=1))
        radiuses = typical_radius * np.power(np.random.uniform(0,1,size=(N_FP,1)),1/d)
        fp_coords = average_pos + radiuses * simulator_utils.random_unit_vectors(N_FP, dim=d)

        if return_dict:
            old_mapping_dict = {ind: ind for ind in range(N_part)}
            new_mapping_dict = {ind: 'fp' for ind in range(N_part, N_part+N_FP)}
            mapping_dict = {**old_mapping_dict,**new_mapping_dict}

            return np.vstack([coords, fp_coords]), mapping_dict

        else:
            return np.vstack([coords, fp_coords])
        

    def remove_fn_from_coords(self, coords, fn_rate: float, return_dict: bool = False):
        coords = np.array(coords)
        N_part, _ = coords.shape
        
        N_FN = int(N_part * fn_rate)

        # randomly select N_part-N_FN particles to keep
        conserved_inds = np.sort(np.random.choice(
            np.arange(N_part),
            size=N_part-N_FN,
            replace=False
        ))

        if return_dict:
            mapping_dict = {new_ind: old_ind for new_ind, old_ind in enumerate(conserved_inds)}

            return coords[conserved_inds], mapping_dict
        else:
            return coords[conserved_inds]
        

    def add_merge_to_coords(self, coords, merge_rate: float, max_distance: float, return_dict: bool = False):
        """
        'max_distance' is a float value indicating the maximum distance between two particles that can be merged.
        """
        coords = np.array(coords)
        N_part, _ = coords.shape
        
        N_merge = int(N_part * merge_rate)

        # use a KDTree to find nearest neighbors (neighbors closer than max_distance)
        tree = scipy_KDTree(coords)
        dist_matrix = tree.sparse_distance_matrix(
                                tree,
                                max_distance=max_distance,
                                output_type='coo_matrix'
                            )

        # all possible pairs of particles that can be merged
        unique_pairs_inds = [
            (i,j) for i,j in zip(dist_matrix.row, dist_matrix.col) if i>j 
        ]


        new_coords = []
        paired_inds = []

        all_coords=[]
        mapping_dict = {}

        for ind_merge in range(N_merge):

            if len(unique_pairs_inds)==0:
                print(f'cannot find pair\ntotal merge: {int(len(paired_inds)/2)}')
                break

            choice_inds = np.random.choice(np.arange(len(unique_pairs_inds)))
            row_ind, col_ind = unique_pairs_inds[choice_inds]
            paired_inds = paired_inds + [row_ind, col_ind]
            # rebuild pairs to prevent particles to be used in two different merges
            unique_pairs_inds = [(i,j) for i,j in unique_pairs_inds \
                if i!=row_ind and j!=col_ind and j!=row_ind and i!=col_ind]

            # place new particle at the center of the two merged particles
            new_coord = (coords[row_ind] + coords[col_ind])/2
            new_coords.append(new_coord)

            mapping_dict[ind_merge] = f'merge_{row_ind}_to_{col_ind}'

        all_coords = new_coords.copy()
        
        # extract the indices of the particles that were not merged
        untouched_indices = np.arange(N_part)[~np.isin(np.arange(N_part), np.unique(paired_inds))]

        all_coords = all_coords + [elem for elem in coords[untouched_indices]]
        all_coords = np.array(all_coords)
        
        for new_ind, old_ind in zip(range(N_merge, N_part-N_merge), untouched_indices):
            mapping_dict[new_ind] = old_ind


        if return_dict:
            return all_coords, mapping_dict
        else:
            return all_coords
        

    def add_split_to_coords(self, coords, split_rate: float, nuclei_sizes = None, return_dict: bool = False):
        """
        Parameter 'nuclei_size' can be given as a float or as an array.
        """

        coords = np.array(coords)
        N_part, d = coords.shape
        
        N_split = int(N_part * split_rate)

        split_inds = np.random.choice(np.arange(N_part), N_split, replace=False)
        
        tree = scipy_KDTree(coords)
        split_coords = coords[split_inds]
        nearest_neighbors_dists, nearest_neighbors_inds = tree.query(split_coords, k=2)
    
        nearest_neighbors_dists = nearest_neighbors_dists[:,1]
        nearest_neighbors_inds = nearest_neighbors_inds[:,1]


        if nuclei_sizes is None:
            split_dists = nearest_neighbors_dists / 4 # approximately half of nuclei size
        else:
            if isinstance(nuclei_sizes, int):
                nuclei_sizes = nuclei_sizes * np.ones(shape=(N_part),dtype=float)
            elif isinstance(nuclei_sizes, np.ndarray):
                if nuclei_sizes.ndim == 2:
                    nuclei_sizes = nuclei_sizes[:,0] 
            
            # choose the distance at which the new particles will be placed
            # from the split particle 
            split_dists = np.minimum(nearest_neighbors_dists, nuclei_sizes[split_inds]/2)

        # random unit vectors that give the axis on which the new particles will be placed
        split_polarities = simulator_utils.random_unit_vectors(N_split, dim=d)

        all_coords = coords.copy()

        # new coords
        new_coords              = split_coords + split_dists[:,None] * split_polarities
        # old coords
        all_coords[split_inds]  = split_coords - split_dists[:,None] * split_polarities

        if return_dict:
            
            # untouched inds
            untouched_indices = np.arange(N_part)[~np.isin(np.arange(N_part), split_inds)]
            untouched_dict = {ind:ind for ind in untouched_indices}

            # displaced inds
            modified_dict = {}
            for ind_new, ind_displaced in enumerate(split_inds, start=N_part):
                modified_dict[ind_displaced] = f'split_{ind_displaced}_{ind_new}'
                modified_dict[ind_new] = f'split_{ind_displaced}_{ind_new}'

            mapping_dict = {**untouched_dict, **modified_dict}

            return np.vstack([all_coords, new_coords]), mapping_dict
        else:
            return np.vstack([all_coords, new_coords])
             
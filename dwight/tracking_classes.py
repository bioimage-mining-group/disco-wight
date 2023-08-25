import numpy as np
import json
from skimage.measure import regionprops as skimage_regionprops
from scipy.spatial import KDTree as scipy_KDTree
import tqdm
from lapsolver import solve_dense as lap_solve_dense
import copy







class Track:
    def __init__(
        self,
        index: int = None,
        tinds: list = None,
        zinds: list = None,
        yinds: list = None,
        xinds: list = None,
        tcoords: list = None,
        zcoords: list = None,
        ycoords: list = None,
        xcoords: list = None,
        xvelocities: list = None,
        yvelocities: list = None,
        zvelocities: list = None,
        nodes_trackabilities: list = None,
        edges_trackabilities: list = None,
        lifetime: float = None,
        scale: list = [1,1,1,1]
    ):

        self.index = index
        
        self.tcoords = tcoords # in real world coordinates
        self.xcoords = xcoords # in real world coordinates
        self.ycoords = ycoords # in real world coordinates
        self.zcoords = zcoords # in real world coordinates
        
        self.tinds = tinds # in frame
        self.xinds = xinds # in voxel
        self.yinds = yinds # in voxel
        self.zinds = zinds # in voxel
        
        self.xvelocities = xvelocities # in real world coordinates
        self.yvelocities = yvelocities # in real world coordinates
        self.zvelocities = zvelocities # in real world coordinates

        self.nodes_trackabilities = nodes_trackabilities
        self.edges_trackabilities = edges_trackabilities
         
        if lifetime is None and not (tcoords is None):
            self.lifetime = max(tcoords) - min(tcoords)
        else:
            self.lifetime = lifetime 

        self.scale = scale

    def from_array(self, array, scale: list = [1,1,1,1]):
        """
        A priori, the coordinates in array should be in array 
        coordinates (i.e be integers).
        """

        self.index = array[0,0]

        self.tinds = array[:, self.attribute_to_index('tinds')]
        self.zinds = array[:, self.attribute_to_index('zinds')]
        self.yinds = array[:, self.attribute_to_index('yinds')]
        self.xinds = array[:, self.attribute_to_index('xinds')]

        self.tcoords = self.tinds * scale[0]
        self.zcoords = self.zinds * scale[1]
        self.ycoords = self.yinds * scale[2]
        self.xcoords = self.xinds * scale[3]

        self.lifetime = max(self.tcoords) - min(self.tcoords)

        self.scale = scale


    def compute_velocities(self):

        self.xvelocities = np.gradient(self.xcoords, self.tcoords)
        self.yvelocities = np.gradient(self.ycoords, self.tcoords) 
        self.zvelocities = np.gradient(self.zcoords, self.tcoords)

    def get_average_velocity(self, along_dim: str = None, average_norm: bool = True):

        if along_dim == None:
            velocity_vecs = np.array([self.xvelocities, self.yvelocities, self.zvelocities]).T
        elif along_dim == 'x':
            velocity_vecs = np.array([self.xvelocities]).T
        elif along_dim == 'y':
            velocity_vecs = np.array([self.yvelocities]).T
        elif along_dim == 'z':
            velocity_vecs = np.array([self.zvelocities]).T

        if average_norm is True:

            velocity_norms = np.linalg.norm(velocity_vecs, axis=1)

            return np.mean(velocity_norms)

        else:
            return np.mean(velocity_vecs, axis=0)

    def attribute_to_index(self, attribute_name: str) -> int:
        return self.attribute_to_index_dict[attribute_name]
        
        
    @property
    def number_of_frames(self):
        return len(self.tinds)
    
    @property
    def velocity_vectors(self):
        return np.array(
            [self.zvelocities, self.yvelocities, self.xvelocities]
        ).T

    @property
    def attribute_to_index_dict(self):
        """
        Tracks informations can be dumped to an array via
        a method of their class. The array has dimensions
        (n_timepoints, n_infos). This dict gives the index
        of each info in the array.
        """
        return {
            'index':   0,
            'tinds':   1,
            'zinds':   2,
            'yinds':   3,
            'xinds':   4,
            'zcoords': 5,
            'ycoords': 6,
            'xcoords': 7
        }

    @property
    def coords(self):
        return np.array([self.xcoords, self.ycoords, self.zcoords]).T



class Tracks():#metaclass=MetaTracks):
    def __init__(self):
        self.tracks = dict()
        

    def empty(self): 
        self.tracks = dict()

    def load_napari_tracks(self, napari_tracks, scale: tuple = [1,1,1,1], formatting='coords'):

        napari_tracks = np.array(napari_tracks)

        if napari_tracks.size == 0:
            return self

        napari_tracks_indices = napari_tracks[:,0]
        
        tracks_indices = np.unique(napari_tracks_indices)

        if formatting == 'coords':

            for track_index in tracks_indices:

                track_data = napari_tracks[napari_tracks_indices == track_index]

                tcoords=track_data[:,1]
                zcoords=track_data[:,2]
                ycoords=track_data[:,3]
                xcoords=track_data[:,4]
                
                # TODO: handle None in coords ?
                if     None in tcoords \
                    or None in zcoords \
                    or None in ycoords \
                    or None in xcoords:
                        
                    raise TypeError

                track = Track(
                    index=int(track_index),
                    tcoords=tcoords,
                    zcoords=zcoords,
                    ycoords=ycoords,
                    xcoords=xcoords,
                    tinds=(tcoords / scale[0]).astype(int),
                    zinds=(zcoords / scale[1]).astype(int),
                    yinds=(ycoords / scale[2]).astype(int),
                    xinds=(xcoords / scale[3]).astype(int),
                    scale=scale
                )

                self.tracks[track.index] = track
        
        elif formatting == 'inds':

            for track_index in tracks_indices:

                track_data = napari_tracks[napari_tracks_indices == track_index]

                tinds=track_data[:,1].astype(int)
                zinds=track_data[:,2].astype(int)
                yinds=track_data[:,3].astype(int)
                xinds=track_data[:,4].astype(int)
                
                # TODO: handle None in coords ?
                if     None in tinds \
                    or None in zinds \
                    or None in yinds \
                    or None in xinds:
                        
                    raise TypeError

                track = Track(
                    index=int(track_index),
                    tinds=tinds,
                    zinds=zinds,
                    yinds=yinds,
                    xinds=xinds,
                    tcoords=(tinds * scale[0]),
                    zcoords=(zinds * scale[1]),
                    ycoords=(yinds * scale[2]),
                    xcoords=(xinds * scale[3]),
                    scale=scale
                )

                self.tracks[track.index] = track
        
        else:
            raise NotImplementedError
        
        return self

    def load_utrack3d_json(
            self,
            path_to_json: str,
            path_to_trackability_json: str = None,
            scale: list = [1,1,1,1],
            utrack_rescale: list = None,
            debug_trackability: bool = False
        ):

        if utrack_rescale is None:
            utrack_rescale = [1, scale[2]/scale[1], 1, 1]

        if debug_trackability:
            tracks_lengths = []
            trackability_lengths = []

        with open(path_to_json) as json_file:
            track_objects = json.load(json_file)

            if not isinstance(track_objects, list):
                track_objects = [track_objects]

            for track_object in track_objects:
                
                # Utrack3D gives result in this weird format where 
                # the anisotropy is handled such that X/Y are scaled 
                # to 1. So technically it outputs in a form of 
                # world-coordinate-way, but I started by extracting
                # the array coordinates first for some reason...
                tinds=np.array(track_object['t']) * utrack_rescale[0]
                zinds=np.array(track_object['z']) * utrack_rescale[1]
                yinds=np.array(track_object['y']) * utrack_rescale[2]
                xinds=np.array(track_object['x']) * utrack_rescale[3]
                
                # TODO: handle None in coords ?
                if     None in tinds \
                    or None in zinds \
                    or None in yinds \
                    or None in xinds:
                        
                    raise TypeError

                track = Track(
                    index=int(track_object['index']),
                    tcoords=tinds * scale[0],
                    zcoords=zinds * scale[1],
                    ycoords=yinds * scale[2],
                    xcoords=xinds * scale[3],
                    tinds=tinds.astype(int),
                    zinds=zinds.astype(int),
                    yinds=yinds.astype(int),
                    xinds=xinds.astype(int),
                    scale=scale
                )

                if debug_trackability:
                    tracks_lengths.append(len(track.tinds))

                track.label_inds = track_object['A']

                self.tracks[track.index] = track
                
        if path_to_trackability_json != None:
            
            with open(path_to_trackability_json) as json_file:
                trackability_data = json.load(json_file)
                
                # retrocompatibility
                if isinstance(trackability_data, dict) and "segTrackability" in trackability_data.keys():
                    trackability_list = trackability_data["segTrackability"]
                elif isinstance(trackability_data, list):
                    trackability_list = trackability_data
                else:
                    raise ValueError

                trackability_property = []
                
                for ind_track, trackability_of_track in enumerate(trackability_list, start=1):

                    track = self.tracks[ind_track]

                    if isinstance(trackability_of_track, list):

                        track.edges_trackabilities = trackability_of_track 

                        # trackability_of_track = (np.array( [trackability_of_track[0]] + trackability_of_track  ) \
                        #                       +  np.array( trackability_of_track + [trackability_of_track[-1]] )) / 2

                        trackability_of_track = np.array([
                            [trackability_of_track[0]] + trackability_of_track,
                            trackability_of_track + [trackability_of_track[-1]]
                        ]) 

                        trackability_of_track = np.min(trackability_of_track, axis=0)
 
                        track.nodes_trackabilities = trackability_of_track 

                        for elem in trackability_of_track: 
                            trackability_property.append(elem)

                        if debug_trackability:
                            trackability_lengths.append(len(trackability_of_track))
                    else:

                        track.edges_trackabilities = [trackability_of_track] 

                        trackability_property.append(trackability_of_track)
                        trackability_property.append(trackability_of_track)

                        track.nodes_trackabilities = [trackability_of_track] * 2

                        if debug_trackability:
                            trackability_lengths.append(2)

                    self.tracks[ind_track] = track

                    # trackability_property.append(1)
                    # if isinstance(trackability_of_track, list):
                    #     for elem in trackability_of_track: 
                    #         trackability_property.append(elem)
                    # else:
                    #     trackability_property.append(trackability_of_track)
                        
            if debug_trackability:
                return tracks_lengths, trackability_lengths

            track_indices = [track.index for track in self for _ in range(track.number_of_frames)]
            trackability_property = np.array(trackability_property)[np.argsort(track_indices)]

            return trackability_property

    def load_coherent_labels(self, labels, scale: tuple = [1,1,1,1]):

        tracks_as_array = []

        for ind_t, labels_t in enumerate(labels):

            props = skimage_regionprops(labels_t)

            for prop in props:
                tracks_as_array.append([
                    prop.label, 
                    ind_t*scale[0],
                    *([c*s for c,s in zip(prop.centroid, scale)])
                ])
        
        # Sort by time
        tracks_as_array = sorted(tracks_as_array, key=lambda l: l[1])
        # Then ID
        tracks_as_array = sorted(tracks_as_array, key=lambda l: l[0])

        return self.load_napari_tracks(tracks_as_array, scale=scale, formatting='coords')

    def load_tracks_list(self, list_of_tracks):

        for track in list_of_tracks:
            self.tracks[track.index] = track

    def compute_velocities(self):
        
        for track in self.tracks.values():
            track.compute_velocities()   
  
    def coherent_segmentation_from_tracks(self, raw_segmented_labels, 
                                                 add_false_negatives: bool = True):
        """
        DONE: add anisotropy !
        DONE: disregard duplicate associations due to track FPs

        TODO: keep leftover blobs (isolated in time because not part of
        a track) in output array !
        """
        
        # future result array
        coherent_labels = np.zeros(raw_segmented_labels.shape, dtype=int)

        # represent tracks as Napari list
        tracks_as_list = np.array(self.dump_tracks_to_napari())

        scale4d = self.tracks[self.indices[0]].scale

        # max_physical_dist = jutils.max_physical_dist_from_array(
        #     array_shape=raw_segmented_labels.shape, 
        #     scale=scale4d[:1]
        # )

        max_track_id = np.max(self.indices)

        for tind in tqdm(range(len(coherent_labels)), desc='Making segmentation coherent'):
            

            ### COORDS FROM THE TRACK HEADS AT THIS T
            tracks_at_t = tracks_as_list[tracks_as_list[:,1]==tind]
            tracks_at_t_ids = tracks_at_t[:,0].astype(int)
            tracks_at_t_coords = [
                    [coord*s for coord, s in zip(coords, scale4d[1:])]
                for coords in tracks_at_t[:,[2,3,4]]
            ] # accounting for anisotropy
            ###
            
            ### COORDS FROM THE BLOBS AT THIS T
            labels_at_t = np.unique(raw_segmented_labels[tind])[1:] # discarding BG
            labels_at_t_properties = skimage_regionprops(raw_segmented_labels[tind])
            labels_centroid_coords = [
                    [coord*s for coord, s in zip(coords, scale4d[1:])]
                for coords in [prop.centroid for prop in labels_at_t_properties]
            ] # accounting for anisotropy
            ###

            # build trees
            raw_labels_at_t_tree = scipy_KDTree(labels_centroid_coords)
            tracks_at_t_tree = scipy_KDTree(tracks_at_t_coords)

            # maximal admitted distance
            max_dist = 3 # NEED TO FIND A MORE SYSTEMATIC WAY :(((

            ### COMPUTE DISTANCE MATRIX AS A COST MATRIX
            sparse_dist_matrix = raw_labels_at_t_tree.sparse_distance_matrix(
                other=tracks_at_t_tree,
                max_distance=max_dist
            ) # assignments above 'max_distance' have cost set to 0
            distance_matrix = sparse_dist_matrix.toarray()
            distance_matrix[distance_matrix == 0] = np.nan
            ###

            # make assignments and calculate costs
            #labels_inds, tracks_inds = scipy_linear_sum_assignment(distance_matrix)
            labels_inds, tracks_inds = lap_solve_dense(distance_matrix)
            costs = distance_matrix[labels_inds, tracks_inds]

            matched_labels_inds = labels_inds[costs <= max_dist] 
            matched_tracks_inds = tracks_inds[costs <= max_dist]

            matched_labels = [labels_at_t[ind] for ind in matched_labels_inds]

            # unmatched_labels_inds = labels_inds[costs > max_dist] 
            # unmatched_tracks_inds = tracks_ind[costs > max_dist]

            unmatched_labels = labels_at_t[
                ~ np.isin(
                    labels_at_t,
                    labels_at_t[matched_labels_inds])
            ]


            # 'tracks_at_t_ids' and 'tracks_at_t_coords', the latter of
            # which was plugged into the tree query, are ordered in the
            # same way (because they both originate from 'tracks_as_list')

            # also, 'labels_at_t' is obtained via 'np.unique', so it is ordered,
            # and 'labels_centroid_coords' is obtained via skimage.regionprops, 
            # which is also ordered
            #TODO : rewrite 'coherentation' loop by looping on blobs

            raw_segmented_labels_at_t = raw_segmented_labels[tind]
            ndim = raw_segmented_labels_at_t.ndim
            props = skimage_regionprops(raw_segmented_labels_at_t)

            for label_index, label in enumerate(labels_at_t):

                prop = props[label_index]

                roi_slices = prop.slice

                segmentation_roi = raw_segmented_labels_at_t[roi_slices]
                coherent_roi = coherent_labels[(tind,)+roi_slices]

                ### THIS DEALS WITH TRUE POSITIVES
                if label in matched_labels:
 
                    label_index_in_matched_labels_ind = matched_labels_inds.tolist().index(label_index)
                    matched_track_ind = matched_tracks_inds[label_index_in_matched_labels_ind]
                    closest_track_head_id = tracks_at_t_ids[matched_track_ind]

                    coherent_roi[segmentation_roi == label] = closest_track_head_id
                    coherent_labels[(tind,)+roi_slices] = coherent_roi

                ### THIS DEALS WITH FALSE NEGATIVES
                else:
                    if add_false_negatives:

                        coherent_roi[segmentation_roi == label] = max_track_id + 1
                        coherent_labels[(tind,)+roi_slices] = coherent_roi
                        max_track_id += 1


        return coherent_labels

    def dump_tracks_to_array(self):

        output_list = []

        for track in self.tracks.values():
            
            for tind, zind, xind, yind, tcoords, zcoords, ycoords, xcoords \
                in zip(track.tinds,track.zinds, track.yinds, track.xinds, 
                       track.tcoords, track.zcoords, track.ycoords, 
                       track.xcoords):
            
                output_list.append(
                    [track.index, tind, zind, xind, yind, tcoords, zcoords, ycoords, xcoords]
                )
                
        return np.array(output_list)
    
                
    def dump_tracks_to_napari(self, return_inds: bool = True, return_graph: bool = False):
        """Quoting the napari doc:
        "The data in the array must be sorted by increasing track_id then time."
        """
        napari_list = []
        
        for track in self.tracks.values():

            if return_inds:
            
                for tind, zind, xind, yind in zip(track.tinds,track.zinds, track.yinds, track.xinds):
                
                    napari_list.append(
                        [track.index, tind, zind, xind, yind]
                    )
            
            else:
                for tcoord, zcoord, xcoord, ycoord in zip(track.tcoords,track.zcoords, track.ycoords, track.xcoords):
            
                    napari_list.append(
                        [track.index, tcoord, zcoord, xcoord, ycoord]
                    )
        # # Start by sorting by track index...
        napari_list = sorted(napari_list, key=lambda l: l[0])
        # # Then sort by time
        # #napari_list = sorted(napari_list, key=lambda l: l[1])
                
        return np.array(napari_list)
    
    def dump_to_XML(self, path_to_xml: str):
        """
        <?xml version="1.0" encoding="UTF-8" standalone="no"?>
        <root>
        <TrackContestISBI2012 snr="2" density="low" scenario="virus"> <!-- Indicates the data set -->
        <particle> <!-- Definition of a particle track -->
        <detection t="4" x="14" y="265" z="5.1"> <!-- Definition of a track point -->
        <detection t="5" x="14.156" y="266.5" z="4.9">
        <detection t="6" x="15.32" y="270.1" z="5.05">
        </particle>
        <particle> <!-- Definition of another particle track -->
        <detection t="14" x="210.14" y="12.5" z="1">
        <detection t="15" x="210.09" y="13.458" z="1.05">
        <detection t="16" x="210.19" y="14.159" z="1.122">
        </particle>
        </TrackContestISBI2012>
        </root>
        """

        with open(f'{path_to_xml}', 'w') as file:

            file.write(
                '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            )
            file.write('<root>\n')
            file.write('<TrackContestISBI2012 snr="2" density="low" scenario="virus">\n')

            for track in self:
                file.write('<particle>\n')

                for tind, zc, yc, xc in zip(track.tinds, track.zcoords, track.ycoords, track.xcoords):
                    file.write(f'<detection t="{tind}" x="{xc}" y="{yc}" z="{zc}"></detection>\n')

                file.write('</particle>\n')

            file.write('</TrackContestISBI2012>\n')
            file.write('</root>')

    def get_subset_of_tracks(self, selected_track_indices: list = None, filters_dict: dict = None,
                             verbose: bool = True):
        """
        The filtering is of type filter.low <= value < filter.high. Note the '<'.

        /!\ Right now, the filtering is only implemented for {t,z,y,x}{inds,coords} 
        filtering. Should make a 'dump_all_attributes_to_array' methods to
        do the filtering on.

        The filtering dictionary is flexible enough that it lets the user
        define several constraints on the same attribute, and interprets it
        as an intersection (AND) of constraints.  
        """
        
        if selected_track_indices is None:
            selected_track_indices = self.indices
        
        sub_tracks = Tracks()
        
        for selected_index in selected_track_indices:

            filtered_track = self.__filter_track(
                filters_dict, self.tracks[selected_index]
            )
            
            if filtered_track is None:
                if verbose is True:
                    print(f'Fully discarded Track {selected_index} while filtering')
                else:
                    pass
            else:
                sub_tracks.tracks[selected_index] = filtered_track
            
        return sub_tracks

    def __filter_track(self, filters_dict: dict, track: Track) -> Track:
        """
        Returns None if the Track is entirely filtered out.
        """


        if filters_dict is None:
            return track
        
        else:

            # create dummy Tracks object to be able to dump to list
            dummy_tracks = Tracks()
            dummy_tracks.tracks[track.index] = track
            track_as_array = dummy_tracks.dump_tracks_to_array()

            for attribute_name, (attr_min, attr_max) in filters_dict.items():

                attr_index = track.attribute_to_index(attribute_name)

                track_as_array = track_as_array[
                    self.__filter_array_min_max(
                        array=track_as_array[:, attr_index],
                        min_val=attr_min,
                        max_val=attr_max
                    )
                ]

            if len(track_as_array) == 0:
                return None
            else:
                filtered_track = Track()
                filtered_track.from_array(track_as_array, scale=track.scale)
                filtered_track.nodes_trackabilities = track.nodes_trackabilities

                return filtered_track

    def __filter_array_min_max(array, min_val: float, max_val: float):

        return np.logical_and(
            array >= min_val,
            array < max_val
        )

    def __getitem__(self, index: int):
        return self.tracks[index]

    def __iter__(self):
        return (track for track in self.tracks.values()) 

    def __len__(self):
        return len(self.tracks)

    @property
    def copy(self):
        return copy.deepcopy(self)

    @property
    def indices(self):
        return list(self.tracks.keys())

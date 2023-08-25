
import numpy as np

from skimage.measure import regionprops as skimage_regionprops
from scipy.spatial import KDTree as scipy_KDTree

import numba
numba.set_num_threads(2)

from lapsolver import solve_dense as lap_solve_dense


def compare_segmentations(gt_segmentation, pred_segmentation, iou_threshold: float = 0, return_inds: bool = False):
    """
    Segmentation are assumed to be 3D (or 2D !).

    By default, the matched/unmatched labels are directly returned.
    'return_inds' allows one to get the ordered continuous indices instead.  
    """

    # iou_matrix = build_iou_matrix(gt_segmentation, pred_segmentation)
    iou_matrix = _intersection_over_union(gt_segmentation, pred_segmentation)[1:,1:]

    # all possible gt indices
    indices_gt = np.arange(iou_matrix.shape[0])
    # all possible prediction indices
    indices_detec = np.arange(iou_matrix.shape[1])
    
    
    ### SOLVE LAP
    # make assignments and calculate costs
    iou_matrix[iou_matrix<=iou_threshold] = np.nan

    tp_gt_inds, tp_pred_inds = lap_solve_dense(iou_matrix)

    fn_gt_inds = [elem for elem in indices_gt if not (elem in tp_gt_inds)]
    fp_pred_inds = [elem for elem in indices_detec if not (elem in tp_pred_inds)]
    ###

    ###
    if len(tp_gt_inds) == 0:
        ious = []
        tp_gt_inds = []
        tp_pred_inds = []
    else:
        ious = iou_matrix[tp_gt_inds, tp_pred_inds]
    ###

    if return_inds:
        return (
            tp_gt_inds,
            tp_pred_inds,
            fp_pred_inds,
            fn_gt_inds
        ), ious
    
    else:
        gt_labels = np.unique(gt_segmentation)[1:]
        pred_labels = np.unique(pred_segmentation)[1:]

        return (
            gt_labels[tp_gt_inds],
            pred_labels[tp_pred_inds],
            pred_labels[fp_pred_inds],
            gt_labels[fn_gt_inds]
        ), ious


@numba.jit(nopython=True)
def _relabel_1d_array(array):
    """
    Relabels a 1D array of labels to be consecutive integers starting at 0.    
    """
    uniques = np.unique(array)
    new_array = np.zeros_like(array)

    for i in range(len(array)):
        for j in range(len(uniques)):
            if array[i] == uniques[j]:
                new_array[i] = j
                break

    return new_array
        

@numba.jit(nopython=True)
def _label_overlap(x, y):
    """ 
    Computes the overlap matrix between two label arrays.
    """
    # put label arrays into standard form then flatten them
    x = _relabel_1d_array(x.ravel())
    y = _relabel_1d_array(y.ravel())
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _intersection_over_union(gt_segmentation, pred_segmentation):
    """
    Computes intersection over union of all labels from the ground
    truth segmentation 'gt_segmentation' and the predicted segmentation
    'pred_segmentation'.
    """
    overlap = _label_overlap(gt_segmentation, pred_segmentation)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou
    
    

def compare_detections(gt_inds_coordinates, prediction_inds_coordinates, max_distance: float, 
                       scale: tuple = None, return_costs: bool = False):

    if scale is None:
        scale = [1] * len(gt_inds_coordinates)
    scale = np.array(scale)                      

    ### BUILD TREES FROM COORDINATES
    gt_tree = scipy_KDTree(gt_inds_coordinates * scale)
    prediction_tree = scipy_KDTree(prediction_inds_coordinates * scale)
    ###

    ### COMPUTE DISTANCE MATRIX AS A COST MATRIX
    sparse_dist_matrix = gt_tree.sparse_distance_matrix(
        other=prediction_tree,
        max_distance=max_distance
    ) # assignments above 'max_distance' have cost set to 0
    distance_matrix = sparse_dist_matrix.toarray()
    ###

    ### SOLVE LAP
    # make assignments and calculate costs
    #distance_matrix[distance_matrix == 0] = max_physical_dist
    distance_matrix[distance_matrix==0] = np.nan

    tp_gt_inds, tp_pred_inds = lap_solve_dense(distance_matrix**2)

    if len(tp_gt_inds) > 0:
        costs = distance_matrix[tp_gt_inds, tp_pred_inds]
    else:
        costs = []
    ###
    
    # all possible gt indices
    indices_gt = np.arange(distance_matrix.shape[0])
    # all possible prediction indices
    indices_detec = np.arange(distance_matrix.shape[1])

    fn_gt_inds = [elem for elem in indices_gt if not (elem in tp_gt_inds)]
    fp_pred_inds = [elem for elem in indices_detec if not (elem in tp_pred_inds)]

    if return_costs:
        return (
            tp_gt_inds,
            tp_pred_inds,
            fp_pred_inds,
            fn_gt_inds
        ), costs
    else:

        return (
            tp_gt_inds,
            tp_pred_inds,
            fp_pred_inds,
            fn_gt_inds
        )
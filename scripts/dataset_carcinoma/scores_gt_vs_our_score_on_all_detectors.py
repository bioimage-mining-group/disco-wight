import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dwight.tracking_classes import Tracks
import dwight.gt_metrics as gtm
from skimage.measure import regionprops

import tifffile

"""
This script is used to generate the following figures for the carcinoma dataset:
    - F1c and F1iou as a function of the distance and IoU thresholds
    - Trackability score histogram for each detector
"""


scale4d = (1,1,1,1)
scale3d = np.array(scale4d[1:])

path_to_root = ...

path_to_labels = f'{path_to_root}/dataset_carcinoma/data'
path_to_tracks = f'{path_to_root}/dataset_carcinoma/utrack'
path_to_save = f'{path_to_root}/dataset_carcinoma/analysis_results'
path_to_figures = f'{path_to_root}/dataset_carcinoma/figures'



all_SRs = np.arange(1,32,2);
all_F1Rs = np.arange(1,10,1)
all_F1vol = np.arange(0.5,0.95,0.05)



folder_names = [
	'gt',
    'sd',
	'cyto',
    'cyto2'
]
label_names = [
	'labels_isoscaled',
    'labels_predicted_sd_pretrained',
	'labels_predicted_cp_pretrained_cyto',
    'labels_predicted_cp_pretrained_cyto2'
]

plot_names = [
    'GT',
    'SDp',
    'CPp1',
    'CPp2',
]


colors = ['black','green', 'tomato', 'skyblue']



plot = True


if not plot:

    F1c_to_save = []
    F1c_std_to_save = []

    F1iou_to_save = []
    F1iou_std_to_save = []

    Tys_to_save = []
    Tys_std_to_save = []


    ### Make JIs
    for ind_name, (name, label_name) in tqdm(enumerate(zip(folder_names[1:], label_names[1:]))):

        F1s_of_model_per_dataset = []
        F1s_std_of_model_per_dataset = []

        F1s_iou_of_model_per_dataset = []
        F1s_iou_std_of_model_per_dataset = []

        for index_dataset in [1,2]:

            labels_gt = tifffile.imread(f'{path_to_labels}/labels_isoscaled_{index_dataset}.tif')
            coords_gt = [np.array([prop.centroid for prop in regionprops(img)]) for img in labels_gt]

            labels_pred = tifffile.imread(f'{path_to_labels}/{label_name}_{index_dataset}.tif')
            coords_pred = [np.array([prop.centroid for prop in regionprops(img)]) for img in labels_pred]

            num_frames = len(labels_gt)

            ### F1_center
            F1s_of_model = []
            F1s_std_of_model = []

            print('starting computation of F1c')

            for ind_F1c, F1c in enumerate(all_F1Rs):

                F1s_of_model_at_F1c = []

                MAX_DISTANCE_FACTOR = F1c

                for tind in range(num_frames):

                    (tp_gt_inds, tp_pred_inds, fp_pred_inds, fn_gt_inds), costs = gtm.compare_detections(
                            gt_inds_coordinates=coords_gt[tind],
                            prediction_inds_coordinates=coords_pred[tind] + 1e-3,
                            max_distance=MAX_DISTANCE_FACTOR,
                            scale=scale4d[1:],
                            return_costs=True
                        )

                    this_F1 = len(tp_gt_inds) / (len(tp_gt_inds) + (len(fp_pred_inds) + len(fn_gt_inds))/2)

                    F1s_of_model_at_F1c.append(this_F1)

                F1s_of_model.append(np.mean(F1s_of_model_at_F1c))
                F1s_std_of_model.append(np.std(F1s_of_model_at_F1c))
            
            F1s_of_model_per_dataset.append(F1s_of_model)
            F1s_std_of_model_per_dataset.append(F1s_std_of_model)

            ### F1_IOU
            F1s_iou_of_model = []
            F1s_iou_std_of_model = []

            print('starting computation of F1iou')

            for ind_F1iou, F1iou in enumerate(all_F1vol):

                F1s_iou_of_model_at_F1c = []

                MAX_VOLUME_FACTOR = F1iou

                for tind in range(num_frames):

                    (tp_gt_labels, tp_pred_labels, fp_pred_labels, fn_gt_labels), ious = gtm.compare_segmentations(
                            gt_segmentation=labels_gt[tind],
                            pred_segmentation=labels_pred[tind],
                            iou_threshold=MAX_VOLUME_FACTOR
                        )


                    this_F1 = len(tp_gt_labels) / (len(tp_gt_labels) + (len(fp_pred_labels) + len(fn_gt_labels))/2)

                    F1s_iou_of_model_at_F1c.append(this_F1)

                F1s_iou_of_model.append(np.mean(F1s_iou_of_model_at_F1c))
                F1s_iou_std_of_model.append(np.std(F1s_iou_of_model_at_F1c))
            
            print('ending computation of F1iou')

            F1s_iou_of_model_per_dataset.append(F1s_iou_of_model)
            F1s_iou_std_of_model_per_dataset.append(F1s_iou_std_of_model)

        F1c_to_save.append(F1s_of_model_per_dataset)
        F1c_std_to_save.append(F1s_std_of_model_per_dataset)
        F1iou_to_save.append(F1s_iou_of_model_per_dataset)
        F1iou_std_to_save.append(F1s_iou_std_of_model_per_dataset)

    np.savetxt(f'{path_to_save}/F1c.txt', np.mean(F1c_to_save,axis=1))
    np.savetxt(f'{path_to_save}/F1c_std.txt', np.mean(F1c_std_to_save,axis=1))
    np.savetxt(f'{path_to_save}/F1iou.txt', np.mean(F1iou_to_save,axis=1))
    np.savetxt(f'{path_to_save}/F1iou_std.txt', np.mean(F1iou_std_to_save,axis=1))
            

    ### Make Tys
    for ind_name, (name, label_name) in tqdm(enumerate(zip(folder_names, label_names))):

        Tys_of_model_per_dataset = []
        Tys_std_of_model_per_dataset = []

        for index_dataset in [1,2]:

            Ty_model = []
            Ty_std_model = []

            for ind_SR, SR in enumerate(all_SRs):

                Ty_at_SR = []

                path_to_trackability = f'{path_to_tracks}/{index_dataset}/{name}/{ind_SR}'

                utracks = Tracks()
                utracks.load_utrack3d_json(
                    path_to_json=f'{path_to_trackability}/tracks.json',
                    scale=scale4d,
                    path_to_trackability_json=f'{path_to_trackability}/trackability.json'
                )

                tracks_Tys_by_time = [[] for _ in range(num_frames)]

                for track in utracks:
                    for tind,ty in zip(track.tinds, track.nodes_trackabilities):
                            
                            tracks_Tys_by_time[tind].append(ty)
                    

                for tind in range(num_frames):
                    this_Ty = np.mean(tracks_Tys_by_time[tind])
                    Ty_at_SR.append(this_Ty)


                Ty_model.append(np.mean(Ty_at_SR))
                Ty_std_model.append(np.std(Ty_at_SR))

            Tys_of_model_per_dataset.append(Ty_model)
            Tys_std_of_model_per_dataset.append(Ty_std_model)
            
        Tys_to_save.append(Tys_of_model_per_dataset)
        Tys_std_to_save.append(Tys_std_of_model_per_dataset)
    
    np.savetxt(f'{path_to_save}/Tys.txt', np.mean(Tys_to_save,axis=1))
    np.savetxt(f'{path_to_save}/Tys_std.txt', np.mean(Tys_std_to_save,axis=1))



else:
    F1c_to_save = np.loadtxt(f'{path_to_save}/F1c.txt')
    F1c_std_to_save = np.loadtxt(f'{path_to_save}/F1c_std.txt')

    F1iou_to_save = np.loadtxt(f'{path_to_save}/F1iou.txt')
    F1iou_std_to_save = np.loadtxt(f'{path_to_save}/F1iou_std.txt')

    Tys_to_save = np.loadtxt(f'{path_to_save}/Tys.txt')
    Tys_std_to_save = np.loadtxt(f'{path_to_save}/Tys_std.txt')




    fig = plt.figure(figsize=(7,3))
        

    ax= fig.add_subplot(1,2,1)
    ax.set_xlabel('Threshold distance '+r'$\tau_c $'+' (um)', fontsize=16)
    ax.set_ylabel(r'$F1_c(\tau_c)$', fontsize=16)
    for ind,(name,c) in enumerate(zip(plot_names[1:], colors[1:])):
        ax.errorbar(all_F1Rs*1.2, F1c_to_save[ind], yerr=F1c_std_to_save[ind],c=c, label=name)
        ax.plot(all_F1Rs*1.2, F1c_to_save[ind], marker='o',c=c)
    ax.legend()

    ax= fig.add_subplot(1,2,2)
    ax.set_xlabel('Threshold IoU '+r'$\tau_{IoU} $'+' (%)', fontsize=16)
    ax.set_ylabel(r'$F1_{IoU}(\tau_{IoU})$', fontsize=16)
    for ind,(name,c) in enumerate(zip(plot_names[1:], colors[1:])):
        ax.errorbar(all_F1vol, F1iou_to_save[ind], yerr=F1iou_std_to_save[ind],c=c, label=name)
        ax.errorbar(all_F1vol, F1iou_to_save[ind], marker ='o',c=c)

    ax.set_xticks(np.linspace(0.5,0.9,5))
    ax.set_xticklabels(range(50,100,10))

    fig.tight_layout()


    fig2 = plt.figure(figsize=(3.5,3))

    ax = fig2.add_subplot(1,1,1)
    
    ax.set_ylim(0.8,1.01)
    ax.set_ylabel('our score (a.u.)', fontsize=16)

    ind_SR25 = [elem for elem in all_SRs].index(25)

    for i, (Ty, Tystd, color) in enumerate(zip(Tys_to_save[:,ind_SR25], Tys_std_to_save[:,ind_SR25], colors)):
        ax.bar(
            x=i, 
            height=Ty, 
            yerr=Tystd, 
            color=color,
            ecolor=color,
            capsize=4,
            )

    
    ax.set_xticks(range(len(folder_names)))
    ax.set_xticklabels(plot_names, fontsize=16)
    fig2.tight_layout()

    fig.savefig(f'{path_to_figures}/JI_data.svg')
    fig.savefig(f'{path_to_figures}/JI_data.pdf')
    fig.savefig(f'{path_to_figures}/JI_data.png')

    fig2.savefig(f'{path_to_figures}/Ty_data.svg')
    fig2.savefig(f'{path_to_figures}/Ty_data.pdf')
    fig2.savefig(f'{path_to_figures}/Ty_data.png')

    fig3 = plt.figure()
    ax = fig3.add_subplot(111)

    ax.errorbar(all_SRs,Tys_to_save[0],yerr=Tys_std_to_save[0], c=colors[0])
    ax.errorbar(all_SRs,Tys_to_save[1],yerr=Tys_std_to_save[1], c=colors[1])
    ax.errorbar(all_SRs,Tys_to_save[2],yerr=Tys_std_to_save[2], c=colors[2])
    ax.errorbar(all_SRs,Tys_to_save[3],yerr=Tys_std_to_save[3], c=colors[3])
         
    plt.show()   
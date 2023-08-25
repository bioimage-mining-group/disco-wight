import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dwight.tracking_classes import Tracks
import dwight.gt_metrics as jcomp
from dwight.utils import load_csv_coords
import os
from scipy.stats import spearmanr

"""
Loads corrupted simulations coordinates for FN+FP conditions, computes
the quality score, and plots along theoretical F1 score as a function 
of the error rate. 
"""



scale4d = (1,)*4
scale3d = np.array(scale4d[1:])

path_to_root = ...

path_to_coords = f'{path_to_root}/simulations/coords'
path_to_tracks = f'{path_to_root}/simulations/utrack'

path_to_figures = f'{path_to_root}/figures'



MAX_TIME = 46

d0 = 16* 0.8489


coords_gt = []
files = sorted(os.listdir(f'{path_to_coords}/coords/gt'))
    
for ind_t,file in enumerate(files):
    c_gt = load_csv_coords(
        path_to_csv=f'{path_to_coords}/coords/gt/{file}'
    )
    coords_gt.append(c_gt)

coords_gt = np.array(coords_gt)


fn_rates = np.arange(0.0, 0.5, 0.05)*100
fp_rates = np.arange(0.0, 0.5, 0.05)*100

plot = True



Ty_matrix = np.zeros((len(fp_rates), len(fn_rates)))
F1_matrix = np.zeros((len(fp_rates), len(fn_rates)))

if not plot:
    for ind_fn, fn_rate in enumerate(tqdm(fn_rates)):
        for ind_fp, fp_rate in enumerate(fp_rates):

            files = sorted(os.listdir(f'{path_to_coords}/fnfp/coords_fn{ind_fn}_fp{ind_fp}'))

            coords_all = []
            for ind_t,file in enumerate(files):
                c_all = load_csv_coords(
                        path_to_csv=f'{path_to_coords}/fnfp/coords_fn{ind_fn}_fp{ind_fp}/{file}'
                    )
                coords_all.append(c_all)

            coords_all = np.array(coords_all)

            JIs_rate = []


            MAX_DISTANCE_FACTOR = 10

            for tind in range(MAX_TIME):

                (tp_gt_inds, tp_pred_inds, fp_pred_inds, all_gt_inds), costs = jcomp.compare_detections(
                        gt_inds_coordinates=coords_gt[tind],
                        prediction_inds_coordinates=coords_all[tind] + 1e-3,
                        max_distance=MAX_DISTANCE_FACTOR,#MAX_DISTANCE_FACTOR * np.linalg.norm(scale4d[1:]),
                        scale=scale4d[1:],
                        return_costs=True
                    )


                this_JI = len(tp_gt_inds) / (len(tp_gt_inds) + len(fp_pred_inds) + len(all_gt_inds))
                JIs_rate.append(this_JI)


            F1_matrix[ind_fp, ind_fn] = np.mean(JIs_rate)


            path_to_trackability = f'{path_to_tracks}/fnfp/coords_fn{ind_fn}_fp{ind_fp}'

            tracks_all = Tracks()
            tracks_all.load_utrack3d_json(
                path_to_json=f'{path_to_trackability}/tracks.json',
                scale=scale4d,
                path_to_trackability_json=f'{path_to_trackability}/trackability.json'
            )

            ### Stardist
            infos_all = [[] for _ in range(MAX_TIME)]

            for track in tracks_all:
                for tind,zind,yind,xind,ty in zip(track.tinds,track.zinds, track.yinds, track.xinds, track.nodes_trackabilities):
                    
                    infos_all[tind].append([track.index, zind,yind,xind, ty])

            infos_all = [np.array(elem) for elem in infos_all]
            ty_all = [elem[:,4] for elem in infos_all]
            ###


            Tys_all = []
            for tind in range(MAX_TIME):
                Tys_all.append(np.mean(ty_all[tind]))
            Ty_matrix[ind_fp, ind_fn] = np.mean(Tys_all)

    path_to_save = f'{path_to_tracks}/fnfp'

    np.savetxt(f'{path_to_save}/Ty_matrix.txt'  ,Ty_matrix)
    np.savetxt(f'{path_to_save}/F1_matrix.txt'  ,F1_matrix)

    plot=True

if plot:    

    Ty_matrix = np.loadtxt(f'{path_to_save}/Ty_matrix.txt')
    F1_matrix = np.loadtxt(f'{path_to_save}/F1_matrix.txt')

    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(1,2,1)

    im1 = ax.imshow(Ty_matrix[::-1], interpolation='none', cmap='Blues_r')

    cbar1 = fig.colorbar(im1, ax=ax,shrink=0.8)
    cbar1.set_label('our score\n ', fontsize=14)
    
    
    ax.set_xlabel('FN rate (%)', fontsize=16)
    ax.set_ylabel('FP rate (%)', fontsize=16)

    ax.set_yticks(np.arange(len(fp_rates))[1::2])
    ax.set_xticks(np.arange(len(fp_rates))[::2])
    ax.set_yticklabels([f'{int(elem)}' for elem in fp_rates][::-1][1::2])
    ax.set_xticklabels([f'{int(elem)}' for elem in fn_rates][::2])

    ax2 = fig.add_subplot(1,2,2)

    im2 = ax2.imshow(F1_matrix[::-1], interpolation='none', cmap='Reds_r')
    cbar2 = fig.colorbar(im2, ax=ax2,shrink=0.8)
    cbar2.set_label(r'$F1_c\left(\tau_c\right)$', fontsize=14)
    
    ax2.set_xlabel('FN rate (%)', fontsize=16)
    ax2.set_ylabel('FP rate (%)', fontsize=16)


    ax2.set_yticks(np.arange(len(fp_rates))[1::2])
    ax2.set_xticks(np.arange(len(fp_rates))[::2])
    ax2.set_yticklabels([f'{int(elem)}' for elem in fp_rates][::-1][1::2])
    ax2.set_xticklabels([f'{int(elem)}' for elem in fn_rates][::2])


    for a in [ax,ax2]:

        for col, pos in zip(['green','yellow','violet'], [3,6,9]):

            a.plot(np.arange(len(fn_rates)), [10-pos]*len(fp_rates),c=col,lw=0.75)


    fig.tight_layout()




    fig2, (ax1, ax2,ax3) = plt.subplots(ncols=3, sharey=True, figsize=(8,3))

    ax1.plot(fn_rates, Ty_matrix[3], '-',marker='.')
    ax1.plot(fn_rates, F1_matrix[3], c='red')
    ax1.plot([-11,-10],[-11,-10], label=f"Spearman's coeff = {spearmanr(Ty_matrix[3], F1_matrix[3])[0]:.2f}")
    ax1.set_ylim(0.5,1)
    ax1.set_xlim(-1.8,46.8)
    ax1.set_yticks([0.55,0.75,1])

    ax2.plot(fn_rates, Ty_matrix[6], '-',marker='.')
    ax2.plot(fn_rates, F1_matrix[6], c='red')
    ax2.plot([-11,-10],[-11,-10], label=f"Spearman's coeff = {spearmanr(Ty_matrix[6], F1_matrix[6])[0]:.2f}")
    ax2.set_ylim(0.5,1)
    ax2.set_xlim(-1.8,46.8)
    ax2.set_yticks([0.55,0.75,1])

    ax3.plot(fn_rates, Ty_matrix[9], '-',marker='.')
    ax3.plot(fn_rates, F1_matrix[9], c='red')
    ax3.plot([-11,-10],[-11,-10], label=f"Spearman's coeff = {spearmanr(Ty_matrix[9], F1_matrix[9])[0]:.2f}")
    ax3.set_ylim(0.5,1)
    ax3.set_xlim(-1.8,46.8)
    ax3.set_yticks([0.5,0.75,1])

    ax1.legend(loc=3, handlelength=0, handleheight=0)
    ax2.legend(loc=3, handlelength=0, handleheight=0)
    ax3.legend(loc=3, handlelength=0, handleheight=0)

    ax1.set_xlabel('FN rate (%)', fontsize=16)
    ax2.set_xlabel('FN rate (%)', fontsize=16)
    ax3.set_xlabel('FN rate (%)', fontsize=16)
    ax1.set_ylabel('evaluation score (a.u)', fontsize=16)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.8) 
        ax2.spines[axis].set_linewidth(1.8) 
        ax3.spines[axis].set_linewidth(1.8) 
    plt.setp(ax1.spines.values(), color='green')
    plt.setp(ax2.spines.values(), color='yellow')
    plt.setp(ax3.spines.values(), color='violet')

    fig2.tight_layout()
    plt.subplots_adjust(wspace=0.02)


fig.savefig(f'{path_to_figures}/JI_vs_Ty_simus_fnfp.svg')
fig.savefig(f'{path_to_figures}/JI_vs_Ty_simus_fnfp.pdf')
fig.savefig(f'{path_to_figures}/JI_vs_Ty_simus_fnfp.png')

fig2.savefig(f'{path_to_figures}/JI_vs_Ty_simus_fnfp_spearman.svg')
fig2.savefig(f'{path_to_figures}/JI_vs_Ty_simus_fnfp_spearman.pdf')
fig2.savefig(f'{path_to_figures}/JI_vs_Ty_simus_fnfp_spearman.png')

print(spearmanr(F1_matrix.ravel(), Ty_matrix.ravel()))

plt.show()   
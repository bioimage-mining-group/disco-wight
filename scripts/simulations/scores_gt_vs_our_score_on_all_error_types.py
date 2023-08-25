import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from labellines import labelLines

from dwight.tracking_classes import Tracks
from dwight.utils import load_csv_coords

"""
Loads corrupted simulations coordinates, computes the quality score,
and plots along theoretical F1 score as a function of the error rate. 
"""


path_to_root = ...

path_to_coords = f'{path_to_root}/simulations/coords'
path_to_tracks = f'{path_to_root}/simulations/utrack'

path_to_figures = f'{path_to_root}/figures'

scale4d = np.array((1,)*4)

# Extract ground truth coordinates
all_coords_gt = []
files = sorted(os.listdir(f'{path_to_coords}/gt'))
    
for ind_t,file in enumerate(files):
    coords_gt = load_csv_coords(
        path_to_csv=f'{path_to_coords}/gt/{file}'
    )
    all_coords_gt.append(coords_gt)

all_coords_gt = np.array(all_coords_gt)


# Prepare to loop over error error types and error rates
meta_all_rates = np.arange(0,50,5)
all_rates_str = [str(elem).zfill(3) for elem in meta_all_rates]
all_rates = meta_all_rates/100

# all theoretical F1 values as a function of the error rates
all_functions = [
    ((1-2*all_rates)/(1-all_rates/2), (1-all_rates)/(1-all_rates/2)), 
    ((1-all_rates)/(1+all_rates/2), 1/(1+all_rates/2)),
    (1/(1+all_rates/2), None), 
    ((1-all_rates)/(1-all_rates/2), None),
]

all_names = ["merge", "split","fp", "fn"]


number_of_frames = len(files)
N_tests = 10


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
axes = [elem for foo in axes for elem in foo]

for ax, name, (fmin, fmax) in tqdm(zip(axes, all_names, all_functions)):

    if name == 'fn':
        ax.set_xlabel(f'FN rate (%)', fontsize=16)
        ax.plot(meta_all_rates, fmin, color='red', alpha=1)
    elif name == 'fp':
        ax.set_xlabel(f'FP rate (%)', fontsize=16)
        ax.plot(meta_all_rates, fmin, color='red', alpha=1)
    else:
        if name == 'fnfp':
            ax.set_xlabel(f'FN+FP rate', fontsize=16)
        else:
            
            ax.set_xlabel(f'{name} rate (%)', fontsize=16)

        # Number of lines for gradient
        N_grad=30
        for i in range(N_grad+1):

            j=i/(2*N_grad)
            alpha = 0.05+1.5*(0.9*j)**2
            
            if i!=N_grad:
                if i==0:
                    ax.plot(meta_all_rates, (1-j)*fmin+j*fmax, color='red', alpha=alpha,label=r'$\tau_c \rightarrow 0$')
                    ax.plot(meta_all_rates, j*fmin+(1-j)*fmax, color='red', alpha=alpha,label=r'$\tau_c \rightarrow \infty$')

                    labelLines(ax.get_lines(), zorder=2.5,xvals=[0.17*100,0.25*100], align=True,alpha=0.5)
                else:
                    ax.plot(meta_all_rates, j*fmin+(1-j)*fmax, color='red', alpha=alpha)
                    ax.plot(meta_all_rates, (1-j)*fmin+j*fmax, color='red', alpha=alpha)
            else:
                ax.plot(meta_all_rates, j*fmin+(1-j)*fmax, color='red', alpha=alpha)


    Ty_err = []

    for ind_test in range(N_tests):

        Ty_test = []

        for all_rate_str in all_rates_str:

            path_to_trackability = f'{path_to_tracks}/{name}/{ind_test}/coords_{name}_{all_rate_str}'

            tracks_all = Tracks()
            tracks_all.load_utrack3d_json(
                path_to_json=f'{path_to_trackability}/tracks.json',
                scale=scale4d,
                path_to_trackability_json=f'{path_to_trackability}/trackability.json'
            )

            ### 
            infos_all = [[] for _ in range(number_of_frames)]

            for track in tracks_all:
                for tind,zind,yind,xind,ty in zip(track.tinds,track.zinds, track.yinds, track.xinds, track.nodes_trackabilities):
                    
                    infos_all[tind].append([track.index, zind,yind,xind, ty])

            infos_all = [np.array(elem) for elem in infos_all]
            ty_all = [elem[:,4] for elem in infos_all]
            ###


            Tys_all = []
            for tind in range(number_of_frames):
                Tys_all.append(np.mean(ty_all[tind])) # mean over all tracks
            Ty_test.append(np.mean(Tys_all)) # mean over all t
        
        Ty_err.append(Ty_test) # information over all tests to get mean and std in postproc


    path_to_save_TYS = f'{path_to_tracks}/{name}' 

    np.savetxt(f'{path_to_save_TYS}/TYs_{name}.txt'  ,np.array(Ty_err))


    Ty_err = np.loadtxt(f'{path_to_save_TYS}/TYs_{name}.txt')
    
    means = np.mean(Ty_err, axis=0)
    stds = np.std(Ty_err, axis=0)


    if name == 'merge':
        # Remove last point of our score on the merge plot
        means = means[:-1]
        stds = stds[:-1]
        all_rates_plot = meta_all_rates[:-1]
    else:
        all_rates_plot = meta_all_rates

    ax.set_ylim(0.5,1.04)
    ax.set_xlim(min(meta_all_rates),max(meta_all_rates)*1.01)
    ax.set_ylabel('evaluation score (a.u.)', fontsize=16)

    ax.plot(all_rates_plot, means, '-',marker='.')
    
# Add legend
ax = axes[3]

ax.plot([-1,-2], [-1,-2], label=r'$F1_c(\tau_c)$', c='r')
ax.plot([-1,-2], [-1,-2], label='our score', c='#1f77b4')

ax.legend(loc='best', prop={'size':12})

# Change rates on merge plot
ax = axes[all_names.index('merge')]

ax.set_xticks([int(elem) for elem in meta_all_rates[:-1][::2]])
ax.set_xticklabels([str(elem) for elem in 2*meta_all_rates[:-1][::2]])

fig.tight_layout()

fig.savefig(f'{path_to_figures}/JI_vs_Ty_simus.svg')
fig.savefig(f'{path_to_figures}/JI_vs_Ty_simus.pdf')
fig.savefig(f'{path_to_figures}/JI_vs_Ty_simus.png')

plt.show()   
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from packajules.segmentation_classes import Tracks
import packajules.comparisons as jcomp
from organo_simulator.utils import load_csv_coords
import os


from labellines import labelLine, labelLines



scale4d = (1,)*4
scale3d = np.array(scale4d[1:])

path2data = '/home/jvanaret/data/data_trackability_study/simulations/test_long'
path2tracks = '/home/jvanaret/data/data_trackability_study/utrack/simulations/all_test_long_SR10'
path2save_JIS = '/home/jvanaret/data/data_trackability_study/simulations/all_test_long_SR10'
path2save_TYS = '/home/jvanaret/data/data_trackability_study/simulations/all_test_long_SR10'


MAX_TIME = 46

d0 = 16* 0.8489


coords_gt = []
files = sorted(os.listdir(f'{path2data}/coords'))
    
for ind_t,file in enumerate(files):
    c_gt = load_csv_coords(
        path_to_csv=f'{path2data}/coords/{file}'
    )
    coords_gt.append(c_gt)

coords_gt = np.array(coords_gt)

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,3))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
axes = [elem for foo in axes for elem in foo]
# axes[-1].remove()
# axes = axes[:-1]

all_SRs = [10];
all_JIRs = [4,6,8,10,16]#np.logspace(np.log10(0.5), np.log10(200), 10)
# all_names = ["fn", "fp", "fnfp", "merge", "split"]
# all_names = ["fn", "fp", "merge", "split"]
all_names = ["merge", "split","fp", "fn"]

meta_all_rates = np.arange(0,50,5)
all_rates_str = [str(elem).zfill(3) for elem in meta_all_rates]
all_rates = meta_all_rates/100

# all_functions = [
#     ((1-all_rates)/(1-all_rates/2), None),
#     (1/(1+all_rates/2), None), 
#     #((1-all_rates/2), np.ones(shape=len(all_rates))), 
#     ((1-2*all_rates)/(1-all_rates/2), (1-all_rates)/(1-all_rates/2)), 
#     ((1-all_rates)/(1+all_rates/2), 1/(1+all_rates/2))
# ]
all_functions = [
    ((1-2*all_rates)/(1-all_rates/2), (1-all_rates)/(1-all_rates/2)), 
    ((1-all_rates)/(1+all_rates/2), 1/(1+all_rates/2)),
    (1/(1+all_rates/2), None), 
    ((1-all_rates)/(1-all_rates/2), None),
]

plot_JIs = True
plot_Tys = True

for ax, name, (fmin, fmax) in tqdm(zip(axes, all_names, all_functions)):

    

    ax.set_ylim(0.5,1.04)
    ax.set_xlim(min(all_rates),max(all_rates))

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_title(name, fontsize=16)
    ax.set_ylabel('evaluation score (a.u.)', fontsize=16)
    if name == 'fn':
        ax.set_xlabel(f'FN rate', fontsize=16)
        ax.plot(all_rates, fmin, color='red', alpha=1)#, label='JIc')
    elif name == 'fp':
        ax.set_xlabel(f'FP rate', fontsize=16)
        ax.plot(all_rates, fmin, color='red', alpha=1)#, label='JIc')
    else:
        if name == 'fnfp':
            ax.set_xlabel(f'FN+FP rate', fontsize=16)
        else:
            
            ax.set_xlabel(f'{name} rate', fontsize=16)

        
        # ax.plot(all_rates, 0.5*fmin+0.5*fmax, color='red', alpha=0.8)#, label='JIc')
        
        N_grad=30
        for i in range(N_grad+1):



            j=i/(2*N_grad)
            alph = 0.05+1.5*(0.9*j)**2
            print(alph)
            
            if i!=N_grad:
                if i==0:
                    ax.plot(all_rates, (1-j)*fmin+j*fmax, color='red', alpha=alph,label=r'$\tau \rightarrow 0$')
                    ax.plot(all_rates, j*fmin+(1-j)*fmax, color='red', alpha=alph,label=r'$\tau \rightarrow \infty$')
                else:
                    ax.plot(all_rates, j*fmin+(1-j)*fmax, color='red', alpha=alph)
                    ax.plot(all_rates, (1-j)*fmin+j*fmax, color='red', alpha=alph)
                #, label=r'$2\hat{\tau}$')
            else:
                ax.plot(all_rates, j*fmin+(1-j)*fmax, color='red', alpha=alph)


        # ax.plot(all_rates, 0.25*fmin+0.75*fmax, color='red', alpha=0.4, label=r'$2\hat{\tau}$')
        # ax.plot(all_rates, 0.75*fmin+0.25*fmax, color='red', alpha=0.4, label=r'$\hat{\tau}/2$')
        # ax.plot(all_rates, 0*fmin+1*fmax, color='red', alpha=0.1, label=r'$10\hat{\tau}$')
        # ax.plot(all_rates, 1*fmin+0*fmax, color='red', alpha=0.1, label=r'$\hat{\tau}/10$')
    # ax.set_ylabel(f'JIc', fontsize=16)

    if plot_JIs:
        JIs_to_save = np.loadtxt(f'{path2save_JIS}/JIs_{name}.txt')
    else:
        JIs_to_save = []


    # for ind_JIR, JIR in enumerate(all_JIRs):

    #     if not plot_JIs:

    #         JI_SR = []

    #         for all_rate_str in all_rates_str:

    #             files = sorted(os.listdir(f'{path2data}/coords_{name}_{all_rate_str}'))
                

    #             coords_all = []
    #             for ind_t,file in enumerate(files):
    #                 c_all = load_csv_coords(
    #                         path_to_csv=f'{path2data}/coords_{name}_{all_rate_str}/{file}'
    #                     )
    #                 coords_all.append(c_all)

    #             coords_all = coords_all

    #             JIs_rate = []


    #             MAX_DISTANCE_FACTOR = JIR

    #             for tind in range(MAX_TIME):

    #                 (tp_gt_inds, tp_pred_inds, fp_pred_inds, all_gt_inds), costs = jcomp.compare_detections(
    #                         gt_inds_coordinates=coords_gt[tind],
    #                         prediction_inds_coordinates=coords_all[tind] + 1e-3,
    #                         max_distance=MAX_DISTANCE_FACTOR,#MAX_DISTANCE_FACTOR * np.linalg.norm(scale4d[1:]),
    #                         scale=scale4d[1:],
    #                         return_costs=True
    #                     )


    #                 this_JI = len(tp_gt_inds) / (len(tp_gt_inds) + (len(fp_pred_inds) + len(all_gt_inds)))
    #                 JIs_rate.append(this_JI)


    #             JI_SR.append(np.mean(JIs_rate))
    #         JIs_to_save.append(JI_SR)
        
    #     else:
    #         if not (name in ['fp', 'fn']):
    #             label = f'{(JIR/d0):.1f}'
    #             # ax.fill_between(all_rates, fmin, fmax, color='k', alpha=0.05)
    #         else:  
    #             label = ''
            
    #         # ax.plot(all_rates, JIs_to_save[ind_JIR], c='red')#label=label,c='red')#c='k')
        
    if not plot_JIs:
        np.savetxt(f'{path2save_JIS}/JIs_{name}.txt'  ,np.array(JIs_to_save))


    if plot_Tys:
        Tys_to_save = np.loadtxt(f'{path2save_TYS}/TYs_{name}.txt')

        if isinstance(Tys_to_save[0], float):
            Tys_to_save = [Tys_to_save]
    else:     
        Tys_to_save = []   
            

    labelLines(ax.get_lines(), zorder=2.5,xvals=[0.17,0.25], align=True,alpha=0.5)

    # if plot_JIs or plot_Tys:
    #     labelLines(ax.get_lines(), zorder=2.5)
    
    for ind_SR,SR in enumerate(all_SRs):

        
        

        if not plot_Tys:

            Ty_SR = []

            for all_rate_str in all_rates_str:

                path2trackability = f'{path2tracks}/coords_{name}_{all_rate_str}_{ind_SR}'

                tracks_all = Tracks()
                tracks_all.load_utrack3d_json(
                    path_to_json=f'{path2trackability}/tracks.json',
                    scale=scale4d,
                    path_to_trackability_json=f'{path2trackability}/trackability.json'
                )

                ### Stardist
                infos_all = [[] for _ in range(MAX_TIME)]

                for track in tracks_all:
                    for tind,zind,yind,xind,ty in zip(track.tinds,track.zinds, track.yinds, track.xinds, track.nodes_trackabilities):
                        
                        infos_all[tind].append([track.index, zind,yind,xind, ty])

                infos_all = [np.array(elem) for elem in infos_all]
                # indices_all = [elem[:,0] for elem in infos_all]
                #coords_all = [elem[:,[1,2,3]] for elem in infos_all]
                ty_all = [elem[:,4] for elem in infos_all]
                ###


                Tys_all = []
                for tind in range(MAX_TIME):
                    Tys_all.append(np.mean(ty_all[tind]))
                Ty_SR.append(np.mean(Tys_all))
            
            Tys_to_save.append(Ty_SR)

        

        else:
            label = 'our score'#if name not in ['fp', 'fn'] else ''
            ax.plot(all_rates, Tys_to_save[ind_SR],label=label)
        
    if not plot_Tys:
        np.savetxt(f'{path2save_TYS}/TYs_{name}.txt'  ,np.array(Tys_to_save))

       
    
    #ax.plot([-10, -9], [-10, -9], c='blue', label='trackability')
    #ax.plot([-10, -9], [-10, -9], c='red',label='JI centroid')
    # ax.plot([-10, -9], [-10, -9], 'k',label='JIc')
    # ax.legend()

    


    # # ax.boxplot(Ty_SR, medianprops=dict(color='blue', linewidth=2), showfliers=False)
    # ax.boxplot(JI_SR, medianprops=dict(color='red', linewidth=2), showfliers=False)



    # ax.set_xlabel('FN+FP rate', fontsize=18)

    # # def f_min(x):
    # #     return (1-x)/(1+x)

    # def f_max(x):
    #     return (1-x/2)/(1+x/2)

    # # ax.plot(np.arange(1,len(all_rates)+1), f_min(all_rates),'k--')
    # ax.plot(np.arange(1,len(all_rates)+1), f_max(all_rates),'k--')

    # ax.set_xlim(1-0.4, len(all_rates)+ 0.4)
    # ax.set_ylim(0,1.07)
    # ax.set_xticklabels([f'{m:.2f}' for m in all_rates]+[f'{m:.2f}' for m in all_rates])


    # # ax.plot([-10, -9], [-10, -9], c='blue', label='trackability')
    # ax.plot([-10, -9], [-10, -9], c='red',label='JI centroid')
    # ax.plot([-10, -9], [-10, -9], 'k--',label='theory')
    # ax.legend()

if plot_Tys or plot_JIs:



    ax = axes[3]

    label = 'eval. score'

    # ax.plot([-1,-2], [-1,-2], label=label)
    ax.plot([-1,-2], [-1,-2], label=r'$F1_c(\tau_c)$', c='r')

    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_axis_off()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ax.legend(loc='best', prop={'size':12})

    fig.tight_layout()

    fig.savefig('/home/jvanaret/Desktop/plots_svg/JI_vs_Ty_simus.svg')
    fig.savefig('/home/jvanaret/Desktop/plots_svg/JI_vs_Ty_simus.pdf')
    fig.savefig('/home/jvanaret/Desktop/plots_svg/JI_vs_Ty_simus.png')

    plt.show()   
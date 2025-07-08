import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os

from utils import *
from plot_utils import *
from AnalyseDefects_dev import AnalyseDefects
from AnalyseDefectsAll import AnalyseDefectsAll


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

plt.style.use('sg_article')
plt.rcParams.update({"text.usetex": True,
                     'figure.figsize': (6, 4),})

def main():

    # STEP 1: Initialize the analysis parameters and data paths
    ##### ---------------------------------------------------

    data_suffix='lfric10bc' #'lk' #'lbc' #  'lfric10bc' #'lambda_minus1'
    LL = 512
    mode = 'all' # 'all' or 'short'

    cluster_dir = 'lustre'

    extract = False
    do_basic_analysis, do_hyperuniformity_analysis, do_merge, do_plotting = False, False, False, True
    calc_pcf = False


    if data_suffix == 'lk':
        prefix_list = []
        suffix_list = ['025', '10']
        Nframes_list = [400, 400] 
        count_suffix = "_periodic_rm0.1"
        label_list = [r'$K_{\mathrm{ref}} / 2 $', 
                    r'$K_{\mathrm{ref}} = 0.05$',
                    r'$2 K_{\mathrm{ref}}$']
    elif data_suffix == 'lbc':
        prefix_list = []
        suffix_list = ['3', '4']
        Nframes_list = [400, 400] 
        count_suffix = "_rm0.1"
        label_list = ['Free-slip', 'Periodic', 'No-slip']
    elif data_suffix == 'lam':
        prefix_list = []
        suffix_list = ['05', '2']
        Nframes_list = [100, 100] 
        count_suffix = "_periodic_rm0.1"
        label_list = [r'$\lambda = 0.5$', 
                    r'$\lambda = 1$',
                r'$\lambda = 2$']
    elif data_suffix == 'lfric10bc':
        prefix_list = [] #['', 'l', '01']
        suffix_list = ['3', '4'] #'01',]# '10']
        Nframes_list = [400, 400] 
        count_suffix = "_rm0.1"
        label_list = ['Free-slip', 'Periodic', 'No-slip']

    output_path = f'data\\na{LL}{data_suffix}'
    save_path = os.path.join(output_path, 'figs')
    save_path0 = 'C:\\Users\\Simon Andersen\\OneDrive - University of Copenhagen\\PhD\\Active Nematic Defect Transition\\figs\\SI'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path0):
        os.makedirs(save_path0)

    defect_list = []

    if len(prefix_list) > 0:
        for i, prefix in enumerate(prefix_list):
            if cluster_dir == 'groups':
                path = f'X:\\na512exp\\na{LL}{prefix}{data_suffix}'
            elif cluster_dir == 'lustre':
                path = f'Z:\\nematic_analysis\\na512exploration\\na{LL}{prefix}{data_suffix}'
            else:
                raise ValueError("Unknown cluster directory. Choose 'groups' or 'lustre'.")
            data_dict = dict(path = path, \
                        suffix = 's' if len(prefix) == 0 else prefix, priority = i, LX = LL, Nframes = Nframes_list[i])
            defect_list.append(data_dict)
    else:
        for i, suffix in enumerate(suffix_list):
            if cluster_dir == 'groups':
                path = f'X:\\na512exp\\na{LL}{data_suffix}{suffix}'
            elif cluster_dir == 'lustre':
                path = f'Z:\\nematic_analysis\\na512exploration\\na{LL}{data_suffix}{suffix}'
            data_dict = dict(path = path, \
                        suffix = suffix, priority = 0, LX = LL, Nframes =  Nframes_list[i])
            defect_list.append(data_dict)


    ad = AnalyseDefects(defect_list, output_path=output_path, count_suffix=count_suffix,)


    # STEP 2: Extract data and perform analysis
    ##### ---------------------------------------------------

    if extract:
        ad.extract_results()
    if do_basic_analysis or do_hyperuniformity_analysis:
        # hyperuniformity parameters
        act_idx_bounds=[0,None]
        Npoints_to_fit = 20
        dens_fluc_dict = dict(act_idx_bounds = [0, None], window_idx_bounds = [50 - Npoints_to_fit, None])

        # sfac fitting parameters
        pval_min = 0.05
        Nbounds = [4,5]
        sfac_dict = dict(Npoints_bounds = Nbounds, act_idx_bounds = act_idx_bounds, pval_min = pval_min)

        # temporal correlation parameter
        temp_corr_simple = True
        nlags = None
        ff_idx = None
        acf_dict = {'nlags_frac': 0.5, 'nlags': nlags, 'max_lag': None,\
                        'alpha': 0.3174, 'max_lag_threshold': 0, 'simple_threshold': 0.2, \
                        'first_frame_idx': ff_idx}

        if do_basic_analysis:
            if do_hyperuniformity_analysis:
                print(' sfac n bounds:', Nbounds)
                
                ad.analyze_defects(temp_corr_simple=temp_corr_simple,
                                    acf_dict=acf_dict,
                                    dens_fluc_dict=dens_fluc_dict, 
                                    sfac_dict=sfac_dict, 
                                    calc_pcf=calc_pcf)
            else:
                ad.analyze_defects(acf_dict=acf_dict,
                                    temp_corr_simple=temp_corr_simple)
    if do_merge:
        ad.merge_results()


    if do_plotting:
        # STEP 3: Init reference class for comparison
        ##### ---------------------------------------------------

        if data_suffix == 'lfric10bc':
            prefix_list2 = ['', 'l'] #['', 'l', '01']
            suffix_list2 = ['', 'l'] #'01',]# '10']
            Nframes_list2 = [100, 400] 
            count_suffix2 = "_periodic_rm0.1"

            data_suffix_ref = 'fric10'
            output_path_ref = f'data\\na{LL}{data_suffix_ref}'
            defect_list_ref = []
    

            if len(prefix_list2) > 0:
                for i, prefix in enumerate(prefix_list2):
                    data_dict = dict(path = f'X:\\na512exp\\na{LL}{prefix}{data_suffix_ref}', \
                                suffix = 's' if len(prefix) == 0 else prefix, priority = i, LX = LL, Nframes = Nframes_list2[i])
                    defect_list_ref.append(data_dict)
            else:
                for i, suffix in enumerate(suffix_list2):
                    data_dict = dict(path = f'X:\\na512exp\\na{LL}{data_suffix_ref}{suffix}', \
                                suffix = suffix, priority = 0, LX = LL, Nframes =  Nframes_list2[i])
                    defect_list_ref.append(data_dict)

            ad2 = AnalyseDefects(defect_list_ref, output_path=output_path_ref,count_suffix=count_suffix2)
        else:
            LL = 512
            output_path_ref = f'data\\na{LL}'
            mode_ref = 'all' # 'all' or 'short'

            defect_list_ref = gen_analysis_dict(LL, mode_ref)
            ad2 = AnalyseDefects(defect_list_ref, output_path=output_path_ref)


        # STEP 4: Perform comparison analysis and plotting
        ##### ---------------------------------------------------

        do_plotting_for = ['fig', 'fig3'] #'fig4']
        make_superfig = True
        use_sus_in_superfig = False

        if make_superfig: 
            width = 11
            height = 4
            fig00, ax00 = plt.subplots(ncols=2, figsize=(width, height))
            ax = ax00[0]
            ax3 = ax00[1]

        act_list_list = [ad.act_list[0], ad2.act_list_merged, ad.act_list[1]]
        marker_list = ['ro', 'gd', 'bs',] if data_suffix=='lfric10bc' else ['ro', 'bs', 'gd'] 

        ## plot av. defect density 
        if 'fig' in do_plotting_for or 'all' in do_plotting_for:
            if not make_superfig: fig, ax = plt.subplots(figsize=(7,5))
        
            av_defect_list = [ad.get_arrays_av(0)[-1], ad2.get_arrays_av(use_merged=True)[-1], ad.get_arrays_av(1)[-1]]

            for i, label in enumerate(label_list):
                av_def = av_defect_list[i] / LL ** 2
                ax.errorbar(act_list_list[i], av_def[:,0], av_def[:,1], fmt = marker_list[i], label=label,
                            elinewidth=1.5, capsize=1.5, capthick=1, markersize = 5, alpha=.5)

            ax.set_xlabel(r'Activity ($\tilde{\zeta}$)')
            ax.set_ylabel(r' Av. defect density ($\overline{\rho_N})$')
            ax.vlines(x=0.022, ymin = -1e-2, ymax=.6e-2, linestyle='--', color='k', lw = 1, alpha=.65, zorder=-10)
            ax.hlines(y=0, xmin=0, xmax=.052, linestyle='-', color='k', lw = 1 )

            ax.tick_params(axis='both',which='major', labelsize=14)
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5e-3))   
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(.5e-3))  

            if data_suffix == 'lbc':
                act_min, act_max = 0.0082, 0.052
                xticks = [0.01, 0.02, 0.03, 0.04, 0.05]
                order = [0, 1, 2]
            elif data_suffix == 'lk':
                act_min, act_max = 0.0052, 0.052
                xticks = [0.01, 0.02, 0.03, 0.04, 0.05]
                order = [0, 1, 2]
            else:
                act_min, act_max = 0.0178, 0.052
                xticks = [0.02, 0.02, 0.03, 0.04, 0.05]
                order = [0, 2,1]


            handles, labels = ax.get_legend_handles_labels()
            
            ax.set_xticks(xticks, xticks)
            ax.set_xlim([act_min, act_max])
            ax.set_ylim(ymin=-.025e-2, ymax = .55e-2)
            if data_suffix != 'lk':ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower right', ncols=1)
            #ax.legend(loc='lower right', ncols=1)
            if not make_superfig: 
                ax.legend(loc='lower right', ncols=1)
                fig.savefig(os.path.join(save_path, f'av_density_{data_suffix}.jpeg'), dpi=620, bbox_inches='tight')


        ## plot susceptibility
        if 'fig2' in do_plotting_for or 'all' in do_plotting_for:

            if not make_superfig: fig2, ax2 = plt.subplots(figsize=(7,5))
            sus_list = [ad.get_susceptibility(0), ad2.get_susceptibility(use_merged=True), ad.get_susceptibility(1)]

            for i, label in enumerate(label_list):
                sus = sus_list[i]
                ax2.errorbar(act_list_list[i], sus[:,0], sus[:,1], fmt = marker_list[i], label=label,
                            elinewidth=1.5, capsize=1.5, capthick=1, markersize = 5, alpha=.5)
            
            ax2.tick_params(axis='both',which='major', labelsize=14)
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(2.5e-3))   
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1))     

            if data_suffix == 'lbc':
                act_min, act_max = 0.0082, 0.052
            elif data_suffix == 'lk':
                act_min, act_max = 0.0052, 0.052
            else:
                act_min, act_max = 0.0178, 0.052
            ax2.set_xlim([act_min, act_max])

    
            ax2.set_ylim(ymin=0, ymax=13)
            ax2.set_xlabel(r'Activity ($\tilde{\zeta}$)')
            ax2.set_ylabel(r'Active susceptibility ($\overline{\chi_a})$')
            ax2.vlines(x=0.022, ymin = -1.2, ymax=20, linestyle='--', color='k', lw = 1, alpha=.65, zorder=-10)

            ax2.hlines(y=0, xmin=0, xmax=0.052, linestyle='-', color='k', lw = 1 )
            
            if not make_superfig: 
                ax2.legend(loc='upper right', ncols=1)
                fig2.savefig(os.path.join(save_path, f'susceptibility_{data_suffix}.jpeg'), dpi=620, bbox_inches='tight')
            if use_sus_in_superfig:
                fig00.savefig(os.path.join(save_path, f'av_density_susceptibility_{data_suffix}.jpeg'), dpi=620, bbox_inches='tight', pad_inches=0.1)

        ## plot hyperuniformity exponent across and for no-slip using 3 approaches
        if 'fig3' in do_plotting_for or 'all' in do_plotting_for:
            
            if not make_superfig: fig3, ax3 = plt.subplots(figsize=(7, 5))
            fit_params_paths = [os.path.join(ad.output_paths[0], 'fit_params_sfac_time_av.npy'), 
                                os.path.join(ad2.output_merged, 'fit_params_sfac_time_av.npy'),
                                os.path.join(ad.output_paths[1], 'fit_params_sfac_time_av.npy')]
            
            for i, label in enumerate(label_list):
                fit_params = np.load(fit_params_paths[i])

                act_idx_min = 0 if data_suffix == 'lbc' else 0
                
                ax3.errorbar(act_list_list[i][act_idx_min:], fit_params[act_idx_min:,0], yerr=fit_params[act_idx_min:,2], \
                            fmt = marker_list[i], elinewidth=1.5, capsize=1.5, capthick=1, markersize = 5, \
                            alpha=.5, label=None if data_suffix=='lk' else label)

            ax3.tick_params(axis='both',which='major', labelsize=14)
            ax3.xaxis.set_minor_locator(ticker.MultipleLocator(2.5e-3))   
            ax3.yaxis.set_minor_locator(ticker.MultipleLocator(.1))

            if data_suffix == 'lbc':
                act_min, act_max = 0.0082, 0.052
                ymin, ymax = -.58, .25
                xticks = [0.01, 0.02, 0.03, 0.04, 0.05]
                order = [0, 1, 2]
            elif data_suffix == 'lk':
                act_min, act_max = 0.0052, 0.052
                ymin, ymax = -.58, .3
                xticks = [0.01, 0.02, 0.03, 0.04, 0.05]
                order = [0, 1, 2]
            else:
                act_min, act_max = 0.0178, 0.052
                ymin, ymax = -.5, .25
                xticks = [0.02, 0.03, 0.04, 0.05]
                order = [0, 2, 1]
            ax3.set_xticks(xticks, xticks)
            ax3.set_xlim([act_min, act_max])
            ax3.set_ylim(ymin=ymin, ymax=ymax)

            handles, labels = ax3.get_legend_handles_labels()
       
            ax3.set_xlabel(r'Activity ($\tilde{\zeta}$)')
            ax3.set_ylabel(r'Hyperuniformity exponent ($\gamma$)')
            ax3.vlines(x=0.022, ymin = -1.2, ymax=1.2, linestyle='--', color='k', lw = 1, alpha=.65, zorder=-10)
            ax3.hlines(y=0, xmin=0, xmax=0.092, linestyle='-', color='k', lw = 1 )
            if data_suffix != 'lk': ax3.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower right', ncols=1)

            if not make_superfig:
                ax3.legend(loc='lower right', ncols=1)
                fig3.savefig(os.path.join(save_path, f'alpha_{data_suffix}.jpeg'), dpi=620, bbox_inches='tight', pad_inches=0.1)
        if make_superfig and not use_sus_in_superfig:
            if data_suffix =='lk': fig00.legend(ncol=3, fontsize = 16, bbox_to_anchor=(0.525, 1.04), loc='upper center')
            fig00.savefig(os.path.join(save_path, f'av_density_alpha_{data_suffix}.jpeg'), dpi=620, bbox_inches='tight', pad_inches=0.1)
            fig00.savefig(os.path.join(save_path, f'av_density_alpha_{data_suffix}.eps'), dpi=620)
            fig00.savefig(os.path.join(save_path0, f'av_density_alpha_{data_suffix}.jpeg'), dpi=620, bbox_inches='tight', pad_inches=0.1)
            fig00.savefig(os.path.join(save_path0, f'av_density_alpha_{data_suffix}.eps'), dpi=620)

        plt.close('all')



if __name__ == "__main__":
    main()
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

    data_suffix='lfric10bc' #'lambda_minus1'
    LL = 512
    mode = 'all' # 'all' or 'short'

    extract = False
    do_basic_analysis, do_hyperuniformity_analysis, do_merge, do_plotting = False, False, False, True
    calc_pcf = False


    if data_suffix == 'lk':
        prefix_list = []
        suffix_list = ['025', '10']
        Nframes_list = [400, 400] 
        count_suffix = "_periodic_rm0.1"
        label_list = [r'$K = 0.025$', 
                    r'$K = 0.05$',
                    r'$K = 0.1$']
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
        count_suffix = "_periodic_rm0.1"
        label_list = ['Free-slip', 'Periodic', 'No-slip']

    output_path = f'data\\na{LL}{data_suffix}'
    save_path = os.path.join(output_path, 'figs')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    defect_list = []

    if len(prefix_list) > 0:
        for i, prefix in enumerate(prefix_list):
            data_dict = dict(path = f'X:\\na512exp\\na{LL}{prefix}{data_suffix}', \
                        suffix = 's' if len(prefix) == 0 else prefix, priority = i, LX = LL, Nframes = Nframes_list[i])
            defect_list.append(data_dict)
    else:
        for i, suffix in enumerate(suffix_list):
            data_dict = dict(path = f'X:\\na512exp\\na{LL}{data_suffix}{suffix}', \
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

        do_plotting_for = ['fig', 'fig2', 'fig3'] #'fig4']

        #fig00, ax00 = plt.subplots(ncols=2, figsize=(10, 10/2.5))
        #ax = ax00[0]
        #ax2 = ax00[1]

        act_list_list = [ad.act_list[0], ad2.act_list_merged, ad.act_list[1]]
        marker_list = ['ro', 'bs', 'gd']

        ## plot av. defect density 
        if 'fig' in do_plotting_for or 'all' in do_plotting_for:
            fig, ax = plt.subplots(figsize=(7,5))
        
            av_defect_list = [ad.get_arrays_av(0)[-1], ad2.get_arrays_av(use_merged=True)[-1], ad.get_arrays_av(1)[-1]]

            for i, label in enumerate(label_list):
                av_def = av_defect_list[i] / LL ** 2
                ax.errorbar(act_list_list[i], av_def[:,0], av_def[:,1], fmt = marker_list[i], label=label,
                            elinewidth=1.5, capsize=1.5, capthick=1, markersize = 5, alpha=.5)

            ax.set_xlabel(r'Activity ($\zeta$)')
            ax.set_ylabel(r' Av. defect density ($\overline{\rho})$')
            ax.vlines(x=0.022, ymin = -1e-2, ymax=.6e-2, linestyle='--', color='k', lw = 1, alpha=.65, zorder=-10)
            ax.hlines(y=0, xmin=0, xmax=.052, linestyle='-', color='k', lw = 1 )

            ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5e-3))   
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(.5e-3))     
            ax.set_xlim([.0178, 0.052])
            ax.set_ylim(ymin=-.05e-2, ymax = .6e-2)
            ax.legend(loc='upper center', ncols=3)
            fig.savefig(os.path.join(save_path, 'av_density.png'), dpi=420, bbox_inches='tight')


        ## plot susceptibility
        if 'fig2' in do_plotting_for or 'all' in do_plotting_for:

            fig2, ax2 = plt.subplots(figsize=(7,5))
            sus_list = [ad.get_susceptibility(0), ad2.get_susceptibility(use_merged=True), ad.get_susceptibility(1)]

            for i, label in enumerate(label_list):
                sus = sus_list[i]
                ax2.errorbar(act_list_list[i], sus[:,0], sus[:,1], fmt = marker_list[i], label=label,
                            elinewidth=1.5, capsize=1.5, capthick=1, markersize = 5, alpha=.5)
                
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(2.5e-3))   
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1))     
            ax2.set_xlim([.0178, 0.052])
            ax2.set_ylim(ymin=0, ymax=13)
            ax2.set_xlabel(r'Activity ($\zeta$)')
            ax2.set_ylabel(r'Active susceptibility ($\overline{\chi_a})$')
            ax2.vlines(x=0.022, ymin = -1.2, ymax=20, linestyle='--', color='k', lw = 1, alpha=.65, zorder=-10)

            ax2.hlines(y=0, xmin=0, xmax=0.052, linestyle='-', color='k', lw = 1 )
            
            ax2.legend(loc='upper right', ncols=1)
            fig2.savefig(os.path.join(save_path, 'susceptibility.png'), dpi=420, bbox_inches='tight')

        #fig00.savefig(os.path.join(save_path, 'av_density_susceptibility.png'), dpi=420, bbox_inches='tight', pad_inches=0.1)

        ## plot hyperuniformity exponent across and for no-slip using 3 approaches
        if 'fig3' in do_plotting_for or 'all' in do_plotting_for:
            
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            fit_params_paths = [os.path.join(ad.output_paths[0], 'fit_params_sfac_time_av.npy'), 
                                os.path.join(ad2.output_merged, 'fit_params_sfac_time_av.npy'),
                                os.path.join(ad.output_paths[1], 'fit_params_sfac_time_av.npy')]
            
            for i, label in enumerate(label_list):
                fit_params = np.load(fit_params_paths[i])
            #   print(len(act_list_list[i]), fit_params.shape)
                ax3.errorbar(act_list_list[i], fit_params[:,0], yerr=fit_params[:,2], fmt = marker_list[i], label=label,
                            elinewidth=1.5, capsize=1.5, capthick=1, markersize = 5, alpha=.5)

            ax3.xaxis.set_minor_locator(ticker.MultipleLocator(2.5e-3))   
            ax3.yaxis.set_minor_locator(ticker.MultipleLocator(.1))     
            ax3.set_xlim([.0178, 0.052])
            ax3.set_ylim(ymin=-.6, ymax=.3)
            ax3.set_xlabel(r'Activity ($\zeta$)')
            ax3.set_ylabel(r'Hyperuniformity exponent ($\gamma$)')
            ax3.vlines(x=0.022, ymin = -1.2, ymax=1.2, linestyle='--', color='k', lw = 1, alpha=.65, zorder=-10)
         #   ax3.vlines(x=0.022, ymin = -.16, ymax=1.2, linestyle='--', color='k', lw = 1)
            ax3.hlines(y=0, xmin=0, xmax=0.052, linestyle='-', color='k', lw = 1 )

            ax3.legend(loc='lower right', ncols=1)
            fig3.savefig(os.path.join(save_path, 'alpha.png'), dpi=420, bbox_inches='tight', pad_inches=0.1)

        ## plot sfac exponent for no-slip
        if 'fig4' in do_plotting_for or 'all' in do_plotting_for:
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            
            label_list_no_slip = [r'$\overline{\delta \rho ^2}$ (time av. of fits)', 
                                rf'$S(k)$ (fit of time av.)',
                                rf'$S(k)$ (time av. of fits)']  
            fit_params_paths_no_slip = [os.path.join(ad.output_paths[1], 'fit_params_count.npy'), 
                                    os.path.join(ad.output_paths[1], 'fit_params_sfac_time_av.npy'),
                                    os.path.join(ad.output_paths[1], 'alpha_list_sfac.npy')]

            for i, label in enumerate(label_list_no_slip):
                fit_params = np.load(fit_params_paths_no_slip[i])
                ax4.errorbar(act_list_list[-1], fit_params[:,0], yerr=fit_params[:,1] if i==2 else fit_params[:,2]
                            , fmt = marker_list[i], label=label, elinewidth=1.5, capsize=1.5, 
                            capthick=1, markersize = 5, alpha=.5)
            
            ax4.xaxis.set_minor_locator(ticker.MultipleLocator(2e-3))
            ax4.set_xlim([.018, 0.052])
            ax4.set_xlabel(r'Activity ($\zeta$)')
            ax4.set_ylabel(r'Hyperuniformity exponent ($\gamma$)')
            ax4.set_ylim(ymin=-1., ymax=.8)
            ax4.vlines(x=0.022, ymin = -1.2, ymax=1.2, linestyle='--', color='k', lw = 1)
            ax4.hlines(y=0, xmin=0, xmax=0.052, linestyle='-', color='k', lw = 1 )
            ax4.legend(loc='lower right', ncols=1)
            fig4.savefig(os.path.join(save_path, 'alpha_no_slip.png'), dpi=420, bbox_inches='tight', pad_inches=0.1)

        plt.close('all')

        if 0:
            fig0, ax0 = plt.subplots(figsize=(9, 6))
            count_var_av = ad.get_arrays_av(-1)[1]
            windows = ad.window_sizes[-1]

            for j, act in enumerate(act_list_list[-1][:4]):
                print(act, np.nansum(count_var_av[:,j,0])/ count_var_av.shape[0])
                ax0.errorbar(windows, count_var_av[:, j, 0], count_var_av[:, j, 1], fmt='.', color=f'C{j}',
                            label=f'Activity {act:.3f}', elinewidth=1.5, capsize=1.5, capthick=1, markersize=3, alpha=.5)
            ax0.legend()
            fig0.savefig(os.path.join(save_path, 'count_var_av_no_slip.png'), dpi=420, bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
    main()
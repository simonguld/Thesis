# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

import sys
import os
import pickle as pkl
import argparse

from pathlib import Path
from multiprocessing.pool import Pool as Pool

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('sg_article')
plt.rcParams.update({"text.usetex": True,})
plt.rcParams['legend.handlelength'] = 0


from AnalyseCID import AnalyseCID
from utils import *
from utils_plot import *

### FUNCTIONS ----------------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

### MAIN ---------------------------------------------------------------------------------------

def main():   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', type=str2bool, default=False)
    parser.add_argument('--analyze', type=str2bool, default=False)
    parser.add_argument('--plot', type=str2bool, default=False)
    parser.add_argument("--nbits", type=lambda s: [int(x) for x in s.split(',')], \
                        help='Comma-separated list, e.g. --nbits 2,3,4', default='4')
    parser.add_argument('--cg', type=int, default=4)
    args = parser.parse_args()

    extract = args.extract
    calc_time_av = args.analyze
    save_figs = args.plot
    nbits_list = args.nbits
    cg = args.cg

    verbose = True
    cap_cid = True
    nexp = 5

    for nbits in nbits_list:
        print(f'\nProcessing for nbits={nbits}, cg={cg}...')

        output_suffix=f'_nb{nbits}cg{cg}'
        uncertainty_multiplier = 20
        act_exclude_dict = {512: [0.02, 0.0225], 1024: [], 2048: [0.0225]}

        L_list = [512, 1024, 2048]
        figs_save_path = f'data\\nematic\\figs\\{output_suffix[1:]}'
        if not os.path.exists(figs_save_path): os.makedirs(figs_save_path )

        if verbose: print(f'Output suffix: {output_suffix}')

        # Initialize dictionaries to hold results

        act_dict = {}
        conv_dict = {}
        cid_dict = {}
        cid_shuffle_dict = {}
        frac_dict = {}
        cid_time_av_dict = {}
        cid_shuffle_time_av_dict = {}
        frac_time_av_dict = {}

        cid_var_dict = {}
        cid_varper_dict = {}
        div_var_dict = {}
        div_varper_dict = {}

        for LX in L_list:

            base_path = f'Z:\\cid\\na{LX}'
            save_path = f'data\\nematic\\na{LX}'

            if not os.path.exists(save_path):
                os.makedirs(save_path )

            info_dict = {'base_path': base_path,
                        'save_path': save_path,
                        'output_suffix': output_suffix,
                        'act_exclude_list': act_exclude_dict[LX],
                        'LX': LX,
                        'nexp': nexp,}

            if extract:
                conv_list = np.load(os.path.join(save_path, f'conv_list.npy'), allow_pickle=True)
                extract_cid_results(info_dict, verbose=True)
                gen_conv_list(conv_list, output_suffix, save_path)
                if verbose: print(f'Extracted CID results for L={LX} and saved to {save_path}')

            with open(os.path.join(save_path, f'cid_params{output_suffix}.pkl'), 'rb') as f:
                            cid_params = pkl.load(f)

            ncubes = cid_params['ncubes']
            npartitions = cid_params['npartitions']

            data_npz = np.load(os.path.join(save_path, f'cid_data{output_suffix}.npz'), allow_pickle=True)
            cid_arr = data_npz['cid']
            cid_shuffle_arr = data_npz['cid_shuffle']
            cid_frac_arr = data_npz['cid_frac']
            if cap_cid:
                frac_mask = cid_frac_arr[...,0] > 1
                cid_frac_arr[...,0][frac_mask] = 1.0
            act_dict[LX] = data_npz['act_list']
            conv_dict[LX] = np.load(os.path.join(save_path, f'conv_list_cubes{output_suffix}.npy'), allow_pickle=True)

            if calc_time_av:
                cid_time_av, cid_var, cid_var_per_exp = calc_time_avs_ind_samples(cid_arr[...,:,:,0], conv_dict[LX], unc_multiplier=uncertainty_multiplier)
                cid_shuffle_time_av, cid_shuffle_var, cid_shuffle_var_per_exp = calc_time_avs_ind_samples(cid_shuffle_arr[...,0], conv_dict[LX], unc_multiplier=uncertainty_multiplier)
                cid_frac_time_av, cid_frac_var, cid_frac_var_per_exp = calc_time_avs_ind_samples(cid_frac_arr[...,0], conv_dict[LX], unc_multiplier=uncertainty_multiplier)

                np.savez_compressed(os.path.join(save_path, f'cid_time_av{output_suffix}.npz'),
                                    cid_time_av=cid_time_av,
                                    cid_var=cid_var,
                                    cid_var_per_exp=cid_var_per_exp,
                                    cid_shuffle_time_av=cid_shuffle_time_av,
                                    cid_shuffle_var=cid_shuffle_var,
                                    cid_shuffle_var_per_exp=cid_shuffle_var_per_exp,
                                    cid_frac_time_av=cid_frac_time_av,
                                    cid_frac_var=cid_frac_var,
                                    cid_frac_var_per_exp=cid_frac_var_per_exp,
                                    act_list=act_dict[LX],
                                    conv_list=conv_dict[LX]
                                    )
                if verbose: print(f'Saved time-averaged CID results to {os.path.join(save_path, f"cid_time_av{output_suffix}.npz")}')
            else:
                time_av_npz = np.load(os.path.join(save_path, f'cid_time_av{output_suffix}.npz'), allow_pickle=True)
                cid_time_av = time_av_npz['cid_time_av']
                cid_shuffle_time_av = time_av_npz['cid_shuffle_time_av']
                cid_frac_time_av = time_av_npz['cid_frac_time_av']
                cid_var = time_av_npz['cid_var']
                cid_var_per_exp = time_av_npz['cid_var_per_exp']
                cid_frac_var = time_av_npz['cid_frac_var']
                cid_frac_var_per_exp = time_av_npz['cid_frac_var_per_exp']
                
            cid_dict[LX] = cid_arr
            cid_shuffle_dict[LX] = cid_shuffle_arr
            frac_dict[LX] = cid_frac_arr
            cid_time_av_dict[LX] = cid_time_av
            cid_shuffle_time_av_dict[LX] = cid_shuffle_time_av
            frac_time_av_dict[LX] = cid_frac_time_av
            cid_var_dict[LX] = cid_var
            cid_varper_dict[LX] = cid_var_per_exp
            div_var_dict[LX] = cid_frac_var
            div_varper_dict[LX] = cid_frac_var_per_exp

        if save_figs:
            
            # Plot divergence and its derivative with respect to activity
            fig, ax0 = plt.subplots(ncols=2, figsize=(10,4))
            marker_shape = ['d-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 'd-', 'p-', 'H-', '8-', '1-', '2-']
            ax = ax0[0]
            axx = ax0[1]

            plot_div = True
            plot_abs = False

            xlim = (0.016, 0.055)
            ymin, ymax = 0, 0

            for i, LX in enumerate(L_list):
                act_list = act_dict[LX]
                cid_time_av = cid_time_av_dict[LX]
                cid_shuffle_time_av = cid_shuffle_time_av_dict[LX]
                cid_frac_time_av = frac_time_av_dict[LX]

                frac_diff = np.diff(1 - cid_frac_time_av[:, 0]) #
                act_diff = np.array(act_list[1:]) - np.array(act_list[:-1])
                act_diff_tot = np.array(act_list[:-1]) + act_diff/2
                deriv_frac = frac_diff / act_diff #np.diff(frac_av[:, 0]) / np.diff(act_list) #/ ada.LX[N]**2
                deriv_frac_err = np.sqrt(cid_frac_time_av[:, 1][1:]**2 + cid_frac_time_av[:,1][:-1]**2) / act_diff

                yvals = 1 - cid_frac_time_av[:,0] if plot_div else cid_frac_time_av[:,0]
                frac_yvals = np.abs(deriv_frac) if plot_abs else deriv_frac

                ax.errorbar(act_list, yvals, yerr=cid_frac_time_av[:,1], fmt=marker_shape[i], lw=1, elinewidth=1.5, label=f'L={LX}', alpha=.6)
                axx.errorbar(act_diff_tot, frac_yvals, yerr=deriv_frac_err, fmt=marker_shape[i], lw=1, elinewidth=1.3, label=f'L={LX}',alpha=.6)

                ymin = min(ymin, np.nanmin(frac_yvals - deriv_frac_err))
                ymax = max(ymax, np.nanmax(frac_yvals + deriv_frac_err))

            ax.vlines(0.022, 0, np.nanmax(yvals*1.1), color='k', linestyle='--', lw=1, zorder=-5)
            ax.set_ylim(0, np.nanmax(yvals*1.1))

            ylabel = r'Divergence ($\mathcal{D}$)' if plot_div else r'CID/CID$_\mathrm{shuffle}$)'
            ax.set_ylabel(ylabel) #
            ax.legend()

            axx.vlines(0.022, ymin*1.2, ymax*1.2, color='k', linestyle='--', lw=1,)
            axx.hlines(0, 0, .1, color='k', linestyle='-', lw=1,)
            axx.set_ylim(ymin*1.1, ymax*1.1)
            axx.set_ylabel(r'd$\mathcal{D}$ / d$\tilde{\zeta}$')
            axx.legend(ncols=1, loc='upper right')

            for i in range(2):
                ax0[i].set_xlim(xlim)
            #  ax0[i].vlines(0.022, 0, np.nanmax(deriv_cid*1.1), color='k', linestyle='--', lw=1,)
                ax0[i].set_xlabel(r'Activity ($\tilde{\zeta}$)')

            fig.savefig(os.path.join(figs_save_path, f'div_ddiv.pdf') ,bbox_inches='tight', dpi=620, pad_inches=.05)
            plt.close()

            ### Plot cid and its derivative with respect to activity
            fig, ax0 = plt.subplots(ncols=2, figsize=(10,4))
            marker_shape = ['d-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 'd-', 'p-', 'H-', '8-', '1-', '2-']

            xlim = (0.016, 0.055)
            ax = ax0[0]
            axx = ax0[1]

            ymin, ymax = 0, 0

            for i, LX in enumerate(L_list):
                act_list = act_dict[LX]
                cid_time_av = cid_time_av_dict[LX]
                cid_shuffle_time_av = cid_shuffle_time_av_dict[LX]
                cid_frac_time_av = frac_time_av_dict[LX]

                cid_diff = np.diff(cid_time_av[:, 0]) #
                act_diff = np.array(act_list[1:]) - np.array(act_list[:-1])
                act_diff_tot = np.array(act_list[:-1]) + act_diff/2
                deriv_cid = cid_diff / act_diff #np.diff(frac_av[:, 0]) / np.diff(act_list) #/ ada.LX[N]**2
                deriv_cid_err = np.sqrt(cid_time_av[:, 1][1:]**2 + cid_time_av[:,1][:-1]**2) / act_diff

                ymin = min(ymin, np.nanmin(deriv_cid - deriv_cid_err))
                ymax = max(ymax, np.nanmax(deriv_cid + deriv_cid_err))

                ax.errorbar(act_list, cid_time_av[:,0], yerr=cid_time_av[:,1], fmt=marker_shape[i], lw=1, label=f'L={LX}', elinewidth=1.5,  alpha=.6)
                axx.errorbar(act_diff_tot, deriv_cid, yerr=deriv_cid_err, fmt=marker_shape[i], label=f'L={LX}', lw=1, elinewidth=1.5, alpha=.6)

            for i in range(2):
                ax0[i].set_xlim(xlim)
                ax0[i].vlines(0.022, 0, np.nanmax(deriv_cid*1.05), color='k', linestyle='--', lw=1,)
                ax0[i].set_xlabel(r'Activity ($\tilde{\zeta}$)')

            ax.set_ylim(0, 0.6)
            ax.set_ylabel(r'CID')
            axx.set_ylim(0, np.nanmax(ymax)*1.1)
            axx.set_ylabel(r'dCID/d$\tilde{\zeta}$')
            axx.legend(ncols=1, loc='upper right')
        
            fig.savefig(os.path.join(figs_save_path, f'cid_dcid.pdf'), \
                        bbox_inches='tight', dpi=620, pad_inches=.05)
            plt.close()

            ### Plot cid variance and derivative
            fig, ax0 = plt.subplots(ncols=3, figsize=(12,4))
            marker_shape = ['s-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 'd-', 'p-', 'H-', '8-', '1-', '2-']

            plot_cid = True
            plot_abs = True

            xlim = (0.016, 0.055)
            ymin, ymax = 0, 0
            for ax in ax0[:]:
                LX = L_list[ax0.tolist().index(ax)]
                act_list = act_dict[LX]
                conv_list = conv_dict[LX]

                cid_time_av = cid_time_av_dict[LX]
                cid_frac_time_av = frac_time_av_dict[LX]

                act_diff = np.array(act_list[1:]) - np.array(act_list[:-1])
                act_diff_tot = np.array(act_list[:-1]) + act_diff/2

                cid_diff = np.diff(cid_time_av[:, 0]) #
                deriv_cid = cid_diff / act_diff #np.diff(frac_av[:, 0]) / np.diff(act_list) #/ ada.LX[N]**2
                deriv_cid_err = np.sqrt(cid_time_av[:, 1][1:]**2 + cid_time_av[:,1][:-1]**2) / act_diff

                varall, varper = cid_var_dict[LX], cid_varper_dict[LX]
                varper_av = np.nanmean(varper, axis=1)
                varper_sem = np.nanstd(varper, axis=1) / np.sqrt(varper.shape[1])

                dcid_vals = np.abs(deriv_cid / np.nanmax(deriv_cid)) if plot_abs else deriv_cid / np.nanmax(deriv_cid)

                dcid_per = deriv_cid / (0.5 * (cid_time_av[:,0][:-1] + cid_time_av[:,0][1:]))
                dcid_per_vals = np.abs(dcid_per / np.nanmax(dcid_per)) if plot_abs else dcid_per / np.nanmax(dcid_per)

                if plot_abs:
                    ylabel_per = r'$\left|d\textrm{CID}/d\tilde{\zeta}\right| / \left|\textrm{CID} \right|$'
                    ylabel = r'$\left|d\textrm{CID}/d$\tilde{\zeta}\right|$'
                    div_ylabel = r'$\left|d\mathcal{D}/d\tilde{\zeta}\right|$' 

                    ylabel_per = r'$\vert d\textrm{CID}/d\tilde{\zeta}\vert / \vert\textrm{CID} \vert$'
                    ylabel = r'$\vert d\textrm{CID}/d\tilde{\zeta}\vert$'
                else:
                    ylabel_per = r'$\frac{d \textrm{CID}}{d\tilde{\zeta}} / \textrm{CID}$'
                    ylabel = r'$\frac{d \textrm{CID}}{d\tilde{\zeta}}$'
                    div_ylabel = r'$\frac{d\mathcal{D}}{d\tilde{\zeta}}$'

            #    ax.plot(act_list, varall / np.nanmax(varall), label=r'$\mathrm{Var} (CID)$ (all)', marker='d', lw=1, alpha=.5)
                ax.errorbar(act_list, varper_av / np.nanmax(varper_av), yerr=varper_sem / np.nanmax(varper_av), fmt='*', alpha=0.5, capsize=3, label=r'$\mathrm{Var} (CID)$', lw=1)
                ax.plot(act_diff_tot, dcid_vals, marker='v', label=ylabel, lw=1, alpha=.5)
                ax.plot(act_diff_tot, dcid_per_vals, marker='^', label=ylabel_per, lw=1, alpha=.5)

                ax.set_title(f'L={LX}')
                ax.set_xlim(0.018,0.035)
                ax.vlines(0.022,0,1,color='k', linestyle='--',zorder=-5,lw=1)#label='Transition region')
                ax.legend(fontsize=10)

            fig.supxlabel(r'Activity $\tilde{\zeta}$', y=0.08)

            figname = 'cid_fluc.pdf'
            fig.savefig(os.path.join(figs_save_path, figname) ,bbox_inches='tight', dpi=620, pad_inches=.05)
            plt.close()

            ### Plot div variance and derivative
            fig, ax0 = plt.subplots(ncols=3, figsize=(12,4))
            marker_shape = ['s-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 'd-', 'p-', 'H-', '8-', '1-', '2-']

            plot_div = True
            plot_abs = True

            xlim = (0.016, 0.055)
            ymin, ymax = 0, 0
            for ax in ax0[:]:
                LX = L_list[ax0.tolist().index(ax)]
                act_list = act_dict[LX]
                conv_list = conv_dict[LX]


                cid_frac_time_av = frac_time_av_dict[LX]

                act_diff = np.array(act_list[1:]) - np.array(act_list[:-1])
                act_diff_tot = np.array(act_list[:-1]) + act_diff/2

                frac_diff = np.diff(1 - cid_frac_time_av[:, 0])
                deriv_frac = frac_diff / act_diff #np.diff(frac_av[:, 0]) / np.diff(act_list) #/ ada.LX[N]**2
                deriv_frac_err = np.sqrt(cid_frac_time_av[:, 1][1:]**2 + cid_frac_time_av[:,1][:-1]**2) / act_diff
                

                yvals = 1 - cid_frac_time_av[:,0] if plot_div else cid_frac_time_av[:,0]
                frac_yvals = np.abs(deriv_frac) if plot_abs else deriv_frac

                varall, varper = div_var_dict[LX], div_varper_dict[LX]
                varper_av = np.nanmean(varper, axis=1)
                varper_sem = np.nanstd(varper, axis=1) / np.sqrt(varper.shape[1])


                div_vals = np.abs(deriv_frac / np.nanmax(np.abs(deriv_frac))) if plot_abs else deriv_frac / np.nanmax(deriv_frac)

                ax.plot(act_list, varall / np.nanmax(varall), label=r'$\mathrm{Var} (\mathcal{D})$ (all)', marker='d', lw=1, alpha=.5)
                ax.errorbar(act_list, varper_av / np.nanmax(varper_av), yerr=varper_sem / np.nanmax(varper_av), fmt='*', alpha=0.5, capsize=3,\
                            label=r'$\mathrm{Var} (\mathcal{D})$ (per exp)', lw=1)
                ax.plot(act_diff_tot, div_vals , marker='v', label=r'Abs(d$\mathcal{D}$ / d$\tilde{\zeta}$)', lw=1, alpha=.5)
                ax.set_title(f'L={LX}')
                ax.set_xlim(0.018,0.035)
                ax.vlines(0.022,0,1,color='k', linestyle='--',zorder=-5,lw=1)#label='Transition region')
                ax.legend(fontsize=12)

            ymin = min(ymin, np.nanmin(frac_yvals))
            ymax = max(ymax, np.nanmax(frac_yvals))
            fig.supxlabel(r'Activity $\tilde{\zeta}$', y=0.08)

            figname = 'div_fluc.pdf'
            fig.savefig(os.path.join(figs_save_path, figname) ,bbox_inches='tight', dpi=620, pad_inches=.05)
            plt.close()

            ### Plot moments
            moment_dict = {}
            div_moment_dict = {}
            for LX in L_list:
                moment_dict[LX] = calc_moments(cid_dict[LX][...,0], conv_dict[LX],)
                div_moment_dict[LX] = calc_moments(1-frac_dict[LX][...,0], conv_dict[LX],)

            fig, ax = plot_moments(moment_dict, act_dict=act_dict, L_list=L_list[1:], moment_label=r'CID',\
                                    plot_binder=False, \
                                savepath=os.path.join(figs_save_path, f'cid_moments{output_suffix}.pdf'))
            plt.close()
            fig, ax = plot_moments(div_moment_dict, act_dict=act_dict, L_list=L_list[1:], moment_label=r'$\mathcal{D}$',\
                                plot_binder=False, \
                                savepath=os.path.join(figs_save_path, f'div_moments{output_suffix}.pdf'))
            plt.close()
            
if __name__ == '__main__':
    main()
# Author: Simon Guldager Andersen
# Date (latest update): 13-09-2024

### SETUP ------------------------------------------------------------------------------------

## Imports:

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from plot_utils import *


class AnalyseDefectsAll:
    def __init__(self, system_size_list, ):
        self.LX = system_size_list
        self.inputs_paths = [f"data\\na{LX}\\merged_results" for LX in self.LX]
        self.act_list = [list(np.load(os.path.join(path, "activity_list.npy"))) for path in self.inputs_paths]
        self.window_sizes = [list(np.load(os.path.join(path, 'window_sizes.npy'))) for path in self.inputs_paths]
        self.Nactivity = [len(act) for act in self.act_list]

        self.output_path = "data\\na_all"

    def get_av_defects(self, LX = 512, density = True):
        """
        Returns av_defects
        """

        idx = self.LX.index(LX)
        norm = LX**2 if density else 1

        try:
            av_defects = np.load(os.path.join(self.inputs_paths[idx], 'av_defects.npy')) / norm
        except:
            print('Average defects not found. Analyse defects first.')
            return
        return av_defects
    
    def get_susceptibility(self, LX = 512,):
        """
        Returns susceptibility
        """

        idx = self.LX.index(LX)
        try:
            susceptibility = np.load(os.path.join(self.inputs_paths[idx], 'susceptibility.npy'))
        except:
            print('Susceptibility not found. Analyse defects first.')
            return
        return susceptibility

    def get_sfac(self, LX, time_av = True,):
        """
        returns kbins, sfac_av, rad, pcf_av
        """

        idx = self.LX.index(LX)
        prefix = 'time_' if time_av else ''

        try:
            sfac_av = np.load(os.path.join(self.inputs_paths[idx], f'sfac_{prefix}av.npy'))
            pcf_av = np.load(os.path.join(self.inputs_paths[idx], f'pcf_{prefix}av.npy'))
        except:
            print('Structure factor or pcf not found. Analyse defects first.')
            return

        rad = np.load(os.path.join(self.inputs_paths[idx], 'rad.npy'))
        kbins = np.loadtxt(os.path.join(self.inputs_paths[idx], 'kbins.txt'))
        return kbins, sfac_av, rad, pcf_av

    def get_alpha(self, LX,):
        """
        returns time_av_of_fits, fit_params_of_time_av, fit_params_time_av_counts
        """     
        idx = self.LX.index(LX)

        try:
            time_av_of_fits = np.load(os.path.join(self.inputs_paths[idx], f'alpha_list_sfac.npy'))
            fit_params_of_time_av = np.load(os.path.join(self.inputs_paths[idx], f'fit_params_sfac_time_av.npy'))
            fit_params_time_av_counts = np.load(os.path.join(self.inputs_paths[idx], f'fit_params_count.npy'))
        except:
            print('Alpha list not found. Analyse hyperuniformity first.')
            return
        return time_av_of_fits, fit_params_of_time_av, fit_params_time_av_counts

    def plot_av_defects(self, fit_dict = {}, LX_list = None, act_bounds = None, \
                        plot_density = True, figsize=(7,4.5), verbose = False, ax = None,inset_box=None):
        """
        fit_dict: dictionary containing the fit parameters with keys
        'fit_func': fit function
        'fit_string': string of the fit function
        'param_guess': guess for the fit parameters
        """
 
        if fit_dict == {}:
            do_fit = False
        else:
            do_fit = True
            fit_func = fit_dict['fit_func']
            fit_string = fit_dict['fit_string']
            param_guess = fit_dict['param_guess']

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            return_fig = True
        else:
            return_fig = False
        axin = ax.inset_axes(inset_box) if inset_box is not None else None
        marker_shape = ['s', 'o', '^', 'v', 'D', 'P', 'X', 'h', 'd', 'p', 'H', '8', '1', '2']

        LX_list = self.LX if LX_list is None else LX_list

        for i, LX in enumerate(self.LX):
            norm = LX**2 if plot_density else 1

            try:
                av_defects = np.load(os.path.join(self.inputs_paths[i], 'av_defects.npy')) / norm
            except:
                print('Average defects not found. Analyse defects first.')
                return
   
            ax.errorbar(self.act_list[i], av_defects[:, 0], yerr = av_defects[:, 1], fmt = marker_shape[i],\
                         label = f'L = {self.LX[i]}', alpha=.6, elinewidth=1.5, capsize=1.5, capthick=1, markersize = 4, color = f'C{i}' )
            if axin is not None:
                axin.errorbar(self.act_list[i], av_defects[:, 0], yerr = av_defects[:, 1], fmt = marker_shape[i], \
                               elinewidth=1.5, capsize=1.5, alpha=.6, capthick=1, markersize = 4, color = f'C{i}')

            if do_fit:

                act_idx_bounds = [0, None]
                if act_bounds is None:
                    act_idx_bounds[0] = 0
                    act_idx_bounds[1] = None
                else:
                    act_idx_bounds[0] = self.act_list[i].index(act_bounds[0])
                    act_idx_bounds[1] = self.act_list[i].index(act_bounds[1]) + 1
     
                activities = np.array(self.act_list[i][act_idx_bounds[0]:act_idx_bounds[1]])
      
      
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)

                    fit = do_chi2_fit(fit_func, activities, av_defects[act_idx_bounds[0]:act_idx_bounds[1], 0], \
                                av_defects[act_idx_bounds[0]:act_idx_bounds[1], 1]*10, parameter_guesses = param_guess, verbose=verbose)
                    Ndof, chi2, pval = get_statistics_from_fit(fit, len(activities), subtract_1dof_for_binning = False)

                  
                    print(f'For LX = {LX}:')
                    print("Valid minimum: ", fit.fmin.is_valid)
                    print('Params: ', fit.values[:])
                    print(f'Ndof = {Ndof}, chi2 = {chi2:.3f}, pval = {pval:.3f}\n')
                    
                ax.plot(activities, fit_func(activities, *fit.values[:]), '-', color = f'C{i}', label=rf'Fit $L = {LX}$', linewidth = 2)
        
        if do_fit:
            ax.text(0.10, 0.95, rf'Fit = {fit_string}', transform=ax.transAxes, fontsize=16, verticalalignment='top', fontweight='bold')

        if return_fig:
            ax.legend(loc='lower right')
            ax.set_xlabel(r'Activity')
            ax.set_ylabel(r'Av. defect density')
            fig.tight_layout()

            if inset_box is None:
                return fig, ax
            else:
                return fig, ax, axin
        else:
            return ax if inset_box is None else ax, axin
    
    def plot_alpha_mean_sfac(self, act_idx_bounds = None, time_av = True,\
                                alpha=0.6, alpha_inset = 0.6, markersize=4,
                                ax = None, labels=True, inset_box = None):
        

        axin = ax.inset_axes(inset_box) if inset_box is not None else None
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        marker_shape = ['s-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 'd-', 'p-', 'H-', '8-', '1-', '2-']
        marker_shape_inset = ['s-', 'o-', '^-', 'v-', 'D-']

        for i, LX in enumerate(self.LX):
            try:
                if time_av:
                    alpha_list = np.load(os.path.join(self.inputs_paths[i], f'fit_params_sfac_time_av.npy'))[:,[0,2]]
                else:
                    alpha_list = np.load(os.path.join(self.inputs_paths[i], f'alpha_list_sfac.npy'))
            except:
                print('Alpha list not found. Analyse hyperuniformity first.')
                return
            

            act_idx_bounds = [0, None] if act_idx_bounds is None else act_idx_bounds
            act_list = self.act_list[i]
            act_list = np.array(act_list[act_idx_bounds[0]:act_idx_bounds[1]])
            alpha_list = alpha_list[act_idx_bounds[0]:act_idx_bounds[1]]

            mask = (np.isnan(alpha_list[:, 0])) | (alpha_list[:,0] == 0.1)
            alpha_list = alpha_list[~mask]
            act_list = act_list[~mask]
            label = f'L = {self.LX[i]}' if labels else None

            ax.errorbar(act_list, alpha_list[:, 0], alpha_list[:, 1], fmt = marker_shape[i], label = label,\
                        alpha = alpha, elinewidth=1.5, capsize=1.5, capthick=1, markersize = markersize, lw=1, color = f'C{i}')
            if axin is not None:
                axin.errorbar(act_list, alpha_list[:, 0], alpha_list[:, 1], fmt = marker_shape_inset[i], label = f'L = {self.LX[i]}',\
                                alpha = alpha_inset, elinewidth=1.5, capsize=1.5, capthick=1, markersize = markersize, lw=1, color = f'C{i}')

        if ax is None:
            ax.legend()
            ax.set_xlabel(r'$\zeta$')
            ax.set_ylabel(rf'$\langle\alpha \rangle$')
            ax.set_title(rf'Time av. of $\alpha $ vs activity')

            fig.tight_layout()
            return fig, ax
        return ax if axin is None else ax, axin

    def plot_alpha_mean(self, use_density_fit = True, include_fluc=False, include_time_av = False, act_idx_bounds = None,):
        

        suffix = 'dens' if use_density_fit else 'count'

        fig, ax = plt.subplots(figsize=(9, 6))
        marker_shape = ['s-', 'o-', '^-', 'v-', 'D-', 'P-', 'X-', 'h-', 'd-', 'p-', 'H-', '8-', '1-', '2-']

        for i, LX in enumerate(self.LX):
            act_list = self.act_list[i]
            if include_fluc:
                try:
                    alpha_list = np.load(os.path.join(self.inputs_paths[i], f'alpha_list_{suffix}.npy'))
                except:
                    print('Alpha list not found. Analyse hyperuniformity first.')
                    return
            try:
                alpha_list_sfac = np.load(os.path.join(self.inputs_paths[i], f'alpha_list_sfac.npy'))
                if include_time_av:
                    alpha_list_sfac_time_av = np.load(os.path.join(self.inputs_paths[i], f'fit_params_sfac_time_av.npy'))[:,[0,2]]
            except:
                print('Alpha list not found. Analyse sfac first.')
                return
            
            act_idx_bounds = [0, None] if act_idx_bounds is None else act_idx_bounds
            act_list = np.array(act_list[act_idx_bounds[0]:act_idx_bounds[1]])

            alpha_list_sfac = alpha_list_sfac[act_idx_bounds[0]:act_idx_bounds[1]]
            alpha_list_sfac_time_av = alpha_list_sfac_time_av[act_idx_bounds[0]:act_idx_bounds[1]]

            if include_fluc:
                alpha_list = alpha_list[act_idx_bounds[0]:act_idx_bounds[1]]
                ax.errorbar(act_list, alpha_list[:, 0], alpha_list[:, 1], fmt = marker_shape[0], \
                        label = f'L = {self.LX[i]}', alpha = .6, elinewidth=1.5, capsize=1.5, capthick=1, markersize = 4, color = f'C{i}')
            if include_time_av:
                ax.errorbar(act_list, alpha_list_sfac_time_av[:, 0], alpha_list_sfac_time_av[:, 1], fmt = marker_shape[2], \
                                alpha = .6, elinewidth=1.5, capsize=1.5, capthick=1, markersize = 4, color = f'C{i}')
            
            ax.errorbar(act_list, alpha_list_sfac[:, 0], alpha_list_sfac[:, 1], fmt = marker_shape[1], \
                            alpha = .6, elinewidth=1.5, capsize=1.5, capthick=1, markersize = 4, color = f'C{i}', label = f'L = {self.LX[i]}')
               
        ax.legend()
        ax.set_xlabel(r'$\zeta$')
        ax.set_ylabel(rf'$\langle\alpha \rangle$')
        ax.set_title(rf'Time av. of $\alpha $ vs activity')
        fig.tight_layout()
        return fig, ax

    def plot_susceptibility(self, act_max_list = [], window_idx_bounds_list = None, \
                            act_bounds = None, verbose = False, save = False):

        fig, ax = plt.subplots(figsize=(9, 6))
        marker_shape = ['s', 'o', '^', 'v', 'D', 'P', 'X', 'h', 'd', 'p', 'H', '8', '1', '2']

        for i, LX in enumerate(self.LX):
            try:
                xi = np.load(os.path.join(self.inputs_paths[i], 'susceptibility.npy'))
            except:
                print('Susceptibilities not found. Analyse defects first.')
                return
            
            act_list = self.act_list[i]
            window_sizes = self.window_sizes[i]

            if window_idx_bounds_list is None:
                window_idx_bounds = [0, len(window_sizes)]
            else:
                window_idx_bounds = window_idx_bounds_list[i]
            if act_bounds is None:
                act_idx_bounds = [0, len(act_list)]
            else:
                act_idx_bounds = [act_list.index(max(act_bounds[0], act_list[0])), act_list.index(min(act_bounds[1], act_list[-2])) + 1]

            window_sizes = window_sizes[window_idx_bounds[0]:window_idx_bounds[1]]
            act_list = act_list[act_idx_bounds[0]:act_idx_bounds[1]]
            xi = xi[window_idx_bounds[0]:window_idx_bounds[1], act_idx_bounds[0]:act_idx_bounds[1], :]

            bin_count = np.bincount(np.argmax(xi[:, :, 0], axis = -1))
            if verbose:
                try:
                    for k, act in enumerate(act_list):
                        if bin_count[k] > 0:
                            print("No. of windows with max susceptibility at activity {:.3f}: {:.0f}".format(act, bin_count[k]))
                except:
                    pass

            act_max = act_list[np.argmax(bin_count)] if len(act_max_list) == 0 else act_max_list[i]
            act_max_idx = act_list.index(act_max)
            xi_normed = np.zeros((len(window_sizes), len(act_list), 2))
    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for j, window in enumerate(window_sizes):
                    norm = xi[j, act_max_idx, 0]
                    xi_normed[j, :, 0] = xi[j, :, 0] / norm
                    xi_normed[j, :, 1] = xi[j, :, 1] / norm
        
                xi_mean = np.nanmean(xi_normed[:,:,0], axis = 0)
                xi_std = np.nanstd(xi_normed[:,:,1], axis = 0) / np.sqrt(len(window_sizes))

            ax.errorbar(act_list, xi_mean, xi_std, label=f'L = {self.LX[i]}', fmt='.-', \
                            capsize=2, capthick=1, elinewidth=1, markeredgewidth=2, alpha = 1, markersize=4, color=f'C{i}') 
            
        ax.legend(loc='lower right', ncol=2)
        ax.set_xlabel(rf'$\zeta$ (activity)')
        ax.set_ylabel(r'$\chi  / \chi_{max}$')


        fig.suptitle(rf'Susceptibility of defect density vs activity', fontsize=20)
        fig.tight_layout()

        if save:
            fig.savefig(os.path.join(self.output_path, 'susceptibility.png'), dpi = 420, pad_inches=0.25)
        return fig, ax

    def plot_sfac_per_activity(self, LX, Npoints_to_fit = 5, act_list = None, 
                               scaling_exp_list = [], scaling_label_list = [], 
                               ax = None, plot_poisson = True, marker_list=[]):

        """
        returns fit_params_time_av
        """
        
        idx = self.LX.index(LX)
        input_path = self.inputs_paths[idx]
        act_list = self.act_list[idx] if act_list is None else act_list

        try:
            kbins = np.loadtxt(os.path.join(input_path, 'kbins.txt'))
            sfac_av = np.load(os.path.join(input_path, f'sfac_time_av.npy'))
        except:
            print('Time-averaged structure factor or pcf not found. Analyse defects first.')
            return

        def fit_func(x, alpha, beta):
                    return beta + alpha * x
        fit_string = rf'$y = \beta + \alpha |k|$'
        Nparams = 2

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
            subplot = False
        else:
            subplot = True

        k_begin_lines_idx = Npoints_to_fit - 1
        kmin, kmax = np.nanmin(kbins), np.nanmax(kbins)
        x = kbins[:Npoints_to_fit]
        
        colors = ['black', 'red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'grey', 'olive', 'lime']

        for i, act in enumerate(act_list):
            act_idx = self.act_list[idx].index(act)
            ax.errorbar(kbins, sfac_av[:, act_idx, 0], yerr = sfac_av[:, act_idx, 1],\
                         fmt = 's' if len(marker_list)==0  else marker_list[i], \
                    alpha = .6, color = colors[i], markersize = 5, label = rf'$\zeta =$ {act}')
            
            if len(scaling_exp_list) > 0:
                scaling_exp = scaling_exp_list[i]
                label = scaling_label_list[i] if len(scaling_label_list) > 0 else None
                ax.plot(x, sfac_av[k_begin_lines_idx, act_idx, 0] * x**scaling_exp / x[-1]**scaling_exp, '--', label = label, alpha=0.5,) 

        if plot_poisson:
            ax.hlines(1, 0, kmax+0.2, label=r'Poisson', linestyles='dashed', colors='k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Wavenumber')
        ax.set_ylabel(r'Time av. structure factor')

        if LX == 2048:
            ax.set_xticks([kmin, 0.01, 0.1, kmax], [np.round(kmin,3), 0.01, 0.1, np.round(kmax,1)])
            ax.set_yticks([0.3, 0.4, 0.6, 1, 5], [0.3, 0.4, 0.6, 1, 5])
        else:
            ax.set_xticks([kmin, 0.1, kmax], [np.round(kmin,3), 0.1, np.round(kmax,1)])
            ax.set_yticks([0.3, 0.4, 0.6, 1,], [0.3, 0.4, 0.6, 1,])
 
        ax.legend(ncol=3)
        ax.set_xlabel(r'Norm of wavenumber ($k$)')
        ax.set_ylabel(r'Structure factor ($\overline{S}$)')

        if not subplot:
            fig.tight_layout()
        if subplot:
            return ax
        else:
            return fig, ax


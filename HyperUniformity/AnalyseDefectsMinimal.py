# Author: Simon Guldager Andersen
# Date (latest update): February 2024

### SETUP ------------------------------------------------------------------------------------

## Imports:

import os
import warnings
import pickle as pkl
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.metrics import ndcg_score

from utils import *
from plot_utils import *


class AnalyseDefectsMinimal:
    def __init__(self, input_list, output_path = 'data'):
        """"
        input_list: list of dictionaries. Each dictionary contains the following keys:
        "path": path to the defect folder
        "suffix": suffix of the defect folder
        "priority": int: defect priority. The higher the more important and will overwrite the lower priority results
        "LX": int: system size in x direction
        "Nframes": int: number of frames in the defect folder

        """
        self.Ndata = len(input_list)
        self.input_list = input_list
        self.input_paths = [input["path"] for input in input_list]
        self.suffixes = [input["suffix"] for input in input_list]
        self.priorities = [input["priority"] for input in input_list]
        self.LX = [int(input["LX"]) for input in input_list]
        self.Nframes = [int(input["Nframes"]) for input in input_list]
        self.Ninfo = [int(input["Ninfo"]) for input in input_list]

        self.output_main_path = output_path 
        self.output_paths = [os.path.join(self.output_main_path, self.suffixes[i]) for i in range(self.Ndata)]
        self.Nexp = []
        self.Nactivity = []
        self.act_list = []
        self.act_dir_list = []
        self.conv_list = []
        self.conv_list_err = []
        
        self.Ndata = len(input_list)

        for i, input in enumerate(self.input_paths):
            Nsubdir = 1
            act = []
            act_dir = []
            self.Nactivity.append(len(os.listdir(input)))
            for j, subdir in enumerate(os.listdir(input)):
        
                subdir_full = os.path.join(input, subdir)
                act.append(np.round(float(subdir_full.split('_')[-1]),4))
                act_dir.append(subdir_full)
                Nsubdir = max(Nsubdir, len(os.listdir(os.path.join(input, subdir))))

                if not os.path.isdir(self.output_paths[i]):
                    os.makedirs(self.output_paths[i])
       
            act, act_dir = zip(*sorted(zip(act, act_dir)))
            # save the activity list
            np.savetxt(os.path.join(self.output_paths[i], 'activity_list.txt'), act)

            self.act_list.append(act)
            self.act_dir_list.append(act_dir)
            self.Nexp.append(Nsubdir)

        for i, output in enumerate(self.output_paths):
            if not os.path.exists(output):
                os.makedirs(output)
            if not os.path.exists(os.path.join(output, 'figs')):
                os.makedirs(os.path.join(output, 'figs'))   

            # load the convergence list if it exists
            try:
                self.conv_list.append(np.loadtxt(os.path.join(output, 'conv_list.txt')).astype(int))
                self.conv_list_err.append(np.loadtxt(os.path.join(output, 'conv_list_err.txt')).astype(int))
            except:
                self.conv_list.append([0] * self.Nactivity[i])
                self.conv_list_err.append([0] * self.Nactivity[i])

    def __get_outpath_path(self, Ndataset = 0, use_merged = False):

        if use_merged:
            output_path = os.path.join(self.output_main_path, 'merged_results')
            N = np.argmin(self.priorities)

            if not os.path.isdir(output_path):
                print(f'Merged results not found. Run merge_results first.')
                return
        else:
            output_path = self.output_paths[Ndataset]
            N = Ndataset
        return output_path, N

    def __calc_av_over_exp(self, data_arr, Ndataset = 0, return_arr = False, save_name = None, save = True,):
        """
        data_arr: array of shape (Nframes, Nactivity, Nexp)
        """
        if save:
            if save_name is None:
                print('No save name given. Data will not be saved.')
                return
            
        av = np.expand_dims(np.nanmean(data_arr, axis = -1), axis = -1)
        std = np.expand_dims(np.nanstd(data_arr, axis = -1) / np.sqrt(data_arr.shape[-1]), axis = -1)
        output_arr = np.concatenate([av, std], axis = -1)

        if save:
            np.save(os.path.join(self.output_paths[Ndataset], save_name + '.npy'), output_arr)
        return output_arr if return_arr else None

    def __plot_defects_per_activity(self, activity, Ndataset = 0, stationarity_dict = {}):
        
        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged = False)

        act_idx = self.act_list[Ndataset].index(activity)   
        Nframes = self.Nframes[Ndataset]

        try:
            defect_arr_av = self.get_arrays_av(Ndataset = Ndataset)[0] 
        except:
            print('Defect array not found. Analyse defects first.')
            return

        title = 'Defect count'
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.errorbar(np.arange(0, Nframes * self.Ninfo[Ndataset], self.Ninfo[Ndataset]), \
                           defect_arr_av[:, act_idx, 0], defect_arr_av[:, act_idx, 1], fmt='.', \
                            alpha = 0.15, markersize=9, label='Activity = {}'.format(activity),) 

        if stationarity_dict != {}:
                x = est_stationarity(defect_arr_av[:, act_idx, 0], **stationarity_dict)[0]
        else:
            x = self.conv_list[Ndataset][act_idx]
            
        x *= self.Ninfo[Ndataset]
        print(x)
        if x > 0:
            ax.axvline(x, color='black', linestyle='--', alpha=0.5)

     #   xticks = np.round(np.linspace(0, Nframes * self.Ninfo[Ndataset], 20)).astype('int'), \
      #         xticklabels = np.round(np.linspace(0, Nframes * self.Ninfo[Ndataset], 20)).astype('int'),
        ax.grid()
        ax.set(xlabel = 'Time step', ylabel = f'{title}', 
               title = f'{title} for activity = {activity}',
               ylim = (0, np.max(defect_arr_av[:, act_idx, 0]) * 1.5))

        fig.tight_layout()
        return

    def update_conv_list(self, Ndataset_list = None, stationarity_dict = {}):
        if Ndataset_list is None:
            Ndataset_list = range(self.Ndata)
        
        for i in Ndataset_list:
            act_list = self.act_list[i]
      
            for j in range(self.Nactivity[i]):
                self.__plot_defects_per_activity(activity = act_list[j], Ndataset = i, stationarity_dict = stationarity_dict)
                plt.show()
                self.conv_list[i][j] = int(input(f'Enter the first frame to use for activity {self.act_list[i][j]}: '))
                self.conv_list_err[i][j] = int(input(f'Enter the error for the first frame to use for activity {self.act_list[i][j]}: '))

            # save the convergence list
            np.savetxt(os.path.join(self.output_paths[i], 'conv_list.txt'), self.conv_list[i])
            np.savetxt(os.path.join(self.output_paths[i], 'conv_list_err.txt'), self.conv_list_err[i])
        return

    def get_arrays_full(self, Ndataset = 0,):
        """
        returns defect_arr
        """
        output_path = self.output_paths[Ndataset]
        try:
            defect_arr = np.load(os.path.join(output_path, 'defect_arr.npy'))  
        except:
            print('Arrays not found. Analyse defects first.')
            return
        return defect_arr
      
    def get_arrays_av(self, Ndataset = 0, use_merged = False):
        """
   
        returns defect_arr_av, av_defects
        """

        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        try:
            defect_arr_av = np.load(os.path.join(output_path, 'defect_arr_av.npy'))
            av_defects = np.load(os.path.join(output_path, 'av_defects.npy'))
        except:
            print('Arrays not found. Analyse defects first.')
            return
        
        else:
            return defect_arr_av, av_defects

    def extract_results(self, save = True):
        """
        Analyse the defects for all the input folders
        """
        for N in range(self.Ndata):

            defect_arr = np.nan * np.zeros((self.Nframes[N], self.Nactivity[N], self.Nexp[N]))
      
            print('Analyse defects for input folder {}'.format(self.input_paths[N]))
            for i, (act, act_dir) in enumerate(zip(self.act_list[N], self.act_dir_list[N])):
                exp_list = []
                exp_dir_list = []

                for file in os.listdir(act_dir):
                    exp_count = file.split('_')[-1]
                    exp_list.append(int(exp_count))
                    exp_dir_list.append(os.path.join(act_dir, file))

                # sort the activity list and the activity directory list
                exp_list, exp_dir_list = zip(*sorted(zip(exp_list, exp_dir_list)))

                for j, (exp, exp_dir) in enumerate(zip(exp_list, exp_dir_list)):
                    with open(os.path.join(exp_dir, 'defect_positions.pkl'), 'rb') as f:
                        defect_list = pkl.load(f)
                    defect_arr[:, i, j] = np.array([len(defect) for defect in defect_list])[-self.Nframes[N]:]
                  #  defect_arr[:, i, j] = np.loadtxt(os.path.join(exp_dir, 'Ndefects_act{}_exp{}.txt'.format(act, exp)))[-self.Nframes[N]:]
            if save:
                np.save(os.path.join(self.output_paths[N], 'defect_arr.npy'), defect_arr)
        return
    
    def analyze_defects(self, Ndataset_list = None, save = True,):

        Ndataset_list = range(self.Ndata) if Ndataset_list is None else Ndataset_list

        for N in Ndataset_list:
            while True:
                try:
                    defect_arr = self.get_arrays_full(N)
                    break
                except:
                    print('Defect array not found. They will be extracted now using normalize = True')
                    self.extract_results(save = True,)

            if len(np.unique(self.conv_list[N])) == 1:
                print(f'NB: All simulations are set to converge at the first frame for dataset {N}. To change this, call update_conv_list.\n')

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.__calc_av_over_exp(defect_arr, N, save_name = 'defect_arr_av', save = save)
          
       
                av_defects = np.zeros((self.Nactivity[N], 2))        
                for i, act in enumerate(self.act_list[N]):
                    av_defects[i, 0] = np.nanmean(defect_arr[self.conv_list[N][i]:, i, :])
                    av_defects[i, 1] = np.nanstd(defect_arr[self.conv_list[N][i]:, i, :]) / np.sqrt(defect_arr[self.conv_list[N][i]:, i, :].size)     
                if save:
                    np.save(os.path.join(self.output_paths[N], 'av_defects.npy'), av_defects)
        return

    def merge_results(self, save_path = None, save = True):

 
        if save_path is None:
            save_path = os.path.join(self.output_main_path, 'merged_results')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        Nbase = np.argmin(self.priorities)
        Nbase_frames = self.Nframes[Nbase]

        try:
            defect_arr_av, av_defects = self.get_arrays_av(Nbase, return_av_counts = True)
        except:
            print('Base dataset not found. Analyse defects first.')
            return
        
        # overwrite the activities with the ones from the other datasets according to self.priorities
        _, Nsorted = zip(*sorted(zip(self.priorities, range(self.Ndata))))

        for N in Nsorted[1:]:

            act_idx_list = []
            for act in self.act_list[N]:
                act_idx_list.append(self.act_list[Nbase].index(act))

            defect_arr_av[:, act_idx_list, :] = np.load(os.path.join(self.output_paths[N], 'defect_arr_av.npy'))[-Nbase_frames:]
            av_defects[act_idx_list] = np.load(os.path.join(self.output_paths[N], 'av_defects.npy'))
            
        if save:
            np.save(os.path.join(save_path, 'activity_list.npy'), self.act_list[Nbase])
            np.save(os.path.join(save_path, 'defect_arr_av.npy'), defect_arr_av)
            np.save(os.path.join(save_path, 'av_defects.npy'), av_defects)
        return
    
    def plot_av_defects(self, Ndataset = 0, fit_dict = {}, plot_density = True, verbose = False, use_merged = False):
        """
        fit_dict: dictionary containing the fit parameters with keys
        'fit_func': fit function
        'fit_string': string of the fit function
        'lower_act_index': index of the activity list to start the fit
        'param_guess': guess for the fit parameters
        """

        _, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        if fit_dict == {}:
            do_fit = False
        else:
            do_fit = True
            fit_func = fit_dict['fit_func']
            fit_string = fit_dict['fit_string']
            lower_act_index = fit_dict['lower_act_index']
            param_guess = fit_dict['param_guess']

        norm = self.LX[Ndataset] ** 2 if plot_density else 1

        try:
            av_defects = self.get_arrays_av(Ndataset)[-1] / norm
        except:
            print('Average defects not found. Analyse defects first.')
            return
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar(self.act_list[Ndataset], av_defects[:, 0], yerr = av_defects[:, 1], fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1, markersize = 4)
        
        ax.set_xlabel(r'$\zeta$')
        ax.set_ylabel(r'$\langle \rho \rangle$')
        ax.set_title('Average defect density vs. activity')

        if do_fit:
            activities = np.array(self.act_list[Ndataset][lower_act_index:])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)

                fit = do_chi2_fit(fit_func, activities, av_defects[lower_act_index:, 0], \
                              av_defects[lower_act_index:, 1], parameter_guesses = param_guess, verbose=verbose)
                d = generate_dictionary(fit, len(activities), chi2_suffix = None)
                text = nice_string_output(d, extra_spacing=4, decimals=3)
                add_text_to_ax(0.02, 0.96, text, ax, fontsize=14)

            ax.plot(activities, fit_func(activities, *fit.values[:]), 'r--', label=rf'Fit: {fit_string}')
            ax.legend(loc='lower right')

        fig.tight_layout()
        return fig, ax
        
    def plot_defects_per_activity(self, Ndataset = 0, Nfirst_frame = 0, act_idx_bounds = None,\
                                   update_conv = False, estimate_stationarity = False, stationarity_dict = {}, plot_density = False, use_merged = False, save = False):
        
        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        if act_idx_bounds is None:
            act_idx_bounds = [0, len(self.act_list[Ndataset])]

        activities = self.act_list[Ndataset][act_idx_bounds[0]:act_idx_bounds[1]]
        norm = self.LX[Ndataset] ** 2 if plot_density else 1
        Nframes = self.Nframes[Ndataset] - Nfirst_frame

        try:
            defect_arr_av = self.get_arrays_av(Ndataset = Ndataset)[0] / norm
        except:
            print('Defect array not found. Analyse defects first.')
            return

        ncols = 1 if update_conv else 4
        nrows = int(np.ceil(len(activities) / ncols))
        title = 'Defect density' if plot_density else 'Defect count'
        height = nrows * 3
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(12,height))
        ax = ax.flatten()  

        for i, act in enumerate(activities):
            act_idx = self.act_list[Ndataset].index(act)
            ax[i].errorbar(np.arange(0, Nframes * self.Ninfo[Ndataset], self.Ninfo[Ndataset]), \
                           defect_arr_av[:, act_idx, 0], defect_arr_av[:, act_idx, 1], fmt='.', \
                            alpha = 0.15, markersize=9, label='Activity = {}'.format(act),) 
            ax[i].text(0.6, 0.2, rf'$\zeta$ = {act}', transform=ax[i].transAxes, fontsize=14, verticalalignment='top')

            
            if estimate_stationarity and stationarity_dict != {}:
                x = est_stationarity(defect_arr_av[:, act_idx, 0], **stationarity_dict)[0] * self.Ninfo[Ndataset]
            elif estimate_stationarity and stationarity_dict == {}:
                print('No stationarity parameters given. Stationarity will not be estimated.')
                x = self.conv_list[Ndataset][act_idx] - Nfirst_frame * self.Ninfo[Ndataset]
            else:
                x = self.conv_list[Ndataset][act_idx] - Nfirst_frame * self.Ninfo[Ndataset]
            print(x)
            if x > 0:
                ax[i].axvline(x,  color='black', linestyle='--', alpha=0.5)
            ax[i].set_ylim(0, np.max(defect_arr_av[:, act_idx, 0]) * 1.5)

        fig.suptitle(f'{title} for different activities (L = {self.LX[Ndataset]})' , fontsize=22, y = 1)
        fig.supxlabel('Time step', fontsize=20, y = 0)
        fig.supylabel(f'{title}', fontsize=20, x=0)
        fig.tight_layout()

        if save:
            if not os.path.isdir(os.path.join(output_path, 'figs')):
                os.makedirs(os.path.join(output_path, 'figs'))
            fig.savefig(os.path.join(output_path, f'figs\\defects_per_activity.png'), dpi = 420, pad_inches=0.15)

        plt.show()
        return fig, ax
    
    def plot_defects_per_exp(self, Ndataset = 0, act_idx_bounds = None, plot_density = False):

        try:
            act_idx_bounds = [0, len(self.act_list[Ndataset])] if act_idx_bounds is None else act_idx_bounds
            activities = self.act_list[Ndataset][act_idx_bounds[0]:act_idx_bounds[1]]
            norm = self.LX[Ndataset] ** 2 if plot_density else 1
            try:
                defect_arr = self.get_arrays_full(Ndataset = Ndataset) / norm
            except:
                print('Defect array not found. Analyse defects first.')
                return

            ncols = 4
            nrows = int(np.ceil(self.Nexp[Ndataset] / ncols))
            height = nrows * 3
            norm = self.LX[Ndataset] ** 2 if plot_density else 1
            title = 'Defect density' if plot_density else 'Defect count'

            for i, act in enumerate(activities):
                fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(16, height))
                ax = ax.flatten()  
                defect_arr_act = (defect_arr[:, i + act_idx_bounds[0], :] / norm).astype(float)
                mini, maxi = np.nanmin(defect_arr_act) * 0.5, np.nanmax(defect_arr_act) * 1.3

                for j in np.arange(self.Nexp[Ndataset]):
                    ax[j].plot(np.arange(self.Nframes[Ndataset]), defect_arr_act[:, j], '.', label='Exp = {}'.format(j), alpha = 0.5)
                    ax[j].legend()  
                    ax[j].set_ylim(mini, maxi)

                fig.suptitle(f'{title} for activity = {act}' , fontsize=18)
                fig.supxlabel('Time step', fontsize=18)
                fig.supylabel(f'{title}', fontsize=18)
                fig.tight_layout()
                plt.show()
        except:
            raise KeyboardInterrupt



def gen_analysis_dict(LL, mode):

    dshort = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_analysis{LL}_LL0.05', \
              suffix = "short", priority = -1, LX = LL, Nframes = 181)
    dlong = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_analysis{LL}_LL0.05_long', \
                suffix = "long", priority = 0, LX = LL, Nframes = 400)
    dvery_long = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_analysis{LL}_LL0.05_very_long',\
                    suffix = "very_long", priority = 3, LX = LL, Nframes = 1500)
    dvery_long2 = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_analysis{LL}_LL0.05_very_long_v2',\
                    suffix = "very_long2", priority = 2, LX = LL, Nframes = 1500)

    if mode == 'all':
        if LL == 2048:
            defect_list = [dshort, dlong]
        else:
            defect_list = [dshort, dlong, dvery_long, dvery_long2] if LL in [256, 512] else [dshort, dlong, dvery_long]
    else:
        defect_list = [dshort]
    
    return defect_list



def main2():
    do_extraction = False
    do_basic_analysis = True
    do_hyperuniformity_analysis = True
    do_merge = True

    system_size_list = [256, 512, 1024, 2048]
    mode = 'all' # 'all' or 'short'

    # order parameter parameters
    shift = True
    shift_by_act = 0.022
    Nscale = True

    # hyperuniformity parameters
    act_idx_bounds=[0,None]
    Npoints_to_fit = 5
    Nbounds = [3,7]

    dens_fluc_dict = dict(fit_densities = True, act_idx_bounds = [0, None], weighted_mean = False, window_idx_bounds = [30 - Npoints_to_fit, None])
    sfac_dict = dict(Npoints_bounds = Nbounds, act_idx_bounds = act_idx_bounds,)

    
    for LL in system_size_list:
        print('Starting analysis for L =', LL)
        time0 = time.time()
        output_path = f'data\\nematic_analysis{LL}_LL0.05'
        
        defect_list = gen_analysis_dict(LL, mode)
        ad = AnalyseDefects(defect_list, output_path=output_path)

        if do_extraction:
            ad.extract_results()
        if do_basic_analysis:
            if do_hyperuniformity_analysis:
                ad.analyze_defects(dens_fluc_dict=dens_fluc_dict, sfac_dict=sfac_dict)
            else:
                ad.analyze_defects()

            # find density at shift_by_act
            av_def_merged = ad.get_arrays_av(use_merged = True)[-1]
            shift_by_def = av_def_merged[ad.act_list[0].index(shift_by_act)][0]

            order_param_function = lambda def_arr, av_def, LX: order_param_func(def_arr, av_def, LX, shift_by_def = shift_by_def, shift = shift)
            sus_binder_dict = dict(Nscale = Nscale, order_param_func = order_param_function)

            ad.analyze_defects(sus_binder_dict=sus_binder_dict)

        if do_merge:
            ad.merge_results()

        print(f'Analysis for L = {LL} done in {time.time() - time0:.2f} s.\n')

if __name__ == "__main__":
    main()



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

from utils import *
from plot_utils import *

### TODO

# go thorugh implementation so far and check for errors
# imp temp corr for def_arr + store
# incorp uncertainty into temp corr
# take into account non-conv eg tau=2 
# expand to sfac and pcf
# incorop in merge etc.
# how to account for the fundamental problem of time series length dependence?


class AnalyseDefects:
    def __init__(self, input_list, output_path = 'data', count_suffix = ''):
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
        self.count_suffix = count_suffix

        self.output_main_path = output_path 
        self.output_paths = [os.path.join(self.output_main_path, self.suffixes[i]) for i in range(self.Ndata)]
        self.Nexp = []
        self.Nactivity = []
        self.act_list = []
        self.act_dir_list = []
        self.window_sizes = []
        self.conv_list = []
        
        self.Ndata = len(input_list)

        for i, input in enumerate(self.input_paths):
            act_dir = []
            if not os.path.isdir(input):
                try:     
                    act = np.loadtxt(os.path.join(self.output_paths[i], 'activity_list.txt'))
                    windows = np.loadtxt(os.path.join(self.output_paths[i], 'window_sizes.txt'))
                    kbins = np.loadtxt(os.path.join(self.output_paths[i], 'kbins.txt'))
                    rad = np.load(os.path.join(self.output_paths[i], 'rad.npy'))
                    self.Nactivity.append(len(act))
                    if self.LX[i] == 2048 and i == 1:
                        Nsubdir = 5
                    else:
                        Nsubdir = 10
                except:
                    print(f'No input folder found for dataset {i}.')
                    continue
            else:
                Nsubdir = 1
                act = []
                self.Nactivity.append(len(os.listdir(input)))

                for _, subdir in enumerate(os.listdir(input)):
            
                    subdir_full = os.path.join(input, subdir)
                    act.append(np.round(float(subdir_full.split('_')[-1]),4))
                    act_dir.append(subdir_full)
                    Nsubdir = max(Nsubdir, len(os.listdir(os.path.join(input, subdir))))

                    if not os.path.isdir(self.output_paths[i]):
                        os.makedirs(self.output_paths[i])

                    if not os.path.isfile(os.path.join(self.output_paths[i], 'window_sizes.txt')):
                        subsubdir = os.path.join(subdir_full, os.listdir(subdir_full)[0])
                        dir_windows = os.path.join(subsubdir, 'window_sizes.txt')
                        windows = np.loadtxt(dir_windows)   
                        np.savetxt(os.path.join(self.output_paths[i], 'window_sizes.txt'), windows)
                    else:
                        windows = np.loadtxt(os.path.join(self.output_paths[i], 'window_sizes.txt'))
                    self.window_sizes.append(windows)
                        
                    if not os.path.isfile(os.path.join(self.output_paths[i], 'kbins.txt')) or not os.path.isfile(os.path.join(self.output_paths[i], 'rad.npy')):
                        subsubdir = os.path.join(subdir_full, os.listdir(subdir_full)[0])
                        dir_kbins = os.path.join(subsubdir, 'kbins.txt')
                        dir_rad = os.path.join(subsubdir, 'rad.txt')

                        # save the kbins and rad if they exist
                        if os.path.isfile(dir_kbins):
                            kbins = np.loadtxt(dir_kbins)
                            np.savetxt(os.path.join(self.output_paths[i], 'kbins.txt'), kbins)
                        if os.path.isfile(dir_rad):
                            rad = np.loadtxt(dir_rad)
                            np.save(os.path.join(self.output_paths[i], 'rad.npy'), rad)
        
                act, act_dir = zip(*sorted(zip(act, act_dir)))
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
            except:
                self.conv_list.append([0] * self.Nactivity[i])

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

    def __calc_av_over_exp(self, data_arr, Ndataset = 0, ddof = 1, return_arr = False, save_name = None, save = True,):
        """
        data_arr: array of shape (Nframes, Nactivity, Nexp)
        """
        if save:
            if save_name is None:
                print('No save name given. Data will not be saved.')
                return
            
        av = np.expand_dims(np.nanmean(data_arr, axis = -1), axis = -1)
        std = np.expand_dims(np.nanstd(data_arr, ddof = ddof, axis = -1) / np.sqrt(data_arr.shape[-1]), axis = -1)
        output_arr = np.concatenate([av, std], axis = -1)
        if save:
            np.save(os.path.join(self.output_paths[Ndataset], save_name + '.npy'), output_arr)
        return output_arr if return_arr else None
 
    def __calc_time_av(self, Ndataset, data_arr, temp_corr_arr, \
                       temp_corr_simple = True, ddof = 1, save_name = None,):
        """
        data_arr must have shape (Nframes, Nsomething, Nact, Nexp)
        returns an array of shape (Nsomething, Nact, 2)
        """

        Nframes, Nsomething, Nact, Nexp = data_arr.shape
        N = Ndataset
        time_av = np.nan * np.zeros((Nsomething, Nact, 2))
     
        for i in range(Nact):
            ff_idx = self.conv_list[N][i]

            Nsamples = (Nframes - ff_idx) * np.ones((Nsomething, Nexp)) - np.nansum(np.isnan(data_arr[ff_idx:, :, i, :]),axis=(0))
            Nind_samples = np.nansum(Nsamples / temp_corr_arr[1 if temp_corr_simple else 0, :, i, :,], axis = -1)
        
            time_av[:, i, 0]  = np.nanmean(data_arr[self.conv_list[N][i]:, :, i, :], axis = (0, -1))
            time_av[:, i, 1] = np.nanstd(data_arr[self.conv_list[N][i]:, :, i, :], axis = (0, -1), ddof = ddof) / np.sqrt(Nind_samples)
        
        if save_name is not None:
            np.save(os.path.join(self.output_paths[N], save_name + '.npy'), time_av)
            return
        else:
            return time_av

    def __calc_sfac_pcf(self, Ndataset = 0, acf_dict = {}, temp_corr_simple = True, ddof = 1, calculate_pcf = True,):
        
        N = Ndataset
        if not os.path.isfile(os.path.join(self.output_paths[N], 'sfac.npz')):
            print('Structure factor not found. Extract results first.')
            return
        
        sfac_path = os.path.join(self.output_paths[N], 'sfac.npz')
        sfac_npz = np.load(sfac_path, 'sfac.npz', allow_pickle = True)
        sfac, sfac_err = sfac_npz['sfac'], sfac_npz['sfac_err']
        sfac_dict = dict(sfac_npz)

        Nframes, Nkbins, Nact, Nexp = sfac.shape[:]
        sfac_time_av = np.nan * np.zeros((Nkbins, Nact, 2))

        self.__calc_av_over_exp(sfac, N, save_name = 'sfac_av',)

        sfac_temp_corr = self.est_corr_time(sfac_dict, npz_target_name='sfac', \
                                            npz_path = sfac_path, Ndataset = N, \
                                            acf_dict = acf_dict, use_error_bound = False)
        
        for i in range(Nact):
            ff_idx = self.conv_list[N][i]

            Nsamples = (Nframes - ff_idx) * np.ones((Nkbins, Nexp)) - np.nansum(np.isnan(sfac[ff_idx:, :, i, :]),axis=(0))
            Nind_samples = np.nansum(Nsamples / sfac_temp_corr[1 if temp_corr_simple else 0, :, i, :,], axis = -1)

            sfac_mean_err = np.nanmean(sfac_err[self.conv_list[N][i]:, :, i, :], axis = (0, -1)) 
            sfac_time_av[:, i, 0]  = np.nanmean(sfac[self.conv_list[N][i]:, :, i, :], axis = (0, -1))
            sfac_time_av[:, i, 1] = (sfac_mean_err + np.nanstd(sfac[self.conv_list[N][i]:, :, i, :], axis = (0, -1), ddof = ddof)) / np.sqrt(Nind_samples)

        np.save(os.path.join(self.output_paths[N], 'sfac_time_av.npy'), sfac_time_av)

        if calculate_pcf:
            if not os.path.isfile(os.path.join(self.output_paths[N], 'pcf.npz')):
                print('Structure factor not found. Extract results first.')
                return

            pcf_path = os.path.join(self.output_paths[N], 'pcf.npz')
            pcf = np.load(pcf_path, 'pcf.npz', allow_pickle=True)['pcf']
            pcf_dict = dict(np.load(pcf_path, allow_pickle = True))
    
            self.__calc_av_over_exp(pcf, N, return_arr = False, save_name = 'pcf_av', save = True)
            pcf_temp_corr = self.est_corr_time(pcf_dict, npz_target_name='pcf', \
                                            npz_path = pcf_path, Ndataset = N, \
                                            acf_dict = acf_dict, use_error_bound = False)       
            self.__calc_time_av(N, pcf, pcf_temp_corr, temp_corr_simple = temp_corr_simple, ddof = ddof, save_name = 'pcf_time_av',)  
        return

    def est_corr_time(self, npz_dict, npz_target_name, npz_path, Ndataset = 0, use_error_bound = True,
                    acf_dict = {'nlags_frac': 0.5, 'nlags': None, 'max_lag': None, \
                                'alpha': 0.3174, 'max_lag_threshold': 0, 'simple_threshold': 0.15, \
                                'first_frame_idx': None},
                    save = True):  
        """ npz_obj is the npz file containing the target array
        target array must have shape (Nframes, Nact, Nexp) or (Nframes, Nsomething, Nact, Nexp)
        """
        
        arr = npz_dict[npz_target_name]
        act_list = self.act_list[Ndataset] 
        conv_list = self.conv_list[Ndataset]

        corr_time_arr = np.nan * np.zeros((2, *arr.shape[1:],)) 
        Nsomething = arr.shape[1] if len(arr.shape) == 4 else None

        max_lag = acf_dict['max_lag']
        alpha = acf_dict['alpha']
        max_lag_threshold = acf_dict['max_lag_threshold']
        simple_threshold = acf_dict['simple_threshold']
        first_frame_idx = acf_dict['first_frame_idx']

        for j, act in enumerate(act_list):
            act_idx = act_list.index(act) if type(act_list) is list else np.where(act_list == act)[0][0]

            if first_frame_idx is None or first_frame_idx > arr.shape[0]:
                conv_idx = conv_list[act_idx]
            else: 
                conv_idx = first_frame_idx
          #  conv_idx = conv_list[act_idx] if first_frame_idx is None else first_frame_idx
            nf = arr.shape[0] - conv_idx
            nlags= int(nf * acf_dict['nlags_frac']) if acf_dict['nlags'] is None else int(acf_dict['nlags'])
            nlags = int(min(nf * acf_dict['nlags_frac'], nlags))

            if nlags > nf:
                continue
        
            arr_vals =  arr[:, :, act_idx, :] if Nsomething else arr[:, act_idx, :]
            acf_arr, confint_arr = calc_acf_for_arr(arr_vals, conv_idx = conv_idx, nlags = nlags, alpha = alpha)

            if Nsomething:
                for k in range(arr.shape[-1]):
                    for i in range(arr.shape[1]):
                        acf_vals = acf_arr[-(nlags + 1):, i, k]
                        confint_vals = confint_arr[-(nlags + 1):, :, i, k]

                        tau, tau_simple = estimate_effective_sample_size(acf_vals,
                                                                    confint_vals = confint_vals, 
                                                                    max_lag = max_lag, 
                                                                    max_lag_threshold = max_lag_threshold, 
                                                                    simple_threshold = simple_threshold,
                                                                    use_error_bound = use_error_bound)    
                        corr_time_arr[:, i, j, k] = [tau, tau_simple,]
            else:
                for k in range(arr.shape[-1]):
                    acf_vals = acf_arr[- (nlags + 1):,k]
                    confint_vals = confint_arr[- (nlags + 1):,:,k]

                    tau, tau_simple = estimate_effective_sample_size(acf_vals,
                                                                confint_vals = confint_vals, 
                                                                max_lag = max_lag, 
                                                                max_lag_threshold = max_lag_threshold, 
                                                                simple_threshold = simple_threshold,
                                                                use_error_bound = use_error_bound)   
                    corr_time_arr[:, j, k] = [tau, tau_simple,]
        if save:
            npz_dict['corr_time_arr'] = corr_time_arr 
            np.savez(npz_path, **npz_dict)
        return corr_time_arr

    def calc_corr_time_old(self, Ndataset = 0, use_error_bound=False,
                         acf_dict = {'nlags_frac': 0.5, 'max_lag': None, 'alpha': 0.3174, 'max_lag_threshold': 0, 'simple_threshold': 0.15},
                         save = True):  

        def_arr = self.get_arrays_full(Ndataset = Ndataset)[0]
        act_list = self.act_list[Ndataset]
   
        corr_time_arr = np.zeros((2, def_arr.shape[-2], def_arr.shape[-1],)) 
  
        max_lag = acf_dict['max_lag']
        alpha = acf_dict['alpha']
        max_lag_threshold = acf_dict['max_lag_threshold']
        simple_threshold = acf_dict['simple_threshold']

        for j, act in enumerate(self.act_list[Ndataset]):
            act_idx = act_list.index(act)

            conv_idx = self.conv_list[Ndataset][act_idx]
            nf = def_arr.shape[0] - conv_idx
            nlags= int(nf * acf_dict['nlags_frac'])  
      
            acf_arr, confint_arr = calc_acf_for_arr(def_arr[:, act_idx, :], conv_idx = conv_idx, nlags = nlags, alpha = alpha)
    
            for k in range(self.Nexp[Ndataset]):

                acf_vals = acf_arr[- (nlags + 1):,k]
                confint_vals = confint_arr[- (nlags + 1):,:,k]

                tau, tau_simple = estimate_effective_sample_size(acf_vals,
                                                            confint_vals = confint_vals, 
                                                            max_lag = max_lag, 
                                                            max_lag_threshold=max_lag_threshold, 
                                                            simple_threshold=simple_threshold,
                                                            use_error_bound=use_error_bound)     
                corr_time_arr[:, j, k] = [tau, tau_simple]
        if save:
            np.save(os.path.join(self.output_paths[Ndataset], 'corr_time.npy'), corr_time_arr)
        return corr_time_arr

    def calc_susceptibility(self, Ndataset = 0, Nframes = None, 
                            order_param_func = None, Nscale = True, \
                            save = True):

        act_list = self.act_list[Ndataset]
        conv_list = self.conv_list[Ndataset]
        output_path = self.output_paths[Ndataset]
        Nact = len(act_list)

        av_def = self.get_arrays_av(Ndataset = Ndataset)[-1]
        def_arr = self.get_arrays_full(Ndataset = Ndataset)[0]

        if order_param_func is None:
            order_param = def_arr
        else:
            order_param = order_param_func(def_arr, av_def, self.LX[Ndataset])
        
        # Initialize arrays
        sus = np.zeros((Nact, 2)) * np.nan

        for i, act in enumerate(act_list):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                Nfirst_frame = conv_list[i] if Nframes is None else max(def_arr.shape[0] - Nframes, conv_list[i])
                sus[i,0] = np.nanmean(order_param[Nfirst_frame:, i, :] ** 2,) - np.nanmean(order_param[Nfirst_frame:, i, :]) ** 2
        if Nscale:
            sus /= av_def[:, 0][:, None]
            sus[:,1] = np.sqrt(sus[:,0]/av_def[:,0]) * av_def[:,1]
        if save:
            np.save(os.path.join(output_path, 'susceptibility.npy'), sus)
        return sus
    
    def print_params(self, Ndataset = 0, act = [], param_keys = ['nstart', 'nsteps']):
        """
        Print out the simulation parameters for the given dataset.
        If act is [], simulation parameters for all activities will be output.
        """

        # change pathlib to WindowsPath to avoid error
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

        act_path = self.act_dir_list[Ndataset]
        act_list = self.act_list[Ndataset] if len(act) == 0 else act

        for activity in act_list:
            # load pkl dictionary
            act_idx = self.act_list[Ndataset].index(activity)
            if activity in [0.02, 0.03, 0.2]:
                activity = str(activity) + '0'
            base_path = os.path.join(act_path[act_idx], f'zeta_{activity}_counter')

            if os.path.isdir(base_path + '_0'):
                dict_path = base_path + '_0'
            elif os.path.isdir(base_path + '_10'):
                dict_path = base_path + '_10'
            elif os.path.isdir(base_path + '_20'):
                dict_path = base_path + '_20'
            else:
                print(f"Parameter dictionary was not found for activity {activity}")
                continue

            dict_path = os.path.join(dict_path, 'model_params.pkl')
   
            with open(dict_path, 'rb') as f:
                param_dict = pkl.load(f)

            print(f"\nFor activity = {activity}:")
            for key in param_keys:
                try:
                    print(f"{key}: {param_dict[key]}")
                except:
                    print(f"{key} not found in parameter dictionary.")
        # reset pathlib
        pathlib.PosixPath = temp
        return

    def update_conv_list(self, Ndataset_list = None):
        if Ndataset_list is None:
            Ndataset_list = range(self.Ndata)
        
        for i in Ndataset_list:
            fig, ax = self.plot_defects_per_activity(Ndataset = i, plot_density = False)
            plt.show()
            for j in range(self.Nactivity[i]):
                self.conv_list[i][j] = int(input(f'Enter the first frame to use for activity {self.act_list[i][j]}: '))

            # save the convergence list
            np.savetxt(os.path.join(self.output_paths[i], 'conv_list.txt'), self.conv_list[i])
        return

    def get_arrays_full(self, Ndataset = 0,):
        """
        returns defect_arr, av_counts
        """
        output_path = self.output_paths[Ndataset]
        try:
            defect_arr = np.load(os.path.join(output_path, 'defect_arr.npz'), allow_pickle = True)['defect_arr']  
            av_counts = np.load(os.path.join(output_path, f'av_counts{self.count_suffix}.npz'), allow_pickle = True)['av_counts']
        except:
            print('Arrays not found. Analyse defects first.')
            return
        return defect_arr, av_counts
      
    def get_arrays_av(self, Ndataset = 0, use_merged = False):
        """  
        returns defect_arr_av, var_counts_av, av_counts_av, av_defects
        """

        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        try:
            defect_arr_av = np.load(os.path.join(output_path, 'defect_arr_av.npy'))
            var_counts_av = np.load(os.path.join(output_path, f'var_counts_av{self.count_suffix}.npy'))
            av_counts_av = np.load(os.path.join(output_path, f'av_counts_av{self.count_suffix}.npy'))
            av_defects = np.load(os.path.join(output_path, 'av_defects.npy'))
        except:
            print('Arrays not found. Analyse defects first.')
            return
        
        return defect_arr_av, var_counts_av, av_counts_av, av_defects
   
    def get_susceptibility(self, Ndataset = 0, use_merged = False):
        """
        returns susceptibility
        """
        if use_merged:
            output_path = os.path.join(self.output_main_path, 'merged_results')
            Ndataset = np.argmin(self.priorities)

            if not os.path.isdir(output_path):
                print(f'Merged results not found. Run merge_results first.')
                return
        else:
            output_path = self.output_paths[Ndataset]
        try:
            susceptibility = np.load(os.path.join(output_path, 'susceptibility.npy'))
        except:
            print('Susceptibilitites not found. Analyse defects first.')
            return
        return susceptibility

    def get_sfac_pcf(self, Ndataset = 0, time_av = True, use_merged = False):
        """
        returns kbins, sfac_av, rad, pcf_av
        """
        if use_merged:
            output_path = os.path.join(self.output_main_path, 'merged_results')
            Ndataset = np.argmin(self.priorities)

            if not os.path.isdir(output_path):
                print(f'Merged results not found. Run merge_results first.')
                return
        else:
            output_path = self.output_paths[Ndataset]

        prefix = 'time_' if time_av else ''

        try:
            sfac_av = np.load(os.path.join(output_path, f'sfac_{prefix}av.npy'))
            pcf_av = np.load(os.path.join(output_path, f'pcf_{prefix}av.npy'))
        except:
            print('Structure factor or pcf not found. Analyse defects first.')
            return

        rad = np.load(os.path.join(output_path, 'rad.npy'))
        kbins = np.loadtxt(os.path.join(output_path, 'kbins.txt'))
    
        return kbins, sfac_av, rad, pcf_av

    def get_sfac_pcf_full(self, Ndataset = 0,):
        """
        returns kbins, sfac, rad, pcf
        """
      
        output_path = self.output_paths[Ndataset]

        try:
            sfac = np.load(os.path.join(output_path, f'sfac.npz'), allow_pickle = True)['sfac']
            pcf = np.load(os.path.join(output_path, f'pcf.npz'))
        except:
            print('Structure factor or pcf not found. Analyse defects first.')
            return

        rad = np.load(os.path.join(output_path, 'rad.npy'))
        kbins = np.loadtxt(os.path.join(output_path, 'kbins.txt'))
    
        return kbins, sfac, rad, pcf
    
    def extract_results(self, save = True,):
        """
        Analyse the defects for all the input folders
        """
        for N in range(self.Ndata):

            defect_arr = np.nan * np.zeros((self.Nframes[N], self.Nactivity[N], self.Nexp[N]))
            av_counts = np.nan * np.zeros([self.Nframes[N], len(self.window_sizes[N]), self.Nactivity[N], self.Nexp[N]])
  
            if os.path.isfile(os.path.join(self.output_paths[N], 'kbins.txt')):
                ext_sfac = True
                kbins = np.loadtxt(os.path.join(self.output_paths[N], 'kbins.txt'))
                rad = np.load(os.path.join(self.output_paths[N], 'rad.npy'))
                sfac = np.nan * np.zeros((self.Nframes[N], len(kbins), 2, self.Nactivity[N], min(10, self.Nexp[N])))
                pcf = np.nan * np.zeros((self.Nframes[N], len(rad), self.Nactivity[N], min(self.Nexp[N], 10)))
            else:
                ext_sfac = False
            
            print('Analyse defects for input folder {}'.format(self.input_paths[N]))
            for i, (act, act_dir) in enumerate(zip(self.act_list[N], self.act_dir_list[N])):
                sfac_counter = 0
                exp_list = []
                exp_dir_list = []

                for file in os.listdir(act_dir):
                    exp_count = file.split('_')[-1]
                    exp_list.append(int(exp_count))
                    exp_dir_list.append(os.path.join(act_dir, file))

                # sort the activity list and the activity directory list
                exp_list, exp_dir_list = zip(*sorted(zip(exp_list, exp_dir_list)))

                print(f'Extracting results for activity {act} ...')

                for j, (exp, exp_dir) in enumerate(zip(exp_list, exp_dir_list)):
                    if os.path.isfile(os.path.join(exp_dir, 'Ndefects.txt')):
                        ndef_path = os.path.join(exp_dir, 'Ndefects.txt')
                    else:
                        ndef_path = os.path.join(exp_dir, 'Ndefects_act{}_exp{}.txt'.format(act, exp))

                    def_temp = np.loadtxt(ndef_path)
                    idx_start = min(self.Nframes[N], len(def_temp))
                    defect_arr[-idx_start:, i, j] = def_temp[-idx_start:]   

                    if os.path.isfile(os.path.join(exp_dir, f'av_counts{self.count_suffix}.npy')):
                        counts = np.load(os.path.join(exp_dir, f'av_counts{self.count_suffix}.npy'))
                    else:
                        counts = np.loadtxt(os.path.join(exp_dir, 'av_counts_act{}_exp{}.txt'.format(act,exp)))
                    
                    if counts.shape[-1] == 1:
                        try:
                            av_counts[-idx_start:, :, i, j] = counts[-idx_start:, :, 0]
                        except:
                            raise ValueError(f'av_counts has the wrong shape: {counts.shape} Aborting ...')
                    else:
                        raise ValueError(f'av_counts has the wrong shape for exp {exp}: {counts.shape} Aborting ...')
                    if ext_sfac and sfac_counter < 10:
                        try:
                            if os.path.isfile(os.path.join(exp_dir, 'sfac.npy')):
                                sfac_temp = np.load(os.path.join(exp_dir, 'sfac.npy'))
                            else:
                                sfac_temp = np.load(os.path.join(exp_dir, 'structure_factor_act{}_exp{}.npy'.format(act,exp)))
                            sfac[-idx_start:, :, :, i, sfac_counter] = sfac_temp[-idx_start:, :,:]
                        except:
                            pass
                        try:
                            pcf[-idx_start:, :, i, sfac_counter] = np.loadtxt(os.path.join(exp_dir, 'pcf.txt'.format(act,exp)))[-idx_start:,:]
                        except:
                            pass
                        if os.path.isfile(os.path.join(exp_dir,'sfac_analysis_completed.txt')):
                            sfac_counter += 1 
      
            if save:
                np.savez(os.path.join(self.output_paths[N], 'defect_arr.npz'), defect_arr = defect_arr)
                np.savez(os.path.join(self.output_paths[N], f'av_counts{self.count_suffix}.npz'), av_counts = av_counts)

                if ext_sfac:
                    np.savez(os.path.join(self.output_paths[N], 'sfac.npz'), sfac_full = sfac, sfac = sfac[:, :, 0, :, :], sfac_err = sfac[:, :, 1, :, :])
                    np.savez(os.path.join(self.output_paths[N], 'pcf.npz'), pcf = pcf)

    def analyze_defects(self, Ndataset_list = None, temp_corr_simple = True, calc_pcf = False,
                        acf_dict = {'nlags_frac': 0.5, 'max_lag': None, 'nlags': None, 'alpha': 0.3174, 
                                    'max_lag_threshold': 0, 'simple_threshold': 0.2, 'first_frame_idx': 0},
                        ddof = 1, sus_dict = {}, dens_fluc_dict = {}, sfac_dict = {}):

        Ndataset_list = range(self.Ndata) if Ndataset_list is None else Ndataset_list

        for N in Ndataset_list:
            while True:
                try:
                    defect_arr, av_counts = self.get_arrays_full(N)
                    break
                except:
                    print('Defect array not found. They will be extracted now using normalize = True')
                    self.extract_results(save = True, normalize = True)

            if len(np.unique(self.conv_list[N])) == 1:
                print(f'NB: All simulations are set to converge at the first frame for dataset {N}. To change this, call update_conv_list.\n')

            # calculate the correlation time
            def_arr_path = os.path.join(self.output_paths[N], f'defect_arr.npz')
            count_arr_path = os.path.join(self.output_paths[N], f'av_counts{self.count_suffix}.npz')
            def_arr_dict = dict(np.load(def_arr_path, allow_pickle=True))
            count_arr_dict = dict(np.load(count_arr_path, allow_pickle=True))
            def_temp_corr = self.est_corr_time(def_arr_dict, npz_target_name='defect_arr', \
                                           npz_path = def_arr_path, Ndataset = N, \
                                            acf_dict = acf_dict, use_error_bound = False)  
            count_temp_corr = self.est_corr_time(count_arr_dict, npz_target_name='av_counts', \
                                           npz_path = count_arr_path, Ndataset = N, \
                                            acf_dict = acf_dict, use_error_bound = False)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.__calc_av_over_exp(defect_arr, N, save_name = 'defect_arr_av',)
                self.__calc_av_over_exp(def_temp_corr, N, save_name = 'def_corr_time_av',)

                # calculate number variance of observation windows
                var_counts = np.zeros((len(self.window_sizes[N]), self.Nactivity[N], 2))
                for i, _ in enumerate(self.act_list[N]):
                    vars = np.nanvar(av_counts[self.conv_list[N][i]:, :, i, :], axis = 0)
                    var_counts[:, i, 0] = np.nanvar(av_counts[self.conv_list[N][i]:, :, i, :], axis = (0,-1))
                    var_counts[:, i, 1] = np.nanstd(vars, axis = -1) / np.sqrt(self.Nexp[N])
                np.save(os.path.join(self.output_paths[N], f'var_counts_av{self.count_suffix}.npy'), var_counts)
       
                # calculate the average number of counts
                self.__calc_time_av(N, av_counts, count_temp_corr, temp_corr_simple = temp_corr_simple, \
                                    ddof = ddof, save_name = f'av_counts_av{self.count_suffix}',)    

                # calculate the average number of defects globally
                av_defects = np.zeros((self.Nactivity[N], 2))    
                for i, _ in enumerate(self.act_list[N]):
                    Ndef_samples = (self.Nframes[N] - self.conv_list[N][i]) * np.ones(self.Nexp[N])
                    Ndef_ind_samples = np.nansum(Ndef_samples / def_temp_corr[1 if temp_corr_simple else 0, i, :,], axis = -1)
    
                    av_defects[i, 0] = np.nanmean(defect_arr[self.conv_list[N][i]:, i, :])
                    av_defects[i, 1] = np.nanstd(defect_arr[self.conv_list[N][i]:, i, :]) / np.sqrt(Ndef_ind_samples) 

                np.save(os.path.join(self.output_paths[N], 'av_defects.npy'), av_defects)

                self.__calc_sfac_pcf(N, acf_dict = acf_dict, temp_corr_simple = temp_corr_simple, \
                                    calculate_pcf = calc_pcf)
                self.calc_susceptibility(N, **sus_dict)

            if dens_fluc_dict != {}:
                self.analyze_hyperuniformity(N, **dens_fluc_dict)
            if sfac_dict != {}:
                self.analyze_sfac(N, **sfac_dict)
                self.analyze_sfac_time_av(N, **sfac_dict)
        return

    def merge_results(self, save_path = None, include_sfac = True, save = True):
        """
        Merge the results from the different datasets according to the priorities in descending order.
        """

        if save_path is None:
            save_path = os.path.join(self.output_main_path, 'merged_results')
        if not os.path.exists(save_path):
            os.makedirs(save_path)


        Nbase = np.argmax(self.priorities)
        Nbase_frames = self.Nframes[Nbase]
        window_sizes = self.window_sizes[Nbase]
        Nbins = len(self.window_sizes[Nbase])
        Nrad = len(np.load(os.path.join(self.output_paths[Nbase], 'rad.npy')))

        # overwrite the activities with the ones from the other datasets according to self.priorities
        _, Nsorted = zip(*sorted(zip(reversed(self.priorities), range(self.Ndata))))

        act_list_full = []
        for N in Nsorted:
            act_list_full += self.act_list[N]

        # initialize arrays
        defect_arr_av = np.nan * np.zeros((Nbase_frames, len(act_list_full), 2))
        var_counts_av = np.nan * np.zeros((len(window_sizes), len(act_list_full), 2))
        av_counts_av = np.nan * np.zeros((len(window_sizes), len(act_list_full), 2))
        av_defects = np.nan * np.zeros((len(act_list_full), 2))
        susceptibility = np.nan * np.zeros((len(act_list_full), 2))
        def_corr_time_av = np.nan * np.zeros((2, len(act_list_full), 2))
        fit_params_count = np.nan * np.zeros((len(act_list_full), 4))

        if include_sfac:
            kbins, _, rad, _ = self.get_sfac_pcf(Nbase, time_av = False)
            sfac_av = np.nan * np.zeros((Nbase_frames, len(kbins), len(act_list_full), 2))
            pcf_av = np.nan * np.zeros((Nbase_frames, len(rad), len(act_list_full), 2))
            sfac_time_av = np.nan * np.zeros((len(kbins), len(act_list_full), 2))
            pcf_time_av = np.nan * np.zeros((len(rad), len(act_list_full), 2))
            alpha_sfac = np.nan * np.zeros((len(act_list_full), 2))
            fit_params_sfac_time_av = np.nan * np.zeros((len(act_list_full), 4))
            fit_params_sfac_time_av_unweighted = np.nan * np.zeros((len(act_list_full), 4))

        for N in Nsorted:

            Nframes = self.Nframes[N]
            act_idx_list = []
            for act in self.act_list[N]:
                act_idx_list.append(act_list_full.index(act))
 
            defect_arr_av[-Nframes:, act_idx_list, :] = np.load(os.path.join(self.output_paths[N], 'defect_arr_av.npy'))
            av_counts_av[:, act_idx_list, :] = np.load(os.path.join(self.output_paths[N], f'av_counts_av{self.count_suffix}.npy'))
            var_counts_av[:, act_idx_list, :] = np.load(os.path.join(self.output_paths[N], f'var_counts_av{self.count_suffix}.npy'))
            av_defects[act_idx_list] = np.load(os.path.join(self.output_paths[N], 'av_defects.npy'))
            def_corr_time_av[:, act_idx_list, :] = np.load(os.path.join(self.output_paths[N], 'def_corr_time_av.npy'))

            susceptibility[act_idx_list] = np.load(os.path.join(self.output_paths[N], 'susceptibility.npy'))
            fit_params_count[act_idx_list] = np.load(os.path.join(self.output_paths[N], f'fit_params_count.npy'))
      
            if include_sfac:  
                sfac_av[-Nframes:, :, act_idx_list, :] = np.load(os.path.join(self.output_paths[N], 'sfac_av.npy'))
                sfac_time_av[:, act_idx_list, :] = np.load(os.path.join(self.output_paths[N], 'sfac_time_av.npy'))[:]

                pcf_av[-Nframes:, :, act_idx_list, :] = np.load(os.path.join(self.output_paths[N], 'pcf_av.npy'))
                pcf_time_av[:, act_idx_list, :] = np.load(os.path.join(self.output_paths[N], 'pcf_time_av.npy'))[:]

                alpha_sfac[act_idx_list] = np.load(os.path.join(self.output_paths[N], f'alpha_list_sfac.npy'))

                fit_params_sfac_time_av[act_idx_list] = np.load(os.path.join(self.output_paths[N], f'fit_params_sfac_time_av.npy'))
                fit_params_sfac_time_av_unweighted[act_idx_list] = np.load(os.path.join(self.output_paths[N], f'fit_params_sfac_time_av_unweighted.npy'))

            
        if save:
            np.save(os.path.join(save_path, 'activity_list.npy'), act_list_full)
            np.save(os.path.join(save_path, 'window_sizes.npy'), window_sizes)
            np.save(os.path.join(save_path, 'defect_arr_av.npy'), defect_arr_av)
            np.save(os.path.join(save_path, f'av_counts_av{self.count_suffix}.npy'), av_counts_av)
            np.save(os.path.join(save_path, f'var_counts_av{self.count_suffix}.npy'), var_counts_av)
            np.save(os.path.join(save_path, 'av_defects.npy'), av_defects)
            np.save(os.path.join(save_path, 'susceptibility.npy'), susceptibility)
            np.save(os.path.join(save_path, 'def_corr_time_av.npy'), def_corr_time_av)
            np.save(os.path.join(save_path, f'fit_params_count.npy'), fit_params_count)
            if include_sfac:
                np.save(os.path.join(save_path, 'sfac_av.npy'), sfac_av)
                np.save(os.path.join(save_path, 'sfac_time_av.npy'), sfac_time_av)
                np.save(os.path.join(save_path, 'pcf_av.npy'), pcf_av)
                np.save(os.path.join(save_path, 'pcf_time_av.npy'), pcf_time_av)
                np.savetxt(os.path.join(save_path, 'kbins.txt'), kbins)
                np.save(os.path.join(save_path, 'rad.npy'), rad)
                np.save(os.path.join(save_path, f'alpha_list_sfac.npy'), alpha_sfac)
                np.save(os.path.join(save_path, f'fit_params_sfac_time_av.npy'), fit_params_sfac_time_av)
                np.save(os.path.join(save_path, f'fit_params_sfac_time_av_unweighted.npy'), fit_params_sfac_time_av_unweighted)
        return

    def merge_sus(self, save_path = None, save = True):

        if save_path is None:
            save_path = os.path.join(self.output_main_path, 'merged_results')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        Nbase = np.argmin(self.priorities)

        try:
            susceptibility = np.load(os.path.join(self.output_paths[Nbase], 'susceptibility.npy'))
        except:
            print('Base dataset not found. Analyse defects first.')
            return
        
        # overwrite the activities with the ones from the other datasets according to self.priorities
        _, Nsorted = zip(*sorted(zip(self.priorities, range(self.Ndata))))

        for N in Nsorted[1:]:
            act_idx_list = []
            for act in self.act_list[N]:
                act_idx_list.append(self.act_list[Nbase].index(act))

            susceptibility[act_idx_list] = np.load(os.path.join(self.output_paths[N], 'susceptibility.npy'))
        if save:
            np.save(os.path.join(save_path, 'susceptibility.npy'), susceptibility)
        return
    
    def analyze_hyperuniformity(self, Ndataset = 0, fit_dict = {}, window_idx_bounds = None, \
                                act_idx_bounds = None, use_merged = False, save = True,):
        

        suffix = 'count'
        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        if window_idx_bounds is None:
            window_idx_bounds = [0, len(self.window_sizes[Ndataset])]
        if act_idx_bounds is None:
            act_idx_bounds = [0, len(self.act_list[Ndataset])]

        window_sizes = self.window_sizes[Ndataset][window_idx_bounds[0]:window_idx_bounds[1]]
        act_list = self.act_list[Ndataset][act_idx_bounds[0]:act_idx_bounds[1]]

        try:
            var_counts = self.get_arrays_av(Ndataset = Ndataset,)[1]
        except:
            print('Density fluctuations not found. Analyse defects first.')
            return

        var_av = var_counts[window_idx_bounds[0]:window_idx_bounds[1], act_idx_bounds[0]:act_idx_bounds[1], 0]
        var_std = var_counts[window_idx_bounds[0]:window_idx_bounds[1], act_idx_bounds[0]:act_idx_bounds[1], 1]

        if fit_dict == {}:
            def fit_func(x, alpha, beta):
                return beta * (2 - alpha) + (2 - alpha) * x
            param_guess = np.array([0.1, 0.3])
        else:
            fit_func = fit_dict['fit_func']
            param_guess = fit_dict['param_guess']

        Nparams = len(param_guess)
            
        fit_params = np.zeros([len(act_list), 2 * Nparams]) * np.nan
        stat_arr = np.zeros([len(act_list), 3]) * np.nan
   
        for i, _ in enumerate(act_list):
            count_var_av = var_av[:, i]
            count_var_std = var_std[:, i]
            zero_mask = (count_var_av > 0) & (count_var_std > 0)

            if len(count_var_av[zero_mask]) < 5:
                continue
            try:
                x = np.log(window_sizes[zero_mask])
                y = np.log(count_var_av[zero_mask])
                yerr = count_var_std[zero_mask] / count_var_av[zero_mask]
            except:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
                fit = do_chi2_fit(fit_func, x, y, yerr, param_guess, verbose = False)

            stat_arr[i, :] = get_statistics_from_fit(fit, len(x), subtract_1dof_for_binning = False)
            fit_params[i, :Nparams] = fit.values[:]
            fit_params[i, Nparams:] = fit.errors[:]

        print("Non-converged fits (p < 0.05): ", np.nansum((stat_arr[:, -1] < 0.05)))

        if save:
            np.save(os.path.join(output_path, f'fit_params_{suffix}.npy'), fit_params)
            np.save(os.path.join(output_path, f'stat_arr_{suffix}.npy'), stat_arr)
        return fit_params, stat_arr

    def analyze_sfac_time_av_old(self, Ndataset = 0, Npoints_bounds = [3,7], act_idx_bounds = None, use_merged = False, save = True,):
        """
        returns fit_params_time_av
        """

        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        if act_idx_bounds is None:
            act_idx_bounds = [0, len(self.act_list[Ndataset])]
        act_list = self.act_list[Ndataset] 

        try:
            kbins = np.loadtxt(os.path.join(output_path, 'kbins.txt'))
            sfac_av = np.load(os.path.join(output_path, f'sfac_time_av.npy'))
        except:
            print('Time-averaged structure factor or pcf not found. Analyse defects first.')
            return

        def fit_func(x, alpha, beta):
                    return beta + alpha * x
        param_guess = np.array([0.1, 0.1])
        fit_string = rf'$y = \beta + \alpha |k|$'
        Nparams = len(param_guess)

        fit_params_sfac_time_av = np.zeros([len(act_list), 2 * Nparams]) * np.nan

        for i, act in enumerate(act_list):
            try:
                x = np.log(kbins)
                y = np.log(sfac_av[:, i, 0])
                yerr = sfac_av[:, i, 1] / sfac_av[:, i, 0]
            except:
                continue

            fit_vals = np.nan * np.zeros((Npoints_bounds[1] - Npoints_bounds[0], Nparams))
            fit_err = np.nan * np.zeros((Npoints_bounds[1] - Npoints_bounds[0], Nparams))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
                for j, Npoints_to_fit in enumerate(range(Npoints_bounds[0], Npoints_bounds[1])):         
                    
                    fit = do_chi2_fit(fit_func, x[:Npoints_to_fit], y[:Npoints_to_fit], yerr[:Npoints_to_fit], param_guess, verbose = False)
                    
                    fit_vals[j] = fit.values[:] if fit._fmin.is_valid else [np.nan, np.nan]
                    fit_err[j] = fit.errors[:] if fit._fmin.is_valid else [np.nan, np.nan]

                nan_mask = np.isnan(fit_vals[:,0])
                fit_vals_valid = fit_vals[~nan_mask]
                fit_err_valid = fit_err[~nan_mask]

                if len(fit_vals_valid) == 0 or len(fit_err_valid) == 0:
                    continue
                
                alpha_weighted_av, alpha_sem = calc_weighted_mean(fit_vals_valid[:,0], fit_err_valid[:,0])
                beta_weighted_av, beta_sem = calc_weighted_mean(fit_vals_valid[:,1], fit_err_valid[:,1])

                fit_params_sfac_time_av[i, :Nparams] = alpha_weighted_av, beta_weighted_av
                fit_params_sfac_time_av[i, Nparams:] = np.std(fit_vals_valid, axis = 0) / np.sqrt(fit_vals_valid.shape[0])

        if save:
            np.save(os.path.join(output_path, f'fit_params_sfac_time_av.npy'), fit_params_sfac_time_av)
        return fit_params_sfac_time_av
    
    def analyze_sfac_time_av(self, Ndataset = 0, Npoints_bounds = [3,7], act_idx_bounds = None,
                            pval_min = 0.01, save = True, save_plots = True, verbose = False):
        """
        returns fit_params_time_av
        """

        def fit_func(x, alpha, beta):
                    return beta + alpha * x
        param_guess = np.array([0.1, 0.1])
        Nparams = len(param_guess)

        if act_idx_bounds is None:
            act_idx_bounds = [0, None]
        act_list = self.act_list[Ndataset][act_idx_bounds[0]:act_idx_bounds[1]] 

        try:
            kbins, sfac_av = self.get_sfac_pcf(Ndataset, time_av = True,)[0:2]
            sfac_av = sfac_av[:, act_idx_bounds[0]:act_idx_bounds[1], :]
        except:
            print('Time-averaged structure factor or pcf not found. Analyse defects first.')
            return
        
        fit_params_weighted = np.zeros([len(act_list), 4]) * np.nan
        fit_params_unweighted = np.zeros([len(act_list), 4]) * np.nan

        for i, act in enumerate(act_list):
            if verbose:
                print(f'\nAnalyzing activity {act}')
    
            it_max = 50
            sfac_nan_mask = np.isnan(sfac_av[:, i, 0])
            try:
                x = np.log(kbins[~sfac_nan_mask])
                y = np.log(sfac_av[~sfac_nan_mask, i, 0])
                yerr = sfac_av[~sfac_nan_mask, i, 1] / sfac_av[~sfac_nan_mask, i, 0] 
            except:
                continue

            fit_vals = np.nan * np.zeros((Npoints_bounds[1] - Npoints_bounds[0], Nparams))
            fit_err = np.nan * np.zeros((Npoints_bounds[1] - Npoints_bounds[0], Nparams))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)

                for j, Npoints_to_fit in enumerate(range(Npoints_bounds[0], Npoints_bounds[1])): 
                    it = 0        
                    yerr_mod = yerr.astype(float)   
                    while it < it_max:
                        fit = do_chi2_fit(fit_func, x[:Npoints_to_fit], y[:Npoints_to_fit], yerr_mod[:Npoints_to_fit],
                                            param_guess, verbose = False)
                        Ndof, chi2, pval = get_statistics_from_fit(fit, len(x[:Npoints_to_fit]), subtract_1dof_for_binning = False)

                        it += 1
                        # increase the error for the first points if the fit is not valid
                        yerr_mod *= 1.05
                        
                        if pval > pval_min:
                            fit_vals[j] = fit.values[:] 
                            fit_err[j] = fit.errors[:]
                            break
                    if verbose:
                        print(f'it: {it}, Npoints: {Npoints_to_fit}, alpha: {fit.values[0]} +/- {fit.errors[0]}, chi2: {chi2}, pval: {pval}')
            
                nan_mask = np.isnan(fit_vals[:,0])
                fit_vals_valid = fit_vals[~nan_mask]
                fit_err_valid = fit_err[~nan_mask]
                Nfits_valid = fit_vals_valid.shape[0]

                if len(fit_vals_valid) == 0 or len(fit_err_valid) == 0:
                    continue

                if Nfits_valid == 1:
                    fit_params_weighted[i, :Nparams] = fit_vals_valid
                    fit_params_weighted[i, Nparams:] = fit_err_valid
                    fit_params_unweighted[i] = fit_params_weighted[i].astype(float)
                else:  
                    alpha_weighted_av, alpha_sem = calc_weighted_mean(fit_vals_valid[:,0], fit_err_valid[:,0])
                    beta_weighted_av, beta_sem = calc_weighted_mean(fit_vals_valid[:,1], fit_err_valid[:,1])
                    alpha_sem *= np.sqrt(Nfits_valid)
                    beta_sem *= np.sqrt(Nfits_valid)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        fit_params_weighted[i, :Nparams] = alpha_weighted_av, beta_weighted_av
                        fit_params_weighted[i, Nparams:] = alpha_sem, beta_sem
                        fit_params_unweighted[i, :Nparams] = np.nanmean(fit_vals_valid, axis = 0)   
                        fit_params_unweighted[i, Nparams:] = np.nanstd(fit_vals_valid, axis = 0)
        if save:
            np.save(os.path.join(self.output_paths[Ndataset], f'fit_params_sfac_time_av.npy'), fit_params_weighted)
            np.save(os.path.join(self.output_paths[Ndataset], f'fit_params_sfac_time_av_unweighted.npy'), fit_params_unweighted)

        if save_plots:
            kmax_idx = Npoints_bounds[1] - 2
            kvals = np.linspace(kbins[0], kbins[kmax_idx], 100)

            for i, act in enumerate(act_list):
                fig, ax = plot_structure_factor(kbins, sfac_av[:,i,0], sfac_av[:, i,1], LX=self.LX[Ndataset])

                if not np.isnan(fit_params_weighted[i, 0]):
                    yvals_weighted = sfac_av[kmax_idx, i, 0] * kvals ** fit_params_weighted[i, 0] / kvals[-1] ** fit_params_weighted[i, 0]
                    ax.plot(kvals, yvals_weighted,
                            label = f'alpha = {fit_params_weighted[i, 0]:.2f} +/- {fit_params_weighted[i, 2]:.2f}, \
                            beta = {np.exp(fit_params_weighted[i, 1]):.2f} +/- {np.exp(fit_params_weighted[i, 1])*fit_params_weighted[i, 3]:.2f}')
                if not np.isnan(fit_params_unweighted[i, 0]):
                    yvals_unweighted = sfac_av[kmax_idx, i, 0] * kvals ** fit_params_unweighted[i, 0] / kvals[-1] ** fit_params_unweighted[i, 0]
                    ax.plot(kvals,yvals_unweighted,
                                label = f'alpha_u = {fit_params_unweighted[i, 0]:.2f} +/- {fit_params_unweighted[i, 2]:.2f}, \
                                beta_u = {np.exp(fit_params_unweighted[i, 1]):.2f} +/- {np.exp(fit_params_unweighted[i, 1])*fit_params_unweighted[i, 3]:.2f}')

                ax.set_title(f'Activity = {act}')
                ax.grid(True)
                ax.legend(fontsize=11) 

                fig.tight_layout()
                fig_savedir = os.path.join(self.output_paths[Ndataset], 'figs', 'sfac')
                if not os.path.exists(fig_savedir):
                    os.makedirs(fig_savedir)
                fig.savefig(os.path.join(fig_savedir, f'sfac_fit_act{act}_Np{Npoints_bounds[0]}.png'), dpi = 420)
                # close the figure to avoid memory issues
                plt.close(fig)
                
        return fit_params_weighted, fit_params_unweighted

    def analyze_sfac(self, Ndataset = 0, Npoints_bounds = [3,8], \
                    act_idx_bounds = None, pval_min = 0.01, \
                    use_merged = False, save = True, plot = False):
    
        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        if act_idx_bounds is None:
            act_idx_bounds = [0, len(self.act_list[Ndataset])]
        act_list = self.act_list[Ndataset][act_idx_bounds[0]:act_idx_bounds[1]]
        
        try:
            sfac_av = np.load(os.path.join(output_path, f'sfac_av.npy'))[:, :, act_idx_bounds[0]:act_idx_bounds[1], :]
            kbins = np.loadtxt(os.path.join(output_path, 'kbins.txt'))
        except:
            print('Structure factor average not found. Analyse defects first.')
            return
        
        def fit_func(x, alpha, beta):
                    return beta + alpha * x
        param_guess = np.array([0.1, 0.1])
        fit_string = rf'$y = \beta + \alpha |k|$'
        Nparams = len(param_guess)

        fit_params = np.zeros([self.Nframes[Ndataset], len(act_list), 2 * Nparams]) * np.nan
        alpha_list = np.zeros([len(act_list), 2]) * np.nan
        
        for i, act in enumerate(act_list):
            for frame in range(self.conv_list[Ndataset][i], self.Nframes[Ndataset]):
                s_av = sfac_av[frame, :, i, 0]
                s_std = sfac_av[frame, :, i, 1]

                nan_mask = np.isnan(s_av)
                if nan_mask.sum() > 0:
                    continue
                try:
                    x = np.log(kbins)
                    y = np.log(s_av)
                    yerr = s_std / s_av
                except:
                    continue

                fit_vals = np.nan * np.zeros((Npoints_bounds[1] - Npoints_bounds[0], Nparams))
                for j, Npoints_to_fit in enumerate(range(Npoints_bounds[0], Npoints_bounds[1])):         
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
                        fit = do_chi2_fit(fit_func, x[:Npoints_to_fit], y[:Npoints_to_fit], yerr[:Npoints_to_fit], param_guess, verbose = False)
                    if fit.fmin.is_valid:
                        fit_vals[j] = fit.values[:]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    fit_params[frame, i, :Nparams] = np.nanmean(fit_vals, axis = 0)
                    nan_fits = np.isnan(fit_vals[:,0]).sum()
                    fit_params[frame, i, Nparams:] = np.nanstd(fit_vals, axis = 0) / np.sqrt(Npoints_bounds[1] - Npoints_bounds[0] - nan_fits)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    alpha_mean = np.nanmean(fit_params[self.conv_list[Ndataset][i]:, i, 0])
                    alpha_err = np.nanstd(fit_params[self.conv_list[Ndataset][i]:, i, 0]) / np.sqrt(self.Nframes[Ndataset] - self.conv_list[Ndataset][i])
            except:
                alpha_mean, alpha_err = np.nan, np.nan

            alpha_list[i] = alpha_mean, alpha_err

        if save:
            np.save(os.path.join(output_path, f'fit_params_sfac.npy'), fit_params)
            np.save(os.path.join(output_path, f'alpha_list_sfac.npy'), alpha_list)

        if plot:
            self.plot_hyperuniformity_sfac(act_list, fit_params = fit_params, Ndataset = Ndataset, \
                                           act_idx_bounds = act_idx_bounds, use_merged = use_merged)
            
        return fit_params

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
        
        ax.set_xlabel(r'Activity ($\zeta$)')
        ax.set_ylabel(r' Av. defect density ($\overline{\rho}$')
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
                                   act_max_idx = None, plot_density = False, use_merged = False, save = False):
        
        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        if act_idx_bounds is None:
            act_idx_bounds = [0, len(self.act_list[Ndataset])]

        activities = self.act_list[Ndataset][act_idx_bounds[0]:act_idx_bounds[1]]
        norm = self.LX[Ndataset] ** 2 if plot_density else 1
        Nframes = self.Nframes[Ndataset] - Nfirst_frame

        try:
            defect_arr_av = self.get_arrays_av(Ndataset = Ndataset, use_merged = use_merged)[0] / norm
        except:
            print('Defect array not found. Analyse defects first.')
            return

        ncols = 4
        nrows = int(np.ceil(len(activities) / ncols))
        title = 'Defect density' if plot_density else 'Defect count'
        height = nrows * 3
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(16,height))
        ax = ax.flatten()  

        for i, act in enumerate(activities):
            act_idx = self.act_list[Ndataset].index(act)
            ax[i].errorbar(np.arange(Nframes), defect_arr_av[:, act_idx, 0], defect_arr_av[:, act_idx, 1], fmt='.', \
                            alpha = 0.15, markersize=9, label='Activity = {}'.format(act),) 
            ax[i].text(0.6, 0.2, rf'$\zeta$ = {act}', transform=ax[i].transAxes, fontsize=14, verticalalignment='top')

            # plot vertical lines to indicate the start of the averaging
            x=self.conv_list[Ndataset][i + act_idx_bounds[0]] - Nfirst_frame
            if x > 0:
                ax[i].axvline(x=self.conv_list[Ndataset][i + act_idx_bounds[0]], color='black', linestyle='--', alpha=0.5)
            ax[i].set_ylim(0, np.nanmax(defect_arr_av[:, act_idx, 0]) * 1.5)

        fig.suptitle(f'{title} for different activities (L = {self.LX[Ndataset]})' , fontsize=22, y = .995)
        fig.supxlabel('Frame', fontsize=20, y = 0)
        fig.supylabel(f'{title}', fontsize=20, x=0)
        fig.tight_layout()

        if save:
            if not os.path.isdir(os.path.join(output_path, 'figs')):
                os.makedirs(os.path.join(output_path, 'figs'))
            fig.savefig(os.path.join(output_path, f'figs\\defects_per_activity.png'), dpi = 420, pad_inches=0.2)

        plt.show()
        return fig, ax
    
    def plot_defects_per_exp(self, Ndataset = 0, act_idx_bounds = None, plot_density = False):

        try:
            act_idx_bounds = [0, len(self.act_list[Ndataset])] if act_idx_bounds is None else act_idx_bounds
            activities = self.act_list[Ndataset][act_idx_bounds[0]:act_idx_bounds[1]]
            norm = self.LX[Ndataset] ** 2 if plot_density else 1
            try:
                defect_arr = self.get_arrays_full(Ndataset = Ndataset)[0]
            except:
                print('Defect array not found. Analyse defects first.')
                return

            ncols = 4
            nrows = int(np.ceil(self.Nexp[Ndataset] / ncols))
            height = nrows * 3
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

    def plot_hyperuniformity_exp_all(self, fit_params = None, stat_arr = None, Ndataset = 0, act_idx_bounds = None, use_merged = False):

        suffix = 'count'

        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        if act_idx_bounds is None:
            act_idx_bounds = [0, len(self.act_list[Ndataset])]

        act_list = self.act_list[Ndataset][act_idx_bounds[0]:act_idx_bounds[1]]

        if isinstance(fit_params, np.ndarray) and isinstance(stat_arr, np.ndarray):
            pass
        else:
            fit_params = np.load(os.path.join(output_path, f'fit_params_{suffix}.npy'))[:, act_idx_bounds[0]:act_idx_bounds[1], :]
            stat_arr = np.load(os.path.join(output_path, f'stat_arr_{suffix}.npy'))[:, act_idx_bounds[0]:act_idx_bounds[1], :]

        if not len(act_list) == fit_params.shape[1]:
            print('The number of activities does not match the number of activities in the fit_params array.')
            return
        
        Nframes = fit_params.shape[0]
        Nparams = int(fit_params.shape[-1] / 2)
        Ngroup = 5

        ncols = 4
        nrows = int(np.ceil(len(act_list) / ncols))
        height = nrows * 4
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20,height))
        ax = ax.flatten()  

        for i, act in enumerate(act_list):
            if i == 0:
                label1 = rf'$\chi^2$ fit p-value > 0.05'
                label2 = rf'$\chi^2$ fit p-value < 0.05'
                label3 = 'Weighted mean'
                label4 = 'Grouped mean'
            else:
                label1 = None
                label2 = None
                label3 = None
                label4 = None

            p_mask = stat_arr[self.conv_list[Ndataset][i]:, i, 2] > 0.05
            
            ax[i].errorbar(np.arange(self.conv_list[Ndataset][i], Nframes)[p_mask], \
                        fit_params[self.conv_list[Ndataset][i]:,i,0][p_mask], \
                            fit_params[self.conv_list[Ndataset][i]:,i,Nparams][p_mask], \
                            fmt='.', \
                        alpha = 0.3, markersize=2, label = label1)
            
            ax[i].errorbar(np.arange(self.conv_list[Ndataset][i], Nframes)[~p_mask], \
                            fit_params[self.conv_list[Ndataset][i]:,i,0][~p_mask], \
                            fit_params[self.conv_list[Ndataset][i]:,i,Nparams][~p_mask], \
                                fmt='r.', \
                            capsize=2, capthick=1, elinewidth=1, markeredgewidth=2, alpha = 0.6, markersize=2, ecolor='red', label = label2)

            x = fit_params[self.conv_list[Ndataset][i]:, i, 0]
            max_idx = len(x) - len(x) % Ngroup 

            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                x_group = np.nanmean(x[:max_idx].reshape(-1, Ngroup), axis=1)
                dx_group = np.nanstd(x[:max_idx].reshape(-1, Ngroup), axis=1, ddof=min(Ngroup-2, 1)) / np.sqrt(Ngroup)
                x_frames = np.arange(self.conv_list[Ndataset][i], self.conv_list[Ndataset][i] + max_idx, Ngroup) + Ngroup / 2

                if len(x_group < 10) > 10:
                    ax[i].plot(x_frames, x_group, '.-', color='orange', alpha=0.8, markersize=4, label = label4)
                
                mean, err, Ndof, chi2, p_val = calc_weighted_mean_vec(x, fit_params[self.conv_list[Ndataset][i]:, i, Nparams])
            
            ax[i].axhline(mean, color='green', linestyle='--', alpha=0.5, lw = 2, label = label3)
            ax[i].text(0.05, 0.97, rf'$\zeta$ = {act}', transform=ax[i].transAxes, fontsize=12, verticalalignment='top')
           
        suptitle = fig.suptitle(f'Hyperuniformity exponent for different activities (L = {self.LX[Ndataset]})', y=1.05)
        fig.supxlabel('Frame')
        fig.supylabel(r'$\alpha$ (est. using $\overline{\delta \rho ^2}$)')
        fig.legend(ncol=4, fontsize = 13,bbox_to_anchor=(0.75, 1.01))
        fig.tight_layout()
        return fig, ax

    def plot_alpha_mean(self, Ndataset = 0, use_density_fit = True, include = ['all'], use_merged = False, save = False, fig_name = None):
            
            """
            include can take values
            'all': include all fits
            'fluc': include only fits based on fluctuations
            'sfac_time_av': include only fits with time averaged structure factor (fit of time av)
            'sfac_av': include only fits with spatially averaged structure factor   (time av of fits)

            """

            suffix = 'dens' if use_density_fit else 'count'
            output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)
            act_list = self.act_list[Ndataset]

            fluc_path = os.path.join(output_path, f'alpha_list_{suffix}.npy')
            sfac_path = os.path.join(output_path, f'alpha_list_sfac.npy')
            sfac_time_av_path = os.path.join(output_path, f'fit_params_sfac_time_av.npy')


            file_name_list = []
            label_list = []

            if 'all' in include:
                file_name_list = [fluc_path, sfac_path, sfac_time_av_path,]
                label_list = [r'$\overline{\delta \rho ^2}$ (time av. of fits)', rf'$S(k)$ (time av. of fits)', rf'$S(k)$ (fit of time av.)']
            else:
                for val in include:
                    if val == 'fluc':
                        file_name_list.append(fluc_path)
                        label_list.append(r'$\overline{\delta \rho ^2}$ (time av. of fits)')
                    elif val == 'sfac_all':
                        file_name_list.extend([sfac_path, sfac_time_av_path,])
                        label_list.extend([rf'$S(k)$ (time av. of fits)', rf'$S(k)$ (fit of time av.)'])
                    elif val == 'sfac_av':
                        file_name_list.append(sfac_path)
                        label_list.append(rf'$S(k)$ (time av. of fits)')
                    elif val == 'sfac_time_av':
                        file_name_list.append(sfac_time_av_path)
                        label_list.append(rf'$S(k)$ (fit of time av.)')
                    else:
                        print(f'Unknown option {val}.')
                        return
    
            fig, ax = plt.subplots(figsize=(9, 6))

            for i, file_name in enumerate(file_name_list):
                try:
                    alpha_list = np.load(file_name)
                except:
                    print(f'File {file_name} not found.')
                    continue

                alpha_std = alpha_list[:, 1] if alpha_list.shape[1] == 2 else alpha_list[:, 2]
                ax.errorbar(act_list, alpha_list[:, 0], alpha_std, fmt = '.-', capsize=2, label = label_list[i], \
                            capthick=1, elinewidth=1, markeredgewidth=2, alpha = 0.5, markersize=4,)
  
            ax.legend(ncol=2, fontsize=12)
            ax.set_xlabel(r'Activity ($\zeta$)')
            ax.set_ylabel(r'Hyperuniformity exponent ($\overline{\alpha}$)')
            ax.set_title(rf'Time av. of $\alpha $ vs activity (L = {self.LX[Ndataset]})')
            fig.tight_layout()
            
            if save:
                if not os.path.isdir(os.path.join(output_path, 'figs')):
                    os.makedirs(os.path.join(output_path, 'figs'))
                fig_name = 'alpha_mean.png' if fig_name is None else fig_name
                fig.savefig(os.path.join(output_path, f'figs\\{fig_name}'), dpi = 420, pad_inches=0.15)
            return fig, ax

    def plot_hyperuniformity_sfac(self, act_list = None, fit_params = None, Ndataset = 0, act_idx_bounds = None, use_merged = False, save = False):

        
        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        if act_idx_bounds is None:
            act_idx_bounds = [0, len(self.act_list[Ndataset])]

        if act_list is None:
            act_list = self.act_list[Ndataset][act_idx_bounds[0]:act_idx_bounds[1]]

        if fit_params is None:
            fit_params = np.load(os.path.join(output_path, f'fit_params_sfac.npy'))[:, act_idx_bounds[0]:act_idx_bounds[1], :]

        if not len(act_list) == fit_params.shape[1]:
            print('The number of activities does not match the number of activities in the fit_params array.')
            return
        
        Nframes = fit_params.shape[0]
        Nparams = int(fit_params.shape[-1] / 2)
        Ngroup = 5

        ncols = 4
        nrows = int(np.ceil(len(act_list) / ncols))
        height = nrows * 4
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20,height))
        ax = ax.flatten()  

        for i, act in enumerate(act_list):
            if i == 0:
                label3 = 'Weighted mean'
                label4 = 'Grouped mean'
            else:
                label3 = None
                label4 = None

            ax[i].errorbar(np.arange(self.conv_list[Ndataset][i], Nframes), \
                        fit_params[self.conv_list[Ndataset][i]:,i,0], \
                            fit_params[self.conv_list[Ndataset][i]:,i,Nparams], \
                            fmt='.', \
                        alpha = 0.3, markersize=2,)

            x = fit_params[self.conv_list[Ndataset][i]:, i, 0]
            max_idx = len(x) - len(x) % Ngroup         
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                x_group = np.nanmean(x[:max_idx].reshape(-1, Ngroup), axis=1)
                dx_group = np.nanstd(x[:max_idx].reshape(-1, Ngroup), axis=1, ddof=min(Ngroup-2, 1)) / np.sqrt(Ngroup)
                x_frames = np.arange(self.conv_list[Ndataset][i], self.conv_list[Ndataset][i] + max_idx, Ngroup) + Ngroup / 2

                if len(x_group < 10) > 10:
                    ax[i].plot(x_frames, x_group, '.-', color='orange', alpha=0.8, markersize=4, label = label4)
                
                mean, err, Ndof, chi2, p_val = calc_weighted_mean_vec(x, fit_params[self.conv_list[Ndataset][i]:, i, Nparams])
            
            ax[i].axhline(mean, color='green', linestyle='--', alpha=0.5, lw = 2, label = label3)
            ax[i].text(0.05, 0.97, rf'$\zeta$ = {act}', transform=ax[i].transAxes, fontsize=12, verticalalignment='top')
           
        fig.suptitle(f'Hyperuniformity exponent for different activities (L = {self.LX[Ndataset]})', y=.995)
        fig.supxlabel('Frame')
        fig.supylabel(rf'$\alpha$ (est. from $S(k)$)', x = 0.005)
        fig.legend(ncol=2)
        fig.tight_layout()

        if save:    
            if not os.path.isdir(os.path.join(output_path, 'figs')):
                os.makedirs(os.path.join(output_path, 'figs'))
            fig.savefig(os.path.join(output_path, f'figs\\alpha_sfac.png'), dpi = 420)
        return fig, ax

    def plot_pair_corr_function_time_av(self, Ndataset = 0, act_idx_bounds = None, use_merged = False, save = False,):
        """
        Plot pair correlation function
        """

        output_path, Ndataset = self.__get_outpath_path(Ndataset, use_merged)

        act_list = self.act_list[Ndataset]

        if act_idx_bounds is None:
            act_idx_bounds = [0, len(act_list)]
        try:
            r, pcf_arr = self.get_sfac_pcf(Ndataset = Ndataset, time_av = True, use_merged = use_merged)[2:]
            pcf_av = pcf_arr[:, act_idx_bounds[0]:act_idx_bounds[1], 0]
            pcf_std = pcf_arr[:, act_idx_bounds[0]:act_idx_bounds[1], 1]
        except:
            print("No pair correlation function data provided. Analyse defects first.")
            return

        title = f"Time av. pair correlation function (L = {self.LX[Ndataset]})" 
        
        fig, ax = plt.subplots()
        for i, act in enumerate(act_list[act_idx_bounds[0]:act_idx_bounds[1]]):
            ax.errorbar(r, pcf_av[:, i], yerr = pcf_std[:, i], fmt = '.', markersize = 4, alpha = 0.5, label = rf'$\zeta$ = {act}')

        ax.set_xlabel(rf"$r$ (radius of observation window)")
        ax.set_ylabel(rf"$g(r)$")
        ax.set_title(title)
        ax.legend(ncol=2)
        fig.tight_layout()

        if save:
            if not os.path.isdir(os.path.join(output_path, 'figs')):
                os.makedirs(os.path.join(output_path, 'figs'))
            fig.savefig(os.path.join(output_path, f'figs\\pcf.png'), dpi = 420, pad_inches=0.25)
        return fig, ax





def order_param_func(def_arr, av_defects, LX, shift_by_def = None, shift = False):

    if isinstance(shift_by_def, float):
        av_def_max = shift_by_def
    else:
        av_def_max = 0
    if shift:
        order_param = def_arr - av_def_max
    else:
        order_param = def_arr 
    order_param /= np.sqrt(av_defects[:,0][None, :, None])
    return order_param
         

def main():
    do_extraction = False
    do_basic_analysis = True
    do_hyperuniformity_analysis = True
    do_merge = True

    system_size_list = [256, 512, 1024, 2048]
    #system_size_list = [2048]
    mode = 'all' # 'all' or 'short'

    # order parameter parameters
    Nscale = True

    # hyperuniformity parameters
    act_idx_bounds=[0,None]
    Npoints_to_fit = 8
    Nbounds = [[3,n] for n in range(5,9)]

    dens_fluc_dict = dict(fit_densities = True, act_idx_bounds = [0, None], weighted_mean = False, window_idx_bounds = [30 - Npoints_to_fit, None])


    for i, LL in enumerate(system_size_list):
        print('\nStarting analysis for L =', LL)
        time0 = time.time()
        output_path = f'data\\nematic_analysis{LL}_LL0.05'
        
        defect_list = gen_analysis_dict(LL, mode)
        ad = AnalyseDefects(defect_list, output_path=output_path)

        if do_extraction:
            ad.extract_results()
        if do_basic_analysis:
            if do_hyperuniformity_analysis:
                sfac_dict = dict(Npoints_bounds = Nbounds[i], act_idx_bounds = act_idx_bounds,)
                ad.analyze_defects(dens_fluc_dict=dens_fluc_dict, sfac_dict=sfac_dict)
            else:
                ad.analyze_defects()
        if do_merge:
            ad.merge_results()

        print(f'Analysis for L = {LL} done in {time.time() - time0:.2f} s.\n\n')


if __name__ == "__main__":
    main()




if 0:
    def __calc_av_over_exp_and_time(self, data_arr, Ndataset = 0, ddof = 1, return_arr = False, save_name = None, save = True,):
        """
        data_arr: array of shape (Nframes, Nactivity, Nexp)
        """
        N = Ndataset
        conv_list = self.conv_list[N]
    
        if save:
            if save_name is None:
                print('No save name given. Data will not be saved.')
                return
            
        av = np.nan * np.zeros((data_arr.shape[1:-1]))
        std = np.nan * np.zeros_like(av)

        for i, _ in enumerate(self.act_list[N]):
            sfac_time_av[:, i, 0] = np.nanmean(sfac[self.conv_list[N][i]:, :, 0, i, :], axis = (-1, 0))
            sfac_time_av[:, i, 1] = np.nanstd(sfac[self.conv_list[N][i]:, :, 0, i, :], axis = (-1, 0), ddof = ddof) / np.sqrt(Nexp * (Nframes - self.conv_list[N][i]))
        if save:
            np.save(os.path.join(self.output_paths[N], 'sfac_time_av.npy'), sfac_time_av)
        av = np.expand_dims(np.nanmean(data_arr, axis = -1), axis = -1)
        std = np.expand_dims(np.nanstd(data_arr, ddof = ddof, axis = -1) / np.sqrt(data_arr.shape[-1]), axis = -1)
        output_arr = np.concatenate([av, std], axis = -1)
        if save:
            np.save(os.path.join(self.output_paths[Ndataset], save_name + '.npy'), output_arr)
        return output_arr if return_arr else None

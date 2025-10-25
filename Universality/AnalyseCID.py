# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------
import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from utils_plot import *

plt.style.use('sg_article')
plt.rcParams.update({"text.usetex": True,})


class AnalyseCID:
    """
    General-purpose class for managing CID data extraction and analysis.
    """

    def __init__(self, params: dict):
        """
        Parameters
        ----------
        params : dict
            Dictionary containing configuration options.
            Required keys:
                - base_path : str
                - save_path : str
                - output_suffix : str
                - L_list : list[int]
                - Nexp_list : list[int]
            Optional keys:
                - data_suffix : str (default '') - save paths are found as save_path{LX}{data_suffix}
                - figs_save_path : str (default None) - path to save figures
                - act_exclude_dict : dict[int, list]
                - uncertainty_multiplier : float (default 1.0)
                - verbose : bool (default True)
                - ddof : int (default 1)
        """

        required = ["base_path", "save_path", "output_suffix", "L_list", "Nexp_list"]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        self.base_path = params["base_path"]
        self.save_path = params["save_path"]
        self.output_suffix = params["output_suffix"]

        self.L_list = params["L_list"]
        self.Nexp = {LX: params["Nexp_list"][i] for i, LX in enumerate(self.L_list)}
        
        self.act_exclude = params.get("act_exclude_dict", {LX: [] for LX in self.L_list})
        self.data_suffix = params.get("data_suffix", "")
        self.figs_save_path = params.get("figs_save_path", None)
        self.verbose = params.get("verbose", True)

        self.uncertainty_multiplier = params.get("uncertainty_multiplier", 1.0)
        self.ddof = params.get("ddof", 1)

        self.base_paths = {LX: f'{self.base_path}{LX}{self.data_suffix}' for LX in self.L_list}
        self.save_paths = {LX: f'{self.save_path}{LX}{self.data_suffix}' for LX in self.L_list}

        self.cid_params = self.__load_cid_params()
        
        # --- Initialize data containers ---
        self.act = {}
        self.conv = {}

        self.cid = {}
        self.cid_shuffle = {}
        self.frac = {}
        self.cid_tav = {}
        self.cid_shuffle_tav = {}
        self.frac_tav = {}
        self.cid_var = {}
        self.frac_var = {}
        self.dcid = {}
        self.dfrac = {}

        self.__load_data()

    def __load_cid_params(self,):
        """Load CID parameters for each system size LX."""
        
        cid_params = {}
        file_name = f'cid_params{self.output_suffix}.pkl'

        for LX in self.L_list:
            save_path = self.save_paths[LX]
            os.makedirs(save_path, exist_ok=True)
            try:
                with open(os.path.join(save_path, file_name), 'rb') as f:
                    cid_params[LX] = pkl.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"CID parameters file not found for LX={LX} at {save_path}")
        return cid_params

    def __load_data_full(self, cap_cid=True):
        """Load full data arrays for each system size LX."""
 
        for LX in self.L_list:
            save_path = self.save_paths[LX]
            os.makedirs(save_path, exist_ok=True)

            try:
                data_npz = np.load(os.path.join(save_path, f'cid_data{self.output_suffix}.npz'), allow_pickle=True)
            except FileNotFoundError:
                raise FileNotFoundError(f"CID data file not found for LX={LX} at {save_path}")
            try:
                self.conv[LX] = np.load(os.path.join(save_path, f'conv_list_cubes{self.output_suffix}.npy'), allow_pickle=True)
            except: pass

            self.act[LX] = data_npz["act_list"]
            self.cid[LX] = data_npz["cid"]
            self.cid_shuffle[LX] = data_npz["cid_shuffle"]

            frac_arr = data_npz["cid_frac"]
            if cap_cid:
                frac_mask = frac_arr[...,0] > 1
                frac_arr[...,0][frac_mask] = 1.0
            self.frac[LX] = frac_arr  
        return

    def __load_data_processed(self,):
        """Load processed data for each system size LX."""
 
        for LX in self.L_list:
            save_path = self.save_paths[LX]
            os.makedirs(save_path, exist_ok=True)

            try:
                tav_npz = np.load(os.path.join(save_path, f'cid_time_av{self.output_suffix}.npz'), allow_pickle=True)
            except Exception:
                print(f"CID time average data not found for LX={LX} at {save_path}, calculating...")
                self.analyze()

            tav_npz = np.load(os.path.join(save_path, f'cid_time_av{self.output_suffix}.npz'), allow_pickle=True)
            self.conv[LX] = tav_npz["conv_list"]
            self.cid_tav[LX] = tav_npz["cid_time_av"]
            self.cid_shuffle_tav[LX] = tav_npz["cid_shuffle_time_av"]
            self.frac_tav[LX] = tav_npz["cid_frac_time_av"]
            self.cid_var[LX] = tav_npz["cid_var_per_exp"] if self.Nexp[LX] > 1 else tav_npz["cid_var"]
            self.frac_var[LX] = tav_npz["cid_frac_var_per_exp"] if self.Nexp[LX] > 1 else tav_npz["cid_frac_var"]
            self.dcid[LX] = tav_npz["cid_derivative"]
            self.dfrac[LX] = tav_npz["cid_frac_derivative"]
        return
    
    def __load_data(self,):
        """Load CID data for each system size LX."""
        self.__load_data_full()
        try:
            self.__load_data_processed()
        except:
            print("Re-analyzing to compute derivatives...")
            self.analyze()
        return
    
    def get_moments(self):
        """Calculate moments of CID and divergence for each system size LX.

        Returns
        -------
        moment_dict : dict
            Dictionary mapping system size to CID moments.
            div_moment_dict : dict
            Dictionary mapping system size to divergence moments."""
        
        moment_dict = {}
        div_moment_dict = {}
        for LX in self.L_list:
            moment_dict[LX] = calc_moments(self.cid[LX][...,0], self.conv[LX],)
            div_moment_dict[LX] = calc_moments(1 - self.frac[LX][...,0], self.conv[LX],)     
        return moment_dict, div_moment_dict

    def extract(self, extractor_func=extract_cid_results, reload=False):
        """Extract CID data for each system size LX.
        
        Parameters
        ----------
        extractor_func : function
            Function to use for extraction. Defaults to `extract_cid_results`.
        reload : bool
            Whether to reload data from disk or use cached data. Defaults to True.
        """
  
        for LX in self.L_list:
            base_path = self.base_paths[LX]
            save_path = self.save_paths[LX]

            info_dict = {"base_path": base_path,
                        "save_path": save_path,
                        "output_suffix": self.output_suffix,
                        "act_exclude_list": self.act_exclude[LX],
                        "LX": LX,
                        "nexp": self.Nexp[LX],}

            if self.verbose: print(f"[Extract] LX={LX} | data_path={base_path}")

            conv_list = np.load(os.path.join(save_path, "conv_list.npy"), allow_pickle=True)
            extractor_func(info_dict, verbose=self.verbose)
            gen_conv_list(conv_list, self.output_suffix, save_path)
   
        if reload: self.__load_data_full()
        return

    def analyze(self,):
        """Analyze CID data to compute time averages and variances."""
        
        multiplier = self.uncertainty_multiplier
        ddof = self.ddof

        for LX in self.L_list:
            save_path = self.save_paths[LX]
  
            act = self.act[LX]
            conv = self.conv[LX]
            Nexp = self.Nexp[LX]
            
            cid_tav, cid_var, cid_var_per_exp = calc_time_avs_ind_samples(self.cid[LX][..., 0], \
                                                conv, Nexp=Nexp, unc_multiplier=multiplier, ddof=ddof)
            cid_shuffle_tav, cid_shuffle_var, cid_shuffle_var_per_exp = calc_time_avs_ind_samples(self.cid_shuffle[LX][..., 0], \
                                                conv, Nexp=Nexp, unc_multiplier=multiplier, ddof=ddof)
            cid_frac_tav, cid_frac_var, cid_frac_var_per_exp = calc_time_avs_ind_samples(self.frac[LX][..., 0], \
                                                conv, Nexp=Nexp, unc_multiplier=multiplier, ddof=ddof)

            # calculate derivatives
            cid_deriv = np.nan * np.ones_like(cid_tav)[:-1]
            frac_deriv = np.nan * np.ones_like(cid_frac_tav)[:-1]

            dcid, dcid_err = calc_derivative(act, cid_tav[:,0], cid_tav[:,1])[1:]
            dfrac, dfrac_err = calc_derivative(act, cid_frac_tav[:,0], cid_frac_tav[:,1])[1:]
            cid_deriv[:,0] = dcid
            cid_deriv[:,1] = dcid_err
            frac_deriv[:,0] = dfrac
            frac_deriv[:,1] = dfrac_err

            np.savez_compressed(
                os.path.join(save_path, f"cid_time_av{self.output_suffix}.npz"),
                cid_time_av=cid_tav, 
                cid_var=cid_var,
                cid_var_per_exp=cid_var_per_exp,
                cid_shuffle_time_av=cid_shuffle_tav,
                cid_shuffle_var=cid_shuffle_var,
                cid_shuffle_var_per_exp=cid_shuffle_var_per_exp,
                cid_frac_time_av=cid_frac_tav,
                cid_frac_var=cid_frac_var,
                cid_frac_var_per_exp=cid_frac_var_per_exp,
                cid_derivative=cid_deriv,
                cid_frac_derivative=frac_deriv,
                act_list=self.act[LX], 
                conv_list=self.conv[LX],)
        self.__load_data_processed()
        return

    def run(self, extractor_func=extract_cid_results):
        """Convenience method to run extraction + analysis."""
        if self.verbose:
            print("=== Starting CID analysis ===")
        self.extract(extractor_func=extractor_func,reload=True)
        self.analyze()
        if verbose:
            print("=== Analysis complete ===")
        if self.verbose:
            print("=== Extraction & Analysis complete ===")

    def plot_cid_and_deriv(self, L_list=None,xlims=None, plot_abs=False, save_path=None):
        """Plot CID and its derivative with respect to activity."""
        save_path = save_path if save_path is not None else self.figs_save_path
        fig, ax = plot_cid_and_derivative(
                    L_list=L_list if L_list is not None else self.L_list, 
                    act_dict=self.act, 
                    cid_time_av_dict=self.cid_tav, 
                    dcid_dict=self.dcid, 
                    xlims=xlims, plot_abs=plot_abs, 
                    savepath=os.path.join(save_path, 'cid_dcid.pdf'))
        return fig, ax

    def plot_div_and_deriv(self, L_list=None, xlims=None, plot_abs=False, save_path=None):
        """Plot divergence and its derivative with respect to activity."""
        save_path = save_path if save_path is not None else self.figs_save_path
        fig, ax = plot_div_and_derivative(
                    L_list=L_list if L_list is not None else self.L_list, 
                    act_dict=self.act, 
                    frac_time_av_dict=self.frac_tav, 
                    dfrac_dict=self.dfrac, 
                    xlims=xlims, 
                    plot_abs=plot_abs, 
                    savepath=os.path.join(save_path, 'div_ddiv.pdf'))
        return fig, ax
    
    def plot_cid_fluc(self, L_list=None, xlims=None, plot_abs=False, save_path=None):
        """Plot CID fluctuations and their derivative with respect to activity."""
        save_path = save_path if save_path is not None else self.figs_save_path
        fig, ax = plot_cid_fluctuations(
            L_list=L_list if L_list is not None else self.L_list, 
            act_dict=self.act, 
            cid_time_av_dict=self.cid_tav, 
            cid_var_dict=self.cid_var, 
            dcid_dict=self.dcid, 
            xlims=xlims, plot_abs=plot_abs, 
            savepath=os.path.join(save_path, 'cid_fluc.pdf'))
        return fig, ax

    def plot_div_fluc(self, L_list=None, xlims=None, plot_div_per=True, plot_abs=False, save_path=None):
        """Plot divergence fluctuations and their derivative with respect to activity."""
        save_path = save_path if save_path is not None else self.figs_save_path
        fig, ax = plot_div_fluctuations(
            L_list=L_list if L_list is not None else self.L_list, 
            act_dict=self.act, 
            frac_time_av_dict=self.frac_tav, 
            dfrac_dict=self.dfrac,
            div_var_dict=self.frac_var,
            xlims=xlims, 
            plot_div_per=plot_div_per,
            plot_abs=plot_abs,
            savepath=os.path.join(save_path, 'div_fluc.pdf'))
        return fig, ax

    def plot_cid_moments(self, L_list=None, xlims=None, plot_binder=False, save_path=None):
        """Plot moments of CID and divergence."""
        save_path = save_path if save_path is not None else self.figs_save_path
        moment_dict, _ = self.get_moments()
        fig, ax = plot_moments(
            moment_dict, 
            act_dict=self.act, 
            L_list=L_list if L_list is not None else self.L_list,
            xlims=xlims,
            moment_label=r'CID', 
            plot_binder=plot_binder, 
            savepath=os.path.join(save_path, f'cid_moments.pdf'))
        return fig, ax

    def plot_div_moments(self, L_list=None, xlims=None, plot_binder=False, save_path=None):
        """Plot moments of divergence."""
        save_path = save_path if save_path is not None else self.figs_save_path
        _, div_moment_dict = self.get_moments()
        fig, ax = plot_moments(
            div_moment_dict, 
            act_dict=self.act, 
            L_list=L_list if L_list is not None else self.L_list,
            xlims=xlims,
            moment_label=r'$\mathcal{D}$', 
            plot_binder=plot_binder, 
            savepath=os.path.join(save_path, f'div_moments.pdf'))
        return fig, ax
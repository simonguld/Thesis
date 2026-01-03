# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

import os
import pickle as pkl

from pathlib import Path
from typing import Dict, List, Any

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

    def __init__(self, params: dict, load_data=True):
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

        self.params = params  # store for debugging/reproducibility
        
        required = ["base_path", "save_path", "output_suffix", "L_list", "Nexp_list"]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        # --- Initialize system sizes and experiment counts ---
        self.L_list: List[int] = params["L_list"]
        Nexp_list: List[int] = params["Nexp_list"]
        if len(self.L_list) != len(Nexp_list): raise ValueError("L_list and Nexp_list must be the same length.")
        self.Nexp = dict(zip(self.L_list, Nexp_list))

        # --- Initialize activity exclusion dict ---
        self.act_exclude = params.get("act_exclude_dict", {LX: [] for LX in self.L_list})

        # --- Initialize paths ---
        self.base_path = params["base_path"]
        self.save_path = params["save_path"]
        self.output_suffix = params["output_suffix"]
        self.figs_save_path = params.get("figs_save_path", None)
        self.data_suffix = params.get("data_suffix", "")

        self.base_paths = {LX: f'{self.base_path}{LX}{self.data_suffix}' for LX in self.L_list}
        self.save_paths = {LX: f'{self.save_path}{LX}{self.data_suffix}' for LX in self.L_list}

        # --- Initialize analysis parameters ---
        self.uncertainty_multiplier = params.get("uncertainty_multiplier", 1.0)
        self.ddof = params.get("ddof", 1)
        self.verbose = params.get("verbose", True)

        # --- Initialize data containers ---
        container_keys = [
            "act", "conv",
            "cid", "cid_shuffle", "frac", "cidm", "fracm",
            "cid_tav", "cid_shuffle_tav", "frac_tav", "cidm_tav", "fracm_tav",
            "cid_var", "frac_var", "cidm_var", "fracm_var",
            "dcid", "dfrac", "dcidm", "dfracm",
        ]
        for key in container_keys:
            setattr(self, key, {})

        # Keep track if minmax data is loaded
        self.minmax_loaded = {LX: False for LX in self.L_list}

        if load_data:
            self.cid_params = self._load_cid_params()
            self._load_data()

    def _load_cid_params(self,):
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
    
    def _assign_processed_fields(self, LX, data):
        """Assign processed data fields for a given system size LX."""
        
        self.conv[LX] = data["conv_list"]
        self.cid_tav[LX] = data["cid_time_av"]
        self.cid_shuffle_tav[LX] = data["cid_shuffle_time_av"]
        self.frac_tav[LX] = data["cid_frac_time_av"]
        self.dcid[LX] = data["cid_derivative"]
        self.dfrac[LX] = data["cid_frac_derivative"]

        if self.Nexp[LX] > 1:
            self.cid_var[LX] = data["cid_var_per_exp"]
            self.frac_var[LX] = data["cid_frac_var_per_exp"]
        else:
            self.cid_var[LX] = data["cid_var"]
            self.frac_var[LX] = data["cid_frac_var"]

        if self.minmax_loaded[LX]:
            self.cidm_tav[LX] = data["cidm_time_av"]
            self.fracm_tav[LX] = data["cid_fracm_time_av"]
            self.dcidm[LX] = data["cidm_derivative"]
            self.dfracm[LX] = data["cid_fracm_derivative"]
            if self.Nexp[LX] > 1:
                self.cidm_var[LX] = data["cidm_var_per_exp"]
                self.fracm_var[LX] = data["cid_fracm_var_per_exp"]
            else:
                self.cidm_var[LX] = data["cidm_var"]
                self.fracm_var[LX] = data["cid_fracm_var"]
        return

    def _load_data_full(self, cap_cid=True):
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
                frac_arr[...,1][frac_mask] = 0.0
            self.frac[LX] = frac_arr  

            if 'cid_minmax' in data_npz:
                self.minmax_loaded[LX] = True
                self.cidm[LX] = data_npz["cid_minmax"]
                fracm_arr = data_npz["frac_minmax"]
                if cap_cid:
                    fracm_mask = fracm_arr[...,0] > 1
                    fracm_arr[...,0][fracm_mask] = 1.0
                    fracm_arr[...,1][fracm_mask] = 0.0  
                self.fracm[LX] = fracm_arr
        return

    def _load_data_processed(self,):
        """Load processed data for each system size LX."""
 
        for LX in self.L_list:
            save_path = self.save_paths[LX]
            os.makedirs(save_path, exist_ok=True)

            npz_path = os.path.join(save_path, f'cid_time_av{self.output_suffix}.npz')
            if not os.path.exists(npz_path):
                if self.verbose:
                    print(f"[INFO] Processed CID data missing for L={LX}, running analyze()...")
                self.analyze()

            tav_npz = np.load(npz_path, allow_pickle=True)
            self._assign_processed_fields(LX, tav_npz)
        return
    
    def _load_data(self,):
        """Load CID data for each system size LX."""
        self.cid_params = self._load_cid_params()
        self._load_data_full()
        try:
            self._load_data_processed()
        except:
            print("Re-analyzing to compute derivatives...")
            self.analyze()
        return

    def get_moments(self, use_min=False):
        """
        Calculate first 4 moments of CID and divergence for each system size LX.

        Returns:
        -------
        moment_dict : dict
            Dictionary mapping system size to CID moments. Shape (4, num_activities).
            div_moment_dict : dict
            Dictionary mapping system size to divergence moments. Shape (4, num_activities).
        """
        
        cid_vals = self.cidm if use_min else self.cid
        frac_vals = self.fracm if use_min else self.frac
        moment_dict = {}
        div_moment_dict = {}
        for LX in self.L_list:
            if use_min:
                if not self.minmax_loaded[LX]:
                    continue
            moment_dict[LX] = calc_moments(cid_vals[LX][...,0], Nexp=self.Nexp[LX], conv_list=self.conv[LX])
            div_moment_dict[LX] = calc_moments(1 - frac_vals[LX][...,0], Nexp=self.Nexp[LX], conv_list=self.conv[LX])
        return moment_dict, div_moment_dict

    def extract(self, extractor_func=None, conv_list_dir=None, reload=True):
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
            os.makedirs(save_path, exist_ok=True)

            if extractor_func is None:
                extractor_function = extract_cid_results if self.Nexp[LX] > 1 else extract_cid_results_single
            else:
                extractor_function = extractor_func

            info_dict = {"base_path": base_path,
                        "save_path": save_path,
                        "output_suffix": self.output_suffix,
                        "act_exclude_list": self.act_exclude[LX],
                        "LX": LX,
                        "nexp": self.Nexp[LX],}

            if self.verbose: print(f"[Extract] LX={LX} | data_path={base_path}")

            # Run extractor function
            extractor_function(info_dict, verbose=self.verbose)

            # Convert conv_list to cubes format
            try:
                conv_list = np.load(os.path.join(save_path if conv_list_dir is None else conv_list_dir, "conv_list.npy"), allow_pickle=True)
                if not os.path.exists(os.path.join(save_path, "conv_list.npy")):
                    np.save(os.path.join(save_path, "conv_list.npy"), conv_list)
            except Exception as e:
                print(f"Error loading conv_list for LX={LX}: {e}")
                print("initializing all-zero conv_list.")
                self._load_data_full()
                conv_list = np.zeros((len(self.act[LX]),), dtype=int)

        gen_conv_list(conv_list, self.output_suffix, save_path)
        if reload: self._load_data_full()
        return

    def analyze(self, use_central_diff=True):
        """Analyze CID data to compute time averages and variances."""
        
        func_deriv = calc_central_derivative if use_central_diff else calc_forward_derivative

        # Load unprocessed data if not already loaded
        self._load_data_full()

        multiplier = self.uncertainty_multiplier
        ddof = self.ddof

        for LX in self.L_list:
            save_path = self.save_paths[LX]
            save_file = os.path.join(save_path, f"cid_time_av{self.output_suffix}.npz")
  
            act = self.act[LX]
            conv = self.conv[LX]
            Nexp = self.Nexp[LX]
            
            # Helper to reduce repetition
            def compute_avs(data):
                return calc_time_avs_ind_samples(data[..., 0],
                    conv, Nexp=Nexp, unc_multiplier=multiplier,
                    ddof=ddof,)
            def compute_avs_dict(data, prefix):
                tav, var, var_exp = compute_avs(data)
                return {f"{prefix}_time_av": tav,f"{prefix}_var": var,f"{prefix}_var_per_exp": var_exp,}
            
            data_to_save = {
                "act_list": act,
                "conv_list": conv,
            }

            data_to_save.update(compute_avs_dict(self.cid[LX], "cid"))
            data_to_save.update(compute_avs_dict(self.cid_shuffle[LX], "cid_shuffle"))
            data_to_save.update(compute_avs_dict(self.frac[LX], "cid_frac"))
            
            if self.minmax_loaded[LX]:
                data_to_save.update(compute_avs_dict(self.cidm[LX], "cidm"))
                data_to_save.update(compute_avs_dict(self.fracm[LX], "cid_fracm"))

            data_to_save["cid_derivative"] = func_deriv(act, data_to_save["cid_time_av"][:,0], data_to_save["cid_time_av"][:,1])
            data_to_save["cid_frac_derivative"] = func_deriv(act, data_to_save["cid_frac_time_av"][:,0], data_to_save["cid_frac_time_av"][:,1])
            
            if self.minmax_loaded[LX]:
                data_to_save["cidm_derivative"] = func_deriv(act, data_to_save["cidm_time_av"][:,0], data_to_save["cidm_time_av"][:,1])
                data_to_save["cid_fracm_derivative"] = func_deriv(act, data_to_save["cid_fracm_time_av"][:,0], data_to_save["cid_fracm_time_av"][:,1])

            np.savez_compressed(save_file, **data_to_save)
            if self.verbose:
                print(f"[Analyze] LX={LX} | saved processed data to {save_file}")

        self._load_data_processed()
        return

    def run(self, extractor_func=None, conv_list_dir=None):
        """Convenience method to run extraction + analysis."""
        if self.verbose:
            print("=== Starting CID analysis ===")
        self.extract(extractor_func=extractor_func,reload=True, conv_list_dir=conv_list_dir)
        self.analyze()
        if self.verbose:
            print("=== Extraction & Analysis complete ===")
        return

    def plot_cid_and_deriv(self, L_list=None, act_critical=None, xlims=None, use_min=False, plot_abs=False, save_path=None):
        """Plot CID and its derivative with respect to activity."""
        save_path = save_path if save_path is not None else self.figs_save_path
        abs_suffix = '_abs' if plot_abs else ''
        min_suffix = 'm' if use_min else ''
        fig, ax = plot_cid_and_derivative(
                    L_list=L_list if L_list is not None else self.L_list, 
                    act_dict=self.act, 
                    cid_time_av_dict=self.cid_tav if not use_min else self.cidm_tav, 
                    dcid_dict=self.dcid if not use_min else self.dcidm, 
                    act_critical=act_critical,
                    xlims=xlims, 
                    plot_abs=plot_abs, 
                    savepath=os.path.join(save_path, f'cid{min_suffix}_dcid{abs_suffix}.pdf') if save_path is not None else None)
        return fig, ax

    def plot_div_and_deriv(self, L_list=None, act_critical=None, use_min=False, xlims=None, plot_abs=False, save_path=None):
        """Plot divergence and its derivative with respect to activity."""
        save_path = save_path if save_path is not None else self.figs_save_path
        abs_suffix = '_abs' if plot_abs else ''
        min_suffix = 'm' if use_min else ''
        fig, ax = plot_div_and_derivative(
                    L_list=L_list if L_list is not None else self.L_list, 
                    act_dict=self.act, 
                    frac_time_av_dict=self.frac_tav if not use_min else self.fracm_tav, 
                    dfrac_dict=self.dfrac if not use_min else self.dfracm,
                    act_critical=act_critical, 
                    xlims=xlims, 
                    plot_abs=plot_abs, 
                    savepath=os.path.join(save_path, f'div{min_suffix}_ddiv{abs_suffix}.pdf') if save_path is not None else None)
        return fig, ax
    
    def plot_cid_fluc(self, L_list=None, act_critical=None, use_min=False, xlims=None, plot_abs=False, save_path=None):
        """Plot CID fluctuations and their derivative with respect to activity."""
        save_path = save_path if save_path is not None else self.figs_save_path
        abs_suffix = '_abs' if plot_abs else ''
        min_suffix = 'm' if use_min else ''
        fig, ax = plot_cid_fluctuations(
            L_list=L_list if L_list is not None else self.L_list, 
            act_dict=self.act, 
            cid_time_av_dict=self.cid_tav if not use_min else self.cidm_tav, 
            cid_var_dict=self.cid_var if not use_min else self.cidm_var, 
            dcid_dict=self.dcid, 
            act_critical=act_critical,
            xlims=xlims, plot_abs=plot_abs, 
            savepath=os.path.join(save_path, f'cid{min_suffix}_fluc{abs_suffix}.pdf') if save_path is not None else None
        )
        return fig, ax

    def plot_div_fluc(self, L_list=None, act_critical=None, use_min=False, xlims=None, plot_div_per=False, plot_abs=False, save_path=None):
        """Plot divergence fluctuations and their derivative with respect to activity."""
        save_path = save_path if save_path is not None else self.figs_save_path
        abs_suffix = '_abs' if plot_abs else ''
        min_suffix = 'm' if use_min else ''
        fig, ax = plot_div_fluctuations(
            L_list=L_list if L_list is not None else self.L_list, 
            act_dict=self.act, 
            frac_time_av_dict=self.frac_tav if not use_min else self.fracm_tav, 
            dfrac_dict=self.dfrac if not use_min else self.dfracm,
            div_var_dict=self.frac_var,
            act_critical=act_critical,
            xlims=xlims, 
            plot_div_per=plot_div_per,
            plot_abs=plot_abs,
            savepath=os.path.join(save_path, f'div{min_suffix}_fluc{abs_suffix}.pdf') if save_path is not None else None
        )
        return fig, ax

    def plot_cid_moments(self, L_list=None, act_critical=None, use_min=False, xlims=None, plot_binder=False, save_path=None):
        """Plot moments of CID and divergence."""
        save_path = save_path if save_path is not None else self.figs_save_path
        min_suffix = 'm' if use_min else ''
        moment_dict, _ = self.get_moments(use_min=use_min)
        fig, ax = plot_moments(
            moment_dict, 
            act_dict=self.act, 
            L_list=L_list if L_list is not None else self.L_list,
            act_critical=act_critical,
            xlims=xlims,
            moment_label=r'CID', 
            plot_binder=plot_binder, 
            savepath=os.path.join(save_path, f'cid{min_suffix}_moments.pdf') if save_path is not None else None
        )
        return fig, ax

    def plot_div_moments(self, L_list=None, act_critical=None, use_min=False, xlims=None, plot_binder=False, save_path=None):
        """Plot moments of divergence."""
        save_path = save_path if save_path is not None else self.figs_save_path
        min_suffix = 'm' if use_min else ''
        _, div_moment_dict = self.get_moments(use_min=use_min)
        fig, ax = plot_moments(
            div_moment_dict, 
            act_dict=self.act, 
            L_list=L_list if L_list is not None else self.L_list,
            act_critical=act_critical,
            xlims=xlims,
            moment_label=r'$\mathcal{D}$', 
            plot_binder=plot_binder, 
            savepath=os.path.join(save_path, f'div{min_suffix}_moments.pdf') if save_path is not None else None
        )
        return fig, ax
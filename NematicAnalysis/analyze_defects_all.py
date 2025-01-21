# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import sys
import time
import numpy as np

from AnalyseDefects_dev import AnalyseDefects

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

### FUNCTIONS ----------------------------------------------------------------------------------


def gen_analysis_dict_old(LL, mode):

    dshort = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_analysis{LL}_LL0.05', \
              suffix = "short", priority = -1, LX = LL, Nframes = 181)
    dlong = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_analysis{LL}_LL0.05_long', \
                suffix = "long", priority = 1, LX = LL, Nframes = 400)
    priority_vl = 2 if LL == 512 else 3
    dvery_long = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_analysis{LL}_LL0.05_very_long',\
                    suffix = "very_long", priority = priority_vl, LX = LL, Nframes = 1500)
    dvery_long2 = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_analysis{LL}_LL0.05_very_long_v2',\
                    suffix = "very_long2", priority = 3 if priority_vl == 2 else 2, LX = LL, Nframes = 1500)

    if mode == 'all':
        if LL == 2048:
            defect_list = [dshort, dlong]
        else:
            defect_list = [dshort, dlong, dvery_long, dvery_long2] if LL in [256] else [dshort, dlong, dvery_long]
    else:
        defect_list = [dshort]
    
    return defect_list

def gen_analysis_dict(LL, mode):

    dshort = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\na{LL}', \
              suffix = "short", priority = 0, LX = LL, Nframes = 181)
    dlong = dict(path = f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\na{LL}l', \
                suffix = "long", priority = 1, LX = LL, Nframes = 400)
    dvery_long = dict(path =  f'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\na{LL}vl', \
                    suffix = "very_long", priority = 2, LX = LL, Nframes = 1500)
    
    if mode == 'all':
        if LL == 2048:
            defect_list = [dshort, dlong]
        else:
            defect_list = [dshort, dlong, dvery_long]
    else:
        defect_list = [dshort]
    
    return defect_list

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
        
### MAIN ---------------------------------------------------------------------------------------


def main():
    do_extraction = False
    do_basic_analysis = True
    do_hyperuniformity_analysis = True
    do_merge = True

    system_size_list_full = [256, 512, 1024, 2048]
    system_size_list = [2048]
    mode = 'all' # 'all' or 'short'
    count_suffix = '_periodic_rm0.1' #_rm0.1'

    # hyperuniformity parameters
    act_idx_bounds=[0,None]
    Npoints_to_fit = 15
    Nbounds = [[3,n] for n in range(5,9)]
    Nbounds_dict = dict(zip(system_size_list_full, Nbounds))
    dens_fluc_dict = dict(act_idx_bounds = [0, None], window_idx_bounds = [50 - Npoints_to_fit, None])
    
    
    calc_pcf = True
    nlags_list = [750/2, 750/2, 750/2, 400/2]
    temp_corr_simple = True

    for i, LL in enumerate(system_size_list):
        print('\nStarting analysis for L =', LL)
        time0 = time.time()
        output_path = f'data\\na{LL}'
        nlags = nlags_list[system_size_list_full.index(LL)]
        
        defect_list = gen_analysis_dict(LL, mode)
        ad = AnalyseDefects(defect_list, output_path=output_path, count_suffix=count_suffix)
        acf_dict = {'nlags_frac': 0.7, 'nlags': nlags, 'max_lag': None, 'alpha': 0.3174, 'max_lag_threshold': 0, 'simple_threshold': 0.2}

        if do_extraction:
            ad.extract_results()
        if do_basic_analysis:
            if do_hyperuniformity_analysis:
                print(' sfac n bounds:', Nbounds_dict[LL])
                sfac_dict = dict(Npoints_bounds = Nbounds_dict[LL], act_idx_bounds = act_idx_bounds,)
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

        print(f'Analysis for L = {LL} done in {time.time() - time0:.2f} s.\n\n')

if __name__ == "__main__":
    main()


# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import time
import numpy as np

from AnalyseDefects_dev import AnalyseDefects

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

### FUNCTIONS ----------------------------------------------------------------------------------


def gen_analysis_dict(LL, mode):

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
    do_extraction = True
    do_basic_analysis = True
    do_hyperuniformity_analysis = True
    do_merge = True

    system_size_list = [256, 512, 1024, 2048]
    system_size_list = [512]
    mode = 'all' # 'all' or 'short'

    # order parameter parameters
    shift = False
    shift_by_act = 0.022
    Nscale = False

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

            # find density at shift_by_act
            av_def_merged = ad.get_arrays_av(use_merged = True)[-1]
            shift_by_def = av_def_merged[ad.act_list[0].index(shift_by_act)][0]

            order_param_function = lambda def_arr, av_def, LX: order_param_func(def_arr, av_def, LX, shift_by_def = shift_by_def, shift = shift)
            sus_binder_dict = dict(Nscale = Nscale, order_param_func = order_param_function)
            ad.analyze_defects(sus_binder_dict=sus_binder_dict)
        if do_merge:
            ad.merge_results()

        print(f'Analysis for L = {LL} done in {time.time() - time0:.2f} s.\n\n')

if __name__ == "__main__":
    main()


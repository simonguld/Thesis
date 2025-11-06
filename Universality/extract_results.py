# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

import os
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('sg_article')
plt.rcParams.update({"text.usetex": True,})
plt.rcParams['legend.handlelength'] = 0


from AnalyseCID import AnalyseCID
from utils import *
from utils_plot import *

### MAIN ---------------------------------------------------------------------------------------

def main():   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_suffix', type=str, default='')
    parser.add_argument('--extract', type=str2bool, default=False)
    parser.add_argument('--analyze', type=str2bool, default=False)
    parser.add_argument('--plot', type=str2bool, default=False)
    parser.add_argument('--seq', type=str2bool, default=False)
    parser.add_argument("--nbits", type=lambda s: [int(x) for x in s.split(',')], \
                        help='Comma-separated list, e.g. --nbits 2,3,4', default='4')
    parser.add_argument('--cg', type=int, default=4)
    args = parser.parse_args()

    extract = args.extract
    analyze = args.analyze
    plot = args.plot

    nbits_list = args.nbits
    cg = args.cg

    data_suffix = args.data_suffix
    if not data_suffix in ['', 'sd', 's', 'ndg']:
        raise ValueError("data_suffix must be one of '', 'sd', 's', or 'ndg'")

    base_path = f'Z:\\cid\\na'
    save_path = f'data\\nematic\\na'
 
    use_seq = args.seq
    verbose = True

    data_dict = {}
    sd_data_dict = {
        'data_suffix': 'sd',
        'L_list': [512],
        'Nexp_list': [10],
        'act_exclude_dict': {512: []},
        'xlims': None,
        'uncertainty_multiplier': 20,
        'act_critical': 0.022
    }

    s_data_dict = {
    'data_suffix': 's',
    'L_list': [2048],
    'Nexp_list': [3],
    'act_exclude_dict': {2048: []},
    'xlims': None,
    'uncertainty_multiplier': 5,
    'act_critical': 2.1
    }

    na_data_dict = {
        'data_suffix': '',
        'L_list': [512, 1024, 2048],
        'Nexp_list': [5]*3,
        'act_exclude_dict': {512: [0.02, 0.0225], 1024: [], 2048: [0.0225]},
        'xlims': (0.016, 0.045),
        'uncertainty_multiplier': 20,
        'act_critical': 0.022
    }
    ndg_data_dict = {
        'data_suffix': 'ndg',
        'L_list': [1024],
        'Nexp_list': [1],
        'act_exclude_dict': {1024: []},
        'xlims': None,
        'uncertainty_multiplier': 20,
        'act_critical': 7
    }

    data_dict = {'sd': sd_data_dict, 'ndg': ndg_data_dict, '': na_data_dict, 's': s_data_dict}
    fig_folder_dict = {'sd': 'sd', 'ndg': 'ndg', '': 'na', 's': 's'}

    cid_dict = {
        'base_path': base_path,
        'save_path': save_path,
        'cg': cg,
        'verbose': verbose,
        'ddof': 1,
        **data_dict[data_suffix]
    }

    for nbits in nbits_list:
        print(f'\nProcessing for nbits={nbits}, cg={cg}...')

        xlims = data_dict[data_suffix]['xlims']

        output_suffix=f'_seq_nb{nbits}cg{cg}' if use_seq else f'_nb{nbits}cg{cg}'
        cid_dict.update({'output_suffix': output_suffix, 'nbits': nbits})

        fig_folder = fig_folder_dict[data_suffix]
        figs_save_path = f'data\\nematic\\figs\\{fig_folder}\\{output_suffix[1:]}'
        if not os.path.exists(figs_save_path): 
            os.makedirs(figs_save_path)

        # Initialize AnalyseCID object
        ac = AnalyseCID(cid_dict, load_data = not extract)
        ac.figs_save_path = figs_save_path

        if extract:
            ac.run()
        if analyze and not extract:
            ac.analyze()

        if plot:
            plot_abs = False
            act_critical = cid_dict['act_critical']
            if nbits == 7 and data_suffix == '':
                L_list = [1024, 2048]
            else:
                L_list = ac.L_list
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                use_min=False
                #L_list = [1024]
                
                ## Plot cid/div and its derivative with respect to activity
                ac.plot_cid_and_deriv(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, act_critical=act_critical, xlims=xlims, plot_abs=True)
                ac.plot_cid_and_deriv(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, act_critical=act_critical, xlims=xlims, plot_abs=False)
                ac.plot_div_and_deriv(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min,   act_critical=act_critical, xlims=xlims, plot_abs=True)
                ac.plot_div_and_deriv(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min,   act_critical=act_critical, xlims=xlims, plot_abs=False)
                plt.close('all')

                ## Plot moments
                ac.plot_cid_moments(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, xlims=xlims, act_critical=act_critical)
                ac.plot_div_moments(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, xlims=xlims, act_critical=act_critical)
                plt.close('all')

                ## Plot fluctuations
                ac.plot_cid_fluc(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, act_critical=act_critical, xlims=xlims, plot_abs=True)
                ac.plot_cid_fluc(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, act_critical=act_critical, xlims=xlims, plot_abs=False)
                ac.plot_div_fluc(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, act_critical=act_critical, xlims=xlims, plot_abs=False, plot_div_per=False)
                ac.plot_div_fluc(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, act_critical=act_critical, xlims=xlims, plot_abs=True, plot_div_per=False)
                plt.close('all')
    
if __name__ == '__main__':
    main()
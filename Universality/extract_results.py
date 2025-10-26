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
    if not data_suffix in ['', 'sd', 'ndg']:
        raise ValueError("data_suffix must be one of '', 'sd', or 'ndg'")

    base_path = f'Z:\\cid\\na'
    save_path = f'data\\nematic\\na'
 
    use_seq = False
    verbose = True

    data_dict = {}
    sd_data_dict = {
        'data_suffix': 'sd',
        'L_list': [512],
        'Nexp_list': [10],
        'act_exclude_dict': {512: []},
        'xlims': None
    }
    na_data_dict = {
        'data_suffix': '',
        'L_list': [512, 1024, 2048],
        'Nexp_list': [5]*3,
        'act_exclude_dict': {512: [0.02, 0.0225], 1024: [], 2048: [0.0225]},
        'xlims': (0.016, 0.045)
    }
    ndg_data_dict = {
        'data_suffix': 'ndg',
        'L_list': [1024],
        'Nexp_list': [1],
        'act_exclude_dict': {1024: []},
        'xlims': None
    }
    data_dict = {'sd': sd_data_dict, 'ndg': ndg_data_dict, '': na_data_dict}
    fig_folder_dict = {'sd': 'sd', 'ndg': 'ndg', '': 'na'}

    cid_dict = {
        'base_path': base_path,
        'save_path': save_path,
        'cg': cg,
        'verbose': verbose,
        'uncertainty_multiplier': 20,
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                plot_abs = False
                ## Plot cid/div and its derivative with respect to activity
                ac.plot_cid_and_deriv(save_path=ac.figs_save_path, xlims=xlims, plot_abs=plot_abs)
                ac.plot_div_and_deriv(save_path=ac.figs_save_path, xlims=xlims, plot_abs=plot_abs)
                plt.close('all')

                ## Plot moments
                ac.plot_cid_moments(L_list=ac.L_list, save_path=ac.figs_save_path, xlims=xlims,)
                ac.plot_div_moments(L_list=ac.L_list, save_path=ac.figs_save_path, xlims=xlims,)
                plt.close('all')

                ## Plot fluctuations
                ac.plot_cid_fluc(save_path=ac.figs_save_path, xlims=xlims, plot_abs=plot_abs)          
                ac.plot_div_fluc(save_path=ac.figs_save_path, xlims=xlims, plot_abs=plot_abs, plot_div_per=False)
                plt.close('all')
    
if __name__ == '__main__':
    main()
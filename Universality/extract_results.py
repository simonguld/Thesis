# Author:  Simon Guldager
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

from asyncio import windows_events
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
    parser.add_argument('-ds', '--data_suffix', type=str, default='')
    parser.add_argument('-e', '--extract', action='store_true', help='Set to extract data')
    parser.add_argument('-a', '--analyze', action='store_true', help='Set to analyze data')
    parser.add_argument('-p', '--plot', action='store_true', help='Set to produce plots')
    parser.add_argument('-s', '--seq', action='store_true', help='Use sequential mode')
    parser.add_argument('-nb', '--nbits', type=lambda s: [int(x) for x in s.split(',')],
                        help='Comma-separated list, e.g. --nbits 2,3,4', default=None)
    #parser.add_argument('-ws', '--window_size', type=lambda s: [int(x) for x in s.split(',')],
    #                    help='Comma-separated list, e.g. --window_size 256', default=None)
    parser.add_argument('-nf', '--nframes', type=lambda s: [int(x) for x in s.split(',')],
                        help='Comma-separated list, e.g. --nframes 256', default=None)
    parser.add_argument('-ws', '--window_size', type=int, default=None)
    parser.add_argument('-cg', '--cg', type=int, default=4)
    args = parser.parse_args()

    extract = args.extract
    analyze = args.analyze
    plot = args.plot

    window_size = args.window_size
    cg = args.cg

    if window_size is not None:
        size_list = args.nframes
        seq = False
        output_suffix_func = lambda size: f'_nx{window_size}nt{size}cg{cg}'
    else:
        size_list = args.nbits
        seq = args.seq
        output_suffix_func = lambda nbits: f'_seq_nb{nbits}cg{cg}' if seq else f'_nb{nbits}cg{cg}'

    data_suffix = args.data_suffix
    if not data_suffix in ['na', 'na512', 'na1024', 'na2048', 'sd', 's', 'ndg', 'pol', 'abp', 'pols']:
        raise ValueError("data_suffix must be one of 'na',\
                          'na512', 'na1024', 'na2048',\
                         'sd', 's', 'ndg', 'pol', 'abp', 'pols'")

    base_path = f'Z:\\cid\\na'
    save_path = f'data\\nematic\\na'
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
    abp_data_dict = {
        'data_suffix': 'abp',
        'L_list': [256],
        'Nexp_list': [1],
        'act_exclude_dict': {256: []},
        'xlims': None,
        'uncertainty_multiplier': 5,
        'act_critical': None
    }

    s_data_dict = {
    'data_suffix': 's',
    'L_list': [2048],
    'Nexp_list': [3],
    'act_exclude_dict': {2048: []},
    'xlims': None,
    'uncertainty_multiplier': 1,
    'act_critical': 2.1
    }
    na_data_dict = {
        'data_suffix': '',
        'L_list': [512, 1024, 2048],
        'Nexp_list': [5]*3,
        'act_exclude_dict': {512: [0.02, 0.0225, 0.0235], 1024: [], 2048: [0.0225]},
        'xlims': (0.016, 0.045),
        'uncertainty_multiplier': 20,
        'act_critical': 0.022
     }
    
    na512_data_dict = {
        'data_suffix': '',
        'L_list': [512,],
        'Nexp_list': [5],
        'act_exclude_dict': {512: [0.02, 0.0225, 0.0235],},
        'xlims': (0.016, 0.045),
        'uncertainty_multiplier': 20,
        'act_critical': 0.022
        }
    na1024_data_dict = {
        'data_suffix': '',
        'L_list': [1024,],
        'Nexp_list': [5],
        'act_exclude_dict': {1024: [],},
        'xlims': (0.016, 0.045),
        'uncertainty_multiplier': 20,
        'act_critical': 0.022
     }
    na2048_data_dict = {
        'data_suffix': '',
        'L_list': [2048,],
        'Nexp_list': [5],
        'act_exclude_dict': {2048: [0.0225],},
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
    pol_data_dict = {
        'data_suffix': 'pol',
        'L_list': [2048],
        'Nexp_list': [1],
        'act_exclude_dict': {2048: [0.05, 0.1, 0.105,0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14,]},
        'xlims': None,
        'uncertainty_multiplier': 5,
        'act_critical': None
    }
    pols_data_dict = {
        'data_suffix': 'pols',
        'L_list': [2048],
        'Nexp_list': [1],
        'act_exclude_dict': {2048: []},
        'xlims': None,
        'uncertainty_multiplier': 1,
        'act_critical': None
    }

    data_dict = {'sd': sd_data_dict, 'ndg': ndg_data_dict, 'na': na_data_dict, 'na512':
                  na512_data_dict, 'na1024': na1024_data_dict, 'na2048': na2048_data_dict, 's': s_data_dict, 
                   'pol': pol_data_dict, 'pols': pols_data_dict, 'abp': abp_data_dict}
    fig_folder_dict = {'sd': 'sd', 'ndg': 'ndg', 'na': 'na',
                    'na512': 'na512',
                    'na1024': 'na1024', 
                    'na2048': 'na2048', 's': 's',
                    'pol': 'pol', 'pols': 'pols', 'abp': 'abp'}

    cid_dict = {
        'base_path': base_path,
        'save_path': save_path,
        'cg': cg,
        'verbose': verbose,
        'ddof': 1,
        **data_dict[data_suffix]
    }

    for size in size_list:
        print(f'\nProcessing for size={size}, cg={cg}...')

        xlims = data_dict[data_suffix]['xlims']

        output_suffix = output_suffix_func(size)
        cid_dict.update({'output_suffix': output_suffix,})

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
            if size == 7 and data_suffix == '':
                L_list = [1024, 2048]
            else:
                L_list = ac.L_list
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                use_min=False
                #L_list = [1024]
                
                ## Plot cid/div and its derivative with respect to activity
                #ac.plot_cid_and_deriv(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, act_critical=act_critical, xlims=xlims, plot_abs=True)
                ac.plot_cid_and_deriv(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min, act_critical=act_critical, xlims=xlims, plot_abs=False)
                #ac.plot_div_and_deriv(L_list=L_list, save_path=ac.figs_save_path, use_min=use_min,   act_critical=act_critical, xlims=xlims, plot_abs=True)
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
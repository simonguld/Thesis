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

    base_path = f'Z:\\cid\\na'
    save_path = f'data\\nematic\\na'

    verbose = True
    use_sd = False
    use_seq = False

    if use_sd:
        nexp = 10
        data_suffix = 'sd'
        L_list = [512]
        Nexp_list = [nexp]
        act_exclude_dict = {512: []}
    else:
        nexp = 5
        data_suffix = ''
        L_list = [512, 1024, 2048]
        Nexp_list = [nexp]*len(L_list)
        act_exclude_dict = {512: [0.02, 0.0225], 1024: [], 2048: [0.0225]}

    xlims = (0.016, 0.045)

    cid_dict = {
            'base_path': base_path,
            'save_path': save_path,
            'data_suffix': data_suffix,
            'cg': cg,
            'L_list': L_list,
            'Nexp_list': Nexp_list,
            'act_exclude_dict': act_exclude_dict,
            'uncertainty_multiplier': 20,
            'verbose': verbose,
            'ddof': 1,
        }

    for nbits in nbits_list:
        print(f'\nProcessing for nbits={nbits}, cg={cg}...')

        output_suffix=f'_seq_nb{nbits}cg{cg}' if use_seq else f'_nb{nbits}cg{cg}'
        cid_dict.update({'output_suffix': output_suffix, 'nbits': nbits})

        figs_save_path = f'data\\nematic\\figs\\{output_suffix[1:]}'
        if not os.path.exists(figs_save_path): os.makedirs(figs_save_path)

        # Initialize AnalyseCID object
        ac = AnalyseCID(cid_dict)

        if extract:
            ac.run()
        if analyze and not extract:
            ac.analyze()
        if plot:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                    
                ## Plot cid/div and its derivative with respect to activity
                ac.plot_cid_and_deriv(save_path=figs_save_path, xlims=xlims, plot_abs=True)
                ac.plot_div_and_deriv(save_path=figs_save_path, xlims=xlims, plot_abs=True)
                plt.close('all')

                ## Plot moments
                ac.plot_cid_moments(L_list=ac.L_list, save_path=figs_save_path, xlims=xlims,)
                ac.plot_div_moments(L_list=ac.L_list, save_path=figs_save_path, xlims=xlims,)
                plt.close('all')

                ## Plot fluctuations
                ac.plot_cid_fluc(save_path=figs_save_path, xlims=xlims, plot_abs=True)          
                ac.plot_div_fluc(save_path=figs_save_path, xlims=xlims, plot_abs=True, plot_div_per=True)
                plt.close('all')
    
if __name__ == '__main__':
    main()
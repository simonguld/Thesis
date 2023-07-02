# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from cycler import cycler

from statsmodels.tsa.stattools import adfuller

sys.path.append('C:\\Users\\Simon Andersen\\Projects\\Projects\\AppStat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

## Set plotting style and print options
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster

d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
d_colors = {'axes.prop_cycle': cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])}
rcParams.update(d)
rcParams.update(d_colors)
np.set_printoptions(precision = 5, suppress=1e-10)


### FUNCTIONS ----------------------------------------------------------------------------------

def do_adf_test(time_series, maxlag = None, autolag = 'AIC', regression = 'c', verbose = True):
    """
    Performs the augmented Dickey-Fuller test on a time series.
    """
    result = adfuller(time_series, maxlag = maxlag, autolag = autolag, regression = regression)
    if verbose:
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print(f'nobs used: {result[3]}')
        print(f'lags used: {result[2]}')
        print(f'Critical Values:')
        
        for key, value in result[4].items():
            print(f'\t{key}: {value}')

    return result


def est_stationarity(time_series, interval_len, Njump, Nconverged, max_sigma_dist = 2):
 
    # Estimate the stationarity of a time series by calculating the mean and standard deviation
    # of the time series in intervals of length interval_len. If the mean of a block is sufficiently
    # close to the mean of the entire time series, then the block is considered stationary.

    Nframes = len(time_series)
    Nblocks = int(Nframes / interval_len)
    converged_mean = np.mean(time_series[Nconverged:])
    global_std = np.std(time_series[Nconverged:], ddof = 1)

    it = 0
    while it * Njump < Nframes - interval_len:
        mean_block = np.mean(time_series[it * Njump: it * Njump + interval_len])
        dist_from_mean = np.abs(mean_block - converged_mean) / global_std

        print("For block: ", it, "  the distance from the converged mean is: ", dist_from_mean, ".")
        if np.abs(mean_block - converged_mean) > max_sigma_dist * global_std:
            it += 1
        else:
            return it * Njump
    return it * Njump


### MAIN ---------------------------------------------------------------------------------------


def main():
    density_arr = np.loadtxt('defect_densities.txt')
    act_list = [0.1, 0.07, 0.025, 0.09, 0.05, 0.034, 0.026, 0.03, 0.08, 0.04, 0.032, 0.024, 0.028, 0.06]

    Nfolders = density_arr.shape[0]
    Nframes = density_arr.shape[1]
    Nframes_adf = 50
    Njump_adf = 25  
    cutoff_arr = np.zeros(Nfolders)
    frames = np.arange(Nframes)

    
    for i in range(Nfolders):
        if 0:
            print("\nFor activity = ", act_list[i], ":")
            p_val = 1
            it = 0
            idx_first_frame = 0
            if act_list[i] <= 0.032:
                while p_val > 0.02:
                    if Nframes_adf + it * Njump_adf < Nframes:
                        print("For first frame: ", it * Njump_adf)
                        results = do_adf_test(density_arr[i,it * Njump_adf: it * Njump_adf + Nframes_adf], autolag = 'AIC', regression = 'c', verbose = True)
                        p_val = results[1]
                        idx_first_frame = it * Njump_adf
                        it += 1      
                    else:
                        idx_first_frame = Nframes - 1
                        print("No stationary region found for activity = ", act_list[i])
                        break

        print("\nFor activity = ", act_list[i], ":")
        idx_first_frame = est_stationarity(density_arr[i], 10, 25, 100, max_sigma_dist=2)
        fig, ax = plt.subplots()
        ax.plot(frames, density_arr[i], label = f'Activity = {act_list[i]}')
        ax.plot([idx_first_frame, idx_first_frame], [0, np.max(density_arr[i])], 'k--', label = f'Left stationary endpoint = {idx_first_frame}')
        ax.plot([idx_first_frame + Nframes_adf, idx_first_frame + Nframes_adf], [0, np.max(density_arr[i])],\
                'k--', label = f'Right stationary endpoint = {idx_first_frame + Nframes_adf}')
        ax.set_xlabel('Frame')
        ax.set_title('For activity = ' + str(act_list[i]))
        ax.set_ylabel('Defect density')
        ax.legend()
        plt.show()




if __name__ == '__main__':
    main()

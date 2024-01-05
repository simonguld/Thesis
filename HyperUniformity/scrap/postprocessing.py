# Author:  Simon Guldager & Lasse Bonn
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
from cycler import cycler
from time import time


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append(os.getcwd())

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

# Set path to data
output_path = 'X:\\output'
bracket = '\\'


### FUNCTIONS ----------------------------------------------------------------------------------


def save_density_plot(dens_defects, activity, idx_first_frame,):
        """
        plot options are defect_density, fluctuations, av_defect_density
        """
        # Plot defect densities
        fig, ax = plt.subplots()
        ax.plot(dens_defects)
        ax.plot([idx_first_frame, idx_first_frame], [0, np.max(dens_defects)], 'k--')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Defect density')
        ax.set_title('Defect density for activity = {}'.format(activity))
        plt.savefig(f'{output_path}{bracket}defect_density_{activity}.png')
        plt.close()

def save_fluctuation_plot(fluctuations, activity, window_sizes):
        fig, ax = plt.subplots()

        ax.plot(window_sizes, fluctuations, '.-',)
        ax.set_xlabel('Window size')
        ax.set_ylabel('Density fluctuations')
        ax.set_title(label = f'Density fluctuations for activity = {float(activity)}')
        plt.savefig(f'{output_path}{bracket}density_fluctuations_act{float(activity)}.png')
        plt.close()

def save_av_defect_density_plot(statistics_arr, activity_list):
    # Plot average defect density
    fig, ax = plt.subplots()
    ax.errorbar(activity_list, statistics_arr[:, 0], yerr = statistics_arr[:, 1], \
                fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1, markersize = 4)
    ax.set_xlabel('Activity')   
    ax.set_ylabel('Defect density')
    ax.set_title('Average defect density vs. activity')
    plt.savefig(f'{output_path}{bracket}av_defect_density.png')
    plt.close()

def load_data(output_path, bracket):

    act_list = np.loadtxt(f'{output_path}{bracket}activity.txt')
    Nexperiments = len(act_list)
    av_defect_densities = np.loadtxt(f'{output_path}{bracket}av_defect_densities.txt')
    fluctuations = np.loadtxt(f'{output_path}{bracket}fluctuations.txt')
    window_sizes = np.loadtxt(f'{output_path}{bracket}fluctuations_window_sizes.txt')
    model_params = pd.read_csv(f'{output_path}{bracket}model_params.csv', index_col = 0)

    try:
        fluctuations_std = np.loadtxt(f'{output_path}{bracket}fluctuations_std.txt')
        return act_list, Nexperiments, av_defect_densities, fluctuations, fluctuations_std, window_sizes, model_params
    except:
        return act_list, Nexperiments, av_defect_densities, fluctuations, window_sizes, model_params


### MAIN ---------------------------------------------------------------------------------------

#TODO

def main():

    # Load data
    try:
        act_list, Nexperiments, av_defect_densities, fluctuations, fluctuations_std, window_sizes, model_params = load_data(output_path, bracket)
    except:
        act_list, Nexperiments, av_defect_densities, fluctuations, window_sizes, model_params = load_data(output_path, bracket)

    # Plot average defect density
    save_av_defect_density_plot(av_defect_densities, act_list)

    # Plot fluctuations for each activity
    for i, act in enumerate(act_list):
        save_fluctuation_plot(fluctuations[i, :], act, window_sizes)

  

if __name__ == '__main__':
    main()



## SCRAP --------------------------------------------------------------------------------------


   # Calc vorticity etc
    """
    Qxx_dat = frame.QQxx.reshape(LX, LY)
    Qyx_dat = frame.QQyx.reshape(LX, LY)

    # get directors
    dx, dy, S = get_dir(Qxx_dat, Qyx_dat, return_S=True)
    vx, vy = mp.base_modules.flow.velocity(frame.ff, LX, LY)

    dyux, dxux = np.gradient(vx)
    dyuy, dxuy = np.gradient(vy)

    vort = dxuy-dyux
    E = dxux + dyuy
    R = E**2 - vort**2
    """

    # Get no. of positive defects
    #nposdef = len([d for d in defects if d['charge']==0.5])
    #print(nposdef)  

    #anim = animate(ar, plot_defects, rng=[1,20], inter = 400, show = True)
    #anim.resume()

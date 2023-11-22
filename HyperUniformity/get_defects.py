# Author: Simon Guldager & Lasse Bonn
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import sys
import pickle
import glob
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib import rcParams
from cycler import cycler
from time import time

import massPy as mp


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


# if development_mode, use only few frames
development_mode = True
if development_mode:
    num_frames = 12


# decide whether to run locally or on cluster
run_locally = True
if run_locally:
    folder_path = 'X:\\nematic_data'
    output_path = 'X:\\output_local'
    bracket = '\\'
else:
    ## Define external paths
    #folder_path = '/groups/astro/kpr279/nematic_data'
    folder_path = '/lustre/astro/rsx187/mmout/active_sample_forperp'
    output_path = '/groups/astro/kpr279/output'
    bracket = '/'

empty_output_folder = True
if empty_output_folder:
    # delete all files in output folder
    files = glob.glob(output_path + bracket + '*')
    for f in files:
        os.remove(f)

    
### FUNCTIONS ----------------------------------------------------------------------------------

def get_dir(Qxx, Qyx, return_S=False):
    """
    get director nx, ny from Order parameter Qxx, Qyx
    """
    S = np.sqrt(Qxx**2+Qyx**2)
    #print(S)
    dx = np.abs(np.sqrt((np.ones_like(S) + Qxx/S)/2))
    #dy = np.sqrt((np.ones_like(S) - Qyx/S)/2)*np.sign(dx)
    #dy = Qyx/(2*s*dx)
    dy = np.sqrt((np.ones_like(S)-Qxx/S)/2)*np.sign(Qyx)
    if return_S:
        return dx, dy, S
    else:
        return dx, dy

def plot_flow_field(frame, engine = plt):
        mp.nematic.plot.velocity(frame, engine)

def plot_defects(frame, engine = plt):
    mp.nematic.plot.director(frame, engine)
    mp.nematic.plot.defects(frame, engine)    

def get_defect_density(defect_list, area, return_charges=False):
        """
        Get defect density for each frame in archive
        parameters:
            defect_list: list of lists of dictionaries holding defect charge and position for each frame 
            area: Area of system
            return_charges: if True, return list of densities of positive and negative defects
        returns:
            dens_defects: list of defect densities
        """
        # Get no. of frames
        Nframes = len(defect_list)

        if return_charges:
            # Initialize list of defect densities
            dens_pos_defects = []
            dens_neg_defects = []
            for i, defects in enumerate(defect_list):
                # Get no. of defects
                nposdef = len([d for d in defects if d['charge'] == 0.5])
                nnegdef = len([d for d in defects if d['charge'] == -0.5])

                dens_pos_defects.append(nposdef / area)
                dens_neg_defects.append(nnegdef / area)

            return dens_pos_defects, dens_neg_defects
        else:
            dens_defects = []
            for i, defects in enumerate(defect_list):
                # Get no. of defects
                ndef = len(defects)
                dens_defects.append(ndef / area)
            return dens_defects

def get_defect_list(archive, LX, LY, idx_first_frame=0, verbose=True):
    """
    Get list of topological defects for each frame in archive
    Parameters:
        archive: massPy archive object
        LX, LY: system size
        verbose: print time to get defect list
    Returns:
        top_defects: list of lists of dictionaries holding defect charge and position for each frame 
    """
    # Initialize list of top. defects
    top_defects = []

    if verbose:
        t_start = time()
    if not development_mode:
        Nframes = archive.__dict__['num_frames']
    else:  
        Nframes = num_frames

    # Loop over frames
    for i in range(idx_first_frame, Nframes):
        # Load frame
        frame = archive._read_frame(i)
        Qxx_dat = frame.QQxx.reshape(LX, LY)
        Qyx_dat = frame.QQyx.reshape(LX, LY)
        # Get defects
        defects = mp.nematic.nematicPy.get_defects(Qxx_dat, Qyx_dat, LX, LY)
        # Add to list
        top_defects.append(defects)

    if verbose:
        t_end = time() - t_start
        # print 2 with 2 decimals
        print('Time to get defect list: %.2f s' % t_end)

    return top_defects

def get_density_fluctuations(top_defect_list, LX, LY, Nwindows, Ndof = 1):
    """
    Calculate defect density fluctuations for different window sizes
    Parameters:
    -----------
    top_defect_list: list of dictionaries, each dictionary contains defect positions and charges for one frame
    LX, LY: int, system size
    Nwindows: int, number of windows to calculate defect density for
    Ndof: int, number of degrees of freedom used to calculate variance (default: 1)
    Returns:
    --------
    defect_densities: array of defect densities for different window sizes

    """
    Nframes = len(top_defect_list)

    # Intialize array of defect densities
    defect_densities = np.zeros([Nframes, Nwindows])

    # Define center point
    center = np.array([LX/2, LY/2])
    # Define max. and min. window size
    max_window_size = LX / 2 - 1
    min_window_size = LX / Nwindows
    # Define window sizes
    window_sizes = np.linspace(min_window_size, max_window_size, Nwindows)

    for frame, defects in enumerate(top_defect_list):
        # Step 1: Convert list of dictionaries to array of defect positions
        Ndefects = len(defects)
        defect_positions = np.empty([Ndefects, 2])
        for i, defect in enumerate(defects):
            defect_positions[i] = defect['pos']
        # Step 2: Calculate distance of each defect to center
        distances = np.linalg.norm(defect_positions - center, axis=1)
        # Step 3: Calculate density for each window size
        for i, window_size in enumerate(window_sizes):
            # Get defects within window
            defects_in_window = len(distances[distances < window_size])
            # Calculate  and store density
            defect_densities[frame, i] = defects_in_window / (np.pi * window_size**2)
    # Calculate fluctuations of defect density
    density_fluctuations = np.var(defect_densities, axis=0, ddof = Ndof)
    return density_fluctuations, window_sizes

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

def animate(oa, fn, rng=[], inter=200, show=True):
    """Show a frame-by-frame animation.

    Parameters:
    oa -- the output archive
    fn -- the plot function (argument: frame, plot engine)
    rng -- range of the frames to be ploted
    interval -- time between frames (ms)
    """
    # set range
    if len(rng)==0:
        rng = [ 1, oa._nframes+1 ]
    # create the figure
    fig = plt.figure()

    # the local animation function
    def animate_fn(i):
        # we want a fresh figure everytime
        fig.clf()
        # add subplot, aka axis
        #ax = fig.add_subplot(111)
        # load the frame
        frame = oa._read_frame(i)
        # call the global function
        fn(frame, plt)

    anim = ani.FuncAnimation(fig, animate_fn,
                             frames=np.arange(rng[0], rng[1]),
                             interval=inter, blit=False)
    if show==True:
      plt.show()
      return

    return anim


def sigma_calculator_2d(field, R, N, center_x, center_y):
    field = np.array(field)  # Convert field to a NumPy array
    indexes = np.random.choice(range(len(field)), N, replace=False)
    points = field
    densities = []
    counts = []

    for i in range(N):
        point0 = points[indexes[i]]
        distance = np.linalg.norm(points - point0, axis=1)
        n = np.sum(distance <= R)
        density = n / (np.pi * R**2)  # Area of the circle
        counts.append(n)
        densities.append(density)

    densities = np.array(densities)
    counts = np.array(counts)

    delta_rho_squared_counts = np.var(counts)/(np.pi * R**2)
    delta_rho_squared_densities = np.var(densities)

    normalized_delta_rho_squared_density = delta_rho_squared_densities

    return delta_rho_squared_counts, normalized_delta_rho_squared_density,densities

def HyperUniformity_2d(field,R):
    Sigma1 = []
    Sigma2 = []
    N=len(R)
    for i in range(N):
          
        # Calculate the center of the circle
        center_x = 0  # X-coordinate of the center
        center_y = 0  # Y-coordinate of the center

        # Calculate sigma using sigma_calculator_2d with the defined circle properties
        sig1, sig2, density= sigma_calculator_2d(field, R[i], len(field), center_x, center_y)

        Sigma1.append(sig1)
        Sigma2.append(sig2)
        
        
        if i==0:
            density0=np.mean(density)
            
    return Sigma1, Sigma2, R,density0

def Hyper(field,R):
    sigma1,sigma2,R, density0=HyperUniformity_2d(field,R)
    ## sigma 1 is  calculated using number of particles, not what we want feel free to take it out
    Sigma2=sigma2/(density0**2) ## calculated using density, it´s the one you want 
    
    return Sigma2, R

### MAIN ---------------------------------------------------------------------------------------

#TODO

# make functional for N realizations
# make idx_first_frame a dyn. variable

def main():
    save_density_plots, save_model_params, check_if_params_are_identical = True, True, False
    calc_density_fluctuations = True

    if save_model_params:
        # Set what params to discard
        params_discard_list = ['Ex','Ey','_path','_compress_full','_compress','_ext']

    # Get list of folders
    dirs = os.listdir(folder_path)

    # Critical activity (samples with no top. defects (activity <= 0.2) or samples with a non-convergant no. of
    # top. defects (activity = 0.022) are discarded)
    zeta_c = 0.022

    Nexperiment = 10
    # Decide which experiment to analyze [from 0 to 9]
    experiment_list = [0,1]
    # Decide how many windows to use for calculating defect density fluctuations
    Nwindows = 5


    # After having looked at the no. of top. defects for each activity, the following cutoffs indicate when the
    # no. of top. defects have converged. To generalize this approach, one could use a statistical test to 
    # check for stationarity in the time series, and then increase the cutoff until convergence occurs.
    # 2 such tests are Augmented Dickey-Fuller (ADF) test and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
    truncated_samples_activities = [0.024, 0.025, 0.026, 0.028, 0.030, 0.032]
    truncated_samples_cutoff = [100, 85, 80, 25, 25, 25]

    # Find folders with activities below the cutoff
    keep_list = []
    pop_idx_list = []
    act_list = []

    if not run_locally:
        for i, dir in enumerate(dirs):
            if int(dir[-1]) == Nexperiment:
                split_name = dir.split('z')[-1]
                zeta = float(split_name.split('_')[0])
               # act_list.append(zeta)
                if zeta <= zeta_c:
                    pop_idx_list.append(i)
                else:
                    keep_list.append(zeta)
            else:
                pop_idx_list.append(i)

        # Remove folders with activities below the cutoff
        for i in reversed(pop_idx_list):
            dirs.pop(i)

    

        print("Simulations with the following activities are kept: ", sorted(keep_list))


    Nfolders = len(dirs)
    

    # Initialize arrays to hold results
    statistics_arr = np.zeros((Nfolders, 2))
    activity_arr = np.zeros((Nfolders, 1))
    fluctuation_arr = np.zeros((Nfolders, Nwindows))

    # Get top. defects for each folder
    for i, dir in enumerate(dirs):
        archive_path = folder_path + '/' + dir
 
        # Load data archive
        ar = mp.archive.loadarchive(archive_path)
        LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
        activity_arr[i] = ar.__dict__['zeta']

        # Initialize defect density arr
        if i == 0:
            if not development_mode:
                Nframes = ar.__dict__['num_frames']
            else:
                Nframes = num_frames
            def_density_arr = np.zeros((Nfolders, Nframes))
        print("act Nframes def_dens_Arr shape", activity_arr[i], Nframes, def_density_arr.shape)

        # Determine when to truncate samples
        if activity_arr[i] in truncated_samples_activities:
            idx_first_frame = truncated_samples_cutoff[truncated_samples_activities.index(activity_arr[i])]
        else:
            idx_first_frame = 0

        if save_model_params:
            if i == 0:
                model_params = ar.__dict__.copy()
                for key in params_discard_list:
                    model_params.pop(key)
                for key in model_params:
                    model_params[key] = [model_params[key]]
            else:
                if check_if_params_are_identical:
                    for key in model_params:
                        model_params[key].append(ar.__dict__[key])
    
        # Get defect list
        top_defects = get_defect_list(ar, LX, LY, idx_first_frame=idx_first_frame)
    
        # Get total defect density
        dens_defects = get_defect_density(top_defects, LX*LY)
        print("dens_defects", dens_defects)

        statistics_arr[i] = np.mean(dens_defects), np.std(dens_defects, ddof = 1)

        if calc_density_fluctuations:
            # Get defect density fluctuations
            fluctuation_arr[i], window_sizes = get_density_fluctuations(top_defects, LX, LY, Nwindows, Ndof=1)


            # Save defect densities
            def_density_arr[i][idx_first_frame:] = dens_defects

            # Plot density fluctuations against windows sizes for given activity
            fig, ax = plt.subplots()
            ax.plot(window_sizes, fluctuation_arr[i], '.-', label = f'Density fluctuations for activity = {float(activity_arr[i])}:.2f')
            ax.set_xlabel('Window size')
            ax.set_ylabel('Density fluctuations')
            ax.set_title('Density fluctuations vs. window size for activity = {}'.format(activity_arr[i]))
            ax.legend()
            plt.savefig(f'{output_path}{bracket}density_fluctuations_{activity_arr[i]}.png')
            plt.close() 


        if save_density_plots:
            # Plot defect densities
            fig, ax = plt.subplots()
            Nframes = len(dens_defects)
            ax.plot(np.arange(idx_first_frame,dens_defects,1), dens_defects, '.-', label = 'Total defects')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Defect density')
            ax.set_title('Defect density vs. frame for activity = {}'.format(activity_arr[i]))
            ax.legend()
    
            plt.savefig(f'{output_path}{bracket}defect_density_{activity_arr[i]}.png')
            plt.close()


    # Save results
    np.savetxt(f'{output_path}{bracket}defect_densities.txt', def_density_arr)
    np.savetxt(f'{output_path}{bracket}av_defect_densities.txt', statistics_arr)
    np.savetxt(f'{output_path}{bracket}activity.txt', activity_arr)
    np.savetxt(f'{output_path}{bracket}fluctuations.txt', fluctuation_arr)
    if save_model_params:
        model_params = pd.DataFrame.from_dict(model_params) 
        model_params.to_csv(f'{output_path}{bracket}model_params.csv')
        if check_if_params_are_identical:
            stds = model_params.describe().loc['std']
            print("No. of parameters varied between simulations: ", len(stds > 1e-5))


    # Plot average defect density
    fig, ax = plt.subplots()
    ax.errorbar(activity_arr, statistics_arr[:, 0], yerr = statistics_arr[:, 1], \
                fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1, markersize = 4)
    ax.set_xlabel('Activity')   
    ax.set_ylabel('Defect density')
    ax.set_title('Average defect density vs. activity')
    plt.savefig(f'{output_path}{bracket}av_defect_density.png')
    plt.close()

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

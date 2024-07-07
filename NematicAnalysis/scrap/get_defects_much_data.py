# Author:  Simon Guldager & Lasse Bonn
# Date (latest update): July 2 2023

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import glob
import warnings
import time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from sklearn.neighbors import KDTree

from matplotlib import rcParams
from cycler import cycler

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
development_mode = False
if development_mode:
    num_frames = 10

# decide whether to run locally or on cluster
run_locally = False
if run_locally:
    folder_path = "C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\nematic_data"
    output_path = 'C:\\Users\\Simon Andersen\\Documents\\Uni\\Speciale\\Hyperuniformity\\output_local'
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

def get_defect_list(archive, LX, LY, idx_first_frame=0, verbose=False):
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
        t_start = time.time()
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
        t_end = time.time() - t_start
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
    min_window_size = (LX / 2) / Nwindows
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


def calc_density_fluctuations(points_arr, window_sizes, N_center_points=None, Ndof=1, dist_to_boundaries=None, normalize=False):
    """
    Calculates the density fluctuations for a set of points in a 2D plane for different window sizes.
    For each window_size (i.e. radius), the density fluctuations are calculated by choosing N_center_points random points
    inside a region determined by dist_to_boundaries and calculating the number of points within a circle of radius R for each
    of these points, from which the number and density variance can be calculated.

    Parameters:
    -----------
    points_arr : (numpy array) - Array of points in 2D plane
    window_sizes : (numpy array or list) - Array of window sizes (i.e. radii) for which to calculate density fluctuations
    N_center_points : (int) - Number of center points to use for each window size. If None, all points are used.
    Ndof : (int) - Number of degrees of freedom to use for variance calculation
    dist_to_boundaries : (float) - Maximum distance to the boundaries. Centers will be chosen within this region.
    normalize : (bool) - If True, the density fluctuations are normalized by the square of the average density of the system.

    Returns:
    --------
    var_counts : (numpy array) - Array containing the number variance for each window size
    var_densities : (numpy array) - Array containing the density variance for each window size
    """

    # If dist_to_boundaries is not given, use the maxium window size
    dist_to_boundaries = window_sizes[-1] if dist_to_boundaries is None else dist_to_boundaries

    # Construct mask for points within boundaries
    center_mask_x = (points_arr[:, 0] - dist_to_boundaries >= np.min(points_arr[:, 0])) & (points_arr[:, 0] + dist_to_boundaries <= np.max(points_arr[:, 0]))
    center_mask_y = (points_arr[:, 1] - dist_to_boundaries >= np.min(points_arr[:, 1])) & (points_arr[:, 1] + dist_to_boundaries <= np.max(points_arr[:, 1]))
    center_mask = center_mask_x & center_mask_y

    # If N is not given, use all points within boundaries
    Npoints = len(points_arr)
    Npoints_within_boundaries = center_mask.sum()
    N_center_points = Npoints_within_boundaries if N_center_points is None else N_center_points

    print("Number of points within boundaries: ", Npoints_within_boundaries)
    print("Number of points to use: ", N_center_points)

    if N_center_points > Npoints_within_boundaries:
        print(f"Warning: N_center_points is larger than the number of points within the boundaries.\
               Using all {Npoints_within_boundaries} points within boundaries instead.")
        N_center_points = Npoints_within_boundaries

    # If N_center_points is equal to Npoints_within_boundaries, use all points within boundaries
    use_all_center_points = (N_center_points == Npoints_within_boundaries)
    if use_all_center_points:
        center_points = points_arr[center_mask]


    # Initialize KDTree
    tree = KDTree(points_arr)

    # Initialize density array, density variance array, and counts variance arrays
    var_counts = np.empty_like(window_sizes, dtype=float)
    var_densities = np.empty_like(var_counts)
    av_counts = np.empty_like(var_counts)

    for i, radius in enumerate(window_sizes):
        if use_all_center_points:
            pass
        else:
            indices = np.random.choice(np.arange(Npoints)[center_mask], N_center_points, replace=False)
            center_points = points_arr[indices]

        # Calculate no. of points within circle for each point
        counts = tree.query_radius(center_points, r=radius, count_only=True)

        # Calculate average counts
        av_counts[i] = np.mean(counts)

        # Calculate number and density variance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            var_counts[i] = np.var(counts, ddof=Ndof)
            densities = counts / (np.pi * radius**2)
            var_densities[i] = np.var(densities, ddof=Ndof)

    if normalize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_densities = np.nanmean(av_counts / (np.pi * window_sizes**2))
            var_densities /= av_densities**2

    return var_counts, var_densities



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

        if np.abs(mean_block - converged_mean) > max_sigma_dist * global_std:
            it += 1
        else:
            return it * Njump, True
    return it * Njump, False

def save_density_plot(dens_defects, activity, idx_first_frame,exp):
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
        plt.savefig(f'{output_path}{bracket}defect_density{exp}_{activity}.png')
        plt.close()

def save_fluctuation_plot(fluctuations, activity, window_sizes):
        fig, ax = plt.subplots()

        ax.plot(window_sizes, fluctuations, '.-',)
        ax.set_xlabel('Window size')
        ax.set_ylabel('Density fluctuations')
        ax.set_title(label = f'Density fluctuations for activity = {float(activity):.2f}')
        plt.savefig(f'{output_path}{bracket}density_fluctuations.png')
        plt.close()

def save_results(statistics_arr, activity_list, fluctuation_arr, window_sizes, model_params, Nexperiments):
       
        if Nexperiments <= 2:
            statistics_arr = np.mean(statistics_arr, axis = 0)
        if Nexperiments > 2:
            fluctuation_arr_std = np.std(fluctuation_arr, axis = 0, ddof=1) / np.sqrt(Nexperiments)
            # Calc. weighted mean of experiments
            stat_arr = np.zeros_like(statistics_arr[0,:,:])
    
            variance =  1 / (1 / statistics_arr[:,:,1]**2).sum(axis=0)
            SEM = np.sqrt(variance)
            stat_arr[:,0] = (statistics_arr[:,:,0] / statistics_arr[:,:,1]**2).sum(axis=0) * variance
            stat_arr[:,1] = SEM

        fluctuation_arr = np.mean(fluctuation_arr, axis = 0)

        # Save results
        np.savetxt(f'{output_path}{bracket}av_defect_densities.txt', stat_arr)
        np.savetxt(f'{output_path}{bracket}activity.txt', np.array(activity_list))
        np.savetxt(f'{output_path}{bracket}fluctuations.txt', fluctuation_arr)
        np.savetxt(f'{output_path}{bracket}fluctuations_window_sizes.txt', window_sizes)
        if Nexperiments > 2:
            np.savetxt(f'{output_path}{bracket}fluctuations_std.txt', fluctuation_arr_std)
    
        model_params = pd.DataFrame.from_dict(model_params) 
        model_params.to_csv(f'{output_path}{bracket}model_params.csv')

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

def gen_debug_txt(message = '', no = 0):
    """
    Generate txt file with message and no.
    """
    with open(f'{output_path}{bracket}debug{no}.txt', 'w') as f:
        f.write(message)
    f.close()
    return


### MAIN ---------------------------------------------------------------------------------------


def main():
    save_density_plots, = True,  

    # Set what params to not save
    params_discard_list = ['Ex','Ey','_path','_compress_full','_compress','_ext', 'model_name']

    # Get list of folders
    dirs_all = os.listdir(folder_path)

    # Critical activity (samples with no top. defects (activity <= 0.2) or samples with a non-convergant no. of
    # top. defects (activity = 0.022) are discarded)
    zeta_c = 0.022

    # Decide which experiment to analyze [from 0 to 9]
    experiment_list = np.arange(10)
    Nexperiments = len(experiment_list)
    Nfolders_per_experiment = 14 
    # Decide how many windows to use for calculating defect density fluctuations
    Nwindows = 30

    # Find folders with activities below the cutoff
    pop_idx_list = []
    activity_list = []

    # Initialize arrays to hold results
    statistics_arr = np.zeros((Nexperiments, Nfolders_per_experiment, 2))
    fluctuation_arr = np.zeros((Nexperiments, Nfolders_per_experiment, Nwindows))
    if Nexperiments > 2:
        fluctuation_arr_std = np.zeros((Nfolders_per_experiment, Nwindows))
    

    for j, exp in enumerate(experiment_list):
        t_start = time.time()
        print("Experiment: ", exp)
        # Extract all files for a given experiment
        dirs = [dir for dir in dirs_all if int(dir[-1]) == exp]

        activity_list = []
        pop_idx_list = []

        if not run_locally:
            for i, dir in enumerate(dirs):
                split_name = dir.split('z')[-1]
                zeta = float(split_name.split('_')[0])
                if zeta <= zeta_c:
                    pop_idx_list.append(i)
                else:
                    activity_list.append(zeta)

            # Remove folders with activities below the cutoff
            for i in reversed(pop_idx_list):
                dirs.pop(i)

        # Sort dirs according to activity
        dirs = [x for _, x in sorted(zip(activity_list, dirs))]

        # Sort activity_list according to activity
        activity_list = sorted(activity_list)

        Nfolders = len(dirs)

        # Get top. defects for each folder
        for i, dir in enumerate(dirs):
            archive_path = folder_path + '/' + dir
           
            # Load data archive
            ar = mp.archive.loadarchive(archive_path)
            LX, LY = ar.__dict__['LX'], ar.__dict__['LY']
            activity_list[i] = ar.__dict__['zeta']
            assert(activity_list[i] == ar.__dict__['zeta'])

            # Get defect list
            top_defects = get_defect_list(ar, LX, LY,)
        
            # Get total defect density
            dens_defects = get_defect_density(top_defects, LX*LY)

            if development_mode:
                idx_first_frame = 0
            else:
                idx_first_frame, converged = est_stationarity(dens_defects, 10, 25, 100, max_sigma_dist=2)
                if not converged:
                    print("Experiment ", exp, " act. ", activity_list[i], " did not converge")
    
            if len(dens_defects) > 2:
                statistics_arr[j, i, :] = np.mean(dens_defects[idx_first_frame:]), \
                                        np.std(dens_defects[idx_first_frame:], ddof = 1) / np.sqrt(len(dens_defects[idx_first_frame:]))
            else:
                statistics_arr[j, i, :] = 0, 1


            # Get defect density fluctuations
            fluctuation_arr[j, i, :], window_sizes = get_density_fluctuations(top_defects[idx_first_frame:], LX, LY, Nwindows, Ndof=1)

            if save_density_plots:
                if exp == 0:
                    save_density_plot(dens_defects, activity_list[i], idx_first_frame, exp)
                elif exp < 10 and activity_list[i] < 0.33:
                    save_density_plot(dens_defects, activity_list[i], idx_first_frame, exp)

        time_end = time.time()
        print("Experiment ", exp, " finished in ", np.round((time_end - t_start) / 60, 2), " minutes")
        gen_debug_txt(f'Experiment {exp} finished in {np.round((time_end - t_start) / 60, 2)} minutes', exp)

    model_params = ar.__dict__.copy()
    for key in params_discard_list:
        try: model_params.pop(key)
        except: continue
        for key in model_params:
            model_params[key] = [model_params[key]]
    
    gen_debug_txt(f'Dictionary successfully saved to csv', 100)

    save_results(statistics_arr, activity_list, fluctuation_arr, window_sizes, model_params, Nexperiments)
    gen_debug_txt(f'stat arrays successfully saved to csv', 1000)


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

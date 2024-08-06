import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

path = 'folder_names2.txt'


def main():

    # read all folders in folder_names.txt into a list
    dirs_all = [folder for folder in open(path).read().split()]
  
    # After having looked at the no. of top. defects for each activity, the following cutoffs indicate when the
    # no. of top. defects have converged. To generalize this approach, one could use a statistical test to 
    # check for stationarity in the time series, and then increase the cutoff until convergence occurs.
    # 2 such tests are Augmented Dickey-Fuller (ADF) test and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
    truncated_samples_activities = [0.024, 0.025, 0.026, 0.028, 0.030, 0.032]
    truncated_samples_cutoff = [100, 85, 80, 25, 25, 25]

    # Find indices of folders with activities below the cutoff
    zeta_c = 0.022
    Nexperiments = 10
    # Decide which experiment to analyze [from 0 to 9]

    

    for exp in range(Nexperiments):
        # Extract all files for a given experiment
        dirs = [dir for dir in dirs_all if int(dir[-1]) == exp]

        keep_list = []
        pop_idx_list = []
 

        for i, dir in enumerate(dirs):
            split_name = dir.split('z')[-1]
            zeta = float(split_name.split('_')[0])
            if zeta <= zeta_c:
                pop_idx_list.append(i)
            else:
                keep_list.append(zeta)

        # Remove folders with activities below the cutoff
        for i in reversed(pop_idx_list):
            dirs.pop(i)

        for dir in dirs:
            print(dir)

        # Sort dirs according to activity
        dirs = [x for _, x in sorted(zip(keep_list, dirs))]

        # Sort keep_list according to activity
        keep_list = sorted(keep_list)

        print(keep_list)
        for dir in dirs:
            print(dir)

        Nfolders = len(dirs)
        print("Simulations with the following activities are kept: ", sorted(keep_list))

        for i, dir in enumerate(dirs):
            act = keep_list[i]

            if act in truncated_samples_activities:
                idx_first_frame = truncated_samples_cutoff[truncated_samples_activities.index(act)]
            else:
                idx_first_frame = 0


            print(exp, act, idx_first_frame)

if __name__ == '__main__':
    main()
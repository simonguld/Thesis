# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import sys
import time

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, integrate, interpolate, optimize
from scipy.special import sici, factorial
from iminuit import Minuit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from cycler import cycler


from utils import *

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

plt.style.use('sg_article')
np.set_printoptions(precision = 5, suppress=1e-10)


### FUNCTIONS ----------------------------------------------------------------------------------

def fit_func_lin(x, a, b):
    return a*x + b

### MAIN ---------------------------------------------------------------------------------------



def main():

    Nruns = 15000
    fit_vals = np.nan * np.zeros((Nruns, 2), dtype = float)
    fit_errs = np.nan * np.zeros((Nruns, 2), dtype = float)
    param_guess_lin = np.array([1, 1], dtype = float)

    x_vals = np.linspace(0, 10, 100)
    y_vals = 3 + 14*x_vals + 2 * (np.random.rand(Nruns, len(x_vals)) - 0.5)
    
    max_err = np.abs(y_vals.mean() / 20)
    dy_vals = np.random.normal(max_err / 5, max_err, size = (Nruns, len(x_vals)))

    


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
        for N in range(Nruns):
            fit = do_chi2_fit(fit_func_lin, x_vals, y_vals[N], dy_vals[N], param_guess_lin, verbose = False)

            if not fit._fmin.is_valid:
                print('Fit not valid')
                continue

            fit_vals[N] = fit.values
            fit_errs[N] = fit.errors
    
        # Now we fit the average values
        y_vals_av = np.mean(y_vals, axis = 0)
        dy_vals_av = np.std(y_vals, axis = 0) / np.sqrt(Nruns)
        fit = do_chi2_fit(fit_func_lin, x_vals, y_vals, dy_vals_av, param_guess_lin, verbose = False)

    print(y_vals_av[0])

    av_of_fit_vals = np.mean(fit_vals, axis = 0)
    av_of_fit_errs = np.mean(fit_errs, axis = 0)#
    av_of_fit_errs = np.std(fit_vals, axis = 0) / np.sqrt(Nruns)

    fit_of_av_vals = fit.values[:]
    fit_of_av_errs = fit.errors[:]

      
    for param in range(2):
        zval, pval = two_sample_test(fit_of_av_vals[param], av_of_fit_vals[param], fit_of_av_errs[param], av_of_fit_errs[param])
        print("\n\nFor parameter ", param)  
        print("z,p: ", zval, pval)
 
    print("Average fit values: ", av_of_fit_vals, " +/- ", av_of_fit_errs)   
    print("Fit of average values: ", fit.values[:], " +/- ", fit.errors[:])

    print("\n")
if __name__ == '__main__':
    main()

## Imports:
import os
import sys
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from iminuit import Minuit
from scipy import stats


sys.path.append('C:\\Users\\Simon Andersen\\Projects\\Projects\\Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure



def get_statistics_from_fit(fitting_object, Ndatapoints, subtract_1dof_for_binning = False):
    
    Nparameters = len(fitting_object.values[:])
    if subtract_1dof_for_binning:
        Ndof = Ndatapoints - Nparameters - 1
    else:
        Ndof = Ndatapoints - Nparameters
    chi2 = fitting_object.fval
    prop = stats.chi2.sf(chi2, Ndof)
    return Ndof, chi2, prop

def do_chi2_fit(fit_function, x, y, dy, parameter_guesses, verbose = True):

    chi2_object = Chi2Regression(fit_function, x, y, dy)
    fit = Minuit(chi2_object, *parameter_guesses)
    fit.errordef = Minuit.LEAST_SQUARES

    if verbose:
        print(fit.migrad())
    else:
        fit.migrad()
    return fit

def generate_dictionary(fitting_object, Ndatapoints, chi2_fit = True, chi2_suffix = None, subtract_1dof_for_binning = False):

    Nparameters = len(fitting_object.values[:])
    if chi2_suffix is None:
        dictionary = {'Entries': Ndatapoints}
    else:
        dictionary = {f'({chi2_suffix}) Entries': Ndatapoints}


    for i in range(Nparameters):
        dict_new = {f'{fitting_object.parameters[i]}': [fitting_object.values[i], fitting_object.errors[i]]}
        dictionary.update(dict_new)
    if subtract_1dof_for_binning:
        Ndof = Ndatapoints - Nparameters - 1
    else:
        Ndof = Ndatapoints - Nparameters
    if chi2_suffix is None:
        dictionary.update({f'Ndof': Ndof})
    else:
        dictionary.update({f'({chi2_suffix}) Ndof': Ndof})

    if chi2_fit:
        chi2 = fitting_object.fval
        p = stats.chi2.sf(chi2, Ndof)
        if chi2_suffix is None:
            dictionary.update({'Chi2': chi2, 'Prop': p})
        else:
            dictionary.update({f'({chi2_suffix}) Chi2': chi2, f'({chi2_suffix}) Prop': p})
    return dictionary


def generate_dictionary(fitting_object, Ndatapoints, chi2_fit = True, chi2_suffix = None, subtract_1dof_for_binning = False):

    Nparameters = len(fitting_object.values[:])
    if chi2_suffix is None:
        chi2_suffix = ''
    else:
        chi2_suffix = f'({chi2_suffix})'
   
    dictionary = {f'{chi2_suffix} Npoints': Ndatapoints}


    for i in range(Nparameters):
        dict_new = {f'{chi2_suffix} {fitting_object.parameters[i]}': [fitting_object.values[i], fitting_object.errors[i]]}
        dictionary.update(dict_new)
    if subtract_1dof_for_binning:
        Ndof = Ndatapoints - Nparameters - 1
    else:
        Ndof = Ndatapoints - Nparameters

    dictionary.update({f'{chi2_suffix} Ndof': Ndof})

    if chi2_fit:
        chi2 = fitting_object.fval
        p = stats.chi2.sf(chi2, Ndof)   
        dictionary.update({f'{chi2_suffix} chi2': chi2, f'{chi2_suffix} pval': p})

    return dictionary

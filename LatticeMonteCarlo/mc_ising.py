# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import sys
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate, interpolate, optimize
from iminuit import Minuit
from matplotlib import rcParams
from cycler import cycler

#import emcee


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Set plotting style
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster
rcParams['lines.linewidth'] = 2 
rcParams['axes.titlesize'] =  18
rcParams['axes.labelsize'] =  18
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 13
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (9,6)
rcParams['axes.prop_cycle'] = cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])
np.set_printoptions(precision = 5, suppress=1e-10)



### FUNCTIONS ----------------------------------------------------------------------------------

def Metropolis_1param(prop_distribution, proposal_distribution, Nsamples, \
    Nburnin, boundaries, starting_point = None):
    """
    Assuming only 1 parameter
    """
    ## Intialize
    accepted_samples = np.empty(Nsamples)
    Naccepted_jumps = 0
    iterations = 0

    ## If starting point is not provided, gen. a random one
    if starting_point is None:
        x_old = boundaries[0] + np.random.rand() * boundaries[1]
    else:
        x_old = starting_point

    while iterations - Nburnin < Nsamples:
        ## Propose a jump
        x_new = proposal_distribution(x_old)

        ## Enforce periodic boundaries
        if x_new > boundaries[1]:
            x_new = max(boundaries[0], boundaries[0] + (x_new - boundaries[1]))
        elif x_new < boundaries[0]:
            x_new = min(boundaries[1], boundaries[1] + (x_new - boundaries[0]))

        ## Accept new step with transition probability p_accept
        prob_fraction = (prop_distribution(x_new) * proposal_distribution(x_old, x_new)) \
            / (prop_distribution(x_old) * proposal_distribution(x_new, x_old))

        p_accept = min(1, prob_fraction)
        if np.random.rand() < p_accept:
            x_old = x_new.astype('float')
            if iterations >= Nburnin:
                Naccepted_jumps +=1

        # Collect values after the burn in phase
        if iterations >= Nburnin:
            accepted_samples[iterations - Nburnin] = x_old

        iterations += 1

    # Calc acceptance_rate
    acceptance_rate = Naccepted_jumps / (Nsamples - Nburnin)

    return accepted_samples, acceptance_rate


def mcrg_ising(system_dims, temp_list, Nsamples, Nburnin, Nrescalings, rescaling_factor, initialization = 'random'):
    """
    Metropolis algorithm for the 2D Ising model with renormalization group.

    Parameters
    ----------
    system_dims : tuple of ints (L, L) 
        System dimensions.
    temp_list : list of floats 
        List of temperatures (actually J/kT, where J is the nearest neighbor interaction) to simulate.
    Nsamples : int
        Number of samples to collect.
    Nburnin : int
        Number of samples to discard.
    rescaling_factor : float
        Rescaling factor to use when rescaling, i.e. applying the Renormaliztion Group.
        NB: The system length must be divisible by this factor.

    Returns
    -------
    magnetization : 3D array of shape (len(temp_list), Nsamples, Nrescalings)
        Magnetization for each temperature, sample and rescaling.

    """

    ## Initialize
    L = system_dims[0]

    ## Check that rescaling factor is valid
    if L % rescaling_factor != 0:
        raise ValueError(f'Rescaling factor {rescaling_factor} is not valid for system length {L}.')

    
    if initialization == 'random':
        ## Initialize system as len(temp_list) x L x L  random spins in {-1, 1} 
        system = np.random.choice([-1, 1], size = [len(temp_list), L, L]).astype('float')
    elif initialization == 'weighted':
        system  = np.zeros((len(temp_list), L, L))
        for i, temp in enumerate(temp_list):
            system[i] = np.random.choice([-1, 1], size = [L, L], p = [1 - magnetization_fraction_estimator(temp), magnetization_fraction_estimator(temp)])
    else:
        raise ValueError(f'Initialization {initialization} is not valid.')

    ## Initialize magnetization array
    magnetization = np.empty((len(temp_list), Nsamples, max(Nrescalings,1)))

    ## Initialize counters
    Ntot = Nsamples + Nburnin
    Naccepted = np.zeros(len(temp_list))
    
    for i in range(Ntot):
        # Calculate random spin flip indices
        flip_idx = np.random.randint(0, L, size = (len(temp_list), 2))
        flip_idx = np.concatenate((np.arange(len(temp_list))[:, None], flip_idx), axis = 1)

        if i%10000 == 0:
            # calculate total energy of system
            print(np.round(i/Ntot * 100, 1), '% done')
          

        # Calculate unitless energy change by only considering nearest neighbors, while enforcing periodic boundaries
        E_prior = - system[flip_idx[:, 0], flip_idx[:, 1], flip_idx[:, 2]] * (system[flip_idx[:, 0], (flip_idx[:, 1] + 1) % L, (flip_idx[:, 2] + 1) % L]  + \
                                                                            system[flip_idx[:, 0], (flip_idx[:, 1] - 1) % L, (flip_idx[:, 2] + 1) % L] + \
                                                                            system[flip_idx[:, 0], (flip_idx[:, 1] + 1) % L, (flip_idx[:, 2] - 1) % L] + \
                                                                            system[flip_idx[:, 0], (flip_idx[:, 1] - 1) % L, (flip_idx[:, 2] - 1) % L])
        
        # Calculate energy change and multiply by temp_list to get beta * delta E
        E_post = - E_prior
        dE = temp_list * (E_post - E_prior)

        # Calculate transition probabilities
        p_accept = np.exp(-dE)

        # Create accept/reject mask
        accept = np.random.rand(len(temp_list)) < p_accept

        # Update system according to accepted flips
        system[flip_idx[accept, 0], flip_idx[accept, 1], flip_idx[accept, 2]] *= -1

        # Update acceptance counters
        Naccepted += accept

        if i >= Nburnin:
            # Update magnetization
            magnetization[:, i - Nburnin, 0] = np.mean(system, axis = (1, 2))

            # Rescale system by taking block averages and find new magnetizations
            rescaled_system = system.astype('float')
            for rescaling in range(1, Nrescalings):
                rescaled_system = np.mean(rescaled_system.reshape((len(temp_list), L // rescaling_factor, rescaling_factor, L // rescaling_factor, rescaling_factor)), axis = (2, 4))

                # Find 0-spin indices
                zero_spin_idx = np.where(rescaled_system == 0)

                # Flip 0-spins randomly
                rescaled_system[zero_spin_idx] = np.random.choice([-1, 1], size = len(zero_spin_idx[0]))

                # Update magnetization
                magnetization[:, i - Nburnin, rescaling] = np.mean(rescaled_system, axis = (1, 2))
   

    return magnetization, Naccepted / Ntot

def plot_grouped_magnetization(magnetization, temp_list, system_dims, Ngroup = 1000, Nburnin = 1):

    ## Initialize
    L = system_dims[0]
    Nrescalings = magnetization.shape[2]
    Nsamples = magnetization.shape[1]

    magnetization_partial_av = np.mean(magnetization.reshape((len(temp_list), Nsamples // Ngroup, Ngroup, Nrescalings)), axis = 2)

    ## Plot magnetization vs temperature for each rescaling
    fig, ax = plt.subplots()
    for i, temp in enumerate(temp_list):
        ax.plot(np.arange(Ngroup, Nsamples + 1, Ngroup), magnetization_partial_av[i, :, 0], 'o', markersize=3, alpha=0.8, label = f'T = {temp:.4f}')
    ax.set_xlabel('Temperature')
    ax.set_ylabel(f'Magnetization (averaged over {Ngroup} samples)')
    ax.legend()
    ax.set_title(f'Magnetization vs temperature for L = {L}')

    return

def plot_magnetization(magnetization, temp_list, system_dims, rescaling_factor, Nburnin = 1):
    """
    Plot results from mcrg_ising.
    """

    ## Initialize
    kc = 0.4406867935097714
    L = system_dims[0]
    Nrescalings = magnetization.shape[2]
    Nsamples = magnetization.shape[1]
    temp_list = kc / temp_list

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  
    # calculate running average
    running_average = np.empty((len(temp_list), Nsamples, Nrescalings))
    for i in range(len(temp_list)):
        for j in range(Nrescalings):
            running_average[i, :, j] = np.cumsum(magnetization[i,:, j]) / np.arange(1, Nsamples + 1)

    ## Plot running average vs iteration for each temperature
    fig, ax = plt.subplots()
    for i, temp in enumerate(temp_list):
        ax.plot(running_average[i, :, 0], '--', color = colors[i], markersize=1, alpha=0.8, label = rf'$T = {temp:.2f} T_c$')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Magnetization')
    ax.legend()
    ax.set_title(f'Magnetization vs iteration')
    ax.set_xticks(np.arange(0,(Nsamples + 1),  int(Nsamples/10)))
    # set tick labels with scientific format
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    Ngroup = 1000
    magnetization_partial_av = np.mean(magnetization.reshape((len(temp_list), Nsamples // Ngroup, Ngroup, Nrescalings)), axis = 2)

    ## Plot non-rescaled magnetization vs iteration for each temperature
    for i, temp in enumerate(temp_list):
        ax.plot(np.arange(Ngroup, Nsamples + 1, Ngroup), magnetization_partial_av[i, :, 0],'.', color = colors[i], markersize=0.5, alpha=0.4, label = rf'$T = {temp:.3f} T_c$')
      #  ax.plot(magnetization[i, :, 0], '.', color = colors[i], markersize=0.5, alpha=0.8, label = rf'$T = {temp:.3f} T_c$')

    if 0:
        ## Plot running average vs iteration for each temperature
        fig, ax = plt.subplots()
        for i, temp in enumerate(temp_list):
            ax.plot(running_average[i, :, 0], '--', markersize=1, alpha=0.8, label = rf'$T = {temp:.3f} T_c$')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Magnetization (running average)')
        ax.legend()
        ax.set_title(f'Magnetization (running average) vs iteration')

        ## Plot non-rescaled magnetization vs iteration for each temperature
        fig, ax = plt.subplots()
        for i, temp in enumerate(temp_list):
            ax.plot(magnetization[i, :, 0], '.', markersize=0.5, alpha=0.8, label = rf'$T = {temp:.3f} T_c$')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Magnetization')
        legend = ax.legend()
        for handle in legend.legendHandles:
            handle.set_markersize(8.0)
        ax.set_title(f'Average Magnetization vs iteration for L = {L}')

    return

def magnetization_fraction_estimator(k):
    """
    Estimate average magnetization per spin given as coupling k = J/kT.
    """
    kc = 0.4406867935097714
    if k < kc:
        K = kc + np.abs(k - kc)
    else:
        K = k
  
    u = np.exp(-4 * K)
    mag = (1+u)**0.25 / (1 - u) ** 0.5 * (1 - 6 * u + u ** 2) ** 0.125
    spin_up_fraction = (1 + mag) / 2
    
    return spin_up_fraction

def initialize_system(system_dims, temp_list):
    """
    Initialize system as len(temp_list) x L x L  random spins in {-1, 1} weighted by magnetization_fraction_estimator.
    """
    ## Initialize
    L = system_dims[0]

    ## Initialize system as len(temp_list) x L x L  random spins in {-1, 1}
    system = np.random.choice([-1, 1], size = [len(temp_list), L, L])

    ## Weight system by magnetization_fraction_estimator
    for i, temp in enumerate(temp_list):
        system[i] = np.random.choice([-1, 1], size = [L, L], p = [1 - magnetization_fraction_estimator(temp), magnetization_fraction_estimator(temp)])

    # Loop over Ntot

    # get random spin flips
    # calc energy changes
    # calc transition probabilities
    # accept/reject
    # calc magnetization for original and rescaled systems after burnin






    

## ARGS: system dims, temp-list, Nsamples, Nburnin, rescaling_factor
# funcs:  energy change estimator, block spin estimator, 
## return: Ntemperatures x Nsamples x Nrescalings values of magnetization, --> find mean and std


### MAIN ---------------------------------------------------------------------------------------


# fix average mag problem when burn in is not 0. for running and average etc
# calc. fraction and find critical exponents
# gen to 3D


# results to include
# plots of magnetization vs iteration for each temp (running average)
# plots of fractions




def main():


    ## Initialize
    L = 64
    kc = 0.4406867935097714
    temp_list = np.array([0.441, 0.443, 0.445]) #, 0.45, 0.46, 0.48, 0.5]
    temp_list = kc / np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    Nsamples = 10_000_000 #¤10_000_000 #44_500_000 #00_000
    Nburnin = 0 #3_500_000
    Nrescalings = 0
    rescaling_factor = 1

    # load results and make histograms for each temperature
    if 1:
        magnetization = np.load('magnetization_N10e7_Nb0.npy')
        # plot histograms
        if 0:
            fig, ax = plt.subplots(ncols = 3, nrows = 2)
            ax = ax.flatten()
            for i, temp in enumerate(temp_list):
                ax[i].hist(magnetization[i, :, 0], bins = 200, color = 'red', label = rf'$T = {kc/temp:.3f} T_c$', histtype ='step')
                ax[i].legend()

        for i, temp in enumerate(temp_list):
            fig, ax =   plt.subplots()
            if i > 2:
                ax.hist(magnetization[i, :, 0], bins = 200, color = 'red', label = rf'$T = {kc/temp:.3f} T_c$', histtype ='stepfilled')
            else:
                ax.hist(magnetization[i, :, 0], bins = 50, color = 'red', label = rf'$T = {kc/temp:.3f} T_c$', histtype ='stepfilled')
            ax.legend()
            ax.set_xlabel('Magnetization')
            ax.set_ylabel('Counts')
           
        plt.show()

    if 0:
        t1 = time.time()

        ## Run simulation
        magnetization, acceptance_rate = mcrg_ising((L, L), temp_list, Nsamples, Nburnin, Nrescalings, rescaling_factor, initialization = 'random')

        # save results
        np.save('magnetization_N10e7_Nb0_random_init.npy', magnetization)



        t2 = time.time()

        Nspins = L**2 * len(temp_list)

        print(f'Elapsed time: {t2 - t1:.2f} s')
        print('Iterations per spin: ',  Nsamples / Nspins)
        print('Spin flip rate: ', (Nsamples * len(temp_list)) / (t2 - t1))
        print(f'Acceptance rate: {acceptance_rate}')
        #print(f'Mean magnetization: {np.mean(magnetization, axis = 1)}')
        for i, temp in enumerate(temp_list):
            print(f'Theoretical magnetization for T = {temp}: {magnetization_fraction_estimator(temp)}')
            print(f'Simulated magnetization for T = {temp}: {np.mean(magnetization[i, :, 0])}', \
                "\u00B1", np.std(magnetization[i, :, 0],ddof=1) / np.sqrt(Nsamples))
    
        ## Plot results
        plot_magnetization(magnetization, temp_list, (L, L), rescaling_factor, Nburnin = Nburnin)

        plt.show()


    

 

if __name__ == '__main__':
    main()

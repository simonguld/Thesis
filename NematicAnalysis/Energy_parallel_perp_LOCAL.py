import matplotlib as mpl
mpl.use('agg')

import numpy as np
import matplotlib.pyplot as plt
from math import atan2

from massPy import archive 
from massPy.base_modules import flow as flow
from massPy.base_modules import plotlib as plotlib

import os, sys
import json

#from itertools import product
#from scipy import ndimage
#from scipy.ndimage import zoom as ndz
#from scipy.stats import binned_statistic
import scipy.stats as stats
#import matplotlib.animation as ani

def energy_spectral_density(Fx):
    # fast fourier transform (complex field)
    psi_fft = np.fft.fft2(Fx)
    # unit-wave vector
    k_hat = (     np.fft.fftfreq(psi_fft.shape[0])[:,None] * psi_fft.shape[0]
             + 1J*np.fft.fftfreq(psi_fft.shape[1])[None,:] * psi_fft.shape[1] )
    k_nrm = np.abs(k_hat)
    np.divide(k_hat, k_nrm, out=k_hat, where=k_nrm!=0)
    k_nrm = k_nrm.flatten()
    
    k_max = min(Fx.shape)//2 + 1 
    kbins = np.arange(.5, k_max, 1.)
    kvals = .5 * (kbins[1:] + kbins[:-1])
    
    psi2_fft = (np.abs(psi_fft)**2).flatten()
    Ek, _, _ = stats.binned_statistic(k_nrm, psi2_fft, statistic="mean", bins=kbins)
    
    return Ek, kvals

def rotate_vector_grid(x, y, angle_rad):
    # Create the rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # Apply the rotation for each element of x and y
    x_rot = cos_theta * x - sin_theta * y
    y_rot = sin_theta * x + cos_theta * y
    
    return x_rot, y_rot

def shift_matrix(matrix, i, j):
    
    return np.roll(np.roll(matrix, -i, axis=0), -j, axis=1)

def modify_dat_file(filename, alpha, zeta, xi):
    with open(filename, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.startswith('alpha'):
            lines[i] = f'alpha\t = {alpha}\n'
        elif line.startswith('zeta'):
            lines[i] = f'zeta\t = {zeta}\n'
        elif line.startswith('xi'):
            lines[i] = f'xi\t = {xi}\n'

    with open(filename, 'w') as file:
        file.writelines(lines)

xi= 0.2
alpha_values= np.linspace(0, 0.1, 11)
zeta = 0.05

for alpha in alpha_values: 
    result_string = f"Hyperuniformity_{round(alpha*100)}"
    
    oname= sys.argv[1]+'/'+result_string
    outname= sys.argv[2]+'/'+result_string
    
    print(oname,flush=True)

    ar =archive.loadarchive(oname)
    
    E_perp = []
    E_parallel = []
    Q_parallel = []
    Q_perp = []
    
    
    for ii in range(ar.num_frames):
        frame = ar[ii]
        
        
        vx, vy = flow.velocity(frame.ff, frame.LX, frame.LY)
        
        vx = plotlib.average(vx, frame.LX, frame.LY, 1)
        vy = plotlib.average(vy, frame.LX, frame.LY, 1)

        E_perp_frame = []
        E_parallel_frame = []
        Q_parallel_frame = []
        Q_perp_frame = []
        
        print(ii,flush=True)
        
        for i in range(vx.shape[0]):
            for j in range(vx.shape[1]):
        #for i in range(1, 1024, 50):
            #for j in range(1, 1024, 50):
                velocity_x = vx[i, j]
                velocity_y = vy[i, j]
                
                

                direction = atan2(velocity_y, velocity_x)
                
                
                v_parallel,v_perp=rotate_vector_grid(vx, vy, - direction)

                # Shift the matrix to have the point i,j at 0,0 in order to calculate the RDF from the origin
                v_parallel= shift_matrix(v_parallel, i, j)
                v_perp= shift_matrix(v_perp, i, j)

                # Find energy spectra for v_parallel and v_perp
                
                Ek_parallel, kv_parallel = energy_spectral_density(v_parallel)
                Ek_perp, kv_perp = energy_spectral_density(v_perp)

                
                
                E_parallel_frame.append(Ek_parallel) 
                E_perp_frame.append(Ek_perp) 
                Q_parallel_frame.append(kv_parallel)
                Q_perp_frame.append(kv_perp) 
                
            

        E_parallel.append( np.mean(E_parallel_frame, axis=0))
        E_perp.append(np.mean(E_perp_frame, axis=0))
        Q_parallel.append( np.mean(Q_parallel_frame, axis=0))
        Q_perp.append( np.mean(Q_perp_frame, axis=0))

        
        

    E_parallel=np.mean(E_parallel, axis=0)
    E_perp=np.mean(E_perp, axis=0)

    Q_parallel=np.mean(Q_parallel, axis=0)
    Q_perp=np.mean(Q_perp, axis=0)

    np.save(outname+'_E_||', E_parallel)
    np.save(outname+'_K_||', Q_parallel)

    np.save(outname+'_E_|_', E_perp)
    np.save(outname+'_K_|_', Q_perp)
    
    


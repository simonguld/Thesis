
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fftpack import fft, fftfreq


def static_structure_factor_2d(field, q_values):
    """
    Calculate the static structure factor for a 2D field.

    Parameters:
    - field: 3D array containing the 2D field varying in time.
    - q_values: Array of wavevectors.

    Returns:
    - sq: Static structure factor values for each q.
    """

    num_frames,num_x,num_y = field.shape


    sq = np.zeros(len(q_values))

    for i, q in enumerate(q_values):
        for j in range(num_x):
            for k in range(num_y):
                # Calculate the contribution to static structure factor for each q
                field_sum = 0
                for l in range(num_frames):
                    field_sum += field[l][j,k]
                sq[i] += np.exp(1j * (q[0] * j + q[1] * k)) * field_sum

    sq /= num_frames  # Normalize by the number of frames

    return sq.real  # Take the real part to remove numerical artifac

def dynamic_structure_factor_2d(field, time_step, q_values):
    """
    Calculate the dynamic structure factor for a 2D field.

    Parameters:
    - field: 3D array containing the 2D field varying in time.
    - time_step: Time step between data points in the time series.
    - q_values: Array of momentum transfer vectors.

    Returns:
    - omega_values: Array of frequency transfer values.
    - sqw: Dynamic structure factor values for each q and omega.
    """

    num_frames, num_x, num_y = field.shape
    frequencies = fftfreq(num_frames, time_step)
    omega_values = 2 * np.pi * frequencies

    sqw = np.zeros((len(q_values), len(omega_values)), dtype=complex)

    for i, q in enumerate(q_values):
        for j in range(num_x):
            for k in range(num_y):
                # Calculate the Fourier transform of the 2D field for each q               
                fluctuations = np.exp(1j * (q[0] * j + q[1] * k)) * field[:,j, k]
                sqw[i, :] += fft(fluctuations)

    return omega_values, sqw

def plot_static_structure_factor(q_values,sq_static):

    plt.figure(figsize=(10, 6))
    plt.plot(np.linalg.norm(q_values, axis=1), sq_static, marker='o', linestyle='-', color='b')
    plt.xlabel('Wave vector (|q|)')
    plt.ylabel('Static Structure Factor (S(q))')
    plt.title('Static Structure Factor')
    plt.ylim(0,100)
    plt.show()

def plot_fixed_dynamic_structure_factor_2d(omega_values,q_values,sqw,xlim=None,ylim=None):
    
    f, s = plt.subplots(figsize=(12,8),dpi=300)
    c = np.arange(np.min(np.linalg.norm(q_values,axis=1)), np.max(np.linalg.norm(q_values,axis=1))+0.001,5)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
    cmap.set_array([])

    for q_val_index in range(len(q_values)):
        s.scatter(omega_values, np.abs(sqw[q_val_index]), marker='o', s=50, color=cmap.to_rgba(np.linalg.norm(q_values[q_val_index])))

        
    cbar = f.colorbar(cmap, ticks=c)
    cbar.set_label("|q|")
    s.set_xlabel('Frequency (ω)')
    s.set_ylabel('|S(q, ω)|')
    s.set_title('Dynamic Structure Factor')

    
    # Set y-axis limits if provided
    if ylim:
        s.set_ylim(ylim)

    # Set x-axis limits if provided
    if xlim:
        s.set_xlim(xlim)


#plot spectrum of a single wavevector
    
def plot_single_dynamic_structure_factor_2d(omega_values,q_val_index,sqw,xlim=None,ylim=None):
    

    plt.scatter(omega_values, np.abs(sqw[q_val_index]), marker='o', s=50)

    plt.xlabel('Frequency (ω)')
    plt.ylabel('|S(q, ω)|')
    plt.title('Dynamic Structure Factor')

    
    # Set y-axis limits if provided
    if ylim:
        plt.ylim(ylim)

    # Set x-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    plt.show()
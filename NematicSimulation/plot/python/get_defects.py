# Author: Lasse Bonn (modified by Simon Guldager)
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy import stats, integrate, interpolate, optimize
from scipy.special import sici, factorial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from cycler import cycler

import massPy as mp


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


## Set path for sim. data
path = 'C:\\Users\\Simon Andersen\\Projects\\Projects\\Thesis\\NematicSimulation\\out\\test1'

path = 'X:\\copy_test_dir'

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

### MAIN ---------------------------------------------------------------------------------------



def main():
    # Load data archive
    ar = mp.archive.loadarchive(path)



    i =30
    frame = ar._read_frame(i)
    step=2
    LX, LY = frame.LX, frame.LY

    Qxx_dat = frame.QQxx.reshape(LX, LY)
    Qyx_dat = frame.QQyx.reshape(LX, LY)


    dx, dy, S = get_dir(Qxx_dat, Qyx_dat, return_S=True)
    vx, vy = mp.base_modules.flow.velocity(frame.ff, LX, LY)

    dyux, dxux = np.gradient(vx)
    dyuy, dxuy = np.gradient(vy)

    vort = dxuy-dyux
    E = dxux + dyuy
    R = E**2 - vort**2

    defects = mp.nematic.nematicPy.get_defects(Qxx_dat, Qyx_dat, LX, LY)
    print(defects[:5])

    nposdef = len([d for d in defects if d['charge']==0.5])
    print(nposdef)  

    f, s = plt.subplots()
    mp.nematic.plot.director(frame, s)
    mp.nematic.plot.defects(frame, s)


    f, s = plt.subplots()
    mp.nematic.plot.velocity(frame, s)


    # animate
    def plot_flow_field(frame, engine = plt):
        mp.nematic.plot.velocity(frame, engine)

    def plot_defects(frame, engine = plt):
        mp.nematic.plot.director(frame, engine)
        mp.nematic.plot.defects(frame, engine)

    #anim = animate(ar, plot_defects, rng=[5,20], inter = 400, show = True)

    #anim.resume()

    plt.show()

if __name__ == '__main__':
    main()

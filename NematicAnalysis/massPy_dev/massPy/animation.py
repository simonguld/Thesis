################################################################################
#
# Plotting routines and tools
#
################################################################################

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np

######################################################################
# animation

def animate(ar, fn, rng=[], inter=200, show=True):
    """Show a frame-by-frame animation.
    
    Parameters:
    ar -- output archive
    fn -- plot function (argument: frame, plot engine)
    rng -- range of the frames to be ploted
    interval -- time between frames (ms)
    """
    # set range
    if len(rng)==0:
        rng = [0, ar._nframes]
    # create the figure
    fig = plt.figure(figsize=(20,20))
    plt.rcParams.update({ 'font.serif': 'Computer Modern Roman',
                          'text.usetex': True,
                          'font.size': 26
                        })

    # the local animation function
    def animate_fn(i):
        # we want a fresh figure everytime
        plt.clf()
        # add subplot, aka axis
        ax = fig.add_subplot(111)
        # load the frame
        frame = ar[i]
        # call the global function
        fn(frame, ax)

    anim = ani.FuncAnimation(fig, animate_fn,
                             frames=np.arange(rng[0], rng[1]),
                             interval=inter, blit=False)
    if show==True:
      plt.show()
      return

    return anim

def save(an, fname, fps, tt='ffmpeg', bitrate=-1):
    writer = ani.writers[tt](fps=fps, bitrate=bitrate)
    an.save(fname, writer=writer)

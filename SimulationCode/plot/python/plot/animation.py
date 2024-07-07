################################################################################
#
# Plotting routines and tools
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

######################################################################
# animation

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
        frame = oa.read_frame(i)
        # call the global function
        fn(frame, plt)

    anim = ani.FuncAnimation(fig, animate_fn,
                             frames=np.arange(rng[0], rng[1]),
                             interval=inter, blit=False)
    if show==True:
      plt.show()
      return

    return anim

def save(an, fname, fps, tt='ffmpeg', bitrate=1800):
    writer = ani.writers[tt](fps=fps, bitrate=bitrate)
    an.save(fname, writer=writer)

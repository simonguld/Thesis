import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import sys
# import and plot
import archive
import plot

##################################################
# Import archive

if len(sys.argv)==1:
    print "Please provide an input file."
    exit(1)

ar = archive.iarchive(sys.argv[1])

##################################################
# plot simple animation of phases

def plotphi(frame, engine=plt):
    f = np.array(frame.phi).reshape((frame.parameters['LX'], frame.parameters['LZ']))
    cax = engine.imshow(f, interpolation='lanczos', cmap='gray', origin='lower')
    cbar = plt.colorbar(cax)
    #engine.contourf(np.arange(0, frame.parameters['LX']),
    #                np.arange(0, frame.parameters['LZ']),
    #                f, levels = [0., .5])
    engine.contour(np.arange(0, frame.parameters['LX']),
                   np.arange(0, frame.parameters['LZ']),
                   f, levels = [.5])

def myplot(frame, engine):
    plotphi(frame, engine)
    #plot.domain(frame, 0, engine)
    engine.axes.set_aspect('equal', adjustable='box')
    engine.set_xlim([0, frame.parameters['LX']-1])
    engine.set_ylim([0, frame.parameters['LZ']-1])
    engine.axis('off')

anim = plot.animate(ar, myplot, show=True)

# to save the movie, rather run:
#anim = plot.animate(ar, myplot, show=False)
#plot.save_animation(anim, 'monolayer.mp4', 5)

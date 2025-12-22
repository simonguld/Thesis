import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import sys
# import and plot
import archive
import plot
from scipy import ndimage

##################################################
# Import archive

if len(sys.argv)==1:
    print "Please provide an input file."
    exit(1)

ar = archive.iarchive(sys.argv[1])

##################################################
# plot simple animation of phases

def myplot(frame, engine):
    plot.phasefields(frame, engine)
    plot.domains(frame, engine)
    engine.axes.set_aspect('equal', adjustable='box')
    engine.set_xlim([0, frame.parameters['LX']-1])
    engine.set_ylim([0, frame.parameters['LY']-1])
    engine.axis('off')

anim = plot.animate(ar, myplot, show=True)
exit(0)

# to save the movie, rather run:
anim = plot.animate(ar, myplot, show=False)
plot.save_animation(anim, 'movie.mp4', 5)

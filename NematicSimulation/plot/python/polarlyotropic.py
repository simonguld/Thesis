import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import sys
# import and plot
from archive.archive import loadarchive
from plot import lyotropic, animation

##################################################
# Import archive

if len(sys.argv)==1:
    print "Please provide an input file."
    exit(1)

ar = loadarchive(sys.argv[1])

##################################################
# Custom animation

def myplot(frame, engine):
    lyotropic.polarity(frame, engine)
    lyotropic.phi(frame, engine)
    #engine.set_xlim([0, frame.parameters['LX']-1])
    #engine.set_ylim([0, frame.parameters['LY']-1])

animation.animate(ar, myplot)

# to save the animation, rather run
#an = animation.animate(ar, myplot, show=False)
#animation.save(an, 'movie.mp4', 5)

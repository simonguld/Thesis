import matplotlib as mpl
#mpl.use('SVG')
import matplotlib.pyplot as plt
import sys
import numpy as np
from math import sqrt
from scipy import ndimage
# import and plot
from archive.archive import loadarchive
from plot import phases, animation

##################################################
# Init

if len(sys.argv)==1:
    print "Please provide an input file."
    exit(1)

ar = loadarchive(sys.argv[1])

oname = ""
if len(sys.argv)==3:
    oname = "_"+sys.argv[2]
    print "Output name is", sys.argv[2]

##################################################
# plot simple animation of phases

def myplot(frame, engine):
    phases.phasefields(frame, engine)
    #phases.domains(frame, engine)
    phases.com(frame, engine)
    phases.ellipses(frame, engine)
    #phases.walls(frame, engine)
    #phases.phase(frame, 0, engine)
    engine.axes.set_aspect('equal', adjustable='box')
    engine.set_xlim([0, frame.parameters['LX']-1])
    engine.set_ylim([0, frame.parameters['LY']-1])
    engine.axis('off')

#fr = ar.read_frame(0)
#plt.figure()
#phases.walls(fr)
#phases.phasefields(fr)
#plt.show()
#exit(0)

animation.animate(ar, myplot, show=True); exit(0)

# to save the movie, rather run:
an = animation.animate(ar, myplot, show=False)
animation.save(an, 'movie'+oname+'.mp4', 5)

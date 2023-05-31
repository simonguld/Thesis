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
    print ("Please provide an input file.")
    exit(1)

test = archive.iarchive(sys.argv[1])

##################################################
# Plot order on the tenth frame

plot.order(test[10])
plt.show()

##################################################
# Simple animation

plot.animate(test, plot.velocitywheel)

##################################################
# Custom animation

def myplot(frame, engine):
    plot.director(frame, engine, scale=True)
    plot.order(frame, engine)
    engine.set_xlim([0, frame.parameters['LX']-1])
    engine.set_ylim([0, frame.parameters['LZ']-1])

plot.animate(test, myplot)

##################################################
# Save animation without showing

def myplot2(frame, engine):
    plot.velocitywheel(frame, engine)
    engine.set_xlim([0, frame.parameters['LX']-1])
    engine.set_ylim([0, frame.parameters['LZ']-1])
    plt.title('Simple movie example')

# save only 20 first frames (5 fps)
#anim = plot.animate(test, myplot2, rng=[1, 20], show=False)
#plot.save_animation(anim, 'movie.mp4', 5)

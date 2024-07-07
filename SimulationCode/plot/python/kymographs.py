import matplotlib as mpl
mpl.use('SVG')
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
# plot all kymographs

# get kymographs
kymo_ux_x = []
kymo_uy_x = []
kymo_ux_y = []
kymo_uy_y = []
for f in ar.read_frames():
    v = phases.get_velocity_field(f.phi, f.veli + f.vela)
    vx = np.array([ i[0] for i in v ]).reshape((f.parameters['LX'], f.parameters['LY']))
    vy = np.array([ i[1] for i in v ]).reshape((f.parameters['LX'], f.parameters['LY']))

    kymo_ux_x += [ np.sum(vx, axis=1) ]
    kymo_uy_x += [ np.sum(vy, axis=1) ]
    kymo_ux_y += [ np.sum(vx, axis=0) ]
    kymo_uy_y += [ np.sum(vy, axis=0) ]

# average
avg = lambda k: ndimage.filters.uniform_filter(k, size=[5, 30], mode='constant')
kymo_ux_x = avg(kymo_ux_x)
kymo_uy_x = avg(kymo_uy_x)
kymo_ux_y = avg(kymo_ux_y)
kymo_uy_y = avg(kymo_uy_y)

# plot
plt.figure()
cm = plt.get_cmap('jet')

plt.subplot(221)
plt.axis('equal')
cax = plt.imshow(kymo_ux_x, origin='lower', cmap=cm, aspect='auto')
plt.colorbar(cax)
plt.title('ux over x')

plt.subplot(222)
plt.axis('equal')
cax = plt.imshow(kymo_uy_x, origin='lower', cmap=cm, aspect='auto')
plt.colorbar(cax)
plt.title('uy over x')

plt.subplot(223)
plt.axis('equal')
cax = plt.imshow(kymo_ux_y, origin='lower', cmap=cm, aspect='auto')
plt.colorbar(cax)
plt.title('ux over y')

plt.subplot(224)
plt.axis('equal')
cax = plt.imshow(kymo_uy_y, origin='lower', cmap=cm, aspect='auto')
plt.colorbar(cax)
plt.title('uy over y')

plt.savefig('kymograph'+oname+'.png')

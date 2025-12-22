################################################################################
#
# Plotting routines for the lyotropic model
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, atan2
import matplotlib.animation as ani
from scipy import ndimage
from itertools import product

################################################################################
# usefull functions

def get_density(ffi):
    """Compute density from an element of the ff array"""
    return sum(ffi)

def get_velocity(ffi):
    """Compute velocity from an element of the ff array"""
    d = get_density(ffi)
    return [
        (ffi[1] - ffi[2] + ffi[5] - ffi[6] - ffi[7] + ffi[8]) / d,
        (ffi[3] - ffi[4] + ffi[5] - ffi[6] + ffi[7] - ffi[8]) / d
    ]

def square(a):
    """Compute the square of an array"""
    return sum([i**2 for i in a])

def norm(a):
    """Compute the norm of an array"""
    return sqrt(square(a))

def get_director(QQxx, QQyx, LX, LY):
    """Compute the director field (S, nx, ny) from the Q_ij = S(2 n_i n_j- delta_ij) tensor"""
    S = np.vectorize(sqrt)(QQyx**2 + QQxx**2)
    nx = np.vectorize(sqrt)((1 + QQxx/S)/2)
    ny = np.sign(QQyx)*np.vectorize(sqrt)((1 - QQxx/S)/2)
    S  =  S.reshape((LX, LY))
    nx = nx.reshape((LX, LY))
    ny = ny.reshape((LX, LY))
    return S, nx, ny

def crop(a, dx, dy):
    s = a.shape[0:2]
    return a[dx:s[0]-dx, dy:s[1]-dy]

################################################################################
# actual plotting routines

def velocitymagnitude(frame, engine=plt, dx=0, dy=0):
    """Plot the magnitude of the velocity"""
    f = np.array([ norm(get_velocity(f)) for f in frame.ff ])
    f = f.reshape((frame.parameters['LX'], frame.parameters['LY']))
    f = crop(f, dx, dy)
    cax = engine.imshow(f.T, interpolation='lanczos', origin='lower')
    #cax = engine.contourf(f)
    cbar = plt.colorbar(cax)

def activity(frame, engine=plt):
    """Plot the activity field"""
    f = np.array(frame.Zeta).reshape((frame.parameters['LX'], frame.parameters['LY']))
    cax = engine.imshow(f.T, interpolation='lanczos', origin='lower')
    cbar = plt.colorbar(cax)

def velocity(frame, engine=plt):
    """Plot the velocity field (quiver)"""
    v = [ get_velocity(i) for i in frame.ff ]
    vx = np.array([ i[0] for i in v ]).reshape((frame.parameters['LX'], frame.parameters['LY']))
    vy = np.array([ i[1] for i in v ]).reshape((frame.parameters['LX'], frame.parameters['LY']))
    cax = engine.quiver(np.arange(0, frame.parameters['LX']),
                        np.arange(0, frame.parameters['LY']),
                        vx.T, vy.T,
                        #pivot='tail', units='dots', scale_units='dots'
                        )

def polarity(frame, engine=plt):
    """Plot the polarity field (quiver)"""
    px=np.array(frame.Px).reshape((frame.parameters['LX'], frame.parameters['LY']))
    py=np.array(frame.Py).reshape((frame.parameters['LX'], frame.parameters['LY']))
    cax = engine.quiver(
            np.arange(0, frame.parameters['LX']),
            np.arange(0, frame.parameters['LY']),
            px.T, py.T,
            color='k', linewidth=1,
            pivot='mid', headlength=0, headaxislength=0,
            scale=1, scale_units='xy'
            )

def polarity_vec(frame, engine=plt):
    """Plot the polarity field (quiver)"""
    px=np.array(frame.Px).reshape((frame.parameters['LX'], frame.parameters['LY']))
    py=np.array(frame.Py).reshape((frame.parameters['LX'], frame.parameters['LY']))
    cax = engine.quiver(
            np.arange(0, frame.parameters['LX']),
            np.arange(0, frame.parameters['LY']),
            px.T, py.T,
            color='k', linewidth=1,
            pivot='mid',
            scale=1, scale_units='xy'
            )

def velocitywheel(frame, engine=plt):
    """Plot the velocity as a colormap, where the colour denotes the direction in radians and the intensity the magnitude"""
    vel = [ get_velocity(f) for f in frame.ff ]
    mnorm = max([ norm(v) for v in vel ])
    cm = plt.get_cmap('hsv')
    img = np.array(
        [ cm((pi + atan2(v[0], v[1]))/(2.*pi), alpha=norm(v)/mnorm) for v in vel ]
    ).reshape((frame.parameters['LX'], frame.parameters['LY'], 4))
    cax = engine.imshow(np.transpose(img, (1,0,2)), interpolation='lanczos', origin='lower', cmap=cm)
    cbar = plt.colorbar(cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['0', '$\pi/2$', '$\pi$', '$3 \pi/2$','$2\pi$'])

def directorwheel(frame, engine=plt, scale=False):
    """Plot the director field"""
    S, nx, ny = get_director(frame.QQxx, frame.QQyx, frame.parameters['LX'], frame.parameters['LY'])
    cm = plt.get_cmap('hsv')
    img = np.array(
        [ cm((pi + atan2(nx[i], ny[i]))/(2.*pi), alpha=S[i]) for i in range(len(S)) ]
    ).reshape((frame.parameters['LX'], frame.parameters['LY'], 4))
    cax = engine.imshow(np.transpose(img, (1,0,2)), interpolation='lanczos', origin='lower', cmap=cm)
    cbar = plt.colorbar(cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['0', '$\pi/2$', '$\pi$', '$3 \pi/2$','$2\pi$'])

def vorticity(frame, engine=plt):
    """Plot the vorticity"""
    # save parameters
    LX = frame.parameters['LX']
    LY = frame.parameters['LY']
    # get velocity
    v = [ get_velocity(f) for f in frame.ff ]
    vx = np.array([ i[0] for i in v ]).reshape((frame.parameters['LX'], frame.parameters['LY']))
    vy = np.array([ i[1] for i in v ]).reshape((frame.parameters['LX'], frame.parameters['LY']))
    # vorticity computed with 3-point stencil
    w = [ [  (vy[(i-1)%LX][j] - 2*vy[i][j] + vy[(i+1)%LX][j])
            -(vx[i][(j-1)%LY] - 2*vx[i][j] + vx[i][(j+1)%LY])
            for j in range(LY)
          ] for i in range(LX)
        ]
    cax = engine.imshow(np.array(w).T, interpolation='lanczos', origin='lower')
    cbar = plt.colorbar(cax)

def order(frame, engine=plt):
    """Plot the order"""
    f = np.array([ frame.QQxx[i]**2 + frame.QQyx[i]**2 for i in range(len(frame.QQxx)) ])
    f = f.reshape((frame.parameters['LX'], frame.parameters['LY']))
    cax = engine.imshow(f.T, interpolation='lanczos', cmap='summer', origin='lower', clim=(0., 1.))
    cbar = plt.colorbar(cax)

def phi(frame, engine=plt):
    """Plot the phi order"""
    f = np.array(frame.phi).reshape((frame.parameters['LX'], frame.parameters['LY']))
    #cax = engine.imshow(f.T, interpolation='lanczos', cmap='summer', origin='lower', clim=(0., 1.))
    cax = engine.imshow(f.T, interpolation='lanczos', cmap='seismic', origin='lower')
    cbar = plt.colorbar(cax)

def phi_contour(frame, engine=plt):
    """Plot the phi field as contour"""
    f = np.array(frame.phi)
    engine.contour(np.arange(0, frame.parameters['LX']),
                   np.arange(0, frame.parameters['LY']),
                   f.reshape((frame.parameters['LX'], frame.parameters['LY'])),
                   levels = [.5])

def director(frame, engine=plt, scale=False):
    """Plot the director field"""
    S, nx, ny = get_director(frame.QQxx, frame.QQyx, frame.parameters['LX'], frame.parameters['LY'])
    x = []
    y = []
    for i, j in product(np.arange(frame.parameters['LX']), np.arange(frame.parameters['LY'])):
        f = S[i,j] if scale else 1.
        x.append(i + .5 - f*nx[i,j]/2.)
        x.append(i + .5 + f*nx[i,j]/2.)
        x.append(None)
        y.append(j + .5 - f*ny[i,j]/2.)
        y.append(j + .5 + f*ny[i,j]/2.)
        y.append(None)
    engine.plot(x, y, color='k', linestyle='-', linewidth=1)

def masks(frame, engine=plt):
    """Plot division/death masks"""
    m1 = np.array([ 1 if i else 0 for i in frame.division_mask ])
    m2 = np.array([ 1 if i else 0 for i in frame.death_mask ])
    engine.contour(np.arange(0, frame.parameters['LX']),
                   np.arange(0, frame.parameters['LY']),
                   m1.reshape((frame.parameters['LX'], frame.parameters['LY'])).T,
                   levels = [.5], colors = ['b'])
    engine.contour(np.arange(0, frame.parameters['LX']),
                   np.arange(0, frame.parameters['LY']),
                   m2.reshape((frame.parameters['LX'], frame.parameters['LY'])).T,
                   levels = [.5], colors = [ 'r' ])

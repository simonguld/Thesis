################################################################################
#
# Plotting routines for the phase field model
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, atan2
import matplotlib.animation as ani
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage, stats
from itertools import product

def get_velocity_field(phases, vel):
    """Compute the collective velocity field from a collection of phase-fields
     and their velocities"""
    v = []
    for k in range(len(phases[0])):
        v = v + [ [ sum([ vel[n][0]*phases[n][k] for n in range(len(phases)) ]),
                    sum([ vel[n][1]*phases[n][k] for n in range(len(phases)) ]) ] ]
    return np.array(v)

def phasefields(frame, engine=plt):
    """Plot all phase fields."""
    for p in frame.phi:
        engine.contour(np.arange(0, frame.parameters['LX']),
                       np.arange(0, frame.parameters['LY']),
                       p.reshape((frame.parameters['LX'], frame.parameters['LY'])).T,
                       #levels = [1e-10, 1e-5, .5])
                       levels = [.5],
                       #color='mediumblue'
                       colors='k')

def phasefieldsdens(frame, engine=plt):
    """Plot all phase fields."""
    totphi = np.zeros(frame.parameters['LX']*frame.parameters['LY'])
    for i in range(0, len(frame.phi)):
        totphi += frame.phi[i]*frame.parameters['walls']
        for j in range(i+1, len(frame.phi)):
            totphi += frame.phi[i]*frame.phi[j]
    #totphi += frame.parameters['walls']*frame.parameters['walls']

    totphi = totphi.reshape((frame.parameters['LX'], frame.parameters['LY'])).T
    cmap = LinearSegmentedColormap.from_list('mycmap', ['grey', 'white'])
    engine.imshow(totphi, interpolation='lanczos', cmap=cmap, origin='lower')

def phasefieldsarea(frame, engine=plt):
    """Plot all phase fields."""
    for i in range(len(frame.phi)):
        p = frame.phi[i]
        engine.contourf(np.arange(0, frame.parameters['LX']),
                       np.arange(0, frame.parameters['LY']),
                       p.reshape((frame.parameters['LX'], frame.parameters['LY'])).T,
                       #levels = [1e-10, 1e-5, .5])
                       levels = [.5, 10.],
                       #color='mediumblue'
                       colors=str(frame.area[i]/(np.pi*frame.parameters['R']**2)))


def get_com(phi):
    """Compute center-of-mass of a phase-field"""
    LX, LY = phi.shape
    # project on each axis
    phix = phi.mean(1)
    phiy = phi.mean(0)
    # map points to [-pi, pi] instead of the box and compute mean sin and cos
    sx = np.dot(phix, np.sin(np.linspace(-np.pi, np.pi, num=LX, endpoint=False)))
    cx = np.dot(phix, np.cos(np.linspace(-np.pi, np.pi, num=LX, endpoint=False)))
    sy = np.dot(phiy, np.sin(np.linspace(-np.pi, np.pi, num=LY, endpoint=False)))
    cy = np.dot(phiy, np.cos(np.linspace(-np.pi, np.pi, num=LY, endpoint=False)))
    # get mean angle
    mx = np.arctan2(sx , cx)
    my = np.arctan2(sy , cy)
    # map back to the box
    return [ (mx+np.pi)/2/np.pi*LX, (my+np.pi)/2/np.pi*LY ]

def com(frame, engine=plt):
    """Compute and plot the center-of-mass of each cell. This is not working with PBC yet..."""
    for p in frame.phi:
        p = p.reshape((frame.parameters['LX'], frame.parameters['LY']))
        c = get_com(p)
        engine.plot(c[0], c[1], 'ro')

def ellipses(frame, engine=plt):
    """Plot the shape-ellipses of each cell."""
    for n in range(frame.parameters['nphases']):
        radius = np.sqrt(frame.area[n]/np.pi/(1-frame.Q_order[n]**2))
        print frame.Q_order[n], radius
        omega  = frame.Q_angle[n]
        p = frame.phi[n].reshape((frame.parameters['LX'], frame.parameters['LY']))
        c = get_com(p)
        an = np.linspace(-omega, 2*np.pi-omega, 100)
        engine.plot(c[0] + radius*(1+10*frame.Q_order[n])*np.cos(an),
                    c[1] + radius*(1-10*frame.Q_order[n])*np.sin(an))

def velc(frame, engine=plt):
    """Print contractile part of the velocity"""
    for i in range(frame.parameters['nphases']):
        p = frame.phi[i].reshape((frame.parameters['LX'], frame.parameters['LY']))
        c = get_com(p)
        v = frame.velc[i]
        # correction factor
        a = frame.parameters['ninfo']*frame.parameters['nsubsteps']
        engine.arrow(c[0], c[1], a*v[0], a*v[1], color='g')


def velp(frame, engine=plt):
    """Print inactive part of the velocity"""
    for i in range(frame.parameters['nphases']):
        p = frame.phi[i].reshape((frame.parameters['LX'], frame.parameters['LY']))
        c = get_com(p)
        v = frame.velp[i]
        # correction factor
        a = frame.parameters['ninfo']*frame.parameters['nsubsteps']
        engine.arrow(c[0], c[1], a*v[0], a*v[1], color='b')

def pol(frame, engine=plt):
    """Print active part of the velocity"""
    for i in range(frame.parameters['nphases']):
        p = frame.phi[i].reshape((frame.parameters['LX'], frame.parameters['LY']))
        c = get_com(p)
        v = frame.pol[i]
        a = 4#frame.parameters['ninfo']*frame.parameters['nsubsteps']
        engine.arrow(c[0], c[1],  a*v[0],  a*v[1], color='k')
        #engine.arrow(c[0], c[1], -a*v[0], -a*v[1], color='k')

def velf(frame, engine=plt):
    """Print active part of the velocity"""
    for i in range(frame.parameters['nphases']):
        p = frame.phi[i].reshape((frame.parameters['LX'], frame.parameters['LY']))
        c = get_com(p)
        v = frame.velf[i]
        # correction factor
        a = frame.parameters['ninfo']*frame.parameters['nsubsteps']
        engine.arrow(c[0], c[1], a*v[0], a*v[1], color='brown')

def vela(frame, engine=plt):
    """Print active part of the velocity"""
    for i in range(frame.parameters['nphases']):
        p = frame.phi[i].reshape((frame.parameters['LX'], frame.parameters['LY']))
        c = get_com(p)
        v = frame.pol[i]
        a = frame.parameters['alpha']/frame.parameters['xi']*frame.parameters['ninfo']*frame.parameters['nsubsteps']
        engine.arrow(c[0], c[1], a*v[0], a*v[1], color='r')


def vel(frame, engine=plt):
    """Print active part of the velocity"""
    for i in range(frame.parameters['nphases']):
        p = frame.phi[i].reshape((frame.parameters['LX'], frame.parameters['LY']))
        c = get_com(p)
        v = frame.vela[i] + frame.veli[i] + frame.velf[i] #+ frame.velc[i]
        a = frame.parameters['ninfo']*frame.parameters['nsubsteps']
        engine.arrow(c[0], c[1], a*v[0], a*v[1], color='k', head_width=1, zorder=10)


def phase(frame, n, engine=plt):
    """Plot single phase as a density plot"""
    f = np.array(frame.phi[n])
    f = f.reshape([frame.parameters['LX'], frame.parameters['LY']])
    cax = engine.imshow(f.T, interpolation='lanczos', cmap='Greys', origin='lower'
#, clim=(0., 1.)
)
    cbar = plt.colorbar(cax)

def velocity_field(frame, size=3, step=1, engine=plt):
    """Plot the total veloctity field assiciated with the cells"""
    v = get_velocity_field(frame.phi, frame.vela+frame.veli)
    vx = np.array([ i[0] for i in v ]).reshape((frame.parameters['LX'], frame.parameters['LY']))
    vy = np.array([ i[1] for i in v ]).reshape((frame.parameters['LX'], frame.parameters['LY']))
    vx = ndimage.filters.uniform_filter(vx, size=size, mode='constant')
    vy = ndimage.filters.uniform_filter(vy, size=size, mode='constant')
    cax = engine.quiver(
            np.arange(0, frame.parameters['LX'])[::step],
            np.arange(0, frame.parameters['LY'])[::step],
            vx[::step, ::step].T, vy[::step, ::step].T, pivot='tail', units='dots', scale_units='dots')

def walls(frame, engine=plt):
    """Plot the wall phase-field"""
    f = frame.parameters['walls']
    f = f.reshape([frame.parameters['LX'], frame.parameters['LY']])
    cax = engine.imshow(f.T, cmap='Greys', origin='lower', clim=(0., 1.))

def domain(frame, n, engine=plt):
    """Plot the restricted domain of a single cell"""
    plot = lambda m, M: engine.fill([ m[0], M[0], M[0], m[0], m[0], None ],
                                    [ m[1], m[1], M[1], M[1], m[1], None ],
                                    color = 'b', alpha=0.04)
    LX = frame.parameters['LX']
    LY = frame.parameters['LY']
    m = frame.domain_min[n]
    M = frame.domain_max[n]

    if(m[0]==M[0]):
        m[0] += 1e-1
        M[0] -= 1e-1
    if(m[1]==M[1]):
        m[1] += 1e-1
        M[1] -= 1e-1

    if(m[0]>M[0] and m[1]>M[1]):
        plot(m, [ LX, LY ])
        plot([ 0, 0 ], M)
        plot([ m[0], 0 ], [ LX, M[1] ])
        plot([0, m[1] ], [ M[0], LY ])
    elif(m[0]>M[0]):
        plot(m, [ LX, M[1] ])
        plot([ 0, m[1] ], M)
    elif(m[1]>M[1]):
        plot(m, [ M[0], LY ])
        plot([ m[0], 0 ], M)
    else:
        plot(m, M)

def domains(frame, engine=plt):
    """Plot the restricted domains of each cell"""
    for n in range(frame.parameters['nphases']):
        domain(frame, n, engine)

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

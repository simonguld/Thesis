from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import pyplot as plt
from ..base_modules import plotlib
from itertools import product
from . import polarPy
import numpy as np

def velocity(frame, engine=plt, avg=4):
    """ Plot the velocity field (quiver) """
    plotlib.velocity(frame, engine, avg)


def velocitystreamline(frame, engine=plt, dens=5):
    """ Plot velocity streamlines """
    plotlib.velocitystreamline(frame, engine, dens)


def velocitymagnitude(frame, engine=plt, dx=0, dy=0):
    """ Plot the magnitude of the velocity """
    plotlib.velocitymagnitude(frame, engine, dx, dy)


def velocitywheel(frame, engine=plt):
    """ Plot the velocity as a colormap, where the colour denotes
    the direction in radians and the intensity its magnitude """
    plotlib.velocitywheel(frame, engine)


def vorticity(frame, engine=plt):
    """ Plot the vorticity """
    plotlib.vorticity(frame, engine)


def okubo_weiss(frame, engine=plt, W0=.0):
    """ Plot the Okubo Weiss parameter """
    plotlib.okubo_weiss(frame, engine, W0)


def order(frame, engine=plt):
    """ Plot the order """
    p = np.sqrt(frame.Px**2 + frame.Py**2)
    p = p.reshape(frame.LX, frame.LY)
    # plot using engine
    im = engine.imshow(p.T, cmap='Greens', interpolation='lanczos', origin='lower')
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax)


def charge(frame, engine=plt):
    """ plot the charge """
    # get charge array
    w = polarPy.get_charge(frame.Px, frame.Py, frame.LX, frame.LY)
    # plot using engine
    # im = engine.imshow(w.T, cmap='coolwarm', origin='lower', clim=(-1.0, 1.0))
    im = engine.imshow(w.T, cmap='bwr', interpolation='lanczos', origin='lower', clim=(-1.0, 1.0))
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.ax.set_yticklabels(['$-1$', '$-1/2$', '$0$', '$1/2$','$1$'])


def diffusive_charge_density(frame, engine=plt):
    """ Plot the Jacobian of the P order parameter """
    D = polarPy.get_diffusive_charge_density(frame.Px, frame.Py, frame.LX, frame.LY)
    # plot using engine
    im = engine.imshow(D.T, cmap='bwr', interpolation='lanczos', origin='lower', clim=(-.1, .1))
    # im = engine.imshow(D.T, cmap='bwr', interpolation='lanczos', origin='lower')
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax)


def polarity(frame, engine=plt, avg=4, normed=True):
    """ Plot the polarity field (quiver) """
    # get polarity
    Px = plotlib.average(frame.Px, frame.LX, frame.LY, avg)
    Py = plotlib.average(frame.Py, frame.LX, frame.LY, avg)
    
    # Px = frame.Px.reshape(frame.LX, frame.LY)[::3, ::3]
    # Py = frame.Py.reshape(frame.LX, frame.LY)[::3, ::3]
    # avg = 3
    # normalize
    if normed:
        Px /= np.sqrt(Px**2 + Py**2)
        Py /= np.sqrt(Px**2 + Py**2)
    # plot using engine
    engine.quiver( np.arange(frame.LX, step=avg)+avg//2, np.arange(frame.LY, step=avg)+avg//2, 
                   Px.T, Py.T, pivot='middle', units='dots', scale_units='dots', width=.7, headwidth=4 )


def polaritywheel(frame, engine=plt):
    """ Plot the polarity as a colormap, where the colour denotes
    the direction in radians and the intensity its magnitude """
    # get polarity
    Px = frame.Px.reshape(frame.LX, frame.LY)
    Py = frame.Py.reshape(frame.LX, frame.LY)
    # get normalized direction (ang in [0,1])
    ang = (np.arctan2(Py, Px) + np.pi) / (2*np.pi)
    # get normalized magnitude (speed in [0,1])
    speed = np.linalg.norm(np.stack([Px, Py]), axis=0)
    speed /= np.amax(speed)
    # plot using engine
    cm = plt.get_cmap('hsv')
    img = cm(ang, alpha=speed)
    im = engine.imshow(np.transpose(img, (1,0,2)), cmap=cm, interpolation='lanczos', origin='lower')
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$','$\pi$'])


def defects(frame, engine=plt, ms=3):
    """ Plot defects of the polarity field """
    # plot using engine
    for d in polarPy.get_defects(frame.Px, frame.Py, frame.LX, frame.LY):
        if d['charge'] == 0.5:
            engine.plot(d["pos"][0], d["pos"][1], 'go', markersize=ms)
        else:
            engine.plot(d["pos"][0], d["pos"][1], 'bs', markersize=ms)

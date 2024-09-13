from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import pyplot as plt
from itertools import product
from ..base_modules import plotlib
from . import nematicPy
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
    # get order S
    S, _, _ = nematicPy.get_director(frame.QQxx, frame.QQyx)
    S = S.reshape(frame.LX, frame.LY)
    # plot using engine
    im = engine.imshow(S.T, cmap='Greens', interpolation='lanczos', origin='lower', clim=(0., 1.))
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax)


def charge(frame, engine=plt):
    """ plot the charge """
    # get charge array
    w = nematicPy.get_charge(frame.QQxx, frame.QQyx, frame.LX, frame.LY)
    # plot using engine
    im = engine.imshow(w.T, cmap='coolwarm', origin='lower', clim=(-.5, .5))
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax, ticks=[-.5, -.25, 0, 0.25, .5])
    cbar.ax.set_yticklabels(['$-1/2$', '$-1/4$', '$0$', '$1/4$','$1/2$'])


def diffusive_charge_density(frame, engine=plt):
    """ Plot the Jacobian of the Q order parameter """
    D = nematicPy.get_diffusive_charge_density(frame.QQxx, frame.QQyx, frame.LX, frame.LY)
    # plot using engine
    # im = engine.imshow(D.T, cmap='bwr', interpolation='lanczos', origin='lower')#, clim=(-1., 1.))
    im = engine.imshow(D.T, cmap='bwr', origin='lower')#, clim=(-1., 1.))
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax)


def director(frame, engine=plt, scale=False, avg=4, ms = 1, alpha=1, lw=.5):
    """ Plot the director field """
    # get nematic field tensor
    Qxx = frame.QQxx.reshape(frame.LX, frame.LY)
    Qyx = frame.QQyx.reshape(frame.LX, frame.LY)
    # get order S and director (nx, ny)
    S, nx, ny = nematicPy.get_director(Qxx, Qyx)
    # plot using engine
    x, y = [], []
    for i, j in product(np.arange(frame.LX, step=avg), np.arange(frame.LY, step=avg)):
        f = avg*(S[i,j] if scale else 1.)
        x.append(i - f*nx[i,j]/2.)
        x.append(i + f*nx[i,j]/2.)
        x.append(None)
        y.append(j - f*ny[i,j]/2.)
        y.append(j + f*ny[i,j]/2.)
        y.append(None)
    engine.plot(x, y, color='k', linestyle='-', markersize=1, linewidth=lw, alpha=alpha)


def directorwheel(frame, engine=plt):
    """ Plot the director as a colormap, where the colour denotes
    the direction in radians and the intensity its magnitude """
    # get nematic field tensor
    Qxx = frame.QQxx.reshape(frame.LX, frame.LY)
    Qyx = frame.QQyx.reshape(frame.LX, frame.LY)
    # get order S and director (nx, ny)
    S, nx, ny = nematicPy.get_director(Qxx, Qyx)
    # get normalized direction (ang in [0,1])
    ang = (np.arctan(ny/nx) + .5*np.pi) / np.pi
    # get normalized magnitude (speed in [0,1])
    speed = np.linalg.norm(np.stack([nx, ny]), axis=0)
    speed /= np.amax(speed)
    # plot using engine
    cm = plt.get_cmap('hsv')
    img = cm(ang, alpha=speed)
    im = engine.imshow(np.transpose(img, (1,0,2)), interpolation='lanczos', origin='lower', cmap=cm)
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['$-\pi/2$', '$-\pi/4$', '$0$', '$\pi/4$','$\pi/2$'])


def defects(frame, engine=plt, arrow_len=0, ms=3, alpha=1):
    """ Plot defects of the nematic field Q.
    arrow_len: If non-zero plot speed of defects as well.
    """
    # plot using engine
    for d in nematicPy.get_defects(frame.QQxx, frame.QQyx, frame.LX, frame.LY, arrow_len!=0):
        if d['charge'] == 0.5:
            engine.plot(d["pos"][0], d["pos"][1], 'go', markersize=ms, alpha=alpha)
            # plot direction of pos defects
            if arrow_len != 0:
                engine.arrow( d['pos'][0], d['pos'][1],
                              arrow_len*np.cos(d['angle']),
                              arrow_len*np.sin(d['angle']),
                              color='r', head_width=.5, head_length=.5
                            )
        else:
            engine.plot(d["pos"][0], d["pos"][1], 'b^', markersize=ms, alpha=alpha)

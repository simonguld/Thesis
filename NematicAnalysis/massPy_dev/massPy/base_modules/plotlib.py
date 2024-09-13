from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import pyplot as plt
from matplotlib import colors
from . import flow
import numpy as np

def crop(a, dx, dy):
    s = a.shape[0:2]
    return a[dx:s[0]-dx, dy:s[1]-dy]


def average(vi, LX, LY, avg):
    return np.mean( vi.reshape(LX//avg, avg, LY//avg, avg), axis=(1,3) )


def velocity(frame, engine=plt, avg=4):
    """ Plot the velocity field (quiver) """
    # get velocity
    vx, vy = flow.velocity(frame)
    vx = average(vx, frame.LX, frame.LY, avg)
    vy = average(vy, frame.LX, frame.LY, avg)
    # plot using engine
    engine.quiver( np.arange(frame.LX, step=avg)+avg//2, np.arange(frame.LY, step=avg)+avg//2, 
                   vx.T, vy.T, pivot='mid', units='dots', scale_units='dots' 
                 )


def velocitystreamline(frame, engine=plt, dens=5):
    vx, vy = flow.velocity(frame)
    engine.streamplot( np.arange(frame.LX), np.arange(frame.LX), 
                       vx.T, vy.T, linewidth=1, arrowsize=.3, density=dens, color='k'
                     )


def velocitymagnitude(frame, engine=plt, dx=0, dy=0):
    """ Plot the magnitude of the velocity """
    # get velocity
    v = flow.velocity(frame)
    # get velocity magnitude (aka. speed)
    v = np.linalg.norm(v, axis=0)
    v = crop(v, dx, dy)
    # plot using engine
    im = engine.imshow(v.T, cmap='turbo', interpolation='lanczos', origin='lower')
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax)


def velocitywheel(frame, engine=plt):
    """ Plot the velocity as a colormap, where the colour denotes
    the direction in radians and the intensity its magnitude """
    # get velocity
    v = flow.velocity(frame)
    # get normalized direction (ang in [0,1])
    ang = (np.arctan2(v[1], v[0]) + np.pi) / (2*np.pi)
    # get normalized magnitude (speed in [0,1])
    speed = np.linalg.norm(v, axis=0)
    speed /= np.amax(speed)
    # plot using engine
    cm = plt.get_cmap('hsv')
    img = cm(ang, alpha=speed)
    im = engine.imshow(np.transpose(img, (1,0,2)), interpolation='lanczos', origin='lower', cmap=cm)
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$','$\pi$'])


def vorticity(frame, engine=plt):
    """ Plot the vorticity """
    # vorticity computed with 5-point stencil
    w = flow.vorticity(frame)
    # plot using engine
    im = engine.imshow(w.T, cmap='turbo', interpolation='lanczos', origin='lower')
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    plt.colorbar(im, cax)


def okubo_weiss(frame, engine=plt, W0=.0):
    """ Plot the Okubo Weiss parameter """
    # Okubo Weiss parameter
    W = flow.okubo_weiss(frame)
    # vorticity
    w = flow.vorticity(frame)
    w = np.where(W < -W0, np.sign(w), 0)
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['blue', 'white', 'red'])
    # plot using engine
    im = engine.imshow(w.T, cmap=cmap, origin='lower')
    # add colorbar
    divider = make_axes_locatable(engine)
    cax = divider.append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(im, cax, ticks=[-.667, 0., 0.667])
    cbar.ax.set_yticklabels(['$-1$', '$0$', '$+1$'])


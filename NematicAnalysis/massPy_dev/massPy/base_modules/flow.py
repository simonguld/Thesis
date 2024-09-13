from . import numdiff
import numpy as np

def density(frame):
    if 'density' in frame.__dict__.keys():
        return frame.density
    else:
        return np.sum(frame.ff, axis=1)

def velocity(frame):
    
    LX, LY = frame.LX, frame.LY
    if 'vx' and 'vy' in frame.__dict__.keys():
        return frame.vx.reshape(LX, LY), frame.vy.reshape(LX, LY)
    else:
        ff = frame.ff
        FFx = frame.FFx
        FFy = frame.FFy
        d = density(frame)

        # calculate the AA_LBF parameter
        isGuo = frame.isGuo if 'isGuo' in frame.__dict__.keys() else True
        AA_LBF = .5 if isGuo else frame.tau

        return np.asarray([ (ff.T[1] - ff.T[2] + ff.T[5] - ff.T[6] - ff.T[7] + ff.T[8] + AA_LBF * FFx) / d,
                            (ff.T[3] - ff.T[4] + ff.T[5] - ff.T[6] + ff.T[7] - ff.T[8] + AA_LBF * FFy) / d
                        ]).reshape(2, LX, LY)

def vorticity(frame):
    vx, vy = velocity(frame)
    return numdiff.curl2D(vx, vy)

def okubo_weiss(frame):
    # see doi:10.1016/j.dsr2.2004.09.013
    # flow & vorticity field
    vx, vy = velocity(frame)
    # nomral strain
    sn = numdiff.derivX(vx) - numdiff.derivY(vy)
    # shear strain
    ss = numdiff.derivX(vy) + numdiff.derivY(vx)
    # vorticity
    w = numdiff.curl2D(vx, vy)
    return sn**2 + ss**2 - w**2

def strain_rate():
    pass

def rotation_rate():
    pass
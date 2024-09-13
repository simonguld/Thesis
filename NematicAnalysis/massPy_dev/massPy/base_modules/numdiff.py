import numpy as np

def gradient(F, axis, pbc=True):
    if pbc:
        return np.gradient(
            np.pad(F, 2, mode='wrap'),
            axis=axis
        )[F.ndim * tuple([slice(2,-2)])]
    else:
        return np.gradient(F, axis=axis)

def derivX(F, pbc=True):
    return gradient(F, 0, pbc)

def derivY(F, pbc=True):
    return gradient(F, 1, pbc)

def curl2D(Fx, Fy, pbc=True):
    return derivX(Fy, pbc) - derivY(Fx, pbc)

def div2D(Fx, Fy, pbc=True):
    return derivX(Fx, pbc) + derivY(Fy, pbc)

def jacobian(Fx, Fy, pbc=True):
    return derivX(Fx, pbc)*derivY(Fy, pbc) - derivX(Fy, pbc)*derivY(Fx, pbc)


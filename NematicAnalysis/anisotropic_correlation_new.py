import numpy as np
from numba import njit
import matplotlib.pyplot as plt
#@njit
#def getQ(px, py):
    # (p_ip_i - I/2); returns qx, qxy, qyy
#    return px**2-0.5, px*py, py**2-0.5


@njit
def corr(s1, s2, d , qxx, qxy, qyy, smean):

    #helper terms
    term = 2*d[0]*d[1]*qxy
    g = (s1-smean)*(s2-smean)
    #correlation functions
    cor_par =  0.5*g*(qxx*d[0]**2 + qyy*d[1]**2 + term + 1)
    cor_perp =  0.5*g*(qyy*d[0]**2 + qxx*d[1]**2 - term + 1)

    return cor_par, cor_perp


@njit
def correct_for_pbc(x1, y1, x2, y2, box_size):
    dx = x1 - x2
    dy = y1 - y2
    dx = dx - box_size*round(dx / box_size)
    dy = dy - box_size*round(dy / box_size)
    return dx, dy


@njit
def doloop(s, Qxx, Qyx, size, cor_par, cor_perp, count):

    n = len(s)
    smean  = np.mean(s)
    for i in range(n):
        for j in range(n):
            x1, y1 = i, j
            qxx, qxy, qyy = Qxx[i, j], Qyx[i, j], -Qxx[i, j]
            #px1, py1 = Px[i, j], Py[i, j]
            s1 = s[i ,j]
            #qxx, qxy, qyy = getQ(px1, py1)

            #coudnt think of how to avoid nested loops 
            for k in range(n):
                for l in range(n):
                    x2, y2 = k, l
                    s2 = s[k, l]

                    dx, dy  = correct_for_pbc(x1, y1, x2, y2, size)
                    #dx = abs(x2-x1)
                    #dy = abs(y2-y1)
                    dist =  np.sqrt(dx**2+dy**2)
                    d = np.array([dx, dy])

                    if dist != 0:
                        a, b = corr(s1, s2, d/dist, qxx, qxy, qyy, smean)
                    else:
                        a, b = 1, 1
                    dist = int(dist)
                    cor_par[dist] += a
                    cor_perp[dist] += b
                    count[dist] += 1

    return cor_par/count, cor_perp/count




def aniso_corr(field, Qxx, Qyx):

    size = len(field)
    dists = np.arange(int(np.sqrt(2)*size/2)+1)
    cor_par = np.zeros_like(dists, dtype=float)
    cor_perp = np.zeros_like(dists, dtype=float)
    count = np.zeros_like(dists)
    cpar, cperp = doloop(field, Qxx, Qyx, size, cor_par, cor_perp, count)
     
    cpar, cperp = cpar/np.var(field), cperp/np.var(field)
    cpar[0], cperp[0] = 1, 1

    return cpar, cperp
#%%

#cpar, cperp = aniso_corr(field, Px, Py)

#%%
#plt.plot(cpar[:50], label = "parallel correlation")
#plt.plot(cperp[:50], label = "Perp correlation")
#plt.legend()

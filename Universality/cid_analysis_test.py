#!/usr/bin/env python
import sys
import os


#sys.path.insert(0,'/groups/astro/robinboe/mass_analysis')
#sys.path.insert(0,'/groups/astro/robinboe/mass_analysis/computable-information-density')

from joblib import Parallel, delayed
from ComputableInformationDensity.cid_interlacings import interlaced_time, cid2d
from massPynpz.nematic.nematicPy import get_defects
from numpy import zeros
import pickle
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from massPynpz import archive 
from massPynpz.nematic import plot as nem_plot
from massPynpz.nematic import nematicPy
from massPynpz.base_modules import plotlib
import warnings
import pickle
import os.path


def main_cid(rep,zeta):

    nframes, nx, ny = 512, 1024, 1024

    # instantiate CID object:
    CID = interlaced_time(8, 16)

    defects = zeros((nframes, nx, ny), dtype=int)
    defect_density = []

    ar = archive.loadarchive(f'/lustre/astro/kpr279/ns2048/output_test_zeta_{zeta}/output_test_zeta_{zeta}_counter_{rep}')

    print(zeta,rep)

    for i in range(nframes):
        frame = ar[i]
        for defect in get_defects(frame.QQxx, frame.QQyx, frame.LX, frame.LY):
            ix, iy = defect['pos']
            ix, iy = int(ix) - nx//2, int(iy) - ny//2
            if (0 <= ix < nx) and (0 <= iy < ny): defects[i, ix, iy] = 1

        defect_density.append(defects[i,:,:].mean())

    # compute cid and cid shuffle:
    cid_, cid_shuff = CID(defects)

    res_cid = {
        'zeta' : ar.zeta,
        'cid' : cid_,
        'cid_shuffle' : cid_shuff,
        'lambda' : 1. - cid_/cid_shuff
    }

    res_defect_density = {
        'zeta' : ar.zeta,
        'density' : defect_density
    }

    with open(f"/lustre/astro/robinboe/HD_CID/cid-results-2048/cid-results-L-2048-zeta-{zeta}-rep-{rep+1}.pkl", "wb") as file:
        pickle.dump(res_cid, file)

    with open(f"/lustre/astro/robinboe/HD_CID/defect-density-results-2048/defect-density-results-L-2048-zeta-{zeta}-rep-{rep+1}.pkl", "wb") as file:
        pickle.dump(res_defect_density, file)
    


def main(rep, zeta):
    main_cid(rep, zeta)

if __name__ == "__main__":
    zeta = float(sys.argv[1])
    rep = int(sys.argv[2])
    main(rep, zeta)


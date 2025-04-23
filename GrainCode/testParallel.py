import GE 
from rve import rve 
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

Rve = rve("parallelTest")
Rve.Mu_roundness = 0.75
Rve.Mu_circularity = 1.0
Rve.Std_roundness = 0.01
Rve.Std_circularity = 0.01
nSims = 1

# Run in parallel
num_cores = multiprocessing.cpu_count()
print (num_cores)
Parallel(n_jobs=num_cores)(delayed(GE.makeParticles)(Rve,i) for i in np.arange(nSims))
Rve.createParticles() #create particles

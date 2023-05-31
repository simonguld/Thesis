import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from grain import Grain

os.chdir('Thesis\SiavashCode')

# Output contact and updated position file
morphTypes = 4
posCOM = np.array([0,0])

# Instantiate grains
grains = np.empty(morphTypes, dtype=object)             
for n in range(morphTypes):
    propFile =  "exampleDir/grainproperty" + str(n)+ ".dat"
    grains[n] = Grain(propFile)



# Collect in patches
for n in range(morphTypes):
	fig, ax = plt.subplots(figsize = (5,5))
	patches = []
	pts = grains[n]._points
	poly = Polygon(grains[n]._points, True) 


	patches.append(poly)
	# Setup up patchCollection 
	pCol = PatchCollection(patches, facecolors='dimgray',edgecolors='black', lw=0.1)
	ax.add_collection(pCol)
	ax.scatter(pts[:,0], pts[:,1],c='r')
	
	plt.show()
#plt.savefig(figDir + '/step_' + str(step) + '.png', format='png' )
#plt.close(fig)


    
#return grains

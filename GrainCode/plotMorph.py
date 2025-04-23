import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import grain
import GE
from scipy.interpolate import splprep, splev
import os 
pad = 2
radBound = 10
count = 0  
numTrials = 1
numParts = 900
for rnd in np.arange(0.8,1.05,0.05):
	Dir = "roundPics/R%.2f/"%rnd
	os.makedirs(Dir,exist_ok = True)	
	for j in range(numTrials):
		for k in range(numParts):
			morphFile = "roundnessTest/morphsR_%.2f/shapes/grainproperty%d.dat"%(rnd,k)
			Particle = grain.Grain(morphFile)

			lset = Particle.lset
			pts = Particle._points + [ 0.10270715, -0.07539873] 
			x,y = pts[:,0],pts[:,1]
			element_list = GE.makeLists(3)
			length = len(element_list[:,0])
			radiusCurvatures = np.zeros(length) #list of radii of curvatures 
			
			tck, u = splprep([x,y], u=None, s=0.0, per = 1) #Fit spline 
			u_new = np.linspace(u.min(), u.max(), length)
			x, y = splev(u_new, tck, der=0) #Get new points on smoothed spline
			fig1 = plt.figure(count )
			ax1 = fig1.add_subplot()
			maxRadius,area = GE.getLsetProps(x,y,lset) #get maximum radius of an inscribed circle & particle area
			for e_num in range(length - 1): #get curvature costs
				element = element_list[e_num]
				i = int(element[0])
				last = int(element[1])
				next = int(element[2])
				weight = int(element[3])  #element weight
				try: 
					cval = GE.calc_curvature(x,y, last, next, i) #current curvature at this point 
					radiusCurvatures[e_num] = abs(1./cval)*(1./maxRadius)
				except: 
					radiusCurvatures[e_num] = radBound 

				#plt.text(x[i],y[i], "%.2f"%radiusCurvatures[e_num])
			radiusCurvaturesNormedFiltered = np.array([radiusCurvatures[i]  if radiusCurvatures[i] < radBound else radBound for i in range(length)])
			roundness = np.mean(radiusCurvaturesNormedFiltered)
		#	print (rnd,j,roundness,radiusCurvaturesNormedFiltered)
			print (rnd,j,roundness,np.amin(radiusCurvatures[1:-1]))
			plt.plot(pts[:,0],pts[:,1])
		#	plt.scatter(x,y,c='r')
			ax1.set_aspect('equal')			
			plt.axis('off')
			plt.savefig(Dir + "%d"%k) 
			plt.close()
		#	plt.show()
			count += 1

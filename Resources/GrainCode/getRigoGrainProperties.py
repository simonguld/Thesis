import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import GE 

numOutputPoints = 1000
badGrains = [14,22,36,52,55,56,70,71,74,75,76,85]
totalParts = 100
for num in range(15,totalParts):
	print ("num",num) 
	if (num in badGrains): continue
	morphName = "rigoGrains/grainproperty%d.dat"%num 
	f = open(morphName,'r')
	lines = f.readlines()
	nPoints = int(lines[3])
	points = np.array([float(val) for val in lines[4].split()])
	points = np.reshape(points,(nPoints,2))*2
	nLset = np.array([int(float(val)) for val in lines[6].split()])
	xdim, ydim = nLset

	plt.plot(points[:,0],points[:,1])
	
	tck, u = splprep([points[:,0],points[:,1]], u=None, s=80.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), numOutputPoints)
	x1, y1 = splev(u_new, tck, der=0)#points generated from smoothe spline 

	lset = GE.getLset(x1,y1)
	 
	xd1,yd1 = splev(u_new,tck,der = 1) #x'(t),y'(t)
	xd2,yd2 = splev(u_new,tck,der = 2) #x''(t),y''(t)

	maxRadius,area = GE.getLsetProps(x1,y1,lset) #get maximum radius of an inscribed circle & particle area

	maxRadius = abs(np.amin(lset))
	radiusCurvatures = np.array([ abs(1./GE.calc_curvature(xd1,xd2,yd1,yd2,i)) for i in range(numOutputPoints - 1)])	
	radiusCurvaturesNormed = radiusCurvatures/maxRadius #Radius of curvatures normed by max radius
	#radiusCurvaturesNormed[radiusCurvaturesNormed > 1] = 1 #to make plotting nice
	cornerMask = [GE.isCorner(radiusCurvaturesNormed,i) for i in range(radiusCurvaturesNormed.shape[0])] #cornerMask[i] = 1 if point i is a corner
	#radiusCurvaturesCorners = radiusCurvaturesNormed[cornerMask]
	#roundnessCheck = np.mean(radiusCurvaturesCorners)
	#xc,yc= x1[:-1][cornerMask],y1[:-1][cornerMask]
	    
       # print (radiusCurvaturesNormed,radiusCurvaturesNormed[radiusCurvaturesNormed < 0.5],np.mean(radiusCurvaturesNormed[radiusCurvaturesNormed < 1]))
       # print (roundness,radiusCurvaturesCorners)
	print (radiusCurvaturesNormed.shape)
	plt.scatter(x1[:-1],y1[:-1],c=radiusCurvaturesNormed,s=10,vmin=0,vmax=1)

	plt.show()

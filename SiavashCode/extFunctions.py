import numpy as np
import os
import scipy
from scipy import interpolate 
from scipy.interpolate import splprep, splev

import matplotlib.path as mpltPath
from utilities import smoothHeaviside,sussman
import numpy.linalg as la
from scipy.spatial import Delaunay
import scipy.stats as stats
import matplotlib.pyplot as plt


epsilon = 0.00001

def getLset(X,Y): #get the level set for a set of points (from Kostas)
	pad = 2
	initTimesteps = 5


	pts = np.column_stack((X,Y))
	ptsCM = pts - np.mean(pts,axis=0)
	nPts = len(pts)
    # Create grid to evaluate level set
	xMin, yMin = np.min(ptsCM,axis=0)
	xMax, yMax = np.max(ptsCM,axis=0)
	cm = [-xMin+pad,-yMin+pad]
	pts = ptsCM + cm
	nX = int(np.ceil(xMax-xMin+2*pad))
	nY = int(np.ceil(yMax-yMin+2*pad))
	x = np.arange(0,nX)
	y = np.arange(0,nY)
	xx,yy = np.meshgrid(x,y)
    # Evaluate signed distance on the grid
	path = mpltPath.Path(pts)
	lset = np.zeros(xx.shape)
	for j in range(nY):
		for k in range(nX):
			xy = [xx[j,k],yy[j,k]]
			dist = la.norm(xy-pts,axis=1)
			idx = np.argmin(dist)
			lset[j,k] = dist[idx]
			inside = path.contains_points([xy])
			lset[j,k] *= -1 if inside else 1
    # Reinitialize level set
	for j in range(initTimesteps):
		lset = sussman(lset,0.1)

	return lset


def getLsetProps(x,y): #get the maximum insribed sphere in a particle & area
	tck, u = splprep([x,y], u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), 50)
	x_new, y_new = splev(u_new, tck, der=0) #First, interpolate more points
	lset = getLset(x_new,y_new)
	maxRad = abs(np.amin(lset))
	area   = np.count_nonzero(lset <= 0)
	return maxRad, area

#calculate curvature given radii list and index into radii list
def calc_curvature(x,y,last,next,i): 
	(x0,y0) = x[last],y[last]
	(x1,y1) = x[i],y[i]
	(x2,y2) = x[next],y[next]
	x = [x0,x1,x2]
	y = [y0,y1,y2]
	tck,uout = splprep([x,y],s=0.,per=False,k=2)
	xd1,yd1 = splev(uout,tck,der = 1) #x'(t),y'(t)
	xd2,yd2 = splev(uout,tck,der = 2) #x''(t),y''(t)
	x1_1,y1_1 = xd1[1],yd1[1] #x1'(p1),y1'(p1)
	x2_1,y2_1 = xd2[1],yd2[1] #x2'(p1),y2'(p1)
	curvature = (x1_1*y2_1 - y1_1*x2_1)/( (x1_1**2 + y1_1**2)**(3./2) )
	return curvature

#calculate spline length given radii list and index into radii list
def calc_perimeter(x,y): 
	nPoints = len(x)

	tck,u = splprep([x,y],s=0.,per=False,k=3)
	u_new = np.linspace(u.min(), u.max(), 100)
	xd1,yd1 = splev(u_new,tck,der = 1) #x'(t),y'(t)
	arcVals = np.sqrt(xd1**2 + yd1**2) #integrate along arc length to get perimeter
	du      = u_new[1] - u_new[0]
	perimeter = scipy.integrate.simps(y = arcVals, x = u_new, dx = du)

	return perimeter


#evaluate particle fitness
def getProps(particle,points):
	x,y = particle[:,0],particle[:,1]
	#plt.plot(x,y)
	#axes = plt.gca()
	#axes.set_xlim([-12,12])
	#axes.set_ylim([-12,12])

	#plt.show()
	tck, u = splprep([x,y], u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), points + 1) #Smooth it 
	x, y = splev(u_new, tck, der=0) #sample points for current length scale
	plt.plot(x,y)
	axes = plt.gca()
	axes.set_xlim([-12,12])
	axes.set_ylim([-12,12])
	#plt.show()

	Particle = np.column_stack((x,y))

	radiusCurvatures = np.zeros(points) #list of radii of curvatures 
	for i in range(points): #get curvature costs
		last = (i - 1)%points
		next = (i + 1)%points
		weight = 1  #element weight
		cval = calc_curvature(x, y, last, next, i) #current curvature at this point
		radiusCurvatures[i] = abs(1./cval)


	maxRadius,area = getLsetProps(x,y) #get maximum radius of an inscribed circle & particle area

	radiusCurvaturesNormed = abs(radiusCurvatures/maxRadius) #Radius of curvatures normed by max radius
	radiusCurvaturesNormedFiltered = [radiusCurvaturesNormed[i] for i in range(points) if radiusCurvaturesNormed[i] < 5] #radius of curvatures less than 1 - major surface features
	#print (radiusCurvaturesNormedFiltered)
	roundness = np.mean(radiusCurvaturesNormedFiltered)

	perimeter = calc_perimeter(x,y) 
	circularity = 2*np.sqrt(np.pi*area)/perimeter
	return (roundness,circularity)

def loadParticle(filename,num):
	particle = np.loadtxt(filename)
	roundness, circularity = getProps(particle,16)
	roundnessArray[num - startNum] = roundness
	

morphDir = "morphsDebug/"
startNum = 0
endNum = 10
roundnessArray = np.zeros(endNum - startNum)
for num in range(startNum,endNum):
	filename = morphDir + str(num) + ".dat"	
	loadParticle(filename,num)
print roundnessArray,np.std(roundnessArray)
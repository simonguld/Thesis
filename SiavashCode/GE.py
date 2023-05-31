#Use genetic algorithms to produce a clone in 2D
#to work at multiple length scales
#To run, import this file (GE.py) and call makeParticles(inputDir)
#inputDir should contain input file ("inputFile.dat"), which is of form:
#aspectRatio \n Mu_roundness \n Std_roundness \n Mu_circularity \n Std_circularity \n numParticles \n

import numpy as np 
import math
import random
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import interpolate 
from scipy.interpolate import splprep, splev
import copy
import pylab

from deap import tools
from deap import base 
from deap import creator 

import matplotlib.path as mpltPath
from utilities import smoothHeaviside,sussman
import numpy.linalg as la
from scipy.spatial import Delaunay
import scipy.stats as stats
from sklearn.decomposition import PCA
from utilities import smoothHeaviside,sussman
#Parameters chosen by user
Std_th = 0.02#0.1 #Standard deviation of theta mutation 
circularityLB = 0.1 #lower bound on circularity 
radBound     = 10 #Only record curvature at points with ROC below this 
Plot         = False #Plot grains

init_stepsize = 1 #initial step size
popSize = 50 #size of population 
MUTP = 0.5 #probability of mutation 
NGEN = 200#Number of generations
CXPB = 0.2 #cross-over probability
weight_r = 0 #weight on match radius, 0 < weight_r < 1
weight_c = 1 - weight_r
Iterations = 2
Subdivisions = 2
eps = 0.5 #for heaviside 
numOutputPoints = 100 #How many points to output

def cartToPol(xl,yl): #convert xl (x list) and yl (y list) to rl and Thetal 
	length = len(xl)
	rl,Thetal = np.zeros(length),np.zeros(length)
	for i in range(length):
		xv,yv = xl[i],yl[i]
		rl[i] = (xv**2 + yv**2)**0.5
		Thv = math.atan2(float(yv),xv)
		if (Thv < 0): Thv = 2*math.pi + Thv
		Thetal[i] = Thv
	return rl,Thetal

#convert polar coordinates (r[i],Theta[i]) to (x,y)
def polToCart(r,Theta,i):
	(rval,theta) = r[i],Theta[i]
	x = rval*math.cos(theta)
	y = rval*math.sin(theta)
	return x,y

#Make element list and flag list (list which states which vertices correspond to which length scales), and theta list, up to It (current iteration)
def makeLists(It):
	if (It):
		points_above = 8*(Subdivisions**(It - 1))
	else: points_above = 0

	points = 8*(Subdivisions**It)
	e_length = points
	element_list = np.ones((e_length,4))*-1
	e_pos = 0 #position in element list
	for i in range(points):
		weight = 1
		element_list[e_pos] = [i, (i -  1) % points, (i + 1) % points, weight]
		e_pos += 1

	return element_list


#calculate curvature given spline derivatives 
def calc_curvature(xd1,xd2,yd1,yd2,i): 
	x1_1,y1_1 = xd1[i],yd1[i] #x1'(p1),y1'(p1)
	x2_1,y2_1 = xd2[i],yd2[i] #x2'(p1),y2'(p1)
	curvature = (x1_1*y2_1 - y1_1*x2_1)/( (x1_1**2 + y1_1**2)**(3./2) )
	return curvature


#check if point i is a corner by comparing against points close to it 
def isCorner(RCs,i):
    cornerLength = int(numOutputPoints/20)
    indices = range(i - cornerLength,i + cornerLength)
    localRCs = RCs.take(indices,mode='wrap') #radius of curvature of points near to point i
    return ( (RCs[i] == min(localRCs)) and (RCs[i] < 1) ) 

def getHullArea(x,y):
		points = np.column_stack((x,y))
		hull = ConvexHull(points)
		return hull,hull.volume


"""
def mutatePart(particle,element_list,Std_r):
	points = len(particle[0])
	xv,yv = getCartVals(particle)
	xv,yv = xv[1:],yv[1:]
	for i in range(points):
		indices = range(i-1,i+2)
		xs,ys = xv.take(indices,mode='wrap'),yv.take(indices,mode='wrap')
		tck, u = splprep([xs,ys], s=0, per=False,k=2) #get spline fit to 3 points
		x, y = splev(u, tck, der=0)#points generated from smoothe spline 
		if (random.random() < MUTP): 
			#that,nhat = np.array(#tangent and normal vector
			rval = particle[0][i]
			Thetaval = particle[1][i]
			particle[0][i] = np.random.normal(rval,Std_r)
			particle[1][i] = np.random.normal(Thetaval, Std_th)
	return particle
"""

#mutation function for particle
def mutatePart(particle,element_list,Std_r):
	points = len(particle[0])
	for element in element_list:
		point = int(element[0])
		if (random.random() < MUTP): 
			rval = particle[0][point]
			Thetaval = particle[1][point]
			particle[0][point] = np.random.normal(rval,Std_r)
			particle[1][point] = np.random.normal(Thetaval, Std_th)
	return particle

#Calculate the first and last index of the element list for this level
def calc_num_e(It):
	if not (It):
		return 0,8
	else:
		e_start = 8*(Subdivisions**(It - 1))
		e_end = 8*(Subdivisions**(It))
		return e_start, e_end

#Get lset from particle 
def getLsetPolar(particle):
	x,y = getCartVals(particle)
	tck, u = splprep([x,y], u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), numOutputPoints)
	x_new, y_new = splev(u_new, tck, der=0)#points generated from smoothe spline 	
	#tck, u = splprep([x,y], u=None, s=0.0, per=1) 
	#u_new = np.linspace(u.min(), u.max(), 20)
	#x_new, y_new = splev(u_new, tck, der=0) #First, interpolate more points
	lset = getLset(x_new,y_new)
	return lset


def getLset(X,Y): #get the level set for a set of points (from Kostas)
	pad = 2
	initTimesteps = 5


	pts = np.column_stack((X,Y))
	ptsCM = pts - np.mean(pts,axis=0)
	nPts = len(pts)
    # Create grid to evaluate level set
	Min = np.min(np.min(ptsCM,axis=0))
	Max = np.max(np.max(ptsCM,axis=0))
	cm = [-Min+pad,-Min+pad]
	pts = ptsCM + cm
	nX = int(np.ceil(Max-Min+2*pad))
	nY = int(np.ceil(Max-Min+2*pad))
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


def getLsetProps(x,y,lset): #get the maximum insribed sphere in a particle & area from lset
	maxRad = abs(np.amin(lset))
	area   = smoothHeaviside(-lset, eps).ravel().sum()
	return maxRad, area

def getPA(lset):
	nX = lset.shape[1]
	nY = lset.shape[0]
	pad = 2
	rho = 1 
	x = np.arange(0,nX)
	y = np.arange(0,nY)
	# Compute mass (density=1, thickness=1)
	m = rho * smoothHeaviside(-lset, eps).ravel().sum()
	# Compute center of mass
	cx = rho / m * np.dot(smoothHeaviside(-lset, eps), x).sum()
	cy = rho / m * np.dot(smoothHeaviside(-lset, eps).T, y).sum()
	# Compute moment of inertia
	x_dev = np.power(x-cx,2)
	y_dev = np.power(y-cy,2)
	xy = np.outer(x - cx, y - cy)

	Ixx =  rho * np.dot(smoothHeaviside(-lset, eps).T, y_dev).sum() 
	Iyy = rho * np.dot(smoothHeaviside(-lset, eps), x_dev).sum()
	Ixy = rho * np.multiply(smoothHeaviside(-lset, eps),xy).sum()
	Izz = rho * np.dot(smoothHeaviside(-lset, eps), x_dev).sum() + rho * np.dot(smoothHeaviside(-lset, eps).T, y_dev).sum()
	
	I = np.array([[Ixx,Ixy],[Ixy,Iyy]])
	w, v = la.eig(I)
	return min(w)/max(w)

#evaluate particle fitness
def evalPart(particle, element_list, It, generation, printCost = 0):
	cost = 0
	length = len(element_list[:,0])
	particleNum = particle.particleNumber
	radiusCurvatures = np.zeros(length) #list of radii of curvatures 

	xv,yv = getCartVals(particle)
	tck, u = splprep([xv,yv], u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), numOutputPoints)
	x, y = splev(u_new, tck, der=0)#points generated from smoothe spline 

	tck, u = splprep([x,y], u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), numOutputPoints)
	x, y = splev(u_new, tck, der=0)#re fit and generate from smoothe spline (to match output)
 
	lset = getLset(x,y)

	xd1,yd1 = splev(u_new,tck,der = 1) #x'(t),y'(t)
	xd2,yd2 = splev(u_new,tck,der = 2) #x''(t),y''(t)

	#set = particle.lset
	maxRadius,area = getLsetProps(x,y,lset) #get maximum radius of an inscribed circle & particle area
	ar = getPA(lset) #get principal axes lengths 

	maxRadius = abs(np.amin(lset))
	radiusCurvatures = np.array([ abs(1./calc_curvature(xd1,xd2,yd1,yd2,i)) for i in range(numOutputPoints - 1)]) #ignore wrap around point	
	radiusCurvaturesNormed = abs(radiusCurvatures/maxRadius) #Radius of curvatures normed by max radius
	#radiusCurvaturesNormed[radiusCurvaturesNormed > 1] = 1 #to make plotting nice
	cornerMask = [isCorner(radiusCurvaturesNormed,i) for i in range(radiusCurvaturesNormed.shape[0])] #cornerMask[i] = 1 if point i is a corner
	radiusCurvaturesCorners = radiusCurvaturesNormed[cornerMask]
	roundness = np.mean(radiusCurvaturesCorners)

	#xc,yc = x[cornerMask],y[cornerMask]
	#centroid = np.column_stack((x,y)).mean(axis=0)
	#maxEnclosingRadii = np.array([np.sqrt((x[i] - centroid[0])**2 + (y[i] - centroid[1])**2) for i in np.arange(len(x))]) 
	#print ("max",np.max(maxEnclosingRadii),maxRadius)
	circularity = ar #(2*np.sqrt(np.pi*area)/(perimeter))**2
 
	particle.roundness = roundness 	
	particle.RCs = radiusCurvaturesCorners

	if (It > 0): #update cost function based on iteration  
		cost += ( (circularity - circularityTarget)/circularity)**2 
		cost +=  ((roundness - roundnessTarget)/roundness)**2
		cost += len(radiusCurvaturesCorners[radiusCurvaturesCorners < 0.05])*100 #penalize large curvatures


	#if (generation % 100 == 0): 
		#print (radiusCurvaturesNormed,roundness)
	#	print (generation,circularity,circularityTarget,roundness,roundnessTarget,cost,len(radiusCurvaturesCorners) )
	if (printCost): 
		print ("roundness in cost",np.nonzero(cornerMask),radiusCurvatures[np.nonzero(cornerMask)[0]],maxRadius,roundness)
	
	#print particle.lset
	return (cost,)

def getCartVals(part): #Get list of cartesian values in order given from r,theta
	length = len(part[0])
	r, Theta = part[0], part[1]
	x = np.zeros(length + 1)
	y = np.zeros(length + 1)
	for i in range(length):
		xv,yv = polToCart(r,Theta,i)
		x[i],y[i] = xv,yv
	x[length], y[length] = x[0],y[0]
	return x,y

def plot(part, It,count): #plot a 2d particle from list of r values)
	plt.figure()
	x,y = getCartVals(part)
	tck, u = splprep([x,y], u=None, s=0.0, per=1, k = 3) 
	u_new = np.linspace(u.min(), u.max(), 10000)
	x_new, y_new = splev(u_new, tck, der=0)
	plt.plot(x_new, y_new, label = "1st Iteration")

	for i in range(len(x) - 1):
		plt.text(x[i], y[i], str(part[2][i]), fontsize=12)


	plt.axes().set_aspect('equal')
	plt.show()
	plt.close()

#output points into file
def outputPoints(part,particleNum, morphDir, element_list):
	x,y = getCartVals(part)
	tck, u = splprep([x,y], u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), numOutputPoints)
	xnew, ynew = splev(u_new, tck, der=0)#points generated from spine
	ptsOut = np.column_stack((xnew,ynew))
	outName = morphDir + str(particleNum) + ".dat"
	print ("output eval",evalPart(part,element_list,2,0,printCost=1))
	np.savetxt(fname = outName, X =  ptsOut, fmt="%.16f")

	"""JUST FOR DEBUGGING"""

	tck, u = splprep([ptsOut[:,0],ptsOut[:,1]], u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), numOutputPoints)
	x, y = splev(u_new, tck, der=0)#points generated from smoothe spline 
 
	xd1,yd1 = splev(u_new,tck,der = 1) #x'(t),y'(t)
	xd2,yd2 = splev(u_new,tck,der = 2) #x''(t),y''(t)

	lset = getLset(x,y) #part.lset
	maxRadius,area = getLsetProps(x,y,lset) #get maximum radius of an inscribed circle & particle area

	ar = getPA(lset) #get principal axes lengths 

	maxRadius = abs(np.amin(lset))
	radiusCurvatures = np.array([ abs(1./calc_curvature(xd1,xd2,yd1,yd2,i)) for i in range(numOutputPoints - 1)])	
	radiusCurvaturesNormed = abs(radiusCurvatures/maxRadius) #Radius of curvatures normed by max radius
	#radiusCurvaturesNormed[radiusCurvaturesNormed > 1] = 1 #to make plotting nice
	cornerMask = [isCorner(radiusCurvaturesNormed,i) for i in range(radiusCurvaturesNormed.shape[0])] #cornerMask[i] = 1 if point i is a corner
	radiusCurvaturesCorners = radiusCurvaturesNormed[cornerMask]
	roundness = np.mean(radiusCurvaturesCorners)
	print ("roundness check",np.nonzero(cornerMask),radiusCurvatures[np.nonzero(cornerMask)[0]],maxRadius,roundness)
	#print ("x is",ptsOut[:,0])
	#print ("y is",ptsOut[:,1])

#re-initialize population after each iteration
def re_init_pop(pop,It,newPart):
	points = 8*(Subdivisions ** (It + 1)) #number of points in new iteration
	for partnum in range(popSize): #iterate through population
		part = copy.deepcopy(newPart)
		x,y = getCartVals(part)
		tck, u = splprep([x,y], u=None, s=0.0, per = 1) #fit spline to current particle
		u_new = np.linspace(u.min(), u.max(), points, endpoint = False) 
		x_new, y_new = splev(u_new, tck, der = 0) 
		r, Theta = cartToPol(x_new, y_new)
		pop[partnum][0] = r
		pop[partnum][1] = Theta

	return pop

#check that ind doesn't have any non-physical (high curvature) features at a lower length scale 
def isPhysical(ind,element_list): 
	evalPart(ind, element_list, 2, 3)
	RCs = ind.RCs
	return (np.amin(RCs) > 0.05)

#initialize genetic algorithm
def make_GE():
	global toolbox
	#initialize DEAP genetic algorithm population
	points = 8
	creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
	creator.create("Particle", list, fitness = creator.FitnessMin, stepsize = init_stepsize)
	toolbox = base.Toolbox()
	toolbox.register("rand_float", random.randint,1,1)
	toolbox.register("pointList", tools.initRepeat, list, toolbox.rand_float, points)
	toolbox.register("particle", tools.initRepeat, creator.Particle, toolbox.pointList, 2)
	toolbox.register("population", tools.initRepeat, list, toolbox.particle, popSize)
	toolbox.register("evaluate",evalPart)
	toolbox.register("mate",tools.cxTwoPoint)
	toolbox.register("select", tools.selTournament, tournsize = 2)

#main program - calls all others
def clone(particleNum, aspectRatio, Mu_roundness, Std_roundness, Mu_circularity, Std_circularity, morphDir, Std_r):
	global roundnessTarget
	global circularityTarget

	make_GE()
	pop = toolbox.population() #initialize population
	Area = 300 
	Min_prin = np.sqrt(Area/(np.pi*aspectRatio))
	Max_prin = Min_prin*aspectRatio
	
	for part in pop: #Some extra initialization not handled by DEAP framework
		Theta = np.linspace(0,2*math.pi,8, endpoint = False)
		r = (Min_prin*Max_prin)/np.sqrt( (Min_prin*np.cos(Theta))**2 + (Max_prin*np.sin(Theta))**2 )
		part[0] = r
		part[1] = Theta
		part.particleNumber = particleNum
		part.lset = getLsetPolar(part)

	
	count = 0
	element_list = makeLists(0)
	for It in range(Iterations):
		

		roundnessTarget = np.random.normal(Mu_roundness,Std_roundness)  #get random target curvature
		circularityTarget = getPA(pop[0].lset)
	#	circularityTarget = stats.truncnorm.rvs((circularityLB - Mu_circularity) / Std_circularity, (1 - Mu_circularity) / Std_circularity, loc=Mu_circularity, scale=Std_circularity, size = 1)[0]
		
		fitnesses = [toolbox.evaluate(pop[i],element_list,It,0) for i in range(len(pop)) ] #get fitnesses of each individual in population
		for ind,fit in zip(pop,fitnesses): #assign fitness values to individuals in population
			ind.fitness.values = fit

		for g in range(NGEN): #begin evolutionary process through generations
			#print("-- Generation %i --" % g)
			#Select next generation of individuals through tournament 
			offspring = toolbox.select(pop,popSize)

			#Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))

			#Apply crossover and mutation on the offspring 

			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				if random.random() < CXPB: 
					child1, child2 = toolbox.mate(child1, child2)
					del child1.fitness.values 
					del child2.fitness.values



			#Now randomly mutate

			for mutant in offspring:
				if (random.random() < MUTP):
					mutant = mutatePart(mutant,element_list,Std_r)
					del mutant.fitness.values

			#calculate the fitness of any offsprings w/ invalid fitness
			invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = [toolbox.evaluate(invalid_ind,element_list,It,g) for invalid_ind in invalid_inds]
			for ind,fit in zip(invalid_inds, fitnesses):
				ind.fitness.values = fit


			#Set the population to be equal to the new offspring, then repeat!
			pop[:]  = offspring
			fitnesses = [toolbox.evaluate(pop[i],element_list,It,g) for i in range(len(pop)) ] 
			count += 1
			
			if ( (np.amin(fitnesses) < 10**-3 and It == 1) or (np.amin(fitnesses) < 10**-6 and It == 0) ): break  
			if (g == (NGEN - 1)): 
				return 0	

		bestInd = pop[np.argmin(fitnesses)]
		if (It == Iterations - 1): #If final iteration, save particle 
			if (not isPhysical(bestInd,element_list)): return 0 
			outputPoints(bestInd, particleNum, morphDir, element_list)
			return 1

		if (Plot): plot(pop[0], It, count) #set up next length scale 
		bestInd = pop[np.argmin(fitnesses)]
		pop = re_init_pop(pop,It,bestInd)
		element_list = makeLists(It + 1)

	count += 1
	return 1 

#program initialize parameters and genetic algorithm
def makeParticles(Rve,particleNum):
	#print (Rve.morphDir)

	(aspectRatio, Mu_roundness, Std_roundness, Mu_circularity, Std_circularity, numParticles, morphDir) = \
		Rve.aspectRatio,Rve.Mu_roundness,Rve.Std_roundness,\
		Rve.Mu_circularity,Rve.Std_circularity,int(Rve.nShapes), Rve.morphDir

	attempt = 0
	Std_r = 0.2 #starting mutation standard deviation on radii
	while (not clone(particleNum, aspectRatio, Mu_roundness, Std_roundness, Mu_circularity, Std_circularity, morphDir, Std_r)):
		attempt += 1
		Std_r/=2.
		if (attempt == 4): return 0
	return 1



import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import grain
import GE
from scipy.interpolate import splprep, splev
import os
import copy
import seaborn
import pathlib




#roundnessList = np.arange(0.3,0.95,0.05)[::-1]
#aspectRatioList = np.arange(0.3,1.0,0.1)[::-1]
roundnessList = np.array([0.3,0.4,0.6,0.7,0.8])#np.arange(0.3,0.6,0.05)[::-1]
aspectRatioList = np.array([0.3,0.5,0.8,0.95]) #np.arange(0.1,0.2,0.1)[::-1]
N = 900 #number of particles 

numOutputPoints = 100

print (roundnessList)
def subplots():
	fig, axs = plt.subplots(aspectRatioList.shape[0],roundnessList.shape[0], squeeze = False,figsize=(50,100))

	for j,roundness in enumerate(roundnessList):
		for i,aspectRatio in enumerate(aspectRatioList):
			morphDirName = "shapeGen3/morphR_%.2f_A_%.2f/"%(roundness,aspectRatio)
			print (morphDirName)
			points = np.loadtxt(morphDirName + "0.dat")
			x1,y1 = points[:,0],points[:,1]
			lset = GE.getLset(x1,y1)

			tck, u = splprep([x1,y1], u=None, s=0.0, per=1) 
			u_new = np.linspace(u.min(), u.max(), numOutputPoints)
			x, y = splev(u_new, tck, der=0)#points generated from smoothe spline 
			axs[i,j].plot(x,y)
		    
			xd1,yd1 = splev(u_new,tck,der = 1) #x'(t),y'(t)
			xd2,yd2 = splev(u_new,tck,der = 2) #x''(t),y''(t)

			maxRadius,area = GE.getLsetProps(x1,y1,lset) #get maximum radius of an inscribed circle & particle area

			maxRadius = abs(np.amin(lset))
			radiusCurvatures = np.array([ abs(1./GE.calc_curvature(xd1,xd2,yd1,yd2,i)) for i in range(numOutputPoints - 1)])	
			radiusCurvaturesNormed = abs(radiusCurvatures/maxRadius) #Radius of curvatures normed by max radius
			#radiusCurvaturesNormed[radiusCurvaturesNormed > 1] = 1 #to make plotting nice
			cornerMask = [GE.isCorner(radiusCurvaturesNormed,i) for i in range(radiusCurvaturesNormed.shape[0])] #cornerMask[i] = 1 if point i is a corner
			radiusCurvaturesCorners = radiusCurvaturesNormed[cornerMask]
			roundnessCheck = np.mean(radiusCurvaturesCorners)
			print ("roundness check",maxRadius,roundnessCheck,roundness)
			xc,yc= x[:-1][cornerMask],y[:-1][cornerMask]
			    
		       # print (radiusCurvaturesNormed,radiusCurvaturesNormed[radiusCurvaturesNormed < 0.5],np.mean(radiusCurvaturesNormed[radiusCurvaturesNormed < 1]))
		       # print (roundness,radiusCurvaturesCorners)
			axs[i,j].set_aspect('equal')
			h = axs[i,j].scatter(xc,yc,c=radiusCurvaturesCorners,s=100,vmin=0,vmax=1)

			if (i == axs.shape[0] - 1): 
			    axs[i,j].set_xlabel(str("%.2f"%roundness)  )
			    axs[i,j].set_xticks([])
			else: 
			    axs[i,j].xaxis.set_visible(False)

			if (j == 0): 
			    axs[i,j].set_ylabel(str("%.2f"%aspectRatio)  )
			    axs[i,j].set_yticks([])
			else: 
			    axs[i,j].yaxis.set_visible(False)


	seaborn.despine(left=True, bottom=True, right=True)
	cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.8])
	cbar_ax.axis('off')
	cbar = fig.colorbar(h,ax=cbar_ax)
	cbar.set_label('Normalized Radius of Curvature', rotation=90)
	cbar.set_ticks(np.arange(0, 1.1, 0.5))
	#plt.tight_layout()
	fig.text(0.5, 0.04, 'Roundness', ha='center')
	fig.text(0.04, 0.5, 'Aspect Ratio', va='center', rotation='vertical')
	plt.show()

def props():
	fig, axs = plt.subplots(aspectRatioList.shape[0],roundnessList.shape[0], squeeze = False,figsize=(50,100))
	for j,roundness in enumerate(roundnessList):
		for i,aspectRatio in enumerate(aspectRatioList):
			roundnesses = []
			for n in range(N):
				morphDirName = "shapeGen3/morphR_%.2f_A_%.2f/"%(roundness,aspectRatio)
				points = np.loadtxt(morphDirName + str(n) + ".dat")
				x1,y1 = points[:,0],points[:,1]
				lset = GE.getLset(x1,y1)

				tck, u = splprep([x1,y1], u=None, s=0.0, per=1) 
				u_new = np.linspace(u.min(), u.max(), numOutputPoints)
				x, y = splev(u_new, tck, der=0)#points generated from smoothe spline 

				xd1,yd1 = splev(u_new,tck,der = 1) #x'(t),y'(t)
				xd2,yd2 = splev(u_new,tck,der = 2) #x''(t),y''(t)

				maxRadius,area = GE.getLsetProps(x1,y1,lset) #get maximum radius of an inscribed circle & particle area

				maxRadius = abs(np.amin(lset))
				radiusCurvatures = np.array([ abs(1./GE.calc_curvature(xd1,xd2,yd1,yd2,i)) for i in range(numOutputPoints - 1)])	
				radiusCurvaturesNormed = abs(radiusCurvatures/maxRadius) #Radius of curvatures normed by max radius
				#radiusCurvaturesNormed[radiusCurvaturesNormed > 1] = 1 #to make plotting nice
				cornerMask = [GE.isCorner(radiusCurvaturesNormed,i) for i in range(radiusCurvaturesNormed.shape[0])] #cornerMask[i] = 1 if point i is a corner
				radiusCurvaturesCorners = radiusCurvaturesNormed[cornerMask]
				roundnessCheck = np.mean(radiusCurvaturesCorners)
				#if (abs(roundnessCheck - roundness > 0.05)): 
				print ("roundness check",roundnessCheck,roundness,n)
				
				roundnesses += [roundnessCheck]
			axs[i,j].hist(roundnesses,range=(roundness - 0.1,roundness + 0.1))
			axs[i,j].set_xlabel("R : %.2f, A : %f"%(roundness,aspectRatio))

	plt.show()

def allPlots():
	for j,roundness in enumerate(roundnessList):
		for i,aspectRatio in enumerate(aspectRatioList):
			for n in range(N):
				morphDirName = "shapeGen3/morphR_%.2f_A_%.2f/"%(roundness,aspectRatio)
				points = np.loadtxt(morphDirName + str(n) + ".dat")
				fig1 = plt.figure()
				ax1 = fig1.add_subplot(111)
				ax1.set_aspect('equal')	
			
				plt.plot(points[:,0],points[:,1],'ro')
				dirName = morphDirName + "particle_plots/"
				if not os.path.exists(dirName):
					pathlib.Path("./" + dirName).mkdir(parents=True, exist_ok=True)
				plt.savefig(dirName + str(n))
				print (dirName)
				plt.close()	
subplots()
#allPlots()
#props()



#Robert Buarque de Macedo
#2020

import helper
import plotConfiguration
import analyzeSim
import computePackingFraction
from rve import rve

import os #python libraries 
import numpy as np
import sys
from grain import Grain
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy.ma as ma  
import pickle
import re
import natsort

import numpy as np
from scipy.optimize import minimize

def calculate_perimeter(a,b):
	perimeter = np.pi * ( 3*(a+b) - np.sqrt( (3*a + b) * (a + 3*b) ) )
	return perimeter

def ellipseCost(AR):
	a,b = AR,1 
	perim = calculate_perimeter(a,b)
	area = np.pi * a * b
	circularityCurr = (2*np.sqrt(np.pi*area)/(perim))**2
	return (circularityCurr - circularity)**2 

roundnessList = np.array([0.3,0.4,0.6,0.7,0.8])#np.arange(0.3,0.6,0.05)[::-1]
aspectRatioList = np.array([0.3,0.5,0.8,0.95]) #np.arange(0.1,0.2,0.1)[::-1]

count = 0 
for i,roundness in enumerate(roundnessList):
	for j,aspectRatio in enumerate(aspectRatioList):
		count += 1 
		#if (count < 97): continue  	
		print (count, roundness, aspectRatio)
		#roundness = 0.7
		#aspectRatio = 0.3#1/res.x
		circularity = 0.1

		#res = minimize(ellipseCost, 1.0, method='nelder-mead',
		#               options={'xatol': 1e-8, 'disp': True})

		print ("AR",aspectRatio)
		attempt = 0 
		mainDirName = "shapeGen3/morphR_%.2f_A_%.2f/"%(roundness,aspectRatio) 
		morphDirName = mainDirName
		Rve = rve(morphDirName,mainDirName) #initialize RVE instance with default variables 
				
		#Parameters for cloning 
		Rve.aspectRatio = aspectRatio # max axis/min axis of starting ellipse
		Rve.Mu_roundness = roundness #Average of roundness targets
		Rve.Std_roundness = 0.001 #standard deviation of roundness targets
		Rve.Mu_circularity = circularity #same, but for circularity
		Rve.Std_circularity = 0.001
		Rve.trial = attempt
		Rve.nGrains = 1
		Rve.nShapes = 1
		check  = Rve.createParticles()
 

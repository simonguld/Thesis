from massPy import archive 
import massPy as mp
import massPy.base_modules.anisotropic_correlation_new as ac
#import massPy.base_modules.anisotropic_correlation_jc as ac
import numpy as np
import matplotlib.pyplot as plt
import sys

var=sys.argv[1]

#path = '/lustre/astro/rsx187/mmout/active_sample/qzk1k30.05_K30.05_qkbt0_z'+str(var)+'_xi1_LX1024_counter0/'
#path = '/lustre/astro/rsx187/mmout/active_sample/'

#path = '/groups/astro/jayeeta/files/jc/pos_data/data'+str(var)
#path = '/groups/astro/jayeeta/varun/massPy/non-eq-fluc/input-files/ukbtqkbt'+str(var)
#path = '/lustre/astro/rsx187/mmout/theta_sample/qzk1k30.05_K30.05_qkbt'+str(var)+'_z0_xi1_LX256_counter0'
path = '/groups/astro/jayeeta/cluster_data_new/aniso-corr/newrun/2D_jc/test4.1-out'

f1 = open('nem_corr_z'+str(var)+'_r256_256_t130-180.dat','w')
#f1 = open('/groups/astro/jayeeta/cluster_data_new/aniso-corr/calculations/active/results/size_1024/r400_time0-30/nem_corr_z'+str(var)+'_r400_1024.dat','w')

ar = mp.archive.loadarchive(path)

# +
nstart = 130
nlast = 180
r1 = 256
#r2 = 50

cpar_t = []
cperp_t = []

for i in range (nstart,nlast,1):
	#i =10
	frame = ar._read_frame(i)
	#step=2
	LX, LY = frame.LX, frame.LY

	Qxx_dat = frame.QQxx.reshape(LX, LY)
	Qyx_dat = frame.QQyx.reshape(LX, LY)

	qxx, qxy, qyy = Qxx_dat, Qyx_dat, -Qxx_dat

	s = mp.base_modules.flow.density(frame.ff)
	s_dat = s.reshape(LX,LY)
	#print(len(s))
	#cpar,cperp = ac.aniso_corr(s_dat, Qxx_dat, Qyx_dat)

	#r1 = 100 #len(s_dat)
	#r2 = 30

	qxx1 = qxx[0:r1, 0:r1]
	qyy1 = qyy[0:r1, 0:r1]
	qxy1 = qxy[0:r1, 0:r1]

	s1 = s_dat[0:r1, 0:r1]

	nem = np.sqrt(qxx1**2 + qxy1**2)

	cpar,cperp = ac.aniso_corr(nem, qxx1, qxy1)
	#print (cpar)	
	#print (cperp)	
	cpar_t.append(cpar)
	cperp_t.append(cperp)

cpar_t = np.array(cpar_t)
cperp_t = np.array(cperp_t)

#print (cpar_t)
#print (cpar_t[0:,1])
#print (np.average(cpar_t[0:,1]))

#cpar_t_avg = []
#cperp_t_avg = []

for j in range (len(cpar)):
	#cpar_t_avg.append(np.average(cpar_t[0:,j]))
	#cperp_t_avg.append(np.average(cperp_t[0:,j]))
	
	cpar_t_avg = np.average(cpar_t[0:,j])
	cperp_t_avg = np.average(cperp_t[0:,j])
	#print >> f, j, cpar_t_avg, cperp_t_avg
	print (j, cpar_t_avg, cperp_t_avg, file = f1)

f1.close()
#print (cpar_t_avg)
#print (cperp_t_avg)

#z_line = np.zeros(len(cpar))
#plt.plot(cpar_t_avg[:r2],'o', label = "parallel correlation")
#plt.plot(cperp_t_avg[:r2],'o', label = "Perp correlation")
#plt.plot(z_line[:r2], 'black')
#plt.legend()
#plt.savefig('corr_0.018_nem_tavrg.png',dpi=300)

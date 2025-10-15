#plot packing fraction vs. t
import numpy as np
import matplotlib.pyplot as plt
As = [5.0,16.0] #[0.0001,0.0002,0.0005] #[0.005,0.01,0.02,0.1] #list of amplitudes to check 
Fs = np.array([0.1,1.0,25.])#[1000,5000,10000]#[500,1000,1500,2000] #[500,1000,1500,2000] #list of time periods to check 
trials = 5
dt = 0.000005
ramps = [0,5./(50000*dt),16./(50000*dt)]

fig, axs = plt.subplots(6,3,figsize=(15, 6), facecolor='w', edgecolor='k')
for ramp in range(0,3):
    count = 0 
    for A in As:
        for F in Fs:
            maxPF,minPF = 0,1
            for trial in np.arange(trials):
                PFs = np.loadtxt("shakeSweep/plots/ramps_%d/A_%f_F_%f_trial_%d/allSteps"% (ramp,A,F,trial) )
                PFs = PFs[PFs > 0]
                axs[count,ramp].plot(np.arange(len(PFs)),PFs,label="A: %f F: %f trial: %d"%(A,F,trial))
                    #plt.show()
                maxPF = max(max(PFs),maxPF)
                minPF = min(min(PFs),minPF) 



            #axs[count].xlabel("Time station")
            #axs[count].ylabel("Packing Fraction")
            axs[count,ramp].set_title("A: %f pix F: %f Hz ramp: %f pix/sec"%(A,F,ramps[ramp]),fontsize=10)
            axs[count,ramp].set_ylim(0.8,0.9)#(minPF,maxPF)#max(PFs))
            if (count != 5): axs[count,ramp].set_xticks([])
            count += 1
            print (ramp,A,F,count)
        #plt.savefig("PFPics/ramp_%d/A_%f_F_%f.png"%(ramp,A,F))
        #axs[count].close()
plt.tight_layout()
plt.savefig("ramps.png",dpi=1000)





#plot packing fraction vs. t
import numpy as np
import matplotlib.pyplot as plt
As = [5.0,16.0] #[0.0001,0.0002,0.0005] #[0.005,0.01,0.02,0.1] #list of amplitudes to check 
Fs = np.array([0.1,1.0,25.])#[1000,5000,10000]#[500,1000,1500,2000] #[500,1000,1500,2000] #list of time periods to check 
count = 1
trials = 5

for A in As:
    for F in Fs:
        maxPF,minPF = 0,1
        for trial in np.arange(trials):
            try: 
                PFs = np.loadtxt("shakeSweep/plots/ramps_2/A_%f_F_%f_trial_%d/allSteps"% (A,F,trial) )
                PFs = PFs[PFs > 0]
                plt.plot(np.arange(len(PFs)),PFs,label="A: %f F: %f trial: %d"%(A,F,trial))
                #plt.show()
                count += 1 
                print (A,F)
                maxPF = max(max(PFs),maxPF)
                minPF = min(min(PFs),minPF) 
                print ("here")
            except:
                pass

        plt.xlabel("Time station")
        plt.ylabel("Packing Fraction")
        plt.title("A: %f F: %f "%(A,F))
        plt.ylim(0.8,0.9)#(minPF,maxPF)#max(PFs))
        plt.savefig("PFPics/ramp_2/A_%f_F_%f.png"%(A,F))
        plt.close()





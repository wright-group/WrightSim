from _transients import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time


# experiment setup and variable definitions

## delays
d1 = 700e-15
d2 = 300e-15

## timing specifications
pw = [280e-15, 180e-15, 600e-15]
t0 = 0
t1 = t0+pw[0]
t2 = t1+d1
t3 = t2+pw[1]
t4 = t3+d2
t5 = t4+pw[2]
times = [t0,t1,t2,t3,t4,t5]

## transition frequencies
wg = 0
w1 = 1675
w2 = 1675+1615
w3 = 9670
omegas = [wg,w1,w2,w3]

## local oscillator frequency factors
wgLO = 1
w1LO = 1.07
w2LO = 1.07
w3LO = 1.01
omega_LOs = [wgLO,w1LO,w2LO,w3LO]

## gammas
gammag = 0 
gamma1 = 2.846
gamma2 = 4.39
gamma3 = 1e12
gammas = [gammag,gamma1,gamma2,gamma3]

## efields
efield1 = 42.9
efield2 = 3.16
efields = [efield1, efield2]

## useful equations
hsthresh = 0.5
E_pulses = lambda t: 0.9*((np.heaviside(t-t0*1e12, hsthresh)-(np.heaviside(t-t1*1e12, hsthresh))) 
                    + (np.heaviside(t-t2*1e12, hsthresh)-(np.heaviside(t-t3*1e12, hsthresh)))
                    + (np.heaviside(t-t4*1e12, hsthresh)-(np.heaviside(t-t5*1e12, hsthresh))))
                    
t_i = np.linspace(0.01, 16000, num=16000)*0.0006e-12

# outputs?
for i,t in enumerate(t_i):
    if t>t4:
        out_i = i
        break
w1_scanrange = np.linspace(1550, 1750, 10)
w2_scanrange = np.linspace(1550, 1750, 10)
simulation_data=np.zeros([len(w2_scanrange),len(w1_scanrange)])

remaining = len(w1_scanrange)*len(w2_scanrange)
for i, w1 in enumerate(w1_scanrange):
    for j, w2 in enumerate(w2_scanrange):
        remaining-=1
        start = time.time()
        ## transition frequencies
        wg = 0
        w3 = 9670
        omegas = [wg,w1,w1+w2,w3]

        # experiment transient outputs
        a = DoveTransient(t=t_i,omegas=omegas, omega_LOs=omega_LOs, gammas=gammas, pulse_timings=times, efields=efields)
        x = a.rho1(renorm=True)                  
        b = DoveTransient(t=t_i,omegas=omegas, omega_LOs=omega_LOs, gammas=gammas, pulse_timings=times, efields=efields)
        y = b.rho2(renorm=True, weight=0.65)     
        c = DoveTransient(t=t_i,omegas=omegas, omega_LOs=omega_LOs, gammas=gammas, pulse_timings=times, efields=efields)
        z = c.rho3(renorm=True, weight=0.47)

        
        t_i_out = t_i[out_i:]
        outcoherence = [out**2 for out in x[out_i:]-y[out_i:]+z[out_i]]
        simulation_data[i,j]=sum(outcoherence)
        end=time.time()

        print(f'{(end-start)*remaining/60} mins remaining', remaining)

plt.imshow(simulation_data, interpolation="nearest", origin="lower")
plt.colorbar()
plt.show()













# plotting
'''
x_axis = t_i*1e12


fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2,2)
trans_noPM = fig.add_subplot(gs[0,0])
trans_PM = fig.add_subplot(gs[0,1])
trans_out_noPM = fig.add_subplot(gs[1,0])
trans_out_PM = fig.add_subplot(gs[1,1])

trans_noPM.plot(x_axis, x)
trans_noPM.plot(x_axis, y)
trans_noPM.plot(x_axis, z, c='red')
trans_noPM.plot(x_axis, [E_pulses(t) for t in x_axis], c='k')
trans_noPM.set_title(r'$\omega_{1}+\omega_{2}+\omega_{3}$')
trans_noPM.set_xlabel('time (ps)')
trans_noPM.set_ylabel('Amplitude')


trans_PM.plot(x_axis, x)
trans_PM.plot(x_axis, -y)
trans_PM.plot(x_axis, z, c='red')
trans_PM.plot(x_axis, [E_pulses(t) for t in x_axis], c='k')
trans_PM.set_title(r'$\omega_{1}-\omega_{2}+\omega_{3}$')
trans_PM.set_xlabel('time (ps)')
trans_PM.set_ylabel('Amplitude')

trans_out_noPM.plot(x_axis, (x+y+z)**2, c='red')
trans_out_noPM.plot(x_axis, [E_pulses(t) for t in x_axis], c='k')
trans_out_noPM.set_xlabel('time (ps)')
trans_out_noPM.set_ylabel('Amplitude')

trans_out_PM.plot(x_axis, (x-y+z)**2, c='red')
trans_out_PM.plot(x_axis, [E_pulses(t) for t in x_axis], c='k')
trans_out_PM.set_xlabel('time (ps)')
trans_out_PM.set_ylabel('Amplitude')

plt.show()
'''


"""
8/9/2022

TODOs:
1) potentially fix rho3 as it should extend longer 


Notes, try a 2D plot of w2-w1 vs w1 where the dependent value is the sum of the intensity 
from t5 to some number of ps where the intensity goes to zero (maybe x_axis[t5:], abstractly?). 

"""
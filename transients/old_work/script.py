from _transientsv2 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider, Button, TextBox
import asyncio


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
w1 = 1620
w2 = 1620+1655
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


# experiment transient outputs
a = DoveTransient(t=t_i,omegas=omegas, omega_LOs=omega_LOs, gammas=gammas, pulse_timings=times, efields=efields)
x = asyncio.run(a.rho1(1620,renorm=True))                  
b = DoveTransient(t=t_i,omegas=omegas, omega_LOs=omega_LOs, gammas=gammas, pulse_timings=times, efields=efields)
y = asyncio.run(b.rho2(renorm=True, weight=0.65))     
c = DoveTransient(t=t_i,omegas=omegas, omega_LOs=omega_LOs, gammas=gammas, pulse_timings=times, efields=efields)
z = asyncio.run(c.rho3(renorm=True, weight=0.47))


# plotting
fig = plt.figure() #fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(8,2)
trans = fig.add_subplot(gs[0:7,0])
intensity_plus = fig.add_subplot(gs[0:3,1])
intensity_minus = fig.add_subplot(gs[4:7,1])
plt.subplots_adjust(bottom=0.20)

x_axis = t_i*1e12

trans_w1 = trans.plot(x_axis, x, linewidth=2.5, alpha=0.7)
#trans_w2 = trans.plot(x_axis, y, linewidth=2.5, alpha=0.7)
#trans_w3 = trans.plot(x_axis, z, linewidth=2.5, alpha=0.7, c='red')
trans_efields = trans.plot(x_axis, [E_pulses(t) for t in x_axis], c='k')
trans.set_title(r'$\omega_{1},\omega_{2},\omega_{3}$')
trans.set_xlabel('time (ps)')
trans.set_ylabel('Amplitude')

int_plus = intensity_plus.plot(x_axis, (x+y)**2)
int_plus_efields = intensity_plus.plot(x_axis, [E_pulses(t) for t in x_axis], c='k')
intensity_plus.set_title(r'$(\omega_{1}+\omega_{2}+\omega_{3})^2$')
intensity_plus.set_xlabel('time (ps)')
intensity_plus.set_ylabel('Intensity')

int_minus = intensity_minus.plot(x_axis, (x-y)**2)
int_minus_efields = intensity_minus.plot(x_axis, [E_pulses(t) for t in x_axis], c='k')
intensity_minus.set_title(r'$(\omega_{1}-\omega_{2}+\omega_{3})^2$')
intensity_minus.set_xlabel('time (ps)')
intensity_minus.set_ylabel('Intensity')


# plot sliders
axd1_slider = plt.axes([0.1, 0.05 , 0.6, 0.03])
axd2_slider = plt.axes([0.1, 0.1, 0.6, 0.03])
axw1_slider = plt.axes([0.1, 0.15, 0.6, 0.03])
axw2_slider = plt.axes([0.1, 0.2, 0.6, 0.03])

d1_slider = Slider(axd1_slider, 'd1', 1e-15, 2000e-15, 700e-15)
d2_slider = Slider(axd2_slider, 'd2', 1e-15, 2000e-15, 700e-15)
w1_slider = Slider(axw1_slider, 'w1', 0.0, 2000, 1620)
w2_slider = Slider(axw2_slider, 'w2', 0.0, 4000, 3240)

# plot textboxes
axd1_tb = plt.axes([0.8, 0.05 , 0.05, 0.03])
axd2_tb = plt.axes([0.8, 0.1, 0.05, 0.03])
axw1_tb = plt.axes([0.8, 0.15, 0.05, 0.03])
axw2_tb = plt.axes([0.8, 0.2, 0.05, 0.03])

d1_tb = TextBox(axd1_tb, '', '')
d2_tb = TextBox(axd2_tb, '', '')
w1_tb = TextBox(axw1_tb, '', '')
w2_tb = TextBox(axw2_tb, '', '')

async def update(val):
    t1 = time.time()
    ## delays
    d1 = d1_slider.val
    d2 = d2_slider.val

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
    w1 = w1_slider.val
    w2 = w2_slider.val
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

    # experiment transient outputs
    a = DoveTransient(t=t_i,omegas=omegas, omega_LOs=omega_LOs, gammas=gammas, pulse_timings=times, efields=efields)              
    b = DoveTransient(t=t_i,omegas=omegas, omega_LOs=omega_LOs, gammas=gammas, pulse_timings=times, efields=efields)  
    c = DoveTransient(t=t_i,omegas=omegas, omega_LOs=omega_LOs, gammas=gammas, pulse_timings=times, efields=efields)
    x,y,z = await asyncio.gather(a.rho1(1620, renorm=True),    
                                b.rho2(renorm=True, weight=0.65),   
                                c.rho3(renorm=True, weight=0.47))
    
    trans_w1[0].set_ydata(x)
    #trans_w2[0].set_ydata(y)
    #trans_w3[0].set_ydata(z)
    trans_efields[0].set_ydata([E_pulses(t) for t in x_axis])
    int_plus[0].set_ydata((x+y)**2)
    int_plus_efields[0].set_ydata([E_pulses(t) for t in x_axis])
    int_minus[0].set_ydata((x-y)**2)
    int_minus_efields[0].set_ydata([E_pulses(t) for t in x_axis])
    print(time.time()-t1)
    
def wrapper(val):
    return asyncio.run(update(val))
def d1_tb_wrapper(val):
    val=float(val)
    return d1_slider.set_val(val)
def d2_tb_wrapper(val):
    val=float(val)
    return d2_slider.set_val(val)
def w1_tb_wrapper(val):
    val=float(val)
    return w1_slider.set_val(val)
def w2_tb_wrapper(val):
    val=float(val)
    return w2_slider.set_val(val)
    
d1_slider.on_changed(wrapper)
d2_slider.on_changed(wrapper)
w1_slider.on_changed(wrapper)
w2_slider.on_changed(wrapper)
d1_tb.on_submit(d1_tb_wrapper)
d2_tb.on_submit(d2_tb_wrapper)
w1_tb.on_submit(w1_tb_wrapper)
w2_tb.on_submit(w2_tb_wrapper)

print("HERE")
plt.show()




"""
w1_scanrange = np.linspace(1550, 1750, 21)
w2_scanrange = np.linspace(1550, 1750, 21)
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
        simulation_data[j,i]=sum(outcoherence)
        end=time.time()

        print(f'{(end-start)*remaining/60} mins remaining', remaining)

plt.imshow(simulation_data, interpolation="nearest", origin="lower")
plt.colorbar()
plt.show()
"""












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
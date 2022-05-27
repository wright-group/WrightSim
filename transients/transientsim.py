#imports 
from ast import increment_lineno
import numpy as np
import matplotlib.pyplot as plt


#constants
c = 3e10 #in cm/s
hbar = (6.6e-27)/2/np.pi
I = 1e17
i = np.linspace(0.01,16000, num=16000)
t_incr = 0.0006e-12
t_i = t_incr*i
N = 1.97e22
F4 = 1.3442
Gamma = 5.15e11
mu_const = 1.14e-18
delta_omega = (77000 - 1e7/532 - 3000)*c*2*np.pi
n = 1.344
F = (n**2 + 2)/3
Rabi = lambda mu: (mu/hbar)*np.sqrt(8*np.pi*I/c)
chi3_CARS = N*F4*mu_const**4 / (24*(hbar**3)*Gamma*(delta_omega**2))

omega0 = 2*np.pi*c*0
omega1 = 2*np.pi*c*2253
omega2 = 2*np.pi*c*3163
omega3 = 2*np.pi*c*12500

gamma0 = 0*c
gamma1 = 2*np.pi*c*2.846
gamma2 = 2*np.pi*c*4.39
gamma3 = 1e12

mu = lambda e, gamma, omega: np.sqrt((n*hbar*c*gamma*np.log(10)*e)/(6.02e20*4*np.pi*omega*F))
chi_CARS = 5.75e-15
e1 = 42.9
e2 = 3.16

mu1 = mu(e1,gamma1,omega1)
mu2 = mu(e2,gamma2,omega2)
mu3 = mu_const

omega1_LO = omega1*1.07
omega2_LO = omega2*1.07
omega3_LO = omega3*1.01

Rabi1 = Rabi(mu1)
Rabi2 = Rabi(mu2)
Rabi3 = Rabi(mu3)

delay = 900e-15
t_pulse = 1000.00001e-15
dt1 = t_pulse
t0 = 0.01*t_pulse
t1 = t_pulse
t2 = delay+t_pulse
t3 = t2+t_pulse
t4 = t3+delay
t5 = t4+t_pulse

i_t4 = t4/t_incr
i_t5 = t5/t_incr

hsthresh = 0.5 #Not in John's work, but added for convenince
Excit = lambda t,t0,t1: np.heaviside(t-t0, hsthresh)-np.heaviside(t-t1, hsthresh)

rho1 = lambda t: ((((np.exp(-gamma0*t) - np.exp(-gamma1*t)) / (2J*(gamma1-gamma0)))\
    * (np.heaviside(t-t0,hsthresh) - np.heaviside(t-t1,hsthresh))\
         + np.heaviside(t-t1,hsthresh) * ((np.exp(-gamma0*t1)-np.exp(-gamma1*t1)) / (2J*(gamma1-gamma0)))\
             *(np.exp(-gamma1*(t-t1))))*np.exp(-1J*omega1*t)*np.exp(1J*omega1_LO*t)*Rabi1)
#####NEED TO ADD OTHER RHO



rho1_i = np.real(rho1(t_i))
#####NEED TO ADD OTHER RHO_I




B1_i = np.sqrt(rho1_i*rho1_i)
#####NEED TO ADD OTHER B_i




rho1_max = np.max(B1_i)
#####NEED TO ADD OTHER RHO_max



A1_i = rho1_i/rho1_max
##### NEED TO ADD OTHER A_i




plt.plot(t_i*1e12, np.real(A1_i))
plt.plot(t_i*1e12, 0.9*(Excit(t_i,t0,t1)))
plt.show()


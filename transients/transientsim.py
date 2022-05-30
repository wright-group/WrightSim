# imports
import numpy as np
import matplotlib.pyplot as plt


# constants and system setup
c = 3e10  # in cm/s
hbar = (6.6e-27) / 2 / np.pi
I = 1e17
i = np.linspace(0.01, 16000, num=16000)
t_incr = 0.0006e-12
t_i = t_incr * i
N = 1.97e22
F4 = 1.3442
Gamma = 5.15e11
mu_const = 1.14e-18
delta_omega = (77000 - 1e7 / 532 - 3000) * c * 2 * np.pi
n = 1.344
F = (n ** 2 + 2) / 3
Rabi = lambda mu: (mu / hbar) * np.sqrt(8 * np.pi * I / c)
chi3_CARS = N * F4 * mu_const ** 4 / (24 * (hbar ** 3) * Gamma * (delta_omega ** 2))

omega0 = 2 * np.pi * c * 0
omega1 = 2 * np.pi * c * 2253
omega2 = 2 * np.pi * c * 3163
omega3 = 2 * np.pi * c * 12500

gamma0 = 0 * c
gamma1 = 2 * np.pi * c * 2.846
gamma2 = 2 * np.pi * c * 4.39
gamma3 = 1e12

mu = lambda e, gamma, omega: np.sqrt((n * hbar * c * gamma * np.log(10) * e) / (6.02e20 * 4 * np.pi * omega * F))
chi_CARS = 5.75e-15
e1 = 42.9
e2 = 3.16

mu1 = mu(e1, gamma1, omega1)
mu2 = mu(e2, gamma2, omega2)
mu3 = mu_const

omega1_LO = omega1 * 1.07
omega2_LO = omega2 * 1.07
omega3_LO = omega3 * 1.01

Rabi1 = Rabi(mu1)
Rabi2 = Rabi(mu2)
Rabi3 = Rabi(mu3)

delay = 900e-15

##Simulation 1
t_pulse = 1000.00001e-15
dt1 = t_pulse
t0 = 0.01 * t_pulse
t1 = t_pulse
t2 = delay + t_pulse
t3 = t2 + t_pulse
t4 = t3 + delay
t5 = t4 + t_pulse

i_t4 = t4 / t_incr
i_t5 = t5 / t_incr

hsthresh = 0.5  # Not in John's work, but added for convenience
Excit = lambda t, t0, t1: np.heaviside(t - t0, hsthresh) - np.heaviside(t - t1, hsthresh)

rho1 = lambda t: ((((np.exp(-gamma0 * t) - np.exp(-gamma1 * t)) / (2J * (gamma1 - gamma0)))
                   * (np.heaviside(t - t0, hsthresh) - np.heaviside(t - t1, hsthresh))
                   + np.heaviside(t - t1, hsthresh) * (
                               (np.exp(-gamma0 * t1) - np.exp(-gamma1 * t1)) / (2J * (gamma1 - gamma0)))
                   * (np.exp(-gamma1 * (t - t1)))) * np.exp(-1J * omega1 * t) * np.exp(1J * omega1_LO * t) * Rabi1)

rho2 = lambda t: ((np.exp(-gamma0 * t1) - np.exp(-gamma1 * t1)) / (2J * (gamma1 - gamma0))
                  * np.exp(-gamma1 * (t2 - t1))) * 1 * \
                 ((1 * ((np.exp(-gamma1 * (t - t2)) - np.exp(-gamma2 * (t - t2))) / (2J * (gamma2 - gamma1)))
                   * (np.heaviside(t - t2, hsthresh) - np.heaviside(t - t3, hsthresh))
                   - np.heaviside(t - t3, hsthresh) * ((np.exp(-gamma1 * (t3 - t2)) - np.exp(-gamma2 * (t3 - t2)))
                                                       / (2J * (gamma2 - gamma1))) * np.exp(-gamma2 * (t - t3)))
                  * np.exp(-1J * omega2 * t)) * np.exp(1J * omega2_LO * t) * Rabi1 * Rabi2

rho3 = lambda t: ((np.exp(-gamma0 * t1) - np.exp(-gamma1 * t1)) / (2J * (gamma1 - gamma0))
                  * np.exp(-gamma1 * (t2 - t1))) \
                 * ((np.exp(-gamma2 * (t3-t2)) - np.exp(-gamma3 * (t3-t2))) / (2J * (gamma3 - gamma2))
                    * np.exp(-gamma2 * (t4 - t3))) * \
                 ((1 * (np.exp(-gamma2 * (t - t4)) / (2J * (delta_omega - gamma2)))
                   * (np.heaviside(t - t4, hsthresh) - np.heaviside(t - t5, hsthresh)))
                  * np.exp(-1J * omega3 * t)) * np.exp(1J * omega3_LO * t) * Rabi1 * Rabi2 * Rabi3

fracSS_1 = 1 - np.exp(-gamma1 * t_pulse)
rho3_out = np.abs(rho3(t5))

rho1_i = np.real(rho1(t_i))
rho2_i = np.real(rho2(t_i))
rho3_i = np.real(rho3(t_i))

B1_i = np.abs(rho1_i)
B2_i = np.abs(rho2_i)
B3_i = np.abs(rho3_i)

rho1_max = np.max(B1_i)
rho2_max = np.max(B2_i)
rho3_max = np.max(B3_i)

A1_i = rho1_i / rho1_max
A2_i = 0.65 * rho2_i / rho2_max
A3_i = 0.4 * rho3_i / rho3_max


# plotting
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.plot(t_i * 1e12, np.real(A1_i), label=u'A1\u1D62', c='red')
ax1.plot(t_i * 1e12, np.real(A2_i), label=u'A2\u1D62', c='springgreen')
ax1.plot(t_i * 1e12, np.real(A3_i), label=u'A3\u1D62', c='dodgerblue')
#ax1.plot(t_i * 1e12, np.real(A1_i)+np.real(A2_i)+np.real(A3_i), label=u'A3\u1D62', c='dodgerblue')
ax1.plot(t_i * 1e12, 0.9 * (Excit(t_i, t0, t1) + Excit(t_i, t2, t3) + Excit(t_i, t4, t5)), label=u'E1\u1D62', c='k')
ax1.set_xlabel(u't\u1D62 \u2219 10\u00B9\u00B2')
ax1.legend()

plt.show()

##Simulation 2
t_pulse = 35.00001e-15
dt2 = t_pulse
t0 = 0.1 * t_pulse
t1 = t_pulse
t2 = delay + t_pulse
t3 = t2 + t_pulse
t4 = t3 + delay
t5 = t4 + t_pulse

i_t4 = t4 / t_incr
i_t5 = t5 / t_incr
# print(i_t4, i_t5) # NOTE: i_t5 doesnt print here what is on John's code...

hsthresh = 0.5  # Not in John's work, but added for convenience
Excit = lambda t, t0, t1: np.heaviside(t - t0, hsthresh) - np.heaviside(t - t1, hsthresh)

rho4 = lambda t: ((((np.exp(-gamma0 * t) - np.exp(-gamma1 * t)) / (2J * (gamma1 - gamma0)))
                   * (np.heaviside(t - t0, hsthresh) - np.heaviside(t - t1, hsthresh))
                   + np.heaviside(t - t1, hsthresh) * (
                               (np.exp(-gamma0 * t1) - np.exp(-gamma1 * t1)) / (2J * (gamma1 - gamma0)))
                   * (np.exp(-gamma1 * (t - t1)))) * np.exp(-1J * omega1 * t) * np.exp(1J * omega1_LO * t) * Rabi1)

rho5 = lambda t: (np.heaviside(t2-t1, hsthresh) * ((np.exp(-gamma0 * t1) - np.exp(-gamma1 * t1))
                                                   / (2J * (gamma1 - gamma0))) * np.exp(-gamma1 * (t2-t1))) \
                 * ((((np.exp(-gamma1 * (t-t2)) - np.exp(-gamma2 * (t-t2))) / (2J * (gamma2 - gamma1)))
                     * (np.heaviside(t-t2, hsthresh) - np.heaviside(t-t3, hsthresh)) + np.heaviside(t-t3, hsthresh)
                     * ((np.exp(-gamma1 * (t3-t2)) - np.exp(-gamma2*(t3-t2))) / (2J * (gamma2 - gamma1)))
                     * np.exp(-gamma2*(t-t3))) * np.exp(-1J * omega2 * t)) * np.exp(1J * omega2_LO * t) * Rabi1 * Rabi2

rho6 = lambda t: (((np.exp(-gamma0 * t2) - np.exp(-gamma1 * t2)) / (2J * (gamma1 - gamma0)))
                  * (np.heaviside(t2-t0, hsthresh - np.heaviside(t2-t1, hsthresh))
                  + np.heaviside(t2-t1, hsthresh) * ((np.exp(-gamma0 * t1) - np.exp(-gamma1 * t1))
                                                     / (2J * (gamma1 - gamma0))) * np.exp(-gamma1 * (t2-t1)))) \
                    * (((np.exp(-gamma1 * (t4-t2)) - np.exp(-gamma2 * (t4-t2))) / (2J * (gamma2 - gamma1)))
                       * (np.heaviside(t4-t2, hsthresh) - np.heaviside(t4-t3, hsthresh))
                       + np.heaviside(t4-t3, hsthresh) * ((np.exp(-gamma2 * (t3-t2)) - np.exp(-gamma3 * (t3-t2)))
                                                          / (2J * (gamma3 - gamma2))) * np.exp(-gamma2 * (t4-t3))) \
                 * 1 * ((1 * ((np.exp(-gamma2 * (t-t4))) / (2J * (delta_omega - gamma2)))
                         * (np.heaviside(t-t4, hsthresh) - np.heaviside(t-t5, hsthresh)))
                        * np.exp(-1J * omega3 * t)) * 1 * Rabi1 * Rabi2 * Rabi3


"""fracSS_1 = 1 - np.exp(-gamma1 * t_pulse)
rho3_out = np.abs(rho3(t5))
print(fracSS_1, rho3_out)"""

rho4_i = np.real(rho4(t_i))
rho5_i = np.real(rho5(t_i))
rho6_i = np.real(rho6(t_i))

B4_i = np.abs(rho4_i)
B5_i = np.abs(rho5_i)
B6_i = np.abs(rho6_i)

rho4_max = np.max(B4_i)
rho5_max = np.max(B5_i)
rho6_max = np.max(B6_i)

A4_i = rho4_i / rho4_max
A5_i = 0.45 * rho5_i / rho5_max
A6_i = 0.3 * rho6_i / rho6_max


# plotting
fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.plot(t_i * 1e12, np.real(A4_i), label=u'A4\u1D62', c='red')
ax2.plot(t_i * 1e12, np.real(A5_i), label=u'A5\u1D62', c='springgreen')
ax2.plot(t_i * 1e12, np.real(A6_i), label=u'A6\u1D62', c='dodgerblue')
# ax2.plot(t_i * 1e12, np.real(A4_i)+np.real(A5_i)+np.real(A6_i), label=u'A6\u1D62', c='dodgerblue')
ax2.plot(t_i * 1e12, 0.9 * (Excit(t_i, t0, t1) + Excit(t_i, t2, t3) + Excit(t_i, t4, t5)), label=u'E1\u1D62', c='k')
ax2.set_xlabel(u't\u1D62 \u2219 10\u00B9\u00B2')
ax2.legend()

plt.show()

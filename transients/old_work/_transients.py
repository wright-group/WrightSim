__all__ = ["DoveTransient"]


# imports 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import asyncio

# global constants
n = 1.344
F = (n ** 2 + 2) / 3 #figuring out what this is
c = 3e10  # in cm/s
hbar = (6.6e-27) / 2 / np.pi
I = 1e17 #dont know what this is 
delta_omega = (77000 - 1e7 / 532 - 3000) * c * 2 * np.pi #dont really know what this is yet




class DoveTransient():
    """
    This class takes the necessary parameters to calculate the transient responses 
    fue to 1, 2, or 3 pulses for DOVE under the approximation of Heaviside pulses.

    The parameters are:
    omegas: list 
        This is a list of frequency values in wavenumbers order by the excitation frequency
    omega_LOs: list
        This is a list of Local Oscillator corrections to the transition frequencies. (relative to 
        omegas list.)
    gammas: list
        This is a list of dephasing gammas (largely ignoring population relaxation) of the transitions
        in wavenumbers. 
    pulse_timings = list
        This is a list of the pulse timings with t0, t1, t2, t3, t4, and t5. t0 is the time the first 
        heaviside pulse is encountered and t5 is the time at which the end of the last pulse is encountered.
    t: list 
        This is a list of the time(s) for which an intensity of the transient oscillations is measured.
    
    """

    def __init__(self,
                omegas=[0,2253,3163,12500], 
                omega_LOs=[1,1.07,1.07,1.01], 
                gammas=[0,2.846,4.39,1e12], 
                pulse_timings=[0,1000e-15,1900e-15,2900e-15,3800e-15,4800e-15], 
                t=[4800e-15],
                efields=[42.9,3.16]):

        self.omegas = np.array(omegas) * 2 * np.pi * c
        self.omega_LOs = np.array(omega_LOs)
        
        self.gammas = np.array(gammas) * 2 * np.pi * c
        
        self.pulse_timings = np.array(pulse_timings)
        self.time = t

        self.efields = efields
        
        self.mu1=self.mu(self.efields[0],self.gammas[1],self.omegas[1])
        self.mu2=self.mu(self.efields[1],self.gammas[2],self.omegas[2])
        self.mu3=1.14e-18

        self.Rabi1=self.Rabi(self.mu1)
        self.Rabi2=self.Rabi(self.mu2)
        self.Rabi3=self.Rabi(self.mu3)

        self.out = None



    async def rho1(self, hsthresh=0.5, renorm=True, weight=1):
        global delta_omega
        gamma0 = self.gammas[0]
        gamma1 = self.gammas[1]
        t0 = self.pulse_timings[0]
        t1 = self.pulse_timings[1]
        omega1 = self.omegas[1]
        omega1_LO = (self.omegas*self.omega_LOs)[1]

        rho = lambda t: ((((np.exp(-gamma0 * t) - np.exp(-gamma1 * t)) / (2J * (gamma1 - gamma0)))
                   * (np.heaviside(t - t0, hsthresh) - np.heaviside(t - t1, hsthresh))
                   + np.heaviside(t - t1, hsthresh) * (
                               (np.exp(-gamma0 * t1) - np.exp(-gamma1 * t1)) / (2J * (gamma1 - gamma0)))
                   * (np.exp(-gamma1 * (t - t1)))) * np.exp(-1J * omega1 * t) * np.exp(1J * omega1_LO * t) * self.Rabi1)

        if renorm: 
            rho_t = np.array(list(map(np.real, map(rho, self.time))))
            rho_max = np.max(np.array(list(map(np.abs, rho_t))))
            self.out = weight*rho_t/rho_max
            return self.out

        elif not renorm:
            self.out = np.array(list(map(rho, self.time)))
            return self.out


    async def rho2(self, hsthresh=0.5, renorm=True, weight=1):
        global delta_omega
        gamma0 = self.gammas[0]
        gamma1 = self.gammas[1]
        gamma2 = self.gammas[2]
        t1 = self.pulse_timings[1]
        t2 = self.pulse_timings[2]
        t3 = self.pulse_timings[3]
        omega2 = self.omegas[2]
        omega2_LO = (self.omegas*self.omega_LOs)[2]

        rho = lambda t: ((np.exp(-gamma0 * t1) - np.exp(-gamma1 * t1)) / (2J * (gamma1 - gamma0))
                  * np.exp(-gamma1 * (t2 - t1))) * 1 * \
                 ((1 * ((np.exp(-gamma1 * (t - t2)) - np.exp(-gamma2 * (t - t2))) / (2J * (gamma2 - gamma1)))
                   * (np.heaviside(t - t2, hsthresh) - np.heaviside(t - t3, hsthresh))
                   - np.heaviside(t - t3, hsthresh) * ((np.exp(-gamma1 * (t3 - t2)) - np.exp(-gamma2 * (t3 - t2)))
                                                       / (2J * (gamma2 - gamma1))) * np.exp(-gamma2 * (t - t3)))
                  * np.exp(-1J * omega2 * t)) * np.exp(1J * omega2_LO * t) * self.Rabi1 * self.Rabi2

        if renorm: 
            rho_t = np.array(list(map(np.real, map(rho, self.time))))
            rho_max = np.max(np.array(list(map(np.abs, rho_t))))
            self.out = weight*rho_t/rho_max
            return self.out

        elif not renorm:
            self.out = np.array(list(map(rho, self.time)))
            return self.out


    async def rho3(self, hsthresh=0.5, renorm=True, weight=1):
        global delta_omega
        gamma0 = self.gammas[0]
        gamma1 = self.gammas[1]
        gamma2 = self.gammas[2]
        gamma3 = self.gammas[3]
        t1 = self.pulse_timings[1]
        t2 = self.pulse_timings[2]
        t3 = self.pulse_timings[3]
        t4 = self.pulse_timings[4]
        t5 = self.pulse_timings[5]
        omega3 = self.omegas[3]
        omega3_LO = (self.omegas*self.omega_LOs)[3]

        rho = lambda t: ((np.exp(-gamma0 * t1) - np.exp(-gamma1 * t1)) / (2J * (gamma1 - gamma0))
                  * np.exp(-gamma1 * (t2 - t1))) \
                 * ((np.exp(-gamma2 * (t3-t2)) - np.exp(-gamma3 * (t3-t2))) / (2J * (gamma3 - gamma2))
                    * np.exp(-gamma2 * (t4 - t3))) * \
                 ((1 * (np.exp(-gamma2 * (t - t4)) / (2J * (delta_omega - gamma2)))
                   * (np.heaviside(t - t4, hsthresh) - np.heaviside(t - t5, hsthresh)))
                  * np.exp(-1J * omega3 * t)) * np.exp(1J * omega3_LO * t) * self.Rabi1 * self.Rabi2 * self.Rabi3

        if renorm: 
            rho_t = np.array(list(map(np.real, map(rho, self.time))))
            rho_max = np.max(np.array(list(map(np.abs, rho_t))))
            self.out = weight*rho_t/rho_max
            return self.out

        elif not renorm:
            self.out = np.array(list(map(rho, self.time)))
            return self.out

    
    def plot(self, t_scale, a_scale=1):
        plt.plot(self.time*t_scale, self.out*a_scale)
        plt.show()  


# tools
    def mu(self, e, gamma, omega):
        global n, F, c, hbar
        return np.sqrt((n * hbar * c * gamma * np.log(10) * e) / (6.02e20 * 4 * np.pi * omega * F))

    def Rabi(self, mu):
        global I, c, hbar
        return (mu / hbar) * np.sqrt(8 * np.pi * I / c)

    #def gausspulse


    
        
        

# TODO: maybe add rho_i, B_i, rho_max, and A_i?? Or figure out how to map stuff to make plots
# maybe add time based plotable tuple with x and y values from class methods.
# May benefit from a separate pulse class (maybe experiment too similar to wrightsim??)
# U.U
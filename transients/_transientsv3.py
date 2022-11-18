# imports
import numpy as np
#import asyncio

# useful constants
c = 2.998e8  # in m/s

# useful functions


def wntohz(wn):
    return wn*c*100*2*np.pi


# modeling functions


def hs(x):
    return np.heaviside(x, 1)


def delta_ij(omega_ij, gamma_ij):
    return omega_ij-1J*gamma_ij


def pulse(ti, tf, t):
    return hs(t-ti)-hs(t-tf)


def fid(rho0_ij,
        delta_ij,
        t):
    return rho0_ij*np.exp(-1J*delta_ij*t)


def ket_abs(rabi_ik,
            delta_kj,
            delta_ij,
            omega,
            omega_ik,
            gamma_ij,
            gamma_kj,
            t):
    rho_ij = 1J/2*rabi_ik*((np.exp(-1J*(delta_kj+omega)*t)-np.exp(-1J*delta_ij*t))
                           / (omega_ik-omega-1J*(gamma_ij-gamma_kj)))
    return rho_ij


def ket_emis(rabi_ik,
             delta_kj,
             delta_ij,
             omega,
             omega_ik,
             gamma_ij,
             gamma_kj,
             t):
    rho_ij = 1J/2*rabi_ik*rho_kj*((np.exp(-1J*(delta_kj-omega)*t)-np.exp(-1J*delta_ij*t))
                                  / (omega_ik+omega-1J*(gamma_ij-gamma_kj)))
    return rho_ij


def bra_abs(rabi_jk,
            delta_ik,
            delta_ij,
            omega,
            omega_kj,
            gamma_ij,
            gamma_ik,
            t):
    rho_ij = -1J/2*rabi_jk*((np.exp(-1J*(delta_ik+omega)*t)-np.exp(-1J*delta_ij*t))
                            / (omega_kj-omega-1J*(gamma_ij-gamma_ik)))
    return rho_ij


def bra_emis(rabi_jk,
             delta_ik,
             delta_ij,
             omega,
             omega_kj,
             gamma_ij,
             gamma_ik,
             t):
    return -1J/2*rabi_jk*((np.exp(-1J*(delta_ik-omega)*t)-np.exp(-1J*delta_ij*t))
                          / (omega_kj+omega-1J*(gamma_ij-gamma_ik)))


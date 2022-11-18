# imports
# cython: infer_types=True

import numpy as np
import cython


# useful constants
c = 2.998e8 # in m/s

# useful functions
wntohz = lambda wn: wn*c*100*2*np.pi


# modeling functions

hs = lambda x: np.heaviside(x, 1)

delta_ij = lambda omega_ij, gamma_ij: omega_ij-1J*gamma_ij

pulse = lambda ti, tf, t: hs(t-ti)-hs(t-tf)

cpdef fid(float complex rho0_ij,
        float complex delta_ij,
        t):
    return rho0_ij*np.exp(-1J*delta_ij*t)

cpdef ket_abs(float rabi_ik,
            float complex delta_kj,
            float complex delta_ij,
            float omega,
            float omega_ik,
            float gamma_ij,
            float gamma_kj,
            t):

    return 1J/2*rabi_ik*((np.exp(-1J*(delta_kj+omega)*t)-np.exp(-1J*delta_ij*t))
                                  /(omega_ik-omega-1J*(gamma_ij-gamma_kj)))

cpdef ket_emis(float rabi_ik,
             float complex delta_kj,
             float complex delta_ij,
             float omega,
             float omega_ik,
             float gamma_ij,
             float gamma_kj,
             t):

    return 1J/2*rabi_ik*((np.exp(-1J*(delta_kj-omega)*t)-np.exp(-1J*delta_ij*t))
                                  /(omega_ik+omega-1J*(gamma_ij-gamma_kj)))

cpdef bra_abs(float rabi_jk,
            float complex delta_ik,
            float complex delta_ij,
            float omega,
            float omega_kj,
            float gamma_ij,
            float gamma_ik,
            t):

    return -1J/2*rabi_jk*((np.exp(-1J*(delta_ik+omega)*t)-np.exp(-1J*delta_ij*t))
                                  /(omega_kj-omega-1J*(gamma_ij-gamma_ik)))

cpdef bra_emis(float rabi_jk,
             float complex delta_ik,
             float complex delta_ij,
             float omega,
             float omega_kj,
             float gamma_ij,
             float gamma_ik,
             t):

    return -1J/2*rabi_jk*((np.exp(-1J*(delta_ik-omega)*t)-np.exp(-1J*delta_ij*t))
                                  /(omega_kj+omega-1J*(gamma_ij-gamma_ik)))

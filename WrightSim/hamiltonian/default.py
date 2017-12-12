
import numpy as np


class Hamiltonian:


    def __init__(self, rho=None, tau=None, mu=None,
                        w_0=None, w_central=7000., coupling=0,
                        propegator='rk', phase_cycle=False,
                        dm_vector=['gg1','ag','ga','aa','2ag','ag2','2aa'],
                        time_orderings=list(range(1,7))):
        if rho is None:
            self.rho = np.zeros(len(dm_vector), dtype=np.complex64)
            self.rho[0] = 1.
        else:
            self.rho = rho

        if tau is None:
            self.tau = np.array([np.inf, 50., 50., np.inf, 50., 50., 50.])
        else:
            self.tau = tau

        if mu is None:
            self.mu = np.array([0., 1., 0., 0., 0., 0., 1.])
        else:
            self.mu = mu

        if w_0 is None:
            w_ag = wa_central
            w_2aa = w_ag - a_coupling
            w_2ag = 2*w_ag - a_coupling
            w_gg = 0.
            w_aa = w_gg
            self.w_0 = np.array( [w_gg, w_ag, -w_ag, w_aa, w_2ag, w_ag, w_2aa] )
        else:
            self.w_0 = w_0

        self.propegator = propegator
        self.phase_cycle = phase_cycle
        self.dm_vector = dm_vector

        self.time_orderings = time_orderings
        self.Gamma = 1./self.tau

    def matrix(self, efields, time, energies):
        E1,E2,E3 = efields[0:3]o

        out1 = self._gen_matrix(E1, E2, E3, time, energies, True)
        out2 = self._gen_matrix(E1, E2, E3, time, energies, False)

        return np.array([out1, out2], dtype=np.complex64)

    def _gen_matrix(self, E1, E2, E3, time, energies, w1first=True):
        """
        creates the coupling array given the input e-fields values for a specific time, t
        w1first selects whether w1 or w2p is the first interacting positive field
        
        Currently neglecting pathways where w2 and w3 require different frequencies
        (all TRIVE space, or DOVE on diagonal)
        
        Matrix formulated such that dephasing/relaxation is accounted for 
        outside of the matrix
        """
        wag  = energies[1]
        w2aa = energies[6]
        
        mu_ag = self.mu[1]
        mu_2aa = self.mu[6]
    
        if w1first==True:
            first  = E1
            second = E3
        else:
            first  = E3
            second = E1

        A_1 = 0.5j * mu_ag * first * np.exp(1j * wag * time)
        A_2 = 0.5j * mu_ag * first * np.exp(1j * wag * time)
        A_2prime = 0.5j * mu_ag * first * np.exp(1j * wag * time)
        B_1 = 0.5j * mu_2aa * first * np.exp(1j * wag * time)
        B_2 = 0.5j * mu_2aa * first * np.exp(1j * wag * time)
        B_2prime = 0.5j * mu_2aa * first * np.exp(1j * wag * time)

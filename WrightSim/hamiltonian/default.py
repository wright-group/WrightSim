
import numpy as np
from ..mixed import propagate


class Hamiltonian:


    def __init__(self, rho=None, tau=None, mu=None,
                        omega=None, w_central=7000., coupling=0,
                        propagator=None, phase_cycle=False,
                        labels=['00','01 -2','10 2\'','10 1','20 1+2\'','11 1-2','11 2\'-2', '10 1-2+2\'', '21 1-2+2\''],
                        time_orderings=list(range(1,7)), recorded_indices = [7, 8]):
        if rho is None:
            self.rho = np.zeros(len(labels), dtype=np.complex64)
            self.rho[0] = 1.
        else:
            self.rho = rho

        if tau is None:
            tau = 50. #fs
        if np.isscalar(tau):
            self.tau = np.array([np.inf, tau, tau, tau, tau, np.inf, np.inf, tau, tau])
        else:
            self.tau = tau

        if mu is None:
            self.mu = np.array([np.nan, 1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.])
        else:
            self.mu = mu

        if omega is None:
            w_ag = w_central
            w_2aa = w_ag - coupling
            w_2ag = 2*w_ag - coupling
            w_gg = 0.
            w_aa = w_gg
            self.omega = np.array( [w_gg, -w_ag, w_ag, w_ag, w_2ag, w_aa, w_aa, w_ag, w_2aa] )
        else:
            self.omega = w_0

        if propagator is None:
            self.propagator = propagate.runge_kutta
        else:
            self.propagator = propagator
        self.phase_cycle = phase_cycle
        self.labels = labels
        self.recorded_indices = recorded_indices

        self.time_orderings = time_orderings
        self.Gamma = 1./self.tau

    def matrix(self, efields, time):
        E1,E2,E3 = efields[0:3]
        return  self._gen_matrix(E1, E2, E3, time, self.omega)

    def _gen_matrix(self, E1, E2, E3, time, energies):
        """
        creates the coupling array given the input e-fields values for a specific time, t
        w1first selects whether w1 or w2p is the first interacting positive field
        
        Currently neglecting pathways where w2 and w3 require different frequencies
        (all TRIVE space, or DOVE on diagonal)
        
        Matrix formulated such that dephasing/relaxation is accounted for 
        outside of the matrix
        """
        wag  = energies[1]
        w2aa = energies[-1]
        
        mu_ag = self.mu[1]
        mu_2aa = self.mu[-1]
    
        A_1 = 0.5j * mu_ag * E1 * np.exp(-1j * wag * time)
        A_2 = 0.5j * mu_ag * E2 * np.exp(1j * wag * time)
        A_2prime = 0.5j * mu_ag * E3 * np.exp(-1j * wag * time)
        B_1 = 0.5j * mu_2aa * E1 * np.exp(-1j * w2aa * time)
        B_2 = 0.5j * mu_2aa * E2 * np.exp(1j * w2aa * time)
        B_2prime = 0.5j * mu_2aa * E3 * np.exp(-1j * w2aa * time)

        out = np.zeros((len(time), len(energies), len(energies)), dtype=np.complex64)

        if 3 in self.time_orderings or 5 in self.time_orderings:
            out[:,1,0] = -A_2
        if 4 in self.time_orderings or 6 in self.time_orderings:
            out[:,2,0] = A_2prime
        if 1 in self.time_orderings or 2 in self.time_orderings:
            out[:,3,0] = A_1
        if 3 in self.time_orderings:
            out[:,5,1] = A_1
        if 5 in self.time_orderings:
            out[:,6,1] = A_2prime
        if 4 in self.time_orderings:
            out[:,4,2] = B_1
        if 6 in self.time_orderings:
            out[:,6,2] = -A_2
        if 2 in self.time_orderings:
            out[:,4,3] = B_2prime
        if 1 in self.time_orderings:
            out[:,5,3] = -A_2
        if 2 in self.time_orderings or 4 in self.time_orderings:
            out[:,7,4] = B_2
            out[:,8,4] = -A_2
        if 1 in self.time_orderings or 3 in self.time_orderings:
            out[:,7,5] = -2 * A_2prime
            out[:,8,5] = B_2prime
        if 5 in self.time_orderings or 6 in self.time_orderings:
            out[:,7,6] = -2 * A_1
            out[:,8,6] = B_1

        for i in range(len(self.Gamma)):
            out[:,i,i] = -1 * self.Gamma[i]

        #NOTE: NISE multiplied outputs by the approriate mu in here
        #      This mu factors out, remember to use it where needed later
        #      Removed for clarity and aligning with Equation S15 of Kohler2016

        return out



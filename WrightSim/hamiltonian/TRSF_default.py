import numpy as np
from ..mixed import propagate


class Hamiltonian:
    
    
    def __init__(self, rho=None, tau=None, mu=None,
                 omega=None, 
                 labels = ['g',
                           'I_1', 'I_2', 'i_1', 'i_2',
                           '2I', '2i', 'c',
                           'a', 'b'
                           ],
                 #time_orderings = list(range(1,4)),
                 phase_cycle = False,
                 propagator = None,
                 recorded_indices = [8, 9]):

        if rho is None:
            self.rho = np.zeros(len(labels), dtype=np.complex128)
            self.rho[0] = 1.
        else:
            self.rho = rho
        
        if tau is None:
            self.tau = np.array([np.inf, 
                                 1e3, 1e3, 1e3, 1e3,
                                 5e2, 5e2, 5e2,
                                 5e1, 5e1])
        else:
            self.tau = tau

        if omega is None:
            w_I  = 1500.
            w_i  = 1600.
            w_2I = 2*w_I
            w_2i = 2*w_i
            w_c  = w_I + w_i
            w_a  = 19000.
            w_b  = 20000.
            self.omega = np.array([0,
                                   w_I, w_I, w_i, w_i,
                                   w_2I, w_2i, w_c,
                                   w_a, w_b])
        else:
            self.omega = omega
        
        if mu is None: 
            # TODO: mu.get('element', default value) so it's easier to customize
            mu = {'I_g'  : 1.0,
                  'i_g'  : 1.0,
                  'a_I'  : 1.0,
                  'b_I'  : 1.0,
                  'a_i'  : 1.0,
                  'b_i'  : 1.0,
                  'a_c'  : 1.0,
                  'b_c'  : 1.0,
                  'a_g'  : 1.0,
                  'b_g'  : 1.0,
                  }

            mu['2I_I'] = np.sqrt(2) * mu['I_g']
            mu['2i_i'] = np.sqrt(2) * mu['i_g']
            mu['c_I'] = mu['i_g']
            mu['c_i'] = mu['I_g']
            mu['a_2I'] = mu['a_g']
            mu['a_2i'] = mu['a_g']
            mu['b_2I'] = mu['b_g']
            mu['b_2i'] = mu['b_g']

        self.mu = mu

        if propagator is None:    
            self.propagator = propagate.runge_kutta
        else:
            self.propagator = propagator
        
        self.phase_cycle = phase_cycle
        self.labels = labels
        self.recorded_indices = recorded_indices
        #self.time_orderings = time_orderings
        self.Gamma = 1. / self.tau

    def matrix(self, efields, time):
        E1,E2,E3 = efields[0:3]
        return  self._gen_matrix(E1, E2, E3, time)

    def _gen_matrix(self, E1, E2, E3, time):
        
        out = np.zeros((len(time), len(self.omega), len(self.omega)), 
                       dtype=np.complex64)

        wn_to_omega = 2*np.pi*3*10**-5
        wI = self.omega[1] * wn_to_omega
        wi = self.omega[3] * wn_to_omega
        wII, wii, wc, wa, wb = self.omega[5:] * wn_to_omega
        
        wII_I = wII - wI
        wii_i = wii - wi
        wc_I = wc - wI
        wc_i = wc - wi

        wa_II = wa - wII
        wb_II = wb - wII
        wa_ii = wa - wii
        wb_ii = wb - wii
        wa_c = wa - wc
        wb_c = wb - wc
        
        mu_Ig = self.mu['I_g']
        mu_ig = self.mu['i_g']    

        mu_2II = self.mu['2I_I']
        mu_2ii = self.mu['2i_i']
        mu_cI  = self.mu['c_I']
        mu_ci  = self.mu['c_i']
        
        mu_a2I = self.mu['a_2I']
        mu_b2I = self.mu['b_2I']
        mu_a2i = self.mu['a_2i']
        mu_b2i = self.mu['b_2i']
        mu_ac  = self.mu['a_c']
        mu_bc  = self.mu['b_c']
        
        out[:,1,0] = 0.5j * mu_Ig * E1 * np.exp(1j * wI * time)
        out[:,2,0] = 0.5j * mu_Ig * E2 * np.exp(1j * wI * time)
        out[:,3,0] = 0.5j * mu_ig * E1 * np.exp(1j * wi * time)
        out[:,4,0] = 0.5j * mu_ig * E2 * np.exp(1j * wi * time)
        
        out[:,5,1] = 0.5j * mu_2II * E2 * np.exp(1j * wII_I * time)
        out[:,5,2] = 0.5j * mu_2II * E1 * np.exp(1j * wII_I * time)
        
        out[:,6,3] = 0.5j * mu_2ii * E2 * np.exp(1j * wii_i * time)
        out[:,6,4] = 0.5j * mu_2ii * E1 * np.exp(1j * wii_i * time)
        
        out[:,7,1] = 0.5j * mu_ci * E2 * np.exp(1j * wc_I * time)
        out[:,7,2] = 0.5j * mu_ci * E1 * np.exp(1j * wc_I * time)
        out[:,7,3] = 0.5j * mu_cI * E2 * np.exp(1j * wc_i * time)
        out[:,7,4] = 0.5j * mu_cI * E1 * np.exp(1j * wc_i * time)

        out[:,8,5] = 0.5j * mu_a2I * E3 * np.exp(1j * wa_II * time)
        out[:,8,6] = 0.5j * mu_a2i * E3 * np.exp(1j * wa_ii * time)
        out[:,8,7] = 0.5j * mu_ac * E3 * np.exp(1j * wa_c * time)
        
        out[:,9,5] = 0.5j * mu_b2I * E3 * np.exp(1j * wb_II * time)
        out[:,9,6] = 0.5j * mu_b2i * E3 * np.exp(1j * wb_ii * time)
        out[:,9,7] = 0.5j * mu_bc * E3 * np.exp(1j * wb_c * time)

        for i in range(self.Gamma.size):
            out[:,i,i] = -self.Gamma[i]

        return out


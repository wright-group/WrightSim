
import numpy as np

from ..mixed import propagate


class Hamiltonian:
    cuda_struct = """
    #include <pycuda-complex.hpp>
    #define I pycuda::complex<double>(0,1)

    struct Hamiltonian {
        int nStates;
        int nMu;
        int nTimeOrderings;
        int nRecorded;
        pycuda::complex<double>* rho;
        pycuda::complex<double>* mu;
        double* omega;
        double* Gamma;

        char* time_orderings;
        int* recorded_indices;
    };
    """
    cuda_mem_size = 4*4 + np.intp(0).nbytes*6


    def __init__(self, rho=None, tau=None, mu=None,
                        omega=None, w_central=7000., coupling=0,
                        propagator=None, phase_cycle=False,
                        labels=['00','01 -2','10 2\'','10 1','20 1+2\'','11 1-2','11 2\'-2', '10 1-2+2\'', '21 1-2+2\''],
                        time_orderings=list(range(1,7)), recorded_indices = [7, 8]):
        if rho is None:
            self.rho = np.zeros(len(labels), dtype=np.complex128)
            self.rho[0] = 1.
        else:
            self.rho = np.array(rho, dtype=np.complex128)

        if tau is None:
            tau = 50. #fs
        if np.isscalar(tau):
            self.tau = np.array([np.inf, tau, tau, tau, tau, np.inf, np.inf, tau, tau])
        else:
            self.tau = tau

        if mu is None:
            self.mu = np.array([1., 1.], dtype=np.complex128)
        else:
            self.mu = np.array(mu, dtype=np.complex128)

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

    def to_device(self, pointer):
        import pycuda.driver as cuda
        rho = cuda.to_device(self.rho)
        mu = cuda.to_device(self.mu)
        omega = cuda.to_device(self.omega)
        Gamma = cuda.to_device(self.Gamma)

        tos = [1 if i in self.time_orderings else 0 for i in range(1,7)]

        time_orderings = cuda.to_device(np.array(tos, dtype=np.int8))
        recorded_indices = cuda.to_device(np.array(self.recorded_indices, dtype=np.int32))

        cuda.memcpy_htod(pointer, np.array([len(self.rho)], dtype=np.int32))
        cuda.memcpy_htod(int(pointer) + 4, np.array([len(self.mu)], dtype=np.int32))
        #TODO: generalize nTimeOrderings
        cuda.memcpy_htod(int(pointer) + 8, np.array([6], dtype=np.int32))
        cuda.memcpy_htod(int(pointer) + 12, np.array([len(self.recorded_indices)], dtype=np.int32))

        cuda.memcpy_htod(int(pointer) + 16, np.intp(int(rho)))
        cuda.memcpy_htod(int(pointer) + 24, np.intp(int(mu)))
        cuda.memcpy_htod(int(pointer) + 32, np.intp(int(omega)))
        cuda.memcpy_htod(int(pointer) + 40, np.intp(int(Gamma)))
        cuda.memcpy_htod(int(pointer) + 48, np.intp(int(time_orderings)))
        cuda.memcpy_htod(int(pointer) + 56, np.intp(int(recorded_indices)))

        

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
        
        mu_ag = self.mu[0]
        mu_2aa = self.mu[-1]
    
        A_1 = 0.5j * mu_ag * E1 * np.exp(-1j * wag * time)
        A_2 = 0.5j * mu_ag * E2 * np.exp(1j * wag * time)
        A_2prime = 0.5j * mu_ag * E3 * np.exp(-1j * wag * time)
        B_1 = 0.5j * mu_2aa * E1 * np.exp(-1j * w2aa * time)
        B_2 = 0.5j * mu_2aa * E2 * np.exp(1j * w2aa * time)
        B_2prime = 0.5j * mu_2aa * E3 * np.exp(-1j * w2aa * time)

        out = np.zeros((len(time), len(energies), len(energies)), dtype=np.complex128)

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

    cuda_matrix_source = """
    __device__ void Hamiltonian_matrix(Hamiltonian ham, pycuda::complex<double>* efields,
                                                           double time, pycuda::complex<double>* out)
    {
        double wag = ham.omega[1];
        double w2aa = ham.omega[8];

        pycuda::complex<double> mu_ag =  1.;//ham.mu[0];
        pycuda::complex<double> mu_2aa = 1.;//ham.mu[1];

        pycuda::complex<double> E1 =  efields[0];
        pycuda::complex<double> E2 =  efields[1];
        pycuda::complex<double> E3 =  efields[2];
        

        pycuda::complex<double> A_1 = 0.5 * I * mu_ag * E1 * pycuda::exp(-1. * I * wag * time);
        pycuda::complex<double> A_2 = 0.5 * I * mu_ag * E2 * pycuda::exp(I * wag * time);
        pycuda::complex<double> A_2prime = 0.5 * I * mu_ag * E3 * pycuda::exp(-1. * I * wag * time);
        pycuda::complex<double> B_1 = 0.5 * I * mu_2aa * E1 * pycuda::exp(-1. * I * w2aa * time);
        pycuda::complex<double> B_2 = 0.5 * I * mu_2aa * E2 * pycuda::exp(I * w2aa * time);
        pycuda::complex<double> B_2prime = 0.5 * I * mu_2aa * E3 * pycuda::exp(-1. * I * w2aa * time);

        for (int i=0; i<ham.nStates * ham.nStates; i++) out[i] = pycuda::complex<double>();

        if(ham.time_orderings[2] || ham.time_orderings[4])
            out[1*ham.nStates + 0] = -1. * A_2;
        if(ham.time_orderings[3] || ham.time_orderings[5])
            out[2*ham.nStates + 0] = A_2prime;
        if(ham.time_orderings[0] || ham.time_orderings[1])
            out[3*ham.nStates + 0] = A_1;
        if(ham.time_orderings[2])
            out[5*ham.nStates + 1] = A_1;
        if(ham.time_orderings[4])
            out[6*ham.nStates + 1] = A_2prime;
        if(ham.time_orderings[3])
            out[4*ham.nStates + 2] = B_1;
        if(ham.time_orderings[5])
            out[6*ham.nStates + 2] = -1. * A_2;
        if(ham.time_orderings[0])
            out[4*ham.nStates + 3] = B_2prime;
        if(ham.time_orderings[1])
            out[5*ham.nStates + 3] = -1. * A_2;
        if(ham.time_orderings[1] || ham.time_orderings[3])
        {
            out[7*ham.nStates + 4] = B_2;
            out[8*ham.nStates + 4] = -1. * A_2;
        }
        if(ham.time_orderings[0] || ham.time_orderings[2])
        {
            out[7*ham.nStates + 5] = -2. * A_2prime;
            out[8*ham.nStates + 5] = B_2prime;
        }
        if(ham.time_orderings[4] || ham.time_orderings[5])
        {
            out[7*ham.nStates + 6] = -2. * A_1;
            out[8*ham.nStates + 6] = B_1;
        }

        for(int i=0; i<ham.nStates; i++) out[i*ham.nStates + i] = -1. * ham.Gamma[i];
    }
"""    


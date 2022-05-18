import numpy as np

from ..mixed import propagate

wn_to_omega = 2 * np.pi * 3 * 10 ** -5


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
    cuda_mem_size = 4 * 4 + np.intp(0).nbytes * 6

    def __init__(
        self,
        rho=None,
        tau=None,
        mu=None,
        omega=None,
        w_central=7000.0,
        coupling=0,
        propagator=None,
        phase_cycle=False,
        labels=[
            "00",
            "01 -2",
            "10 2'",
            "10 1",
            "20 1+2'",
            "11 1-2",
            "11 2'-2",
            "10 1-2+2'",
            "21 1-2+2'",
        ],
        time_orderings=list(range(1, 7)),
        recorded_indices=[7, 8],
    ):
        """Create a Hamiltonian object.

        Parameters
        ----------
        rho : 1-D array <Complex> (optional)
            The initial density vector, length defines N.
            Default is 1. in the ground state (index 0), and 0. everywhere else.
        tau : scalar or 1-D array (optional)
            The lifetime of the states in femptoseconds.
            For coherences, this is the pure dephasing time.
            For populations, this is the population lifetime.
            If tau is scalar, the same dephasing time is used for all coherences,
                              Populations do not decay by default (inf lifetime).
            This value is converted into a rate of decay, Gamma, by the multiplicative inverse.
            Default is 50.0 fs dephasing for all coherences, infinite for populations.
        mu : 1-D array <Complex> (optional)
            The dipole moments used either in computing state densities or output intensities
            Array like structures are converted to the proper data type.
            Order matters, and meaning is dependent on the individual Hamiltonian.
            Default is two values, both initially 1.0.
        omega : 1-D array <float64> (optional)
            The energies of various transitions.
            The default uses w_central and coupling parameters to compute the appropriate
                values for a TRIVE Hamiltonian
        w_central : float (optional)
            The cetral frequency of a resonance for a TRIVE Hamiltonian.
            Used only when ``omega`` is ``None``.
        coupling : float (optional)
            The copuling of states for a TRIVE Hamiltonian.
            Used only when ``omega`` is ``None``.
        propagator : function (optional)
            The evolution function to use to evolve the density matrix over time.
            Default is ``runge_kutta``.
        phase_cycle : bool (optional)
            Whether or not to use phase cycling.
            Not applicable to all Hamiltonians.
            Default is ``False``
        labels : list of string (optional)
            Human readable labels for the states. Not used in computation, only to keep track.
        time_orderings : list of int (optional)
            Which time orderings to use in the simulation.
            Default is all for a three interaction process: ``[1,2,3,4,5,6]``.
        recorded_indices : list of int (optional)
            Which density vector states to record through the simulation.
            Default is [7, 8], the output of a TRIVE Hamiltonian.

        """
        if rho is None:
            self.rho = np.zeros(len(labels), dtype=np.complex128)
            self.rho[0] = 1.0
        else:
            self.rho = np.array(rho, dtype=np.complex128)

        if tau is None:
            tau = 50.0  # fs
        if np.isscalar(tau):
            self.tau = np.array([np.inf, tau, tau, tau, tau, np.inf, np.inf, tau, tau])
        else:
            self.tau = tau

        # TODO: Think about dictionaries or some other way of labeling Mu values
        if mu is None:
            self.mu = np.array([1.0, 1.0], dtype=np.complex128)
        else:
            self.mu = np.array(mu, dtype=np.complex128)

        if omega is None:
            w_ag = w_central
            w_2aa = w_ag - coupling
            w_2ag = 2 * w_ag - coupling
            w_gg = 0.0
            w_aa = w_gg
            self.omega = np.array([w_gg, -w_ag, w_ag, w_ag, w_2ag, w_aa, w_aa, w_ag, w_2aa])
            self.omega *= wn_to_omega
        else:
            self.omega = omega

        if propagator is None:
            self.propagator = propagate.runge_kutta
        else:
            self.propagator = propagator
        self.phase_cycle = phase_cycle
        self.labels = labels
        self.recorded_indices = recorded_indices

        self.time_orderings = time_orderings
        self.Gamma = 1.0 / self.tau

    @property
    def omega_wn(self):
        return self.omega / wn_to_omega

    def to_device(self, pointer):
        """Transfer the Hamiltonian to a C struct in CUDA device memory.

        Currently expects a pointer to an already allocated chunk of memory.
        """
        import pycuda.driver as cuda

        # TODO: Reorganize to allocate here and return the pointer, this is more friendly
        # Transfer the arrays which make up the hamiltonian
        rho = cuda.to_device(self.rho)
        mu = cuda.to_device(self.mu)
        omega = cuda.to_device(self.omega)
        Gamma = cuda.to_device(self.Gamma)

        # Convert time orderings into a C boolean array of 1 and 0, offset by one
        tos = [1 if i in self.time_orderings else 0 for i in range(1, 7)]

        # Transfer time orderings and recorded indices
        time_orderings = cuda.to_device(np.array(tos, dtype=np.int8))
        recorded_indices = cuda.to_device(np.array(self.recorded_indices, dtype=np.int32))

        # Transfer metadata about the lengths of feilds
        cuda.memcpy_htod(pointer, np.array([len(self.rho)], dtype=np.int32))
        cuda.memcpy_htod(int(pointer) + 4, np.array([len(self.mu)], dtype=np.int32))
        # TODO: generalize nTimeOrderings
        cuda.memcpy_htod(int(pointer) + 8, np.array([6], dtype=np.int32))
        cuda.memcpy_htod(int(pointer) + 12, np.array([len(self.recorded_indices)], dtype=np.int32))

        # Set pointers in the struct
        cuda.memcpy_htod(int(pointer) + 16, np.intp(int(rho)))
        cuda.memcpy_htod(int(pointer) + 24, np.intp(int(mu)))
        cuda.memcpy_htod(int(pointer) + 32, np.intp(int(omega)))
        cuda.memcpy_htod(int(pointer) + 40, np.intp(int(Gamma)))
        cuda.memcpy_htod(int(pointer) + 48, np.intp(int(time_orderings)))
        cuda.memcpy_htod(int(pointer) + 56, np.intp(int(recorded_indices)))

    def matrix(self, efields, time):
        """Generate the time dependant Hamiltonian Coupling Matrix.

        Parameters
        ----------
        efields : ndarray<Complex>
            Contains the time dependent electric fields.
            Shape (M x T) where M is number of electric fields, and T is number of timesteps.
        time : 1-D array <float64>
            The time step values

        Returns
        -------
        ndarray <Complex> 
            Shape T x N x N array with the full Hamiltonian at each time step.
            N is the number of states in the Density vector.
        """
        # TODO: Just put the body of this method in here, rather than calling _gen_matrix
        E1, E2, E3 = efields[0:3]
        return self._gen_matrix(E1, E2, E3, time, self.omega)

    def _gen_matrix(self, E1, E2, E3, time, energies):
        """
        creates the coupling array given the input e-fields values for a specific time, t
        
        Currently neglecting pathways where w2 and w3 require different frequencies
        (all TRIVE space, or DOVE on diagonal)
        
        Matrix formulated such that dephasing/relaxation is accounted for 
        outside of the matrix
        """
        # Define transition energies
        wag  = energies[2]
        w2aa = energies[-1]

        # Define dipole moments
        mu_ag = self.mu[0]
        mu_2aa = self.mu[-1]

        # Define helpful variables
        A_1 = 0.5j * mu_ag * E1 * np.exp(1j * wag * time)
        A_2 = 0.5j * mu_ag * E2 * np.exp(-1j * wag * time)
        A_2prime = 0.5j * mu_ag * E3 * np.exp(1j * wag * time)
        B_1 = 0.5j * mu_2aa * E1 * np.exp(1j * w2aa * time)
        B_2 = 0.5j * mu_2aa * E2 * np.exp(-1j * w2aa * time)
        B_2prime = 0.5j * mu_2aa * E3 * np.exp(1j * w2aa * time)

        # Initailze the full array of all hamiltonians to zero
        out = np.zeros((len(time), len(energies), len(energies)), dtype=np.complex128)

        # Add appropriate array elements, according to the time orderings
        if 3 in self.time_orderings or 5 in self.time_orderings:
            out[:, 1, 0] = -A_2
        if 4 in self.time_orderings or 6 in self.time_orderings:
            out[:, 2, 0] = A_2prime
        if 1 in self.time_orderings or 2 in self.time_orderings:
            out[:, 3, 0] = A_1
        if 3 in self.time_orderings:
            out[:, 5, 1] = A_1
        if 5 in self.time_orderings:
            out[:, 6, 1] = A_2prime
        if 4 in self.time_orderings:
            out[:, 4, 2] = B_1
        if 6 in self.time_orderings:
            out[:, 6, 2] = -A_2
        if 2 in self.time_orderings:
            out[:, 4, 3] = B_2prime
        if 1 in self.time_orderings:
            out[:, 5, 3] = -A_2
        if 2 in self.time_orderings or 4 in self.time_orderings:
            out[:, 7, 4] = B_2
            out[:, 8, 4] = -A_2
        if 1 in self.time_orderings or 3 in self.time_orderings:
            out[:, 7, 5] = -2 * A_2prime
            out[:, 8, 5] = B_2prime
        if 5 in self.time_orderings or 6 in self.time_orderings:
            out[:, 7, 6] = -2 * A_1
            out[:, 8, 6] = B_1

        # Add Gamma along the diagonal
        for i in range(len(self.Gamma)):
            out[:, i, i] = -1 * self.Gamma[i]

        # NOTE: NISE multiplied outputs by the approriate mu in here
        #      This mu factors out, remember to use it where needed later
        #      Removed for clarity and aligning with Equation S15 of Kohler2017

        return out

    cuda_matrix_source = """
    /**
     *  Hamiltonian_matrix: Computes the Hamiltonian matrix for an individual time step.
     *  NOTE: This differs from the Python implementation, which computes the full time 
     *          dependant hamiltonian, this only computes for a single time step
     *          (to conserve memory).
     * 
     *  Parameters
     *  ----------
     *  Hamiltonian ham: A struct which represents a hamiltonian,
     *                   containing orrays omega, mu, and Gamma
     *  cmplx* efields: A pointer to an array containg the complex valued
     *                  electric fields to use for evaluation
     *  double time: the current time step counter
     *
     *  Output
     *  -------
     *  cmplx* out: an N x N matrix containing the transition probabilities
     *
     */
    __device__ void Hamiltonian_matrix(Hamiltonian ham, pycuda::complex<double>* efields,
                                       double time, pycuda::complex<double>* out)
    {
        // Define state energies
        double wag = ham.omega[1];
        double w2aa = ham.omega[8];

        // Define dipoles
        //TODO: don't assume one, generalize
        pycuda::complex<double> mu_ag =  1.;//ham.mu[0];
        pycuda::complex<double> mu_2aa = 1.;//ham.mu[1];

        // Define the electric field values
        pycuda::complex<double> E1 =  efields[0];
        pycuda::complex<double> E2 =  efields[1];
        pycuda::complex<double> E3 =  efields[2];

        // Define helpful variables
        pycuda::complex<double> A_1 = 0.5 * I * mu_ag * E1 * pycuda::exp(-1. * I * wag * time);
        pycuda::complex<double> A_2 = 0.5 * I * mu_ag * E2 * pycuda::exp(I * wag * time);
        pycuda::complex<double> A_2prime = 0.5 * I * mu_ag * E3 * pycuda::exp(-1. * I * wag * time);
        pycuda::complex<double> B_1 = 0.5 * I * mu_2aa * E1 * pycuda::exp(-1. * I * w2aa * time);
        pycuda::complex<double> B_2 = 0.5 * I * mu_2aa * E2 * pycuda::exp(I * w2aa * time);
        pycuda::complex<double> B_2prime = 0.5 * I * mu_2aa * E3 * pycuda::exp(-1. * I * w2aa * time);

        //TODO: zero once, take this loop out of the inner most loop
        for (int i=0; i<ham.nStates * ham.nStates; i++) out[i] = pycuda::complex<double>();

        // Fill in appropriate matrix elements
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

        // Put Gamma along the diagonal
        for(int i=0; i<ham.nStates; i++) out[i*ham.nStates + i] = -1. * ham.Gamma[i];
    }
"""

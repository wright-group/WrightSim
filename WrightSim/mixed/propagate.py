import numpy as np
from pycuda import driver as cuda

def runge_kutta(t, efields, n_recorded, hamiltonian):
    """
    Evolves the hamiltonian in time, given the three 1d array of E-field points 
    for each E-field and the hamiltonian

    H[pulse_permutations, t, ij_to, ij_from] is a rank 4 array of the hamiltonian 2D 
    matrix for each time point and necessary pulse permutations

    Uses Runge-Kutta method to integrate 
        --> assumes eqaully spaced t-points (t_int is spacing)

    Unlike earlier implementations, delays, phase, etc., must be incorporated
    externally to the function.  Rotating wave may still be specified.
    """
    # can only call on n_recorded and t after efield_object.E is called
    dt = np.abs(t[1]-t[0])
    # extract attributes of the system
    rho_emitted = np.empty((len(hamiltonian.recorded_indices), n_recorded), dtype=np.complex64)

    # H has 3 dimensions: time and the 2 matrix dimensions
    H = hamiltonian.matrix(efields, t)
    # index to keep track of elements of rho_emitted
    emitted_index = 0
    rho_i = hamiltonian.rho.copy()
    for k in range(len(t)-1):
        # now sum over p equivalent pulse permutations (e.g. TRIVE near-
        # diagonal has 2 permutations)
        # calculate delta rho based on previous rho values
        temp_delta_rho = np.dot(H[k], rho_i)
        temp_rho_i = rho_i + temp_delta_rho*dt
        delta_rho = np.dot(H[k+1], temp_rho_i)
        rho_i += dt/2 * (temp_delta_rho + delta_rho)
        # if we are close enough to final coherence emission, start 
        # storing these values
        if k >= len(t) - n_recorded:
            for rec,native in enumerate(hamiltonian.recorded_indices):
                rho_emitted[rec, emitted_index] = rho_i[native]
            emitted_index += 1
    # Last timestep
    temp_delta_rho = np.dot(H[-1], rho_i)
    rho_i += temp_delta_rho*dt
    for rec,native in enumerate(hamiltonian.recorded_indices):
        rho_emitted[rec, emitted_index] = rho_i[native]

   # rho_emitted[s,t], s is out_group index, t is time index
    return rho_emitted

runge_kutta_cuda_source = """
    __device__ pycuda::complex<double>* runge_kutta(double time_start, double time_end, double dt, 
                                                   int nEFields, pycuda::complex<double> *efields,
                                                   int n_recorded, Hamiltonian ham,
                                                   pycuda::complex<double> *out)
    {
        pycuda::complex<double> *H_cur = malloc(ham->nStates * ham->nStates * sizeof(pycuda::complex<double>));
        pycuda::complex<double> *H_next = malloc(ham->nStates * ham->nStates * sizeof(pycuda::complex<double>));

        int out_index = 0;
        int index=0;
        int npoints = (int) ((time_end-time_start)/dt);

        H_next = Hamiltonian_matrix(nEFields, efields + nEFields*index, t);
        for(double t = time_start; t < time_end; t += dt)
        {   
            H_cur = H_next;
            H_next = Hamiltonian_matrix(nEFields, efields + nEFields*(index+1), t+dt);
            //TODO: write dot
            pycuda::complex<double>* temp_delta_rho = dot(H_cur, ham->rho, ham->nStates);
            //TODO: write muladd/see if I can use the pycuda one?
            pycuda::complex<double>* temp_rho_i = muladd(ham->rho, 1., temp_delta_rho, dt);
            pycuda::complex<double>* delta_rho = dot(H_next, temp_rho_i, ham->nStates);
            //TODO add to ham->rho

            if(index > npoints - n_recorded)
            {
                for(int i=0; i < ham->nRecorded; i++)
                    out[out_index + i * n_recorded] = ham->rho[ham->recorded_indices[i]];
                out_index++;
            }
            index++;
        }
        
        pycuda::complex<double>* temp_delta_rho = dot(H_cur, ham->rho, ham->nStates);
        ham->rho = muladd(ham->rho, 1., temp_delta_rho, dt);
        for(int i=0; i < ham->nRecorded; i++)
            out[out_index + i * n_recorded] = ham->rho[ham->recorded_indices[i]];
    }
